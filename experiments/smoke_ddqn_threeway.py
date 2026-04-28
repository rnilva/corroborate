"""§3 acceptance smoke — three-way verdict on the DDQN sweep.

Loads `experiments/data/ddqn/runs.parquet`, then for each
(env, capacity) combination:

1. **Mechanism verdict** — paired Δ on `mechanism.jensen_gap`,
   predicted_direction='a_lt_b' (DDQN should *reduce* the gap).
2. **Outcome verdict** — paired Δ on each of three outcome
   reductions (`late_window_mean`, `eval_final_mean`,
   `eval_best_burst_mean`), predicted_direction='a_gt_b'.
3. **Link verdict** — Pearson r across envs between mechanism Δ
   (g) and outcome Δ (g). Predicted positive — mechanism
   reduction should track outcome improvement.

Plus an HP-sensitivity table: outcome by replay capacity,
holding env fixed. Reveals whether DDQN's effect is robust to
the buffer-capacity HP.

Run: `uv run python experiments/smoke_ddqn_threeway.py`."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import polars as pl

from corroborate.aggregate import (
    link_pearson_across_envs,
    paired_comparison_from_runs,
)
from corroborate.persistence import read_runrows, write_comparisonrows
from corroborate.schema import ComparisonRow, RunRow
from corroborate.statistics import (
    PooledStats,
    random_effects_summary,
    random_effects_verdict,
)


_DATA_PATH = Path(__file__).parent / 'data' / 'ddqn' / 'runs.parquet'
_COMPARISONS_PATH = (
    Path(__file__).parent / 'data' / 'ddqn' / 'comparisons.parquet'
)


_OUTCOME_PATHS = (
    'outcome.late_window_mean',
    'outcome.eval_final_mean',
    'outcome.eval_best_burst_mean',
)


def _f(x: float | int | str | bool | None, prec: int = 3) -> str:
    if x is None:
        return '—'
    if not isinstance(x, (int, float)):
        return repr(x)
    if isinstance(x, float) and x != x:  # NaN
        return 'NaN'
    return f'{x:.{prec}f}'


def _by_env_capacity_intervention(
    rows: list[RunRow],
) -> dict[tuple[str, int], dict[str, list[RunRow]]]:
    """Index runs by (env, capacity) → intervention → [runs].
    Capacity is the swept HP that varies the leaf signature; we
    pair within each capacity level."""
    out: dict[tuple[str, int], dict[str, list[RunRow]]] = defaultdict(
        lambda: defaultdict(list),
    )
    for r in rows:
        env_name = r.measurements.get('env_name')
        intervention = r.measurements.get('intervention_name')
        capacity = r.measurements.get('replay.capacity')
        if not isinstance(env_name, str):
            continue
        if not isinstance(intervention, str):
            continue
        if isinstance(capacity, bool) or not isinstance(capacity, (int, float)):
            continue
        out[(env_name, int(capacity))][intervention].append(r)
    return out


def _mechanism_cmp(
    treatment: list[RunRow], baseline: list[RunRow],
    env_name: str, capacity: int,
) -> ComparisonRow:
    return paired_comparison_from_runs(
        treatment_runs=treatment, baseline_runs=baseline,
        outcome_path='mechanism.jensen_gap',
        pair_by=('seed',),  # within-env pairing on seed
        predicted_direction='a_lt_b',  # DDQN should REDUCE the gap
        extra_measurements={
            'comparison_kind': 'mechanism',
            'env_name': env_name,
            'replay.capacity': capacity,
        },
    )


def _outcome_cmp(
    treatment: list[RunRow], baseline: list[RunRow],
    outcome_path: str, env_name: str, capacity: int,
) -> ComparisonRow:
    return paired_comparison_from_runs(
        treatment_runs=treatment, baseline_runs=baseline,
        outcome_path=outcome_path,
        pair_by=('seed',),
        predicted_direction='a_gt_b',  # DDQN should INCREASE return
        extra_measurements={
            'comparison_kind': 'outcome',
            'env_name': env_name,
            'replay.capacity': capacity,
        },
    )


def _print_verdict_row(
    label: str, cmp: ComparisonRow, gp_key: str,
) -> None:
    g = cmp.measurements.get(f'{gp_key}.effect_size_g')
    se = cmp.measurements.get(f'{gp_key}.se')
    q = cmp.measurements.get(f'{gp_key}.derived_q')
    rc = cmp.refutation_class.value if cmp.refutation_class else '—'
    print(
        f'    {label:<32} {cmp.verdict.value:<22} {rc:<14} '
        f'g={_f(g):>7} se={_f(se):>7} q={_f(q):>5} '
        f'powered={str(cmp.adequately_powered):>5}'
    )


def _print_pooled_row(
    label: str,
    cmps: list[ComparisonRow],
    path_key: str,
    *,
    predicted_direction: str,
) -> None:
    """Render the random-effects pooled verdict for a list of
    per-env ComparisonRows. Reads (g, se) from each comparison's
    measurements at `<path_key>.effect_size_g` / `<path_key>.se`,
    pools via DerSimonian-Laird, and prints pooled_g + 95% PI +
    tau² + I² + Popperian verdict."""
    pairs: list[tuple[float, float]] = []
    for cmp in cmps:
        g = cmp.measurements.get(f'{path_key}.effect_size_g')
        se = cmp.measurements.get(f'{path_key}.se')
        if isinstance(g, (int, float)) and isinstance(se, (int, float)):
            pairs.append((float(g), float(se)))
    if not pairs:
        print(f'    {label:<32} (no per-env (g, se) pairs)')
        return
    pool: PooledStats = random_effects_summary(pairs)
    # `random_effects_verdict` accepts None / 'a_gt_b' / 'a_lt_b' /
    # 'two_sided'; we pass strings directly here for the smoke.
    from corroborate.hypothesis import Direction as _Dir
    verdict, refutation = random_effects_verdict(
        pool,
        predicted_direction=predicted_direction,  # type: ignore[arg-type]
    )
    rc = refutation.value if refutation else '—'
    print(
        f'    {label:<32} {verdict.value:<22} {rc:<14} '
        f'pooled_g={_f(pool.pooled_g):>7} '
        f'PI=[{_f(pool.pi_lo):>7}, {_f(pool.pi_hi):>7}] '
        f'tau²={_f(pool.tau2):>5} I²={_f(pool.I2):>5} '
        f'n_envs={pool.n_cells}'
    )


def main() -> None:
    rows = read_runrows(_DATA_PATH)
    print(f'loaded {len(rows)} RunRows from {_DATA_PATH.name}')
    grouped = _by_env_capacity_intervention(rows)
    keys = sorted(grouped.keys())

    # ============ Per (env, capacity) three-way verdict ============

    print('\n' + '=' * 110)
    print('PER (env, capacity) THREE-WAY VERDICTS')
    print('=' * 110)

    # Cache mechanism + outcome comparisons (per capacity) for the
    # link verdict that crosses envs at the same capacity.
    mech_by_capacity: dict[int, list[ComparisonRow]] = defaultdict(list)
    out_by_capacity: dict[
        int, dict[str, list[ComparisonRow]]
    ] = defaultdict(lambda: defaultdict(list))

    all_comparisons: list[ComparisonRow] = []

    for env, capacity in keys:
        ig = grouped[(env, capacity)]
        if 'ddqn' not in ig or 'vanilla_dqn' not in ig:
            continue
        treatment = ig['ddqn']
        baseline = ig['vanilla_dqn']

        print(f'\n  {env} | replay.capacity={capacity} | '
              f'n_treatment={len(treatment)} n_baseline={len(baseline)}')

        # Mechanism (Jensen gap; DDQN should reduce it).
        if any('mechanism.jensen_gap' in r.measurements for r in treatment):
            mech_cmp = _mechanism_cmp(treatment, baseline, env, capacity)
            _print_verdict_row(
                'mechanism.jensen_gap', mech_cmp, 'mechanism.jensen_gap',
            )
            mech_by_capacity[capacity].append(mech_cmp)
            all_comparisons.append(mech_cmp)

        # Outcomes (return; DDQN should increase).
        for path in _OUTCOME_PATHS:
            if not any(path in r.measurements for r in treatment):
                continue
            out_cmp = _outcome_cmp(treatment, baseline, path, env, capacity)
            _print_verdict_row(path, out_cmp, path)
            out_by_capacity[capacity][path].append(out_cmp)
            all_comparisons.append(out_cmp)

    # ============ Cross-env LINK verdict per capacity ============

    print('\n' + '=' * 110)
    print('CROSS-ENV LINK VERDICTS (Pearson r between mechanism Δg + outcome Δg)')
    print('=' * 110)

    for capacity in sorted(out_by_capacity.keys()):
        print(f'\n  replay.capacity={capacity}')
        mech_cmps = mech_by_capacity.get(capacity, [])
        if not mech_cmps:
            print('    (no mechanism comparisons at this capacity)')
            continue
        for outcome_path in _OUTCOME_PATHS:
            out_cmps = out_by_capacity[capacity].get(outcome_path, [])
            if len(out_cmps) < 3:
                continue
            link = link_pearson_across_envs(
                mech_cmps, out_cmps,
                mechanism_path='mechanism.jensen_gap.effect_size_g',
                outcome_path=f'{outcome_path}.effect_size_g',
                group_by='env_name',
                extra_measurements={
                    'comparison_kind': 'link',
                    'outcome_path': outcome_path,
                    'replay.capacity': capacity,
                },
            )
            all_comparisons.append(link)
            r = link.measurements.get('link.pearson_r')
            n_envs = link.measurements.get('n_paired_envs')
            rc = link.refutation_class.value if link.refutation_class else '—'
            print(
                f'    link({outcome_path:<32}) '
                f'{link.verdict.value:<22} {rc:<14} '
                f'r={_f(r):>7} n_envs={n_envs} '
                f'powered={str(link.adequately_powered)}'
            )

    # ============ Random-effects pooled verdict per capacity ============

    print('\n' + '=' * 110)
    print('RANDOM-EFFECTS POOLED VERDICTS (DerSimonian-Laird across envs)')
    print('=' * 110)

    for capacity in sorted(out_by_capacity.keys()):
        print(f'\n  replay.capacity={capacity}')

        # Mechanism: predicted_direction='a_lt_b' (DDQN reduces gap).
        mech_cmps = mech_by_capacity.get(capacity, [])
        _print_pooled_row(
            'mechanism.jensen_gap',
            mech_cmps, 'mechanism.jensen_gap',
            predicted_direction='a_lt_b',
        )

        # Outcomes: predicted_direction='a_gt_b' (DDQN increases return).
        for outcome_path in _OUTCOME_PATHS:
            out_cmps = out_by_capacity[capacity].get(outcome_path, [])
            _print_pooled_row(
                outcome_path, out_cmps, outcome_path,
                predicted_direction='a_gt_b',
            )

    # ============ HP-sensitivity (capacity sweep) ============

    print('\n' + '=' * 110)
    print('HP-SENSITIVITY: outcome by replay.capacity, holding env fixed')
    print('=' * 110)

    df = pl.read_parquet(_DATA_PATH)
    if df.is_empty() or 'env_name' not in df.columns:
        print('\n  (runs.parquet empty or missing env_name; skipping)')
    else:
        summary = df.group_by(
            ['env_name', 'intervention_name', 'replay.capacity'],
        ).agg([
            pl.col('outcome.late_window_mean').mean().alias('late_mean'),
            pl.col('outcome.late_window_mean').std().alias('late_sd'),
            pl.col('outcome.late_window_mean').count().alias('n'),
        ]).sort(['env_name', 'intervention_name', 'replay.capacity'])
        print()
        print(summary)

    # Persist all comparisons to a single parquet for downstream
    # repeatability. Read via `read_comparisonrows(_COMPARISONS_PATH)`
    # and project by `comparison_kind` / `replay.capacity` /
    # `outcome_path` to recover the §3 verdict shape without
    # recomputing.
    if all_comparisons:
        write_comparisonrows(all_comparisons, _COMPARISONS_PATH)
        print(
            f'\nwrote {len(all_comparisons)} comparisons → '
            f'{_COMPARISONS_PATH.name}',
        )

    print('\nAll verdicts rendered.')


if __name__ == '__main__':
    main()
