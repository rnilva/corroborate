"""§3 acceptance smoke — three-way verdict on the DDQN sweep.

Loads `experiments/data/ddqn/runs.parquet`, then for each
`replay.capacity`:

1. **Mechanism verdict** — paired Δ on `mechanism.jensen_gap`,
   predicted_direction='a_lt_b' (DDQN should *reduce* the gap).
2. **Outcome verdict** — paired Δ on each of three outcome
   reductions (`late_window_mean`, `eval_final_mean`,
   `eval_best_burst_mean`), predicted_direction='a_gt_b'.
3. **Link verdict** — Pearson r across envs between mechanism Δ
   (g) and outcome Δ (g). Predicted positive — mechanism
   reduction should track outcome improvement.

The mechanism + outcome verdicts go through
`HypothesisComparisonRow.from_cells(...)` — the canonical
aggregator that internally pairs by `('seed',)` within each env,
runs per-env Hedges' g, and pools across envs via random-effects
DerSimonian-Laird in one pass. `row.per_group` carries per-env
stats; `row.pooled` carries the pooled summary.

Run: `uv run python experiments/smoke_ddqn_threeway.py`."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import polars as pl

from corroborate.aggregate import link_pearson_across_envs
from corroborate.hypothesis import Direction, Hypothesis
from corroborate.persistence import read_runrows, write_comparisonrows
from corroborate.schema import (
    ComparisonRow,
    HypothesisComparisonRow,
    RunRow,
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


def _by_capacity_intervention(
    rows: list[RunRow],
) -> dict[int, dict[str, list[RunRow]]]:
    """Index runs by capacity → intervention → [runs]."""
    out: dict[int, dict[str, list[RunRow]]] = defaultdict(
        lambda: defaultdict(list),
    )
    for r in rows:
        intervention = r.measurements.get('intervention_name')
        capacity = r.measurements.get('replay.capacity')
        if not isinstance(intervention, str):
            continue
        if isinstance(capacity, bool) or not isinstance(capacity, (int, float)):
            continue
        out[int(capacity)][intervention].append(r)
    return out


def _print_per_group(row: HypothesisComparisonRow) -> None:
    """Render per-env GroupStats from a HypothesisComparisonRow."""
    for gs in sorted(row.per_group, key=lambda g: repr(g.group_value)):
        rc = gs.refutation_class.value if gs.refutation_class else '—'
        print(
            f'      {str(gs.group_value):<28} '
            f'{gs.verdict.value:<22} {rc:<14} '
            f'g={_f(gs.effect_size_g):>7} se={_f(gs.se):>7} '
            f'q={_f(gs.derived_q):>5} n_pairs={gs.n_pairs} '
            f'powered={str(gs.adequately_powered):>5}'
        )


def _print_pooled(label: str, row: HypothesisComparisonRow) -> None:
    """Render the pooled summary line for one HypothesisComparisonRow."""
    rc = row.refutation_class.value if row.refutation_class else '—'
    if row.pooled is None:
        print(f'    {label:<32} {row.verdict.value:<22} '
              f'{rc:<14} (single-group)')
        return
    print(
        f'    {label:<32} {row.verdict.value:<22} {rc:<14} '
        f'pooled_g={_f(row.pooled.pooled_g):>7} '
        f'PI=[{_f(row.pooled.pi_lo):>7}, {_f(row.pooled.pi_hi):>7}] '
        f'tau²={_f(row.pooled.tau2):>5} I²={_f(row.pooled.I2):>5} '
        f'n_envs={row.pooled.n_cells}'
    )


def _comparison_from_hypothesis_row(
    row: HypothesisComparisonRow, *,
    extra_measurements: dict[str, object],
) -> ComparisonRow:
    """Project a HypothesisComparisonRow's per-env stats into
    individual ComparisonRows for the link verdict + persistence.

    The link verdict (`link_pearson_across_envs`) operates on
    per-env (env_name, effect_size_g) pairs, so we surface those
    from row.per_group as flat-keyed measurements compatible with
    the existing primitive."""
    raise NotImplementedError(
        'projection only used inline; see _link_inputs_from_row',
    )


def _link_inputs_from_row(
    row: HypothesisComparisonRow,
    *,
    outcome_path: str,
) -> list[ComparisonRow]:
    """Synthesize per-env `ComparisonRow`s from a stratified
    HypothesisComparisonRow's `per_group`. Each ComparisonRow
    carries `<outcome_path>.effect_size_g` and `env_name` so
    `link_pearson_across_envs` can pair them on env."""
    from corroborate.verdict import Verdict
    out: list[ComparisonRow] = []
    for gs in row.per_group:
        # Match OLD smoke's behaviour: store effect_size_g + se
        # independently. link_pearson_across_envs filters NaN
        # internally; no need to pre-filter here.
        measurements: dict[str, object] = {
            'env_name': gs.group_value
                if isinstance(gs.group_value, (int, float, bool, str))
                else str(gs.group_value),
        }
        if gs.effect_size_g is not None:
            measurements[f'{outcome_path}.effect_size_g'] = float(
                gs.effect_size_g,
            )
        if gs.se is not None:
            measurements[f'{outcome_path}.se'] = float(gs.se)
        cmp = ComparisonRow(
            id=str(gs.group_value),
            parent_id=None,
            cycle_id=None,
            timestamp=row.timestamp,
            treatment_arm_id='',
            baseline_arm_id='',
            predicted_direction=row.predicted_direction,
            verdict=gs.verdict,
            refutation_class=gs.refutation_class,
            adequately_powered=gs.adequately_powered,
            measurements=measurements,  # type: ignore[arg-type]
        )
        out.append(cmp)
    return out


def main() -> None:
    rows = read_runrows(_DATA_PATH)
    print(f'loaded {len(rows)} RunRows from {_DATA_PATH.name}')
    by_cap = _by_capacity_intervention(rows)
    capacities = sorted(by_cap.keys())

    all_comparisons: list[ComparisonRow] = []
    rows_by_cap: dict[
        int, dict[str, HypothesisComparisonRow],
    ] = defaultdict(dict)

    for capacity in capacities:
        ig = by_cap[capacity]
        if 'ddqn' not in ig or 'vanilla_dqn' not in ig:
            continue
        treatment = ig['ddqn']
        baseline = ig['vanilla_dqn']

        print('\n' + '=' * 110)
        print(f'replay.capacity={capacity} | n_treatment={len(treatment)} '
              f'n_baseline={len(baseline)}')
        print('=' * 110)

        # Mechanism (DDQN should REDUCE Jensen gap).
        mech_h: Hypothesis = Hypothesis(
            name='ddqn', intervention={}, bridges=(),
            predicted_direction='a_lt_b',
        )
        mech_row = HypothesisComparisonRow.from_cells(
            mech_h, treatment, baseline,
            outcome_path='mechanism.jensen_gap',
            pair_by=('seed',),
            group_by='env_name',
        )
        rows_by_cap[capacity]['mechanism.jensen_gap'] = mech_row
        print(f'\n  mechanism.jensen_gap (per-env, sorted)')
        _print_per_group(mech_row)

        # Outcomes (DDQN should INCREASE return).
        out_h: Hypothesis = Hypothesis(
            name='ddqn', intervention={}, bridges=(),
            predicted_direction='a_gt_b',
        )
        for path in _OUTCOME_PATHS:
            row = HypothesisComparisonRow.from_cells(
                out_h, treatment, baseline,
                outcome_path=path,
                pair_by=('seed',),
                group_by='env_name',
            )
            rows_by_cap[capacity][path] = row
            print(f'\n  {path} (per-env, sorted)')
            _print_per_group(row)

    # ============ Cross-env LINK verdict per capacity ============

    print('\n' + '=' * 110)
    print('CROSS-ENV LINK VERDICTS (Pearson r between mechanism Δg + outcome Δg)')
    print('=' * 110)

    for capacity in capacities:
        rows_for_cap = rows_by_cap.get(capacity, {})
        if 'mechanism.jensen_gap' not in rows_for_cap:
            continue
        mech_row = rows_for_cap['mechanism.jensen_gap']
        mech_cmps = _link_inputs_from_row(
            mech_row, outcome_path='mechanism.jensen_gap',
        )
        if len(mech_cmps) < 3:
            continue
        print(f'\n  replay.capacity={capacity}')
        for outcome_path in _OUTCOME_PATHS:
            if outcome_path not in rows_for_cap:
                continue
            out_cmps = _link_inputs_from_row(
                rows_for_cap[outcome_path], outcome_path=outcome_path,
            )
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

    # ============ Pooled summary per capacity (one line per kind) ============

    print('\n' + '=' * 110)
    print('RANDOM-EFFECTS POOLED VERDICTS (DerSimonian-Laird across envs)')
    print('=' * 110)

    for capacity in capacities:
        rows_for_cap = rows_by_cap.get(capacity, {})
        if not rows_for_cap:
            continue
        print(f'\n  replay.capacity={capacity}')
        if 'mechanism.jensen_gap' in rows_for_cap:
            _print_pooled(
                'mechanism.jensen_gap',
                rows_for_cap['mechanism.jensen_gap'],
            )
        for outcome_path in _OUTCOME_PATHS:
            if outcome_path in rows_for_cap:
                _print_pooled(outcome_path, rows_for_cap[outcome_path])

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

    # Persist link comparisons for downstream repeatability.
    if all_comparisons:
        write_comparisonrows(all_comparisons, _COMPARISONS_PATH)
        print(
            f'\nwrote {len(all_comparisons)} link comparisons → '
            f'{_COMPARISONS_PATH.name}',
        )

    print('\nAll verdicts rendered.')


if __name__ == '__main__':
    main()
