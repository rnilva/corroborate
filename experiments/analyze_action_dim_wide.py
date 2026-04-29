"""Combined action-dim analysis on `action_dim_sweep/` (4 envs) +
`action_dim_wide/` (6 envs) = 10 envs spanning |A| ∈ {2, 3, 4, 5, 10}
at converging HPs.

Computes:
  - Per-env paired g on `mechanism.jensen_gap` (DDQN vs vanilla,
    pair-by seed, predicted DDQN < vanilla) — the action_dim
    dependency at the mechanism edge.
  - Per-env paired g on `outcome.eval_final_mean` — the link
    edge.
  - Random-effects pool of mechanism g across envs.
  - Meta-regression of mechanism g on log(action_dim).

Usage:
  uv run python experiments/analyze_action_dim_wide.py
"""
from __future__ import annotations

import math
from pathlib import Path

import polars as pl

from corroborate._polars_boundary import to_dicts as _to_dicts
from corroborate.aggregate import paired_comparison_from_runs
from corroborate.meta_regression import StratumObservation, meta_regression
from corroborate.rl.env_catalogue import get
from corroborate.schema import RunRow
from corroborate.statistics import (
    PooledStats, random_effects_summary, random_effects_verdict,
)
from corroborate.verdict import Verdict


_CORPORA: tuple[Path, ...] = (
    Path('experiments/data/action_dim_sweep/runs.parquet'),
    Path('experiments/data/action_dim_wide/runs.parquet'),
)


def _load_combined() -> list[RunRow]:
    rows: list[RunRow] = []
    for p in _CORPORA:
        if not p.exists():
            continue
        df = pl.read_parquet(p)
        rows.extend(RunRow.from_row_dict(d) for d in _to_dicts(df))
    return rows


def _per_env_g(
    runs: list[RunRow], env_name: str, *, outcome_path: str,
    predicted_direction: str,
) -> tuple[float, float, int, str]:
    er = [r for r in runs if r.measurements.get('env_name') == env_name]
    ddqn = [r for r in er if r.measurements.get('intervention_name') == 'ddqn']
    vanilla = [r for r in er if r.measurements.get('intervention_name') == 'vanilla_dqn']
    if not ddqn or not vanilla:
        return float('nan'), float('nan'), 0, 'missing'
    cmp = paired_comparison_from_runs(
        ddqn, vanilla,
        outcome_path=outcome_path,
        pair_by=('seed',),
        predicted_direction=predicted_direction,  # type: ignore[arg-type]
    )
    g = cmp.measurements.get(f'{outcome_path}.effect_size_g', float('nan'))
    se = cmp.measurements.get(f'{outcome_path}.se', float('nan'))
    n = cmp.measurements.get('n_pairs', 0)
    g_f = float(g) if isinstance(g, (int, float)) and not math.isnan(float(g)) else float('nan')
    se_f = float(se) if isinstance(se, (int, float)) and not math.isnan(float(se)) else float('nan')
    n_i = int(n) if isinstance(n, (int, float)) else 0
    return g_f, se_f, n_i, cmp.verdict.value


def _format_pool(label: str, p: PooledStats, v: Verdict) -> str:
    if p.n_cells < 2:
        return f'  {label:<22} n_envs={p.n_cells} (too few)'
    return (
        f'  {label:<22} n_envs={p.n_cells} '
        f'g_pooled={p.pooled_g:+.3f} '
        f'I²={p.I2:.2f} '
        f'PI=[{p.pi_lo:+.3f}, {p.pi_hi:+.3f}]  '
        f'verdict={v.value}'
    )


def main() -> None:
    runs = _load_combined()
    print('=' * 100)
    print(f'Combined action-dim analysis  — {len(runs)} cells across both corpora')
    print('=' * 100)

    envs = sorted({r.measurements.get('env_name') for r in runs if isinstance(r.measurements.get('env_name'), str)})
    print()
    print(f'  {"env":<25} {"|A|":>4} {"n":>4} '
          f'{"g_mech":>8} {"se_m":>5} {"verdict_m":<22}  '
          f'{"g_link":>8} {"se_l":>5} {"verdict_l":<22}')
    print('-' * 130)
    g_se_mech: list[tuple[str, int, float, float]] = []
    g_se_link: list[tuple[str, int, float, float]] = []
    for env in envs:
        try:
            n_a = get(env).n_actions
        except Exception:
            n_a = 0
        gm, sm, nm, vm = _per_env_g(
            runs, env, outcome_path='mechanism.jensen_gap',
            predicted_direction='a_lt_b',
        )
        gl, sl, nl, vl = _per_env_g(
            runs, env, outcome_path='outcome.eval_final_mean',
            predicted_direction='a_gt_b',
        )
        print(
            f'  {env:<25} {n_a:>4} {nm:>4} '
            f'{gm:>+8.3f} {sm:>5.2f} {vm:<22}  '
            f'{gl:>+8.3f} {sl:>5.2f} {vl:<22}'
        )
        if not math.isnan(gm) and not math.isnan(sm) and sm > 0:
            g_se_mech.append((env, n_a, gm, sm))
        if not math.isnan(gl) and not math.isnan(sl) and sl > 0:
            g_se_link.append((env, n_a, gl, sl))

    # Random-effects pools.
    print()
    pool_m = random_effects_summary([(g, se) for _, _, g, se in g_se_mech])
    v_m, _ = random_effects_verdict(pool_m, predicted_direction='a_lt_b')
    pool_l = random_effects_summary([(g, se) for _, _, g, se in g_se_link])
    v_l, _ = random_effects_verdict(pool_l, predicted_direction='a_gt_b')
    print('Random-effects pools:')
    print(_format_pool('mechanism (a_lt_b)', pool_m, v_m))
    print(_format_pool('link (a_gt_b)', pool_l, v_l))

    # Meta-regression of mechanism g on log(action_dim).
    if len(g_se_mech) >= 4:
        print()
        print(f'Meta-regression: mechanism g ~ log(action_dim)')
        obs = [
            StratumObservation(
                stratum_id=env, g=g, se=se,
                covariates={'log_action_dim': math.log(n_a)},
            )
            for env, n_a, g, se in g_se_mech if n_a >= 2
        ]
        res = meta_regression(obs)
        print(f'  n_strata={res.n_strata}  R²={res.r_squared:+.3f}  intercept={res.intercept:+.3f}')
        for c in res.coefficients:
            sig = '✓ SIGNIFICANT' if c.is_significant else ' '
            print(
                f'  {c.name:<18} β={c.coefficient:+.3f}  '
                f'CI=[{c.ci_lo:+.3f}, {c.ci_hi:+.3f}]  '
                f'p={c.p_value:.4f}  {sig}'
            )
        if res.cleavage_axes:
            print(f'  CLEAVAGE: {", ".join(res.cleavage_axes)}')
        else:
            print('  no cleavage axis (no significant covariate)')

    # Stratify by |A| ≥ 3 vs |A| = 2.
    print()
    print('Stratified by |A|:')
    above_2 = [(g, se) for _, n_a, g, se in g_se_mech if n_a >= 3]
    eq_2 = [(g, se) for _, n_a, g, se in g_se_mech if n_a == 2]
    if above_2:
        p3 = random_effects_summary(above_2)
        v3, _ = random_effects_verdict(p3, predicted_direction='a_lt_b')
        print(_format_pool('|A| ≥ 3 mechanism', p3, v3))
    if eq_2:
        p2 = random_effects_summary(eq_2)
        v2, _ = random_effects_verdict(p2, predicted_direction='a_lt_b')
        print(_format_pool('|A| = 2 mechanism', p2, v2))


if __name__ == '__main__':
    main()
