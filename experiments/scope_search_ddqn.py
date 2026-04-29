"""Scope search for DDQN's link via meta-regression.

The framework's literal purpose: given a mechanism that activates
(jensen_gap reduces under DDQN) but doesn't reliably translate to
outcome (link broken on average), find the **measurable scope** —
a numeric covariate whose value predicts where the link holds.

This is the empirical-side counterpart to invariants. When the
corpus-level pool of paired g(outcome) is heterogeneous across
envs, meta-regression of per-env g on per-env covariates can
identify the cleavage axis. A significant coefficient is a numeric
threshold on a measurable (env feature, mediator value) — the
*scope claim's content*, not a "list of envs".

### Candidate scope variables

The principled candidate is **baseline jensen_gap** — the magnitude
of the theorem's premise violation. Hasselt: vanilla DQN's
overestimation bias is what DDQN corrects. If the bias is small,
correcting it can't help. If the bias is huge but Q-network is
fundamentally miscalibrated, correction may also be irrelevant.
Predicted shape: link strength is non-monotone in baseline_gap —
peaks at some "moderate" magnitude.

Other candidates exercised:
- `gap_reduction` — how much DDQN actually reduces the gap (the
  mechanism's *empirical* activation magnitude per env)
- `out_scale_log` — outcome's natural magnitude (controls
  normalization)
- `gap_per_outcome` — gap relative to the outcome scale (a
  unitless premise-violation index)
- `solve_rate` — fraction of cells reaching env solve threshold
  (regime indicator)
- `state_entropy_mean` — late-window state visit entropy (clean
  mediator from the panel audit)

### Method

1. Per env (literature/derived threshold only, 200k cells): paired
   Hedges' g on `outcome.eval_best_burst_mean` (DDQN vs vanilla,
   pair-by seed, predicted `a_gt_b` — DDQN should be ≥ vanilla).
2. Per env: compute candidate covariates.
3. Meta-regress: per-env g as a function of (intercept +
   covariates). Inverse-variance-weighted OLS, t-distribution CIs.
4. Report cleavage axes — covariates with CI excluding zero.

### Honest scope of this analysis

- 13 envs with thresholds → ~10 valid pairs after dropping degenerate.
  With 1 covariate + intercept, regression has ~8 dof — workable
  but each additional covariate burns a dof.
- Per-env paired g uses `predicted_direction='a_gt_b'` so HELD/null
  semantics align with "DDQN should beat vanilla on outcome".
- Covariates are env-summary statistics; per-cell scope analysis
  would need a different aggregation (e.g., bin cells by gap
  magnitude across envs).

Usage:
  uv run python experiments/scope_search_ddqn.py
  uv run python experiments/scope_search_ddqn.py --covariate baseline_gap_log
"""
from __future__ import annotations

import argparse
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import polars as pl

from corroborate._polars_boundary import to_dicts as _to_dicts
from corroborate.aggregate import paired_comparison_from_runs
from corroborate.meta_regression import (
    MetaRegressionResult, StratumObservation, meta_regression,
)
from corroborate.rl.env_solve_thresholds import (
    SOLVE_THRESHOLDS, envs_with_threshold,
)
from corroborate.schema import RunRow


_RUNS = Path('experiments/data/ddqn/runs_with_mediators.parquet')
_OUTCOME_PATH = 'outcome.eval_best_burst_mean'
_GAP_PATH = 'mechanism.jensen_gap'


def _load(env: str) -> tuple[list[RunRow], list[RunRow], pl.DataFrame]:
    df = pl.read_parquet(_RUNS).filter(
        (pl.col('env_name') == env) & (pl.col('total_steps') == 200000)
    )
    rows = [RunRow.from_row_dict(d) for d in _to_dicts(df)]
    ddqn = [r for r in rows if r.measurements.get('intervention_name') == 'ddqn']
    vanilla = [
        r for r in rows
        if r.measurements.get('intervention_name') == 'vanilla_dqn'
    ]
    return ddqn, vanilla, df


def _per_env_g(env: str) -> tuple[float, float, int, dict[str, float]] | None:
    """Return (g, se, n_pairs, covariates) for one env, or None when
    the env can't be fit (degenerate gap or non-positive SE)."""
    ddqn, vanilla, df = _load(env)
    if not ddqn or not vanilla:
        return None

    cmp = paired_comparison_from_runs(
        ddqn, vanilla,
        outcome_path=_OUTCOME_PATH,
        pair_by=('seed',),
        predicted_direction='a_gt_b',
    )
    g_v = cmp.measurements.get(f'{_OUTCOME_PATH}.effect_size_g')
    se_v = cmp.measurements.get(f'{_OUTCOME_PATH}.se')
    n_v = cmp.measurements.get('n_pairs')
    if not all(isinstance(x, (int, float)) and not math.isnan(float(x))
               for x in (g_v, se_v, n_v)):
        return None
    g = float(g_v)  # type: ignore[arg-type]
    se = float(se_v)  # type: ignore[arg-type]
    n = int(n_v)  # type: ignore[arg-type]
    if se <= 0.0:
        return None

    # Per-env covariates.
    summary = (
        df.group_by('intervention_name').agg([
            pl.col(_GAP_PATH).mean().alias('gap_mean'),
            pl.col(_OUTCOME_PATH).mean().alias('out_mean'),
            pl.col(_OUTCOME_PATH).std().alias('out_std'),
            pl.col('mediator.state_visit_entropy_late').mean().alias('ent_mean'),
        ])
    )
    arm: dict[str, dict[str, float]] = {}
    for r in summary.iter_rows(named=True):
        arm[r['intervention_name']] = {
            'gap': float(r['gap_mean']) if r['gap_mean'] is not None else 0.0,
            'out': float(r['out_mean']) if r['out_mean'] is not None else 0.0,
            'out_std': float(r['out_std']) if r['out_std'] is not None else 0.0,
            'ent': float(r['ent_mean']) if r['ent_mean'] is not None else 0.0,
        }
    if 'ddqn' not in arm or 'vanilla_dqn' not in arm:
        return None

    baseline_gap = arm['vanilla_dqn']['gap']
    ddqn_gap = arm['ddqn']['gap']
    out_scale = max(abs(arm['vanilla_dqn']['out']), 1e-6)
    gap_reduction = baseline_gap - ddqn_gap
    state_entropy_mean = (arm['ddqn']['ent'] + arm['vanilla_dqn']['ent']) / 2.0

    # Solve rate (vanilla, since "where the env is solvable at all").
    spec = SOLVE_THRESHOLDS[env]
    if spec.threshold is not None:
        van_solved = df.filter(
            (pl.col('intervention_name') == 'vanilla_dqn')
            & (pl.col(_OUTCOME_PATH) >= spec.threshold)
        ).height
        van_n = df.filter(pl.col('intervention_name') == 'vanilla_dqn').height
        solve_rate = van_solved / van_n if van_n > 0 else 0.0
    else:
        solve_rate = float('nan')

    covariates: dict[str, float] = {
        'baseline_gap_log': math.log10(max(abs(baseline_gap), 1e-6)),
        'gap_reduction_log_signed': (
            math.copysign(math.log10(max(abs(gap_reduction), 1e-6)), gap_reduction)
        ),
        'out_scale_log': math.log10(out_scale),
        'gap_per_outcome_log': math.log10(
            max(abs(baseline_gap), 1e-6) / out_scale,
        ),
        'state_entropy_mean': state_entropy_mean,
        'solve_rate': solve_rate,
    }
    return g, se, n, covariates


def _print_result(label: str, res: MetaRegressionResult) -> None:
    print(f'\n--- meta_regression: link_g ~ {label} ---')
    print(f'  n_strata={res.n_strata}  R²={res.r_squared:+.3f}')
    print(f'  intercept={res.intercept:+.3f}')
    for c in res.coefficients:
        sig = '✓' if c.is_significant else ' '
        print(
            f'  {c.name:<28} β={c.coefficient:+.3f}  '
            f'CI=[{c.ci_lo:+.3f}, {c.ci_hi:+.3f}]  '
            f'p={c.p_value:.3f}  {sig}'
        )
    if res.cleavage_axes:
        print(f'  CLEAVAGE: {", ".join(res.cleavage_axes)}')
    else:
        print('  no cleavage axis (no significant covariate)')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--covariate', default=None,
        help='single covariate to regress on (default: pairwise + multi)',
    )
    args = parser.parse_args()

    print('=' * 100)
    print('Scope search — DDQN link strength regressed on per-env covariates')
    print(f'  outcome: {_OUTCOME_PATH}  link predicted: a_gt_b (DDQN ≥ vanilla)')
    print('=' * 100)
    envs = envs_with_threshold()
    obs: list[StratumObservation] = []
    print()
    print(
        f'  {"env":<25} {"n":<3} {"g":>7} {"se":>5} '
        f'{"base_gap_log":>13} {"gap_red":>10} {"out_log":>8} '
        f'{"g/out_log":>10} {"ent":>6} {"solve":>6}'
    )
    print('-' * 110)
    for env in envs:
        result = _per_env_g(env)
        if result is None:
            print(f'  {env:<25} (skipped — degenerate)')
            continue
        g, se, n, cov = result
        print(
            f'  {env:<25} {n:<3} '
            f'{g:>+7.3f} {se:>5.2f} '
            f'{cov["baseline_gap_log"]:>+13.2f} '
            f'{cov["gap_reduction_log_signed"]:>+10.2f} '
            f'{cov["out_scale_log"]:>+8.2f} '
            f'{cov["gap_per_outcome_log"]:>+10.2f} '
            f'{cov["state_entropy_mean"]:>+6.2f} '
            f'{cov["solve_rate"]:>6.2f}'
        )
        obs.append(StratumObservation(
            stratum_id=env, g=g, se=se, covariates=cov,
        ))

    if len(obs) < 4:
        print(f'\nToo few valid envs ({len(obs)}); cannot meta-regress.')
        return

    if args.covariate:
        single = [
            StratumObservation(
                stratum_id=o.stratum_id, g=o.g, se=o.se,
                covariates={args.covariate: o.covariates[args.covariate]},
            )
            for o in obs
        ]
        _print_result(args.covariate, meta_regression(single))
        return

    # Pairwise — one covariate at a time, n=10ish so we can afford
    # a few of these without burning dof.
    candidates = (
        'baseline_gap_log',
        'gap_reduction_log_signed',
        'out_scale_log',
        'gap_per_outcome_log',
        'state_entropy_mean',
        'solve_rate',
    )
    for c in candidates:
        single = [
            StratumObservation(
                stratum_id=o.stratum_id, g=o.g, se=o.se,
                covariates={c: o.covariates[c]},
            )
            for o in obs
        ]
        try:
            _print_result(c, meta_regression(single))
        except ValueError as e:
            print(f'\n--- {c}: skipped ({e}) ---')


if __name__ == '__main__':
    main()
