"""Per-burst time-series summary across all envs in a corpus.

Computes the per-burst within-pair Pearson r(Δjensen_bias,
Δreturn) for every env in the supplied corpus, paired DDQN-vs-
vanilla by seed. Prints one row per (env, burst) plus a summary
line per env.

Builds on the same primitive used in `analyze_fourrooms_mediators`
but runs across the full env list — answers "does the per-burst
within-pair coupling pattern hold beyond the action_dim_* corpora?"

Usage:
  uv run python experiments/analyze_per_burst_summary.py
  uv run python experiments/analyze_per_burst_summary.py --corpus ddqn
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import scipy.stats as ss


def _load(
    corpus: str, env: str,
    treatment_arm: str = 'ddqn',
    baseline_arm: str = 'vanilla_dqn',
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
] | None:
    """Read treatment + baseline cells for `env` in `corpus`;
    return per-cell per-burst (bias, return) arrays paired by
    seed.

    Returns (van_bias, van_ret, ddqn_bias, ddqn_ret), each shape
    `(n_pairs, n_bursts)` (van=baseline, ddqn=treatment in
    variable names). Returns None if corpus is missing data or
    no paired cells exist for the env."""
    base = Path('experiments/data') / corpus
    runs_path = base / 'runs.parquet'
    if not runs_path.exists():
        runs_path = base / 'runs_with_mediators.parquet'
    if not runs_path.exists():
        return None
    traces_path = base / 'traces.parquet'
    if not traces_path.exists():
        return None
    runs_df = pl.read_parquet(str(runs_path)).filter(
        pl.col('env_name') == env
    )
    if runs_df.height == 0:
        return None
    # Collect ids per arm.
    ddqn_ids = runs_df.filter(
        pl.col('intervention_name') == treatment_arm
    ).select(['id', 'seed']).to_dicts()
    van_ids = runs_df.filter(
        pl.col('intervention_name') == baseline_arm
    ).select(['id', 'seed']).to_dicts()
    if not ddqn_ids or not van_ids:
        return None

    # Read trace columns for these ids.
    all_ids = [d['id'] for d in ddqn_ids] + [d['id'] for d in van_ids]
    trace_df = pl.read_parquet(
        str(traces_path),
        columns=['id', 'predicted_q_at_start', 'mc_return'],
    ).filter(pl.col('id').is_in(all_ids))
    bias_by_id: dict[str, np.ndarray] = {}
    return_by_id: dict[str, np.ndarray] = {}
    for row in trace_df.iter_rows(named=True):
        cid = row['id']
        pred = np.asarray(row['predicted_q_at_start'], dtype=np.float64)
        actual = np.asarray(row['mc_return'], dtype=np.float64)
        if pred.ndim != 2 or actual.ndim != 2 or pred.shape != actual.shape:
            continue
        bias_by_id[cid] = (pred - actual).mean(axis=-1)
        return_by_id[cid] = actual.mean(axis=-1)

    # Pair by seed.
    seed_to_van = {d['seed']: d['id'] for d in van_ids}
    seed_to_ddqn = {d['seed']: d['id'] for d in ddqn_ids}
    common_seeds = sorted(set(seed_to_van) & set(seed_to_ddqn))
    common_seeds = [
        s for s in common_seeds
        if seed_to_van[s] in bias_by_id and seed_to_ddqn[s] in bias_by_id
    ]
    if not common_seeds:
        return None
    van_bias = np.stack([bias_by_id[seed_to_van[s]] for s in common_seeds])
    van_ret = np.stack([return_by_id[seed_to_van[s]] for s in common_seeds])
    ddqn_bias = np.stack([bias_by_id[seed_to_ddqn[s]] for s in common_seeds])
    ddqn_ret = np.stack([return_by_id[seed_to_ddqn[s]] for s in common_seeds])
    return van_bias, van_ret, ddqn_bias, ddqn_ret


def _per_env_summary(env: str, arrays: tuple[np.ndarray, ...]) -> tuple[
    str, int, int, list[float], list[float], list[float], list[float]
]:
    """Return (env, n_pairs, n_bursts, per_burst_r, per_burst_p,
    per_burst_dbias_mean, per_burst_dret_mean)."""
    van_bias, van_ret, ddqn_bias, ddqn_ret = arrays
    n_pairs, n_bursts = van_bias.shape
    delta_bias = ddqn_bias - van_bias
    delta_ret = ddqn_ret - van_ret
    r_per_burst: list[float] = []
    p_per_burst: list[float] = []
    db_mean: list[float] = []
    dr_mean: list[float] = []
    for b in range(n_bursts):
        db = delta_bias[:, b]
        dr = delta_ret[:, b]
        if float(db.std()) == 0.0 or float(dr.std()) == 0.0:
            r_per_burst.append(float('nan'))
            p_per_burst.append(float('nan'))
        else:
            r = ss.pearsonr(db, dr)
            r_per_burst.append(float(r.statistic))
            p_per_burst.append(float(r.pvalue))
        db_mean.append(float(db.mean()))
        dr_mean.append(float(dr.mean()))
    return env, n_pairs, n_bursts, r_per_burst, p_per_burst, db_mean, dr_mean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', default='ddqn')
    parser.add_argument('--total-steps', type=int, default=200000)
    parser.add_argument('--treatment-arm', default='ddqn')
    parser.add_argument('--baseline-arm', default='vanilla_dqn')
    args = parser.parse_args()
    corpus: str = args.corpus
    treatment_arm: str = args.treatment_arm
    baseline_arm: str = args.baseline_arm

    runs_path = Path('experiments/data') / corpus / 'runs.parquet'
    if not runs_path.exists():
        runs_path = (
            Path('experiments/data') / corpus / 'runs_with_mediators.parquet'
        )
    df = pl.read_parquet(str(runs_path)).filter(
        pl.col('total_steps') == args.total_steps,
    )
    envs = sorted(df['env_name'].unique().to_list())
    print(f'corpus={corpus}  envs={len(envs)}  total_steps={args.total_steps}')
    print()

    rows: list[tuple] = []
    for env in envs:
        arrays = _load(corpus, env, treatment_arm, baseline_arm)
        if arrays is None or arrays[0].shape[0] < 4:
            print(f'  {env}: skipped (insufficient pairs)')
            continue
        rows.append(_per_env_summary(env, arrays))

    # Header.
    print()
    print(f'{"env":<25} {"|pairs|":>7} '
          f'{"min_r":>7} {"max_r":>7} {"mean_r":>7} '
          f'{"sig_neg":>8} {"sig_pos":>8} '
          f'{"mean_db":>10} {"mean_dr":>10}')
    print('-' * 105)
    for env, n, nb, r_list, p_list, db, dr in rows:
        finite_r = [r for r in r_list if r == r]
        if not finite_r:
            continue
        sig_neg = sum(
            1 for i, r in enumerate(r_list)
            if r == r and r < 0 and p_list[i] < 0.05
        )
        sig_pos = sum(
            1 for i, r in enumerate(r_list)
            if r == r and r > 0 and p_list[i] < 0.05
        )
        print(
            f'{env:<25} {n:>7} '
            f'{min(finite_r):>+7.3f} {max(finite_r):>+7.3f} '
            f'{float(np.mean(finite_r)):>+7.3f} '
            f'{sig_neg:>8} {sig_pos:>8} '
            f'{float(np.mean(db)):>+10.3f} '
            f'{float(np.mean(dr)):>+10.3f}'
        )

    # ============ Cross-burst lag correlation ============
    # For each env, compute Pearson r(Δbias[k], Δret[k+τ]) across
    # all valid (pair, k) pairs, for τ ∈ {-3, -2, -1, 0, +1, +2, +3}.
    # Forward asymmetry (τ > 0 stronger than τ < 0) is consistent
    # with bias-reduction temporally preceding outcome benefit.
    print()
    print(f'{"env":<25} ' + ' '.join(f'τ={t:>+2}' for t in (-3, -2, -1, 0, 1, 2, 3)))
    print('-' * 95)
    for env in envs:
        arrays = _load(corpus, env, treatment_arm, baseline_arm)
        if arrays is None or arrays[0].shape[0] < 4:
            continue
        van_bias, van_ret, ddqn_bias, ddqn_ret = arrays
        delta_bias = ddqn_bias - van_bias
        delta_ret = ddqn_ret - van_ret
        n_pairs, n_bursts = delta_bias.shape
        cells: list[str] = []
        for tau in (-3, -2, -1, 0, 1, 2, 3):
            xs: list[float] = []
            ys: list[float] = []
            for k in range(n_bursts):
                k2 = k + tau
                if k2 < 0 or k2 >= n_bursts:
                    continue
                xs.extend(delta_bias[:, k].tolist())
                ys.extend(delta_ret[:, k2].tolist())
            if (
                len(xs) < 4
                or float(np.std(xs)) == 0.0
                or float(np.std(ys)) == 0.0
            ):
                cells.append('   nan')
                continue
            r = ss.pearsonr(np.asarray(xs), np.asarray(ys))
            cells.append(f'{r.statistic:>+5.2f}')
        print(f'{env:<25} ' + ' '.join(f'{c:>5}' for c in cells))


if __name__ == '__main__':
    main()
