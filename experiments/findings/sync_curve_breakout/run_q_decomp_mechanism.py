"""Algorithmic decomposition of why Q-regime sign inverts DDQN's
effect on outcome: per-step DDQN-correction magnitude.

The new measurable `ddqn_bootstrap_gap_late = mean over late 50%
of (max(target_q) − target_q[argmax_online])` quantifies HOW MUCH
DDQN's bootstrap value diverges from vanilla's per step.

Predictions:

1. Across τ: more staleness → larger gap (online and target
   diverge → max_target and target[argmax_online] diverge).
   Universal.

2. Across Q-regime: gap should grow with staleness IN BOTH
   regimes. The asymmetric outcome effect must come from how
   the gap maps to outcome — which DEPENDS on Q-regime sign.

3. Direct test: regression
   Δ_outcome = β₀ + β_g·gap + β_q·q_late_mean
                + β_int·(gap × q_late_mean) + ε
   Predicts: β_int significantly nonzero — gap's effect on
   outcome depends on Q-regime sign.
"""
from __future__ import annotations

import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import t as t_dist, spearmanr

import corroborate_rl.dqn.measurables  # register

DDQN = 'bootstrap=partial(Claim:bootstrap;greedification=Claim:double_greedify)'


def main() -> None:
    base = Path('experiments/data/polyak_tau_q_decomp')
    panels = []
    for sub in sorted(base.iterdir()):
        if not sub.is_dir() or not (sub / 'measurements.parquet').exists():
            continue
        runs = pl.read_parquet(sub / 'runs.parquet')
        ms = pl.read_parquet(sub / 'measurements.parquet')
        collide = [c for c in ms.columns if c in runs.columns and c != 'id']
        if collide:
            runs = runs.drop(collide)
        df = runs.join(ms, on='id', how='left')
        panels.append(df)
    df = pl.concat(panels, how='diagonal_relaxed')
    print(f'cells: {df.height}')
    print(df.group_by(['env_name', 'arm_key']).agg(pl.len().alias('n')).sort(['env_name','arm_key']))

    # Per-cell summary
    print()
    print('=== Per-(env, arm, τ) ddqn_bootstrap_gap_late ===\n')
    print(f'{"env":<20} {"arm":<10} {"τ":>7} {"gap_late mean":>15} {"q_late mean":>14}')
    print('-' * 75)
    agg = df.group_by(['env_name', 'arm_key', 'target_sync.tau']).agg(
        pl.col('ddqn_bootstrap_gap_late').mean().alias('gap'),
        pl.col('q_late_mean').mean().alias('q'),
        pl.len().alias('n'),
    ).sort(['env_name', 'arm_key', 'target_sync.tau'])
    for row in agg.iter_rows(named=True):
        env = row['env_name']
        arm = 'baseline' if row['arm_key'] == 'baseline' else 'ddqn'
        print(f'{env:<20} {arm:<10} {row["target_sync.tau"]:>7.3f} '
              f'{row["gap"]:>+15.4f} {row["q"]:>+14.3f}')

    # Pair on (env, τ, seed); compute Δ_outcome AND read baseline's gap_late
    pair_keys = ['env_name', 'gamma', 'sync_period', 'total_steps', 'seed', 'target_sync.tau']
    select_cols = ['eval_best_burst_mean', 'ddqn_bootstrap_gap_late', 'q_late_mean', 'target_staleness_late']
    v = df.filter(pl.col('arm_key')=='baseline').select(pair_keys + select_cols).rename(
        {c: f'{c}_v' for c in select_cols}
    )
    d = df.filter(pl.col('arm_key')==DDQN).select(pair_keys + ['eval_best_burst_mean']).rename(
        {'eval_best_burst_mean': 'eval_best_burst_mean_d'}
    )
    j = v.join(d, on=pair_keys, how='inner').filter(
        pl.col('eval_best_burst_mean_v').is_finite() & pl.col('eval_best_burst_mean_d').is_finite()
    ).filter(
        pl.col('ddqn_bootstrap_gap_late_v').is_finite()
        & pl.col('q_late_mean_v').is_finite()
    )
    print()
    print(f'paired cells with all measurables: {j.height}')

    # ===== TEST: gap × q_late_mean interaction on Δ_outcome =====
    do = (j['eval_best_burst_mean_d'] - j['eval_best_burst_mean_v']).to_numpy()
    gap = j['ddqn_bootstrap_gap_late_v'].to_numpy()
    q = j['q_late_mean_v'].to_numpy()

    n = len(do)
    X = np.column_stack([np.ones(n), gap, q, gap * q])
    beta, _, rank, _ = np.linalg.lstsq(X, do, rcond=None)
    y_pred = X @ beta
    resid = do - y_pred
    df_resid = n - rank
    sigma2 = float((resid ** 2).sum() / df_resid)
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t_stats = beta / se
    p_vals = 2 * (1.0 - t_dist.cdf(np.abs(t_stats), df=df_resid))

    print()
    print('=== TEST: Δ_outcome = β₀ + β_g·gap + β_q·q + β_int·(gap × q) + ε ===')
    print()
    print(f'{"term":<22} {"β":>14} {"SE":>12} {"t":>8} {"p":>10}')
    print('-' * 80)
    names = ['intercept', 'ddqn_bootstrap_gap_late', 'q_late_mean', 'gap × q_late_mean']
    for nm, b, s, t, p in zip(names, beta, se, t_stats, p_vals):
        print(f'{nm:<22} {b:>+14.5f} {s:>12.5f} {t:>+8.3f} {p:>10.4g}')

    # ===== Per-Q-regime stratified gap → outcome =====
    print()
    print('=== Per-Q-regime: ATE(gap → Δ_outcome) ===\n')
    print(f'{"regime":<12} {"n":>4} {"slope":>10} {"se":>10} {"t":>8} {"p":>10}')
    print('-' * 60)
    for regime_label, mask in [('q > 0', q > 0), ('q < 0', q < 0)]:
        if mask.sum() < 10: continue
        gx = gap[mask]
        gy = do[mask]
        if gx.std() == 0: continue
        # OLS slope
        cov_xy = float(np.cov(gx, gy, ddof=1)[0, 1])
        var_x = float(np.var(gx, ddof=1))
        slope = cov_xy / var_x
        intercept = float(gy.mean()) - slope * float(gx.mean())
        y_pred_r = slope * gx + intercept
        resid_r = gy - y_pred_r
        ss_x = float(((gx - gx.mean()) ** 2).sum())
        sigma_r = math.sqrt(float((resid_r ** 2).sum() / max(len(gx) - 2, 1)))
        slope_se = sigma_r / math.sqrt(ss_x) if ss_x > 0 else float('nan')
        t_slope = slope / slope_se if slope_se > 0 else float('nan')
        from scipy.stats import t as t_dist2
        p_slope = 2 * (1 - t_dist2.cdf(abs(t_slope), df=len(gx) - 2))
        print(f'{regime_label:<12} {int(mask.sum()):>4} {slope:>+10.3f} {slope_se:>10.3f} {t_slope:>+8.3f} {p_slope:>10.4g}')

    out = Path('experiments/findings/sync_curve_breakout/q_decomp_mechanism.json')
    out.write_text(json.dumps({
        'n_pairs': n,
        'beta_intercept': float(beta[0]),
        'beta_gap': float(beta[1]),
        'beta_q': float(beta[2]),
        'beta_interaction': float(beta[3]),
        't_interaction': float(t_stats[3]),
        'p_interaction': float(p_vals[3]),
    }, indent=2, default=str))
    print(f'\nwrote: {out}')


if __name__ == '__main__':
    import math
    main()
