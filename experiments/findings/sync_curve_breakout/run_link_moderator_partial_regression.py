"""Cross-env partial regression on the link panel: does
target_staleness_late absorb bootstrap_fraction, or are they
independent link predictors?

The standing residual from `findings_residual_unexplained` and
`findings_chain_bottlenecks`: bootstrap_fraction → g_link is
robust after 7+ controls. But bf is L-derived (≈ 1 − 1/E[L]),
similar to polarity — a length-channel proxy. Question: does
the polarity-orthogonal mediator `target_staleness_late`
absorb bf's predictive power, or do both survive?

Three possible outcomes:
1. **Both survive** (no interaction) → 2 independent channels.
2. **Stale absorbs bf** (bf null after adding stale) → bf was
   a length-channel proxy for the staleness chain.
3. **bf absorbs stale** → bf captures something deeper.

Method: per-env meta-regression on g_link, restricted to
mech-HELD pairs (Δ_jens < 0). Cluster-robust SE clustering by
env_name, per `findings_chain_bottlenecks` 2026-05-02 lesson.

Variables:
- target: paired-g of `eval_best_burst_mean` per env (mech-HELD)
- predictors: env-mean bootstrap_fraction, env-mean
  target_staleness_late, optional log_obs_dim, log_horizon

Output: panel JSON + table.
"""
from __future__ import annotations

import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import t as t_dist

DDQN = 'bootstrap=partial(Claim:bootstrap;greedification=Claim:double_greedify)'


def _per_env_g_link_mech_held(df: pl.DataFrame) -> dict[str, dict]:
    """For each env, compute paired g(outcome) restricted to
    mech-HELD pairs (Δ_jens < 0). Returns dict env → {g, se,
    n_pairs, env_features}."""
    out: dict[str, dict] = {}
    pair_keys = ['env_name', 'gamma', 'total_steps', 'sync_period', 'seed']
    cols = ['eval_best_burst_mean', 'jensen_gap', 'bootstrap_fraction',
            'target_staleness_late', 'log_obs_dim', 'log_horizon',
            'effective_horizon']
    for env in sorted(df['env_name'].unique()):
        sub = df.filter(pl.col('env_name') == env)
        v = sub.filter(pl.col('arm_key') == 'baseline').select(pair_keys + cols).rename(
            {c: f'{c}_v' for c in cols}
        )
        d = sub.filter(pl.col('arm_key') == DDQN).select(pair_keys + cols).rename(
            {c: f'{c}_d' for c in cols}
        )
        j = v.join(d, on=pair_keys, how='inner').filter(
            pl.col('eval_best_burst_mean_v').is_finite()
            & pl.col('eval_best_burst_mean_d').is_finite()
            & pl.col('jensen_gap_v').is_finite()
            & pl.col('jensen_gap_d').is_finite()
        )
        if j.height < 5:
            continue
        d_o = (j['eval_best_burst_mean_d'] - j['eval_best_burst_mean_v']).to_numpy()
        d_jens = (j['jensen_gap_d'] - j['jensen_gap_v']).to_numpy()
        # Mech-HELD: Δ_jens < 0
        mech_mask = d_jens < 0
        if mech_mask.sum() < 5:
            continue
        d_o_h = d_o[mech_mask]
        n = len(d_o_h)
        if d_o_h.std() == 0:
            continue
        sd = d_o_h.std(ddof=1)
        g_raw = float(d_o_h.mean()) / sd
        # Hedges' correction
        j_corr = 1.0 - 3.0 / (4.0 * (n - 1) - 1.0) if n > 2 else 1.0
        g = j_corr * g_raw
        se = math.sqrt(1.0 / n + g_raw ** 2 / (2.0 * n))
        # Env-mean features (baseline arm only, for stability)
        bf_env = float(j.filter(d_jens < 0)['bootstrap_fraction_v'].mean())
        stale_env = float(j.filter(d_jens < 0)['target_staleness_late_v'].mean())
        log_obs_env = float(j['log_obs_dim_v'][0]) if j.height > 0 else float('nan')
        log_horizon_env = float(j['log_horizon_v'][0]) if j.height > 0 else float('nan')
        out[env] = {
            'env': env,
            'g_link': g, 'se_g': se, 'n_pairs_mech_held': n,
            'bf_env_mean': bf_env,
            'stale_env_mean': stale_env,
            'log_obs_dim': log_obs_env,
            'log_horizon': log_horizon_env,
        }
    return out


def _ols_with_cluster_se(
    X: np.ndarray, y: np.ndarray, cluster_ids: list[str],
) -> dict:
    """OLS + Liang-Zeger cluster-robust SE clustering by
    cluster_ids. Returns dict with beta, se, t, p per coef."""
    n, k = X.shape
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    # OLS sandwich: (X'X)^-1 (X' meat X) (X'X)^-1 with
    # meat = sum_g (X_g' e_g e_g' X_g) over clusters g
    XtX_inv = np.linalg.pinv(X.T @ X)
    clusters = sorted(set(cluster_ids))
    meat = np.zeros((k, k))
    for c in clusters:
        idx = [i for i, x in enumerate(cluster_ids) if x == c]
        if not idx:
            continue
        Xg = X[idx, :]
        eg = resid[idx]
        Xe = Xg.T @ eg
        meat += np.outer(Xe, Xe)
    n_clust = len(clusters)
    # CR1 small-sample adjustment
    finite_correction = (n_clust / max(n_clust - 1, 1)) * ((n - 1) / max(n - k, 1))
    cov_cr1 = finite_correction * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(cov_cr1))
    t_stats = beta / se
    df_clust = n_clust - 1
    p_vals = 2 * (1 - t_dist.cdf(np.abs(t_stats), df=df_clust))
    return {
        'beta': beta.tolist(), 'se': se.tolist(),
        't': t_stats.tolist(), 'p': p_vals.tolist(),
        'n_obs': n, 'n_clusters': n_clust,
    }


def main() -> None:
    runs = pl.read_parquet('experiments/data/ddqn/runs.parquet')
    ms = pl.read_parquet('experiments/data/ddqn/measurements.parquet')
    collide = [c for c in ms.columns if c in runs.columns and c != 'id']
    if collide:
        runs = runs.drop(collide)
    df = runs.join(ms, on='id', how='inner')
    print(f'cells: {df.height}, envs: {df["env_name"].n_unique()}')

    panel = _per_env_g_link_mech_held(df)
    print(f'\nper-env mech-HELD g_link panel: {len(panel)} envs')
    print(f'{"env":<26} {"n":>4} {"g_link":>9} {"se":>8} {"bf":>8} {"stale":>10} {"log_obs":>9}')
    print('-' * 85)
    rows = []
    for env, info in sorted(panel.items()):
        print(f'{env:<26} {info["n_pairs_mech_held"]:>4} '
              f'{info["g_link"]:>+9.3f} {info["se_g"]:>8.3f} '
              f'{info["bf_env_mean"]:>8.4f} {info["stale_env_mean"]:>10.6f} '
              f'{info["log_obs_dim"]:>9.3f}')
        rows.append(info)

    # Filter to only envs where bf and stale are finite
    rows = [r for r in rows if not any(
        math.isnan(v) for v in (r['bf_env_mean'], r['stale_env_mean'], r['g_link'])
    )]
    print(f'\nusable rows: {len(rows)}')

    if len(rows) < 4:
        print('insufficient envs for regression')
        return

    # Build panel for regression
    g = np.array([r['g_link'] for r in rows], dtype=np.float64)
    bf = np.array([r['bf_env_mean'] for r in rows], dtype=np.float64)
    stale = np.array([r['stale_env_mean'] for r in rows], dtype=np.float64)
    cluster_ids = [r['env'] for r in rows]
    # Standardize predictors so coefficients are comparable
    bf_z = (bf - bf.mean()) / bf.std() if bf.std() > 0 else bf
    stale_z = (stale - stale.mean()) / stale.std() if stale.std() > 0 else stale

    n = len(rows)
    print()
    print('=== Models on n_envs={} (cluster SE by env) ==='.format(n))

    results = {}
    # Model 1: marginal — bf only
    X = np.column_stack([np.ones(n), bf_z])
    r1 = _ols_with_cluster_se(X, g, cluster_ids)
    print(f'\nModel 1 (g_link ~ bf):')
    print(f'  intercept: β={r1["beta"][0]:+.4f} t={r1["t"][0]:+.2f} p={r1["p"][0]:.4g}')
    print(f'  bf:        β={r1["beta"][1]:+.4f} t={r1["t"][1]:+.2f} p={r1["p"][1]:.4g}')
    results['m1_bf'] = r1

    # Model 2: marginal — stale only
    X = np.column_stack([np.ones(n), stale_z])
    r2 = _ols_with_cluster_se(X, g, cluster_ids)
    print(f'\nModel 2 (g_link ~ stale):')
    print(f'  intercept: β={r2["beta"][0]:+.4f} t={r2["t"][0]:+.2f} p={r2["p"][0]:.4g}')
    print(f'  stale:     β={r2["beta"][1]:+.4f} t={r2["t"][1]:+.2f} p={r2["p"][1]:.4g}')
    results['m2_stale'] = r2

    # Model 3: joint
    X = np.column_stack([np.ones(n), bf_z, stale_z])
    r3 = _ols_with_cluster_se(X, g, cluster_ids)
    print(f'\nModel 3 (g_link ~ bf + stale):')
    print(f'  intercept: β={r3["beta"][0]:+.4f} t={r3["t"][0]:+.2f} p={r3["p"][0]:.4g}')
    print(f'  bf:        β={r3["beta"][1]:+.4f} t={r3["t"][1]:+.2f} p={r3["p"][1]:.4g}')
    print(f'  stale:     β={r3["beta"][2]:+.4f} t={r3["t"][2]:+.2f} p={r3["p"][2]:.4g}')
    results['m3_joint'] = r3

    # Model 4: with interaction
    X = np.column_stack([np.ones(n), bf_z, stale_z, bf_z * stale_z])
    r4 = _ols_with_cluster_se(X, g, cluster_ids)
    print(f'\nModel 4 (g_link ~ bf + stale + bf×stale):')
    print(f'  intercept: β={r4["beta"][0]:+.4f} t={r4["t"][0]:+.2f} p={r4["p"][0]:.4g}')
    print(f'  bf:        β={r4["beta"][1]:+.4f} t={r4["t"][1]:+.2f} p={r4["p"][1]:.4g}')
    print(f'  stale:     β={r4["beta"][2]:+.4f} t={r4["t"][2]:+.2f} p={r4["p"][2]:.4g}')
    print(f'  bf×stale:  β={r4["beta"][3]:+.4f} t={r4["t"][3]:+.2f} p={r4["p"][3]:.4g}')
    results['m4_interaction'] = r4

    # Outcome reading
    print()
    print('=== Reading ===')
    bf_marginal = r1['beta'][1]
    stale_marginal = r2['beta'][1]
    bf_joint = r3['beta'][1]
    stale_joint = r3['beta'][2]
    print(f'bf marginal slope:    {bf_marginal:+.3f} (p={r1["p"][1]:.3g})')
    print(f'bf joint slope:       {bf_joint:+.3f} (p={r3["p"][1]:.3g}) — drop {(1-bf_joint/bf_marginal)*100:+.0f}%')
    print(f'stale marginal slope: {stale_marginal:+.3f} (p={r2["p"][1]:.3g})')
    print(f'stale joint slope:    {stale_joint:+.3f} (p={r3["p"][2]:.3g}) — drop {(1-stale_joint/stale_marginal)*100:+.0f}%')

    out_path = Path(
        'experiments/findings/sync_curve_breakout/'
        'link_moderator_partial.json',
    )
    out_path.write_text(json.dumps({
        'panel': rows, 'models': results,
    }, indent=2, default=str))
    print(f'\nwrote: {out_path}')


if __name__ == '__main__':
    main()
