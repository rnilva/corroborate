"""Per-burst panel partial regression — same question as
`run_link_moderator_partial_regression.py` but at the
(env, burst) panel level for more degrees of freedom.

Per `findings_chain_bottlenecks` 2026-05-02: per-env panel
(N=14-18) is underpowered for >2 covariates; per-burst panel
(N≈149) gives proper df. Cluster SE by env is still required
because bursts within env share trajectory.

Per-burst paired g(outcome) restricted to mech-HELD bursts.
Env-level covariates (bf, stale) since they're env-structural.
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


def _per_burst_panel(df: pl.DataFrame) -> list[dict]:
    """Per (env, burst) compute paired g(outcome) using mc_return
    per burst as outcome, restricted to mech-HELD pairs at each
    burst (Δ_jens_per_burst < 0)."""
    pair_keys = ['env_name', 'gamma', 'total_steps', 'sync_period', 'seed']
    rows = []
    for env in sorted(df['env_name'].unique()):
        sub = df.filter(pl.col('env_name') == env)
        v = sub.filter(pl.col('arm_key') == 'baseline').select(
            pair_keys + ['mc_return__mean_axis_-1', 'jensen_bias_per_eps__mean_axis_-1',
                         'bootstrap_fraction', 'target_staleness_late', 'log_obs_dim']
        )
        d = sub.filter(pl.col('arm_key') == DDQN).select(
            pair_keys + ['mc_return__mean_axis_-1', 'jensen_bias_per_eps__mean_axis_-1']
        )
        v = v.rename({'mc_return__mean_axis_-1': 'mc_v', 'jensen_bias_per_eps__mean_axis_-1': 'jens_v'})
        d = d.rename({'mc_return__mean_axis_-1': 'mc_d', 'jensen_bias_per_eps__mean_axis_-1': 'jens_d'})
        j = v.join(d, on=pair_keys, how='inner').filter(
            pl.col('bootstrap_fraction').is_finite()
            & pl.col('target_staleness_late').is_finite()
        )
        if j.height == 0:
            continue
        # Per-cell env features
        bf_env = float(j['bootstrap_fraction'].mean())
        stale_env = float(j['target_staleness_late'].mean())
        log_obs = float(j['log_obs_dim'][0]) if j.height > 0 else float('nan')

        # Per-burst arrays — each cell has a list of per-burst means
        for cell in j.iter_rows(named=True):
            mc_v = cell['mc_v']
            mc_d = cell['mc_d']
            jens_v = cell['jens_v']
            jens_d = cell['jens_d']
            if not all(isinstance(x, list) for x in (mc_v, mc_d, jens_v, jens_d)):
                continue
            n_bursts = min(len(mc_v), len(mc_d), len(jens_v), len(jens_d))
            for b in range(n_bursts):
                rows.append({
                    'env': env, 'burst': b, 'seed': cell['seed'],
                    'd_o': mc_d[b] - mc_v[b],
                    'd_jens': jens_d[b] - jens_v[b],
                    'bf_env': bf_env, 'stale_env': stale_env,
                    'log_obs': log_obs,
                })
    return rows


def _per_env_burst_g(rows: list[dict]) -> list[dict]:
    """Aggregate per (env, burst) to a paired-g panel restricted to
    mech-HELD pairs at that burst."""
    by_eb: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in rows:
        if not (math.isfinite(r['d_o']) and math.isfinite(r['d_jens'])):
            continue
        by_eb[(r['env'], r['burst'])].append(r)

    panel = []
    for (env, b), pairs in by_eb.items():
        # Mech-HELD: Δ_jens < 0
        mech_held = [p for p in pairs if p['d_jens'] < 0]
        if len(mech_held) < 5:
            continue
        d_o_arr = np.array([p['d_o'] for p in mech_held])
        if d_o_arr.std(ddof=1) == 0:
            continue
        n = len(d_o_arr)
        sd = d_o_arr.std(ddof=1)
        g_raw = float(d_o_arr.mean()) / sd
        j_corr = 1.0 - 3.0 / (4.0 * (n - 1) - 1.0) if n > 2 else 1.0
        g = j_corr * g_raw
        # SE of g
        se_g = math.sqrt(1.0 / n + g_raw ** 2 / (2.0 * n))
        panel.append({
            'env': env, 'burst': b, 'g_link': g, 'se_g': se_g,
            'n_pairs': n,
            'bf_env': mech_held[0]['bf_env'],
            'stale_env': mech_held[0]['stale_env'],
            'log_obs': mech_held[0]['log_obs'],
        })
    return panel


def _ols_with_cluster_se(
    X: np.ndarray, y: np.ndarray, cluster_ids: list[str],
) -> dict:
    n, k = X.shape
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
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

    rows = _per_burst_panel(df)
    print(f'per-(env, burst, seed) rows: {len(rows)}')
    panel = _per_env_burst_g(rows)
    print(f'(env, burst) g_link panel (mech-HELD): {len(panel)}')

    # Cross-env collinearity
    panel_envs = sorted(set(p['env'] for p in panel))
    bf_per_env = {p['env']: p['bf_env'] for p in panel}
    stale_per_env = {p['env']: p['stale_env'] for p in panel}
    bf_arr = np.array([bf_per_env[e] for e in panel_envs])
    st_arr = np.array([stale_per_env[e] for e in panel_envs])
    rho = float(np.corrcoef(bf_arr, st_arr)[0, 1])
    print(f'cross-env ρ(bf, stale) = {rho:+.3f} (collinearity check)')

    # Build regression matrices
    g_arr = np.array([p['g_link'] for p in panel])
    bf_arr = np.array([p['bf_env'] for p in panel])
    stale_arr = np.array([p['stale_env'] for p in panel])
    log_obs_arr = np.array([p['log_obs'] for p in panel])
    cluster_ids = [p['env'] for p in panel]
    bf_z = (bf_arr - bf_arr.mean()) / bf_arr.std()
    stale_z = (stale_arr - stale_arr.mean()) / stale_arr.std()
    log_obs_z = (log_obs_arr - log_obs_arr.mean()) / log_obs_arr.std()
    n = len(panel)
    n_envs = len(set(cluster_ids))

    print(f'\nN obs = {n}, N envs (clusters) = {n_envs}')
    print()
    print('=== Per-burst panel + cluster SE by env ===')

    for label, X in [
        ('m1 g_link ~ bf', np.column_stack([np.ones(n), bf_z])),
        ('m2 g_link ~ stale', np.column_stack([np.ones(n), stale_z])),
        ('m3 g_link ~ bf + stale', np.column_stack([np.ones(n), bf_z, stale_z])),
        ('m4 g_link ~ bf + stale + log_obs', np.column_stack([np.ones(n), bf_z, stale_z, log_obs_z])),
        ('m5 g_link ~ bf + stale + bf*stale', np.column_stack([np.ones(n), bf_z, stale_z, bf_z * stale_z])),
    ]:
        r = _ols_with_cluster_se(X, g_arr, cluster_ids)
        names = ['intercept', 'bf', 'stale', 'log_obs/inter']
        print(f'\n{label}:')
        for i, b in enumerate(r['beta']):
            nm = names[i] if i < len(names) else f'x{i}'
            print(f'  {nm:<14} β={b:+.4f}  t={r["t"][i]:+.2f}  p={r["p"][i]:.4g}')

    out_path = Path(
        'experiments/findings/sync_curve_breakout/'
        'link_moderator_per_burst.json',
    )
    out_path.write_text(json.dumps({
        'rho_bf_stale_cross_env': rho,
        'panel_size': n, 'n_envs': n_envs,
        'panel': panel,
    }, indent=2, default=str))
    print(f'\nwrote: {out_path}')


if __name__ == '__main__':
    main()
