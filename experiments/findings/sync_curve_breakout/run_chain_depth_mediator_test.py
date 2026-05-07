"""Test chain_depth_mediator as the unified link predictor on
existing ddqn corpus data.

Three layers of tests, increasing rigor:

1. **Per-env panel**: env-mean chain_depth_mediator vs g_link
   (mech-HELD). Cross-env correlation.
2. **Joint regression**: chain_depth_mediator vs bf vs stale —
   does the composite ABSORB its constituents?
3. **Per-burst panel** with cluster SE (per `findings_chain_
   bottlenecks` lesson).
"""
from __future__ import annotations

import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import json
import math
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import t as t_dist, spearmanr

import corroborate_rl.dqn.measurables  # register
from corroborate.measurables import compute_missing_columns

DDQN = 'bootstrap=partial(Claim:bootstrap;greedification=Claim:double_greedify)'


def _per_env_g_link(df: pl.DataFrame) -> list[dict]:
    pair_keys = ['env_name', 'gamma', 'total_steps', 'sync_period', 'seed']
    cols = ['eval_best_burst_mean', 'jensen_gap', 'bootstrap_fraction',
            'target_staleness_late', 'chain_depth_mediator', 'effective_horizon']
    panel = []
    for env in sorted(df['env_name'].unique()):
        sub = df.filter(pl.col('env_name') == env)
        v = sub.filter(pl.col('arm_key') == 'baseline').select(pair_keys + cols)
        d = sub.filter(pl.col('arm_key') == DDQN).select(pair_keys + ['eval_best_burst_mean', 'jensen_gap'])
        v = v.rename({c: f'{c}_v' for c in cols})
        d = d.rename({'eval_best_burst_mean': 'out_d', 'jensen_gap': 'jens_d'})
        j = v.join(d, on=pair_keys, how='inner').filter(
            pl.col('eval_best_burst_mean_v').is_finite()
            & pl.col('out_d').is_finite()
            & pl.col('jensen_gap_v').is_finite()
            & pl.col('jens_d').is_finite()
        )
        if j.height < 5: continue
        d_o = (j['out_d'] - j['eval_best_burst_mean_v']).to_numpy()
        d_jens = (j['jens_d'] - j['jensen_gap_v']).to_numpy()
        mech_mask = d_jens < 0
        if mech_mask.sum() < 5: continue
        d_o_h = d_o[mech_mask]
        n = int(mech_mask.sum())
        sd = float(d_o_h.std(ddof=1))
        if sd == 0: continue
        g_raw = float(d_o_h.mean()) / sd
        j_corr = 1.0 - 3.0 / (4.0 * (n - 1) - 1.0) if n > 2 else 1.0
        g = j_corr * g_raw
        # env-level features (mech-HELD baseline cells)
        cdm = float(j.filter(d_jens < 0)['chain_depth_mediator_v'].mean())
        bf = float(j.filter(d_jens < 0)['bootstrap_fraction_v'].mean())
        stale = float(j.filter(d_jens < 0)['target_staleness_late_v'].mean())
        eff_h = float(j.filter(d_jens < 0)['effective_horizon_v'].mean())
        panel.append({
            'env': env, 'g_link': g, 'n': n,
            'cdm': cdm, 'bf': bf, 'stale': stale, 'eff_h': eff_h,
        })
    return panel


def _ols(X: np.ndarray, y: np.ndarray) -> dict:
    """Plain OLS (no cluster), returns coefs + se + t + p."""
    n, k = X.shape
    if n <= k:
        return {}
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    df_r = n - k
    sigma2 = float((resid ** 2).sum() / df_r)
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t_stats = beta / se
    p_vals = 2 * (1 - t_dist.cdf(np.abs(t_stats), df=df_r))
    ss_total = float(((y - y.mean()) ** 2).sum())
    ss_res = float((resid ** 2).sum())
    r2 = 1 - ss_res / ss_total if ss_total > 0 else float('nan')
    return {
        'beta': beta.tolist(), 'se': se.tolist(),
        't': t_stats.tolist(), 'p': p_vals.tolist(),
        'r2': r2, 'n': n, 'k': k,
    }


def main() -> None:
    runs = pl.read_parquet('experiments/data/ddqn/runs.parquet')
    ms = pl.read_parquet('experiments/data/ddqn/measurements.parquet')
    collide = [c for c in ms.columns if c in runs.columns and c != 'id']
    runs = runs.drop(collide)
    df = runs.join(ms, on='id', how='left')
    # Compute chain_depth_mediator if not present
    df = compute_missing_columns(df, ['chain_depth_mediator'])
    print(f'cells: {df.height}, chain_depth_mediator finite: '
          f'{df["chain_depth_mediator"].is_finite().sum()}')

    panel = _per_env_g_link(df)
    print(f'\nper-env mech-HELD panel: {len(panel)} envs')
    print(f'{"env":<26} {"n":>4} {"g_link":>9} {"cdm":>8} {"bf":>8} {"stale":>10} {"eff_h":>9}')
    print('-' * 85)
    for p in sorted(panel, key=lambda r: -r['cdm']):
        print(f'{p["env"]:<26} {p["n"]:>4} {p["g_link"]:>+9.3f} '
              f'{p["cdm"]:>8.4f} {p["bf"]:>8.4f} {p["stale"]:>10.5f} '
              f'{p["eff_h"]:>9.2f}')

    # Cross-env Spearman correlations
    print()
    print('=== Cross-env Spearman ρ vs g_link ===')
    g = np.array([p['g_link'] for p in panel])
    for feat in ('cdm', 'bf', 'stale', 'eff_h'):
        x = np.array([p[feat] for p in panel])
        rho, p_val = spearmanr(x, g)
        print(f'  ρ(g_link, {feat:<8}) = {rho:+.3f}  p={p_val:.4g}  n={len(g)}')

    # Multi-feature regressions (no cluster, since already per-env)
    print()
    print('=== Per-env OLS regressions on g_link (n_envs={}) ==='.format(len(panel)))
    cdm = np.array([p['cdm'] for p in panel])
    bf = np.array([p['bf'] for p in panel])
    stale = np.array([p['stale'] for p in panel])
    eff_h = np.array([p['eff_h'] for p in panel])
    # standardize
    cdm_z = (cdm - cdm.mean()) / cdm.std() if cdm.std() > 0 else cdm
    bf_z = (bf - bf.mean()) / bf.std() if bf.std() > 0 else bf
    stale_z = (stale - stale.mean()) / stale.std() if stale.std() > 0 else stale
    eff_h_z = (eff_h - eff_h.mean()) / eff_h.std() if eff_h.std() > 0 else eff_h
    n = len(panel)

    for label, X in [
        ('m1 g_link ~ cdm', np.column_stack([np.ones(n), cdm_z])),
        ('m2 g_link ~ bf', np.column_stack([np.ones(n), bf_z])),
        ('m3 g_link ~ stale', np.column_stack([np.ones(n), stale_z])),
        ('m4 g_link ~ eff_h', np.column_stack([np.ones(n), eff_h_z])),
        ('m5 g_link ~ cdm + bf', np.column_stack([np.ones(n), cdm_z, bf_z])),
        ('m6 g_link ~ cdm + stale', np.column_stack([np.ones(n), cdm_z, stale_z])),
        ('m7 g_link ~ cdm + bf + stale', np.column_stack([np.ones(n), cdm_z, bf_z, stale_z])),
    ]:
        r = _ols(X, g)
        if not r:
            continue
        names = ['intercept', 'cdm', 'bf', 'stale']
        coef_strs = []
        for i, b in enumerate(r['beta']):
            nm = names[i] if i < len(names) else f'x{i}'
            coef_strs.append(f'{nm}={b:+.3f}(p={r["p"][i]:.3g})')
        print(f'{label:<35} R²={r["r2"]:>6.3f}  ' + '  '.join(coef_strs))

    # Save
    out = Path('experiments/findings/sync_curve_breakout/chain_depth_mediator_test.json')
    out.write_text(json.dumps({'panel': panel}, indent=2, default=str))
    print(f'\nwrote: {out}')


if __name__ == '__main__':
    main()
