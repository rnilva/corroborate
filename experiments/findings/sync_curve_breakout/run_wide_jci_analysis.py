"""Wide-range JCI / partial Spearman analysis across (env, sync_period)
strata.

Builds a per-(env, sync, total_steps) panel of paired-Δ statistics
for each of {jensen_gap, target_staleness_late, effective_horizon,
bootstrap_fraction, q_divergence_score, eval_best_burst_mean}, then
runs JCI-style stratified partial Spearman to ask whether each
candidate mediator predicts Δ_outcome AFTER conditioning on Δ_jens
within the (env, sync) stratum.

Three layers:

1. **Per-stratum panel**: paired g + mean Δ for each measurable per
   (env, sync, total_steps).

2. **Cross-stratum partial Spearman** ρ(Δ_M, Δ_outcome | Δ_jens) via
   `stratified_partial_spearman_rho` (JCI form, Fisher-z pooled across
   env strata).

3. **Per-env-per-sync correlation panel**: within each stratum, the
   per-pair r(Δ_M, Δ_outcome) and r(Δ_M, Δ_outcome | Δ_jens). Surfaces
   regimes where the link operates vs where it's broken.

Used to characterize:
  - which (env, sync) regimes have an active staleness link
  - which envs reverse sign at high sync (Q-amplification regime)
  - whether candidates other than staleness compete

Output: `wide_jci_panel.json`.
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
from scipy.stats import pearsonr, spearmanr

import corroborate_rl.dqn.measurables  # register
from corroborate.graph.discovery import (
    partial_spearman_rho, stratified_partial_spearman_rho,
)

DDQN = 'bootstrap=partial(Claim:bootstrap;greedification=Claim:double_greedify)'

# Envs with non-degenerate trace data + outcome variance.
ENVS = (
    'Acrobot-v1', 'Asterix-MinAtar', 'Breakout-MinAtar',
    'CartPole-v1', 'FourRooms-misc', 'Freeway-MinAtar',
    'MetaMaze-misc', 'MountainCar-v0', 'Pong-misc',
    'SpaceInvaders-MinAtar',
)

CANDIDATES = (
    'jensen_gap',
    'target_staleness_late',
    'effective_horizon',
    'bootstrap_fraction',
    'q_divergence_score',
)


def _pair_stratum(
    df: pl.DataFrame, env: str, sync: int, total_steps: int,
) -> pl.DataFrame | None:
    """Pair (DDQN, baseline) within a (env, sync, total_steps) cell.
    Returns paired DataFrame with Δ for each candidate measurable +
    Δ_outcome, or None if no pairs."""
    cells = df.filter(
        (pl.col('env_name') == env)
        & (pl.col('sync_period') == sync)
        & (pl.col('total_steps') == total_steps)
        & pl.col('arm_key').is_in(['baseline', DDQN])
    )
    if cells.height < 4:
        return None
    pair_keys = ['env_name', 'corpus', 'gamma', 'total_steps', 'sync_period', 'seed']
    select_cols = pair_keys + ['eval_best_burst_mean'] + [
        c for c in CANDIDATES if c in cells.columns
    ]
    v = cells.filter(pl.col('arm_key') == 'baseline').select(select_cols).rename(
        {c: f'{c}_v' for c in select_cols if c not in pair_keys}
    )
    d = cells.filter(pl.col('arm_key') == DDQN).select(select_cols).rename(
        {c: f'{c}_d' for c in select_cols if c not in pair_keys}
    )
    j = v.join(d, on=pair_keys, how='inner').filter(
        pl.col('eval_best_burst_mean_v').is_finite()
        & pl.col('eval_best_burst_mean_d').is_finite()
    )
    if j.height < 4:
        return None
    j = j.with_columns([
        (pl.col(f'{c}_d') - pl.col(f'{c}_v')).alias(f'd_{c}')
        for c in ['eval_best_burst_mean'] + list(CANDIDATES)
        if f'{c}_d' in j.columns
    ])
    return j


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 4 or x.std() == 0 or y.std() == 0:
        return float('nan'), float('nan')
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 4:
            return float('nan'), float('nan')
        x, y = x[m], y[m]
    if x.std() == 0 or y.std() == 0:
        return float('nan'), float('nan')
    r, p = pearsonr(x, y)
    return float(r), float(p)


def main() -> None:
    df = pl.read_parquet('experiments/data/cache/ddqn.parquet')

    # 1. Per-stratum panel
    panel: list[dict] = []
    for env in ENVS:
        env_df = df.filter(pl.col('env_name') == env)
        if env_df.height == 0:
            continue
        sync_vals = sorted(env_df['sync_period'].unique().to_list())
        steps_vals = sorted(env_df['total_steps'].unique().to_list())
        for sync in sync_vals:
            for steps in steps_vals:
                paired = _pair_stratum(df, env, sync, steps)
                if paired is None or paired.height < 8:
                    continue
                row: dict[str, object] = {
                    'env': env, 'sync': sync, 'total_steps': steps,
                    'n_pairs': paired.height,
                }
                # Mean Δs
                d_o = paired['d_eval_best_burst_mean'].to_numpy()
                row['mean_d_outcome'] = float(d_o.mean())
                row['sd_d_outcome'] = float(d_o.std(ddof=1))
                if row['sd_d_outcome'] > 0:
                    g_raw = float(d_o.mean()) / row['sd_d_outcome']
                    n = paired.height
                    j_corr = 1.0 - 3.0 / (4.0 * (n - 1) - 1.0) if n > 2 else 1.0
                    row['g_outcome'] = j_corr * g_raw
                # Per-candidate stats
                for c in CANDIDATES:
                    col = f'd_{c}'
                    if col not in paired.columns:
                        continue
                    arr = paired[col].drop_nans().to_numpy()
                    if len(arr) < 4:
                        continue
                    row[f'mean_d_{c}'] = float(arr.mean())
                    if arr.std(ddof=1) > 0:
                        row[f'g_{c}'] = float(arr.mean()) / float(arr.std(ddof=1))
                    # within-stratum r(Δc, Δoutcome)
                    arr_full = paired[col].to_numpy()
                    if len(arr_full) == len(d_o):
                        r, p = _safe_pearson(arr_full, d_o)
                        row[f'r_{c}_outcome'] = r
                        row[f'p_{c}_outcome'] = p
                # Within-stratum partial: ρ(Δ_stale, Δ_o | Δ_jens)
                if 'd_target_staleness_late' in paired.columns and 'd_jensen_gap' in paired.columns:
                    ds = paired['d_target_staleness_late'].to_numpy()
                    dj = paired['d_jensen_gap'].to_numpy()
                    mask = np.isfinite(ds) & np.isfinite(dj) & np.isfinite(d_o)
                    if mask.sum() >= 5:
                        rho, pp = partial_spearman_rho(ds[mask], d_o[mask], dj[mask])
                        row['rho_part_stale_o_given_jens'] = rho
                        row['p_part_stale_o_given_jens'] = pp
                panel.append(row)

    print(f'panel rows: {len(panel)}', flush=True)

    # 2. Cross-stratum stratified partial Spearman ρ(Δ_M, Δ_outcome | Δ_jens, strata)
    # using ALL pooled paired cells with stratification by (env, sync).
    # Compute within-stratum partial ρ per (env, sync) and Fisher-z pool.
    print()
    print('=== Stratified (env, sync) partial Spearman ρ(Δ_M, Δ_outcome | Δ_jens) ===\n')
    print(f'{"mediator":<28} {"ρ_pool":>10} {"p":>10} {"n_pooled":>10} {"n_strata":>10}', flush=True)
    print('-' * 80)

    # Build per-stratum vectors
    per_stratum_data: dict[str, dict[tuple[str, int, int], np.ndarray]] = {
        'Δ_jens': {},
        'Δ_outcome': {},
    }
    for c in CANDIDATES:
        per_stratum_data[f'Δ_{c}'] = {}

    for env in ENVS:
        env_df = df.filter(pl.col('env_name') == env)
        if env_df.height == 0:
            continue
        for sync in sorted(env_df['sync_period'].unique().to_list()):
            for steps in sorted(env_df['total_steps'].unique().to_list()):
                paired = _pair_stratum(df, env, sync, steps)
                if paired is None or paired.height < 8:
                    continue
                key = f'{env}|{sync}|{steps}'  # string-keyed for np.unique
                per_stratum_data['Δ_outcome'][key] = paired['d_eval_best_burst_mean'].to_numpy()
                if 'd_jensen_gap' in paired.columns:
                    per_stratum_data['Δ_jens'][key] = paired['d_jensen_gap'].to_numpy()
                for c in CANDIDATES:
                    if f'd_{c}' in paired.columns:
                        per_stratum_data[f'Δ_{c}'][key] = paired[f'd_{c}'].to_numpy()

    # Pool: for each candidate M, build x=Δ_M, y=Δ_outcome, z=Δ_jens, strata=key
    # then run stratified_partial_spearman_rho.
    for c in CANDIDATES:
        c_key = f'Δ_{c}'
        if c_key not in per_stratum_data:
            continue
        xs, ys, zs, st = [], [], [], []
        for key in sorted(per_stratum_data[c_key]):
            x = per_stratum_data[c_key][key]
            y = per_stratum_data['Δ_outcome'].get(key)
            z = per_stratum_data['Δ_jens'].get(key)
            if y is None or z is None or len(x) != len(y) or len(x) != len(z):
                continue
            mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
            if mask.sum() < 5:
                continue
            xs.append(x[mask])
            ys.append(y[mask])
            zs.append(z[mask])
            st.extend([key] * int(mask.sum()))
        if not xs:
            continue
        x_pooled = np.concatenate(xs)
        y_pooled = np.concatenate(ys)
        z_pooled = np.concatenate(zs)
        rho, p = stratified_partial_spearman_rho(
            x_pooled, y_pooled, z_pooled, st, min_stratum_size=5,
        )
        n_pooled = len(x_pooled)
        n_strata = len(set(st))
        print(f'{c:<28} {rho:>+10.4f} {p:>10.4g} {n_pooled:>10} {n_strata:>10}', flush=True)

    # 3. Per-stratum panel — within-stratum r table for staleness
    print()
    print('=== Per-(env, sync) within-stratum r(Δ_stale, Δ_outcome) ===\n')
    print(f'{"env":<22} {"sync":>6} {"steps":>8} {"n":>4} '
          f'{"g_out":>7} {"g_jens":>7} {"g_stale":>8} {"r_stale_o":>10} {"ρ_part":>8}', flush=True)
    print('-' * 100)

    for row in sorted(panel, key=lambda r: (r['env'], r['sync'], r['total_steps'])):
        env = row['env']
        sync = row['sync']
        steps = row['total_steps']
        n = row['n_pairs']
        g_out = row.get('g_outcome', float('nan'))
        g_jens = row.get('g_jensen_gap', float('nan'))
        g_stale = row.get('g_target_staleness_late', float('nan'))
        r_stale_o = row.get('r_target_staleness_late_outcome', float('nan'))
        rho_p = row.get('rho_part_stale_o_given_jens', float('nan'))

        def _f(x: float, prec: int = 3) -> str:
            return f'{x:>+.{prec}f}' if isinstance(x, float) and not math.isnan(x) else 'nan'

        print(
            f'{env:<22} {sync:>6} {steps:>8} {n:>4} '
            f'{_f(g_out):>7} {_f(g_jens):>7} {_f(g_stale):>8} '
            f'{_f(r_stale_o):>10} {_f(rho_p):>8}',
            flush=True,
        )

    out = Path('experiments/findings/sync_curve_breakout/wide_jci_panel.json')
    out.write_text(json.dumps(panel, indent=2, default=str))
    print(f'\nwrote: {out}', flush=True)


if __name__ == '__main__':
    main()
