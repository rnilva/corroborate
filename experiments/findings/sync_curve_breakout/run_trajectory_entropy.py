"""Trajectory entropy: how stochastic are the trajectories the policy
generates?

For each cell, look at the 5 eval episodes per burst:
- σ(mc_return) within burst = how much returns vary across rollouts
- σ(episode_length) within burst = how much lengths vary
- Both are LOW when policy generates uniform trajectories (low traj entropy)
- Both are HIGH when policy generates diverse trajectories (high traj entropy)

The "DDQN prefers min-entropy trajectories" hypothesis predicts:
- DDQN cells have LOWER within-burst σ than vanilla cells
- The Δ should correlate with polarity (goal envs benefit more from
  trajectory concentration)

Aggregate per-cell: mean across bursts of within-burst σ.
Compare DDQN vs vanilla per env via paired_g.
"""
from __future__ import annotations

import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import json
import math
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import pearsonr, spearmanr

TRACE_SOURCES = [
    'experiments/data/expectile_3way',
    'experiments/data/capacity_sweep_fourrooms',
    'experiments/data/minatar_sync_intervention',
    'experiments/data/l2_x_gamma_acrobot',         # Acrobot γ=0.99/0.999, n=180
    'experiments/data/gamma_sweep_acrobot_high',   # Acrobot γ-sweep
    'experiments/data/adaptive_dqn_acrobot',       # Acrobot adaptive controller
]
DDQN = 'bootstrap=partial(Claim:bootstrap;greedification=Claim:double_greedify)'
BASELINE = 'baseline'


def _within_burst_std(arr_2d: np.ndarray, axis: int = -1) -> float:
    """Mean across bursts of within-burst (across eval episodes) std."""
    if arr_2d.ndim != 2 or arr_2d.shape[axis] < 2:
        return float('nan')
    stds = arr_2d.std(axis=axis, ddof=1)
    return float(stds.mean())


def _polarity(row: dict) -> float:
    el = row.get('episode_length')
    mc = row.get('mc_return')
    if el is None or mc is None: return float('nan')
    el_flat = np.asarray(el, dtype=np.float64).flatten()
    mc_flat = np.asarray(mc, dtype=np.float64).flatten()
    if el_flat.shape != mc_flat.shape or el_flat.size < 3: return float('nan')
    if el_flat.std() == 0 or mc_flat.std() == 0: return float('nan')
    return float(np.corrcoef(el_flat, mc_flat)[0, 1])


def _eval_best(row: dict) -> float:
    mc = row.get('mc_return')
    if mc is None: return float('nan')
    arr = np.asarray(mc, dtype=np.float64)
    if arr.ndim != 2: return float('nan')
    return float(arr.mean(axis=1).max())


def _process_corpus(path: Path) -> list[dict]:
    runs_path, traces_path = path / 'runs.parquet', path / 'traces.parquet'
    if not runs_path.exists() or not traces_path.exists(): return []
    runs_cols = pl.read_parquet(runs_path, n_rows=1).columns
    runs_keep = [c for c in ['id', 'env_name', 'arm_key', 'seed', 'gamma', 'sync_period', 'total_steps'] if c in runs_cols]
    runs = pl.read_parquet(runs_path, columns=runs_keep)
    traces = pl.read_parquet(traces_path, columns=['id', 'episode_length', 'mc_return'])
    df = runs.join(traces, on='id', how='inner')
    rows = []
    for r in df.iter_rows(named=True):
        mc_2d = np.asarray(r['mc_return'], dtype=np.float64) if r['mc_return'] is not None else None
        el_raw = r['episode_length']
        # episode_length may be (n_bursts, n_eps) ints; convert defensively
        if el_raw is not None:
            el_2d = np.asarray(el_raw, dtype=np.float64)
        else:
            el_2d = None
        mc_std = _within_burst_std(mc_2d) if mc_2d is not None else float('nan')
        el_std = _within_burst_std(el_2d) if el_2d is not None else float('nan')
        rows.append({
            'corpus': path.name, 'env_name': r['env_name'], 'arm_key': r['arm_key'],
            'seed': r['seed'], 'gamma': r.get('gamma'),
            'sync_period': r.get('sync_period'), 'total_steps': r.get('total_steps'),
            'mc_return_within_burst_std': mc_std,
            'episode_length_within_burst_std': el_std,
            'env_reward_polarity': _polarity(r),
            'eval_best_burst_mean': _eval_best(r),
        })
    return rows


def main() -> None:
    all_rows = []
    for src in TRACE_SOURCES:
        p = Path(src)
        if p.is_dir():
            all_rows += _process_corpus(p)
            for sub in p.iterdir():
                if sub.is_dir() and (sub / 'runs.parquet').exists() and (sub / 'traces.parquet').exists():
                    all_rows += _process_corpus(sub)
    df = pl.DataFrame(all_rows)
    print(f'cells: {len(df)}')

    print()
    print(f'{"env":<25} {"σmc_v":>8} {"σmc_d":>8} {"Δσmc":>8} {"Δσmc/σmc_v":>11} | {"σel_v":>8} {"σel_d":>8} {"Δσel":>8} {"polarity":>10}')
    print('-' * 120)
    panel = []
    for env in sorted(df['env_name'].unique()):
        sub = df.filter(pl.col('env_name') == env)
        v = sub.filter((pl.col('arm_key') == BASELINE) & pl.col('mc_return_within_burst_std').is_not_nan())
        d = sub.filter((pl.col('arm_key') == DDQN) & pl.col('mc_return_within_burst_std').is_not_nan())
        if len(v) == 0 or len(d) == 0: continue
        sigma_mc_v = float(v['mc_return_within_burst_std'].mean())
        sigma_mc_d = float(d['mc_return_within_burst_std'].mean())
        sigma_el_v = float(v['episode_length_within_burst_std'].drop_nans().mean() or float('nan'))
        sigma_el_d = float(d['episode_length_within_burst_std'].drop_nans().mean() or float('nan'))
        polarity_v = float(v['env_reward_polarity'].drop_nans().mean() or float('nan'))
        rel_d_mc = (sigma_mc_d - sigma_mc_v) / sigma_mc_v if sigma_mc_v > 1e-9 else float('nan')
        print(f'{env:<25} {sigma_mc_v:>8.3f} {sigma_mc_d:>8.3f} {sigma_mc_d-sigma_mc_v:>+8.3f} {rel_d_mc:>+11.1%} | {sigma_el_v:>8.3f} {sigma_el_d:>8.3f} {sigma_el_d-sigma_el_v:>+8.3f} {polarity_v:>+10.3f}')
        panel.append({'env':env,'sigma_mc_v':sigma_mc_v,'sigma_mc_d':sigma_mc_d,'sigma_el_v':sigma_el_v,'sigma_el_d':sigma_el_d,'polarity':polarity_v,'rel_d_mc':rel_d_mc,'n_v':len(v),'n_d':len(d)})

    print()
    print('=== Cross-env tests ===')
    rel_dmc = np.array([p['rel_d_mc'] for p in panel])
    pols = np.array([p['polarity'] for p in panel])
    finite = np.isfinite(rel_dmc) & np.isfinite(pols)
    if finite.sum() >= 3:
        rho, pv = spearmanr(rel_dmc[finite], pols[finite])
        print(f'  ρ(Δσmc/σmc_v, polarity) = {rho:+.3f}, p = {pv:.3g}, n_envs = {int(finite.sum())}')

    # Sign test: does DDQN systematically have lower σmc?
    sign_lower_mc = sum(1 for p in panel if not math.isnan(p['rel_d_mc']) and p['rel_d_mc'] < 0)
    n_total_mc = sum(1 for p in panel if not math.isnan(p['rel_d_mc']))
    sign_lower_el = sum(1 for p in panel if not math.isnan(p['sigma_el_v']) and not math.isnan(p['sigma_el_d']) and p['sigma_el_d'] < p['sigma_el_v'])
    n_total_el = sum(1 for p in panel if not math.isnan(p['sigma_el_v']) and not math.isnan(p['sigma_el_d']))
    from scipy.stats import binomtest
    print(f'  ΔσMC: {sign_lower_mc}/{n_total_mc} envs DDQN lower σmc (binomial p={binomtest(sign_lower_mc, n_total_mc, p=0.5).pvalue:.3g})')
    print(f'  Δσel: {sign_lower_el}/{n_total_el} envs DDQN lower σel (binomial p={binomtest(sign_lower_el, n_total_el, p=0.5).pvalue:.3g})')

    out = Path('experiments/findings/sync_curve_breakout/trajectory_entropy_panel.json')
    out.write_text(json.dumps({'per_env': panel}, indent=2))
    print(f'wrote: {out}')


if __name__ == '__main__':
    main()
