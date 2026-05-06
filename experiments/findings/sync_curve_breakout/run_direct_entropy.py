"""Direct entropy of the greedy policy's marginal action distribution.

For each cell, compute H(online_argmax_per_step) — Shannon entropy of
the distribution of greedy-argmax actions over the training trajectory.

- Lower H = policy commits to fewer distinct actions across visited states
  → 'lower-entropy' policy in marginal-distribution sense
- Higher H = policy uses more distinct actions → 'higher-entropy'

Note: this is MARGINAL entropy over the visited state distribution, not
state-conditional. Per-state argmax is deterministic (no Q-distribution
available in trace). Marginal H captures whether DDQN concentrates its
greedy-action histogram more than vanilla.

Tests:
1. Per-env paired Δ_H (DDQN − vanilla): is DDQN systematically lower H?
2. Cross-env ρ(Δ_H, env_reward_polarity): does the entropy effect vary by polarity?
3. Per-cell r(Δ_H, Δoutcome): does within-seed entropy reduction track outcome?
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
]
DDQN = 'bootstrap=partial(Claim:bootstrap;greedification=Claim:double_greedify)'
BASELINE = 'baseline'


def _shannon_entropy(actions: np.ndarray, n_actions: int) -> float:
    """Marginal Shannon entropy (in nats) of action histogram."""
    if actions.size == 0:
        return float('nan')
    counts = np.bincount(actions.astype(np.int64), minlength=n_actions)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def _entropy_late(row: dict, n_actions: int) -> float:
    """H(online_argmax) over late 50% of training (greedy policy stabilized)."""
    arr = row.get('online_argmax_per_step')
    if arr is None: return float('nan')
    a = np.asarray(arr, dtype=np.int64)
    if a.size < 4: return float('nan')
    return _shannon_entropy(a[a.size // 2 :], n_actions)


def _entropy_early(row: dict, n_actions: int) -> float:
    arr = row.get('online_argmax_per_step')
    if arr is None: return float('nan')
    a = np.asarray(arr, dtype=np.int64)
    if a.size < 4: return float('nan')
    return _shannon_entropy(a[: a.size // 4], n_actions)


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
    traces_cols_avail = pl.read_parquet(traces_path, n_rows=1).columns
    traces_keep = ['id', 'online_argmax_per_step', 'episode_length', 'mc_return']
    if 'n_actions' in traces_cols_avail:
        traces_keep.append('n_actions')
    traces = pl.read_parquet(traces_path, columns=traces_keep)
    df = runs.join(traces, on='id', how='inner')
    rows = []
    for r in df.iter_rows(named=True):
        # n_actions: prefer trace col, else infer from argmax max
        argmax = np.asarray(r.get('online_argmax_per_step') or [], dtype=np.int64)
        if 'n_actions' in r and r.get('n_actions') is not None:
            n_actions = int(r['n_actions'])
        elif argmax.size > 0:
            n_actions = int(argmax.max() + 1)
        else:
            n_actions = 4
        rows.append({
            'corpus': path.name, 'env_name': r['env_name'], 'arm_key': r['arm_key'],
            'seed': r['seed'], 'gamma': r.get('gamma'),
            'sync_period': r.get('sync_period'), 'total_steps': r.get('total_steps'),
            'n_actions': n_actions,
            'H_late': _entropy_late(r, n_actions),
            'H_early': _entropy_early(r, n_actions),
            'H_max': math.log(n_actions),
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
    print(f'{"env":<25} {"|A|":>4} {"H_max":>6} {"H_v":>6} {"H_d":>6} {"ΔH":>7} {"ΔH/H_max":>9} {"polarity":>10} {"r(ΔH,Δmc)":>12}')
    print('-' * 110)
    panel = []
    for env in sorted(df['env_name'].unique()):
        sub = df.filter(pl.col('env_name') == env)
        v = sub.filter((pl.col('arm_key') == BASELINE) & pl.col('H_late').is_not_nan())
        d = sub.filter((pl.col('arm_key') == DDQN) & pl.col('H_late').is_not_nan())
        if len(v) == 0 or len(d) == 0: continue
        n_actions = int(sub['n_actions'].mean())
        H_max = math.log(n_actions)
        H_v = float(v['H_late'].mean())
        H_d = float(d['H_late'].mean())
        delta_H = H_d - H_v
        polarity_v = float(v['env_reward_polarity'].drop_nans().mean() or float('nan'))
        # paired r within env
        pair_keys = ['corpus', 'gamma', 'sync_period', 'total_steps', 'seed']
        v_p = v.select(pair_keys + ['H_late', 'eval_best_burst_mean']).rename({'H_late':'Hv','eval_best_burst_mean':'ov'})
        d_p = d.select(pair_keys + ['H_late', 'eval_best_burst_mean']).rename({'H_late':'Hd','eval_best_burst_mean':'od'})
        j = v_p.join(d_p, on=pair_keys, how='inner').filter(pl.col('Hv').is_not_nan() & pl.col('Hd').is_not_nan() & pl.col('ov').is_not_nan() & pl.col('od').is_not_nan())
        arr = j.to_pandas()
        if len(arr) >= 3:
            dH = (arr['Hd']-arr['Hv']).to_numpy()
            dM = (arr['od']-arr['ov']).to_numpy()
            if dH.std() > 0 and dM.std() > 0:
                r_HM, _ = pearsonr(dH, dM)
            else:
                r_HM = float('nan')
        else:
            r_HM = float('nan')
        print(f'{env:<25} {n_actions:>4d} {H_max:>6.3f} {H_v:>6.3f} {H_d:>6.3f} {delta_H:>+7.3f} {delta_H/H_max:>+9.3f} {polarity_v:>+10.3f} {r_HM:>+12.3f}')
        panel.append({'env':env,'n_actions':n_actions,'H_max':H_max,'H_v':H_v,'H_d':H_d,'delta_H':delta_H,'polarity':polarity_v,'r_HM':r_HM,'n_pairs':len(arr)})

    print()
    print('=== Cross-env tests ===')
    deltas = np.array([p['delta_H'] for p in panel])
    pols = np.array([p['polarity'] for p in panel])
    rs = np.array([p['r_HM'] for p in panel])
    finite = np.isfinite(deltas) & np.isfinite(pols)
    if finite.sum() >= 3:
        rho, p = spearmanr(deltas[finite], pols[finite])
        print(f'  ρ(ΔH, polarity) = {rho:+.3f}, p = {p:.3g}, n_envs = {int(finite.sum())}')
    finite_r = np.isfinite(rs)
    print(f'  per-env r(ΔH, Δmc): {[f"{r:+.2f}" for r in rs[finite_r]]}')
    if finite_r.sum() >= 3:
        z_pool, ws = 0.0, 0.0
        for r, n in zip(rs[finite_r], np.array([p['n_pairs'] for p in panel])[finite_r]):
            r_clamp = max(-0.999, min(0.999, r))
            z = 0.5 * math.log((1 + r_clamp) / (1 - r_clamp))
            z_pool += z * (n - 3); ws += (n - 3)
        if ws > 0:
            print(f'  Fisher-z pooled r(ΔH, Δmc) = {math.tanh(z_pool/ws):+.3f}')

    # Sign-test: does DDQN universally have lower H?
    delta_H_finite = deltas[np.isfinite(deltas)]
    n_lower = int((delta_H_finite < 0).sum())
    n_total = len(delta_H_finite)
    from scipy.stats import binomtest
    bt = binomtest(n_lower, n_total, p=0.5)
    print(f'  ΔH sign: {n_lower}/{n_total} envs have DDQN H < vanilla H, two-sided binomial p = {bt.pvalue:.3g}')

    out = Path('experiments/findings/sync_curve_breakout/direct_entropy_panel.json')
    out.write_text(json.dumps({'per_env': panel}, indent=2))
    print(f'wrote: {out}')


if __name__ == '__main__':
    main()
