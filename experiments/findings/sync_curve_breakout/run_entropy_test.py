"""Test: does DDQN actually have lower-entropy policy than vanilla?

The "DDQN prefers min-entropy trajectories" hypothesis is intuitive but
unproven. Quick empirical check on local traces:

1. q_gap_late = mean(online_max_q − online_min_q) over late 50% of training
   - Larger q_gap = argmax sticks out more = lower-entropy greedy policy
   - Tests whether DDQN concentrates Q-distribution more than vanilla
2. Per-env paired_g(q_gap_late) DDQN vs vanilla — sign tells us
   the direction of difference
3. Correlate Δq_gap_late with env_reward_polarity — does DDQN's
   concentration effect differ by polarity?
4. Per-cell: does Δq_gap_late predict Δoutcome with polarity-conditional sign?
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


def _q_gap_late(row: dict) -> float:
    """Mean of (online_max_q - online_min_q) over late 50% of training."""
    omax = row.get('online_max_q_per_step')
    omin = row.get('online_min_q_per_step')
    if omax is None or omin is None:
        return float('nan')
    omax_arr = np.asarray(omax, dtype=np.float64)
    omin_arr = np.asarray(omin, dtype=np.float64)
    if omax_arr.shape != omin_arr.shape or omax_arr.size < 4:
        return float('nan')
    n = omax_arr.size
    start = n // 2
    gap = omax_arr[start:] - omin_arr[start:]
    return float(gap.mean())


def _polarity_continuous(row: dict) -> float:
    """Per-cell Pearson(episode_length, mc_return)."""
    el = row.get('episode_length')
    mc = row.get('mc_return')
    if el is None or mc is None:
        return float('nan')
    el_flat = np.asarray(el, dtype=np.float64).flatten()
    mc_flat = np.asarray(mc, dtype=np.float64).flatten()
    if el_flat.shape != mc_flat.shape or el_flat.size < 3:
        return float('nan')
    if el_flat.std() == 0 or mc_flat.std() == 0:
        return float('nan')
    return float(np.corrcoef(el_flat, mc_flat)[0, 1])


def _eval_best_burst_mean(row: dict) -> float:
    """Hasselt outcome: max over bursts of per-burst mc mean."""
    mc = row.get('mc_return')
    if mc is None: return float('nan')
    arr = np.asarray(mc, dtype=np.float64)
    if arr.ndim != 2: return float('nan')
    return float(arr.mean(axis=1).max())


def _process_corpus(path: Path) -> list[dict]:
    runs_path = path / 'runs.parquet'
    traces_path = path / 'traces.parquet'
    if not runs_path.exists() or not traces_path.exists():
        return []
    runs = pl.read_parquet(runs_path, columns=['id', 'env_name', 'arm_key', 'seed', 'gamma', 'sync_period', 'total_steps'])
    traces = pl.read_parquet(traces_path, columns=[
        'id', 'online_max_q_per_step', 'online_min_q_per_step', 'episode_length', 'mc_return', 'done',
    ])
    df = runs.join(traces, on='id', how='inner')
    rows = []
    for r in df.iter_rows(named=True):
        gamma = r.get('gamma') or 0.99
        done_arr = np.asarray(r.get('done', []), dtype=np.float64) if r.get('done') is not None else np.array([])
        bf = float(1.0 - done_arr.mean()) if done_arr.size > 0 else float('nan')
        eff_h = 1.0 / (1.0 - gamma * bf) if (gamma < 1.0 and not math.isnan(bf)) else float('nan')
        rows.append({
            'corpus': path.name,
            'env_name': r['env_name'],
            'arm_key': r['arm_key'],
            'seed': r['seed'],
            'gamma': gamma,
            'sync_period': r.get('sync_period'),
            'total_steps': r.get('total_steps'),
            'q_gap_late': _q_gap_late(r),
            'env_reward_polarity': _polarity_continuous(r),
            'eval_best_burst_mean': _eval_best_burst_mean(r),
            'eff_h_new': eff_h,
        })
    return rows


def main() -> None:
    all_rows = []
    for src_dir in TRACE_SOURCES:
        path = Path(src_dir)
        if path.is_dir():
            all_rows += _process_corpus(path)
        # sub-dirs (capacity_sweep_fourrooms has them)
        for sub in (path.iterdir() if path.is_dir() else []):
            if sub.is_dir() and (sub / 'runs.parquet').exists() and (sub / 'traces.parquet').exists():
                all_rows += _process_corpus(sub)
    df = pl.DataFrame(all_rows)
    print(f'cells: {len(df)}')

    # --- Test 1: per-env mean q_gap_late, vanilla vs DDQN
    print()
    print(f'{"env":<25} {"n_v":>4} {"n_d":>4} {"q_gap_v":>9} {"q_gap_d":>9} {"Δq_gap":>9} {"polarity":>10} {"per-cell n":>11} {"r(Δqgap, Δmc)":>14}')
    print('-' * 120)
    panel = []
    for env in sorted(df['env_name'].unique()):
        sub = df.filter(pl.col('env_name') == env)
        v = sub.filter((pl.col('arm_key') == BASELINE) & pl.col('q_gap_late').is_not_nan())
        d = sub.filter((pl.col('arm_key') == DDQN) & pl.col('q_gap_late').is_not_nan())
        if len(v) == 0 or len(d) == 0: continue
        q_gap_v = float(v['q_gap_late'].mean())
        q_gap_d = float(d['q_gap_late'].mean())
        polarity_v = float(v['env_reward_polarity'].drop_nans().mean() or float('nan'))
        # paired Δq_gap and Δoutcome per (corpus, sync, gamma, seed)
        pair_keys = ['corpus', 'gamma', 'sync_period', 'total_steps', 'seed']
        v_p = v.select(pair_keys + ['q_gap_late', 'eval_best_burst_mean']).rename({'q_gap_late':'qg_v','eval_best_burst_mean':'ov'})
        d_p = d.select(pair_keys + ['q_gap_late', 'eval_best_burst_mean']).rename({'q_gap_late':'qg_d','eval_best_burst_mean':'od'})
        j = v_p.join(d_p, on=pair_keys, how='inner').filter(
            pl.col('qg_v').is_not_nan() & pl.col('qg_d').is_not_nan() & pl.col('ov').is_not_nan() & pl.col('od').is_not_nan()
        )
        arr = j.to_pandas()
        if len(arr) >= 3 and (arr['qg_d']-arr['qg_v']).std() > 0 and (arr['od']-arr['ov']).std() > 0:
            r_qm, p_qm = pearsonr((arr['qg_d']-arr['qg_v']).to_numpy(), (arr['od']-arr['ov']).to_numpy())
        else:
            r_qm, p_qm = float('nan'), float('nan')
        print(f'{env:<25} {len(v):>4d} {len(d):>4d} {q_gap_v:>9.3f} {q_gap_d:>9.3f} {q_gap_d-q_gap_v:>+9.3f} {polarity_v:>+10.3f} {len(arr):>11d} {r_qm:>+14.3f}')
        panel.append({'env':env,'q_gap_v':q_gap_v,'q_gap_d':q_gap_d,'delta_qg':q_gap_d-q_gap_v,'polarity':polarity_v,'r_qm':r_qm,'n_pairs':len(arr)})

    # --- Test 2: cross-env, does Δq_gap correlate with polarity?
    print()
    print('=== Cross-env: ρ(Δq_gap, env_reward_polarity) ===')
    deltas = np.array([p['delta_qg'] for p in panel])
    pols = np.array([p['polarity'] for p in panel])
    finite = np.isfinite(deltas) & np.isfinite(pols)
    if finite.sum() >= 3:
        rho, p = spearmanr(deltas[finite], pols[finite])
        print(f'  Spearman ρ = {rho:+.3f}, p = {p:.3g}, n_envs = {int(finite.sum())}')
        rho_pe, p_pe = pearsonr(deltas[finite], pols[finite])
        print(f'  Pearson  r = {rho_pe:+.3f}, p = {p_pe:.3g}')

    # --- Test 3: cross-env, does Δq_gap correlate with r(Δqg, Δmc)?
    print()
    print('=== Cross-env: per-env r(Δq_gap, Δoutcome) — does Δq_gap mediate outcome? ===')
    rs = np.array([p['r_qm'] for p in panel])
    finite_rs = np.isfinite(rs)
    print(f'  n_envs with finite r(Δqg, Δmc): {int(finite_rs.sum())}')
    print(f'  per-env r values: {[f"{r:+.2f}" for r in rs[finite_rs]]}')
    if finite_rs.sum() >= 3:
        # pool via Fisher-z
        z_vals = []
        for r, p in zip(rs[finite_rs], np.array(panel)[finite_rs]):
            r_clamp = max(-0.999, min(0.999, r))
            z = 0.5 * math.log((1 + r_clamp) / (1 - r_clamp))
            z_vals.append((z, p['n_pairs'] - 3))
        total_w = sum(w for z, w in z_vals)
        z_pool = sum(z * w for z, w in z_vals) / total_w if total_w > 0 else float('nan')
        rho_pool = math.tanh(z_pool)
        print(f'  Fisher-z pooled r(Δqg, Δmc) = {rho_pool:+.3f}')

    out = Path('experiments/findings/sync_curve_breakout/entropy_test_panel.json')
    out.write_text(json.dumps({'per_env': panel}, indent=2))
    print()
    print(f'wrote: {out}')


if __name__ == '__main__':
    main()
