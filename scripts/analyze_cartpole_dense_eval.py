"""Per-burst + window AUC analysis on the CartPole dense-eval
corpus (eval_every=2500, total_steps=100000 → 40 bursts).

CartPole canonical at 1M / eval_every=100000 has both arms at
~98 raw outcome (γ-discounted 500-step cap, both arms hit cap
on 82% of episodes by burst 0 — memory `findings_substrate_
realization_variance` audit). The transient is pre-100k. This
script reads the dense-eval corpus to characterize:

- Per-burst trajectory at 2.5k granularity (every 5th printed)
- Burst-window AUC (early / mid / late / final / full) — same
  shape as `findings_dense_eval_acrobot_transient`.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
CORPUS = REPO / 'experiments' / 'data' / 'cartpole_dense_eval'

DDQN = 'bootstrap=partial(Claim:bootstrap;greedification=Claim:double_greedify)'
BASE = 'baseline'


def main() -> None:
    runs = pl.read_parquet(str(CORPUS / 'runs.parquet'))
    traces = pl.read_parquet(str(CORPUS / 'traces.parquet')).select(['id', 'mc_return'])
    df = runs.join(traces, on='id')
    print(f'shape: {df.shape}')
    print(df.group_by('arm_key').agg(pl.len().alias('n')).sort('arm_key'))

    def per_burst(arm: str) -> tuple[np.ndarray, np.ndarray]:
        sub = df.filter(pl.col('arm_key') == arm)
        arr = np.array(sub['mc_return'].to_list())  # (seeds, bursts, episodes)
        per_seed = arr.mean(axis=-1)
        return per_seed.mean(axis=0), per_seed.std(axis=0) / np.sqrt(per_seed.shape[0])

    bm, bs = per_burst(BASE)
    dm, ds = per_burst(DDQN)
    n_bursts = bm.shape[0]
    print(f'\nCartPole dense-eval per-burst (every 5th, n=30 per arm, {n_bursts} bursts):')
    print(f'{"burst":>5s} {"step":>8s} {"vanilla":>9s} {"DDQN":>9s} {"Δ":>7s} {"z":>6s}')
    for b in range(0, n_bursts, 5):
        step = (b + 1) * 2500
        delta = dm[b] - bm[b]
        se = np.sqrt(bs[b] ** 2 + ds[b] ** 2)
        z = delta / se if se > 0 else float('nan')
        print(f'{b:>5d} {step:>8d} {bm[b]:>+9.2f} {dm[b]:>+9.2f} {delta:>+7.2f} {z:>+6.2f}')

    print(f'\nBurst-window AUC decomposition:')

    def window_seed_means(arm: str, sl: slice) -> np.ndarray:
        sub = df.filter(pl.col('arm_key') == arm)
        arr = np.array(sub['mc_return'].to_list())
        return arr[:, sl, :].mean(axis=(1, 2))

    def stat(name: str, sl: slice) -> None:
        b = window_seed_means(BASE, sl)
        d = window_seed_means(DDQN, sl)
        delta = d.mean() - b.mean()
        sd = math.sqrt((b.var(ddof=1) + d.var(ddof=1)) / 2)
        cohen = delta / sd if sd > 0 else float('nan')
        se = math.sqrt((b.var(ddof=1) + d.var(ddof=1)) / d.size)
        z = delta / se if se > 0 else float('nan')
        print(f'  {name:<22s}: vanilla={b.mean():+7.3f}  DDQN={d.mean():+7.3f}  Δ={delta:+6.3f}  d={cohen:+5.2f}  z={z:+5.2f}')

    q1 = n_bursts // 4
    q2 = n_bursts // 2
    q3 = 3 * n_bursts // 4
    stat(f'early (0-{q1 - 1})', slice(0, q1))
    stat(f'mid (early {q1}-{q2 - 1})', slice(q1, q2))
    stat(f'mid (late {q2}-{q3 - 1})', slice(q2, q3))
    stat(f'late ({q3}-{n_bursts - 1})', slice(q3, n_bursts))
    stat(f'final (last 4)', slice(n_bursts - 4, n_bursts))
    stat(f'full (0-{n_bursts - 1})', slice(0, n_bursts))


if __name__ == '__main__':
    main()
