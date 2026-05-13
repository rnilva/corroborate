"""Focused Acrobot per-burst dig.

At canonical scope + bsuite-excluded, Acrobot was the only env
where action-margin/dormancy mediated the Q-channel:
  ρ(q, mc | bg) = +0.651
  + margin + dormancy = +0.485  (Δ = -0.166, ~25% reduction)

This script tests at per-burst resolution (with newly-backfilled
q_argmax_margin_per_burst, q_action_std_per_burst):

1. Per-burst baseline ρ(q_pb, mc_pb | bg_pb) on Acrobot canonical
   cells.
2. Add per-burst margin / std as conditioning — does the
   mediation pattern hold within-cell at per-burst level?
3. Compare cell-level vs per-burst within-cell vs across cells
   to see where the mediation lives."""
from __future__ import annotations

import math
import warnings

import numpy as np
import polars as pl
from scipy import stats


CACHE = 'experiments/data/cache/ddqn.parquet'


def _rank(x):
    return stats.rankdata(x)


def _partial(y, x, controls):
    mask = np.isfinite(y) & np.isfinite(x)
    for j in range(controls.shape[1]):
        mask &= np.isfinite(controls[:, j])
    y, x, controls = y[mask], x[mask], controls[mask]
    n = len(y)
    if n < 10:
        return float('nan'), n
    y_r, x_r = _rank(y), _rank(x)
    c_r = np.column_stack([_rank(controls[:, j]) for j in range(controls.shape[1])])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            y_res = y_r - c_r @ np.linalg.lstsq(c_r, y_r, rcond=None)[0]
            x_res = x_r - c_r @ np.linalg.lstsq(c_r, x_r, rcond=None)[0]
        except np.linalg.LinAlgError:
            return float('nan'), n
    if y_res.std() == 0 or x_res.std() == 0:
        return float('nan'), n
    return float(np.corrcoef(y_res, x_res)[0, 1]), n


def main() -> None:
    canonical = (
        (pl.col('env_name') == 'Acrobot-v1')
        & (pl.col('gamma') == 0.99)
        & (pl.col('sync_period') == 100)
        & (pl.col('replay.capacity') == 50000)
        & (pl.col('optimizer.inner.lr') == 0.0001)
        & (pl.col('q_network.hidden') == '(64,64)')
        & ((pl.col('n_step') == 1) | pl.col('n_step').is_null())
        & pl.col('action_duplicate_k').is_null()
        & (pl.col('reward_scale').is_null() | (pl.col('reward_scale') == 1.0))
        & pl.col('target_sync.tau').is_null()
        & (pl.col('total_steps') == 200000)
        & (pl.col('wrappers') == '()')
    )
    df = pl.scan_parquet(CACHE).filter(canonical).select([
        'corpus', 'arm_key', 'seed',
        'bootstrap_gap_magnitude_per_burst',
        'q_per_burst',
        'mc_return_raw__mean_axis_-1',
        'q_argmax_margin_per_burst',
        'q_action_std_per_burst',
    ]).filter(
        pl.col('bootstrap_gap_magnitude_per_burst').is_not_null()
        & pl.col('q_per_burst').is_not_null()
        & pl.col('mc_return_raw__mean_axis_-1').is_not_null()
    ).collect()

    print(f'Acrobot canonical cells: {df.height}')
    print()

    # Unfold per-burst
    rows = []
    has_margin_pb = 'q_argmax_margin_per_burst' in df.columns
    has_std_pb = 'q_action_std_per_burst' in df.columns
    for i, cell in enumerate(df.iter_rows(named=True)):
        bg = np.asarray(cell['bootstrap_gap_magnitude_per_burst'] or [], dtype=np.float64)
        q = np.asarray(cell['q_per_burst'] or [], dtype=np.float64)
        mc = np.asarray(cell['mc_return_raw__mean_axis_-1'] or [], dtype=np.float64)
        margin_pb = (
            np.asarray(cell.get('q_argmax_margin_per_burst') or [], dtype=np.float64)
            if has_margin_pb else np.array([])
        )
        std_pb = (
            np.asarray(cell.get('q_action_std_per_burst') or [], dtype=np.float64)
            if has_std_pb else np.array([])
        )
        n_b = min(bg.size, q.size, mc.size)
        for j in range(n_b):
            if not (np.isfinite(bg[j]) and np.isfinite(q[j]) and np.isfinite(mc[j])):
                continue
            rows.append({
                'cell_id': i, 'burst': j,
                'bg': bg[j], 'q': q[j], 'mc': mc[j],
                'margin_pb': float(margin_pb[j]) if j < margin_pb.size else float('nan'),
                'std_pb': float(std_pb[j]) if j < std_pb.size else float('nan'),
            })
    panel = pl.DataFrame(rows)
    print(f'Per-burst panel: {panel.height} rows from {panel["cell_id"].n_unique()} cells')
    print()

    q_arr = panel.get_column('q').to_numpy()
    mc_arr = panel.get_column('mc').to_numpy()

    # Layered tests
    print(f'{"controls":<55s} | partial ρ(q, mc | controls)')
    print('-' * 85)
    tests = [
        (['bg'], ['bg']),
        (['bg', 'cell_id'], ['bg', 'cell_id']),
        (['bg', 'cell_id', 'burst'], ['bg', 'cell_id', 'burst']),
        (['bg', 'margin_pb'], ['bg', 'margin_pb']),
        (['bg', 'cell_id', 'margin_pb'], ['bg', 'cell_id', 'margin_pb']),
        (['bg', 'cell_id', 'burst', 'margin_pb'], ['bg', 'cell_id', 'burst', 'margin_pb']),
        (['bg', 'margin_pb', 'std_pb'], ['bg', 'margin_pb', 'std_pb']),
        (['bg', 'cell_id', 'burst', 'margin_pb', 'std_pb'], ['bg', 'cell_id', 'burst', 'margin_pb', 'std_pb']),
    ]
    for label, cols in tests:
        ctrl = panel.select(cols).to_numpy()
        rho, n = _partial(mc_arr, q_arr, ctrl)
        print(f'{", ".join(label):<55s} | {rho:+.3f}  (n={n})')


if __name__ == '__main__':
    main()
