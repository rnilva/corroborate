"""Project the 8 §5 candidate mediators across the corpus.

Reads the canonical `runs.parquet` + `traces.parquet` (+ zarr
arrays), joins by id, computes the eight mediator scalars per
cell, writes `runs_with_mediators.parquet` with new `mediator.*`
columns alongside the original run-level columns.

After this, `experiments/smoke_per_env_mediator_pc.py` consumes
the enriched runs to reproduce PAPER §5's within-env Pearson and
§6's per-env PC with the rich variable set.

Run: `uv run python experiments/compute_mediators.py`."""
from __future__ import annotations

# CPU is fine — pure numpy on persisted traces.
import os

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

from pathlib import Path

import polars as pl

import numpy as np

from corroborate.persistence import iter_trace_records
from corroborate.rl.dqn.measurables import (
    fill_ratio_late,
    greedy_match_late,
    q_gap_growth,
    q_gap_late,
    q_max_growth,
    td_residual_late,
    v_vs_max_delta_late,
)


def _epsilon_late(
    *, eps_init: float, eps_final: float,
    anneal_steps: int, total_steps: int,
) -> float:
    """Mean of the linear-ε schedule value over the late 50% of
    training. Closed-form from leaves; not a Measurable because
    it reads HP scalars, not a per-step record. Inlined here as
    a §5-analysis-specific projection — `mediator.epsilon_late`
    measurement on the enriched runs parquet."""
    if anneal_steps <= 0 or total_steps <= 0:
        return float('nan')
    lo = total_steps // 2
    if lo >= total_steps:
        return float('nan')
    steps = np.arange(lo, total_steps, dtype=np.float64)
    progress = np.minimum(steps / float(anneal_steps), 1.0)
    eps = eps_init + (eps_final - eps_init) * progress
    return float(np.mean(eps))


_DATA_DIR = Path(__file__).parent / 'data' / 'ddqn'
_RUNS_PATH = _DATA_DIR / 'runs.parquet'
_TRACES_PATH = _DATA_DIR / 'traces.parquet'
_ZARR_PATH = _DATA_DIR / 'arrays.zarr'
_OUT_PATH = _DATA_DIR / 'runs_with_mediators.parquet'


_LEAF_KEYS_FOR_EPSILON: tuple[str, ...] = (
    'action_select.schedule.eps_init',
    'action_select.schedule.eps_final',
    'action_select.schedule.anneal_steps',
    'total_steps',
)
_LEAF_KEY_REPLAY_CAPACITY = 'replay.capacity'


def main() -> None:
    if not _RUNS_PATH.exists() or not _TRACES_PATH.exists():
        raise SystemExit(
            f'corpus not found: {_RUNS_PATH} or {_TRACES_PATH}',
        )
    print(f'reading runs:    {_RUNS_PATH}')
    print(f'reading traces:  {_TRACES_PATH}')
    if _ZARR_PATH.exists():
        print(f'reading arrays:  {_ZARR_PATH}')

    runs_df = pl.read_parquet(_RUNS_PATH)
    print(f'  {runs_df.height} rows × {len(runs_df.columns)} cols')

    # Build per-id lookup of the leaf HPs needed by epsilon_late /
    # fill_ratio_late. Cheap — runs.parquet has only scalar columns.
    leaf_lookup: dict[str, dict[str, object]] = {}
    for row in runs_df.iter_rows(named=True):
        rid = row.get('id')
        if isinstance(rid, str):
            leaf_lookup[rid] = {
                k: row.get(k) for k in (
                    _LEAF_KEY_REPLAY_CAPACITY,
                    'action_select.schedule.eps_init',
                    'action_select.schedule.eps_final',
                    'action_select.schedule.anneal_steps',
                    'total_steps',
                )
            }

    # Streaming read: only the trace columns mediators actually
    # consume. iter_trace_records bounds peak memory at
    # O(batch_size × per-row-size) instead of materialising the
    # full trace store. ~50 GB → ~600 MB peak.
    needed_columns = (
        'id',
        'online_max_q_per_step', 'online_min_q_per_step',
        'online_mean_q_per_step',
        'online_argmax_per_step', 'target_argmax_per_step',
        'td_error', 'buf_size',
    )
    print(f'  streaming traces (cols={len(needed_columns)}, '
          f'batch_size=32, zarr skipped)')

    # Project mediators per cell as we stream.
    mediator_rows: list[dict[str, object]] = []
    n_skipped = 0
    for record in iter_trace_records(
        _TRACES_PATH, columns=needed_columns,
    ):
        cell_id = record.get('id')
        if not isinstance(cell_id, str) or cell_id not in leaf_lookup:
            n_skipped += 1
            continue
        leafs = leaf_lookup[cell_id]
        capacity_v = leafs.get(_LEAF_KEY_REPLAY_CAPACITY)
        capacity = int(capacity_v) if capacity_v is not None else 0  # type: ignore[arg-type]
        eps_init_v = leafs.get('action_select.schedule.eps_init')
        eps_final_v = leafs.get('action_select.schedule.eps_final')
        anneal_v = leafs.get('action_select.schedule.anneal_steps')
        total_v = leafs.get('total_steps')
        eps_init = float(eps_init_v) if eps_init_v is not None else float('nan')  # type: ignore[arg-type]
        eps_final = (
            float(eps_final_v) if eps_final_v is not None else float('nan')  # type: ignore[arg-type]
        )
        anneal_steps = int(anneal_v) if anneal_v is not None else 0  # type: ignore[arg-type]
        total_steps = int(total_v) if total_v is not None else 0  # type: ignore[arg-type]
        mediator_rows.append({
            'id': cell_id,
            'mediator.q_gap_late': q_gap_late(record),
            'mediator.q_gap_growth': q_gap_growth(record),
            'mediator.q_max_growth': q_max_growth(record),
            'mediator.v_vs_max_delta_late': v_vs_max_delta_late(record),
            'mediator.td_residual_late': td_residual_late(record),
            'mediator.greedy_match_late': greedy_match_late(record),
            'mediator.fill_ratio_late': fill_ratio_late(
                record, capacity=capacity,
            ),
            'mediator.epsilon_late': _epsilon_late(
                eps_init=eps_init, eps_final=eps_final,
                anneal_steps=anneal_steps, total_steps=total_steps,
            ),
        })
    if n_skipped:
        print(f'  warning: {n_skipped} runs had no trace (skipped)')

    mediator_df = pl.DataFrame(mediator_rows)
    enriched = runs_df.join(mediator_df, on='id', how='left')
    enriched.write_parquet(_OUT_PATH)
    print(f'wrote {enriched.height} rows × {len(enriched.columns)} cols '
          f'→ {_OUT_PATH.name}')

    print()
    print('=' * 72)
    print('Mediator summary (over all cells)')
    print('=' * 72)
    for c in sorted(c for c in enriched.columns if c.startswith('mediator.')):
        series = enriched[c]
        non_null = series.drop_nulls()
        if non_null.is_empty():
            print(f'  {c:<32}  (all NaN)')
            continue
        print(
            f'  {c:<32}  '
            f'n={non_null.len():>4}  '
            f'mean={float(non_null.mean() or 0.0):>+.3f}  '
            f'std={float(non_null.std() or 0.0):>.3f}  '
            f'min={float(non_null.min() or 0.0):>+.3f}  '
            f'max={float(non_null.max() or 0.0):>+.3f}',
        )


if __name__ == '__main__':
    main()
