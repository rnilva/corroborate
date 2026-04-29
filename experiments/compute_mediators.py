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

from corroborate.persistence import read_tracerows
from corroborate.rl.dqn.mediators import (
    epsilon_late,
    fill_ratio_late,
    greedy_match_late,
    q_gap_growth,
    q_gap_late,
    q_max_growth,
    td_residual_late,
    v_vs_max_delta_late,
)


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

    # Mediators only read 1-D per-step series (online_*_q_per_step,
    # td_error, buf_size, *_argmax_per_step) — all of which live
    # in the parquet, not zarr. Skip the zarr load entirely; it
    # holds eval-burst tensors (mc_return, predicted_q_at_start,
    # ...) that no mediator reads. This keeps the read step at
    # ~5 GB RAM instead of ~50 GB.
    traces = read_tracerows(_TRACES_PATH, zarr_path=None)
    print(f'  {len(traces)} traces (parquet-only, zarr skipped)')

    traces_by_id = {t.id: t for t in traces}

    # Project mediators per row, indexed by run id.
    mediator_rows: list[dict[str, object]] = []
    n_skipped = 0
    for row in runs_df.iter_rows(named=True):
        cell_id = row.get('id')
        if cell_id is None or cell_id not in traces_by_id:
            n_skipped += 1
            continue
        trace = traces_by_id[cell_id]
        record = trace.leaves
        # arrays from zarr also live on the trace (TraceRow.arrays);
        # mediators only need 1-D series (which are in `leaves`).
        capacity_v = row.get(_LEAF_KEY_REPLAY_CAPACITY)
        capacity = int(capacity_v) if capacity_v is not None else 0
        eps_init_v = row.get('action_select.schedule.eps_init')
        eps_final_v = row.get('action_select.schedule.eps_final')
        anneal_v = row.get('action_select.schedule.anneal_steps')
        total_v = row.get('total_steps')
        eps_init = float(eps_init_v) if eps_init_v is not None else float('nan')
        eps_final = (
            float(eps_final_v) if eps_final_v is not None else float('nan')
        )
        anneal_steps = int(anneal_v) if anneal_v is not None else 0
        total_steps = int(total_v) if total_v is not None else 0
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
            'mediator.epsilon_late': epsilon_late(
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
