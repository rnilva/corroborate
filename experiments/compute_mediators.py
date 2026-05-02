"""Project the 8 §5 candidate measurables across the corpus.

Reads the canonical `runs.parquet` + `traces.parquet`, joins by
id, computes the eight measurable scalars per cell, writes
`runs_with_mediators.parquet` with new `mediator.*` columns
alongside the original run-level columns.

After this, downstream analyses consume the enriched runs via
`experiments/analyze.py --stages pc` (with the appropriate
mediator paths surfaced as variables).

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
    learning_curve_auc,
    plateau_slope_late,
    q_gap_growth,
    q_gap_late,
    q_max_growth,
    return_at_25pct_steps,
    state_coverage_kl_uniform_late,
    state_visit_entropy_late,
    td_residual_late,
    td_within_batch_var_late,
    time_to_threshold,
    v_vs_max_delta_late,
)


def _peak_idx_for_arr(
    arr_len: int, peak_step: float, total_steps: int,
) -> int:
    """Map a per-cell peak step (in training-step units) to an
    array index for a per-step trajectory of length `arr_len`.
    Returns -1 when the inputs are unusable."""
    if arr_len < 2 or total_steps < 1 or peak_step < 0:
        return -1
    return min(arr_len, int(peak_step * arr_len / total_steps))


def _q_gap_peak_truncated_late(
    record: dict[str, object], peak_step: float, total_steps: int,
) -> float:
    """Mean of (online_max − online_min) over [peak/2, peak] of
    the per-step trajectory — the peak-truncated late-half mean.
    Plain Python; mirrors `mean_peak_window`'s window logic."""
    max_q_v = record.get('online_max_q_per_step')
    min_q_v = record.get('online_min_q_per_step')
    if max_q_v is None or min_q_v is None:
        return float('nan')
    arr = (
        np.asarray(max_q_v, dtype=np.float64)
        - np.asarray(min_q_v, dtype=np.float64)
    )
    peak_idx = _peak_idx_for_arr(len(arr), peak_step, total_steps)
    if peak_idx < 4:
        return float('nan')
    lo = max(0, peak_idx // 2)
    return float(np.mean(arr[lo:peak_idx]))


def _td_residual_peak_truncated_late(
    record: dict[str, object], peak_step: float, total_steps: int,
) -> float:
    """Mean of |td_error| over [peak/2, peak]."""
    td_v = record.get('td_error')
    if td_v is None:
        return float('nan')
    arr = np.abs(np.asarray(td_v, dtype=np.float64))
    peak_idx = _peak_idx_for_arr(len(arr), peak_step, total_steps)
    if peak_idx < 4:
        return float('nan')
    lo = max(0, peak_idx // 2)
    return float(np.mean(arr[lo:peak_idx]))


def _greedy_match_peak_truncated_late(
    record: dict[str, object], peak_step: float, total_steps: int,
) -> float:
    """Mean of (online_argmax == target_argmax) over [peak/2,
    peak]. Higher = more online-target greedy agreement before
    the cell's peak."""
    online_v = record.get('online_argmax_per_step')
    target_v = record.get('target_argmax_per_step')
    if online_v is None or target_v is None:
        return float('nan')
    online = np.asarray(online_v, dtype=np.int64)
    target = np.asarray(target_v, dtype=np.int64)
    peak_idx = _peak_idx_for_arr(len(online), peak_step, total_steps)
    if peak_idx < 4:
        return float('nan')
    lo = max(0, peak_idx // 2)
    match = (online == target).astype(np.float64)
    return float(np.mean(match[lo:peak_idx]))


def _learning_curve_auc_peak_truncated(
    record: dict[str, object], peak_step: float, total_steps: int,
) -> float:
    """Trapezoidal AUC of per-burst mean return over bursts
    [0, peak_burst_idx]. Captures the rising portion of the
    learning curve, avoiding post-peak collapse."""
    mc_v = record.get('mc_return')
    if mc_v is None:
        return float('nan')
    mc = np.asarray(mc_v, dtype=np.float64)
    if mc.ndim != 2 or mc.size == 0:
        return float('nan')
    burst_means = np.mean(mc, axis=1)
    n_bursts = len(burst_means)
    if n_bursts < 2 or total_steps < 1:
        return float('nan')
    eval_every = max(1, total_steps // n_bursts)
    peak_burst_idx = min(n_bursts, int(peak_step // eval_every) + 1)
    if peak_burst_idx < 2:
        return float('nan')
    sub = burst_means[:peak_burst_idx]
    return float(np.trapezoid(sub) / (sub.size - 1))


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

    runs_df = pl.read_parquet(_RUNS_PATH)
    print(f'  {runs_df.height} rows × {len(runs_df.columns)} cols')

    # Build per-id lookup of the leaf HPs needed by epsilon_late /
    # fill_ratio_late, plus `eval_best_burst_step` used by
    # peak-truncated reductions. Cheap — runs.parquet has only
    # scalar columns.
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
                    'eval_best_burst_step',
                )
            }

    # Streaming read: only the trace columns mediators actually
    # consume. iter_trace_records bounds peak memory at
    # O(batch_size × per-row-size) instead of materialising the
    # full trace store. ~50 GB → ~600 MB peak.
    needed_columns = (
        'id',
        # 1-D per-step trajectory series.
        'online_max_q_per_step', 'online_min_q_per_step',
        'online_mean_q_per_step',
        'online_argmax_per_step', 'target_argmax_per_step',
        'td_error', 'td_error_within_batch_std', 'buf_size',
        'state_hash',  # F1 state-coverage family.
        # 2-D arrays for D3 value-curve family + jensen-gap
        # mechanism (nested-list columns in parquet).
        'mc_return',
        'predicted_q_at_start',
    )
    print(
        f'  streaming traces (cols={len(needed_columns)}, '
        f'batch_size=32)'
    )

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
        peak_step_v = leafs.get('eval_best_burst_step')
        peak_step = (
            float(peak_step_v) if peak_step_v is not None else float('nan')  # type: ignore[arg-type]
        )
        mediator_rows.append({
            'id': cell_id,
            'mediator.q_gap_late': q_gap_late(record),
            'mediator.q_gap_growth': q_gap_growth(record),
            'mediator.q_max_growth': q_max_growth(record),
            'mediator.v_vs_max_delta_late': v_vs_max_delta_late(record),
            'mediator.td_residual_late': td_residual_late(record),
            'mediator.td_within_batch_var_late': (
                td_within_batch_var_late(record)
            ),
            'mediator.greedy_match_late': greedy_match_late(record),
            'mediator.fill_ratio_late': fill_ratio_late(
                record, capacity=capacity,
            ),
            'mediator.epsilon_late': _epsilon_late(
                eps_init=eps_init, eps_final=eps_final,
                anneal_steps=anneal_steps, total_steps=total_steps,
            ),
            # D3 value-curve family — reductions on `mc_return`.
            'mediator.learning_curve_auc': learning_curve_auc(record),
            'mediator.time_to_threshold': time_to_threshold(record),
            'mediator.return_at_25pct_steps': return_at_25pct_steps(record),
            'mediator.plateau_slope_late': plateau_slope_late(record),
            # F1 state-coverage family. (Action-margin proxy
            # would equal the existing `v_vs_max_delta_late`;
            # deferred until a per-step second-max reduction is
            # added to the collect harness.)
            'mediator.state_visit_entropy_late':
                state_visit_entropy_late(record),
            'mediator.state_coverage_kl_uniform_late':
                state_coverage_kl_uniform_late(record),
            # H2 peak-truncated reductions: per-cell-aware
            # windows that avoid post-peak contamination.
            # Plain Python compositions (the framework's typed
            # `mean_peak_window` primitive in `reductions.py` is
            # available for any future Bridge / invariant; for
            # cache-population here, a direct numpy reduction is
            # simpler).
            'mediator.q_gap_peak_truncated_late':
                _q_gap_peak_truncated_late(
                    dict(record), peak_step, total_steps,
                ),
            'mediator.td_residual_peak_truncated_late':
                _td_residual_peak_truncated_late(
                    dict(record), peak_step, total_steps,
                ),
            'mediator.greedy_match_peak_truncated_late':
                _greedy_match_peak_truncated_late(
                    dict(record), peak_step, total_steps,
                ),
            'mediator.learning_curve_auc_peak_truncated':
                _learning_curve_auc_peak_truncated(
                    dict(record), peak_step, total_steps,
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
