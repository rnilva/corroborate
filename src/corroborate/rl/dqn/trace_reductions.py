"""Canonical trace post-reductions for DQN traces.

The substrate's `dqn_step` emits per-step `online_q_per_action` /
`target_q_per_action` shape `(steps, n_actions)`. Persisting those
2-D arrays per cell is wasteful — every analysis subsequently
reduces them to per-step scalars (max/min/mean/std/argmax across
actions). This module ships the canonical reduction exprs so
sweep scripts don't re-author them every time.

Use:
    from corroborate.rl.dqn.trace_reductions import (
        Q_TRACE_REDUCTIONS, Q_TRACE_DROPS,
    )
    apply_trace_reductions(
        traces, add=Q_TRACE_REDUCTIONS, drop=Q_TRACE_DROPS,
    )

Reductions emitted (per cell, length = total_steps):
- `online_max_q_per_step`, `online_min_q_per_step`,
  `online_mean_q_per_step`, `online_std_q_per_step` —
  per-step (max, min, mean, std-across-actions) of the
  online Q vector. The std is the σ_Q input to
  `jensen_dormancy_gap` (`σ × √(2 log |A|)`).
- `online_argmax_per_step`, `target_argmax_per_step` —
  per-step argmax-action indices. Used by
  `mediator.greedy_match_late` (DDQN's argmax decoupling
  signature).
- `target_max_q_per_step` — per-step max of target Q
  (vanilla's bootstrap target).

Drops `online_q_per_action` and `target_q_per_action` after
reduction — the per-step scalars carry enough information for
all current measurables and bridges, and the drop shrinks
parquets ~64×."""
from __future__ import annotations

import statistics

import polars as pl


def _per_step_max_q(s: pl.Series) -> list[float]:
    return [max(per_action) for per_action in s.to_list()]


def _per_step_min_q(s: pl.Series) -> list[float]:
    return [min(per_action) for per_action in s.to_list()]


def _per_step_mean_q(s: pl.Series) -> list[float]:
    return [
        sum(p) / len(p) if p else float('nan')
        for p in s.to_list()
    ]


def _per_step_argmax_q(s: pl.Series) -> list[int]:
    return [
        int(max(range(len(p)), key=lambda i: p[i])) if p else -1
        for p in s.to_list()
    ]


def _per_step_std_q(s: pl.Series) -> list[float]:
    out: list[float] = []
    for p in s.to_list():
        out.append(
            float('nan') if p is None or len(p) < 2
            else float(statistics.pstdev(p))
        )
    return out


Q_TRACE_REDUCTIONS: tuple[pl.Expr, ...] = (
    pl.col('online_q_per_action').map_elements(
        _per_step_max_q, return_dtype=pl.List(pl.Float64),
    ).alias('online_max_q_per_step'),
    pl.col('target_q_per_action').map_elements(
        _per_step_max_q, return_dtype=pl.List(pl.Float64),
    ).alias('target_max_q_per_step'),
    pl.col('online_q_per_action').map_elements(
        _per_step_min_q, return_dtype=pl.List(pl.Float64),
    ).alias('online_min_q_per_step'),
    pl.col('online_q_per_action').map_elements(
        _per_step_mean_q, return_dtype=pl.List(pl.Float64),
    ).alias('online_mean_q_per_step'),
    pl.col('online_q_per_action').map_elements(
        _per_step_std_q, return_dtype=pl.List(pl.Float64),
    ).alias('online_std_q_per_step'),
    pl.col('online_q_per_action').map_elements(
        _per_step_argmax_q, return_dtype=pl.List(pl.Int64),
    ).alias('online_argmax_per_step'),
    pl.col('target_q_per_action').map_elements(
        _per_step_argmax_q, return_dtype=pl.List(pl.Int64),
    ).alias('target_argmax_per_step'),
)

Q_TRACE_DROPS: tuple[str, ...] = (
    'online_q_per_action',
    'target_q_per_action',
)


__all__ = ['Q_TRACE_REDUCTIONS', 'Q_TRACE_DROPS']
