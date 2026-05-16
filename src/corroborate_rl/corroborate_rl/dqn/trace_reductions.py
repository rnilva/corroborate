"""Canonical trace post-reductions for DQN traces.

The substrate's `dqn_step` emits per-step `online_q_per_action` /
`target_q_per_action` shape `(steps, n_actions)`. Persisting those
2-D arrays per cell is wasteful — every analysis subsequently
reduces them to per-step scalars (max/min/mean/std/argmax across
actions). This module ships the canonical reduction exprs so
sweep scripts don't re-author them every time.

Use:
    from corroborate_rl.dqn.trace_reductions import (
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

import math
from collections.abc import Mapping, Sequence

import numpy as np
import polars as pl

from corroborate.measurables import Measurable, from_key, register_as
from corroborate.measurables.reductions import reduce_axis


# ============ Module-top shared sources (trace-time leaves) ============
#
# 2-D source columns produced by the JAX kernel (shape `(T, A)` per
# row): per-step per-action online + target Q estimates. Declared
# once here so the trace-time `Q_TRACE_REDUCTIONS` entries compose
# Measurables rooted at the same `from_key` instance via Python
# identity; `compose_of` chains then reach back through these
# shared roots uniformly.

ONLINE_Q_PER_ACTION = from_key('online_q_per_action')
TARGET_Q_PER_ACTION = from_key('target_q_per_action')


def _measurable_as_polars_expr(
    m: Measurable[Mapping[str, object], object],
    return_dtype: pl.DataType,
) -> pl.Expr:
    """Wrap a Measurable as a polars expression that invokes it per
    row at trace-persistence time.

    Each polars row is materialised as a struct over `m.reads`; the
    Measurable is called on the resulting dict and its output
    becomes the cell value. Output is `.tolist()`-converted when
    ndarray so the polars `List` return-dtype path doesn't have to
    do the conversion itself.

    `alias(m.name)` pins the emitted column name to the Measurable's
    name — the substrate must `register_as(...)` each composition
    with a stable hand-picked name to preserve cache-column
    contracts (`pl.col('online_max_q_per_step')` in downstream
    scope predicates).
    """
    reads_list = list(m.reads)

    def _invoke(row: object) -> object:
        if not isinstance(row, Mapping):
            return None
        result = m(dict(row))
        if isinstance(result, np.ndarray):
            return result.tolist()
        return result

    return pl.struct(reads_list).map_elements(
        _invoke,
        return_dtype=return_dtype,
    ).alias(m.name)


def _per_step_target_q_at_online_argmax(s: object) -> list[float]:
    """Per-step DDQN bootstrap value: `target_q_per_action[
    argmax_a online_q_per_action[a]]`. Operates on a row-level
    struct of `{online_q_per_action: list[list[float]],
    target_q_per_action: list[list[float]]}`."""
    if not isinstance(s, dict):
        return []
    online = s.get('online_q_per_action')
    target = s.get('target_q_per_action')
    if online is None or target is None:
        return []
    out: list[float] = []
    for o_step, t_step in zip(online, target):
        if not o_step or not t_step:
            out.append(float('nan'))
            continue
        argmax_o = max(range(len(o_step)), key=lambda i: o_step[i])
        if argmax_o < len(t_step):
            out.append(float(t_step[argmax_o]))
        else:
            out.append(float('nan'))
    return out


def _per_step_argmax_q(s: pl.Series) -> list[int]:
    return [
        int(max(range(len(p)), key=lambda i: p[i])) if p else -1
        for p in s.to_list()
    ]


def _per_step_top1_top2_margin(s: pl.Series) -> list[float]:
    """Per-step argmax-margin: `Q(top1) − Q(top2)` across actions.
    Captures argmax-bias-sensitivity: small margin → argmax fragile
    to bias differential; large margin → bias differential below
    margin → argmax robust → DDQN's bias-correction is policy-
    irrelevant on this state. Continuous structural variable; |A|
    only sets an upper bound on flip-paths."""
    out: list[float] = []
    for p in s.to_list():
        if p is None or len(p) < 2:
            out.append(float('nan'))
            continue
        sorted_q = sorted(p, reverse=True)
        out.append(float(sorted_q[0] - sorted_q[1]))
    return out


def _q_action_temporal_corr_at_state_late(s: object) -> float:
    """Per-cell scalar: mean pairwise off-diagonal Pearson r of
    `online_q_per_action[t, a_i]` vs `online_q_per_action[t, a_j]`
    across all (i, j) action pairs, computed within each state
    that's revisited ≥ 5 times in late training, then averaged
    over states. Range [-1, 1].

    **What this actually measures:** TEMPORAL Q-value correlation
    across action heads at the same state over training time. NOT
    direct gradient-overlap (the theoretical intra-state α). The
    distinction matters: linear FA `Q(s, a) = W_a · obs + b_a` has
    **zero** gradient overlap by construction (W_a, W_a' are
    independent rows), but the temporal proxy here can still
    return values near 1 because:

    1. Limited-capacity FA collapses Q(s, ·) toward a low-rank
       function → actions appear correlated over time even with
       independent weight updates.
    2. Common training signal (rewards, target-network updates)
       drives all action Q's in correlated ways via the
       env-trajectory.

    Empirical evidence (FA-depth pilot 2026-05-13): linear MLP[]
    on FR gives temporal_corr ≈ 0.97-1.00, deep MLP[64,64] gives
    0.94-0.99 — they're INDISTINGUISHABLE on this measure. The
    theoretical intra-state α distinction lives at the gradient
    level (not Q-value level) and requires a separate probe
    (`q_action_grad_overlap_late` — not implemented; see
    `findings_fa_depth_within_env.md`).

    Useful as: a **Q-functional-rank-deficiency** indicator.
    High value (→1) means action Q's at this state are
    effectively a 1-D process over training — informative for
    detecting capacity-limited or degenerate regimes, not
    informative for distinguishing FA architecture coupling.

    Computed BEFORE `Q_TRACE_DROPS` removes the per-action vector
    so the source column remains visible.

    Operates on a row-level struct of
    `{online_q_per_action: list[list[float]], state_hash_per_step: list[int]}`.
    Returns NaN when no state has ≥ 5 late-window visits or all
    visits have zero variance (degenerate input)."""
    if not isinstance(s, Mapping):
        return float('nan')
    q = s.get('online_q_per_action')
    h = s.get('state_hash_per_step')
    if q is None or h is None:
        return float('nan')
    if not isinstance(q, Sequence) or not isinstance(h, Sequence):
        return float('nan')
    n = len(q)
    if n != len(h) or n < 2:
        return float('nan')
    half = n // 2
    q_late = q[half:]
    h_late = h[half:]
    # Bucket visits by state hash
    buckets: dict[int, list[Sequence[float]]] = {}
    for q_t, h_t in zip(q_late, h_late):
        if q_t is None or h_t is None:
            continue
        buckets.setdefault(int(h_t), []).append(q_t)
    pair_corrs: list[float] = []
    for visits in buckets.values():
        if len(visits) < 5:
            continue
        # Per-action time series at this state
        n_actions = len(visits[0])
        if n_actions < 2:
            continue
        cols: list[list[float]] = [[] for _ in range(n_actions)]
        for q_vec in visits:
            if len(q_vec) != n_actions:
                cols = []
                break
            for a, v in enumerate(q_vec):
                cols[a].append(float(v))
        if not cols:
            continue
        # Pairwise off-diagonal Pearson r
        for i in range(n_actions):
            xi = cols[i]
            mx = sum(xi) / len(xi)
            var_x = sum((v - mx) ** 2 for v in xi)
            if var_x == 0.0:
                continue
            for j in range(i + 1, n_actions):
                yj = cols[j]
                my = sum(yj) / len(yj)
                var_y = sum((v - my) ** 2 for v in yj)
                if var_y == 0.0:
                    continue
                cov = sum(
                    (xv - mx) * (yv - my) for xv, yv in zip(xi, yj)
                )
                r = cov / math.sqrt(var_x * var_y)
                if math.isfinite(r):
                    pair_corrs.append(r)
    if not pair_corrs:
        return float('nan')
    return sum(pair_corrs) / len(pair_corrs)


# ============ Trace-time Measurable compositions ============
#
# Each of the 5 simple reductions below is `reduce_axis(SOURCE,
# axis=-1, op=...)` wrapped under `register_as` so it carries the
# stable hand-picked column name (`online_max_q_per_step` etc.) —
# preserving the existing cache-column / scope-predicate contract
# while the underlying Measurable threads `compose_of=(SOURCE,)`
# for structural lineage.
#
# The remaining 4 closures stay as polars closures: top1/top2-
# margin and per-step argmax aren't covered by the framework's
# vocabulary (would need new primitives), `_per_step_target_q_at
# _online_argmax` is per-step indexing (a one-shot scalar
# `select_at` doesn't apply), and `_q_action_temporal_corr_at_
# state_late` is irreducible domain logic.

_online_max_q_m = register_as(
    reduce_axis(ONLINE_Q_PER_ACTION, axis=-1, op='max'),
    name='online_max_q_per_step',
)
_target_max_q_m = register_as(
    reduce_axis(TARGET_Q_PER_ACTION, axis=-1, op='max'),
    name='target_max_q_per_step',
)
_online_min_q_m = register_as(
    reduce_axis(ONLINE_Q_PER_ACTION, axis=-1, op='min'),
    name='online_min_q_per_step',
)
_online_mean_q_m = register_as(
    reduce_axis(ONLINE_Q_PER_ACTION, axis=-1, op='mean'),
    name='online_mean_q_per_step',
)
_online_std_q_m = register_as(
    reduce_axis(ONLINE_Q_PER_ACTION, axis=-1, op='std'),
    name='online_std_q_per_step',
)


Q_TRACE_REDUCTIONS: tuple[pl.Expr, ...] = (
    _measurable_as_polars_expr(_online_max_q_m, pl.List(pl.Float64)),
    _measurable_as_polars_expr(_target_max_q_m, pl.List(pl.Float64)),
    _measurable_as_polars_expr(_online_min_q_m, pl.List(pl.Float64)),
    _measurable_as_polars_expr(_online_mean_q_m, pl.List(pl.Float64)),
    _measurable_as_polars_expr(_online_std_q_m, pl.List(pl.Float64)),
    pl.col('online_q_per_action').map_elements(
        _per_step_top1_top2_margin, return_dtype=pl.List(pl.Float64),
    ).alias('online_top12_margin_per_step'),
    pl.col('online_q_per_action').map_elements(
        _per_step_argmax_q, return_dtype=pl.List(pl.Int64),
    ).alias('online_argmax_per_step'),
    pl.col('target_q_per_action').map_elements(
        _per_step_argmax_q, return_dtype=pl.List(pl.Int64),
    ).alias('target_argmax_per_step'),
    # DDQN's bootstrap value per step: target Q at online's argmax.
    # Per-step argmax-indexing isn't covered by the framework's
    # `select_at` (one-shot scalar reducer); stays as a polars
    # closure until a per-step indexer primitive earns its keep.
    pl.struct(['online_q_per_action', 'target_q_per_action']).map_elements(
        _per_step_target_q_at_online_argmax,
        return_dtype=pl.List(pl.Float64),
    ).alias('target_q_at_online_argmax_per_step'),
    # Per-cell scalar (NOT per-step list) — intra-state coupling
    # proxy. Computed here because `online_q_per_action` is dropped
    # immediately after; no @measurable can recover it post-hoc.
    pl.struct(['online_q_per_action', 'state_hash_per_step']).map_elements(
        _q_action_temporal_corr_at_state_late,
        return_dtype=pl.Float64,
    ).alias('q_action_temporal_corr_at_state_late'),
)

Q_TRACE_DROPS: tuple[str, ...] = (
    'online_q_per_action',
    'target_q_per_action',
)


__all__ = ['Q_TRACE_REDUCTIONS', 'Q_TRACE_DROPS']
