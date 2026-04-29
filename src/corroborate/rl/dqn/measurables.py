"""DDQN measurables — declared scalar / series derivations of
the per-step record. Each is a `@measurable` registered in the
framework's name-keyed registry.

These are the *substrate-level* derivations bridges, invariants,
and diagnostics consume across the DDQN claim graph. Reductions
live here (not in `train_phase`) so that:

- The training loop only emits raw record fields plus the
  cheapest in-loop sum-stats.
- Multiple bridges sharing a derivation (q_mean reads in three
  places) compute it exactly once per record, memoized by the
  framework.
- Authors who want a NEW measurable add it here without changing
  `train_phase` or any bridge body.

`reads` declarations are explicit so the redundancy primitive
(`compute_redundancy(h, G)`) can fingerprint each bridge by its
transitive read-set.

Two output shapes coexist:

1. **Per-step series Measurables** — return `NDArray[Float64]`
   shape `(steps,)`. Q-distribution shape (`q_mean_per_step`,
   `q_max_per_step`, `q_std_per_step`, `q_gap_per_step`),
   target-network coverage (`target_q_mean_per_step`,
   `target_q_max_per_step`), TD-error magnitude
   (`td_error_norm_per_step`).

2. **Per-cell scalar Measurables** — return `float`. Late-window
   reductions over per-step columns persisted by the trace
   reductions (`online_max_q_per_step`, `online_min_q_per_step`,
   `online_mean_q_per_step`, `online_argmax_per_step`,
   `target_argmax_per_step`, `td_error`, `buf_size`). Used as
   inputs to PAPER §5/§6's mediator analysis (within-env Pearson
   + per-env PC); the framework treats them as plain
   Measurables — the "mediator" framing is paper-section domain
   language, not a framework concept."""
from __future__ import annotations

from collections.abc import Mapping

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from corroborate.measurable import measurable


# ============ Online Q distribution ============

# Inputs are `online_q_per_action` shape `(steps, n_actions)` — the
# in-loop batch-averaged per-step Q vector. Reductions collapse the
# action axis to per-step scalars.

@measurable(reads=('online_q_per_action',))
def q_mean_per_step(
    record: Mapping[str, object],
) -> npt.NDArray[np.float64]:
    """Mean Q across actions, per training step. Series shape
    `(steps,)`."""
    arr = np.asarray(record['online_q_per_action']).astype(np.float64)
    return arr.mean(axis=-1) if arr.ndim >= 1 else arr


@measurable(reads=('online_q_per_action',))
def q_max_per_step(
    record: Mapping[str, object],
) -> npt.NDArray[np.float64]:
    """Max Q across actions, per training step."""
    arr = np.asarray(record['online_q_per_action']).astype(np.float64)
    return arr.max(axis=-1) if arr.ndim >= 1 else arr


@measurable(reads=('online_q_per_action',))
def q_std_per_step(
    record: Mapping[str, object],
) -> npt.NDArray[np.float64]:
    """Std Q across actions, per step. Captures the spread of
    the action-value distribution at each training step."""
    arr = np.asarray(record['online_q_per_action']).astype(np.float64)
    return arr.std(axis=-1) if arr.ndim >= 1 else arr


@measurable(reads=('online_q_per_action',))
def q_gap_per_step(
    record: Mapping[str, object],
) -> npt.NDArray[np.float64]:
    """Gap between max and second-max Q-values per step. The
    DDQN-relevant signal: a small gap means the greedy action is
    barely distinguishable from a runner-up — the regime where
    overestimation matters most."""
    arr = np.asarray(record['online_q_per_action']).astype(np.float64)
    if arr.ndim < 2 or arr.shape[-1] < 2:
        return np.zeros(arr.shape[:-1], dtype=np.float64)
    sorted_arr = np.sort(arr, axis=-1)
    return sorted_arr[..., -1] - sorted_arr[..., -2]


# ============ Target Q distribution ============

@measurable(reads=('target_q_per_action',))
def target_q_mean_per_step(
    record: Mapping[str, object],
) -> npt.NDArray[np.float64]:
    """Mean target-Q across actions, per step."""
    arr = np.asarray(record['target_q_per_action']).astype(np.float64)
    return arr.mean(axis=-1) if arr.ndim >= 1 else arr


@measurable(reads=('target_q_per_action',))
def target_q_max_per_step(
    record: Mapping[str, object],
) -> npt.NDArray[np.float64]:
    """Max target-Q across actions, per step."""
    arr = np.asarray(record['target_q_per_action']).astype(np.float64)
    return arr.max(axis=-1) if arr.ndim >= 1 else arr


# ============ TD-error magnitude ============

@measurable(reads=('td_error',))
def td_error_norm_per_step(
    record: Mapping[str, object],
) -> npt.NDArray[np.float64]:
    """Per-step √mean(td_error²) — a positive scalar magnitude."""
    arr = np.asarray(record['td_error']).astype(np.float64)
    if arr.ndim == 0:
        return np.abs(arr)
    if arr.ndim == 1:
        return np.abs(arr)
    return np.sqrt(np.mean(arr ** 2, axis=tuple(range(1, arr.ndim))))


# ============ Pearson r — online vs target Q populations ============

@measurable(reads=('pearson_stats',))
def pearson_r_online_target(record: Mapping[str, object]) -> float:
    """Population-level Pearson r between online-Q and target-Q
    over all (step, sample, action) triples observed during
    training. Aggregates the per-step `pearson_stats` sufficient
    statistics. Same scalar `hasselt_covariance_gap` reduces to;
    exposed here as a reusable measurable so other bridges /
    diagnostics can read it without redoing the math."""
    ps = jnp.asarray(record['pearson_stats'])
    agg = ps.mean(axis=0)
    m_on, m_tg, m_on_sq, m_tg_sq, m_xy = (
        float(agg[0]), float(agg[1]), float(agg[2]),
        float(agg[3]), float(agg[4]),
    )
    var_on = m_on_sq - m_on ** 2
    var_tg = m_tg_sq - m_tg ** 2
    if var_on <= 1e-12 or var_tg <= 1e-12:
        return float('nan')
    cov = m_xy - m_on * m_tg
    return cov / (var_on * var_tg) ** 0.5


# ============ Per-cell scalar Measurables ============
#
# Late-window reductions over the per-step columns persisted by
# the trace reductions in `experiments/collect_ddqn_runs.py`'s
# `TRACE_POST_REDUCTIONS`: `online_max_q_per_step`,
# `online_min_q_per_step`, `online_mean_q_per_step`,
# `online_argmax_per_step`, `target_argmax_per_step` (plus the
# raw `td_error` and `buf_size`).
#
# Used as inputs to PAPER §5/§6's mediator analysis. The framework
# treats them as plain Measurables — the "candidate mediator"
# framing is paper-section domain language, not a framework
# concept. They become claims-in-the-framework-sense when wrapped
# in a Bridge or Invariant (`bounded(q_gap_late, 1e3)`,
# `paired_hedges_g(treatment_q_gap_late, baseline_q_gap_late)`).
# The §5/§6 analysis uses them via corpus-level Pearson / PC,
# which is a third pattern (corpus-level diagnostic).


def _mean_window(
    arr: npt.NDArray[np.float64], lo: float, hi: float,
) -> float:
    """Mean over the fractional window [lo, hi] of `arr`'s
    leading axis. Returns NaN for empty input. Inlined from
    `corroborate.reductions.mean_window` to keep the per-cell
    numpy hot path direct (no `Measurable` indirection)."""
    n = arr.shape[0]
    if n == 0:
        return float('nan')
    i_lo = int(lo * n)
    i_hi = max(int(hi * n), i_lo + 1)
    return float(np.mean(arr[i_lo:i_hi]))


def _record_array(
    record: Mapping[str, object], key: str,
) -> npt.NDArray[np.float64] | None:
    """Pull `record[key]` as a float64 numpy array. Returns None
    if the key is absent so callers can NaN-propagate."""
    if key not in record:
        return None
    return np.asarray(record[key], dtype=np.float64)


@measurable(reads=('online_max_q_per_step', 'online_min_q_per_step'))
def q_gap_late(record: Mapping[str, object]) -> float:
    """Mean of (online_max_q − online_min_q) over the last 50% of
    training. Action-margin scalar — sharp policy preferences
    correlate with reward in Breakout / MNISTBandit /
    SpaceInvaders (PAPER §5.1)."""
    max_q = _record_array(record, 'online_max_q_per_step')
    min_q = _record_array(record, 'online_min_q_per_step')
    if max_q is None or min_q is None:
        return float('nan')
    return _mean_window(max_q - min_q, 0.5, 1.0)


@measurable(reads=('online_max_q_per_step', 'online_min_q_per_step'))
def q_gap_growth(record: Mapping[str, object]) -> float:
    """(late_half_mean − early_half_mean) of the q_gap signal.
    Captures whether action-margin widens as training progresses
    (positive growth → emergent policy sharpness)."""
    max_q = _record_array(record, 'online_max_q_per_step')
    min_q = _record_array(record, 'online_min_q_per_step')
    if max_q is None or min_q is None:
        return float('nan')
    gap = max_q - min_q
    early = _mean_window(gap, 0.0, 0.5)
    late = _mean_window(gap, 0.5, 1.0)
    return float(late - early)


@measurable(reads=('online_max_q_per_step',))
def q_max_growth(record: Mapping[str, object]) -> float:
    """late_quarter / max(|early_quarter|, 1e-9) of online_max_q.
    Value-curve growth — vanilla DQN's Jensen bias typically
    pushes this above 1; DDQN attenuates."""
    arr = _record_array(record, 'online_max_q_per_step')
    if arr is None:
        return float('nan')
    early = _mean_window(arr, 0.0, 0.25)
    late = _mean_window(arr, 0.75, 1.0)
    return float(late / max(abs(early), 1e-9))


@measurable(reads=('online_mean_q_per_step', 'online_max_q_per_step'))
def v_vs_max_delta_late(record: Mapping[str, object]) -> float:
    """Mean of |q_mean − q_max| over the late 50%. DDQN's
    mechanism signature: vanilla's overestimation widens the
    action-Q distribution (large delta); DDQN's decoupled
    selection narrows it. The Hasselt 2010 Jensen-bias proxy at
    the per-step level."""
    mean_q = _record_array(record, 'online_mean_q_per_step')
    max_q = _record_array(record, 'online_max_q_per_step')
    if mean_q is None or max_q is None:
        return float('nan')
    return _mean_window(np.abs(mean_q - max_q), 0.5, 1.0)


@measurable(reads=('td_error',))
def td_residual_late(record: Mapping[str, object]) -> float:
    """Mean of |TD residual| over the late 50%. TD-convergence
    scalar — Acrobot's r=+0.84 vs GaussianBandit's r=−0.81
    (sign-flip across regimes) is the canonical motivation for
    per-env PC in PAPER §6."""
    arr = _record_array(record, 'td_error')
    if arr is None:
        return float('nan')
    return _mean_window(arr, 0.5, 1.0)


@measurable(reads=('online_argmax_per_step', 'target_argmax_per_step'))
def greedy_match_late(record: Mapping[str, object]) -> float:
    """Mean of (online_argmax == target_argmax) over the late
    50%. DDQN's slot swap explicitly decouples these argmaxes;
    large match ⇒ DDQN's mechanism is *inactive* on this cell
    (the two estimators agree, so vanilla and DDQN reduce to
    each other)."""
    online = _record_array(record, 'online_argmax_per_step')
    target = _record_array(record, 'target_argmax_per_step')
    if online is None or target is None:
        return float('nan')
    match = (online == target).astype(np.float64)
    return _mean_window(match, 0.5, 1.0)


@measurable(reads=('buf_size',))
def fill_ratio_late(
    record: Mapping[str, object], *, capacity: int,
) -> float:
    """Mean of buf_size / capacity over the late 50%. Coverage
    scalar — when buf < capacity throughout training the agent is
    still in the early-replay regime; when full, bootstrapping
    operates on a stationary buffer.

    NOTE: `capacity` is an extra kwarg the framework's auto-
    resolver does NOT fill — it's neither a record key nor a
    registered measurable. Callers must pass it directly:
    `fill_ratio_late(record, capacity=10_000)`. The Measurable
    wrapper still tracks `reads=('buf_size',)` for downstream
    redundancy / reads-set fingerprinting."""
    if capacity <= 0:
        return float('nan')
    arr = _record_array(record, 'buf_size')
    if arr is None:
        return float('nan')
    return _mean_window(arr / float(capacity), 0.5, 1.0)
