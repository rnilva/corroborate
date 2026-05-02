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

from corroborate.measurable import Measurable, measurable


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


@measurable(reads=('td_error_within_batch_std',))
def td_within_batch_var_late(record: Mapping[str, object]) -> float:
    """Mean of within-batch std(|TD-error|) over the late 50%.

    Captures *training-signal heterogeneity*: at each training step,
    the spread of |TD-error| across the batch's transitions. High
    value = the batch averages diverse transitions (varied gradient
    directions); low value = the batch is dominated by similar
    transitions (correlated gradients).

    Theoretically a candidate mediator for `replay.capacity → solve`:
    larger replay → less correlated samples → higher within-batch
    variance → less catastrophic forgetting. Crucially *not*
    deterministic from `capacity` (depends on which transitions
    happened to be in the batch) and *not* a re-encoding of the
    outcome (reads `td_error_within_batch_std`, disjoint from
    `mc_return`)."""
    arr = _record_array(record, 'td_error_within_batch_std')
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


@measurable(reads=(
    'online_argmax_per_step', 'target_argmax_per_step',
    'eval_step_index',
))
def greedy_match_per_burst(
    record: Mapping[str, object],
) -> npt.NDArray[np.float64]:
    """Per-burst fraction of (online_argmax == target_argmax).
    Shape `(n_bursts,)`. Chunks the per-step argmax disagreement
    into n_bursts equal windows along the training trajectory.

    Direct measurement of "DDQN-mechanism-activation rate over
    training". 1.0 means the two argmaxes always agreed in that
    burst's training window (DDQN ≡ DQN per-step); 0.0 means
    they always disagreed (every TD update saw the slot swap
    bite). Used to characterize DDQN intrinsically — does the
    mechanism even fire? — without arm-comparison.

    `1 - greedy_match_per_burst` is the activation frequency."""
    online = _record_array(record, 'online_argmax_per_step')
    target = _record_array(record, 'target_argmax_per_step')
    eval_idx = _record_array(record, 'eval_step_index')
    if online is None or target is None or eval_idx is None:
        return np.zeros((0,), dtype=np.float64)
    match = (online == target).astype(np.float64)
    n_bursts = int(eval_idx.shape[0])
    if n_bursts == 0:
        return np.zeros((0,), dtype=np.float64)
    n_steps = match.shape[0]
    edges = np.linspace(0, n_steps, n_bursts + 1, dtype=np.int64)
    return np.array(
        [match[edges[i]:edges[i+1]].mean() for i in range(n_bursts)],
        dtype=np.float64,
    )


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


# ============ Value-curve features (PAPER §4.6 family 2) ============
#
# Reductions on the eval-burst learning curve `mc_return`. v10 §6
# left 6 envs with no detected mediator (CartPole, Catch-bsuite,
# FourRooms, MemoryChain, Asterix-MinAtar, plus one more); §4.6
# names value-curve features as a candidate mediator family that
# the existing q_gap / td_residual / greedy_match set doesn't
# capture. AUC, time-to-threshold, and plateau slope project the
# learning curve to scalars suitable as additional covariates in
# meta-regression.


def _burst_means_1d(
    record: Mapping[str, object],
) -> npt.NDArray[np.float64] | None:
    """Reduce `mc_return: (n_bursts, K)` to per-burst means
    `(n_bursts,)`. Returns None when the record lacks the field
    or the array is degenerate."""
    if 'mc_return' not in record:
        return None
    arr = np.asarray(record['mc_return'], dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        return None
    return np.mean(arr, axis=1)


@measurable(reads=('mc_return',))
def learning_curve_auc(record: Mapping[str, object]) -> float:
    """Trapezoidal AUC of the per-burst mean MC return. Larger AUC
    means the agent reaches and holds higher returns earlier and
    longer in training — a candidate mediator for envs where the
    *shape* of the learning curve, not just the final value,
    drives performance.

    AUC is normalised by the burst-axis length so it has the same
    units as `mc_return` (a return value), not return·step. This
    keeps the scale interpretable across runs of different total
    step budgets."""
    burst_means = _burst_means_1d(record)
    if burst_means is None or burst_means.size < 2:
        return float('nan')
    return float(np.trapezoid(burst_means) / (burst_means.size - 1))


@measurable(reads=('mc_return',))
def time_to_threshold(
    record: Mapping[str, object], *, target_frac: float = 0.5,
) -> float:
    """First burst index (as a fraction of n_bursts) where the
    burst mean reaches `target_frac × max(burst_means)`. Returns
    1.0 if the threshold is never crossed (the agent never
    reaches `target_frac` of its own peak — a degenerate case
    that callers should treat as 'never converged').

    Sample-efficiency proxy: smaller is faster convergence. The
    fractional-of-n encoding makes the scalar comparable across
    cells with different `n_bursts`."""
    burst_means = _burst_means_1d(record)
    if burst_means is None or burst_means.size == 0:
        return float('nan')
    peak = float(np.max(burst_means))
    if peak <= 0.0:
        return float('nan')
    threshold = target_frac * peak
    crossed = np.where(burst_means >= threshold)[0]
    if crossed.size == 0:
        return 1.0
    return float(crossed[0]) / float(burst_means.size - 1) if (
        burst_means.size > 1
    ) else 0.0


@measurable(reads=('mc_return',))
def return_at_25pct_steps(record: Mapping[str, object]) -> float:
    """Per-burst mean return at the 25%-of-training checkpoint —
    a smooth analog of 'sample-efficiency at 25% budget'.
    Compared with `outcome.eval_final_mean` it tells you whether
    most of the learning happened in the first quarter or
    spread out."""
    burst_means = _burst_means_1d(record)
    if burst_means is None or burst_means.size == 0:
        return float('nan')
    idx = burst_means.size // 4
    return float(burst_means[idx])


@measurable(reads=('mc_return',))
def plateau_slope_late(
    record: Mapping[str, object], *, frac: float = 0.25,
) -> float:
    """Least-squares slope of per-burst mean return over the last
    `frac` of bursts. Positive = still improving; near-zero =
    plateaued; negative = degrading. Distinguishes 'converged at
    high return' (slope ≈ 0, high level) from 'still climbing'
    (slope > 0)."""
    burst_means = _burst_means_1d(record)
    if burst_means is None or burst_means.size < 4:
        return float('nan')
    n = burst_means.size
    i_lo = int((1.0 - frac) * n)
    if n - i_lo < 2:
        return float('nan')
    y = burst_means[i_lo:]
    x = np.arange(y.size, dtype=np.float64)
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    cov = float(np.sum((x - x_mean) * (y - y_mean)))
    var = float(np.sum((x - x_mean) ** 2))
    if var <= 0.0:
        return float('nan')
    return cov / var


# ============ F1: State-coverage family ============
#
# Shannon entropy + KL-against-uniform of the `state_hash`
# distribution over the late window. v10 §4.6 candidate mediator
# family 3 (on-policy state-distribution coverage). `state_hash`
# is logged per step by `cell_runner` (default returns 0 for
# image envs whose hash space is astronomical; bsuite chains and
# small-discrete envs produce non-trivial distributions).
#
# A separate "action-margin" measurable wasn't authored: the
# obvious 1-D-trace-derivable proxy `mean(|q_mean − q_max|)` is
# already the existing `v_vs_max_delta_late` — same formula. A
# *true* Q* − Q_2nd action-margin requires adding a per-step
# second-max reduction to the collect harness; deferred until a
# substrate change is forced.


def _state_distribution_late_half(
    record: Mapping[str, object],
) -> npt.NDArray[np.int64] | None:
    """Slice `state_hash` to the late 50% and return as int64.
    None when the field is absent or the slice is empty."""
    if 'state_hash' not in record:
        return None
    arr = np.asarray(record['state_hash'], dtype=np.int64)
    n = arr.shape[0] if arr.ndim >= 1 else 0
    if n < 2:
        return None
    return arr[n // 2:]


@measurable(reads=('state_hash',))
def state_visit_entropy_late(record: Mapping[str, object]) -> float:
    """Shannon entropy (in nats) of the `state_hash` distribution
    over the late 50% of training. Higher entropy = more uniform
    visitation across states; lower = concentrated visitation.

    Returns NaN when `state_hash` is missing, the slice has < 2
    samples, or only one bucket was visited (entropy degenerate
    at zero — could return 0.0 but NaN is more honest about
    'this measurement carries no signal here').

    Useful for image-state envs (`state_hash` returns 0
    sentinel) — the measurable returns NaN there because every
    state hashes to the same bucket. For tabular bsuite chains
    and small-discrete envs the entropy is meaningful."""
    late = _state_distribution_late_half(record)
    if late is None:
        return float('nan')
    counts = np.bincount(late.astype(np.int64))
    nonzero = counts[counts > 0]
    if nonzero.size <= 1:
        return float('nan')
    p = nonzero.astype(np.float64) / float(nonzero.sum())
    return float(-np.sum(p * np.log(p)))


@measurable(reads=('state_hash',))
def state_coverage_kl_uniform_late(
    record: Mapping[str, object],
) -> float:
    """KL(observed-state-distribution || uniform-over-visited-
    buckets) over the late 50%. Zero when visitation is uniform
    across distinct visited states; positive otherwise. Larger =
    more concentrated visitation (the agent revisits some states
    far more than others among the buckets it touches at all).

    Reference distribution is uniform OVER VISITED buckets, not
    uniform over the full state space — that's the cell-by-cell
    'how concentrated is the visitation pattern' question, not
    'what fraction of the state space did the agent see'
    (which would need the global bucket cardinality, env-
    specific). Returns NaN under the same degenerate cases as
    `state_visit_entropy_late`."""
    late = _state_distribution_late_half(record)
    if late is None:
        return float('nan')
    counts = np.bincount(late.astype(np.int64))
    nonzero = counts[counts > 0]
    if nonzero.size <= 1:
        return float('nan')
    p = nonzero.astype(np.float64) / float(nonzero.sum())
    n = float(nonzero.size)
    # KL(p || uniform-over-visited) = sum p_i log(p_i * n)
    # = sum p_i log p_i + log n
    # = log n - H(p)   where H(p) = -sum p_i log p_i
    h = float(-np.sum(p * np.log(p)))
    return float(np.log(n) - h)


# ============ Post-run per-cell measurables (universal-ready) ============
#
# Phase A measurables: per-cell scalar / array derivations that
# bridges or analyses currently compute INLINE. Declaring them as
# named, registered fixtures lets the framework cache the result
# once per cell and serve to any consumer by name. The "universal"
# parquet (per the architectural principle) is then just (1)
# claim outputs + (2) a chosen subset of these cached measurables.
#
# Each must depend only on the per-cell record (runs.parquet row
# fields and/or persisted trace fields), not on cross-cell
# pairing — pairing is an analysis primitive, NOT a measurable.


@measurable(reads=('outcome.eval_best_burst_mean', 'reward_scale'))
def outcome_native(record: Mapping[str, object]) -> float:
    """Outcome divided by `reward_scale` — the agent's policy
    quality in units invariant under reward magnitude scaling.
    The optimal policy achieves the same `outcome_native`
    regardless of how reward is scaled (because optimal value
    scales linearly with reward; division cancels). Hedges' g
    standardizes this away too via pooled SD; bridges that test
    interventional effects ON reward magnitude must use this
    native unit, NOT standardized g.

    Returns the raw outcome unchanged when `reward_scale` is
    absent (legacy corpora that didn't record the column). NaN
    when reward_scale is exactly zero (defensive)."""
    outcome = record.get('outcome.eval_best_burst_mean')
    if not isinstance(outcome, (int, float)):
        return float('nan')
    rs = record.get('reward_scale', 1.0)
    if not isinstance(rs, (int, float)) or float(rs) == 0.0:
        return float('nan')
    return float(outcome) / float(rs)


@measurable(reads=('mc_return',))
def mc_variance_per_burst(
    record: Mapping[str, object],
) -> npt.NDArray[np.float64]:
    """Per-burst variance of `mc_return` across the K eval
    episodes within each burst. Shape `(n_bursts,)`.

    The within-burst spread of returns is a proxy for *epistemic
    uncertainty* in the agent's current policy — when the same
    policy produces variable episodic returns, the agent is
    sampling a high-variance distribution. Distinct from
    *between-env* mc_variance (which captures structural env
    noise). The bridge `mc_variance_attenuates_g_link__between_env`
    consumes this for the per-(env, burst) panel."""
    arr = np.asarray(record['mc_return'], dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        return np.zeros((0,), dtype=np.float64)
    return arr.var(axis=1)


@measurable(reads=('mc_return',))
def log_mc_variance_per_burst(
    record: Mapping[str, object],
    mc_variance_per_burst: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """`log(mc_variance_per_burst)`. Composes the variance
    measurable. Zero-variance bursts (converged-deterministic
    runs — e.g. Catch reaches optimum and every episode returns
    the same scalar) produce NaN, NOT a fudge floor. Downstream
    consumers that average across cells must filter NaN; the
    +1e-9 epsilon shortcut leaks deterministic-success bursts
    in as -20.7 outliers and contaminates the average."""
    arr = np.asarray(mc_variance_per_burst, dtype=np.float64)
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    mask = arr > 0
    out[mask] = np.log(arr[mask])
    return out


@measurable(reads=('mc_return',))
def mc_cv_per_burst(
    record: Mapping[str, object],
) -> npt.NDArray[np.float64]:
    """Per-burst coefficient of variation: `√var / |mean|`. Shape
    `(n_bursts,)`. NaN when |mean| is zero (degenerate, no
    well-defined CV) or when mc_return is mis-shaped.

    Variance scales with reward magnitude squared (k²) — it
    confounds with reward-scale interventions. CV is unitless,
    invariant under positive linear scaling of reward, and so
    captures *intrinsic relative noisiness* of returns rather
    than absolute spread. If `log_mc_variance` attenuates DDQN
    because it correlates with reward magnitude, CV should NOT
    reproduce the attenuation; if there's a true relative-noise
    moderator, CV will surface it."""
    arr = np.asarray(record['mc_return'], dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        return np.zeros((0,), dtype=np.float64)
    means = arr.mean(axis=1)
    stds = arr.std(axis=1)
    out = np.full(means.shape, np.nan, dtype=np.float64)
    mask = np.abs(means) > 0
    out[mask] = stds[mask] / np.abs(means[mask])
    return out


@measurable(reads=('mc_return',))
def log_mc_cv_per_burst(
    record: Mapping[str, object],
    mc_cv_per_burst: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """`log(mc_cv_per_burst)`. Zero-CV bursts (deterministic
    success or failure) produce NaN — same NaN-honest discipline
    as `log_mc_variance_per_burst`."""
    arr = np.asarray(mc_cv_per_burst, dtype=np.float64)
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    mask = arr > 0
    out[mask] = np.log(arr[mask])
    return out


# ============ Lifted from cell_runner._eval_outcomes (Phase 3A) ============
#
# Three eval-burst reductions persisted on every cell. Each is a
# `@measurable` reading `mc_return` (and `eval_step_index` for the
# step-provenance reduction). Substrates that want the standard set
# wire them via `dqn_default_measurables()` rather than naming each
# explicitly.
#
# Names retain the `outcome.` prefix so downstream consumers
# (paper_full_range.py §3-§7, dqn_bridges.py claim_bridges) keep
# reading the same column keys until Phase 5's bare-name pass.

@measurable(name='outcome.eval_final_mean', reads=('mc_return',))
def outcome_eval_final_mean(record: Mapping[str, object]) -> float:
    """`mean(mc_return[-1, :])`. The LAST eval burst's mean MC
    return — honest "final policy performance" (greedy, no
    exploration noise). Vulnerable to late-training instability.
    NaN when `mc_return` is missing or degenerate."""
    if 'mc_return' not in record:
        return float('nan')
    mc = np.asarray(record['mc_return'], dtype=np.float64)
    if mc.ndim != 2 or mc.size == 0:
        return float('nan')
    return float(mc[-1, :].mean())


@measurable(name='outcome.eval_best_burst_mean', reads=('mc_return',))
def outcome_eval_best_burst_mean(record: Mapping[str, object]) -> float:
    """`max_i(mean(mc_return[i, :]))`. Best-burst-seen during
    training. Robust to instability, slightly optimistic — the
    standard reduction for unstable-RL evaluation."""
    if 'mc_return' not in record:
        return float('nan')
    mc = np.asarray(record['mc_return'], dtype=np.float64)
    if mc.ndim != 2 or mc.size == 0:
        return float('nan')
    return float(mc.mean(axis=1).max())


@measurable(
    name='outcome.eval_best_burst_step',
    reads=('mc_return', 'eval_step_index'),
)
def outcome_eval_best_burst_step(record: Mapping[str, object]) -> float:
    """Provenance: training step at which the best burst occurred.
    Lets consumers see whether 'best' is at convergence or an
    early lucky checkpoint. Returns NaN when either input is
    missing or shapes are inconsistent (cell_runner persisted as
    `int` historically; lifting to `float` is honest about NaN-
    sentinel and matches the framework's `Measurable[R, float]`
    pre-registration shape)."""
    if 'mc_return' not in record or 'eval_step_index' not in record:
        return float('nan')
    mc = np.asarray(record['mc_return'], dtype=np.float64)
    eval_steps = np.asarray(record['eval_step_index'])
    if mc.ndim != 2 or mc.size == 0:
        return float('nan')
    if eval_steps.ndim != 1:
        return float('nan')
    best_idx = int(mc.mean(axis=1).argmax())
    if best_idx >= eval_steps.size:
        return float('nan')
    return float(eval_steps[best_idx])


# ============ Lifted from cell_runner._mechanism_measurements (Phase 3B) ============

@measurable(
    name='mechanism.jensen_gap',
    reads=('predicted_q_at_start', 'mc_return'),
)
def mechanism_jensen_gap(record: Mapping[str, object]) -> float:
    """`max(0, mean(predicted_q_at_start − mc_return))`. Hasselt
    2010/2016: vanilla DQN's positive Jensen-bias is what DDQN
    reduces by decoupling action selection (online) from value
    evaluation (target). Smaller is better. Clipped at 0 because
    only positive bias is the Jensen signature; negative
    (under-estimation) is a different phenomenon. NaN-honest:
    returns NaN when either input is missing.

    The full-named version of `jensen_overestimation_gap()` —
    same formula, registered under `mechanism.jensen_gap` so
    column names stay continuous through Phase 3 (Phase 5
    normalises the prefix)."""
    if (
        'predicted_q_at_start' not in record
        or 'mc_return' not in record
    ):
        return float('nan')
    predicted = np.asarray(record['predicted_q_at_start'], dtype=np.float64)
    actual = np.asarray(record['mc_return'], dtype=np.float64)
    if predicted.size == 0 or actual.size == 0:
        return float('nan')
    return float(max(0.0, (predicted - actual).mean()))


# ============ Substrate helper — default measurable set ============

def dqn_default_measurables() -> tuple[
    Measurable[Mapping[str, object], float], ...,
]:
    """The standard pre-registered measurable set every DDQN
    Hypothesis includes: three outcome reductions + Jensen-gap.
    Substrates that want the full set call this; ad-hoc
    hypotheses can construct a custom tuple instead.

    Author-side ergonomics: `Hypothesis(..., measurables=
    dqn_default_measurables())`. Each entry is a `Measurable[
    Mapping[str, object], float]` registered globally via
    `@measurable`, so corpus-side analyses that read the
    persisted columns by name (`outcome.eval_final_mean`,
    `mechanism.jensen_gap`) stay untouched."""
    return (
        outcome_eval_final_mean,
        outcome_eval_best_burst_mean,
        outcome_eval_best_burst_step,
        mechanism_jensen_gap,
    )
