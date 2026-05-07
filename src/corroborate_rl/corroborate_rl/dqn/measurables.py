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

import math
from collections.abc import Mapping

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from corroborate.graph.causal import Direction, Tier
from corroborate.bridge.bridge import Bridge
from corroborate.measurables import Measurable, measurable, register
from corroborate.measurables.reductions import (
    cv_safe,
    from_key,
    log_safe,
    mean_window,
    reduce_axis,
)
from corroborate_rl import env_catalogue


# ============ Pearson r — online vs target Q populations ============
#
# (The per-step Q-distribution measurables — `q_{mean,max,std,gap}_
# _per_step`, `target_q_{mean,max}_per_step`, `td_error_norm_per_
# step` — were removed: they read `online_q_per_action` /
# `target_q_per_action`, which `Q_TRACE_REDUCTIONS` in
# `trace_reductions.py` drops at trace persistence time. They could
# only resolve on synthetic test records, never on persisted
# corpora — pure scaffold.)

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
    leading axis. Returns NaN for empty or 0-d input (e.g. a null
    trace cell that `_record_array` decoded as a scalar ndarray).
    Inlined from `corroborate.reductions.mean_window` to keep the
    per-cell numpy hot path direct (no `Measurable` indirection)."""
    if arr.ndim == 0:
        return float('nan')
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


@measurable(reads=('episode_length', 'mc_return'))
def env_reward_polarity(record: Mapping[str, object]) -> float:
    """Per-cell Pearson r between `episode_length` and `mc_return`
    over all (burst, eval-episode) pairs. The endogenous proxy for
    env reward polarity:

    - r → +1: longer episodes ⇒ higher return ('survival' polarity).
      The env rewards being alive longer (per-step rewards
      accumulate; episode terminates on death). Examples:
      CartPole, Breakout, SpaceInvaders, Asterix.

    - r → −1: longer episodes ⇒ lower return ('goal' polarity).
      The env rewards reaching a target quickly (per-step penalty
      until terminal reward). Examples: Acrobot, FourRooms,
      MountainCar, DiscountingChain.

    - r ≈ 0: episode length and return decoupled. Either fixed-
      length episodes (Freeway), bandits (1-step), or saturated
      outcomes (Catch).

    Empirical (vanilla baseline cells, cross-env): ρ_pool of this
    measurable matches the hand-coded categorical polarity at
    Spearman = +0.88 (n_envs=6, p=0.02), with 6/6 sign-match.

    Used as the moderator for the eff_h-mediator bridges that
    would otherwise need a hand-coded env catalogue (cf. the env-
    polarity formal proof in `findings_polarity_mediator.md`):
    GOAL envs (polarity < 0) have negative slope coupling between
    Δ_eff_h and Δ_outcome; SURVIVAL envs (polarity > 0) have
    positive slope coupling. The pool ρ values were −0.798 and
    +0.240 respectively (formal proof n_envs=8, binomial p=0.004).
    """
    el = _record_array(record, 'episode_length')
    mc = _record_array(record, 'mc_return')
    if el is None or mc is None:
        return float('nan')
    el_arr = np.asarray(el, dtype=np.float64).flatten()
    mc_arr = np.asarray(mc, dtype=np.float64).flatten()
    if el_arr.shape != mc_arr.shape or el_arr.size < 3:
        return float('nan')
    if el_arr.std() == 0 or mc_arr.std() == 0:
        return float('nan')
    r = float(np.corrcoef(el_arr, mc_arr)[0, 1])
    return r if math.isfinite(r) else float('nan')


@measurable(reads=('predicted_q_at_start', 'mc_return'))
def q_mc_calibration_pearson(record: Mapping[str, object]) -> float:
    """Per-cell Pearson r between `predicted_q_at_start` and
    `mc_return` over all (burst, eval-episode) pairs (typically
    20 bursts × 5 episodes = 100 points).

    Measures Q's *predictive validity for its own policy's
    return*. r→1 means Q correctly ranks initial states by
    expected MC; r→0 means Q is uninformative; r<0 means Q is
    inversely calibrated (rare).

    On Breakout-MinAtar 1M, the within-seed Δ (DDQN − vanilla)
    of this measurable is the strongest known mediator of
    Δ_outcome: stratified partial Spearman ρ(Δ_calibration,
    Δ_mc_late | Δ_q_b19) = +0.701, p ≈ 0 (n=120 pooled across
    4 sync values). It survives controlling for late-Q
    amplification, and Q-amplification's coefficient attenuates
    when calibration is added as control. Combined, the two
    mediators absorb the cross-sync log_sync→outcome effect
    (residual ρ = −0.09 ns).

    Distinguishable from `pearson_r_online_target` (which is
    online-Q vs target-Q population correlation, a target-
    staleness diagnostic): this one is Q vs realized return —
    the OPE-style validity of the value function."""
    qs = _record_array(record, 'predicted_q_at_start')
    mc = _record_array(record, 'mc_return')
    if qs is None or mc is None:
        return float('nan')
    qs_flat = np.asarray(qs).flatten()
    mc_flat = np.asarray(mc).flatten()
    if qs_flat.size != mc_flat.size or qs_flat.size < 3:
        return float('nan')
    if qs_flat.std() == 0 or mc_flat.std() == 0:
        return float('nan')
    r = float(np.corrcoef(qs_flat, mc_flat)[0, 1])
    return r if math.isfinite(r) else float('nan')


@measurable(reads=('online_max_q_per_step', 'target_max_q_per_step'))
def target_staleness_late(record: Mapping[str, object]) -> float:
    """Mean over the late 50% of training of the *relative* gap
    `|online_max_q − target_max_q| / max(|online|, |target|, 1e-6)`.

    The endogenous delegate of `sync_period`. Sync_period is the
    HP knob; target staleness is the measurable mediator that
    sits on the causal path: target staleness IS what changes
    when sync_period varies, and DDQN's bootstrap rule (which
    uses Q_target at argmax_online) diverges from vanilla's
    (max_a Q_target) in proportion to it.

    Relative normalisation rather than absolute: at low sync the
    Q-explosion regime gives huge absolute gaps (hundreds-of-
    thousands) that swamp the late-burst signal; the relative
    form is monotone with sync (~0.002 at sync=100 to ~0.027 at
    sync=10000 on Breakout-MinAtar at burst 19, 1M steps).

    Late-50% window matches the canonical late-quarter Hasselt
    convention while keeping enough samples for a stable mean.
    """
    omax = _record_array(record, 'online_max_q_per_step')
    tmax = _record_array(record, 'target_max_q_per_step')
    if omax is None or tmax is None:
        return float('nan')
    abs_gap = np.abs(omax - tmax)
    denom = np.maximum(np.maximum(np.abs(omax), np.abs(tmax)), 1e-6)
    return _mean_window(abs_gap / denom, 0.5, 1.0)


@measurable(reads=('online_max_q_per_step', 'target_max_q_per_step'))
def target_staleness_early(record: Mapping[str, object]) -> float:
    """Same as `target_staleness_late` but over the EARLY 25% of
    training. Captures the burn-in regime where sync_period's
    grip on online-target divergence is largest (cf.
    `findings_target_staleness_collinear.md`: relative gap at
    burst 0 ranges from 1.9% (sync=100) to 41.6% (sync=10000) on
    Breakout). Useful as the substrate-level mediator for
    early-policy-quality bridges where the late window has
    already stabilised."""
    omax = _record_array(record, 'online_max_q_per_step')
    tmax = _record_array(record, 'target_max_q_per_step')
    if omax is None or tmax is None:
        return float('nan')
    abs_gap = np.abs(omax - tmax)
    denom = np.maximum(np.maximum(np.abs(omax), np.abs(tmax)), 1e-6)
    return _mean_window(abs_gap / denom, 0.0, 0.25)


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


@measurable(reads=('eval_best_burst_mean', 'reward_scale'))
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
    outcome = record.get('eval_best_burst_mean')
    if not isinstance(outcome, (int, float)):
        return float('nan')
    rs = record.get('reward_scale', 1.0)
    if not isinstance(rs, (int, float)) or float(rs) == 0.0:
        return float('nan')
    return float(outcome) / float(rs)


# mc_return reductions: 2-D shape (n_bursts, n_episodes). The
# per-cell scalars below collapse the inner (episode) axis with
# `reduce_axis(..., op='mean')` then window the leading (burst)
# axis with `mean_window(..., lo, hi)`. Composed factories carry
# `reads=('mc_return',)` automatically; cells without the column
# fail at `from_key` and the cache builder NaN-stores them. Each
# is rebound under a stable substrate name so existing bridges
# referencing 'mc_return_first_quarter' etc. resolve unchanged.

mc_return_first_quarter = Measurable(
    fn=mean_window(
        reduce_axis(from_key('mc_return'), axis=-1, op='mean'),
        0.0, 0.25,
    ).fn,
    name='mc_return_first_quarter',
    reads=('mc_return',),
)
register(mc_return_first_quarter)

mc_return_last_quarter = Measurable(
    fn=mean_window(
        reduce_axis(from_key('mc_return'), axis=-1, op='mean'),
        0.75, 1.0,
    ).fn,
    name='mc_return_last_quarter',
    reads=('mc_return',),
)
register(mc_return_last_quarter)


@measurable(reads=('gamma',))
def effective_horizon(
    record: Mapping[str, object],
    bootstrap_fraction: float,  # injected via @measurable name resolution
) -> float:
    """Chain-depth amplifier `1 / (1 − γ · bf)` — the geometric
    expected discount over the bootstrap chain, where `bf =
    1 - mean(done)` is the realised per-step non-termination
    probability.

    Derivation: under per-step termination probability
    `p_term = 1 − bf`, the effective per-step discount is
    `γ · (1 - p_term) = γ · bf`; the geometric series of these
    sums to `1 / (1 − γ · bf)`. Any per-step quantity (a DDQN
    bias correction, a reward) integrates to that times the
    amplifier along the bootstrap chain.

    Reads `gamma` (HP) directly; `bootstrap_fraction` injected
    by name from the measurable registry, which closes over
    `done` (trajectory). The closure therefore touches realised
    cell dynamics — eff_h IS truly endogenous per the topological
    check (cf. ENDOGENEITY_TOPOLOGY.md), unlike the prior
    γ-only form which was a pure HP transform.

    Caveat: γ is still an HP knob. `eff_h = γ · bf`-driven means
    cell-trajectory dependence enters via `bf`; the formula is
    half-synthetic in the sense that γ is author-chosen. A fully
    trajectory-derived amplifier would estimate the discount
    factor from observed returns; not currently needed.

    NaN propagates from invalid `gamma` (≥1, <0, missing) or
    from `bootstrap_fraction` itself (no `done` data). Denominator
    ≤ 0 is impossible for valid γ < 1 and bf ∈ [0, 1] but
    handled defensively."""
    gamma = record.get('gamma')
    if not isinstance(gamma, (int, float)):
        return float('nan')
    g = float(gamma)
    if math.isnan(g) or g >= 1.0 or g < 0.0:
        return float('nan')
    if math.isnan(bootstrap_fraction):
        return float('nan')
    denom = 1.0 - g * bootstrap_fraction
    if denom <= 0.0:
        return float('nan')
    return 1.0 / denom


# Per-burst variance / CV / log: variance over the inner
# (episode) axis, then optionally elementwise log. `cv_safe`
# composes mean+std internally; `log_safe` is NaN-on-non-positive.
# Each rebound under a stable name for back-compat with bridges
# that reference these by string source/target.

mc_variance_per_burst = Measurable(
    fn=reduce_axis(from_key('mc_return'), axis=-1, op='var').fn,
    name='mc_variance_per_burst',
    reads=('mc_return',),
)
register(mc_variance_per_burst)

log_mc_variance_per_burst = Measurable(
    fn=log_safe(
        reduce_axis(from_key('mc_return'), axis=-1, op='var'),
    ).fn,
    name='log_mc_variance_per_burst',
    reads=('mc_return',),
)
register(log_mc_variance_per_burst)

mc_cv_per_burst = Measurable(
    fn=cv_safe(from_key('mc_return'), axis=-1).fn,
    name='mc_cv_per_burst',
    reads=('mc_return',),
)
register(mc_cv_per_burst)

log_mc_cv_per_burst = Measurable(
    fn=log_safe(cv_safe(from_key('mc_return'), axis=-1)).fn,
    name='log_mc_cv_per_burst',
    reads=('mc_return',),
)
register(log_mc_cv_per_burst)


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

@measurable(name='late_window_mean', reads=('ep_return', 'done'))
def late_window_mean(record: Mapping[str, object]) -> float:
    """Late-window 10% mean of episode returns, restricted to
    terminal steps (`done > 0.5`). The codebase's standard outcome
    reduction over training trajectories: averages the last 10%
    of episode-end returns. Returns NaN if no terminal step
    survives the late window (e.g. an env that never terminates
    in the budget).

    Same formula as `masked_window_mean('ep_return', 'done', 0.1)`
    in `reductions.py`; lifted here as a registered substrate
    measurable so cell_runner persists it through the same
    channel as the rest of `dqn_default_measurables()`."""
    values = record.get('ep_return')
    done = record.get('done')
    if not isinstance(values, np.ndarray) or not isinstance(done, np.ndarray):
        # JAX arrays satisfy the same axis / shape protocol — np.asarray
        # at the boundary handles the cross-backend case.
        try:
            values = np.asarray(values)
            done = np.asarray(done)
        except (TypeError, ValueError):
            return float('nan')
    fraction = 0.1
    n = int(values.shape[0])
    cutoff = int((1.0 - fraction) * n)
    time_mask = np.arange(n) >= cutoff
    keep_mask = time_mask & (np.asarray(done) > 0.5)
    n_kept = int(np.sum(keep_mask))
    if n_kept == 0:
        return float('nan')
    masked = np.where(keep_mask, values, 0.0)
    return float(np.sum(masked) / n_kept)


@measurable(name='eval_final_mean', reads=('mc_return',))
def eval_final_mean(record: Mapping[str, object]) -> float:
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


@measurable(name='eval_best_burst_mean', reads=('mc_return',))
def eval_best_burst_mean(record: Mapping[str, object]) -> float:
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
    name='eval_best_burst_step',
    reads=('mc_return', 'eval_step_index'),
)
def eval_best_burst_step(record: Mapping[str, object]) -> float:
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

def _jensen_bias_per_eps_fn(
    record: Mapping[str, object],
) -> npt.NDArray[np.floating]:
    """Per-(burst, episode) Jensen-bias proxy: Q − MC. Positive →
    Q overestimates; the per-episode quantity *before* any reduction.

    Shape `(n_bursts, n_episodes)` — identical to the raw inputs.
    Per-burst analyses compose this with `reduce_axis(_, axis=-1,
    op='mean')` to get the per-burst gap; trajectory-level analyses
    compose with two reductions to get the scalar `jensen_gap`
    (clipped at 0 in `jensen_gap` for the structural-floor
    convention).

    Linearity of mean: `mean(Q − MC) = mean(Q) − mean(MC)`, so
    declaring the per-eps quantity once and reducing later
    produces the same result as reducing first and subtracting.
    Composing first is the cleaner shape — one reduction wraps
    the whole thing, and non-linear reductions (variance) get the
    *paired-difference* answer rather than the (potentially
    negative) `var(Q) − var(MC)` artefact."""
    q = np.asarray(record['predicted_q_at_start'], dtype=np.float64)
    mc = np.asarray(record['mc_return'], dtype=np.float64)
    return q - mc


# Explicit Measurable construction (rather than @measurable
# decorator) so the generic record-type parameter `R` lands as
# `Mapping[str, object]` for downstream reduction composition. The
# decorator's PEP 695 inference leaves R unbound when there's no
# call-site context, which makes `reduce_axis(jensen_bias_per_eps,
# ...)` partially-unknown at downstream consumers (e.g. bridges
# composing the per-burst-mean form). Explicit construction +
# `register(_)` is the same shape `mc_variance_per_burst` and the
# other NDArray-returning measurables in this module use.
jensen_bias_per_eps: Measurable[
    Mapping[str, object], npt.NDArray[np.floating],
] = Measurable(
    fn=_jensen_bias_per_eps_fn,
    name='jensen_bias_per_eps',
    reads=('predicted_q_at_start', 'mc_return'),
)
register(jensen_bias_per_eps)


# ============ Canonical per-burst-mean reductions (substrate-shared) ============
# Both `experiments.findings.ddqn_universe` and
# `experiments.findings.dqn_bridges` need per-burst-mean reductions over
# `mc_return` (link-side outcome projection) and `jensen_bias_per_eps`
# (mech-side bias projection). When each module composed these inline
# via `reduce_axis(...)`, the registry tripped on the second module's
# import — same name (`mc_return__mean_axis_-1`), distinct fresh
# `Measurable` instances. Authoring once here, importing in both
# findings modules, gives one canonical Measurable instance per name —
# no registration collision possible.
from corroborate.measurables.reductions import from_key, reduce_axis as _reduce_axis  # noqa: E402

mc_return_per_burst_mean: Measurable[
    Mapping[str, object], npt.NDArray[np.floating],
] = _reduce_axis(from_key('mc_return'), axis=-1, op='mean')

jensen_bias_per_burst_mean: Measurable[
    Mapping[str, object], npt.NDArray[np.floating],
] = _reduce_axis(jensen_bias_per_eps, axis=-1, op='mean')


@measurable(
    name='jensen_gap',
    reads=('predicted_q_at_start', 'mc_return'),
)
def jensen_gap(record: Mapping[str, object]) -> float:
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


# ============ Lifted from rl/dqn/invariants.py (Phase 4A) ============
#
# Theorem-gap measurables that previously lived as factory-returned
# `Measurable[..., float]` instances attached via `attach_invariant`
# to specific Claims. Phase 4 of the Bridge-collapse refactor moves
# these into the registry-based measurable channel — substrate
# enumerates them on `Hypothesis.measurables` and cell_runner
# persists each as a named scalar column.


@measurable(
    name='jensen_dormancy_gap',
    reads=('predicted_q_at_start', 'mc_return',
           'online_std_q_per_step', 'env_name'),
)
def jensen_dormancy_gap_measurable(record: Mapping[str, object]) -> float:
    """`max(0, structural_floor − observed_bias)` — gap between
    the Jensen-alone structural floor (`σ_late × √(2 log |A|)`)
    and the empirical overestimation (`mean(predicted_q_at_start −
    mc_return)`).

    Convention: gap = 0 ⇒ Jensen-premise active (DDQN's correction
    has something to bite on). gap > 0 ⇒ Jensen-premise dormant
    (observed bias is below what Jensen-alone predicts at this
    |A| and σ_Q; mechanism is structurally weak).

    Reads the persisted per-step σ_Q reduction
    (`online_std_q_per_step`) rather than raw
    `online_q_per_action` — the latter is dropped by
    `Q_TRACE_REDUCTIONS` at trace persistence time, so the
    measurable would always NaN post-hoc on persisted corpora.
    `n_actions` comes from the env catalogue keyed by
    `record['env_name']`.

    Returns NaN when inputs are missing or shapes are degenerate.
    The full registered version of `rl/dqn/invariants.py:
    jensen_dormancy_gap()` — Phase 4 lifts the invariant to the
    measurable channel."""
    predicted_v = record.get('predicted_q_at_start')
    actual_v = record.get('mc_return')
    sigma_v = record.get('online_std_q_per_step')
    env = record.get('env_name')
    if predicted_v is None or actual_v is None or sigma_v is None:
        return float('nan')
    if not isinstance(env, str):
        return float('nan')
    predicted = np.asarray(predicted_v, dtype=np.float64)
    actual = np.asarray(actual_v, dtype=np.float64)
    sigma_per_step = np.asarray(sigma_v, dtype=np.float64)
    if predicted.size == 0 or actual.size == 0 or sigma_per_step.size < 2:
        return float('nan')
    try:
        spec = env_catalogue.get(env)
    except KeyError:
        return float('nan')
    n_actions = int(spec.n_actions)
    if n_actions < 2:
        return float('nan')
    observed_bias = float(max(0.0, (predicted - actual).mean()))
    late_lo = int(sigma_per_step.shape[0]) // 2
    sigma = float(sigma_per_step[late_lo:].mean())
    if not (sigma == sigma and abs(sigma) < float('inf')):
        return float('nan')
    floor = sigma * math.sqrt(2.0 * math.log(n_actions))
    return float(max(0.0, floor - observed_bias))


# Hasselt-2010 Jensen premise as a self-loop INVARIANT bridge.
# The bridge is the typed, declarative source of truth: the
# threshold `0.0`, the predicate `AT_MOST`, the source measurable
# `jensen_dormancy_gap`, and the bridge's name (which becomes the
# emitted column name) all sit on Bridge fields rather than
# encoded in a measurable function body. The framework synthesizes
# the per-cell verdict measurable from the bridge — the substrate
# registers the synthesized one; the bridge's metadata is what
# carries the predicate.
jensen_dormancy_premise_active_bridge: Bridge = Bridge(
    name='jensen_dormancy_premise_active',
    source='jensen_dormancy_gap',
    target='jensen_dormancy_gap',
    direction=Direction.AT_MOST,
    tier=Tier.INVARIANT,
    threshold=0.0,
)
jensen_dormancy_premise_active: Measurable[Mapping[str, object], object] = (
    jensen_dormancy_premise_active_bridge.to_invariant_measurable()
)
register(jensen_dormancy_premise_active)


# ============ Substrate helper — default measurable set ============

def dqn_default_measurables() -> tuple[
    Measurable[Mapping[str, object], object], ...,
]:
    """The standard pre-registered measurable set every DDQN
    Hypothesis includes: three outcome reductions + Jensen-gap +
    Jensen-dormancy invariant verdict. Substrates that want the
    full set call this; ad-hoc hypotheses can construct a custom
    tuple instead.

    Author-side ergonomics: `Hypothesis(..., measurables=
    dqn_default_measurables())`. Each entry is a `Measurable[
    Mapping[str, object], MeasurementLeaf]` registered globally
    via `@measurable`, so corpus-side analyses read the persisted
    columns by their plain measurable names (`eval_final_mean`,
    `jensen_gap`, `jensen_dormancy_premise_active`).

    Returned as `Measurable[..., object]` to admit both float and
    string return types — the dormancy-premise verdict is
    categorical (`'held'` / `'invariant_violation'` /
    `'power_insufficient'`), the others are scalar floats.
    `Measurable.T` is covariant (per the regular-class +
    `@property` form), so `Measurable[..., float]` and
    `Measurable[..., str]` both lift cleanly to the
    `Measurable[..., object]` upper bound at the tuple boundary —
    no `cast` needed."""
    return (
        late_window_mean,
        eval_final_mean,
        eval_best_burst_mean,
        eval_best_burst_step,
        jensen_gap,
        jensen_dormancy_gap_measurable,
        jensen_dormancy_premise_active,
        # CLAIM 13 mediator (cf. `findings_target_staleness_mediator.md`):
        # target_staleness_{late,early} carries the dominant share of
        # DDQN's outcome benefit (3/4 envs HELD: FourRooms 0.27,
        # MountainCar 0.66, Breakout sync=100 0.65). Pre-registered
        # so future sweeps persist these scalars at cell-write time
        # — avoids the "trace-dependent measurables need backfill"
        # rebuild gap (cf. `feedback_cache_trace_dependent_measurables`).
        target_staleness_late,
        target_staleness_early,
        # CLAIM 15 substrate variable: log10 of the Polyak τ target-
        # update rate. NaN when target_sync.tau isn't set (most
        # periodic_copy sweeps). Consumed by
        # `paired_continuous_do_dowhy` in the staleness-amplification
        # bridges (`staleness_amplifies_ddqn_outcome__*_polyak`).
        log_tau,
    )


# ============ Env-structural measurables ============
#
# Per-cell scalars derived from the env catalogue keyed by
# `record['env_name']`. These replace inline static dicts in
# bridges (`_DDQN_200K_BOOTSTRAP_FRACTION`, etc.) — bridges now
# reference these by name in their `covariates` param, and the
# cache builder materialises them per cell across whatever corpus
# is in scope. NaN when env_name isn't a string or isn't
# registered in the catalogue.


def _env_spec_for(record: Mapping[str, object]) -> object:
    """Resolve `env_catalogue.get(record['env_name'])` defensively
    — returns the EnvSpec or None on missing/unknown env. Typed
    `object` because env_catalogue.EnvSpec isn't imported at
    module top-level (avoiding a circular dependency for code
    that loads measurables before env_catalogue's gymnax-side
    initialisation completes)."""
    name = record.get('env_name')
    if not isinstance(name, str):
        return None
    try:
        return env_catalogue.get(name)
    except KeyError:
        return None


@measurable(reads=('target_sync.tau',))
def log_tau(record: Mapping[str, object]) -> float:
    """`log10(target_sync.tau)` — the Polyak target-update rate on
    log scale. NaN-propagates when `target_sync.tau` is missing (HP
    not set, e.g. periodic_copy regimes) or non-positive.

    Used as the continuous treatment variable in the polyak-τ
    intervention bridges (`paired_continuous_do_dowhy` consumes
    cells with this column). Log-space matches the geometric
    mixing semantics: τ → 0 is "very stale" (1/τ-step memory),
    τ → 1 is "no staleness". The 3-log-decade sweep
    [0.001, 0.01, 0.1, 1.0] gives log_tau ∈ [-3, -2, -1, 0]."""
    tau = record.get('target_sync.tau')
    if not isinstance(tau, (int, float)):
        return float('nan')
    if tau <= 0:
        return float('nan')
    return math.log10(float(tau))


@measurable(reads=('env_name',))
def log_action_dim(record: Mapping[str, object]) -> float:
    """`log(max(n_actions, 2))` — the discrete-action dimensionality
    on log scale. Matches Hasselt's overestimation-bias floor
    `√(2 log|A|)` only logarithmically, so log_action_dim is the
    natural covariate for cross-env meta-regressions of bias."""
    spec = _env_spec_for(record)
    if not isinstance(spec, env_catalogue.EnvSpec):
        return float('nan')
    return math.log(max(int(spec.n_actions), 2))


@measurable(reads=('env_name',))
def log_obs_dim(record: Mapping[str, object]) -> float:
    """`log(max(obs_dim, 1))` — total flattened observation
    dimensionality on log scale. Useful as a structural covariate
    for cross-env regressions; image envs (MinAtar) have
    `obs_dim ≈ 10⁴`, vector envs ≈ 4-25."""
    spec = _env_spec_for(record)
    if not isinstance(spec, env_catalogue.EnvSpec):
        return float('nan')
    return math.log(max(int(spec.obs_dim), 1))


@measurable(reads=('env_name',))
def log_horizon(record: Mapping[str, object]) -> float:
    """`log(max(horizon, 1))` — episode-length cap on log scale.
    Falls back to 1000 (gymnax's default cap) when the env's
    `horizon` is None."""
    spec = _env_spec_for(record)
    if not isinstance(spec, env_catalogue.EnvSpec):
        return float('nan')
    h = spec.horizon if spec.horizon is not None else 1000
    return math.log(max(int(h), 1))


@measurable(reads=('env_name',))
def r_max(record: Mapping[str, object]) -> float:
    """Per-step reward upper bound from the env catalogue. Used
    by `q_divergence_score` to compute the Bellman fixed-point
    Q-bound `r_max / (1 - γ)`."""
    spec = _env_spec_for(record)
    if not isinstance(spec, env_catalogue.EnvSpec):
        return float('nan')
    return float(spec.r_max)


@measurable(reads=('env_name',))
def r_min(record: Mapping[str, object]) -> float:
    """Per-step reward lower bound. Sibling of `r_max`; used in
    contexts that need the symmetric reward span (e.g. signed
    Bellman bounds for envs with negative rewards)."""
    spec = _env_spec_for(record)
    if not isinstance(spec, env_catalogue.EnvSpec):
        return float('nan')
    return float(spec.r_min)


# ============ Episode dynamics measurables ============


@measurable(reads=('done',))
def bootstrap_fraction(record: Mapping[str, object]) -> float:
    """Fraction of update steps that bootstrap (i.e. don't
    terminate). `1 - mean(done)` over the per-step trajectory.

    A bootstrap-fraction of 1.0 means the agent never reaches a
    terminal state during training (long-horizon envs like
    Acrobot at high γ); 0.0 means every step terminates (bandit
    envs where the agent never bootstraps from its own Q
    estimate). The covariate predicts DDQN-link strength: bias
    compounds along bootstrapped chains, so envs in the high-
    bootstrap regime show stronger Hasselt-mechanism → outcome
    translation.

    NaN when `done` is missing or empty; the framework's typed
    None handling in the cache builder propagates this without
    column-erasure."""
    arr = record.get('done')
    if arr is None:
        return float('nan')
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0:
        return float('nan')
    return float(1.0 - a.mean())


# ============ Bellman-bound measurable ============


@measurable(reads=('jensen_gap', 'gamma'))
def q_divergence_score(
    record: Mapping[str, object],
    r_max: float,  # injected via @measurable name resolution
    r_min: float,  # injected via @measurable name resolution
) -> float:
    """`jensen_gap / (R / (1 - gamma))` — the overestimation-bias
    gap normalised by the L∞ Bellman fixed-point Q-bound,
    `R = max(|r_min|, |r_max|)`. Per-cell scalar.

    Reading: scores below ~1 mean Q stays within the theoretical
    bound and DDQN's correction translates to outcome; scores
    above ~1000 mean Q has diverged orders of magnitude beyond
    the bound and DDQN's link to outcome attenuates (CLAIM 11
    in `findings_minatar_link_attenuation.md`).

    Uses `max(|r_min|, |r_max|)` rather than just `r_max` so the
    bound is well-defined on envs with non-positive `r_max`
    (Acrobot, MountainCar — `r_min=-1, r_max=0` → bound=1/(1-γ)).
    Reduces to the textbook `r_max / (1-γ)` whenever `r_max ≥
    |r_min|`, including all positive-reward envs (CartPole,
    Catch-bsuite, MinAtar). NaN on degenerate inputs (gamma ≥ 1,
    missing fields, both r_max and r_min ≤ 0 in absolute value).

    Composes the env-driven `r_max` + `r_min` measurables with
    the cell's runs.parquet `jensen_gap` and `gamma` columns —
    transitive_reads(`q_divergence_score`) closes over
    `{jensen_gap, gamma, env_name}`."""
    jens = record.get('jensen_gap')
    gamma = record.get('gamma')
    if not isinstance(jens, (int, float)):
        return float('nan')
    if not isinstance(gamma, (int, float)):
        return float('nan')
    g = float(gamma)
    if g >= 1.0 or g < 0.0:
        return float('nan')
    if math.isnan(r_max) or math.isnan(r_min):
        return float('nan')
    r_abs = max(abs(r_max), abs(r_min))
    if r_abs <= 0.0:
        return float('nan')
    bound = r_abs / (1.0 - g)
    return float(jens) / bound
