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
from corroborate.measurables import (
    Measurable,
    measurable,
    register,
    register_as,
)
from corroborate.measurables.reductions import (
    cv_safe,
    from_key,
    log_safe,
    mean_window,
    reduce_axis,
)
from corroborate_rl import env_catalogue


# ============ Module-top shared sources ============
#
# Each trace column the substrate reads is declared once here so
# every consumer shares the SAME `from_key` instance via Python
# identity. Within this module, `compose_of` trees of downstream
# Measurables (e.g., `mean_window(ONLINE_MAX_Q, ...)`) root at
# the shared instance — `signature()` recursion then catches
# semantic changes to the source uniformly across all consumers.

ONLINE_MAX_Q = from_key('online_max_q_per_step')
ONLINE_MIN_Q = from_key('online_min_q_per_step')
ONLINE_MEAN_Q = from_key('online_mean_q_per_step')
ONLINE_STD_Q = from_key('online_std_q_per_step')
ONLINE_ARGMAX = from_key('online_argmax_per_step')
ONLINE_TOP12_MARGIN = from_key('online_top12_margin_per_step')
TARGET_MAX_Q = from_key('target_max_q_per_step')
TARGET_AT_ARGMAX = from_key('target_q_at_online_argmax_per_step')
TARGET_ARGMAX = from_key('target_argmax_per_step')
PREDICTED_Q_AT_START = from_key('predicted_q_at_start')
MC_RETURN = from_key('mc_return')
EP_LENGTH = from_key('episode_length')
EVAL_STEP_INDEX = from_key('eval_step_index')
BUF_SIZE = from_key('buf_size')
TD_ERROR = from_key('td_error')
TD_WB_STD = from_key('td_error_within_batch_std')
Q_ACTION_GRAD_OVERLAP = from_key('q_action_grad_overlap_per_step')
BOOTSTRAP_ACTION_MISMATCH = from_key('bootstrap_action_mismatch_per_step')
Q_INTER_STATE_GRAD_OVERLAP = from_key('q_inter_state_grad_overlap_per_step')


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


def _windowed_mean(
    arr: npt.NDArray[np.float64], lo: float, hi: float,
) -> float:
    """Substrate-private fallback for bodies whose final reduction
    is a windowed-mean of an *intermediate* ndarray (e.g., a per-
    cell `max_q - min_q` derived inline). Framework `mean_window`
    operates on `Measurable` operands; bodies producing derived
    arrays inline can't go through it, so this helper exists for
    the binary-arithmetic case. NaN on degenerate (0-d / empty)
    input."""
    if arr.ndim == 0:
        return float('nan')
    n = arr.shape[0]
    if n == 0:
        return float('nan')
    i_lo = int(lo * n)
    i_hi = max(int(hi * n), i_lo + 1)
    return float(np.mean(arr[i_lo:i_hi]))


@measurable(reads=('online_max_q_per_step', 'online_min_q_per_step'))
def q_gap_late(record: Mapping[str, object]) -> float:
    """Mean of (online_max_q − online_min_q) over the last 50% of
    training. Action-margin scalar — sharp policy preferences
    correlate with reward in Breakout / MNISTBandit /
    SpaceInvaders (PAPER §5.1)."""
    try:
        max_q = ONLINE_MAX_Q(record)
        min_q = ONLINE_MIN_Q(record)
    except KeyError:
        return float('nan')
    return _windowed_mean(max_q - min_q, 0.5, 1.0)


@measurable(reads=('online_max_q_per_step', 'online_min_q_per_step'))
def q_gap_growth(record: Mapping[str, object]) -> float:
    """(late_half_mean − early_half_mean) of the q_gap signal.
    Captures whether action-margin widens as training progresses
    (positive growth → emergent policy sharpness)."""
    try:
        max_q = ONLINE_MAX_Q(record)
        min_q = ONLINE_MIN_Q(record)
    except KeyError:
        return float('nan')
    gap = max_q - min_q
    early = _windowed_mean(gap, 0.0, 0.5)
    late = _windowed_mean(gap, 0.5, 1.0)
    return float(late - early)


@measurable(reads=('target_max_q_per_step', 'target_q_at_online_argmax_per_step'))
def ddqn_bootstrap_gap(record: Mapping[str, object]) -> float:
    """Mean of `target_max_q − target_q_at_online_argmax` across ALL
    training steps. The DDQN-correction magnitude per step,
    integrated over the full bootstrap-update process:

      vanilla bootstrap value  =  max_a Q_target(s', a)
      DDQN bootstrap value     =  Q_target(s', argmax_a Q_online(s', a))
      gap                      =  vanilla_value  −  DDQN_value  ≥ 0

    Full-trajectory mean (no windowing) — the clip-wedge per step
    integrated over all training. Polarity-invariant
    (non-negative by construction).

    This is the *algorithmic* clip-wedge: it fires whenever
    `argmax_online ≠ argmax_target`, regardless of whether the
    Hasselt bias premise is active. On dormant cells
    (jens=0), it still produces a non-zero gap whose effect on
    outcome is the Q-magnitude-regularization channel —
    independent of bias correction. Use this as the predictor
    for "DDQN helps via clip channel" bridges in dormant scope.

    Sign-aware interpretation by Q-regime (uniform signed Q-shift,
    polarity-blind statement):
    - The clip always pulls bootstrap-target Q DOWN (less
      positive, by construction `target_q[argmax_online] ≤
      max_a target_q`).
    - In envs with positive Q (r_max > 0): |Q| DECREASES.
    - In envs with negative Q (r_max ≤ 0): |Q| INCREASES (Q
      becomes more negative).
    The downstream effect on outcome may depend on a separate
    env-level property — `env_reward_polarity` (the framework's
    Pearson(length, return) measurable, REACH/SURVIVAL axis) —
    distinct from r_max sign. See CLAIM 3 in
    `experiments/findings/ddqn/` for the polarity-
    moderation test of the clip→outcome channel.

    Don't use Δ_q as the predictor for the clip channel (conflates
    with bias-reduction); use THIS quantity instead — it's the
    clip-wedge directly.

    Prefer this over the legacy `_late` variant (which used an
    arbitrary 50% cut-off); `_late` is kept for backward compat
    with bridges authored before the convention was reconsidered."""
    try:
        target_max = TARGET_MAX_Q(record)
        target_at_online = TARGET_AT_ARGMAX(record)
    except KeyError:
        return float('nan')
    n = min(target_max.shape[0], target_at_online.shape[0])
    if n == 0:
        return float('nan')
    gap = target_max[:n] - target_at_online[:n]
    return float(np.mean(gap))


@measurable(reads=(
    'target_max_q_per_step', 'target_q_at_online_argmax_per_step',
    'episode_length', 'mc_return',
))
def clip_wedge_polarity_aligned(record: Mapping[str, object]) -> float:
    """`ddqn_bootstrap_gap × sign(env_reward_polarity)`.

    Polarity-moderation test in a single predictor: if the
    clip-channel's effect on outcome changes sign with
    `env_reward_polarity` (the framework's REACH/SURVIVAL axis,
    Pearson r between episode_length and mc_return), this product
    aligns the sign for a unified positive-direction test.

    Empirical motivation (per `findings_clip_channel_polarity.md`):
    on dormant cells, per-env partial-Spearman ρ(clip_wedge,
    outcome | jens) was +0.33/+0.25 on Asterix/Breakout (SURVIVAL-
    polarity) and -0.51/-0.16 on Acrobot/FourRooms (REACH-polarity).
    Pooled it averaged to ~0 (washing out the polarity-conditional
    structure). The polarity-aligned predictor folds that
    interaction into a single quantity testable with the framework's
    existing `stratified_partial_spearman` primitive.

    Sign convention:
    - polarity > 0 (SURVIVAL): aligned = +clip_wedge (predicted
      positive r with outcome)
    - polarity < 0 (REACH): aligned = −clip_wedge (predicted
      positive r with outcome, since original r is negative)
    - polarity ≈ 0: aligned = 0 (no signal in either direction)

    Implementation: computes both clip-wedge and polarity inline
    from trace columns (the framework's measurable dependency
    system doesn't guarantee eval order for derived measurables).
    Cheap — both underlying quantities are already O(n_steps)."""
    # Clip wedge (full-trajectory mean of target_max - target_at_argmax)
    try:
        target_max = TARGET_MAX_Q(record)
        target_at_online = TARGET_AT_ARGMAX(record)
    except KeyError:
        return float('nan')
    n = min(target_max.shape[0], target_at_online.shape[0])
    if n == 0:
        return float('nan')
    clip = float(np.mean(target_max[:n] - target_at_online[:n]))
    # Polarity (Pearson r between episode_length and mc_return)
    ep_len = record.get('episode_length')
    mc_ret = record.get('mc_return')
    if ep_len is None or mc_ret is None:
        return float('nan')
    # Flatten nested per-burst lists. `episode_length` is
    # List[List[int]] (n_bursts × n_episodes_per_burst);
    # `mc_return` likewise.
    try:
        ep_flat = np.concatenate([np.asarray(b, dtype=np.float64) for b in ep_len])
        mc_flat = np.concatenate([np.asarray(b, dtype=np.float64) for b in mc_ret])
    except (TypeError, ValueError):
        return float('nan')
    if len(ep_flat) != len(mc_flat) or len(ep_flat) < 3:
        return float('nan')
    ep_var = float(np.var(ep_flat))
    mc_var = float(np.var(mc_flat))
    if ep_var <= 0 or mc_var <= 0:
        return 0.0  # constant series → no polarity signal
    pol = float(np.corrcoef(ep_flat, mc_flat)[0, 1])
    if math.isnan(pol):
        return float('nan')
    # Sign-align: +clip for survival, -clip for reach, 0 for null
    if pol > 0:
        return clip
    elif pol < 0:
        return -clip
    return 0.0


@measurable(reads=(
    'target_max_q_per_step', 'target_q_at_online_argmax_per_step',
))
def bootstrap_gap_magnitude(record: Mapping[str, object]) -> float:
    """`mean(target_max − target_q_at_online_argmax)` per training
    step — the per-step magnitude of DDQN's algorithmic correction.

    Twin to `ddqn_bootstrap_gap` (which is currently all-NaN in
    cache because cell_runner persists it before the trace join).
    This sibling lives outside `dqn_default_measurables()` so it's
    computed fresh at ingestion time when trace columns are
    available, dodging the cell_runner stringification path.

    **MC-free**: defined entirely in terms of network outputs
    (online + target Q). Use as the bias-magnitude predictor for
    falsification bridges that would otherwise hit the `jens = Q
    − MC` tautology — `corr(Δ_jens, Δ_mc)` is algebraically
    pinned even at the stratum level (partial-r given Δ_Q = 1.0
    empirically); `bootstrap_gap_magnitude` doesn't contain MC
    anywhere so its regression on Δ_outcome is non-tautological.

    Returns NaN when trace cols are absent."""
    try:
        target_max = TARGET_MAX_Q(record)
        target_at_online = TARGET_AT_ARGMAX(record)
    except KeyError:
        return float('nan')
    n = min(target_max.shape[0], target_at_online.shape[0])
    if n == 0:
        return float('nan')
    return float(np.mean(target_max[:n] - target_at_online[:n]))


@measurable(reads=(
    'target_max_q_per_step', 'target_q_at_online_argmax_per_step',
))
def bootstrap_gap_frac_active(record: Mapping[str, object]) -> float:
    """Fraction of training steps with bg > 0.001 — the
    "wedge-active rate." Twin to `bootstrap_gap_magnitude` (mean
    aggregation), but sensitive to the FREQUENCY of arm-
    disagreement rather than the integrated magnitude.

    The per-step `target_max − target_q_at_online_argmax`
    distribution is heavy-tailed and dominated by post-convergence
    zeros at canonical 1M. The mean reduction averages toward
    zero and hides arm differences. `frac_active` instead asks:
    on what fraction of training steps did online's argmax
    disagree with target's argmax (above a tiny tolerance)?

    Empirically arm-separable: FourRooms d_bg_frac001 = -0.021
    (t=-3.7), MountainCar d_bg_frac001 = -0.036 (t=-1.9), both
    DDQN < vanilla. The 0.001 threshold trims float-noise from
    bg ≈ 0 events (online/target literally agree on argmax →
    bg=0 exactly); 0.001 is well below the q99 tail (~0.013-0.25
    across canonical envs).

    **MC-free** — same MC-free property as `bootstrap_gap_magnitude`;
    no `jens = Q − MC` tautology."""
    try:
        target_max = TARGET_MAX_Q(record)
        target_at_online = TARGET_AT_ARGMAX(record)
    except KeyError:
        return float('nan')
    n = min(target_max.shape[0], target_at_online.shape[0])
    if n == 0:
        return float('nan')
    gap = target_max[:n] - target_at_online[:n]
    return float((gap > 0.001).mean())


@measurable(reads=(
    'target_max_q_per_step', 'target_q_at_online_argmax_per_step',
))
def bootstrap_gap_q99(record: Mapping[str, object]) -> float:
    """99th-percentile of per-step bootstrap-gap magnitude — the
    tail of online/target argmax disagreement.

    Companion to `bootstrap_gap_frac_active`. Where `frac_active`
    asks "how often do they disagree?", `q99` asks "when they
    disagree at all, how large is the wedge?" Together the two
    decompose what the mean reduction collapses.

    Useful on envs where DDQN preserves bigger wedges per
    disagreement event (e.g., FourRooms d_bg_q999=+0.006, t=+2.5
    — DDQN's online net stays consistent with its own argmax
    even when target disagrees, so the wedge stays open).

    **MC-free** — same MC-free property as the family."""
    try:
        target_max = TARGET_MAX_Q(record)
        target_at_online = TARGET_AT_ARGMAX(record)
    except KeyError:
        return float('nan')
    n = min(target_max.shape[0], target_at_online.shape[0])
    if n == 0:
        return float('nan')
    gap = target_max[:n] - target_at_online[:n]
    return float(np.quantile(gap, 0.99))


@measurable(reads=('target_max_q_per_step', 'target_q_at_online_argmax_per_step'))
def ddqn_bootstrap_gap_late(record: Mapping[str, object]) -> float:
    """**LEGACY** — mean clip-wedge over the late 50% of training
    only. Prefer `ddqn_bootstrap_gap` (full-trajectory) for new
    bridges; this `_late` variant exists for backward compat with
    bridges authored before the convention was reconsidered. The
    `_late` cut-off is arbitrary.

    Mean of `target_max_q − target_q_at_online_argmax` over the
    late 50% of training. The DDQN-correction magnitude per step:

      vanilla bootstrap value  =  max_a Q_target(s', a)
      DDQN bootstrap value     =  Q_target(s', argmax_a Q_online(s', a))
      gap                      =  vanilla_value  −  DDQN_value  ≥ 0

    The gap is non-negative by construction (max ≥ value at any
    specific argmax). Larger gap → DDQN deviates more from vanilla
    on this step → bigger correction.

    Sign-aware interpretation by Q-regime (cf. `q_late_mean`):
    - Q > 0 (sparse-positive): vanilla's max-bias inflates Q
      ABOVE truth. Gap is the inflated-Q amount. DDQN removes
      the inflation. Mechanism beneficial.
    - Q < 0 (dense-penalty): vanilla's max picks the LEAST
      negative Q. DDQN at online's argmax picks a MORE negative
      Q (online's argmax may not align with target's max in this
      regime). Gap is the "less optimism" amount. DDQN removes
      the optimism. Mechanism harmful (per the Q-regime story).

    Used to decompose the algorithmic mechanism behind the ATE
    sign-flip: bridge `staleness_amplifies_ddqn_outcome__sparse_
    goal_polyak` predicts gap × Q-sign correlates with
    Δ_outcome."""
    try:
        target_max = TARGET_MAX_Q(record)
        target_at_online = TARGET_AT_ARGMAX(record)
    except KeyError:
        return float('nan')
    n = min(target_max.shape[0], target_at_online.shape[0])
    if n == 0:
        return float('nan')
    gap = target_max[:n] - target_at_online[:n]
    return _windowed_mean(gap, 0.5, 1.0)


@measurable(reads=('online_argmax_per_step', 'state_hash'))
def state_conditional_argmax_entropy_late(
    record: Mapping[str, object],
) -> float:
    """State-conditional Shannon entropy of `online_argmax_per_step`
    over the late 50% of training.

    `H(argmax | state)` = E_s [ -Σ_a p(a|s) log p(a|s) ] where
    p(a|s) is the empirical fraction of late-training steps in
    state-bucket s where the online network's argmax was action a.

    Distinguishes (a) state-differentiated policy from (b) Q-flat
    indecisive policy that marginal `argmax_entropy_late` confounds:
      (a) state-differentiated: H(argmax | state) ≈ 0 (decisive
          per state, just different argmax per state region) →
          high mutual_info(state, argmax) → STATE-CONDITIONAL
          structure.
      (b) Q-flat: H(argmax | state) ≈ H(argmax) ≈ log(|A|) →
          mutual_info ≈ 0 → no state-conditional information.

    Returns NaN when state_hash is missing or all-zero (image envs
    pre-state-hash-registration), or fewer than 2 distinct
    (state, argmax) pairs are observed."""
    argmax_arr = record.get('online_argmax_per_step')
    state_arr = record.get('state_hash')
    if argmax_arr is None or state_arr is None:
        return float('nan')
    a = np.asarray(argmax_arr, dtype=np.int64).flatten()
    s = np.asarray(state_arr, dtype=np.int64).flatten()
    n = min(a.size, s.size)
    if n < 4:
        return float('nan')
    half = n // 2
    a_late, s_late = a[half:], s[half:]
    unique_s = np.unique(s_late)
    if unique_s.size < 2:
        return float('nan')
    total_h = 0.0
    total_w = 0
    for s_val in unique_s:
        mask = s_late == s_val
        n_s = int(mask.sum())
        if n_s < 2:
            continue
        a_in_s = a_late[mask]
        counts = np.bincount(a_in_s)
        nonzero = counts[counts > 0]
        if nonzero.size <= 1:
            h_s = 0.0
        else:
            p = nonzero.astype(np.float64) / float(nonzero.sum())
            h_s = float(-np.sum(p * np.log(p)))
        total_h += h_s * n_s
        total_w += n_s
    if total_w == 0:
        return float('nan')
    return total_h / total_w


@measurable(reads=('online_argmax_per_step', 'state_hash'))
def mutual_info_state_argmax_late(
    record: Mapping[str, object],
) -> float:
    """Mutual information I(state; argmax) over the late 50% of
    training. `I = H(argmax) − H(argmax | state)` — measures how
    much of the action-distribution variance is explained by
    state-bucket identity.

    High MI: DDQN's policy is STATE-DIFFERENTIATED (different
    state regions yield different argmaxes; marginal action
    diversity is structured by state).
    Low MI + high marginal entropy: Q-FLAT noise (same near-
    uniform action distribution regardless of state).
    Low MI + low marginal entropy: COLLAPSED policy (same action
    everywhere).

    The (a) vs (b) disambiguation primary measurable for the
    policy-structure-channel claim per memory
    `findings_ddqn_mediator_heterogeneity`.

    Returns NaN under same degenerate cases as
    `state_conditional_argmax_entropy_late`."""
    argmax_arr = record.get('online_argmax_per_step')
    state_arr = record.get('state_hash')
    if argmax_arr is None or state_arr is None:
        return float('nan')
    a = np.asarray(argmax_arr, dtype=np.int64).flatten()
    s = np.asarray(state_arr, dtype=np.int64).flatten()
    n = min(a.size, s.size)
    if n < 4:
        return float('nan')
    half = n // 2
    a_late, s_late = a[half:], s[half:]
    unique_s = np.unique(s_late)
    if unique_s.size < 2:
        return float('nan')
    # Marginal H(argmax)
    counts_a = np.bincount(a_late)
    nz_a = counts_a[counts_a > 0]
    if nz_a.size <= 1:
        return float('nan')
    p_a = nz_a.astype(np.float64) / float(nz_a.sum())
    h_a = float(-np.sum(p_a * np.log(p_a)))
    # Conditional H(argmax | state)
    total_h = 0.0
    total_w = 0
    for s_val in unique_s:
        mask = s_late == s_val
        n_s = int(mask.sum())
        if n_s < 2:
            continue
        a_in_s = a_late[mask]
        counts = np.bincount(a_in_s)
        nonzero = counts[counts > 0]
        if nonzero.size <= 1:
            h_s = 0.0
        else:
            p = nonzero.astype(np.float64) / float(nonzero.sum())
            h_s = float(-np.sum(p * np.log(p)))
        total_h += h_s * n_s
        total_w += n_s
    if total_w == 0:
        return float('nan')
    h_cond = total_h / total_w
    return h_a - h_cond


@measurable(reads=('online_argmax_per_step',))
def argmax_entropy_late(record: Mapping[str, object]) -> float:
    """Shannon entropy (nats) of `online_argmax_per_step`'s
    distribution over the late 50% of training.

    H = -Σ p_a log p_a where p_a is the empirical fraction of
    late-training steps where the online network's argmax was
    action a.

    Low entropy: policy decisively prefers a small set of
    actions (committed / converged on a strategy).
    High entropy: policy distributes argmax broadly across
    actions (uncertain / exploratory / Q-flat).

    Connection to DDQN mechanism: in regimes where vanilla's
    Q-values flatten across actions (rescue regime, sparse-
    reward + low rs), the argmax distribution becomes dispersed
    not because the policy is good but because the policy is
    indecisive. DDQN's bias correction can either:
      (a) Sharpen the policy further if it has signal to commit
          on (g_link via decisive-policy improvement), OR
      (b) Maintain higher entropy by avoiding spurious
          commitment to wrong actions when Q is flat (rescue
          regime: keeps exploring until reward signal arrives).

    Per-cell measurable; bridge body computes paired difference
    DDQN_entropy − vanilla_entropy. Sign of the difference
    distinguishes (a) vs (b) regimes.

    Returns nan if `online_argmax_per_step` is missing or empty."""
    arr = record.get('online_argmax_per_step')
    if arr is None:
        return float('nan')
    try:
        a = list(arr)
    except TypeError:
        return float('nan')
    if not a:
        return float('nan')
    import numpy as np_
    arr_np = np_.asarray(a, dtype=np_.int64)
    n = len(arr_np)
    if n < 2:
        return float('nan')
    late = arr_np[n // 2:]
    if len(late) == 0:
        return float('nan')
    counts = np_.bincount(late)
    p = counts / len(late)
    p_pos = p[p > 0]
    return float(-(p_pos * np_.log(p_pos)).sum())


@measurable(reads=('online_argmax_per_step',))
def argmax_mode_freq_late(record: Mapping[str, object]) -> float:
    """Fraction of late-training steps where `online_argmax_
    per_step` equals the mode action.

    Range [1/|A|, 1.0]. 1.0 = always picks the same action
    (fully committed). 1/|A| = uniform across actions
    (fully indecisive). Companion to `argmax_entropy_late`;
    different scaling, same underlying signal.

    Per-cell measurable. Bridge body computes paired
    DDQN_mode_freq − vanilla_mode_freq."""
    arr = record.get('online_argmax_per_step')
    if arr is None:
        return float('nan')
    try:
        a = list(arr)
    except TypeError:
        return float('nan')
    if not a:
        return float('nan')
    import numpy as np_
    arr_np = np_.asarray(a, dtype=np_.int64)
    n = len(arr_np)
    if n < 2:
        return float('nan')
    late = arr_np[n // 2:]
    if len(late) == 0:
        return float('nan')
    counts = np_.bincount(late)
    return float(counts.max()) / float(len(late))


@measurable(reads=('online_std_q_per_step',))
def q_action_std_late(record: Mapping[str, object]) -> float:
    """Mean of `online_std_q_per_step` over the late 50% of
    training — the cross-action Q-stdev at non-terminal states,
    averaged over late training.

    Hasselt 2010's overestimation theorem says the per-step
    max-bias is approximately `c · σ_action` where σ_action is
    the cross-action Q-stdev (noise level across action choices)
    and `c = √(2 ln(|A|)/π)`. So this is the operationally-
    measured local noise that drives Hasselt's overestimation.

    Distinct from `q_late_mean` (mean Q level): SNR = |q_late_
    mean| / q_action_std_late captures whether action-noise is
    small relative to true value differences (high SNR, max-bias
    benign) or comparable (low SNR, max-bias dominates). Reward
    scaling shifts q_late_mean but doesn't necessarily shift
    σ_action proportionally — hence the rs-coupling of DDQN's
    benefit may operate through SNR rather than through
    accumulated bias / chain depth.

    Returns nan if `online_std_q_per_step` is absent (cache
    lacking trace column) or if the late window is empty."""
    try:
        arr = ONLINE_STD_Q(record)
    except KeyError:
        return float('nan')
    return _windowed_mean(arr, 0.5, 1.0)


@measurable(reads=('online_max_q_per_step', 'online_min_q_per_step',
                   'online_std_q_per_step'))
def q_range_to_std_late(record: Mapping[str, object]) -> float:
    """Mean of (max_Q − min_Q) / σ_Q over the late 50% of training,
    averaged across non-terminal states.

    **Conservative proxy for argmax-vulnerability**:
    `(max − min) / σ` ≥ `(top1 − top2) / σ`, so when this ratio
    is small, the *true* margin-to-noise ratio is also small →
    argmax fragile under bias. When this ratio is large,
    inconclusive (could still be argmax-fragile if K-1 actions
    cluster low and 1 outlier high, but bias-vulnerability is
    less likely on average).

    Computable from existing trace columns; intended as the
    continuous G2 predicate in the three-gate decomposition (see
    `docs/DDQN_THREE_GATES.md`) until `q_argmax_margin_late`
    populates from new sweeps. For binary actions (CartPole),
    `max − min = top1 − top2` exactly, so this proxy is tight.

    Returns nan when any of the trace columns is missing or the
    late window is empty."""
    try:
        max_arr = ONLINE_MAX_Q(record)
        min_arr = ONLINE_MIN_Q(record)
        std_arr = ONLINE_STD_Q(record)
    except KeyError:
        return float('nan')
    if max_arr.size != min_arr.size or max_arr.size != std_arr.size:
        return float('nan')
    n = max_arr.size
    lo = int(n * 0.5)
    if lo >= n:
        return float('nan')
    rng = max_arr[lo:] - min_arr[lo:]
    sd = std_arr[lo:]
    valid = sd > 1e-9
    if not valid.any():
        return float('nan')
    return float(np.mean(rng[valid] / sd[valid]))


@measurable(reads=('online_top12_margin_per_step', 'eval_step_index'))
def q_argmax_margin_per_burst(
    record: Mapping[str, object],
) -> npt.NDArray[np.float64]:
    """Per-burst analog of `q_argmax_margin_late`. Returns
    `(n_bursts,)` — per-burst mean of `online_top12_margin_per_step`.

    Used to test whether action-margin mediates the per-burst
    Q-channel within canonical-config scope (where cell-level
    margin mediation drops to ~6%). See
    `findings_two_channel_cross_corpus.md`."""
    try:
        arr = ONLINE_TOP12_MARGIN(record)
        eval_idx = EVAL_STEP_INDEX(record)
    except KeyError:
        return np.zeros((0,), dtype=np.float64)
    if arr.ndim < 1 or eval_idx.ndim < 1:
        return np.zeros((0,), dtype=np.float64)
    n = int(arr.shape[0])
    n_bursts = int(eval_idx.shape[0])
    if n == 0 or n_bursts == 0:
        return np.zeros((0,), dtype=np.float64)
    edges = np.linspace(0, n, n_bursts + 1, dtype=np.int64)
    return np.array(
        [float(arr[edges[i]:edges[i+1]].mean()) for i in range(n_bursts)],
        dtype=np.float64,
    )


@measurable(reads=('online_std_q_per_step', 'eval_step_index'))
def q_action_std_per_burst(
    record: Mapping[str, object],
) -> npt.NDArray[np.float64]:
    """Per-burst analog of `q_action_std_late`. Returns
    `(n_bursts,)` — per-burst mean of `online_std_q_per_step`
    (cross-action Q-stdev at non-terminal states)."""
    try:
        arr = ONLINE_STD_Q(record)
        eval_idx = EVAL_STEP_INDEX(record)
    except KeyError:
        return np.zeros((0,), dtype=np.float64)
    if arr.ndim < 1 or eval_idx.ndim < 1:
        return np.zeros((0,), dtype=np.float64)
    n = int(arr.shape[0])
    n_bursts = int(eval_idx.shape[0])
    if n == 0 or n_bursts == 0:
        return np.zeros((0,), dtype=np.float64)
    edges = np.linspace(0, n, n_bursts + 1, dtype=np.int64)
    return np.array(
        [float(arr[edges[i]:edges[i+1]].mean()) for i in range(n_bursts)],
        dtype=np.float64,
    )


@measurable(reads=('online_top12_margin_per_step',))
def q_argmax_margin_late(record: Mapping[str, object]) -> float:
    """Mean of `online_top12_margin_per_step` over the late 50%
    of training — per-step (Q(top1) − Q(top2)) averaged across
    late states. Captures **argmax-bias-sensitivity**: the
    continuous structural variable that |A| ≥ 3 only approximates.

    Reading: small margin → argmax flips easily under bias →
    DDQN's bias-correction translates to policy improvement.
    Large margin → argmax robust → bias is policy-irrelevant
    (CartPole's |A|=2 + decisive states keep margin large).

    Companion to `q_action_std_late` (the σ_action Hasselt floor).
    The dimensionless ratio `q_argmax_margin_late /
    q_action_std_late` is the natural argmax-robustness score —
    when ratio < 1, bias differential can flip argmax; when > 1,
    bias is below margin scale.

    Returns nan if the trace column is absent."""
    try:
        arr = ONLINE_TOP12_MARGIN(record)
    except KeyError:
        return float('nan')
    return _windowed_mean(arr, 0.5, 1.0)


@measurable(reads=('n_actions',))
def hasselt_implied_per_step_bias(
    record: Mapping[str, object],
    q_action_std_late: float,  # injected via @measurable name resolution
) -> float:
    """Theoretical per-step max-bias `c · σ_action` where
    `c = √(2 ln(|A|) / π)` — the closed-form Hasselt 2010
    overestimation under iid-normal-noise assumption on Q
    estimates across actions.

    This is what the chain-amplifier's accumulated bias is built
    from: per-step bias × effective_horizon ≈ jensen_gap. So if
    the chain-amplifier story is right, jensen_gap should track
    `hasselt_implied_per_step_bias × effective_horizon` cross-env.

    NaN propagates from missing/invalid inputs (n_actions ≤ 1,
    σ_action non-finite)."""
    n_actions = record.get('n_actions')
    if not isinstance(n_actions, (int, float)) or n_actions <= 1:
        return float('nan')
    if not math.isfinite(q_action_std_late) or q_action_std_late < 0.0:
        return float('nan')
    c = math.sqrt(2.0 * math.log(float(n_actions)) / math.pi)
    return c * q_action_std_late


@measurable(reads=())
def q_signal_to_noise_late(
    record: Mapping[str, object],
    q_late_mean: float,            # injected
    q_action_std_late: float,      # injected
) -> float:
    """`|q_late_mean| / q_action_std_late` — Q-value signal-to-
    noise at non-terminal states, late training.

    High SNR: Q values are large relative to cross-action stdev
    → max-bias is small relative to value differences → DDQN's
    correction is marginal.
    Low SNR: σ_action comparable to or larger than mean(Q) →
    max-bias dominates the signal → DDQN's correction matters.

    The cross-env interpretive frame for the rs-coupling
    finding (FourRooms rs=0.1 g_out=+3.0 vs Acrobot rs=0.1 g_out=
    -0.17): if rs scales mean(Q) but not σ_action proportionally,
    SNR drops at small rs. Where SNR drops AND env structure has
    chain depth, DDQN's relative benefit grows. Tests the
    'reward scale interacts with noise structure' hypothesis.

    NaN propagates from invalid inputs (σ_action ≤ 0)."""
    if not math.isfinite(q_late_mean) or not math.isfinite(q_action_std_late):
        return float('nan')
    if q_action_std_late <= 0.0:
        return float('nan')
    return abs(q_late_mean) / q_action_std_late


@measurable(reads=('online_max_q_per_step',))
def q_late_mean(record: Mapping[str, object]) -> float:
    """Mean of `online_max_q_per_step` over the late 50% of
    training — a scalar summary of where the value function ends
    up.

    Sign of `q_late_mean` reflects the env's Q-regime, which is
    determined exogenously by `r_min`:

    - `r_min ≥ 0` (sparse-terminal-positive, e.g. FourRooms):
      Q* ∈ [0, R_max/(1−γ)] is positive bounded above. Vanilla
      DQN's max-bootstrap pushes Q ABOVE the true value — wrong
      actions get inflated values, policy degenerates. DDQN's
      argmax/max separation corrects upward bias.
      `q_late_mean > 0` per-cell.

    - `r_min < 0` (dense-penalty, e.g. Acrobot, MountainCar):
      Q* ∈ [−|r_min|/(1−γ), 0] is negative bounded below.
      Vanilla overestimation pushes Q LESS NEGATIVE than truth
      (mild optimism aiding exploration through the penalty
      floor). DDQN's correction removes this optimism — sometimes
      hurts. `q_late_mean < 0` per-cell.

    The endogenous downstream of `r_min`. Used as a regime-
    selector predicate in bridges that test claims dependent on
    the sign of Hasselt's bias direction."""
    try:
        arr = ONLINE_MAX_Q(record)
    except KeyError:
        return float('nan')
    return _windowed_mean(arr, 0.5, 1.0)


@measurable(reads=('online_max_q_per_step', 'eval_step_index'))
def q_per_burst(
    record: Mapping[str, object],
) -> npt.NDArray[np.float64]:
    """Per-burst mean of `online_max_q_per_step`. Returns `(n_bursts,)`.

    Per-burst analog of `q_late_mean` (full-trajectory late-50%
    reduction). Chunks per-step Q into n_bursts equal training-step
    windows, takes mean per window. The Q-magnitude channel at
    per-burst granularity for the two-channel decomposition
    (`findings_ddqn_reward_sign_conditional.md`):
    `bg_per_burst → mc_per_burst` and `q_per_burst → mc_per_burst`
    are independent direct edges in the per-burst PC graph."""
    try:
        arr = ONLINE_MAX_Q(record)
        eval_idx = EVAL_STEP_INDEX(record)
    except KeyError:
        return np.zeros((0,), dtype=np.float64)
    n = int(arr.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=np.float64)
    n_bursts = int(eval_idx.shape[0])
    if n_bursts == 0:
        return np.zeros((0,), dtype=np.float64)
    edges = np.linspace(0, n, n_bursts + 1, dtype=np.int64)
    return np.array(
        [float(arr[edges[i]:edges[i+1]].mean()) for i in range(n_bursts)],
        dtype=np.float64,
    )


@measurable(reads=('online_max_q_per_step',))
def q_max_growth(record: Mapping[str, object]) -> float:
    """late_quarter / max(|early_quarter|, 1e-9) of online_max_q.
    Value-curve growth — vanilla DQN's Jensen bias typically
    pushes this above 1; DDQN attenuates."""
    try:
        arr = ONLINE_MAX_Q(record)
    except KeyError:
        return float('nan')
    early = _windowed_mean(arr, 0.0, 0.25)
    late = _windowed_mean(arr, 0.75, 1.0)
    return float(late / max(abs(early), 1e-9))


@measurable(reads=('q_action_grad_overlap_per_step',))
def q_action_grad_overlap_late(record: Mapping[str, object]) -> float:
    """Late-window mean of per-step `q_action_grad_overlap_per_step`:
    mean off-diagonal cosine similarity of `∂Q(s, a_i)/∂θ` for action
    pairs (i, j) at a single sampled batch state per training step,
    averaged over the late 50% of training.

    THE theoretical intra-state α from `findings_fa_depth_within_env`:
    when the trunk's gradient is updated, this is the cross-action
    correlation that propagates the update into all action heads.

    Closed-form references:
      - Tabular Q (independent (s,a) entries): α = 0
      - Linear FA `Q(s,a) = W_a · obs + b_a` (independent rows): α = 0
      - Shared-trunk MLP: α > 0, magnitude depends on trunk
        capacity + action-head dot products

    THIS is the measurement that distinguishes FA architectures
    in the way the FA-coherence theory specifies. The
    `q_action_temporal_corr_at_state_late` reduction measured
    Q-VALUE temporal correlation, which conflates this gradient-
    overlap with limited-capacity Q-rank-deficiency and
    trajectory-induced drift; it returns near-1 for BOTH linear
    and deep FA. Use `q_action_grad_overlap_late` for the
    architectural-α test; use the temporal correlation only as
    a Q-rank-deficiency proxy."""
    try:
        arr = Q_ACTION_GRAD_OVERLAP(record)
    except KeyError:
        return float('nan')
    if arr.ndim == 0 or arr.shape[0] < 2:
        return float('nan')
    return _windowed_mean(arr, 0.5, 1.0)


@measurable(reads=('bootstrap_action_mismatch_per_step',))
def bootstrap_action_mismatch_late(
    record: Mapping[str, object],
) -> float:
    """Late-window mean of the cross-action bootstrap rate:
    fraction of training transitions where
    `argmax_a' Q_online(s', a') ≠ action_taken_at_s`.

    **Theory connection**: DDQN's contribution is decorrelating
    the argmax-selection from the value-estimation in the
    bootstrap target. That decorrelation matters EXACTLY when
    the bootstrap pulls Q(s, a) toward Q(s', a') with a' ≠ a —
    cross-action TD updates. When the rate is high (s' decisions
    diverge from s decisions frequently), DDQN's mechanism has
    leverage. When near zero (within-action bootstrap), the
    update is trivially Q(s, a) ← r + γ · Q(s', a) and DDQN
    has nothing to fix.

    Conjunction with axis (ii) `γ → 1` and axis (iii) `r → 0`:
    when r → 0 the TD target reduces to γ · max_a' Q(s', a');
    when γ → 1 the propagation horizon is long; when a' ≠ a
    frequently the cross-action correction accumulates. ALL
    THREE jointly are the regime where DDQN should help most.

    Cheap probe added 2026-05-13 in `train_phase` — single
    argmax + compare per batch step. Negligible cost."""
    try:
        arr = BOOTSTRAP_ACTION_MISMATCH(record)
    except KeyError:
        return float('nan')
    if arr.ndim == 0 or arr.shape[0] < 2:
        return float('nan')
    return _windowed_mean(arr, 0.5, 1.0)


@measurable(reads=('q_inter_state_grad_overlap_per_step',))
def q_inter_state_grad_overlap_late(
    record: Mapping[str, object],
) -> float:
    """Late-window mean of inter-state α — cosine overlap of
    `∂Q(s, a)/∂θ` vs `∂Q(s', a)/∂θ` at paired (s, s') from the
    replay batch (s' = batch.next_obs[0]), averaged across actions.

    Theory's axis (i) of FA-coherence: spatial smoothness of Q
    GRADIENTS under the FA at neighboring states. Distinct from:
    - `q_action_grad_overlap_late`: intra-state α (action-head
      overlap at the same state).
    - `q_trajectory_autocorr_late`: trajectory-Q autocorr
      (confounded by env-dynamics smoothness).
    - `q_autocorr_late`: training-batch-mean autocorr (confounded
      by network nonlinearity).

    Probe added 2026-05-13 in `train_phase`; populated by sweeps
    run after that. Pre-existing corpora have NaN for this
    column."""
    try:
        arr = Q_INTER_STATE_GRAD_OVERLAP(record)
    except KeyError:
        return float('nan')
    if arr.ndim == 0 or arr.shape[0] < 2:
        return float('nan')
    return _windowed_mean(arr, 0.5, 1.0)


@measurable(reads=('online_max_q_per_step',))
def q_autocorr_late(record: Mapping[str, object]) -> float:
    """Lag-1 Pearson autocorrelation of `online_max_q_per_step`
    over the late 50% of training. Proxy for function-approximator
    spatial coherence: how strongly the FA enforces
    `Q(s_t, a*) ≈ Q(s_{t+1}, a*)` for consecutive trajectory
    states. High autocorr (→ 1) means the FA is over-smoothing
    across nearby states — a single TD update at one state shifts
    Q at neighbors via shared trunk gradients. Low autocorr
    (→ 0) means states discriminate cleanly under the FA.

    Empirical (post-fix vanilla, n=15 strata, 8 envs):
    cross-env r(q_autocorr_late, log(jens / σ_Q)) = +0.71 (p=0.003)
    — single strongest predictor of σ-normalized argmax-bias,
    beating chain depth, ep_len, and |A|. Pattern: FR/MetaMaze
    (slow-drift maze states) autocorr ~0.7-0.999 with high jens;
    Acrobot (fast-pendulum dynamics) autocorr ~0.06 with low
    jens despite long chains.

    Mechanism: argmax overestimate at Q(s, argmax_a) is pushed
    through the trunk; high autocorr means that push redistributes
    onto Q(s', a) for s' ≈ s, amplifying spatial bias coverage.
    DDQN's argmax-decorrelation breaks this loop — should help
    most where autocorr is high."""
    try:
        arr = ONLINE_MAX_Q(record)
    except KeyError:
        return float('nan')
    if arr.ndim == 0 or arr.shape[0] < 2:
        return float('nan')
    half = arr.shape[0] // 2
    late = arr[half:]
    if late.shape[0] < 2:
        return float('nan')
    x = late[:-1]
    y = late[1:]
    if np.std(x) == 0 or np.std(y) == 0:
        return float('nan')
    r = float(np.corrcoef(x, y)[0, 1])
    return r if math.isfinite(r) else float('nan')


@measurable(reads=('predicted_q_per_step', 'active_per_step'))
def q_trajectory_autocorr_late(
    record: Mapping[str, object],
) -> float:
    """**Inter-state** Q correlation along actual eval trajectories,
    averaged over late-half bursts.

    For each (burst, episode) in the late half of `predicted_q_per_step`
    (shape `(n_bursts, n_episodes, episode_cap)`), masks to active
    steps and computes the lag-1 Pearson autocorr of Q values
    along the trajectory. Returns the mean across episodes/bursts.

    **Distinct from `q_autocorr_late`** which uses
    `online_max_q_per_step` — that trace is per-training-step
    mean over a RANDOM REPLAY BATCH (`phases.py:199`), so the
    autocorr is between consecutive batch-means, not between
    consecutive trajectory states. The training-batch quantity
    confounds (a) network nonlinearity (linear FA produces
    smoother batch-means than deep MLP → higher autocorr), (b)
    replay-buffer stability, and (c) Q-network convergence —
    none of which is the theory's "FA spatial coherence ALONG A
    TRAJECTORY" axis.

    **Distinct from `q_action_grad_overlap_late`** which measures
    *intra*-state α (cosine overlap of ∂Q/∂θ across action heads
    AT the same state — action-head shared trunk). The theory's
    FA-degeneracy axis (i) requires *inter*-state α: how Q at
    state s relates to Q at neighboring state s' under the FA.
    This measurable is the natural empirical proxy via the
    realized trajectory.

    Computed at EVAL time on the agent's trajectory (with current
    policy's action selection). Reflects (env, trained-policy)
    pair; the vanilla-arm value at the converged policy is the
    intended proxy for env-architecture-determined spatial
    smoothness."""
    q = record.get('predicted_q_per_step')
    a = record.get('active_per_step')
    if q is None or a is None:
        return float('nan')
    q_arr = np.asarray(q, dtype=np.float64)
    a_arr = np.asarray(a, dtype=np.float64)
    if q_arr.ndim != 3 or q_arr.shape != a_arr.shape:
        return float('nan')
    n_bursts = q_arr.shape[0]
    if n_bursts < 2:
        return float('nan')
    half = n_bursts // 2
    late_q = q_arr[half:]
    late_a = a_arr[half:]
    corrs: list[float] = []
    n_b_late, n_episodes, _ = late_q.shape
    for b in range(n_b_late):
        for e in range(n_episodes):
            mask = late_a[b, e] > 0
            q_traj = late_q[b, e][mask]
            if q_traj.size < 2:
                continue
            x = q_traj[:-1]
            y = q_traj[1:]
            if np.std(x) == 0 or np.std(y) == 0:
                continue
            r = float(np.corrcoef(x, y)[0, 1])
            if math.isfinite(r):
                corrs.append(r)
    if not corrs:
        return float('nan')
    return float(np.mean(corrs))


@measurable(reads=('online_max_q_per_step', 'eval_step_index'))
def q_autocorr_per_burst(
    record: Mapping[str, object],
) -> npt.NDArray[np.float64]:
    """Per-burst version of `q_autocorr_late`. Splits
    `online_max_q_per_step` (shape `(total_steps,)`) into N equal
    chunks where N = `len(eval_step_index)` (one window per eval
    burst), and returns the lag-1 Pearson autocorrelation within
    each window.

    Returns shape `(n_bursts,)`. Each value is the autocorr over
    the corresponding training-step chunk, NaN when the chunk has
    zero std (constant Q).

    Use when the fixed late-50% window of `q_autocorr_late`
    silently mixes learning + converged phases. The per-burst
    trajectory makes the convergence point visible: autocorr
    typically rises monotonically and saturates after the agent
    locks onto a policy. Downstream analyses can take `[-1]` for
    the final burst, mean over the last K bursts, or fit a slope
    `autocorr ~ burst_idx`.

    Long-term fix (deferred per
    `project_convergence_detect_pelt.md` /
    `project_convergence_detect_spectral.md`): replace the fixed-
    chunk windowing with PELT-detected segment boundaries OR
    ADF-stationarity detection on TD-error. Per-burst is the
    cheap-ship-now option."""
    try:
        arr = ONLINE_MAX_Q(record)
        step_idx = EVAL_STEP_INDEX(record)
    except KeyError:
        return np.zeros((0,), dtype=np.float64)
    if arr.ndim != 1 or step_idx.ndim != 1:
        return np.zeros((0,), dtype=np.float64)
    n_bursts = int(step_idx.shape[0])
    if n_bursts < 1 or arr.shape[0] < n_bursts * 2:
        return np.zeros((0,), dtype=np.float64)
    chunks = np.array_split(arr, n_bursts)
    out = np.full((n_bursts,), np.nan, dtype=np.float64)
    for i, chunk in enumerate(chunks):
        if chunk.size < 2:
            continue
        x = chunk[:-1]
        y = chunk[1:]
        if np.std(x) == 0 or np.std(y) == 0:
            continue
        r = float(np.corrcoef(x, y)[0, 1])
        if math.isfinite(r):
            out[i] = r
    return out


@measurable(reads=(
    'mc_return_from_step', 'active_per_step', 'gamma',
))
def reward_nonzero_frac(
    record: Mapping[str, object],
    eps: float = 1e-3,
) -> float:
    """Fraction of active eval-trajectory steps with non-zero per-
    step reward. Proxy for env-level reward informativeness — high
    (→ 1) means every step gets reward (CartPole-style dense
    return); low (→ 0) means reward is sparse and the TD target
    reduces to `γ · Q(s, a*)` (the degenerate self-referential map
    in `findings_unified_degeneracy_theory.md`).

    Reconstructs per-step reward via
    `r[t] = mc[t] − γ · mc[t+1]`
    on `active[t] == 1` steps, and terminal `r[T−1] = mc[T−1]`
    (no future). Then divides the count of `|r| > eps` by the
    active-step count.

    `eps = 1e-3` is a numerical-noise threshold (mc_return_from_step
    is float32 in cache, so true zeros may carry ~1e-7 jitter).
    Sweep-relevant rewards are O(1).

    **Policy-conditional caveat**: this is computed over eval
    trajectories, so its value reflects the (env, trained policy)
    pair. On a sparse-reward env with a trained agent that solves
    the task, density ≈ `1 / episode_length` (terminal-reward
    only). On the same env with vanilla collapse, density → 0
    (agent never reaches goal). Use the VANILLA-arm value per env
    as the operational proxy for env-level reward sparsity (the
    theory's `uninf-r` axis). Shaped envs (via `PotentialReward`)
    show density → 1 even on bare-reward envs because the
    potential transform adds per-step signal."""
    mc = record.get('mc_return_from_step')
    active = record.get('active_per_step')
    gamma_v = record.get('gamma')
    if mc is None or active is None:
        return float('nan')
    if not isinstance(gamma_v, (int, float)):
        return float('nan')
    gamma = float(gamma_v)
    mc_arr = np.asarray(mc, dtype=np.float64)
    a_arr = np.asarray(active, dtype=np.float64)
    if mc_arr.ndim < 1 or mc_arr.shape != a_arr.shape or mc_arr.size == 0:
        return float('nan')
    # Shift mc backwards along the trailing (step) axis. Treat past-
    # episode-end positions as `r[t] = mc[t]` (no future): set
    # next_mc to 0 where active[t+1] == 0 or at the array end.
    next_mc = np.roll(mc_arr, -1, axis=-1)
    next_active = np.roll(a_arr, -1, axis=-1)
    # Last step of trailing axis has no `t+1`: force terminal.
    next_mc[..., -1] = 0.0
    next_active[..., -1] = 0.0
    next_mc = next_mc * next_active
    rewards = mc_arr - gamma * next_mc
    mask = a_arr > 0
    denom = float(mask.sum())
    if denom < 1.0:
        return float('nan')
    nonzero = (np.abs(rewards) > eps) & mask
    return float(nonzero.sum() / denom)


@measurable(reads=(
    'predicted_q_per_step', 'mc_return_from_step', 'active_per_step',
))
def mean_per_state_cumulative_bias_per_burst(
    record: Mapping[str, object],
) -> npt.NDArray[np.float64]:
    """Per-burst version of `mean_per_state_cumulative_bias_late`:
    returns shape `(n_bursts,)` with each element the active-weighted
    mean of `predicted_q_per_step − mc_return_from_step` across all
    visited states in that burst.

    Companion to `jensen_bias_per_burst_mean` (which probes only s_0
    per burst). Use with `paired_link_per_burst` to test whether the
    chain-traced per-state mean carries link signal beyond
    bias-at-start at the per-burst resolution."""
    q = record.get('predicted_q_per_step')
    mc = record.get('mc_return_from_step')
    active = record.get('active_per_step')
    if q is None or mc is None or active is None:
        return np.zeros((0,), dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64)
    mc_arr = np.asarray(mc, dtype=np.float64)
    a_arr = np.asarray(active, dtype=np.float64)
    if q_arr.ndim < 3 or q_arr.size == 0:
        return np.zeros((0,), dtype=np.float64)
    bias = q_arr - mc_arr
    weighted = bias * a_arr
    num = weighted.sum(axis=(1, 2))
    den = a_arr.sum(axis=(1, 2))
    out = np.where(den > 0, num / den, np.nan)
    return out.astype(np.float64)


@measurable(reads=(
    'predicted_q_per_step', 'mc_return_from_step',
    'active_per_step', 'eval_step_index',
))
def mean_per_state_cumulative_bias_late(
    record: Mapping[str, object],
) -> float:
    """Mean per-state cumulative bias `Q(s_t) − G(s_t)` averaged
    across all visited states (active steps) over the late half
    of training bursts.

    `predicted_q_per_step`, `mc_return_from_step`, `active_per_step`
    have shape `(n_bursts, K, episode_cap)` where:
    - `predicted_q_per_step[b, k, t]` = max_a Q_online(s_t, a) at
      state t of episode k in burst b (the agent's value prediction
      at each visited state, NOT just the start).
    - `mc_return_from_step[b, k, t]` = Σ_{j≥t} γ^(j−t) r_j (the
      remaining-chain MC ground truth from state t — the bias-free
      target that Q(s_t) is approximating).
    - `active_per_step[b, k, t]` = 1.0 while episode k is still
      running at step t, 0.0 after `done`.

    Per-state bias = `predicted_q_per_step − mc_return_from_step`.
    Averaged over the late-half-of-bursts × all rollouts × all
    active steps. Inactive (post-done) steps are excluded by mask.

    **Why this matters relative to `jensen_gap`**: `jensen_gap`
    aggregates `predicted_q_at_start − mc_return` — bias measured
    only at episode-start states (t=0), which sit at maximal chain
    depth. Long-trajectory envs see large bias-at-start by virtue
    of chain depth alone, not necessarily because per-step bias is
    larger. This measurable de-confounds chain-position from
    bias-rate by averaging across all visited states (each at its
    own remaining-chain depth)."""
    q = record.get('predicted_q_per_step')
    mc = record.get('mc_return_from_step')
    active = record.get('active_per_step')
    if q is None or mc is None or active is None:
        return float('nan')
    q_arr = np.asarray(q, dtype=np.float64)
    mc_arr = np.asarray(mc, dtype=np.float64)
    active_arr = np.asarray(active, dtype=np.float64)
    if q_arr.ndim < 1 or q_arr.size == 0:
        return float('nan')
    n_b = q_arr.shape[0]
    if n_b < 2:
        return float('nan')
    late_q = q_arr[n_b // 2:]
    late_mc = mc_arr[n_b // 2:]
    late_active = active_arr[n_b // 2:]
    bias = late_q - late_mc
    weighted = bias * late_active
    total_weight = late_active.sum()
    if total_weight < 1.0:
        return float('nan')
    return float(weighted.sum() / total_weight)


@measurable(reads=('online_argmax_per_step',))
def argmax_persistence_late(record: Mapping[str, object]) -> float:
    """Fraction of consecutive late-window step-pairs where the
    online argmax is the same step-to-step. Reduces over the
    *temporal* axis (different from `argmax_mode_freq_late`,
    which reduces over the categorical-action axis).

    Reads `online_argmax_per_step` (per-step argmax over batch-mean
    Q values). For the late 50% of training, computes
    `mean(argmax[t] == argmax[t-1])`.

    Range: [1/|A|_eff, 1.0]. 1.0 = network's argmax never flips
    between training steps (extremely stable). 1/|A|_eff =
    flips at chance rate (every step is independent argmax draw).

    Theoretical reading: bias correction → less Q oscillation step-
    to-step → stable argmax → directed policy → fewer wasted
    actions → shorter episodes. This decomposes the
    bias-correction → length-reduction step that polarity-tautology
    reductions like `effective_horizon` cannot — argmax persistence
    is independent of polarity by construction.

    Caveat: the argmax here is over batch-mean Q at each training
    step, not over per-state Q at visited states. Step-to-step
    flips reflect both real Q instability AND batch composition
    drift. Two consecutive batches drawing very different
    transitions can produce different argmaxes even with stable Q.
    Read in conjunction with `argmax_entropy_late` (which captures
    the across-action histogram) to disambiguate."""
    try:
        arr = ONLINE_ARGMAX(record)
    except KeyError:
        return float('nan')
    n = arr.shape[0]
    if n < 4:
        return float('nan')
    late_start = n // 2
    late = arr[late_start:]
    if len(late) < 2:
        return float('nan')
    matches = (late[1:] == late[:-1]).astype(np.float64)
    return float(np.mean(matches))


@measurable(reads=('online_max_q_per_step',))
def q_max_temporal_cv_late(record: Mapping[str, object]) -> float:
    """Temporal coefficient of variation (std / |mean|) of
    `online_max_q_per_step` over the late 50% of training.

    Reduces *across training time*: high CV = max Q oscillates
    in time at scale comparable to its mean; low CV = max Q is
    stable in time. Distinct from `q_action_std_late` which
    reduces across actions at each step.

    Theoretical reading: bias correction → bounded Q-iteration →
    stable max-Q over time → less temporal noise driving the
    policy. A direct measurement of "is the value estimate
    converged in time" — the assumption that Hasselt's per-step
    bias *integrates* into a stable bias requires the underlying
    Q to itself be temporally stable.

    Returns NaN if `online_max_q_per_step` absent or if late-window
    mean is below 1e-9 in magnitude (CV undefined)."""
    try:
        arr = ONLINE_MAX_Q(record)
    except KeyError:
        return float('nan')
    n = arr.shape[0]
    if n < 4:
        return float('nan')
    late = arr[n // 2:]
    if len(late) < 2:
        return float('nan')
    mu = float(np.mean(late))
    if abs(mu) < 1e-9:
        return float('nan')
    sd = float(np.std(late, ddof=1))
    return sd / abs(mu)


@measurable(reads=('mc_return_from_step', 'episode_length', 'gamma', 'mc_return'))
def env_disc_raw_alignment(record: Mapping[str, object]) -> float:
    """Per-cell Pearson r between **discounted** and **undiscounted**
    episode returns across all (burst, eval-episode) pairs. An
    endogenous proxy for how much discounting bites in this env:

    - r → +1: episode lengths similar across episodes (or rewards
      concentrated near terminal step). Discounting reshapes uniformly;
      raw and disc co-vary perfectly. Outcomes interchangeable up to
      a constant scaling. Examples: fixed-length envs.

    - r ≈ 0 or low: episode lengths vary wildly. Discounting
      gives short-quick episodes much higher disc-return than
      long-success ones with same raw return. Raw and disc rank
      cells differently. Outcome choice MATTERS.

    Use as a scope predicate for bridges that consume an
    outcome measure: where `env_disc_raw_alignment > 0.7`,
    `eval_best_burst_raw_mean` and `eval_best_burst_mean` are
    interchangeable; below that, the bridge must commit to one
    semantics and justify it. Companion to `env_reward_polarity`
    (the REACH/SURVIVE moderator)."""
    raw = _compute_mc_return_raw(record)
    if raw.size == 0:
        return float('nan')
    try:
        mc = MC_RETURN(record)
    except KeyError:
        return float('nan')
    mc_arr = np.asarray(mc, dtype=np.float64)
    if raw.shape != mc_arr.shape:
        return float('nan')
    rf = raw.flatten()
    mf = mc_arr.flatten()
    if rf.size < 3 or rf.std() == 0 or mf.std() == 0:
        return float('nan')
    r = float(np.corrcoef(rf, mf)[0, 1])
    return r if math.isfinite(r) else float('nan')


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
    try:
        el = EP_LENGTH(record)
        mc = MC_RETURN(record)
    except KeyError:
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
    try:
        qs = PREDICTED_Q_AT_START(record)
        mc = MC_RETURN(record)
    except KeyError:
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
    try:
        omax = ONLINE_MAX_Q(record)
        tmax = TARGET_MAX_Q(record)
    except KeyError:
        return float('nan')
    abs_gap = np.abs(omax - tmax)
    denom = np.maximum(np.maximum(np.abs(omax), np.abs(tmax)), 1e-6)
    return _windowed_mean(abs_gap / denom, 0.5, 1.0)


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
    try:
        omax = ONLINE_MAX_Q(record)
        tmax = TARGET_MAX_Q(record)
    except KeyError:
        return float('nan')
    abs_gap = np.abs(omax - tmax)
    denom = np.maximum(np.maximum(np.abs(omax), np.abs(tmax)), 1e-6)
    return _windowed_mean(abs_gap / denom, 0.0, 0.25)


@measurable(reads=('online_mean_q_per_step', 'online_max_q_per_step'))
def v_vs_max_delta_late(record: Mapping[str, object]) -> float:
    """Mean of |q_mean − q_max| over the late 50%. DDQN's
    mechanism signature: vanilla's overestimation widens the
    action-Q distribution (large delta); DDQN's decoupled
    selection narrows it. The Hasselt 2010 Jensen-bias proxy at
    the per-step level."""
    try:
        mean_q = ONLINE_MEAN_Q(record)
        max_q = ONLINE_MAX_Q(record)
    except KeyError:
        return float('nan')
    return _windowed_mean(np.abs(mean_q - max_q), 0.5, 1.0)


@measurable(reads=('td_error',))
def td_residual_late(record: Mapping[str, object]) -> float:
    """Mean of |TD residual| over the late 50%. TD-convergence
    scalar — Acrobot's r=+0.84 vs GaussianBandit's r=−0.81
    (sign-flip across regimes) is the canonical motivation for
    per-env PC in PAPER §6."""
    try:
        arr = TD_ERROR(record)
    except KeyError:
        return float('nan')
    return _windowed_mean(arr, 0.5, 1.0)


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
    try:
        arr = TD_WB_STD(record)
    except KeyError:
        return float('nan')
    return _windowed_mean(arr, 0.5, 1.0)


@measurable(reads=('online_argmax_per_step', 'target_argmax_per_step'))
def greedy_match_late(record: Mapping[str, object]) -> float:
    """Mean of (online_argmax == target_argmax) over the late
    50%. DDQN's slot swap explicitly decouples these argmaxes;
    large match ⇒ DDQN's mechanism is *inactive* on this cell
    (the two estimators agree, so vanilla and DDQN reduce to
    each other)."""
    try:
        online = ONLINE_ARGMAX(record)
        target = TARGET_ARGMAX(record)
    except KeyError:
        return float('nan')
    match = (online == target).astype(np.float64)
    return _windowed_mean(match, 0.5, 1.0)


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
    try:
        online = ONLINE_ARGMAX(record)
        target = TARGET_ARGMAX(record)
        eval_idx = EVAL_STEP_INDEX(record)
    except KeyError:
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


@measurable(reads=(
    'target_max_q_per_step', 'target_q_at_online_argmax_per_step',
    'eval_step_index',
))
def bootstrap_gap_magnitude_per_burst(
    record: Mapping[str, object],
) -> npt.NDArray[np.float64]:
    """Per-burst bootstrap_gap magnitude. Returns `(n_bursts,)`.
    Chunks the per-step `target_max − target_q_at_online_argmax`
    into n_bursts equal training-step windows, takes mean per
    window.

    Per-burst analog of `bootstrap_gap_magnitude` (full-trajectory
    mean). Surfaces phase-specific bg dynamics that the full-
    trajectory reduction averages away. Critical for causal
    analysis on training trajectories with non-monotone phases
    (Q-explosion → convergence, rescue regimes, etc.)."""
    try:
        target_max = TARGET_MAX_Q(record)
        target_argonline = TARGET_AT_ARGMAX(record)
        eval_idx = EVAL_STEP_INDEX(record)
    except KeyError:
        return np.zeros((0,), dtype=np.float64)
    n = min(target_max.shape[0], target_argonline.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=np.float64)
    gap = target_max[:n] - target_argonline[:n]
    n_bursts = int(eval_idx.shape[0])
    if n_bursts == 0:
        return np.zeros((0,), dtype=np.float64)
    edges = np.linspace(0, n, n_bursts + 1, dtype=np.int64)
    return np.array(
        [float(gap[edges[i]:edges[i+1]].mean()) for i in range(n_bursts)],
        dtype=np.float64,
    )


@measurable(reads=('online_argmax_per_step', 'eval_step_index', 'n_actions'))
def argmax_entropy_per_burst(
    record: Mapping[str, object],
) -> npt.NDArray[np.float64]:
    """Per-burst Shannon entropy (nats) of `online_argmax_per_step`
    distribution. Returns `(n_bursts,)`.

    Per-burst analog of `argmax_entropy_late` (late-50% window).
    Each burst's entropy is computed from the argmax counts in
    that burst's training-step window. Surfaces phase-specific
    policy-decisiveness dynamics that the late-window reduction
    averages away."""
    try:
        argmax = ONLINE_ARGMAX(record)
        eval_idx = EVAL_STEP_INDEX(record)
    except KeyError:
        return np.zeros((0,), dtype=np.float64)
    n_actions_v = record.get('n_actions')
    if not isinstance(n_actions_v, int):
        return np.zeros((0,), dtype=np.float64)
    n_bursts = int(eval_idx.shape[0])
    if n_bursts == 0:
        return np.zeros((0,), dtype=np.float64)
    n_steps = argmax.shape[0]
    edges = np.linspace(0, n_steps, n_bursts + 1, dtype=np.int64)
    out = np.zeros((n_bursts,), dtype=np.float64)
    for i in range(n_bursts):
        chunk = argmax[edges[i]:edges[i+1]].astype(np.int64)
        if chunk.size == 0:
            out[i] = float('nan')
            continue
        counts = np.bincount(chunk, minlength=int(n_actions_v))
        p = counts.astype(np.float64) / counts.sum()
        p_nz = p[p > 0]
        if p_nz.size == 0:
            out[i] = float('nan')
            continue
        out[i] = float(-(p_nz * np.log(p_nz)).sum())
    return out


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
    try:
        arr = BUF_SIZE(record)
    except KeyError:
        return float('nan')
    return _windowed_mean(arr / float(capacity), 0.5, 1.0)


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


# Raw-return per-burst reduction, mirroring
# `mc_return__mean_axis_-1`: per-burst mean of undiscounted
# episode return. Computes the (n_bursts, n_episodes) raw return
# inline from trace columns (see `_compute_mc_return_raw` below),
# then averages over the episode axis to yield (n_bursts,). Direct
# trace-column reads so the dependency walker sees the leaf
# trace cols (rather than an intermediate measurable name the
# build_measurements satisfiability check can't see).
def _mc_return_raw_per_burst_mean(
    record: Mapping[str, object],
) -> npt.NDArray[np.floating]:
    raw = _compute_mc_return_raw(record)
    if raw.ndim != 2 or raw.size == 0:
        return np.full((0,), float('nan'), dtype=np.float64)
    return raw.mean(axis=1)


mc_return_raw_per_burst_mean = Measurable(
    fn=_mc_return_raw_per_burst_mean,
    name='mc_return_raw__mean_axis_-1',
    reads=('mc_return_from_step', 'episode_length', 'gamma'),
)
register(mc_return_raw_per_burst_mean)


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


@measurable(reads=('jensen_gap', 'effective_horizon'))
def per_step_max_bias(record: Mapping[str, object]) -> float:
    """Per-step max-bias = `jensen_gap / effective_horizon`.

    Hasselt 2010's overestimation theorem bounds the per-step
    max-action bias `ε`. The framework's `jensen_gap` measures
    the *accumulated* bias over the bootstrap chain — under the
    chain-amplifier reading,
        accumulated_bias ≈ ε × effective_horizon
    where `effective_horizon = 1/(1−γ·bf)` is the geometric chain
    depth. So `jensen_gap / effective_horizon` is the
    operationally-defined per-step bias under this model — the
    direct analogue of the per-step quantity the theorem
    constrains.

    This decomposes the chain-amplifier into its two factors:
    the per-step bias (Hasselt's quantity) and the chain-depth
    multiplier. Bridges that test the multiplicative structure
    `g_outcome ≈ −β · g_per_step_bias · effective_horizon`
    cross-env need this measurable as a scope-discriminator and
    as the regression's per-step covariate.

    Reads `effective_horizon` from the record (cached column)
    rather than via the measurable resolver, so the framework
    doesn't try to re-derive it from missing trace data when
    only the cache parquet is available.

    NaN propagates from invalid inputs: jensen_gap or
    effective_horizon non-finite, or eff_horizon ≤ 0."""
    jens = record.get('jensen_gap')
    eff_h = record.get('effective_horizon')
    if not isinstance(jens, (int, float)):
        return float('nan')
    if not isinstance(eff_h, (int, float)):
        return float('nan')
    j, h = float(jens), float(eff_h)
    if not math.isfinite(j) or not math.isfinite(h) or h <= 0.0:
        return float('nan')
    return j / h


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


def _compute_mc_return_raw(
    record: Mapping[str, object],
) -> npt.NDArray[np.floating]:
    """Per-(burst, episode) **undiscounted** episode return,
    reconstructed from `mc_return_from_step` (per-step discounted
    value-to-go).

    Math: at each step `t`, `mc[t] = r[t] + γ · mc[t+1]` →
    `r[t] = mc[t] - γ · mc[t+1]`. Last-step `r[T-1] = mc[T-1]`
    (no future). Summing per-step rewards over actual episode
    length recovers the undiscounted episode return.

    Free function (not a `@measurable`): both
    `mc_return_raw__mean_axis_-1` and `eval_best_burst_raw_mean`
    inline this so the dependency walker (`transitive_reads`)
    sees direct trace-column reads (`mc_return_from_step`,
    `episode_length`, `gamma`) rather than an intermediate
    `mc_return_raw` measurable. The intermediate would require
    the measurable graph to walk through `reads=` (it currently
    walks only through injected-param names), so the
    `build_measurements` `all(r in joined.columns for r in
    leaf_reads)` check would fail and the measurable would be
    skipped."""
    if (
        'mc_return_from_step' not in record
        or 'episode_length' not in record
        or 'gamma' not in record
    ):
        return np.full((0, 0), float('nan'), dtype=np.float64)
    mc_from_step = np.asarray(
        record['mc_return_from_step'], dtype=np.float64,
    )
    lengths = np.asarray(record['episode_length'], dtype=np.int64)
    gamma_v = record['gamma']
    if not isinstance(gamma_v, (int, float)):
        return np.full((0, 0), float('nan'), dtype=np.float64)
    gamma = float(gamma_v)
    if mc_from_step.ndim != 3 or lengths.ndim != 2:
        return np.full((0, 0), float('nan'), dtype=np.float64)
    n_bursts, n_episodes, _ = mc_from_step.shape
    if lengths.shape != (n_bursts, n_episodes):
        return np.full((0, 0), float('nan'), dtype=np.float64)
    raw = np.zeros((n_bursts, n_episodes), dtype=np.float64)
    for b in range(n_bursts):
        for e in range(n_episodes):
            T = int(lengths[b, e])
            if T <= 0:
                continue
            v = mc_from_step[b, e, :T]
            if T == 1:
                raw[b, e] = float(v[0])
                continue
            inner = v[:-1] - gamma * v[1:]
            raw[b, e] = float(inner.sum() + v[-1])
    return raw


@measurable(
    name='eval_best_burst_raw_mean',
    reads=('mc_return_from_step', 'episode_length', 'gamma'),
)
def eval_best_burst_raw_mean(record: Mapping[str, object]) -> float:
    """Undiscounted counterpart of `eval_best_burst_mean`:
    `max_i(mean(mc_return_raw[i, :]))` where `mc_return_raw` is
    reconstructed inline from `mc_return_from_step` +
    `episode_length` + `gamma`. The best-burst-seen metric on the
    raw (undiscounted) return — γ-invariant policy quality. Use
    for bridges that compare across γ or across envs with
    different reward scaling."""
    raw = _compute_mc_return_raw(record)
    if raw.ndim != 2 or raw.size == 0:
        return float('nan')
    return float(raw.mean(axis=1).max())


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
# Both `experiments.findings.ddqn` and
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

# Per-burst std across the n_episodes evaluation rollouts. Captures
# pure environment-stochasticity at a fixed policy (same Q, K
# independent episode rollouts → σ across rollouts). Independent of
# algorithm choice (vanilla / DDQN / etc.) at fixed policy snapshot
# — useful as a SCOPE predicate to distinguish "env intrinsically
# noisy" (MetaMaze-misc CV ≈ 0.7) from "env quiet" (FourRooms CV ≈
# 0.09), so cross-env outcome-magnitude bridges can scope on
# detectability. See `findings_dowhy_two_mediator_chain.md` and the
# meta-discussion of CLAIM 22's MetaMaze underpower.
mc_return_per_burst_std: Measurable[
    Mapping[str, object], npt.NDArray[np.floating],
] = _reduce_axis(from_key('mc_return'), axis=-1, op='std')

jensen_bias_per_burst_mean: Measurable[
    Mapping[str, object], npt.NDArray[np.floating],
] = _reduce_axis(jensen_bias_per_eps, axis=-1, op='mean')


@measurable(
    name='outcome_episode_sigma',
    reads=('mc_return',),
)
def outcome_episode_sigma(record: Mapping[str, object]) -> float:
    """Mean across bursts of σ(mc_return) across the K eval episodes
    within each burst. Captures **pure env stochasticity at a fixed
    policy snapshot** — independent of algorithm at any single burst
    (same Q, K independent rollouts → σ across rollouts).

    Useful as a SCOPE predicate to distinguish env-intrinsic noise
    (MetaMaze ≈ 17, procedurally generated mazes) from quiet envs
    (FourRooms ≈ 0.06, fixed deterministic layout). Cross-env
    outcome-magnitude bridges should scope on this to filter cells
    where mean-effect detection is impossible at typical n_seeds —
    framework's POWER_INSUFFICIENT verdict gate."""
    mc = record.get('mc_return')
    if mc is None:
        return float('nan')
    arr = np.asarray(mc, dtype=np.float64)
    if arr.ndim == 0 or arr.shape[-1] < 2:
        return float('nan')
    return float(np.mean(np.std(arr, axis=-1)))


@measurable(
    name='outcome_episode_cv',
    reads=('mc_return',),
)
def outcome_episode_cv(record: Mapping[str, object]) -> float:
    """Coefficient of variation of `outcome_episode_sigma` —
    σ across episodes / |mean across all (burst, episode)|.
    Reward-scale-free version of the env-stochasticity proxy;
    comparable across envs with different reward magnitudes.

    MetaMaze ≈ 0.7 (extreme), MountainCar ≈ 0.07, Acrobot ≈ 0.06,
    FourRooms ≈ 0.09. Predicate `outcome_episode_cv < 0.15`
    cleanly excludes envs where 30-seed mean-effect detection
    is dramatically underpowered."""
    mc = record.get('mc_return')
    if mc is None:
        return float('nan')
    arr = np.asarray(mc, dtype=np.float64)
    if arr.ndim == 0 or arr.shape[-1] < 2:
        return float('nan')
    sigma = float(np.mean(np.std(arr, axis=-1)))
    abs_mean = abs(float(np.mean(arr)))
    if abs_mean < 1e-9:
        return float('nan')
    return sigma / abs_mean


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
    normalises the prefix).

    LITERATURE CONTEXT: this is the Q − MC composite (Category B
    in `findings_jensen_gap_literature_audit.md`), operationally
    identical to Hasselt 2016 / REDQ 2021 / "Revisited" 2025.
    For Hasselt 2010 Theorem 1's structural floor
    `σ × √(2 ln K)`, see `jensen_dormancy_gap` (which encodes
    this floor internally as `max(0, floor − jens_gap)`)."""
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


@measurable(
    name='bias_std_across_states',
    reads=('predicted_q_at_start', 'mc_return'),
)
def bias_std_across_states(record: Mapping[str, object]) -> float:
    """REDQ 2021's "std-of-bias" — the second moment of the
    Q − MC composite across eval-episode samples within a cell.

    REDQ Figure 3 reports both `mean(MC − Q)` and `std(MC − Q)`
    over 1000 replay-buffer states every 50k steps as headline
    figures. The framework's `jensen_gap` matches the mean; this
    measurable matches the std for apples-to-apples comparison
    against REDQ's reporting protocol.

    See `findings_jensen_gap_literature_audit.md` for the
    comparison study design."""
    if (
        'predicted_q_at_start' not in record
        or 'mc_return' not in record
    ):
        return float('nan')
    predicted = np.asarray(record['predicted_q_at_start'], dtype=np.float64)
    actual = np.asarray(record['mc_return'], dtype=np.float64)
    if predicted.size < 2 or actual.size < 2:
        return float('nan')
    diff = predicted - actual
    return float(np.std(diff, ddof=1))


@measurable(
    name='effective_alpha',
    # NOTE: `bootstrap.greedification.alpha` IS read inside the body
    # but only when present (dampened-α cells); for vanilla / pure
    # DDQN cells the measurable falls back to arm_key string parsing.
    # If we declare it in `reads`, `build_measurements` gates the
    # entire measurable on the column's presence and skips computing
    # it for non-dampened corpora — leaving α=NaN for vanilla / DDQN
    # cells where it should be 0.0 / 1.0. The declared `reads` is
    # only `arm_key` (always present); the dampened-α leaf is read
    # opportunistically inside the function.
    #
    # **KNOWN GAP** (2026-05-11): this workaround sidesteps a
    # framework limitation — `transitive_reads`/`signature()` don't
    # support an "optional reads" axis. As a result, if a cached
    # cell was computed when `bootstrap.greedification.alpha` was
    # ABSENT (arm_key fallback → 1.0 or 0.0), and the column is
    # later added with a real α value, the cached scalar will NOT
    # invalidate: function bytecode unchanged ⇒ signature unchanged,
    # cell value non-null ⇒ partial-nullity recompute skipped. For
    # this corpus the dampened-α cells were materialised once with
    # the column present, so the trap doesn't fire — but it's a
    # latent staleness hazard for incremental cache extensions.
    # Proper fix: framework-level `optional_reads=(...)` parameter
    # on `@measurable` that hashes into `signature()` and is
    # treated as soft-dependency by `transitive_reads`. Tracked as
    # TODO; not blocking current audit.
    reads=('arm_key',),
)
def effective_alpha(record: Mapping[str, object]) -> float:
    """Derives the effective α-level for the dampened-greedify spectrum.

    Maps cells from THREE distinct algorithmic arms to a unified
    α ∈ [0, 1] continuum for cross-config slope analysis:
      - `dampened_double_greedify(α)` cells → read α directly from
        the leaf parameter
      - `max_greedify` (vanilla baseline) cells → α = 0.0
        (algebraically equivalent: `dampened_double_greedify(0.0)` =
        `α·v_DDQN + (1−α)·v_vanilla` with α=0 returns v_vanilla)
      - `double_greedify` (pure DDQN) cells → α = 1.0
        (similarly, α=1 returns v_DDQN)

    Used by the second-layer theorem α-sweep bridges to combine
    existing ddqn baseline + DDQN cells (the α=0 and α=1
    endpoints) with new dampened-α sweeps (intermediate α values),
    avoiding redundant re-runs of the endpoints. See
    `docs/SECOND_LAYER_THEOREM.md` and
    `experiments/findings/second_layer_theorem.py`.

    Returns NaN for cells from arms outside the clean DDQN-vs-vanilla
    contrast (action-duplicate, n-step, expectile, polyak, etc.) —
    bridges are responsible for restricting scope to clean cells via
    additional predicates."""
    alpha_raw = record.get('bootstrap.greedification.alpha')
    if alpha_raw is not None:
        try:
            v = float(alpha_raw)  # type: ignore[arg-type]
            if not math.isnan(v):
                return v
        except (TypeError, ValueError):
            pass
    arm = str(record.get('arm_key', ''))
    # Pure DDQN (not dampened, not adaptive, not expectile)
    if (
        'dampened' not in arm
        and 'adaptive' not in arm
        and 'expectile' not in arm
        and 'double_greedify' in arm
    ):
        return 1.0
    # Vanilla baseline
    if arm == 'baseline':
        return 0.0
    return float('nan')


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


def _jensen_dormancy_gap_per_burst_fn(
    record: Mapping[str, object],
) -> npt.NDArray[np.floating]:
    """Per-burst dormancy gap (shape (n_bursts,)).

    For burst i:
      observed_bias[i] = max(0, (predicted[i, :] - mc[i, :]).mean())
      σ_Q[i]           = mean of online_std_q_per_step over the
                         training-step window that produced burst i
      floor[i]         = σ_Q[i] × √(2 log K)
      dormancy[i]      = max(0, floor[i] - observed_bias[i])

    Resolves the scalar-`jensen_dormancy_gap` / best-burst-
    `eval_best_burst_mean` measurement-frame misalignment surfaced
    by the dormancy bridge audit (2026-05-11). The scalar form
    collapses ALL bursts via `.mean()`; downstream bridges then
    pair the collapsed dormancy with `eval_best_burst_mean` (which
    picks the BEST burst), producing artifactual "DDQN helps
    despite dormancy" readings when the best-burst was achieved
    during a transiently-active phase.

    The per-step σ_Q trace is uniformly sampled across training;
    we chunk it into `n_bursts` equal windows. Light approximation
    (assumes uniform burst cadence over training steps); matches
    the `eval_every` convention used to produce `mc_return`.

    Returns shape (n_bursts,) array of NaN-or-float values.
    Bursts where any input is degenerate (zero σ window, missing
    n_actions) report NaN at that index; downstream reductions
    (`reduce_axis`) skip NaN per polars semantics.
    """
    predicted_v = record.get('predicted_q_at_start')
    actual_v = record.get('mc_return')
    sigma_v = record.get('online_std_q_per_step')
    env = record.get('env_name')
    if predicted_v is None or actual_v is None or sigma_v is None:
        return np.asarray([float('nan')], dtype=np.float64)
    if not isinstance(env, str):
        return np.asarray([float('nan')], dtype=np.float64)
    predicted = np.asarray(predicted_v, dtype=np.float64)
    actual = np.asarray(actual_v, dtype=np.float64)
    sigma_per_step = np.asarray(sigma_v, dtype=np.float64)
    if predicted.ndim != 2 or actual.ndim != 2:
        return np.asarray([float('nan')], dtype=np.float64)
    if predicted.shape[0] != actual.shape[0]:
        return np.asarray([float('nan')], dtype=np.float64)
    n_bursts = int(predicted.shape[0])
    if n_bursts == 0 or sigma_per_step.size < n_bursts:
        return np.asarray([float('nan')] * max(1, n_bursts), dtype=np.float64)
    try:
        spec = env_catalogue.get(env)
    except KeyError:
        return np.asarray([float('nan')] * n_bursts, dtype=np.float64)
    n_actions = int(spec.n_actions)
    if n_actions < 2:
        return np.asarray([float('nan')] * n_bursts, dtype=np.float64)
    sqrt_term = math.sqrt(2.0 * math.log(n_actions))
    chunk = int(sigma_per_step.shape[0]) // n_bursts
    out = np.empty(n_bursts, dtype=np.float64)
    for i in range(n_bursts):
        lo = i * chunk
        hi = (i + 1) * chunk if i < n_bursts - 1 else int(sigma_per_step.shape[0])
        sigma_i = float(sigma_per_step[lo:hi].mean()) if hi > lo else float('nan')
        if not (sigma_i == sigma_i and abs(sigma_i) < float('inf')):
            out[i] = float('nan')
            continue
        observed_i = float(max(0.0, (predicted[i, :] - actual[i, :]).mean()))
        floor_i = sigma_i * sqrt_term
        out[i] = max(0.0, floor_i - observed_i)
    return out


jensen_dormancy_gap_per_burst: Measurable[
    Mapping[str, object], npt.NDArray[np.floating],
] = Measurable(
    fn=_jensen_dormancy_gap_per_burst_fn,
    name='jensen_dormancy_gap_per_burst',
    reads=('predicted_q_at_start', 'mc_return',
           'online_std_q_per_step', 'env_name'),
)
register(jensen_dormancy_gap_per_burst)


# Scalar reductions of the per-burst dormancy. The `_at_best_burst`
# variant picks the dormancy gap AT the burst where the outcome's
# best-burst-mean was selected — resolves the
# `jensen_dormancy_gap` ⊥ `eval_best_burst_mean` window
# misalignment by collocating both measurements at the same burst.
def _dormancy_at_best_burst_fn(record: Mapping[str, object]) -> float:
    """Dormancy gap at the BURST where outcome's best-mean occurred.

    `eval_best_burst_mean = max_i(mc_return[i, :].mean())`. This
    measurable returns `jensen_dormancy_gap_per_burst[argmax_i]` —
    the dormancy at the SAME burst the outcome was scored at.
    Use when a bridge needs "outcome and mechanism state at the
    same window" rather than "outcome at peak burst, mech averaged
    over training".
    """
    mc_v = record.get('mc_return')
    if mc_v is None:
        return float('nan')
    mc = np.asarray(mc_v, dtype=np.float64)
    if mc.ndim != 2 or mc.size == 0:
        return float('nan')
    best_idx = int(mc.mean(axis=1).argmax())
    dormancy_per_burst = _jensen_dormancy_gap_per_burst_fn(record)
    if best_idx >= dormancy_per_burst.size:
        return float('nan')
    v = float(dormancy_per_burst[best_idx])
    if math.isnan(v):
        return float('nan')
    return v


jensen_dormancy_gap_at_best_burst: Measurable[
    Mapping[str, object], float,
] = Measurable(
    fn=_dormancy_at_best_burst_fn,
    name='jensen_dormancy_gap_at_best_burst',
    reads=('predicted_q_at_start', 'mc_return',
           'online_std_q_per_step', 'env_name'),
)
register(jensen_dormancy_gap_at_best_burst)


@measurable(
    name='jensen_floor',
    reads=('online_std_q_per_step', 'env_name'),
)
def jensen_floor_measurable(record: Mapping[str, object]) -> float:
    """Hasselt 2010 Theorem 1 structural floor for argmax-noise bias:
    `mean_t(σ_action(s_t)) × √(2 ln K)`, averaged over the FULL
    training trajectory (not the arbitrary late-half window used by
    `jensen_dormancy_gap`).

    `σ_action` is the across-action SD of online Q at each replay
    step. K = env's native n_actions from the catalogue (action-
    duplicate wrappers should be handled separately via
    `effective_n_actions`; pending).

    Caveat: σ_action upper-bounds σ_noise (the strict Hasselt σ).
    σ_action² = σ_noise² + σ_structural² where σ_structural is the
    across-action variation in TRUE Q values. The measurement
    overestimates argmax-noise bias by the structural component.
    Strict isolation requires multi-pass stochastic forward or
    cross-seed canonical-state instrumentation.

    Companion to `jensen_dormancy_gap` (which uses late-half σ):
      - This measurable: floor over FULL trajectory
      - `jensen_dormancy_gap`: max(0, floor_late − observed_bias)

    For DDQN's effect on argmax-noise, compare Δ(jensen_floor)
    between arms (cancels structural Q variation if it's
    arm-invariant). Absolute floor is an upper bound.

    Returns NaN when inputs are missing or degenerate."""
    sigma_v = record.get('online_std_q_per_step')
    env = record.get('env_name')
    if sigma_v is None or not isinstance(env, str):
        return float('nan')
    sigma_per_step = np.asarray(sigma_v, dtype=np.float64)
    if sigma_per_step.size < 2:
        return float('nan')
    try:
        spec = env_catalogue.get(env)
    except KeyError:
        return float('nan')
    n_actions = int(spec.n_actions)
    if n_actions < 2:
        return float('nan')
    sigma = float(np.nanmean(sigma_per_step))
    if not np.isfinite(sigma):
        return float('nan')
    return float(sigma * math.sqrt(2.0 * math.log(n_actions)))


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
        # CLAIM 15 substrate variables.
        log_tau,
        # `q_late_mean` is the endogenous Q-regime indicator:
        # sign(q_late_mean) > 0 ⇔ r_min ≥ 0 ⇔ vanilla bias direction
        # is upward (sparse-terminal-positive); sign < 0 ⇔
        # dense-penalty. Used as bridge scope predicate to select
        # the regime where Hasselt's bias has a specific sign.
        q_late_mean,
        # Clip-wedge: per-step `target_max_q − target_q_at_online_argmax`.
        # `ddqn_bootstrap_gap` (full-trajectory) is the canonical
        # predictor for the Q-regularization channel — fires on
        # dormant cells (jens=0) wherever argmax_online ≠ argmax_target.
        # `_late` is legacy (arbitrary 50% cut-off); kept for
        # backward compat with existing bridges.
        # `clip_wedge_polarity_aligned` = clip × sign(polarity),
        # the moderation-aware predictor for CLAIM 3's polarity-
        # conditional channel sign.
        ddqn_bootstrap_gap,
        clip_wedge_polarity_aligned,
        ddqn_bootstrap_gap_late,
        # Chain-traced cumulative bias: averages bias `Q(s_t) − G(s_t)`
        # across all visited states (each at its own remaining-chain
        # depth), de-confounding chain-position from bias-rate.
        # Distinct from `jensen_gap` which only probes start states
        # (chain-deepest endpoint).
        mean_per_state_cumulative_bias_late,
        # Algorithmic activation rate: 1 − greedy_match_late = rate at
        # which DDQN's argmax/max correction fires per step on this
        # cell's trajectory. Per-cell scalar that's *not*
        # polarity-locked (vs eff_h) and *not* bias-magnitude
        # (vs jensen_gap) — captures the algorithmic side of DDQN's
        # mechanism distinct from cumulative-bias measures.
        greedy_match_late,
        # Function-approximator coherence: lag-1 autocorrelation of
        # online_max_q across consecutive training steps. Captures
        # how strongly the FA enforces Q(s,a) ≈ Q(s',a') for s ≈ s' —
        # the bias-amplification mechanism orthogonal to TD-chain
        # depth. r=+0.71 cross-env predictor of log(jens/σ_Q),
        # beating ep_len + chain + |A| in the post-fix panel
        # (`findings_fa_coherence_bias`).
        q_autocorr_late,
        # Inter-state α: lag-1 autocorr of Q along actual eval
        # TRAJECTORY states (vs `q_autocorr_late` which is
        # training-batch-mean autocorr — see docstring distinction).
        # This is the proper empirical proxy for FA spatial
        # coherence under the unified-degeneracy theory's axis (i).
        q_trajectory_autocorr_late,
        # Note: `q_autocorr_per_burst` (vector — shape (n_bursts,))
        # is registered but NOT in this scalar tuple. cell_runner
        # persists scalar `MeasurementLeaf` only; vector-shaped
        # returns would stringify via `_leaf_scalar` and break
        # ingestion. Bridges request it through `transitive_reads`
        # at the cache-build path which handles vectors. Same
        # pattern as `mean_per_state_cumulative_bias_per_burst` and
        # `mc_return_raw_per_burst_mean`.
        # Env-level reward informativeness proxy: fraction of
        # active eval-trajectory steps with |r| > eps, recovered
        # from `mc_return_from_step` + γ. Vanilla-arm value per env
        # is the theory's `uninf-r` axis (third leg of the
        # FA-degeneracy conjunction in
        # `findings_unified_degeneracy_theory.md`).
        reward_nonzero_frac,
        # FA intra-state α — gradient-overlap probe at a sampled
        # batch state per training step. THE theoretical intra-
        # state α that distinguishes shared-trunk MLP (α > 0) from
        # linear FA (α = 0) from tabular (α = 0). Persisted as a
        # per-step trace column by `train_phase`'s gradient-
        # overlap probe block.
        q_action_grad_overlap_late,
        # FA-coherence INTER-state α (theory's axis i). Cosine
        # overlap of per-action gradients at paired (s, s') from
        # the replay batch's (obs, next_obs). The proper
        # architectural measurement; see docstring distinction
        # from intra-state α and trajectory autocorr.
        q_inter_state_grad_overlap_late,
        # Cross-action bootstrap rate: fraction of training
        # transitions where argmax_a' Q_online(s', a') differs from
        # the action taken at s. THE regime where DDQN's argmax-
        # target decoupling has leverage. Conjunct with γ→1 + r→0
        # to operationalize "when DDQN should help" hypothesis.
        bootstrap_action_mismatch_late,
        # Raw (undiscounted) eval-return scalar — γ-invariant
        # policy-quality metric for cross-γ / cross-env bridges.
        # Reconstructs per-(burst, episode) raw return inline from
        # trace columns; reduces to the best-burst scalar
        # (counterpart of `eval_best_burst_mean`).
        eval_best_burst_raw_mean,
        # Per-burst array measurable `mc_return_raw_per_burst_mean`
        # is intentionally NOT in this tuple. cell_runner persists
        # scalar MeasurementLeaf only — non-scalar (NDArray) returns
        # would be stringified via `_leaf_scalar` and break ingestion
        # (the partial-null branch in `compute_missing_columns`
        # preserves non-null strings instead of recomputing). The
        # per-burst array is computed at ingestion time by
        # `build_measurements` from the joined trace columns
        # (`transitive_reads` walks `mc_return_from_step`,
        # `episode_length`, `gamma` from the registered measurable).
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
def train_episode_length_mean(record: Mapping[str, object]) -> float:
    """Mean training-time episode length (steps per episode under
    the ACTUAL TRAINING POLICY, not random nor eval).

    Computed as `total_steps / max(sum(done), 1)`. This is the
    "T" relevant for bootstrap-chain dynamics: the typical chain
    length the agent's TD updates compound over.

    Distinct from:
      - `eval_episode_cap`: env-spec timeout (upper bound)
      - `env_params.max_steps_in_episode`: env-spec horizon
      - random-policy episode length: distorted by exploration
        (REACH envs cap at horizon, SURVIVE envs die fast)

    Why it matters: the random-policy probe undercounts episode
    length on REACH envs (random never reaches → cap; trained
    reaches in much fewer steps) and overcounts on SURVIVE envs
    (random dies fast; trained survives much longer). Predictors
    of v_jens that use `T × density` should use THIS quantity,
    not random or env-spec horizon.

    NaN when `done` is missing or empty."""
    arr = record.get('done')
    if arr is None:
        return float('nan')
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0:
        return float('nan')
    n_episodes = max(int(a.sum()), 1)  # avoid zero-div on no-terminate
    return float(a.size / n_episodes)


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
