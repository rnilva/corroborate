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
from corroborate.measurables.measurable import get_registered
from corroborate.measurables.reductions import (
    cv_safe,
    from_key,
    log_safe,
    mean_window,
    reduce_axis,
)
from corroborate_rl import env_catalogue
from corroborate_rl.dqn._temporal_reduction import temporal_reduction


def _registered(name: str) -> Measurable[Mapping[str, object], object]:
    """Fetch a measurable freshly registered via `@temporal_reduction`,
    raising on the (unreachable-by-construction) miss. The
    `@temporal_reduction` decorator returns the window-reduction
    kernel — when the public Python name must remain a
    `Measurable[Mapping, object]` (e.g., exported in
    `dqn_default_measurables`), rebind via `name = _registered('name')`
    immediately after the decorator call."""
    m = get_registered(name)
    if m is None:
        raise RuntimeError(
            f'expected measurable {name!r} to be registered',
        )
    return m


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
Q_INTER_STATE_GRAD_OVERLAP_RANDOM = from_key(
    'q_inter_state_grad_overlap_random_per_step',
)


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


def _weighted_conditional_entropy(
    a_late: np.ndarray, s_late: np.ndarray, unique_s: np.ndarray,
) -> float:
    """Plug-in estimate of `H(argmax | state) = Σ_s (n_s/N) ·
    H(argmax | state=s)` over the full empirical distribution.

    All buckets contribute: singleton buckets (n_s=1) have a point-mass
    conditional → `H = 0`, which is the correct plug-in value (not
    a degenerate skip). Normalising by full `N = Σ n_s` keeps the
    support consistent with the marginal `H(argmax)`, which makes
    `H(X|Y) ≤ H(X)` hold by construction (chain rule of entropy).

    Like all plug-in entropy estimators this is biased downward in
    the small-sample regime — when state-bucket cardinality is
    high relative to N, conditional entropy reads lower than the
    underlying truth. Miller-Madow `(K-1)/(2N)` would correct, but
    we keep the unadjusted form: the bias direction is consistent
    across arms, so cross-arm MI deltas are unaffected.

    NaN only when every bucket has zero observations (impossible
    if `unique_s` was derived from `s_late`)."""
    total_h = 0.0
    total_w = 0
    for s_val in unique_s:
        mask = s_late == s_val
        n_s = int(mask.sum())
        if n_s == 0:
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


@measurable(reads=('online_argmax_per_step', 'state_hash_per_step'))
def state_conditional_argmax_entropy_late(
    record: Mapping[str, object],
) -> float:
    """State-conditional Shannon entropy of `online_argmax_per_step`
    over the late 50% of training.

    `H(argmax | state)` = E_s [ -Σ_a p(a|s) log p(a|s) ] where
    p(a|s) is the empirical fraction of late-training steps in
    state-bucket `s` where the online network's argmax was action
    `a`. Paired with `mutual_info_state_argmax_late` (and the
    marginal `argmax_entropy_late`) to decompose action-distribution
    structure: marginal entropy alone cannot distinguish a
    decisive-per-state policy with action-diversity ACROSS states
    from a Q-flat policy with action-diversity WITHIN states.

    Plug-in estimator over all observed state buckets — see
    `_weighted_conditional_entropy` for the finite-sample bias note.

    Returns NaN when `state_hash` is missing (env-side hash not
    registered) or fewer than 2 distinct state buckets are
    observed in the late slice."""
    argmax_arr = record.get('online_argmax_per_step')
    state_arr = record.get('state_hash_per_step')
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
    return _weighted_conditional_entropy(a_late, s_late, unique_s)


@measurable(reads=('online_argmax_per_step', 'state_hash_per_step'))
def mutual_info_state_argmax_late(
    record: Mapping[str, object],
) -> float:
    """Mutual information `I(state; argmax)` over the late 50% of
    training. `I = H(argmax) − H(argmax | state)` — the share of
    action-distribution structure explained by state-bucket identity.

    Joint reading with marginal `argmax_entropy_late` distinguishes:
      - **State-differentiated policy**: high MI, marginal H is
        whatever the action mix needs to be. Different state regions
        yield different argmaxes — structure is in the conditioning.
      - **Q-flat / noisy policy**: MI ≈ 0, marginal H ≈ log|A|.
        Near-uniform action distribution INDEPENDENT of state.
      - **Collapsed policy**: MI ≈ 0, marginal H ≈ 0. Same action
        everywhere.

    Algorithm-agnostic: any algorithm whose online network emits a
    per-step argmax stream can be analysed with this. The hash
    (env-side) provides the state-bucket; the measurable reads the
    per-step argmax stream that the cell runner already records.

    Plug-in estimator. `MI ≥ 0` holds by construction (chain rule
    on matched supports); the `max(0, ...)` clip absorbs fp rounding
    only. Returns NaN under the same degenerate cases as
    `state_conditional_argmax_entropy_late`, or when the marginal
    action distribution collapses to a single action (MI undefined)."""
    argmax_arr = record.get('online_argmax_per_step')
    state_arr = record.get('state_hash_per_step')
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
    counts_a = np.bincount(a_late)
    nz_a = counts_a[counts_a > 0]
    if nz_a.size <= 1:
        return float('nan')
    p_a = nz_a.astype(np.float64) / float(nz_a.sum())
    h_a = float(-np.sum(p_a * np.log(p_a)))
    h_cond = _weighted_conditional_entropy(a_late, s_late, unique_s)
    if not np.isfinite(h_cond):
        return float('nan')
    # I ≥ 0 in theory; clip to absorb fp-rounding negatives.
    return max(0.0, h_a - h_cond)


@measurable(reads=())
def q_margin_burst_autocorr_per_lag(
    record: Mapping[str, object],
    q_argmax_margin_per_burst: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Per-lag Pearson autocorrelation of per-burst mean action-gap
    (top1 − top2 of Q over actions). Operates on the existing
    `q_argmax_margin_per_burst` array via param-name injection.

    Captures whether the action-gap (i.e. the per-state policy
    decisiveness) is smooth across training (high lag-k autocorr)
    or noisy (decaying). Finer-resolution than `q_burst_autocorr_per_lag`
    on max-Q: the action gap is the actual quantity governing
    argmax stability per state, so vanilla's bias-amplified
    fluctuations might show up here even when per-burst-mean
    max-Q looks smooth."""
    del record
    a = np.asarray(q_argmax_margin_per_burst, dtype=np.float64).flatten()
    n = a.size
    if n < 3:
        return np.array([float('nan')])
    autocorr = np.zeros((n - 1,), dtype=np.float64)
    for k in range(1, n):
        x = a[:-k]
        y = a[k:]
        if x.size < 3 or x.std() < 1e-9 or y.std() < 1e-9:
            autocorr[k - 1] = float('nan')
            continue
        autocorr[k - 1] = float(np.corrcoef(x, y)[0, 1])
    return autocorr


@measurable(reads=())
def q_margin_burst_autocorr_lag1(
    record: Mapping[str, object],
    q_margin_burst_autocorr_per_lag: npt.NDArray[np.float64],
) -> float:
    del record
    a = np.asarray(q_margin_burst_autocorr_per_lag, dtype=np.float64).flatten()
    if a.size < 1 or not np.isfinite(a[0]):
        return float('nan')
    return float(a[0])


@measurable(reads=())
def q_margin_burst_autocorr_long(
    record: Mapping[str, object],
    q_margin_burst_autocorr_per_lag: npt.NDArray[np.float64],
) -> float:
    del record
    a = np.asarray(q_margin_burst_autocorr_per_lag, dtype=np.float64).flatten()
    if a.size < 1:
        return float('nan')
    n = a.size
    tail_start = max(1, int(0.75 * n))
    tail = a[tail_start:]
    finite = tail[np.isfinite(tail)]
    if finite.size == 0:
        return float('nan')
    return float(np.median(finite))


@measurable(reads=())
def q_margin_burst_autocorr_ratio(
    record: Mapping[str, object],
    q_margin_burst_autocorr_per_lag: npt.NDArray[np.float64],
) -> float:
    del record
    a = np.asarray(q_margin_burst_autocorr_per_lag, dtype=np.float64).flatten()
    if a.size < 2 or not np.isfinite(a[0]) or abs(a[0]) <= 1e-9:
        return float('nan')
    n = a.size
    tail_start = max(1, int(0.75 * n))
    tail = a[tail_start:]
    finite = tail[np.isfinite(tail)]
    if finite.size == 0:
        return float('nan')
    return float(np.median(finite)) / float(a[0])


@measurable(reads=())
def q_std_burst_autocorr_per_lag(
    record: Mapping[str, object],
    q_action_std_per_burst: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Per-lag Pearson autocorrelation of per-burst cross-action Q SD
    (i.e., how spread the Q values are across actions at a state,
    averaged per burst). Operates on `q_action_std_per_burst`."""
    del record
    a = np.asarray(q_action_std_per_burst, dtype=np.float64).flatten()
    n = a.size
    if n < 3:
        return np.array([float('nan')])
    autocorr = np.zeros((n - 1,), dtype=np.float64)
    for k in range(1, n):
        x = a[:-k]
        y = a[k:]
        if x.size < 3 or x.std() < 1e-9 or y.std() < 1e-9:
            autocorr[k - 1] = float('nan')
            continue
        autocorr[k - 1] = float(np.corrcoef(x, y)[0, 1])
    return autocorr


@measurable(reads=())
def q_std_burst_autocorr_lag1(
    record: Mapping[str, object],
    q_std_burst_autocorr_per_lag: npt.NDArray[np.float64],
) -> float:
    del record
    a = np.asarray(q_std_burst_autocorr_per_lag, dtype=np.float64).flatten()
    if a.size < 1 or not np.isfinite(a[0]):
        return float('nan')
    return float(a[0])


@measurable(reads=())
def q_std_burst_autocorr_long(
    record: Mapping[str, object],
    q_std_burst_autocorr_per_lag: npt.NDArray[np.float64],
) -> float:
    del record
    a = np.asarray(q_std_burst_autocorr_per_lag, dtype=np.float64).flatten()
    if a.size < 1:
        return float('nan')
    n = a.size
    tail_start = max(1, int(0.75 * n))
    tail = a[tail_start:]
    finite = tail[np.isfinite(tail)]
    if finite.size == 0:
        return float('nan')
    return float(np.median(finite))


def _state_repeat_rate(s_arr: np.ndarray, window: int) -> float:
    """For each step t, was state_hash[t] in the trailing `window`
    steps? Returns fraction of steps that repeat. O(n) via sliding-
    window dict."""
    n = s_arr.size
    if n <= window + 1:
        return float('nan')
    matches = 0
    counts: dict[int, int] = {}
    from collections import deque as _deque
    queue: 'list[int]' = []
    for i in range(n):
        h = int(s_arr[i])
        if h in counts:
            matches += 1
        queue.append(h)
        counts[h] = counts.get(h, 0) + 1
        if len(queue) > window:
            old = queue.pop(0)
            counts[old] -= 1
            if counts[old] == 0:
                del counts[old]
    return matches / n


def _state_repeat_rate_within_episode(
    s_arr: np.ndarray, done_arr: np.ndarray, window: int,
) -> float:
    """Within-episode-only repeat rate. For each step t, the
    trailing-window lookback is BOUNDED by the most recent episode
    start (i.e., the step after the last done before t). Cross-
    episode initial-state matches are excluded.

    Implementation: sliding-window dict that's RESET to empty
    whenever a `done` is encountered. O(n)."""
    n = s_arr.size
    if n != done_arr.size or n <= window + 1:
        return float('nan')
    matches = 0
    eligible = 0
    counts: dict[int, int] = {}
    queue: 'list[int]' = []
    for i in range(n):
        h = int(s_arr[i])
        # Lookback exists only if this step has eligible trailing
        # window (i.e., at least 1 prior within-episode step).
        if queue:
            eligible += 1
            if h in counts:
                matches += 1
        # Update sliding window.
        queue.append(h)
        counts[h] = counts.get(h, 0) + 1
        if len(queue) > window:
            old = queue.pop(0)
            counts[old] -= 1
            if counts[old] == 0:
                del counts[old]
        # If this step ended the episode, reset window — next step
        # is a fresh episode-start with no within-episode lookback.
        if done_arr[i] > 0.5:
            counts.clear()
            queue.clear()
    if eligible == 0:
        return float('nan')
    return matches / eligible


@measurable(reads=('state_hash_per_step', 'done'))
def state_repeat_rate_within_episode_window64_late(
    record: Mapping[str, object],
) -> float:
    """Like `state_repeat_rate_window64_late` but RESTRICTED to
    within-episode repeats — cross-episode initial-state matches
    are excluded.

    Addresses the episode-length-artifact concern: a weak agent
    that dies often might inflate the window-64 repeat rate purely
    via initial-state-after-reset matches, not via actual policy
    cycling. This measurable counts only repeats where the matched
    earlier step is from the SAME episode as the current step
    (no `done` boundary between them).

    If the original `state_repeat_rate_window64_late` arm-difference
    SHRINKS substantially under this measure → the original signal
    was largely the episode-reset artifact.
    If the arm-difference SURVIVES → the loop signal is real
    within-episode cycling."""
    state_arr = record.get('state_hash_per_step')
    done_arr = record.get('done')
    if state_arr is None or done_arr is None:
        return float('nan')
    s = np.asarray(state_arr, dtype=np.int64).flatten()
    d = np.asarray(done_arr, dtype=np.float64).flatten()
    n = min(s.size, d.size)
    if n < 128:
        return float('nan')
    half = n // 2
    return _state_repeat_rate_within_episode(s[half:], d[half:], window=64)


@measurable(reads=('done',))
def episode_count_late(
    record: Mapping[str, object],
) -> float:
    """Number of completed episodes (done==1 transitions) in the
    late 50% of training. Direct measure of episode count — used
    to verify the episode-length-artifact concern."""
    done_arr = record.get('done')
    if done_arr is None:
        return float('nan')
    d = np.asarray(done_arr, dtype=np.float64).flatten()
    n = d.size
    if n < 2:
        return float('nan')
    half = n // 2
    return float((d[half:] > 0.5).sum())


@measurable(reads=('done',))
def mean_episode_length_late(
    record: Mapping[str, object],
) -> float:
    """Mean training-step episode length in late 50%. Computed as
    (late-window length) / (episode count + epsilon) for
    interpretation as steps-per-episode."""
    done_arr = record.get('done')
    if done_arr is None:
        return float('nan')
    d = np.asarray(done_arr, dtype=np.float64).flatten()
    n = d.size
    if n < 2:
        return float('nan')
    half = n // 2
    late = d[half:]
    n_ep = int((late > 0.5).sum())
    if n_ep == 0:
        return float(late.size)
    return float(late.size) / float(n_ep)


@measurable(reads=('state_hash_per_step',))
def state_repeat_rate_window64_late(
    record: Mapping[str, object],
) -> float:
    """Fraction of late-50% steps whose state_hash also appears in
    the trailing 64-step window. Captures short-range state cycling
    (within ~1-2 episode lengths for SI/FR-class envs).

    Tests the "loop-allowing dynamics" hypothesis: vanilla policies
    at γ→1 sparse-reward might cycle through small state subsets
    within episodes (state_hash[t] revisited at t+10, t+20, ...).
    High repeat rate = strong within-window cycling. Low = each
    step visits a "new" state in the recent window.

    Degenerate-state-hash (constant 0) → repeat rate = 1.0
    trivially. Pair with `state_hash_n_unique_late > 1.5` to
    filter."""
    state_arr = record.get('state_hash_per_step')
    if state_arr is None:
        return float('nan')
    s = np.asarray(state_arr, dtype=np.int64).flatten()
    n = s.size
    if n < 128:
        return float('nan')
    half = n // 2
    return _state_repeat_rate(s[half:], window=64)


@measurable(reads=('state_hash_per_step',))
def state_repeat_rate_window256_late(
    record: Mapping[str, object],
) -> float:
    """Like `state_repeat_rate_window64_late` but with 256-step
    window. Captures longer-range cycling (cross-episode revisits
    of similar states)."""
    state_arr = record.get('state_hash_per_step')
    if state_arr is None:
        return float('nan')
    s = np.asarray(state_arr, dtype=np.int64).flatten()
    n = s.size
    if n < 512:
        return float('nan')
    half = n // 2
    return _state_repeat_rate(s[half:], window=256)


@measurable(reads=('online_max_q_per_step', 'mc_return'))
def q_burst_autocorr_per_lag(
    record: Mapping[str, object],
) -> npt.NDArray[np.float64]:
    """Per-lag Pearson autocorrelation of per-burst-mean max-Q.

    Chunks `online_max_q_per_step` into n_bursts equal pieces
    (n_bursts inferred from `mc_return.shape[0]`). Computes mean
    max-Q per chunk → 1-D sequence of length n_bursts. For each
    lag k ∈ {1, ..., n_bursts-1}:

        autocorr[k-1] = Pearson(per_burst_Q[:-k], per_burst_Q[k:])

    Returns `(n_bursts-1,)` array.

    Captures whether Q's trajectory is smooth (high autocorr at
    all lags — trend dominates noise) vs noisy (autocorr decays
    with lag — fluctuations dominate). Companion to
    `state_burst_jaccard_per_lag` on the Q-magnitude side; tests
    whether vanilla's bias-chain produces noisier Q dynamics than
    DDQN's clipped chain.

    Returns array of NaN when inputs missing or n_bursts < 3."""
    q = record.get('online_max_q_per_step')
    mc = record.get('mc_return')
    if q is None or mc is None:
        return np.array([float('nan')])
    q_arr = np.asarray(q, dtype=np.float64).flatten()
    mc_arr = np.asarray(mc, dtype=np.float64)
    if mc_arr.ndim != 2 or mc_arr.shape[0] < 3:
        return np.array([float('nan')])
    n_bursts = int(mc_arr.shape[0])
    if q_arr.size < n_bursts * 2:
        return np.array([float('nan')])
    chunks = np.array_split(q_arr, n_bursts)
    per_burst = np.array([float(c.mean()) for c in chunks])
    autocorr = np.zeros((n_bursts - 1,), dtype=np.float64)
    for k in range(1, n_bursts):
        x = per_burst[:-k]
        y = per_burst[k:]
        if x.size < 3 or x.std() < 1e-9 or y.std() < 1e-9:
            autocorr[k - 1] = float('nan')
            continue
        autocorr[k - 1] = float(np.corrcoef(x, y)[0, 1])
    return autocorr


@measurable(reads=())
def q_burst_autocorr_lag1(
    record: Mapping[str, object],
    q_burst_autocorr_per_lag: npt.NDArray[np.float64],
) -> float:
    """Pearson autocorrelation of per-burst max-Q at lag 1. High =
    Q changes smoothly between adjacent bursts; lower = jumpy."""
    del record
    a = np.asarray(q_burst_autocorr_per_lag, dtype=np.float64).flatten()
    if a.size < 1 or not np.isfinite(a[0]):
        return float('nan')
    return float(a[0])


@measurable(reads=())
def q_burst_autocorr_long(
    record: Mapping[str, object],
    q_burst_autocorr_per_lag: npt.NDArray[np.float64],
) -> float:
    """Median Pearson autocorrelation of per-burst max-Q across the
    longest 25% of lags. High = Q trajectory smooth across the
    whole of training (monotonic trend dominates). Lower = the
    long-lag correlation has decayed (noise dominates)."""
    del record
    a = np.asarray(q_burst_autocorr_per_lag, dtype=np.float64).flatten()
    if a.size < 1:
        return float('nan')
    n = a.size
    tail_start = max(1, int(0.75 * n))
    tail = a[tail_start:]
    finite = tail[np.isfinite(tail)]
    if finite.size == 0:
        return float('nan')
    return float(np.median(finite))


@measurable(reads=())
def q_burst_autocorr_ratio(
    record: Mapping[str, object],
    q_burst_autocorr_per_lag: npt.NDArray[np.float64],
) -> float:
    """`q_burst_autocorr_long / q_burst_autocorr_lag1` — Q-side
    smoothness-vs-noise indicator. Near 1 = autocorr retained at
    all lags (smooth Q trajectory). Near 0 = autocorr decays with
    lag (noisy Q dynamics around trend)."""
    del record
    a = np.asarray(q_burst_autocorr_per_lag, dtype=np.float64).flatten()
    if a.size < 2 or not np.isfinite(a[0]) or abs(a[0]) <= 1e-9:
        return float('nan')
    n = a.size
    tail_start = max(1, int(0.75 * n))
    tail = a[tail_start:]
    finite = tail[np.isfinite(tail)]
    if finite.size == 0:
        return float('nan')
    long_val = float(np.median(finite))
    return long_val / float(a[0])


@measurable(reads=('state_hash_per_step', 'mc_return'))
def state_burst_jaccard_per_lag(
    record: Mapping[str, object],
) -> npt.NDArray[np.float64]:
    """Per-lag Jaccard autocorrelation of per-burst state-hash sets.

    For each burst t, compute `S_t` = set of unique `state_hash_per_step`
    values visited during burst t (one chunk of size n_steps/n_bursts).
    Then for each lag k ∈ {1, ..., n_bursts−1}, compute the mean
    Jaccard similarity `|S_t ∩ S_{t+k}| / |S_t ∪ S_{t+k}|` averaged
    over all valid t.

    Returns: `(n_bursts−1,)` array. Element `[k−1]` is the mean
    Jaccard at lag k.

    This is the categorical analog of `q_autocorr_per_burst` —
    captures whether the policy cycles through similar state sets
    over time (high Jaccard at large lag = trajectory rut) or moves
    through evolving state distributions (decaying Jaccard with lag
    = preserved/expanding exploration). No arbitrary early/late
    split: uses ALL pairs (t, t+k) weighted naturally by lag.

    NaN when the inputs are missing, n_bursts < 3, or state_hash is
    degenerate (constant 0)."""
    state_arr = record.get('state_hash_per_step')
    mc = record.get('mc_return')
    if state_arr is None or mc is None:
        return np.array([float('nan')])
    s = np.asarray(state_arr, dtype=np.int64).flatten()
    mc_arr = np.asarray(mc, dtype=np.float64)
    if mc_arr.ndim != 2 or mc_arr.shape[0] < 3:
        return np.array([float('nan')])
    n_bursts = int(mc_arr.shape[0])
    if s.size < n_bursts * 2:
        return np.array([float('nan')])
    chunks = np.array_split(s, n_bursts)
    sets_per_burst: list[set[int]] = [set(c.tolist()) for c in chunks]
    # Degenerate-hash short circuit.
    all_hashes: set[int] = set()
    for ss in sets_per_burst:
        all_hashes |= ss
    if len(all_hashes) < 2:
        return np.array([float('nan')])
    jaccard_at_lag = np.zeros((n_bursts - 1,), dtype=np.float64)
    for k in range(1, n_bursts):
        vals: list[float] = []
        for t in range(n_bursts - k):
            S1, S2 = sets_per_burst[t], sets_per_burst[t + k]
            union = S1 | S2
            if not union:
                continue
            intersect = S1 & S2
            vals.append(len(intersect) / len(union))
        jaccard_at_lag[k - 1] = float(np.mean(vals)) if vals else float('nan')
    return jaccard_at_lag


@measurable(reads=())
def state_burst_jaccard_lag1(
    record: Mapping[str, object],
    state_burst_jaccard_per_lag: npt.NDArray[np.float64],
) -> float:
    """Mean Jaccard between consecutive bursts (lag = 1). Both arms
    typically high (same policy generates similar state distributions
    one burst apart). Reads `state_burst_jaccard_per_lag` via
    framework param-name injection."""
    del record
    a = np.asarray(state_burst_jaccard_per_lag, dtype=np.float64).flatten()
    if a.size < 1 or not np.isfinite(a[0]):
        return float('nan')
    return float(a[0])


@measurable(reads=())
def state_burst_jaccard_long(
    record: Mapping[str, object],
    state_burst_jaccard_per_lag: npt.NDArray[np.float64],
) -> float:
    """Median Jaccard across the longest 25% of lags. High = trajectory
    rut (policy cycles through same states across all of training).
    Low = drift (policy moves through different distributions over
    time). Median of the long-lag region, not the single max-lag
    pair (which is noisy with only 1 pair contributing)."""
    del record
    a = np.asarray(state_burst_jaccard_per_lag, dtype=np.float64).flatten()
    if a.size < 1:
        return float('nan')
    n = a.size
    tail_start = max(1, int(0.75 * n))
    tail = a[tail_start:]
    finite = tail[np.isfinite(tail)]
    if finite.size == 0:
        return float('nan')
    return float(np.median(finite))


@measurable(reads=())
def state_burst_jaccard_ratio(
    record: Mapping[str, object],
    state_burst_jaccard_per_lag: npt.NDArray[np.float64],
) -> float:
    """`state_burst_jaccard_long / state_burst_jaccard_lag1` — the
    direct rut-vs-drift indicator. Near 1 = state sets stay similar
    across all lags (trajectory rut, no drift). Near 0 = state sets
    decorrelate at long lag (policy moves through state space)."""
    del record
    a = np.asarray(state_burst_jaccard_per_lag, dtype=np.float64).flatten()
    if a.size < 2 or not np.isfinite(a[0]) or a[0] <= 1e-9:
        return float('nan')
    n = a.size
    tail_start = max(1, int(0.75 * n))
    tail = a[tail_start:]
    finite = tail[np.isfinite(tail)]
    if finite.size == 0:
        return float('nan')
    long_val = float(np.median(finite))
    return long_val / float(a[0])


@measurable(reads=('state_hash_per_step',))
def state_hash_entropy_early(
    record: Mapping[str, object],
) -> float:
    """Shannon entropy (nats) of `state_hash_per_step` over the
    EARLY 50% of training — symmetric counterpart to
    `state_hash_entropy_late`.

    In the early window the behavior policy is dominated by
    ε-greedy exploration (ε still high). Early state-distribution
    therefore reflects ε-random + env-dynamics + initial-Q
    argmax, with the LEARNED-POLICY contribution still small.

    Used to test whether arm-induced differences in late-window
    state diversity are UPSTREAM (visible early too → DDQN affects
    behavior from the start) or DOWNSTREAM (only diverge late →
    state diversity is a manifestation of late-window policy
    quality).

    Same degenerate-state-hash caveat as `state_hash_entropy_late`."""
    state_arr = record.get('state_hash_per_step')
    if state_arr is None:
        return float('nan')
    s = np.asarray(state_arr, dtype=np.int64).flatten()
    n = s.size
    if n < 4:
        return float('nan')
    half = n // 2
    s_early = s[:half]
    counts = np.bincount(s_early - s_early.min())
    nonzero = counts[counts > 0]
    if nonzero.size <= 1:
        return 0.0
    p = nonzero.astype(np.float64) / float(nonzero.sum())
    return float(-np.sum(p * np.log(p)))


@measurable(reads=('state_hash_per_step',))
def state_hash_n_unique_early(
    record: Mapping[str, object],
) -> float:
    """Number of distinct `state_hash_per_step` values in the early
    50% window. Counterpart to `state_hash_n_unique_late`."""
    state_arr = record.get('state_hash_per_step')
    if state_arr is None:
        return float('nan')
    s = np.asarray(state_arr, dtype=np.int64).flatten()
    n = s.size
    if n < 4:
        return float('nan')
    half = n // 2
    return float(np.unique(s[:half]).size)


@measurable(reads=('online_argmax_per_step', 'state_hash_per_step'))
def policy_churn_early(
    record: Mapping[str, object],
) -> float:
    """State-conditional policy churn over the EARLY 50% of training.

    Symmetric counterpart to `policy_churn_late`. In the early window
    the behavior policy is ε-greedy-dominated, so consecutive
    same-state appearances are partly stochastic-ε actions — the
    early churn rate carries less "policy flux" signal than the
    late form and more "ε-random + early-Q-instability" noise.

    Used as a baseline: if late churn differs between arms but early
    churn doesn't, the arm-induced late churn is downstream of
    learning. If early churn already differs, the clip is changing
    behavior from the start."""
    argmax_arr = record.get('online_argmax_per_step')
    state_arr = record.get('state_hash_per_step')
    if argmax_arr is None or state_arr is None:
        return float('nan')
    a = np.asarray(argmax_arr, dtype=np.int64).flatten()
    s = np.asarray(state_arr, dtype=np.int64).flatten()
    n = min(a.size, s.size)
    if n < 4:
        return float('nan')
    half = n // 2
    a_early, s_early = a[:half], s[:half]
    n_flips = 0
    n_pairs = 0
    for s_val in np.unique(s_early):
        mask = s_early == s_val
        if int(mask.sum()) < 2:
            continue
        a_in_s = a_early[mask]
        flips = a_in_s[1:] != a_in_s[:-1]
        n_flips += int(flips.sum())
        n_pairs += int(flips.size)
    if n_pairs == 0:
        return float('nan')
    return float(n_flips) / float(n_pairs)


@measurable(reads=('state_hash_per_step',))
def state_hash_entropy_late(
    record: Mapping[str, object],
) -> float:
    """Shannon entropy (nats) of the `state_hash_per_step` distribution
    over the late 50% of training.

    Diagnostic of state-visitation diversity. High entropy = the
    policy visits many distinct state-hash buckets in roughly even
    proportions. Low entropy = the policy concentrates on a small
    set of states.

    Used to resolve the churn-Finding interpretive ambiguity (see
    `finding_ddqn_reduces_policy_churn` docstring): DDQN's higher
    `policy_churn_late` at γ→1 sparse-reward could be (a) wider state
    distribution causing inter-state argmax differences to register
    as "flips" at the same hash bucket, OR (b) true policy flux at
    the same state across training updates. State-visitation entropy
    isolates (a): if DDQN > vanilla entropy, some of the higher
    churn is state-distribution drift, not pure policy churn.

    Caveat: degenerates to 0 when `state_hash` is the constant-0
    default (`default_state_hash`). Pair with
    `state_hash_n_unique_late > 1.5` scope predicate to filter out
    the degenerate case."""
    state_arr = record.get('state_hash_per_step')
    if state_arr is None:
        return float('nan')
    s = np.asarray(state_arr, dtype=np.int64).flatten()
    n = s.size
    if n < 4:
        return float('nan')
    half = n // 2
    s_late = s[half:]
    counts = np.bincount(s_late - s_late.min())
    nonzero = counts[counts > 0]
    if nonzero.size <= 1:
        return 0.0
    p = nonzero.astype(np.float64) / float(nonzero.sum())
    return float(-np.sum(p * np.log(p)))


@measurable(reads=('state_hash_per_step',))
def state_hash_n_unique_late(
    record: Mapping[str, object],
) -> float:
    """Number of distinct `state_hash_per_step` values observed in
    the late 50% of training. Diagnostic for whether the env's
    registered `state_hash` is non-degenerate.

    1.0 = the substrate's `default_state_hash` returned 0 for every
    step (env catalogue did not register a real hash). Any
    state-conditional measurable evaluated on this cell degenerates
    to a global / no-conditioning form.

    > 1 = the env has a meaningful state_hash; state-conditional
    measurables compute their intended quantity.

    Returns NaN if the column is missing. Returns int-as-float; the
    `>1.5` scope predicate that guards Schaul-aligned bridges
    catches the constant-0 case (1.0) and admits the meaningful
    case (≥ 2)."""
    state_arr = record.get('state_hash_per_step')
    if state_arr is None:
        return float('nan')
    s = np.asarray(state_arr, dtype=np.int64).flatten()
    n = s.size
    if n < 4:
        return float('nan')
    half = n // 2
    return float(np.unique(s[half:]).size)


# Per-burst siblings of the state-hash and state-repeat-rate late
# measurables. Authored via `@temporal_reduction` (same window-
# reduction kernel, two windowing strategies). The late-half versions
# above stay as authored — only the per-burst sibling is registered
# here, leaving the existing scalar measurables and their docstrings
# untouched.


def _window_n_unique(window: npt.NDArray[np.float64]) -> float:
    """Count of distinct state-hash values in the window."""
    if window.size < 2:
        return float('nan')
    return float(np.unique(window.astype(np.int64)).size)


def _window_shannon_entropy(window: npt.NDArray[np.float64]) -> float:
    """Shannon entropy (nats) of state-hash distribution in window."""
    if window.size < 2:
        return float('nan')
    s = window.astype(np.int64)
    counts = np.bincount(s - s.min())
    counts = counts[counts > 0]
    if counts.size <= 1:
        return 0.0
    p = counts.astype(np.float64) / float(counts.sum())
    return float(-np.sum(p * np.log(p)))


def _window_state_repeat_rate_64(window: npt.NDArray[np.float64]) -> float:
    """Fraction of steps whose state appears in the trailing 64-step
    sub-window — same kernel as `_state_repeat_rate(s, window=64)`
    but applied to the supplied window (not the late half)."""
    if window.size < 128:
        return float('nan')
    return _state_repeat_rate(window.astype(np.int64), window=64)


temporal_reduction(
    reads=('state_hash_per_step',),
    per_burst_name='state_hash_n_unique_per_burst',
)(_window_n_unique)

temporal_reduction(
    reads=('state_hash_per_step',),
    per_burst_name='state_hash_entropy_per_burst',
)(_window_shannon_entropy)

temporal_reduction(
    reads=('state_hash_per_step',),
    per_burst_name='state_repeat_rate_window64_per_burst',
)(_window_state_repeat_rate_64)


@measurable(reads=('online_argmax_per_step', 'state_hash_per_step'))
def policy_churn_late(
    record: Mapping[str, object],
) -> float:
    """State-conditional policy churn over the late 50% of training,
    in the form of Schaul et al. 2022 "The Phenomenon of Policy Churn"
    (NeurIPS, arXiv:2206.00730).

    For each state that appears at least twice in the late window,
    walks the time-ordered argmax sequence at that state and counts
    consecutive-pair flips (`argmax[i] != argmax[i-1]`). Pools over
    all (state, consecutive-pair) tuples weighted by occurrence
    count. Returned value is the empirical fraction of consecutive
    state-revisit pairs where the online network's greedy choice
    changed.

    Schaul's exact form is `W(π_t, π_{t+1}|s) = ½ Σ_a |π_t(a|s) −
    π_{t+1}(a|s)|` evaluated on a fixed eval set between consecutive
    policy snapshots; for deterministic policy (DQN's greedy), W
    reduces to `1[argmax flipped]`. Our trace stream is the natural-
    rollout per-step argmax, so consecutive APPEARANCES of the same
    state-hash serve as the consecutive-snapshot sample pair (the
    policy DID advance between those two appearances because steps
    elapsed). The proxy is exact when the state recurs and the
    policy's argmax at that state is the only thing that changed
    between appearances; it conflates with batch-composition drift
    when the appearances are far apart.

    Range: [0, 1]. Higher → more churn (policy thrashes between
    revisits on the same state). 0 → policy fully stable at each
    revisited state.

    Complementary to `state_conditional_argmax_entropy_late`
    (static argmax distribution at each state — does the policy
    commit to one action there?). A policy with full commitment
    (entropy 0) has churn 0; a noisy/exploration policy has both
    entropy > 0 AND churn > 0; a drift policy can have entropy >
    0 but churn near 0 (monotonic argmax sequence: aaaabbbb → 1
    flip out of 7 pairs).

    Lit positioning: see THEORY_bootstrap_dominance.md §11. This is
    the direct sibling for Schaul's `W(π,π')` on our existing
    trace shape — no substrate change required, only the per-step
    argmax + state-hash columns that are already standard.

    **Caveat — degenerate `state_hash` envs.** When an env's
    `state_hash` is `default_state_hash` (returns 0 for every obs),
    `np.unique(s_late)` collapses to `[0]` and this measurable
    DEGENERATES to a GLOBAL consecutive-step argmax-flip rate.
    Envs in `env_catalogue.py` with `state_hash=None` or no
    `state_hash=` kwarg fall into this case (FourRooms-misc,
    MetaMaze-misc, image-obs envs without registered hashes).
    Envs WITH explicit hashes — Asterix, Breakout, Freeway,
    SpaceInvaders (MinAtar), CartPole, Acrobot, MountainCar —
    return strict state-conditional churn. Check the env's
    catalogue entry before treating this measurable's value as
    Schaul-aligned. Schaul's published quantity requires a
    non-trivial state hash."""
    argmax_arr = record.get('online_argmax_per_step')
    state_arr = record.get('state_hash_per_step')
    if argmax_arr is None or state_arr is None:
        return float('nan')
    a = np.asarray(argmax_arr, dtype=np.int64).flatten()
    s = np.asarray(state_arr, dtype=np.int64).flatten()
    n = min(a.size, s.size)
    if n < 4:
        return float('nan')
    half = n // 2
    a_late, s_late = a[half:], s[half:]
    n_flips = 0
    n_pairs = 0
    for s_val in np.unique(s_late):
        mask = s_late == s_val
        if int(mask.sum()) < 2:
            continue
        a_in_s = a_late[mask]
        flips = a_in_s[1:] != a_in_s[:-1]
        n_flips += int(flips.sum())
        n_pairs += int(flips.size)
    if n_pairs == 0:
        return float('nan')
    return float(n_flips) / float(n_pairs)


@temporal_reduction(
    reads=('online_argmax_per_step',),
    late_name='argmax_entropy_late',
)
def _argmax_entropy_window(window: npt.NDArray[np.float64]) -> float:
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

    Late-only: the per-burst sibling `argmax_entropy_per_burst`
    is authored separately because it reads `n_actions` to
    `minlength`-pad the bincount — the trailing zeros are
    entropy-neutral but the explicit dependency is preserved
    in the existing measurable.

    Returns nan if `online_argmax_per_step` is missing or empty."""
    if window.size == 0:
        return float('nan')
    late = window.astype(np.int64)
    counts = np.bincount(late)
    p = counts.astype(np.float64) / float(late.size)
    p_pos = p[p > 0]
    return float(-(p_pos * np.log(p_pos)).sum())


argmax_entropy_late = _registered('argmax_entropy_late')


@temporal_reduction(
    reads=('online_argmax_per_step',),
    late_name='argmax_mode_freq_late',
)
def _argmax_mode_freq_window(window: npt.NDArray[np.float64]) -> float:
    """Fraction of late-training steps where `online_argmax_
    per_step` equals the mode action.

    Range [1/|A|, 1.0]. 1.0 = always picks the same action
    (fully committed). 1/|A| = uniform across actions
    (fully indecisive). Companion to `argmax_entropy_late`;
    different scaling, same underlying signal.

    Per-cell measurable. Bridge body computes paired
    DDQN_mode_freq − vanilla_mode_freq."""
    if window.size == 0:
        return float('nan')
    late = window.astype(np.int64)
    counts = np.bincount(late)
    return float(counts.max()) / float(late.size)


argmax_mode_freq_late = _registered('argmax_mode_freq_late')


@temporal_reduction(
    reads=('online_std_q_per_step',),
    late_name='q_action_std_late',
    per_burst_name='q_action_std_per_burst',
)
def _q_action_std_window_mean(window: npt.NDArray[np.float64]) -> float:
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

    The per-burst sibling `q_action_std_per_burst` returns the
    same statistic computed within each eval-burst training-step
    window — the per-burst granularity for panel bridges that
    cannot collapse training-time phase structure into a single
    late-half scalar."""
    if window.size == 0:
        return float('nan')
    return float(window.mean())


q_action_std_late = _registered('q_action_std_late')


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


@measurable(
    reads=('online_std_q_per_step', 'online_top12_margin_per_step',
           'eval_step_index', 'n_actions'),
)
def q_lambda_a_per_burst(
    record: Mapping[str, object],
) -> npt.NDArray[np.float64]:
    """Per-burst $\\Lambda_a^{\\mathrm{cell}}$ trajectory under the
    Theorem 3 (THEORY note §6.1) operational definition:
    $\\Lambda_a^{\\mathrm{burst}} = \\sigma_{\\mathrm{action}}^{\\mathrm{burst}}
     \\cdot \\sqrt{2 \\ln K} / \\Delta^{\\mathrm{burst}}$, where the
    burst-window averaged σ_action and top12-margin replace the
    late-window aggregates of `q_action_std_late` /
    `q_argmax_margin_late`.

    Returns `(n_bursts,)`. Burst boundaries split the per-step
    trace into `n_bursts` equal windows (same convention as
    `q_action_std_per_burst` / `q_argmax_margin_per_burst`),
    where `n_bursts = len(eval_step_index)`.

    Use cases:
    - (A4'a) magnitude-alignment test: CV of the converged-tail
      window (last 20% of bursts) bounds σ_clip variation across
      "one-step from converged" iterates. Operationalised by
      `q_lambda_a_tail_cv` below.
    - Geometric-series argmax-accumulation gap visualisation:
      `q_lambda_a_growth_ratio = tail_mean / init_mean` quantifies
      how far the converged σ_Λa drifts from the early-burst
      ("one-step from init") value — the open limitation
      acknowledged in v9's Status section.

    NaN when any trace input is missing or the burst window is
    empty."""
    try:
        sigma = ONLINE_STD_Q(record)
        margin = ONLINE_TOP12_MARGIN(record)
        eval_idx = EVAL_STEP_INDEX(record)
    except KeyError:
        return np.zeros((0,), dtype=np.float64)
    n_actions = record.get('n_actions')
    if not isinstance(n_actions, (int, float)) or n_actions <= 1:
        return np.zeros((0,), dtype=np.float64)
    n = int(sigma.shape[0])
    n_bursts = int(eval_idx.shape[0])
    if n == 0 or n_bursts == 0:
        return np.zeros((0,), dtype=np.float64)
    if margin.shape[0] != n:
        return np.zeros((0,), dtype=np.float64)
    edges = np.linspace(0, n, n_bursts + 1, dtype=np.int64)
    coeff = math.sqrt(2.0 * math.log(float(n_actions)))
    out = np.empty(n_bursts, dtype=np.float64)
    for i in range(n_bursts):
        s = float(sigma[edges[i]:edges[i+1]].mean())
        m = float(margin[edges[i]:edges[i+1]].mean())
        out[i] = s * coeff / m if m > 1e-9 else float('nan')
    return out


@measurable(reads=())
def q_lambda_a_tail_mean(
    record: Mapping[str, object],
    q_lambda_a_per_burst: npt.NDArray[np.float64],
) -> float:
    """Mean of `q_lambda_a_per_burst` over the converged tail
    (last 20% of bursts). The operational σ_Λa^env^cell that
    Theorem 3's empirical signature consumes; pooled cross-seed
    gives σ_Λa^env."""
    arr = np.asarray(q_lambda_a_per_burst, dtype=np.float64)
    if arr.size == 0:
        return float('nan')
    lo = int(arr.size * 0.8)
    if lo >= arr.size:
        return float('nan')
    tail = arr[lo:]
    finite = tail[np.isfinite(tail)]
    if finite.size == 0:
        return float('nan')
    return float(finite.mean())


@measurable(reads=())
def q_lambda_a_tail_cv(
    record: Mapping[str, object],
    q_lambda_a_per_burst: npt.NDArray[np.float64],
) -> float:
    """Coefficient of variation of `q_lambda_a_per_burst` over the
    converged tail (last 20% of bursts). Operationalises the
    (A4'a) magnitude-alignment test from Theorem 3 v9: under
    (A4'a), σ_clip is order-of-magnitude aligned across
    converged-iterate one-step bootstraps; CV in the tail bounds
    that drift. Cell-level CV < 0.2 → tail-stable. NaN propagates."""
    arr = np.asarray(q_lambda_a_per_burst, dtype=np.float64)
    if arr.size == 0:
        return float('nan')
    lo = int(arr.size * 0.8)
    if lo >= arr.size:
        return float('nan')
    tail = arr[lo:]
    finite = tail[np.isfinite(tail)]
    if finite.size < 2:
        return float('nan')
    mean = float(finite.mean())
    if abs(mean) < 1e-12:
        return float('nan')
    return float(finite.std(ddof=1) / abs(mean))


@measurable(reads=())
def q_lambda_a_init_mean(
    record: Mapping[str, object],
    q_lambda_a_per_burst: npt.NDArray[np.float64],
) -> float:
    """Mean of `q_lambda_a_per_burst` over the init window (first
    10% of bursts). Reference quantity for the geometric-series
    growth ratio."""
    arr = np.asarray(q_lambda_a_per_burst, dtype=np.float64)
    if arr.size == 0:
        return float('nan')
    hi = max(1, int(arr.size * 0.1))
    init = arr[:hi]
    finite = init[np.isfinite(init)]
    if finite.size == 0:
        return float('nan')
    return float(finite.mean())


@measurable(reads=())
def q_lambda_a_growth_ratio(
    record: Mapping[str, object],
    q_lambda_a_tail_mean: float,
    q_lambda_a_init_mean: float,
) -> float:
    """`q_lambda_a_tail_mean / q_lambda_a_init_mean` — quantifies
    the geometric-series argmax-accumulation gap (THEORY §6.1
    open limitation, parallel to §9.3's Robbins-Monro gap for
    Theorem 1). Larger ratio → σ_Λa drifted more during training;
    the converged signature accumulates farther from the one-step
    bootstrap from init. NaN when init is degenerate."""
    if not math.isfinite(q_lambda_a_init_mean) or abs(q_lambda_a_init_mean) < 1e-9:
        return float('nan')
    if not math.isfinite(q_lambda_a_tail_mean):
        return float('nan')
    return float(q_lambda_a_tail_mean / q_lambda_a_init_mean)


@measurable(reads=('gamma', 'eval_step_index'))
def q_lambda_a_horizon_normalised_per_burst(
    record: Mapping[str, object],
    q_lambda_a_per_burst: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Per-burst $\\Lambda_a^{\\mathrm{cell}}$ divided by the
    effective-horizon factor $(1 - \\gamma^t) / (1 - \\gamma)$
    at the burst's training-step count $t$.

    Under Lemma 2's Bellman bias accumulation, σ_aniso[t] ≈
    σ_aniso[∞] · (1 - γ^t). Dividing the per-burst trajectory by
    `(1 - γ^t)/(1-γ)` (the effective horizon at step $t$ for a
    γ-discounted geometric series) gives an "asymptote-normalised"
    trajectory that should equal σ_aniso[∞]/(1-γ) at every burst
    if the geometric-series accumulation captures the trajectory
    shape. The growth_ratio of the normalised trajectory then
    reduces from ~2.4× (raw) to ~1× if the gap is purely
    geometric-series scaling, OR stays ~2.4× if there's residual
    non-Bellman structure (FA dynamics, replay-as-prior etc.).

    Empirical test of the geometric-series gap's structural source."""
    gamma = record.get('gamma')
    if not isinstance(gamma, (int, float)) or gamma >= 1.0 or gamma < 0.0:
        return np.zeros((0,), dtype=np.float64)
    g = float(gamma)
    arr = np.asarray(q_lambda_a_per_burst, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0,), dtype=np.float64)
    try:
        eval_idx = EVAL_STEP_INDEX(record)
    except KeyError:
        return np.zeros((0,), dtype=np.float64)
    if eval_idx.size == 0 or eval_idx.size != arr.size:
        return np.zeros((0,), dtype=np.float64)
    steps = np.asarray(eval_idx, dtype=np.float64)
    eff_horizon = (1.0 - np.power(g, steps)) / (1.0 - g)
    eff_horizon = np.where(eff_horizon > 1e-9, eff_horizon, np.nan)
    return arr / eff_horizon


@measurable(
    reads=('online_std_q_per_step', 'online_top12_margin_per_step',
           'n_actions'),
)
def q_lambda_a_early_window_mean(
    record: Mapping[str, object],
) -> float:
    """Λ_a^cell averaged over the very early training window
    (steps 100-1000) — captures the transient regime where
    Bellman bias is still accumulating from initialisation.

    Sub-burst granularity needed: at γ=0.95 Bellman bias saturates
    by step ~90; at γ=0.99 by step ~460; at γ=0.999 by step ~4600.
    The first burst-window (steps 0-20000) is past saturation for
    all γ ≤ 0.999, so per-burst measurements miss the transient.

    Combined with `q_lambda_a_tail_mean`, the ratio
    `tail_mean / early_window_mean` gives a finer-grained test of
    whether the geometric-series accumulation matches Bellman's
    `1 / (1 − γ^t)` prediction or has residual NN-side structure."""
    try:
        sigma = ONLINE_STD_Q(record)
        margin = ONLINE_TOP12_MARGIN(record)
    except KeyError:
        return float('nan')
    n_actions = record.get('n_actions')
    if not isinstance(n_actions, (int, float)) or n_actions <= 1:
        return float('nan')
    if sigma.size < 1000 or margin.size < 1000:
        return float('nan')
    lo, hi = 100, 1000
    s = float(sigma[lo:hi].mean())
    m = float(margin[lo:hi].mean())
    if m <= 1e-9:
        return float('nan')
    return s * math.sqrt(2.0 * math.log(float(n_actions))) / m


@measurable(reads=('gamma',))
def q_lambda_a_bellman_growth_predicted(
    record: Mapping[str, object],
) -> float:
    """Bellman's predicted growth ratio `1 / (1 − γ^N)` at the
    early-window midpoint (N=550 steps, midpoint of 100-1000).

    Closed-form prediction: if the geometric-series gap is pure
    Bellman bias accumulation, then σ_aniso[t] ≈ σ_aniso[∞]·(1 − γ^t),
    so the growth from early (t=550) to converged (t→∞) is
    `1 / (1 − γ^550)`. This measurable gives the theoretical
    prediction per cell; pairing with the empirical
    `q_lambda_a_early_to_tail_ratio` (next) tests it."""
    gamma = record.get('gamma')
    if not isinstance(gamma, (int, float)) or gamma >= 1.0 or gamma < 0.0:
        return float('nan')
    g = float(gamma)
    denom = 1.0 - math.pow(g, 550)
    if denom <= 1e-9:
        return float('nan')
    return 1.0 / denom


@measurable(reads=())
def q_lambda_a_early_to_tail_ratio(
    record: Mapping[str, object],
    q_lambda_a_tail_mean: float,
    q_lambda_a_early_window_mean: float,
) -> float:
    """Empirical growth ratio from early-window (steps 100-1000)
    to converged-tail (last 20% of bursts). Compared against
    `q_lambda_a_bellman_growth_predicted` — if the ratio matches
    `1/(1−γ^550)`, the gap is pure Bellman; if the empirical ratio
    is systematically larger (or differently γ-scaled), residual
    NN training dynamics drive the gap."""
    if not math.isfinite(q_lambda_a_tail_mean):
        return float('nan')
    if not math.isfinite(q_lambda_a_early_window_mean):
        return float('nan')
    if abs(q_lambda_a_early_window_mean) < 1e-9:
        return float('nan')
    return q_lambda_a_tail_mean / q_lambda_a_early_window_mean


@measurable(reads=())
def q_lambda_a_horizon_normalised_growth_ratio(
    record: Mapping[str, object],
    q_lambda_a_horizon_normalised_per_burst: npt.NDArray[np.float64],
) -> float:
    """Growth ratio of the horizon-normalised per-burst Λ_a
    (tail mean / init mean). Tests whether the geometric-series
    gap is purely Bellman-accumulation scaling (ratio → 1 after
    normalisation) or has residual non-geometric structure (ratio
    stays > 1)."""
    arr = np.asarray(q_lambda_a_horizon_normalised_per_burst, dtype=np.float64)
    if arr.size == 0:
        return float('nan')
    lo_init = max(1, int(arr.size * 0.1))
    lo_tail = int(arr.size * 0.8)
    if lo_tail >= arr.size or lo_init > lo_tail:
        return float('nan')
    init = arr[:lo_init]
    tail = arr[lo_tail:]
    init_finite = init[np.isfinite(init)]
    tail_finite = tail[np.isfinite(tail)]
    if init_finite.size == 0 or tail_finite.size == 0:
        return float('nan')
    init_mean = float(init_finite.mean())
    tail_mean = float(tail_finite.mean())
    if abs(init_mean) < 1e-9:
        return float('nan')
    return tail_mean / init_mean


@temporal_reduction(
    reads=('online_top12_margin_per_step',),
    late_name='q_argmax_margin_late',
    per_burst_name='q_argmax_margin_per_burst',
)
def _q_argmax_margin_window_mean(window: npt.NDArray[np.float64]) -> float:
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

    The per-burst sibling `q_argmax_margin_per_burst` is used to
    test whether action-margin mediates the per-burst Q-channel
    within canonical-config scope (where cell-level margin
    mediation drops to ~6%). See
    `findings_two_channel_cross_corpus.md`."""
    if window.size == 0:
        return float('nan')
    return float(window.mean())


q_argmax_margin_late = _registered('q_argmax_margin_late')


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


@measurable(reads=('arm_key',))
def arm_is_baseline(record: Mapping[str, object]) -> float:
    """Numeric indicator for baseline arm membership.

    Returns 1.0 if `arm_key == "baseline"`, 0.0 otherwise. Used
    as the binary `x` in per-cell partial-Spearman bridges that
    test whether mediators absorb the arm → outcome effect.
    Encoding arm as a continuous {0, 1} variable makes Spearman
    ρ equivalent to a Wilcoxon-Mann-Whitney rank-sum statistic
    (Hollander & Wolfe 1973), so the resulting partial-ρ is a
    monotone-equivalent of the partial Wilcoxon-Mann-Whitney
    framework's nonparametric mediation effect.

    Returns nan if `arm_key` is absent or non-string."""
    arm = record.get('arm_key')
    if not isinstance(arm, str):
        return float('nan')
    return 1.0 if arm == 'baseline' else 0.0


@measurable(reads=(
    'n_actions', 'action_duplicate_k',
    'q_action_std_late', 'q_argmax_margin_late',
))
def lambda_a_late(record: Mapping[str, object]) -> float:
    """Per-cell Λ_a — the bias-asymmetry index
    `σ_clip · √(2 ln K_eff) / Δ_v` (Cor 3.2,
    THEORY_bootstrap_dominance v3).

    Under (A2) iid Gaussian and (A3) iid sampling, Theorem 3 says
    the agent's argmax is preserved iff
    `γ · σ_clip · √(2 ln K) < Δ_v`. This per-cell quantity is the
    structural predictor of how DDQN's bootstrap clip interacts
    with bias asymmetry across actions: small Λ_a → bias dominated
    by mean (Type B, asymmetric — DDQN's clip helps); large Λ_a →
    bias variance dominates (Type A, uniform-across-actions —
    DDQN's clip corrupts argmax).

    K_eff = n_actions × max(1, action_duplicate_k).

    Reads `q_action_std_late` (σ_clip proxy) and
    `q_argmax_margin_late` (Δ_v proxy) from the record's cached
    scalar columns rather than via the injection resolver — the
    resolver would recompute them from `online_std_q_per_step`
    traces, which are evicted post-ingest on most corpora.
    Reading the cached scalars lets Λ_a populate from cache alone
    once σ_clip + Δ_v are present.

    Returns nan when any input is non-finite, q_argmax_margin_late
    ≈ 0, or K_eff < 2."""
    n_actions_obj = record.get('n_actions')
    if not isinstance(n_actions_obj, (int, float)) or n_actions_obj < 2:
        return float('nan')
    adk_obj = record.get('action_duplicate_k')
    if (
        isinstance(adk_obj, (int, float))
        and math.isfinite(adk_obj)
        and adk_obj >= 1
    ):
        k_dup = int(adk_obj)
    else:
        k_dup = 1
    k_eff = int(n_actions_obj) * k_dup
    if k_eff < 2:
        return float('nan')
    sigma_clip_obj = record.get('q_action_std_late')
    delta_v_obj = record.get('q_argmax_margin_late')
    if not (
        isinstance(sigma_clip_obj, (int, float))
        and isinstance(delta_v_obj, (int, float))
    ):
        return float('nan')
    sigma_clip = float(sigma_clip_obj)
    delta_v = float(delta_v_obj)
    if not (math.isfinite(sigma_clip) and math.isfinite(delta_v)):
        return float('nan')
    if abs(delta_v) < 1e-9:
        return float('nan')
    return sigma_clip * math.sqrt(2.0 * math.log(float(k_eff))) / delta_v


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


@measurable(reads=())
def sigma_over_jens_late(
    record: Mapping[str, object],
    q_action_std_late: float,  # injected
    jensen_gap: float,         # injected
) -> float:
    """`q_action_std_late / jensen_gap` — per-cell σ/jens ratio,
    the regime-discriminator covariate at γ→1.

    High σ/jens: bias is small relative to action-spread → bias is
    noise on top of a directed Q signal (DDQN's clip removes
    informative noise → potentially harms).
    Low σ/jens: bias dominates action-spread → policy mostly
    follows noise → DDQN's clip removes the dominant-bias-driven
    misranking (potentially helps).

    Replaces the hardcoded `_SIGMA_OVER_JENS_PER_ENV` constant in
    `sigma_over_jens_regime.py`. The cross-env panel aggregator
    is `mean(sigma_over_jens_late)` over baseline-arm cells per
    env, computed at bridge resolution via `DerivedCovariateSpec`
    so HP-mixing in per-env aggregates is surfaced as scope-
    drift rather than frozen into a snapshot constant.

    NaN when jensen_gap is non-finite, ≤ 0, or σ_action is
    non-finite."""
    if not math.isfinite(q_action_std_late):
        return float('nan')
    if not math.isfinite(jensen_gap) or jensen_gap <= 0.0:
        return float('nan')
    return q_action_std_late / jensen_gap


@temporal_reduction(
    reads=('online_max_q_per_step',),
    late_name='q_late_mean',
    per_burst_name='q_per_burst',
)
def _q_max_window_mean(window: npt.NDArray[np.float64]) -> float:
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
    the sign of Hasselt's bias direction.

    Per-burst sibling `q_per_burst` is the Q-magnitude channel at
    per-burst granularity for the two-channel decomposition
    (`findings_ddqn_reward_sign_conditional.md`):
    `bg_per_burst → mc_per_burst` and `q_per_burst → mc_per_burst`
    are independent direct edges in the per-burst PC graph."""
    if window.size == 0:
        return float('nan')
    return float(window.mean())


q_late_mean = _registered('q_late_mean')


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


@temporal_reduction(
    reads=('q_action_grad_overlap_per_step',),
    late_name='q_action_grad_overlap_late',
)
def _q_action_grad_overlap_window_mean(
    window: npt.NDArray[np.float64],
) -> float:
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
    a Q-rank-deficiency proxy.

    Late-only: no per-burst sibling — the bridge consumers all
    read this as the trajectory-averaged FA-architectural
    invariant, not as a per-burst panel."""
    if window.size == 0:
        return float('nan')
    return float(window.mean())


q_action_grad_overlap_late = _registered('q_action_grad_overlap_late')


@temporal_reduction(
    reads=('bootstrap_action_mismatch_per_step',),
    late_name='bootstrap_action_mismatch_late',
)
def _bootstrap_action_mismatch_window_mean(
    window: npt.NDArray[np.float64],
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
    argmax + compare per batch step. Negligible cost.

    Literature positioning. Adjacent to Schaul et al. 2022 "Policy
    Churn" (NeurIPS, arXiv:2206.00730), which measures argmax
    *between policy snapshots* (consecutive update steps).
    `bootstrap_action_mismatch` measures argmax *between (online net,
    target net) at the same step* — a within-time, between-network
    mismatch rate, not a between-time churn. Both capture the
    policy-structure axis (Theorem 1 Cor 1.1 has explicit
    "argmax-preservation" condition); the framework's three orthogonal
    diagnostic axes are documented in THEORY_bootstrap_dominance.md §11."""
    if window.size == 0:
        return float('nan')
    return float(window.mean())


bootstrap_action_mismatch_late = _registered('bootstrap_action_mismatch_late')


@temporal_reduction(
    reads=('q_inter_state_grad_overlap_per_step',),
    late_name='q_inter_state_grad_overlap_late',
)
def _q_inter_state_grad_overlap_window_mean(
    window: npt.NDArray[np.float64],
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
    if window.size == 0:
        return float('nan')
    return float(window.mean())


q_inter_state_grad_overlap_late = _registered(
    'q_inter_state_grad_overlap_late',
)


@temporal_reduction(
    reads=('q_inter_state_grad_overlap_random_per_step',),
    late_name='q_inter_state_grad_overlap_random_late',
)
def _q_inter_state_grad_overlap_random_window_mean(
    window: npt.NDArray[np.float64],
) -> float:
    """Late-window mean of "lag-k" baseline: cosine overlap of
    `∂Q(s, a)/∂θ` vs `∂Q(s_random, a)/∂θ` at paired
    (s, s_random) = (batch.obs[0], batch.obs[-1]) — two states from
    uniform-random replay positions, generally different trajectories.

    Diagnostic baseline for continuous-state envs (LL, MC) where
    the lag-1 measure saturates at 1 because consecutive
    observations differ by infinitesimal continuous deltas. The
    discriminative signal in the cross-env smoothness claim is the
    DIFFERENCE `q_inter_state_grad_overlap_late −
    q_inter_state_grad_overlap_random_late`:
    - Discrete envs: lag-1 > random-pair (trajectory-adjacency
      confers extra smoothness above global baseline) → diff > 0.
    - Continuous envs: both saturate near 1 → diff ≈ 0.

    Probe added 2026-05-22 in `train_phase`; pre-existing corpora
    have NaN."""
    if window.size == 0:
        return float('nan')
    return float(window.mean())


q_inter_state_grad_overlap_random_late = _registered(
    'q_inter_state_grad_overlap_random_late',
)


@measurable(
    reads=(
        'q_inter_state_grad_overlap_per_step',
        'q_inter_state_grad_overlap_random_per_step',
    ),
)
def q_inter_state_grad_overlap_excess_late(
    record: Mapping[str, object],
) -> float:
    """Adjacent-pair smoothness EXCESS over random-pair baseline:
    `q_inter_state_grad_overlap_late −
     q_inter_state_grad_overlap_random_late`.

    The intended discriminative measurable for cross-env
    smoothness comparisons. Continuous-state envs saturate both
    components near 1, driving excess to 0 (no trajectory-adjacency
    signal). Discrete envs separate the two terms (lag-1 captures
    adjacency-specific overlap; random-pair captures global Q
    smoothness)."""
    try:
        adj = Q_INTER_STATE_GRAD_OVERLAP(record)
        rand = Q_INTER_STATE_GRAD_OVERLAP_RANDOM(record)
    except KeyError:
        return float('nan')
    if (
        adj.ndim == 0 or rand.ndim == 0
        or adj.shape[0] < 2 or rand.shape[0] < 2
    ):
        return float('nan')
    return _windowed_mean(adj, 0.5, 1.0) - _windowed_mean(rand, 0.5, 1.0)


@temporal_reduction(
    reads=('online_max_q_per_step',),
    late_name='q_autocorr_late',
)
def _q_autocorr_window(window: npt.NDArray[np.float64]) -> float:
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
    most where autocorr is high.

    Late-only via `@temporal_reduction`: the per-burst sibling
    `q_autocorr_per_burst` is authored separately with `np.array_
    split` windowing (different from the decorator's `np.linspace`
    edges convention)."""
    if window.size < 2:
        return float('nan')
    x = window[:-1]
    y = window[1:]
    if np.std(x) == 0 or np.std(y) == 0:
        return float('nan')
    r = float(np.corrcoef(x, y)[0, 1])
    return r if math.isfinite(r) else float('nan')


q_autocorr_late = _registered('q_autocorr_late')


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


def _per_burst_q_and_mc(
    record: Mapping[str, object],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None:
    """Build per-burst (q_window_mean, mc_burst_mean) for the cross-
    lagged scalar measurables. Returns None when shapes are
    inconsistent or n_bursts < 3 (lag-1 Pearson needs ≥3 pairs)."""
    try:
        q_arr = ONLINE_MAX_Q(record)
        mc_arr = MC_RETURN(record)
    except KeyError:
        return None
    if q_arr.ndim != 1 or mc_arr.ndim != 2:
        return None
    n_bursts = int(mc_arr.shape[0])
    if n_bursts < 3 or q_arr.shape[0] < n_bursts * 2:
        return None
    # Equal-length training-step windows ending at each burst's eval
    # boundary (same windowing as `q_autocorr_per_burst`).
    chunks = np.array_split(q_arr, n_bursts)
    q_per_burst = np.fromiter(
        (chunk.mean() for chunk in chunks),
        dtype=np.float64, count=n_bursts,
    )
    mc_per_burst = mc_arr.mean(axis=1).astype(np.float64)
    return q_per_burst, mc_per_burst


def _lag1_pearson(
    x_prev: npt.NDArray[np.float64], y_curr: npt.NDArray[np.float64],
) -> float:
    """Lag-1 Pearson of `corr(y[1:], x[:-1])`. Returns NaN when
    either series has zero variance."""
    if x_prev.size < 2 or y_curr.size < 2:
        return float('nan')
    if np.std(x_prev) == 0 or np.std(y_curr) == 0:
        return float('nan')
    r = float(np.corrcoef(x_prev, y_curr)[0, 1])
    return r if math.isfinite(r) else float('nan')


@measurable(reads=('online_max_q_per_step', 'mc_return'))
def q_mc_burst_correlation_late(record: Mapping[str, object]) -> float:
    """Contemporaneous Pearson r between per-burst window-mean Q
    and per-burst MC return, over the LATE half of bursts.

    Why it matters: the σ/jens regime discriminator (per
    `findings_sigma_over_jens_regime_discriminator`) assumes
    vanilla's bias is uniform across actions — argmax preserves
    meaningful policy structure. That assumption only holds when
    Q is COUPLED to MC (Q-function tracks return).

    In the FR γ=0.999 vanilla regime, Q grows monotonically to
    100 while MC stays at ~0.005 — Q is decoupled from MC. The
    argmax structure is whatever Q's noise propagates into, not
    a meaningful policy. DDQN's clip prevents the unbounded
    explosion → DDQN rescues regardless of σ/jens shape.

    This measurable lets the σ/jens bridges scope-restrict to
    cells where Q-MC coupling is non-trivial (r > 0.5 say),
    excluding the regime-C "vanilla collapse" cases.

    Returns NaN when n_bursts < 3, when MC variance is too small
    to compute r reliably (z-score floor), or shapes inconsistent.
    Computed on the late HALF of bursts only — late training is
    where the regime is most clearly expressed.

    Distinct from the lag-1 cross-correlations
    (`q_burst_to_mc_cross_lag1`, `mc_burst_to_q_cross_lag1`)
    which test temporal precedence; this one is contemporaneous."""
    pair = _per_burst_q_and_mc(record)
    if pair is None:
        return float('nan')
    q_per_burst, mc_per_burst = pair
    n = q_per_burst.size
    if n < 4:
        return float('nan')
    half = n // 2
    q_late = q_per_burst[half:]
    mc_late = mc_per_burst[half:]
    if np.std(q_late) == 0 or np.std(mc_late) == 0:
        return float('nan')
    r = float(np.corrcoef(q_late, mc_late)[0, 1])
    return r if math.isfinite(r) else float('nan')


@measurable(reads=('online_max_q_per_step', 'mc_return'))
def q_burst_autoregression_lag1(record: Mapping[str, object]) -> float:
    """Lag-1 Pearson autocorrelation of per-burst window-mean Q,
    one scalar per cell. Captures Q's burst-to-burst persistence.

    Why this matters: in the FR γ=0.999 vanilla Q-explosion regime
    (`findings_q_explosion_direct_evidence`), Q drifts via self-
    bootstrap with a1 ≈ 0.82 — Q[t] ≈ 0.82·Q[t-1] + drift. High
    autoregression + observational decoupling (see
    `q_burst_to_mc_cross_lag1`) is the dynamical signature of the
    pure-overestimation-only regime. Acrobot-style cells have low
    autoregression (Q converges fast, ≈ 0.1-0.2).

    Distinct from `q_autocorr_late` (lag-1 across consecutive
    training steps WITHIN a burst — measures FA spatial coherence)
    and `q_autocorr_per_burst` (per-burst vector of the same). This
    one is across-burst — captures the cross-burst dynamical
    persistence of the windowed Q signal."""
    pair = _per_burst_q_and_mc(record)
    if pair is None:
        return float('nan')
    q_per_burst, _ = pair
    return _lag1_pearson(q_per_burst[:-1], q_per_burst[1:])


@measurable(reads=('online_max_q_per_step', 'mc_return'))
def q_burst_to_mc_cross_lag1(record: Mapping[str, object]) -> float:
    """Lag-1 cross-correlation `corr(Q_burst[t-1], MC_burst[t])` —
    does past Q predict future MC? One scalar per cell.

    The 'Q leads MC' marker. In healthy Q-learning (Acrobot DDQN,
    z=+3.01 per `findings_q_explosion_direct_evidence`'s lagged
    analysis), Q-estimate improvements precede policy quality
    improvements with a positive lag. In the FR vanilla Q-
    explosion regime, this is ≈ 0 (z=-0.73 NS) — Q grows
    internally with no propagation to policy."""
    pair = _per_burst_q_and_mc(record)
    if pair is None:
        return float('nan')
    q_per_burst, mc_per_burst = pair
    return _lag1_pearson(q_per_burst[:-1], mc_per_burst[1:])


@measurable(reads=('online_max_q_per_step', 'mc_return'))
def mc_burst_to_q_cross_lag1(record: Mapping[str, object]) -> float:
    """Lag-1 cross-correlation `corr(MC_burst[t-1], Q_burst[t])` —
    does past observation inform current Q? The complement of
    `q_burst_to_mc_cross_lag1`. Healthy Q-learning has both
    directions positive (bidirectional Q↔MC coupling, Acrobot
    DDQN pattern: MC→Q z=+2.97)."""
    pair = _per_burst_q_and_mc(record)
    if pair is None:
        return float('nan')
    q_per_burst, mc_per_burst = pair
    return _lag1_pearson(mc_per_burst[:-1], q_per_burst[1:])


@measurable(reads=('online_max_q_per_step', 'mc_return'))
def bootstrap_dominated_burst_fraction(
    record: Mapping[str, object],
    theta_q_rel: float = 0.05,
    theta_mc: float = 0.02,
) -> float:
    """Fraction of bursts where Q grows substantially while MC stays flat.

    Per-burst predicate: `dQ/|Q_mid| > theta_q_rel` AND `dMC < theta_mc`,
    where d is the burst-to-burst forward difference. Returns the
    fraction of valid (n_bursts − 1) burst pairs satisfying it.

    Operationalizes the bootstrap-dominated under-learning regime of
    Theorem 1 / `findings-fr-g999-rescue-unified-narrative`: the bias
    chain compounds Q via `γ max Q` while the reward signal can't
    pull the policy out. γ-agnostic and env-agnostic (relative-Q
    threshold handles cross-env scale; absolute-MC threshold catches
    "no policy progress").

    Empirical signal at canonical 1M:
        FR γ=0.99   vanilla / DDQN: ~0.05 / 0.07 (not under-learning)
        FR γ=0.999  vanilla / DDQN: ~0.62 / 0.13 (vanilla pure
                                                 under-learning;
                                                 DDQN escapes)
        Asterix γ=0.999 vanilla / DDQN: ~0.26 / 0.22 (intermediate —
                                                     Asterix has Q
                                                     and MC BOTH
                                                     growing, Q faster)

    Pairs with `DDQN/Vanilla Λ_a ratio` (computed at the
    cross-arm bridge level) to discriminate Type A under-learning
    (DDQN suppresses anisotropy → outcome rescue) from Type B
    (DDQN doesn't suppress → outcome harm). See the under-learning
    Finding panel for the full discrimination.

    Returns NaN when n_bursts < 2 or the per-burst extraction fails.
    """
    pair = _per_burst_q_and_mc(record)
    if pair is None:
        return float('nan')
    q_per_burst, mc_per_burst = pair
    if q_per_burst.size < 2:
        return float('nan')
    dq = np.diff(q_per_burst)
    dmc = np.diff(mc_per_burst)
    q_mid = (q_per_burst[:-1] + q_per_burst[1:]) / 2.0
    dq_rel = dq / np.maximum(np.abs(q_mid), 1e-6)
    mask = (dq_rel > theta_q_rel) & (dmc < theta_mc)
    return float(mask.mean())


@measurable(reads=('reward', 'target_max_q_per_step', 'gamma'))
def bootstrap_self_reference_fraction(
    record: Mapping[str, object],
    eps: float = 1e-3,
) -> float:
    """Late-window mean fraction of |bootstrap target| attributable
    to `γ × max_a Q_target(s', a)` (self-reference) vs `|r(s, a)|`
    (observed reward) at training steps.

    For each training step t in the late half:
        |self_ref[t]| = |γ × target_max_q_per_step[t]|
        |reward[t]|   = |reward[t]|
        frac[t]       = |self_ref[t]| / (|self_ref[t]| + |reward[t]| + eps)

    Returns mean over the late half of training steps.

    Interpretation:
    - frac → 1.0: bootstrap target dominated by γ × Q (self-
      reference). Rewards negligible vs Q magnitude — the
      Q-explosion regime per `findings_q_explosion_direct_evidence`
      (sparse-reward × γ→1 vanilla: r=0 almost always while
      γ × Q ~ 100).
    - frac → 0.0: bootstrap target dominated by |r|. Rare in
      sequential decision problems where Q is cumulative.

    Note: this measure naturally saturates near 1.0 because |r|
    (per-step) is typically much smaller than γ × |Q| (cumulative)
    in any non-degenerate RL task. It discriminates Q-explosion
    cases (FR γ=0.999 vanilla: frac ≈ 1.0) from cases where
    vanilla observes reward more (FR γ=0.99 vanilla: frac
    moderately lower because vanilla finds goal sometimes,
    injecting r=1 events). Cross-env comparisons less meaningful
    due to scale differences.

    Uses `target_max_q_per_step` (target net's max Q) at the same
    index as `reward`. Strictly the bootstrap target uses Q at
    the NEXT state s_{t+1}; using `target_max_q_per_step[t]` is a
    one-step-shift approximation that's fine for late-window
    averages (the late training distribution is stable)."""
    r = record.get('reward')
    q = record.get('target_max_q_per_step')
    gamma_v = record.get('gamma')
    if r is None or q is None:
        return float('nan')
    if not isinstance(gamma_v, (int, float)):
        return float('nan')
    gamma = float(gamma_v)
    r_arr = np.asarray(r, dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64)
    if r_arr.ndim != 1 or q_arr.ndim != 1 or r_arr.size != q_arr.size:
        return float('nan')
    n = r_arr.size
    if n < 2:
        return float('nan')
    start = n // 2
    r_abs = np.abs(r_arr[start:])
    q_abs = np.abs(gamma * q_arr[start:])
    denom = q_abs + r_abs + eps
    frac = q_abs / denom
    return float(frac.mean())


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
    the across-action histogram) to disambiguate.

    Literature positioning. Closest published sibling is Schaul et al.
    2022 "The Phenomenon of Policy Churn" (NeurIPS, arXiv:2206.00730),
    which defines `W(π, π'|s) = ½ Σ_a |π(a|s) − π'(a|s)|` per-state
    and aggregates to a churn fraction. Our argmax_persistence is the
    *temporal-self-pair* variant (consecutive STEPS, not consecutive
    SNAPSHOTS on a fixed eval set) — cheaper to trace, but conflates
    real policy instability with batch-composition drift. See
    THEORY_bootstrap_dominance.md §11 for the full positioning."""
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


@temporal_reduction(
    reads=('online_max_q_per_step',),
    late_name='q_max_temporal_cv_late',
)
def _q_max_temporal_cv_window(window: npt.NDArray[np.float64]) -> float:
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
    if window.size < 2:
        return float('nan')
    mu = float(np.mean(window))
    if abs(mu) < 1e-9:
        return float('nan')
    sd = float(np.std(window, ddof=1))
    return sd / abs(mu)


q_max_temporal_cv_late = _registered('q_max_temporal_cv_late')


@measurable(reads=('mc_return',))
def env_disc_raw_alignment(
    record: Mapping[str, object],
    mc_return_raw_episodes: npt.NDArray[np.floating],  # injected
) -> float:
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
    if mc_return_raw_episodes.size == 0:
        return float('nan')
    try:
        mc = MC_RETURN(record)
    except KeyError:
        return float('nan')
    mc_arr = np.asarray(mc, dtype=np.float64)
    if mc_return_raw_episodes.shape != mc_arr.shape:
        return float('nan')
    rf = mc_return_raw_episodes.flatten()
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


@temporal_reduction(
    reads=('td_error',),
    late_name='td_residual_late',
)
def _td_residual_window_mean(window: npt.NDArray[np.float64]) -> float:
    """Mean of |TD residual| over the late 50%. TD-convergence
    scalar — Acrobot's r=+0.84 vs GaussianBandit's r=−0.81
    (sign-flip across regimes) is the canonical motivation for
    per-env PC in PAPER §6."""
    if window.size == 0:
        return float('nan')
    return float(window.mean())


td_residual_late = _registered('td_residual_late')


@temporal_reduction(
    reads=('td_error_within_batch_std',),
    late_name='td_within_batch_var_late',
)
def _td_within_batch_var_window_mean(
    window: npt.NDArray[np.float64],
) -> float:
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
    if window.size == 0:
        return float('nan')
    return float(window.mean())


td_within_batch_var_late = _registered('td_within_batch_var_late')


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
    """Slice `state_hash_per_step` to the late 50% and return as
    int64. None when the field is absent or the slice is empty.

    Reads `state_hash_per_step` (the per-step trace, NOT the
    `state_hash` LEAF config callable). Pre-2026-05-16 the trace
    column was named `state_hash` and collided with the leaf at
    cache-join time; renamed in commit TBD."""
    if 'state_hash_per_step' not in record:
        return None
    arr = np.asarray(record['state_hash_per_step'], dtype=np.int64)
    n = arr.shape[0] if arr.ndim >= 1 else 0
    if n < 2:
        return None
    return arr[n // 2:]


@measurable(reads=('state_hash_per_step',))
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


@measurable(reads=('state_hash_per_step',))
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
# episode return. Injects `mc_return_raw_episodes` (registered
# below) for the (n_bursts, n_episodes) raw return, then
# averages over the episode axis to yield (n_bursts,). The
# parameter-name injection channel propagates the helper's
# closure hash AND its leaf trace-col reads to the dependency
# walker — when the helper's body changes, this consumer's
# signature auto-invalidates without manual hash bumps.
def _mc_return_raw_per_burst_mean(
    record: Mapping[str, object],
    mc_return_raw_episodes: npt.NDArray[np.floating],  # injected
) -> npt.NDArray[np.floating]:
    del record
    if mc_return_raw_episodes.ndim != 2 or mc_return_raw_episodes.size == 0:
        return np.full((0,), float('nan'), dtype=np.float64)
    return mc_return_raw_episodes.mean(axis=1)


mc_return_raw_per_burst_mean = Measurable(
    fn=_mc_return_raw_per_burst_mean,
    name='mc_return_raw__mean_axis_-1',
    reads=(),
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


@measurable(name='mc_first_nonzero_burst', reads=('mc_return',))
def mc_first_nonzero_burst(
    record: Mapping[str, object],
    threshold: float = 0.1,
) -> float:
    """Burst index where per-burst-mean MC return first crosses
    `threshold`. Sentinel value `float(n_bursts)` if MC never
    crosses — usable as a "never anchored" indicator in arm-diff
    analyses (vanilla stuck cells return n_bursts while escaped
    cells return finite small index).

    Used for the FR γ=0.999 temporal-ordering hypothesis: at the
    canonical sparse-positive-reward + γ→1 scope, the burst at
    which MC first anchors (policy aligns to reward signal) should
    be EARLIER for DDQN than for vanilla. Paired with
    `q_first_cross_burst` to test whether policy anchoring precedes
    bias-dominance Q crossing.

    Threshold default 0.1 is appropriate for FR (positive bounded
    [0,1] reward where 0.1 is meaningfully nonzero). Substrates
    with other reward scales should pass threshold explicitly.

    Returns NaN only if `mc_return` is missing or malformed."""
    if 'mc_return' not in record:
        return float('nan')
    mc = np.asarray(record['mc_return'], dtype=np.float64)
    if mc.ndim != 2 or mc.shape[0] < 1:
        return float('nan')
    per_burst = mc.mean(axis=1)
    n = per_burst.size
    crossings = np.where(per_burst > threshold)[0]
    if crossings.size == 0:
        return float(n)  # sentinel: "never crossed"
    return float(crossings[0])


@measurable(name='q_first_cross_burst', reads=('online_max_q_per_step', 'mc_return'))
def q_first_cross_burst(
    record: Mapping[str, object],
    threshold: float = 9.2,
) -> float:
    """Burst index where per-burst-mean online_max_Q first crosses
    `threshold`. Sentinel value `float(n_bursts)` if Q never
    crosses — usable as a "bias never dominated" indicator.

    Used for FR γ=0.999 temporal-ordering hypothesis: at canonical
    scope, `threshold=9.2` is 50% of the Lemma 2 asymptote
    `γb/(1-γ)` ≈ 18.4 for K=4, σ_action≈0.04 (per memory
    `findings_fr_g999_rescue_unified_narrative`). Vanilla cells
    cross this around burst 5-8; DDQN cells never cross (Q stays
    ~1).

    Other envs need different thresholds tied to their Lemma 2
    asymptote. The threshold parameterizes per-bridge author
    commitment to a specific bias-dominance level.

    Chunks `online_max_q_per_step` into n_bursts equal pieces
    (n_bursts from mc_return.shape[0]). Returns NaN on missing data."""
    if 'mc_return' not in record or 'online_max_q_per_step' not in record:
        return float('nan')
    mc = np.asarray(record['mc_return'], dtype=np.float64)
    if mc.ndim != 2 or mc.shape[0] < 1:
        return float('nan')
    n_bursts = mc.shape[0]
    q = np.asarray(record['online_max_q_per_step'], dtype=np.float64)
    if q.ndim != 1 or q.size < n_bursts:
        return float('nan')
    chunks = np.array_split(q, n_bursts)
    per_burst = np.array([c.mean() for c in chunks])
    crossings = np.where(per_burst > threshold)[0]
    if crossings.size == 0:
        return float(n_bursts)  # sentinel: "never crossed"
    return float(crossings[0])


@measurable(name='mc_growth_max_minus_initial', reads=('mc_return',))
def mc_growth_max_minus_initial(record: Mapping[str, object]) -> float:
    """Per-cell scalar: max(per-burst-mean MC) − per-burst-mean MC[0].

    Captures how much the policy improved over training in raw MC
    units. Stuck cells: ~0. Rescued cells: substantial positive value.

    Used (paired with `q_growth_max_minus_initial`) to compute
    `policy_growth_fraction` — a threshold-free measure of "what
    fraction of trajectory progress is policy-side vs bias-side."

    For envs where MC can be negative (Acrobot, MountainCar), this
    gives positive growth too (max − initial ≥ 0). The sign convention
    is just absolute progress, not direction relative to optimal."""
    if 'mc_return' not in record:
        return float('nan')
    mc = np.asarray(record['mc_return'], dtype=np.float64)
    if mc.ndim != 2 or mc.size == 0:
        return float('nan')
    per_burst = mc.mean(axis=1)
    if per_burst.size < 2:
        return float('nan')
    return float(per_burst.max() - per_burst[0])


@measurable(
    name='q_growth_max_minus_initial',
    reads=('online_max_q_per_step', 'mc_return'),
)
def q_growth_max_minus_initial(record: Mapping[str, object]) -> float:
    """Per-cell scalar: max(per-burst-mean Q) − per-burst-mean Q[0].

    Captures how much vanilla's bootstrap chain (or DDQN's clipped
    chain) inflated Q over training. Vanilla FR γ=0.999 cells:
    Q grows ~3.8 → ~12 → growth ~8.2. DDQN cells: ~0.9 → ~1.0 →
    growth ~0.1.

    Paired with `mc_growth_max_minus_initial` to compute
    `policy_growth_fraction`. Chunks `online_max_q_per_step` into
    n_bursts using `mc_return.shape[0]` for n_bursts."""
    if 'mc_return' not in record or 'online_max_q_per_step' not in record:
        return float('nan')
    mc = np.asarray(record['mc_return'], dtype=np.float64)
    if mc.ndim != 2 or mc.shape[0] < 2:
        return float('nan')
    n_bursts = mc.shape[0]
    q = np.asarray(record['online_max_q_per_step'], dtype=np.float64)
    if q.ndim != 1 or q.size < n_bursts:
        return float('nan')
    chunks = np.array_split(q, n_bursts)
    per_burst = np.array([c.mean() for c in chunks])
    return float(per_burst.max() - per_burst[0])


@measurable(
    name='policy_growth_fraction',
    reads=('online_max_q_per_step', 'mc_return'),
)
def policy_growth_fraction(
    record: Mapping[str, object],
    eps: float = 1e-6,
) -> float:
    """Per-cell scalar: fraction of trajectory growth attributable to
    policy (MC) vs bias-chain (Q).

    `policy_growth_fraction = mc_growth / (|mc_growth| + |q_growth| + ε)`

    Where mc_growth = max(per_burst_MC) − per_burst_MC[0]
    and q_growth = max(per_burst_Q) − per_burst_Q[0].

    Threshold-free: no authored cutpoints, no Lemma-2-asymptote
    estimate, no methodological fraction. Just the relative magnitude
    of policy-side vs bias-side trajectory progress.

    Interpretation:
    - Near 0: trajectory progress was all bias-chain growth (Q grew,
      MC didn't). Vanilla-in-failure-trap signature.
    - Near 1: trajectory progress was all policy-side (MC grew, Q
      stable). DDQN-rescue-clean signature.
    - Intermediate: mixed growth.

    Empirical at FR γ=0.999 (n=30 per arm, canonical k=4):
        vanilla stuck cells: ~0.00
        vanilla escape (rare): ~0.03 (post-escape Q inflation dominates)
        DDQN cells: ~0.89

    Empirical at SI γ=0.999 (n=30 per arm):
        vanilla: ~0.065 (MC grew 22→30, Q grew 2→102)
        DDQN:    ~0.24  (MC grew 23→50, Q grew 1.8→88)

    Cross-env meta-comparison NOT recommended (the ratio's scale
    depends on env-specific Q and MC ranges). Within-env arm-diff
    is the appropriate use. The threshold-free formulation is the
    paper-grade replacement for `policy_anchors_before_bias`.

    Literature positioning (see `THEORY_bootstrap_dominance.md` §11).
    Hybrid of the value-magnitude axis (Hasselt et al. 2018 "soft
    divergence" — `max_a Q` shape) and the bias-decomposition axis
    (Fu et al. 2019 oracle-FQI; Q-vs-MC tracking). No published work
    uses precisely this ratio form; the closest analogues are
    Hasselt 2018's qualitative Q-trajectory shape (no MC anchor) and
    Yue 2023's SEEM eigenvalue (predicts divergence, doesn't
    decompose recovery). This measurable's substantive contribution
    is the per-cell decomposition of WHICH recovery mechanism
    (reward-anchored policy convergence vs DDQN-clip-prevented
    Q-blowup) is operating within the soft-divergence regime."""
    mc_growth = mc_growth_max_minus_initial.fn(record)
    q_growth = q_growth_max_minus_initial.fn(record)
    if np.isnan(mc_growth) or np.isnan(q_growth):
        return float('nan')
    denom = abs(mc_growth) + abs(q_growth) + eps
    return mc_growth / denom


@measurable(
    name='policy_anchors_before_bias',
    reads=('online_max_q_per_step', 'mc_return'),
)
def policy_anchors_before_bias(
    record: Mapping[str, object],
    mc_threshold: float = 0.1,
    q_threshold: float = 9.2,
) -> float:
    """Binary indicator (0.0 / 1.0): does the policy reach `mc > mc_threshold`
    BEFORE Q crosses `q_threshold`?

    1.0 iff mc_first_nonzero_burst < q_first_cross_burst (policy
    anchored to reward signal before bias dominated Q).
    0.0 iff Q crossed first OR MC never anchored. Captures the
    temporal-ordering claim at FR γ=0.999:

    - **Vanilla failed cell**: MC stuck at 0; Q grows to ~12. MC
      sentinel = n_bursts; Q sentinel = ~5. Q first → indicator 0.
    - **Vanilla escape (rare)**: MC grows late; Q crosses earlier
      (Q grew during pre-escape exploration). Q first → indicator 0.
    - **DDQN cell**: MC grows; Q stays ~1 (never crosses 9.2).
      MC first → indicator 1.

    Predicted at canonical FR γ=0.999: arm_mean_diff(this) > 0.5
    (DDQN proportion >> vanilla proportion). Pre-registration in
    `experiments.findings.ddqn_three_conditions.bridges` —
    commit hash records the prediction before resolution.

    Default thresholds (mc=0.1, q=9.2) appropriate for FR γ=0.999;
    substrates at other γ or reward scales pass explicitly."""
    mc_first = mc_first_nonzero_burst.fn(record, threshold=mc_threshold)
    q_first = q_first_cross_burst.fn(record, threshold=q_threshold)
    if np.isnan(mc_first) or np.isnan(q_first):
        return float('nan')
    return 1.0 if mc_first < q_first else 0.0


@measurable(name='mc_burst_trend', reads=('mc_return',))
def mc_burst_trend(record: Mapping[str, object]) -> float:
    """Per-cell Spearman ρ(burst_index, per-burst-mean MC return).

    Captures whether the policy IMPROVES over training (ρ → +1),
    STAYS FLAT (ρ → 0), or DEGRADES (ρ → -1).

    Use cases where the trajectory shape matters but a single late
    aggregate hides it:
    - Freeway γ=0.999 vanilla: peak around burst 10-15, then
      declines. Late_window_mean alone reports the declined value
      but doesn't reveal the peak-then-decay shape; trend is
      negative or null at late half, positive at full trajectory.
    - SI γ=0.999 vanilla: MC flat across training (trend ≈ 0)
      vs DDQN where trend ≈ +0.4 (climbing).
    - FR γ=0.999 vanilla: trend ≈ 0 (10/30 cells truly flat at
      zero); DDQN trend ≈ +0.5 (climbing).

    Returns NaN when n_bursts < 3 or MC has zero variance across
    bursts (e.g., truly flat). Zero-variance cells are NOT
    re-coded to ρ=0 — NaN signals "no trend signal" rather than
    "flat trend signal," letting analyses decide how to handle.
    Substrates wanting flat-as-zero can post-process."""
    if 'mc_return' not in record:
        return float('nan')
    mc = np.asarray(record['mc_return'], dtype=np.float64)
    if mc.ndim != 2 or mc.shape[0] < 3:
        return float('nan')
    per_burst = mc.mean(axis=1)
    if np.std(per_burst) == 0:
        return float('nan')
    from scipy.stats import spearmanr
    burst_idx = np.arange(per_burst.size, dtype=np.float64)
    rho, _ = spearmanr(burst_idx, per_burst)
    return float(rho) if np.isfinite(rho) else float('nan')


@measurable(name='td_burst_trend', reads=('td_error', 'mc_return'))
def td_burst_trend(record: Mapping[str, object]) -> float:
    """Per-cell Spearman ρ(burst_index, per-burst-mean |TD error|).

    Captures whether the training loss CONVERGES (ρ → 0 with low
    magnitude), GROWS (ρ → +1 — vanilla Q-explosion regime), or
    SHRINKS (ρ → -1 — clean convergence).

    Use case: at γ=0.999 the burst dynamics panel shows vanilla
    |TD| keeps climbing across training in most envs (Asterix,
    Breakout, SI, Freeway). DDQN's |TD| stabilizes. The "outcome
    looks fine" reading hides "loss never converges" without this
    trajectory signal.

    Chunks `td_error` (per-training-step list) into n_bursts
    equal pieces using `mc_return.shape[0]` for n_bursts. Same
    chunking convention as `_per_burst_q_and_mc`.

    Returns NaN when n_bursts < 3, td_error missing, or zero
    variance across burst-chunks."""
    if 'mc_return' not in record or 'td_error' not in record:
        return float('nan')
    mc = np.asarray(record['mc_return'], dtype=np.float64)
    if mc.ndim != 2 or mc.shape[0] < 3:
        return float('nan')
    n_bursts = mc.shape[0]
    td = np.asarray(record['td_error'], dtype=np.float64)
    if td.ndim != 1 or td.size < n_bursts:
        return float('nan')
    chunks = np.array_split(np.abs(td), n_bursts)
    per_burst = np.array([c.mean() for c in chunks])
    if np.std(per_burst) == 0:
        return float('nan')
    from scipy.stats import spearmanr
    burst_idx = np.arange(per_burst.size, dtype=np.float64)
    rho, _ = spearmanr(burst_idx, per_burst)
    return float(rho) if np.isfinite(rho) else float('nan')


@measurable(name='outcome_peak_width_relative', reads=('mc_return',))
def outcome_peak_width_relative(record: Mapping[str, object]) -> float:
    """Per-cell peak BROADNESS: fraction of bursts whose per-burst-mean
    MC sits within 20% of the peak's range above the minimum.

    Range-based threshold handles negative-MC envs (Acrobot, MC).
    `threshold = min + 0.8 × (peak − min)`. Count bursts where MC ≥
    threshold, then divide by n_bursts.

    Interpretation:
    - Near 1.0: trajectory is FLAT (most bursts near peak — agent
      converges early and stays).
    - 0.5: half the trajectory is within 20% of peak range.
    - Near 1/n_bursts: NARROW spike (only the peak burst itself
      and 1-2 neighbors qualify).

    Use case: resolves the "DDQN best-burst Δ is +5.30 but per-burst
    plot looks flat" puzzle at Breakout γ=0.999. The plot showed
    similar typical-burst performance; the +5.30 Δ comes from
    cell-specific peak heights at varied timings. Peak-width
    distinguishes the cells' peak SHAPE: vanilla Breakout γ=0.999
    has mean peak-width 0.21 (huge spread, some broad some narrow);
    DDQN has 0.094 — DDQN peaks are NARROWER but more cells reach
    high peaks. So DDQN's gain at Breakout is "more cells reach
    transient highs," not "broader sustained policy."

    Pairs with `outcome_late_to_best_ratio` (stability of post-peak
    region) to give the full peak-shape characterization."""
    if 'mc_return' not in record:
        return float('nan')
    mc = np.asarray(record['mc_return'], dtype=np.float64)
    if mc.ndim != 2 or mc.size == 0:
        return float('nan')
    per_burst = mc.mean(axis=1)
    n = per_burst.size
    if n < 4:
        return float('nan')
    peak = float(per_burst.max())
    floor = float(per_burst.min())
    if peak == floor:
        return 1.0  # truly flat — every burst at peak
    threshold = floor + 0.8 * (peak - floor)
    return float((per_burst >= threshold).sum()) / float(n)


@measurable(name='outcome_post_peak_decay_5', reads=('mc_return',))
def outcome_post_peak_decay_5(record: Mapping[str, object]) -> float:
    """Ratio of mean MC return in the 5-burst window AFTER the peak
    to the peak value itself.

    Detects post-peak policy decay. For sustained-peak policies:
    ratio ≈ 1.0. For peak-then-decay: ratio < 1 (positive env) or
    > 1 (negative env, more-negative is worse).

    Distinct from `outcome_late_to_best_ratio` which uses the LATE
    WINDOW (last 25%) regardless of peak timing. This measurable
    aligns the window TO the peak — captures the local trajectory
    shape around each cell's own peak. Useful for cells that peak
    mid-training (where late_to_best_ratio aggregates over a window
    that may or may not include the peak).

    Empirical at γ=0.999 (median across cells):
        Asterix:  vanilla=0.65, DDQN=0.60 — similar decay
        Breakout: vanilla=0.67, DDQN=0.67 — similar
        Freeway:  vanilla=0.70, DDQN=0.67 — similar
        SI:       vanilla=0.46, DDQN=0.57 — DDQN less post-peak decay

    NaN when peak is within 5 bursts of trajectory end (insufficient
    post-peak window) or when peak and post-peak window have
    opposite signs (ratio uninterpretable).
    """
    if 'mc_return' not in record:
        return float('nan')
    mc = np.asarray(record['mc_return'], dtype=np.float64)
    if mc.ndim != 2 or mc.size == 0:
        return float('nan')
    per_burst = mc.mean(axis=1)
    n = per_burst.size
    if n < 6:
        return float('nan')
    peak_idx = int(np.argmax(per_burst))
    peak_val = float(per_burst[peak_idx])
    if peak_idx + 6 > n:
        return float('nan')  # not enough post-peak window
    post_peak_window = per_burst[peak_idx + 1:peak_idx + 6]
    post_peak_mean = float(post_peak_window.mean())
    if peak_val == 0.0:
        return float('nan')
    if (peak_val > 0 and post_peak_mean < 0) or (peak_val < 0 and post_peak_mean > 0):
        return float('nan')
    return post_peak_mean / peak_val


@measurable(name='outcome_late_to_best_ratio', reads=('mc_return',))
def outcome_late_to_best_ratio(record: Mapping[str, object]) -> float:
    """Ratio of late-quarter-mean MC return to best-burst MC return.

    Decomposes the stability/dynamics distinction at the per-cell
    level: pairs with `eval_best_burst_mean` (dynamics) and
    `late_window_mean` (stability) per
    `findings_outcome_metric_distinction`.

    Interpretation (when best and late have same sign):
    - ratio ≈ 1.0: late performance ≈ peak. Stable policy at
      convergence (FR γ=0.999 vanilla mostly-zero cells: ratio
      undefined; FR converged cells: ratio ≈ 1.0).
    - ratio < 1.0 (positive env): peak-then-decay. DDQN's effect
      may be "prevents decay" rather than "improves peak." Freeway
      γ=0.999 vanilla shows ratio ≈ 0.5 (peak ~15, late ~7).
    - ratio > 1.0 (negative env): peak-then-worsening (rare).

    Sign-mismatch case (best > 0, late < 0 or vice versa) returns
    NaN — the ratio is uninterpretable across sign boundary; use
    `late_window_mean` and `eval_best_burst_mean` directly.

    Computation: best_burst over per-burst means; late_window over
    last 25% of bursts."""
    if 'mc_return' not in record:
        return float('nan')
    mc = np.asarray(record['mc_return'], dtype=np.float64)
    if mc.ndim != 2 or mc.size == 0:
        return float('nan')
    per_burst = mc.mean(axis=1)
    n = per_burst.size
    if n < 4:
        return float('nan')
    best = float(per_burst.max())
    late_start = int(0.75 * n)
    late = float(per_burst[late_start:].mean())
    if best == 0.0:
        return float('nan')
    if (best > 0 and late < 0) or (best < 0 and late > 0):
        return float('nan')
    return late / best


@measurable(reads=('mc_return_from_step', 'episode_length', 'gamma'))
def mc_return_raw_episodes(
    record: Mapping[str, object],
) -> npt.NDArray[np.floating]:
    """Per-(burst, episode) **undiscounted** episode return,
    reconstructed from `mc_return_from_step` (per-step discounted
    value-to-go).

    Math: at each step `t`, `mc[t] = r[t] + γ · mc[t+1]` →
    `r[t] = mc[t] - γ · mc[t+1]`. With `mc[T] = 0`, summing
    per-step rewards gives:

      Σ_{t=0..T-1} r_t
        = Σ_t (v[t] − γ·v[t+1])
        = v[0] + (1 − γ) · Σ_{t=1..T-1} v[t]

    Forward form avoids the cancellation error of the backward-
    recurrence `Σ (v[t] − γ·v[t+1]) + v[T-1]`, ~T× fewer FLOPs.

    Registered as a measurable (rather than a free function) so
    the dependency walker tracks its closure hash via the
    parameter-name injection channel — when this body changes,
    every consumer that injects `mc_return_raw_episodes` gets its
    signature auto-invalidated. Returns a (n_bursts, n_episodes)
    array (~50 floats per cell at typical sweep shapes; safe to
    persist)."""
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
    one_minus_gamma = 1.0 - gamma
    for b in range(n_bursts):
        for e in range(n_episodes):
            T = int(lengths[b, e])
            if T <= 0:
                continue
            v = mc_from_step[b, e, :T]
            if T == 1:
                raw[b, e] = float(v[0])
                continue
            raw[b, e] = float(v[0] + one_minus_gamma * v[1:].sum())
    return raw


@measurable(name='eval_best_burst_raw_mean', reads=())
def eval_best_burst_raw_mean(
    record: Mapping[str, object],
    mc_return_raw_episodes: npt.NDArray[np.floating],  # injected
) -> float:
    """Undiscounted counterpart of `eval_best_burst_mean`:
    `max_i(mean(mc_return_raw[i, :]))`. The best-burst-seen
    metric on the raw (undiscounted) return — γ-invariant policy
    quality. Use for bridges that compare across γ or across
    envs with different reward scaling.

    Injects `mc_return_raw_episodes` so this measurable's
    closure-hash auto-invalidates when the reconstruction logic
    changes."""
    del record
    if mc_return_raw_episodes.ndim != 2 or mc_return_raw_episodes.size == 0:
        return float('nan')
    return float(mc_return_raw_episodes.mean(axis=1).max())


@measurable(name='eval_full_auc_mean', reads=('mc_return',))
def eval_full_auc_mean(record: Mapping[str, object]) -> float:
    """`mean(mc_return)` over all bursts × episodes. The
    full-trajectory AUC of the training curve — captures
    integrated DDQN benefit across early-learning AND
    late-stability windows, while `eval_best_burst_mean` only
    captures a single peak.

    The honest estimator for slow-converging envs where DDQN's
    benefit accumulates across bursts (MetaMaze γ=0.99 d=+0.71
    via AUC vs d=+0.30 via best-burst; memory
    `findings_metamaze_translates_after_eval_power`)."""
    if 'mc_return' not in record:
        return float('nan')
    mc = np.asarray(record['mc_return'], dtype=np.float64)
    if mc.ndim != 2 or mc.size == 0:
        return float('nan')
    return float(mc.mean())


@measurable(name='eval_final_raw_mean', reads=())
def eval_final_raw_mean(
    record: Mapping[str, object],
    mc_return_raw_episodes: npt.NDArray[np.floating],  # injected
) -> float:
    """Undiscounted counterpart of `eval_final_mean`:
    `mean(mc_return_raw[-1, :])`. γ-invariant final-burst policy
    quality for cross-γ comparisons where discounted-MC weights
    later-step rewards differently across γ.

    At Breakout γ=0.99, discounted `eval_final_mean` shows DDQN
    marginally helping (+0.08) while undiscounted `eval_final_raw_mean`
    shows DDQN marginally harming (-2.23). The discount kills
    late-step weight in 50-step episodes (γ^50 ≈ 0.61), so
    discounted and undiscounted measure different things at
    intermediate γ. The raw version is the natural game-score
    metric."""
    del record
    if mc_return_raw_episodes.ndim != 2 or mc_return_raw_episodes.size == 0:
        return float('nan')
    return float(mc_return_raw_episodes[-1, :].mean())


@measurable(name='eval_full_auc_raw_mean', reads=())
def eval_full_auc_raw_mean(
    record: Mapping[str, object],
    mc_return_raw_episodes: npt.NDArray[np.floating],  # injected
) -> float:
    """Undiscounted counterpart of `eval_full_auc_mean`:
    `mean(mc_return_raw)` over all bursts × episodes. γ-invariant
    integrated AUC for cross-γ / cross-env comparisons."""
    del record
    if mc_return_raw_episodes.ndim != 2 or mc_return_raw_episodes.size == 0:
        return float('nan')
    return float(mc_return_raw_episodes.mean())


@measurable(name='eval_late_burst_mean', reads=('mc_return',))
def eval_late_burst_mean(record: Mapping[str, object]) -> float:
    """Mean of `mc_return` over the LAST 30% of bursts (rounded
    up; minimum 1 burst). Convergence-region policy quality —
    captures DDQN's late-stability benefit (vanilla can drift down
    after peaking; this metric reads the trained-agent endpoint
    rather than the pre-collapse peak that `eval_best_burst_mean`
    picks).

    Pair with `eval_best_burst_mean`: a large best > late gap
    flags vanilla's late-burst Q-collapse (Acrobot k=16 pattern,
    memory `findings_per_burst_acrobot_k_sweep`)."""
    if 'mc_return' not in record:
        return float('nan')
    mc = np.asarray(record['mc_return'], dtype=np.float64)
    if mc.ndim != 2 or mc.size == 0:
        return float('nan')
    n_bursts = mc.shape[0]
    n_late = max(1, (n_bursts + 2) // 3)  # ceil(n/3)
    return float(mc[-n_late:, :].mean())


@measurable(name='eval_late_burst_raw_mean', reads=())
def eval_late_burst_raw_mean(
    record: Mapping[str, object],
    mc_return_raw_episodes: npt.NDArray[np.floating],  # injected
) -> float:
    """Undiscounted counterpart of `eval_late_burst_mean`:
    last-30%-bursts mean of `mc_return_raw`. γ-invariant
    convergence-region policy quality."""
    del record
    if mc_return_raw_episodes.ndim != 2 or mc_return_raw_episodes.size == 0:
        return float('nan')
    n_bursts = mc_return_raw_episodes.shape[0]
    n_late = max(1, (n_bursts + 2) // 3)
    return float(mc_return_raw_episodes[-n_late:, :].mean())


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
        # "Lag-k" baseline: cosine overlap at a random-batch-partner
        # pair (batch.obs[0], batch.obs[-1]) — sampled from different
        # trajectory positions. Continuous-state envs saturate both
        # adjacent and random pairs near 1; the EXCESS measurable
        # below is the discriminative signal.
        q_inter_state_grad_overlap_random_late,
        # Adjacent-pair smoothness EXCESS over random-pair baseline.
        # `q_inter_state_grad_overlap_late − q_inter_state_grad_overlap_random_late`.
        # Robust cross-env smoothness comparator: ≈ 0 at saturated
        # (continuous) envs; > 0 where adjacency confers extra
        # smoothness above global Q smoothness.
        q_inter_state_grad_overlap_excess_late,
        # Cross-action bootstrap rate: fraction of training
        # transitions where argmax_a' Q_online(s', a') differs from
        # the action taken at s. THE regime where DDQN's argmax-
        # target decoupling has leverage. Conjunct with γ→1 + r→0
        # to operationalize "when DDQN should help" hypothesis.
        bootstrap_action_mismatch_late,
        # Cross-burst lagged dynamics. The Q-explosion dynamical
        # signature (`findings_q_explosion_direct_evidence`):
        # - High Q autoregression (a1 ≈ 0.82 in FR γ=0.999 VAN)
        #   means Q drifts via self-bootstrap.
        # - Null Q→MC cross-lag (z=-0.73 NS in FR VAN) means Q is
        #   decoupled from observations — pure overestimation.
        # - Bidirectional Q↔MC coupling (Acrobot DDQN, z=+3 both
        #   ways) marks healthy mech-link translation.
        # Pair these with `reward_nonzero_frac` as substrate-level
        # regime classifiers.
        q_burst_autoregression_lag1,
        q_burst_to_mc_cross_lag1,
        mc_burst_to_q_cross_lag1,
        # Raw (undiscounted) eval-return scalar — γ-invariant
        # policy-quality metric for cross-γ / cross-env bridges.
        # Reconstructs per-(burst, episode) raw return inline from
        # trace columns; reduces to the best-burst scalar
        # (counterpart of `eval_best_burst_mean`).
        eval_best_burst_raw_mean,
        eval_final_raw_mean,
        eval_full_auc_raw_mean,
        eval_late_burst_raw_mean,
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
