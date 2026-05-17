"""Causal chain: DDQN's clip introduces argmax noise that, at
γ=0.999, propagates to outcome harm — when vanilla's argmax was
already meaningful (uniform-overestimation regime).

This module decomposes the σ/jens hypothesis into testable causal
edges, replacing the bare regression
`finding_sigma_over_jens_regime_discriminator` (which was REFUTED
because it tested a marginal cross-env correlate, not a mechanism).

The mechanism (per
`findings_sigma_over_jens_regime_discriminator.md`):

DDQN's clip replaces `max_a Q_target(s', a)` with
`Q_target(s', argmax_online(s'))`. Two effects:

  PRO: when Q estimates have positive max-bias, using
       argmax_online (decoupled estimator) breaks the bias
       upward-pressure (Hasselt 2010).
  CON: when argmax_online ≠ argmax_target (they DO disagree
       because of fresh gradient updates), DDQN bootstraps from
       a sub-optimal-per-target action. This is a NEW error
       vanilla doesn't have.

Both PRO and CON propagate through Bellman backups → both scale
by 1/(1−γ) at γ→1. At γ=0.999, both are amplified ~1000×.

When vanilla's argmax was preserved (uniform Q-overestimation,
e.g. Asterix γ=0.999), PRO is small (vanilla's argmax wasn't
corrupted to begin with) and CON dominates → DDQN HARMS outcome.

This is the CON-side mechanism. Three causal edges, three bridges:

  Edge 1 — `ddqn_clip_increases_state_conditional_argmax_entropy`:
    DDQN's per-state argmax variability is HIGHER than vanilla's.
    Using H(argmax | state) so we isolate "argmax noise per
    state" from "state-discriminative policy" — `argmax_persistence`
    conflates these because consecutive states are different
    (a good state-discriminative policy will have low persistence
    by construction). Predicted: a_gt_b (DDQN H_cond > vanilla
    H_cond).

  Edge 2 — `mismatch_predicts_outcome_harm__within_ddqn`:
    Within DDQN cells at γ=0.999, more bootstrap action mismatch
    correlates with worse outcome. Tests that the mechanism's
    proximate effect (mismatch) translates to the distal effect
    (outcome). Predicted: ρ < 0.

  Edge 3 — `delta_h_cond_predicts_delta_outcome_xenv`:
    Across envs at γ=0.999, the per-env arm-diff in per-state
    argmax entropy (DDQN−vanilla) correlates with the per-env
    arm-diff in outcome. Tests the dose-response form: more
    clip-induced per-state argmax-noise → more outcome harm.
    Δ_H_cond > 0 (DDQN noisier), Δ_out < 0 (DDQN worse) → ρ < 0
    across envs.

If all three HELD → mechanism chain SUPPORTED. The framework's
`composed_verdict` AND-aggregates them in
`finding_ddqn_clip_argmax_harm_chain`."""
from __future__ import annotations

import math

import polars as pl

from corroborate.analyses.cross_stratum_arm_diff_slope import (
    CrossStratumArmDiffSlopeResult,
)
from corroborate.analyses.stratified_arm_diff_pooled import (
    StratifiedArmDiffPooledResult,
)
from corroborate.analyses.stratified_partial_spearman import (
    StratifiedPartialSpearmanResult,
)
from corroborate.bridge.bridge import Direction, Tier, claim_bridge
from corroborate.bridge.verdict import RefutationClass, Verdict

from experiments.findings.ddqn._arms import (
    DDQN_ARM, INTERVENTION, VANILLA_ARM,
)
from experiments.findings.ddqn._scope import CANONICAL_HP_EXCLUDING_GAMMA


# File-level scope shared by every bridge in this module. Three
# requirements composed:
#   1. γ=0.999 (the regime where the mechanism amplifies via 1/(1-γ))
#   2. CANONICAL_HP_EXCLUDING_GAMMA — canonical-shape HPs per env
#      (sync, capacity, hidden, channels, lr, wrappers, etc.) so
#      cross-env effects aren't confounded by HP-sweep variants
#      (Acrobot has 2130 γ=0.999 cells across HP-varied corpora;
#      MountainCar 1440; CartPole 720 — most of those are HP
#      sweeps that contaminate the mechanism test if not filtered).
#   3. q_mc_burst_correlation_late >= 0.3 — vanilla's Q tracks MC,
#      excluding regime-C Q-explosion cases (FR γ=0.999 unshaped)
#      where vanilla's argmax doesn't carry meaningful policy
#      structure and the mechanism doesn't apply.
_GAMMA_999_LEARNABLE_CANONICAL_SCOPE: pl.Expr = (
    (pl.col('gamma') == 0.999)
    & CANONICAL_HP_EXCLUDING_GAMMA
    & pl.col('q_mc_burst_correlation_late').is_finite()
    & (pl.col('q_mc_burst_correlation_late') >= 0.3)
)


# ============ Edge 1: clip increases per-state argmax noise ============

@claim_bridge(
    source=INTERVENTION,
    target='state_conditional_argmax_entropy_late',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=(
        _GAMMA_999_LEARNABLE_CANONICAL_SCOPE
        & pl.col('state_conditional_argmax_entropy_late').is_finite()
    ),
    predicted_direction='a_gt_b',
)
def ddqn_clip_increases_state_conditional_argmax_entropy(
    stratified_arm_diff_pooled: StratifiedArmDiffPooledResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    source: str = 'state_conditional_argmax_entropy_late',
    stratify_by: tuple[str, ...] = ('env_name',),
    scope_predictor: str = 'jensen_gap',
    min_baseline_predictor: float = 0.5,
    min_seeds_per_arm: int = 5,
    harm_floor: float = 0.2,
    min_strata: int = 1,
) -> tuple[Verdict, RefutationClass | None]:
    """Edge 1 of the clip-argmax-harm chain.

    Test that DDQN's clip introduces *per-state* argmax noise:
    `H(argmax | state)` is HIGHER under DDQN than vanilla at
    γ=0.999. Using state-CONDITIONAL entropy isolates "argmax
    noise WITHIN a state" from "argmax variability ACROSS
    states" — the latter being just the agent's state-
    discriminative policy structure, which DOESN'T indicate a
    noisy mechanism.

    HELD: per-env pooled Cohen's d on H_cond is ≥ `harm_floor`
    (default +0.2) AT p < 0.05. That is, DDQN's H_cond is at
    least 0.2 SD above vanilla's — more action variability for
    the same observed state.

    REFUTED (SIGN_FLIP): d ≤ −`harm_floor` (DDQN policy is
    MORE deterministic per state — would contradict the
    mechanism).

    CAVEAT 1: state_conditional_argmax_entropy_late requires
    `state_hash_per_step` — env must have a registered state_hash
    callable. Available on 8 envs: Acrobot (card=46656), CartPole
    (10000), MountainCar (400), LunarLander (65536), and 4
    MinAtar envs (Asterix/Breakout/Freeway/SI, card=512 each).
    FourRooms-misc, MetaMaze-misc, PacMan-jumanji, bsuite,
    SlidingTile, Snake → NaN, dropped by the scope predicate.

    CAVEAT 2: state_hash cardinality vs trace length. Acrobot
    at 46k buckets over 1M steps → ~21 visits/bucket — H_cond
    is undersampled. MountainCar at 400 buckets → ~2500
    visits/bucket — well-resolved. The discrimination power of
    H_cond depends on this ratio.

    CAVEAT 3: empirical evidence at γ=0.99 (from
    `findings-mi-state-argmax-disambiguation`) shows DDQN's
    marginal H_marg LOWER than vanilla's on SI (d=-0.57 sig),
    not higher. The chain's Edge 1 prediction is that γ=0.999
    REVERSES this (clip-noise amplification dominates), but
    that's an open empirical question this bridge will test."""
    del treatment_arm, baseline_arm, source, stratify_by
    del scope_predictor, min_baseline_predictor, min_seeds_per_arm
    if stratified_arm_diff_pooled.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT, None
    d = stratified_arm_diff_pooled.pooled_d
    p = stratified_arm_diff_pooled.pooled_p_value
    if math.isnan(d) or math.isnan(p):
        return Verdict.POWER_INSUFFICIENT, None
    abs_floor = abs(harm_floor)
    if d >= abs_floor and p < 0.05:
        return Verdict.HELD, None
    if d <= -abs_floor and p < 0.05:
        return Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP
    return Verdict.POWER_INSUFFICIENT, None


# ============ Edge 2: mismatch → outcome harm WITHIN DDQN ============

@claim_bridge(
    source='bootstrap_action_mismatch_late',
    target='eval_best_burst_raw_mean',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=(
        _GAMMA_999_LEARNABLE_CANONICAL_SCOPE
        & pl.col('bootstrap_action_mismatch_late').is_finite()
        & pl.col('eval_best_burst_raw_mean').is_finite()
        & pl.col('arm_key').str.contains('double_greedify')
    ),
    predicted_direction='a_lt_b',
)
def mismatch_predicts_outcome_harm__within_ddqn(
    stratified_partial_spearman: StratifiedPartialSpearmanResult,
    *,
    x: str = 'bootstrap_action_mismatch_late',
    y: str = 'eval_best_burst_raw_mean',
    conditioning: str = 'jensen_gap',
    stratify_by: str = 'env_name',
    min_stratum_size: int = 5,
    min_rho: float = 0.3,
    min_strata: int = 2,
) -> Verdict:
    """Edge 2 of the clip-argmax-harm chain.

    Within DDQN cells at γ=0.999, test that bootstrap-action
    mismatch (clip-induced argmax disagreement) correlates with
    WORSE outcome. Per-env Spearman partialed by jensen_gap to
    rule out the mismatch-just-tracks-bias confound, Fisher-z
    pool across envs.

    Predicted ρ_pool < 0 — more mismatch → worse outcome.

    HELD if pooled partial-r ≤ −`min_rho`. NO_EFFECT (SIGN_FLIP)
    if pooled-r ≥ +min_rho (would contradict the mechanism).
    POWER_INSUFFICIENT otherwise."""
    del x, y, conditioning, stratify_by, min_stratum_size
    if stratified_partial_spearman.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT
    rho = stratified_partial_spearman.rho_pooled
    if math.isnan(rho):
        return Verdict.POWER_INSUFFICIENT
    if rho <= -min_rho:
        return Verdict.HELD
    if rho >= min_rho:
        return Verdict.NO_EFFECT
    return Verdict.POWER_INSUFFICIENT


# ============ Edge 3: cross-env dose-response ============

@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_raw_mean',
    direction=Direction.INVERSE,
    tier=Tier.INTERVENTIONAL,
    scope=(
        _GAMMA_999_LEARNABLE_CANONICAL_SCOPE
        & pl.col('state_conditional_argmax_entropy_late').is_finite()
        & pl.col('eval_best_burst_raw_mean').is_finite()
    ),
    predicted_direction='a_lt_b',
)
def delta_h_cond_predicts_delta_outcome_xenv(
    cross_stratum_arm_diff_slope: CrossStratumArmDiffSlopeResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    target: str = 'eval_best_burst_raw_mean',
    predictor: str = 'state_conditional_argmax_entropy_late',
    stratify_by: tuple[str, ...] = ('env_name',),
    min_seeds_per_arm: int = 5,
    rho_threshold_held: float = 0.5,
    p_threshold: float = 0.10,
    null_threshold: float = 0.2,
    min_strata: int = 3,
) -> tuple[Verdict, RefutationClass | None]:
    """Edge 3 of the clip-argmax-harm chain — dose-response.

    Across envs at γ=0.999 (learnable), Spearman ρ between
    per-env Δ_H_cond (DDQN−vanilla, H(argmax|state)) and
    per-env Δ_outcome (DDQN−vanilla). Predicts ρ < 0: when
    DDQN's clip makes the per-state argmax noisier (Δ_H_cond > 0)
    the outcome drops more (Δ_out < 0).

    Tests the mechanism at the cohort level: does the dose
    (clip-induced per-state argmax noise) predict the response
    (outcome drop) across envs?

    HELD if ρ ≤ −`rho_threshold_held` AND p ≤ `p_threshold`.
    NO_EFFECT (SIGN_FLIP) if ρ ≥ +`rho_threshold_held` (more
    noise → better outcome, contradicting mechanism).

    `min_strata=3` accommodates the panel after the
    learnability scope drops some envs. The eligible cohort
    (state_hash registered + γ=0.999 cells available + Q-MC
    coupled) is up to 8 envs: Acrobot, CartPole, MountainCar,
    LunarLander, and 4 MinAtar."""
    del treatment_arm, baseline_arm, target, predictor, stratify_by
    del min_seeds_per_arm
    if cross_stratum_arm_diff_slope.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT, None
    rho = cross_stratum_arm_diff_slope.rho
    p = cross_stratum_arm_diff_slope.p_value
    if math.isnan(rho):
        return Verdict.POWER_INSUFFICIENT, None
    if rho <= -rho_threshold_held and (math.isnan(p) or p <= p_threshold):
        return Verdict.HELD, None
    if rho >= rho_threshold_held:
        return Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP
    if abs(rho) < null_threshold:
        return Verdict.NO_EFFECT, RefutationClass.NULL_EFFECT
    return Verdict.POWER_INSUFFICIENT, None


BRIDGES = (
    ddqn_clip_increases_state_conditional_argmax_entropy,
    mismatch_predicts_outcome_harm__within_ddqn,
    delta_h_cond_predicts_delta_outcome_xenv,
)
