"""Causal chain: DDQN's clip reduces Q-smoothness → outcome harm.

Replaces the refuted clip-argmax-noise chain
(`clip_argmax_harm_mechanism`). The cross-arm signature on
Asterix γ=0.999 is real (DDQN does cut Q-smoothness, d=-2.13
z=-8.24), but disentangling smoothness from jens at single-env
scope is impossible — both move ~50% in lockstep under DDQN.

The chain has two edges with different scope shapes by design:

  Edge 1 — `ddqn_cuts_q_smoothness_asterix_gamma_999`
    SINGLE-ENV: mechanism-active test. DDQN's
    q_inter_state_grad_overlap_late is LOWER than vanilla's on
    Asterix γ=0.999. Asks "is the mechanism engaged on the
    target env?". Empirical d=-2.13 z=-8.24 → comfortably HELD.

  Edge 2 — `q_smoothness_predicts_outcome__cross_stratum`
    CROSS-STRATUM: independence test. Across the (env_name, γ)
    Δ panel (Asterix/Breakout/Freeway × γ=0.95/0.999), does
    Δ_smoothness predict Δ_outcome AFTER controlling for
    Δ_jens? Asks "is smoothness an independent mediator or
    jens's shadow?". Consumes
    `cross_stratum_arm_diff_partial_spearman`. Predicted
    direction `a_gt_b` (smoother Q → higher outcome).

If both HELD → chain SUPPORTED. The framework's
`composed_verdict` AND-aggregates them. The asymmetric-scope
authoring is honest: Edge 1 establishes the mechanism is active
on the target env; Edge 2 establishes the mediation isn't
jens's shadow at the cross-stratum scope where they can be
decoupled.

Empirical state (ddqn_sweeps cache post-backfill):
  Edge 1: Asterix γ=0.999 d=-2.13 z=-8.24 → HELD.
  Edge 2: 6-stratum panel partial-r=-0.325 p=0.63 → UNDERPOWERED.
"""
from __future__ import annotations

import math

import polars as pl
from scipy.stats import norm

from corroborate.analyses.link.cross_stratum_arm_diff_partial_spearman import (
    CrossStratumArmDiffPartialSpearmanResult,
)
from corroborate.analyses.panel.stratified_arm_diff_pooled import (
    StratifiedArmDiffPooledResult,
)
from corroborate.bridge.bridge import Direction, Tier, claim_bridge
from corroborate.bridge.verdict import RefutationClass, Verdict

from experiments.findings.ddqn._arms import (
    DDQN_ARM, INTERVENTION, VANILLA_ARM,
)
from experiments.findings.ddqn._scope import CANONICAL_HP_EXCLUDING_GAMMA


# Edge-1 scope: Asterix γ=0.999, canonical-shape HPs. Single-env
# mechanism-active test.
_ASTERIX_GAMMA_999_SCOPE: pl.Expr = (
    (pl.col('env_name') == 'Asterix-MinAtar')
    & (pl.col('gamma') == 0.999)
    & CANONICAL_HP_EXCLUDING_GAMMA
)

# Edge-2 scope: MinAtar (Asterix, Breakout, Freeway) × γ∈{0.95,
# 0.999} × canonical-shape HPs. 6-stratum cross-stratum panel.
_CROSS_STRATUM_GAMMA_SCOPE: pl.Expr = (
    pl.col('env_name').is_in(
        ['Asterix-MinAtar', 'Breakout-MinAtar', 'Freeway-MinAtar']
    )
    & pl.col('gamma').is_in([0.95, 0.999])
    & CANONICAL_HP_EXCLUDING_GAMMA
)


# ============ Edge 1: DDQN cuts Q-smoothness on Asterix γ=0.999 ============

@claim_bridge(
    source=INTERVENTION,
    target='q_inter_state_grad_overlap_late',
    direction=Direction.INVERSE,
    tier=Tier.INTERVENTIONAL,
    scope=(
        _ASTERIX_GAMMA_999_SCOPE
        & pl.col('q_inter_state_grad_overlap_late').is_finite()
    ),
    predicted_direction='a_lt_b',
)
def ddqn_cuts_q_smoothness_asterix_gamma_999(
    stratified_arm_diff_pooled: StratifiedArmDiffPooledResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    source: str = 'q_inter_state_grad_overlap_late',
    stratify_by: tuple[str, ...] = ('env_name',),
    scope_predictor: str = 'jensen_gap',
    min_baseline_predictor: float = 50.0,
    min_seeds_per_arm: int = 5,
    harm_floor: float = -0.5,
    alpha: float = 0.05,
    min_strata: int = 1,
) -> tuple[Verdict, RefutationClass | None]:
    """Edge 1: mechanism-active test on Asterix γ=0.999.

    DDQN's clip reduces Q-function smoothness across consecutive
    trajectory states. `q_inter_state_grad_overlap_late` is the
    inner-product alignment of dQ/dθ between consecutive states
    in the late training half — high overlap = trunk gradients
    propagate coherently (smooth value surface), low overlap =
    states discriminate sharply.

    HELD: pooled Cohen's d ≤ harm_floor AND p < alpha. Predicted
    a_lt_b.
    REFUTED (SIGN_FLIP): d ≥ +harm_floor (DDQN MORE smooth than
    vanilla — would contradict the mechanism)."""
    del treatment_arm, baseline_arm, source, stratify_by
    del scope_predictor, min_baseline_predictor, min_seeds_per_arm
    res = stratified_arm_diff_pooled
    if res.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT, None
    # DL random-effects pool requires ≥2 strata for τ² estimation;
    # at n_strata=1 the pool returns NaN. Fall back to the single
    # stratum's fixed-effects Cohen's d, with the closed-form
    # two-sided p from the z-statistic d / se.
    if res.n_strata == 1 and math.isnan(res.pooled_d):
        s = res.per_stratum[0]
        if math.isnan(s.cohen_d) or math.isnan(s.cohen_se) or s.cohen_se <= 0:
            return Verdict.POWER_INSUFFICIENT, None
        d = s.cohen_d
        z = d / s.cohen_se
        p = 2.0 * (1.0 - float(norm.cdf(abs(z))))
    else:
        d = res.pooled_d
        p = res.pooled_p_value
        if math.isnan(d) or math.isnan(p):
            return Verdict.POWER_INSUFFICIENT, None
    abs_floor = abs(harm_floor)
    if d <= -abs_floor and p < alpha:
        return Verdict.HELD, None
    if d >= abs_floor and p < alpha:
        return Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP
    return Verdict.POWER_INSUFFICIENT, None


# ============ Edge 2: cross-stratum partial-r ↔ jens conditioning ============

@claim_bridge(
    source='q_inter_state_grad_overlap_late',
    target='eval_best_burst_raw_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=(
        _CROSS_STRATUM_GAMMA_SCOPE
        & pl.col('q_inter_state_grad_overlap_late').is_finite()
        & pl.col('jensen_gap').is_finite()
        & pl.col('eval_best_burst_raw_mean').is_finite()
    ),
    predicted_direction='a_gt_b',
)
def q_smoothness_predicts_outcome__cross_stratum(
    cross_stratum_arm_diff_partial_spearman: CrossStratumArmDiffPartialSpearmanResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    predictor: str = 'q_inter_state_grad_overlap_late',
    target: str = 'eval_best_burst_raw_mean',
    confound: str = 'jensen_gap',
    stratify_by: tuple[str, ...] = ('env_name', 'gamma'),
    min_seeds_per_arm: int = 5,
    held_rho: float = 0.5,
    null_rho_band: float = 0.2,
    alpha: float = 0.10,
    min_strata: int = 5,
) -> tuple[Verdict, RefutationClass | None]:
    """Edge 2: cross-stratum independence test.

    Partial Spearman ρ(Δ_smoothness, Δ_outcome | Δ_jens) across
    (env_name, gamma) strata. Tests whether smoothness has
    independent predictive power over outcome AFTER controlling
    for jens — discriminates "independent mediator" from
    "jens-shadow".

    HELD: partial ρ ≥ held_rho with p < alpha (smoothness is an
    independent positive mediator).
    REFUTED (SIGN_FLIP): partial ρ ≤ -held_rho with p < alpha
    (smoothness predicts in the wrong direction — contradicts the
    mechanism story).
    NO_EFFECT (NULL): |partial ρ| ≤ null_rho_band with adequate
    power (smoothness is jens-shadow at the cross-stratum scope).
    """
    del (
        treatment_arm, baseline_arm, predictor, target, confound,
        stratify_by, min_seeds_per_arm,
    )
    res = cross_stratum_arm_diff_partial_spearman
    if res.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT, None
    rho = res.rho
    p = res.p_value
    if math.isnan(rho) or math.isnan(p):
        return Verdict.POWER_INSUFFICIENT, None
    if rho >= held_rho and p < alpha:
        return Verdict.HELD, None
    if rho <= -held_rho and p < alpha:
        return Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP
    if abs(rho) <= null_rho_band and p < alpha:
        return Verdict.NO_EFFECT, RefutationClass.NULL_EFFECT
    return Verdict.POWER_INSUFFICIENT, None


BRIDGES = (
    ddqn_cuts_q_smoothness_asterix_gamma_999,
    q_smoothness_predicts_outcome__cross_stratum,
)
