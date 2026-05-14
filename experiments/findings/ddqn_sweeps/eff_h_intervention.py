"""Eff_h intervention mediator test on GOAL-polarity γ-sweep.

The canonical `eff_h_mediates_g_link__{goal,survival}_envs` bridges
in `ddqn/mediation.py` are POLARITY-STRUCTURE checks — at canonical
γ=0.99 (pinned), eff_h = 1/(1−γ·bf) is a monotone function of bf
(bootstrap_fraction = 1 − mean(done)). Within env, ρ(eff_h, outcome)
is structurally determined by polarity: on SURVIVAL-polarity envs
longer episodes = more cumulative reward (positive ρ); on
GOAL-polarity envs faster solving = shorter episodes = more reward
(negative ρ). The partial-Spearman conditioning on jens doesn't
break this because jens is independent of bf.

This bridge tests the SUBSTANTIVE mediator claim: across (env, γ)
strata where γ varies in {0.99, 0.995, 0.999}, does DDQN's
intervention on effective_horizon (Δ_eff_h = mean(DDQN eff_h) −
mean(vanilla eff_h)) predict DDQN's intervention on outcome
(Δ_outcome)? With γ as the manipulated variable creating real
predictor variance, eff_h becomes an interpretable dose-response
covariate rather than a polarity-tautology proxy.

**Predicted direction**: ρ < 0. On GOAL-polarity envs, better DDQN
policy → faster solving → lower bf → MORE-NEGATIVE Δ_eff_h.
Simultaneously, better DDQN policy → higher cumulative reward →
MORE-POSITIVE Δ_outcome. The bigger DDQN's policy improvement, the
more negative Δ_eff_h AND the more positive Δ_outcome → negative
correlation across strata.

**Why GOAL only, not SURVIVAL**: γ-sweep cells exist for 4
GOAL-polarity envs (Acrobot, FourRooms, MetaMaze, MountainCar at
γ ∈ {0.99, 0.995, 0.999}) — 8–10 (env, γ) strata available. No
γ-sweep cells exist for SURVIVAL-polarity envs at canonical, so
no intervention data is available to falsify the polarity-structure
finding there. SURVIVAL mediator would require a designed
truncation-wrapper sweep (out of scope here).

**Scope**: ddqn_sweeps γ-sweep cohort with GOAL polarity. Strata
(env, γ); each contributes one (Δ_eff_h, Δ_outcome) point.
Spearman ρ across strata, calibrated for n_strata≥8.
"""
from __future__ import annotations

import polars as pl

from corroborate.analyses.cross_stratum_arm_diff_slope import (
    CrossStratumArmDiffSlopeResult,
)
from corroborate.bridge.bridge import Direction, Tier, claim_bridge
from corroborate.bridge.predicates import finite, finite_lt
from corroborate.bridge.verdict import RefutationClass, Verdict

from experiments.findings.ddqn._arms import (
    DDQN_ARM, INTERVENTION, VANILLA_ARM,
)
from experiments.findings.ddqn._verdicts import (
    cross_stratum_signed_spearman_verdict,
)


_GOAL_GAMMA_SWEEP_SCOPE: pl.Expr = (
    pl.col('gamma').is_in([0.99, 0.995, 0.999])
    & finite_lt('env_reward_polarity', -0.3)
    & finite('effective_horizon')
    & finite('eval_best_burst_raw_mean')
    & ((pl.col('n_step') == 1) | pl.col('n_step').is_null())
    & pl.col('action_duplicate_k').is_null()
    & (pl.col('reward_scale').is_null() | (pl.col('reward_scale') == 1.0))
    & pl.col('target_sync.tau').is_null()
)


@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_raw_mean',
    direction=Direction.INVERSE,
    tier=Tier.INTERVENTIONAL,
    scope=_GOAL_GAMMA_SWEEP_SCOPE,
    predicted_direction='a_lt_b',
)
def eff_h_intervention_mediates_outcome__goal_envs_gamma_sweep(
    cross_stratum_arm_diff_slope: CrossStratumArmDiffSlopeResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    predictor: str = 'effective_horizon',
    target: str = 'eval_best_burst_raw_mean',
    stratify_by: tuple[str, ...] = ('env_name', 'gamma'),
    min_seeds_per_arm: int = 5,
    rho_threshold_held: float = 0.6,
    p_threshold: float = 0.05,
    null_threshold: float = 0.2,
    sign_flip_threshold: float = 0.5,
    min_strata: int = 8,
) -> tuple[Verdict, RefutationClass | None]:
    """Cross-(env, γ) Spearman ρ of per-stratum arm-diff Δ_eff_h vs
    arm-diff Δ_outcome. Predicted ρ ≤ −0.6 on GOAL polarity.

    Verdict matrix (per `cross_stratum_signed_spearman_verdict`):
      HELD                  : ρ ≤ −0.6 AND p ≤ 0.05
      NO_EFFECT (SIGN_FLIP) : ρ ≥ +0.5 (decisive wrong-direction)
      NO_EFFECT (NULL_EFFECT) : |ρ| < 0.2 (clean null both directions)
      POWER_INSUFFICIENT    : in-between, or n_strata < 8

    Substantive interpretation of HELD: across γ-sweep strata,
    DDQN's effect on eff_h tracks its effect on outcome — i.e.,
    eff_h IS a meaningful mediator of γ-induced chain-depth
    amplification (Hasselt 2010 chain claim, axis ii of
    unified-degeneracy theory).

    Substantive interpretation of NULL_EFFECT: γ-sweep doesn't
    produce a coherent (Δ_eff_h, Δ_outcome) dose-response — the
    canonical polarity-structure correlation `eff_h_polarity_
    structure_check__goal_envs` was structural, not mediated.

    `cross_stratum_arm_diff_slope` uses independent-samples per-arm
    means (no seed-pairing). Both arms' eff_h means are computed
    independently from their seed cohorts; the Δ is the difference."""
    del (
        treatment_arm, baseline_arm, predictor, target,
        stratify_by, min_seeds_per_arm,
    )
    return cross_stratum_signed_spearman_verdict(
        cross_stratum_arm_diff_slope,
        sign=-1,
        rho_threshold_held=rho_threshold_held,
        p_threshold=p_threshold,
        null_threshold=null_threshold,
        sign_flip_threshold=sign_flip_threshold,
        min_strata=min_strata,
    )


BRIDGES = (
    eff_h_intervention_mediates_outcome__goal_envs_gamma_sweep,
)
