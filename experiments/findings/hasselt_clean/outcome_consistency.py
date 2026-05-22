"""DDQN's total effect on outcome — an *independent* empirical
claim, separate from the Hasselt chain.

The Hasselt chain (`chain.py`: B1 theorem + B2 link + B3 mech)
establishes the bias-clip *mechanism*: σ-floor predicts bias,
bias correlates with outcome, DDQN reduces bias. Each edge is a
falsifiable empirical claim about the chain's structure.

DDQN's *total effect* on outcome — whether the intervention
ends up improving the agent's actual return — is a structurally
distinct claim. Pearl's mediation framing makes this explicit:

  total effect = direct effect + indirect effect (via mediator)

The chain B3 + B2 quantifies the INDIRECT path (DDQN → bias →
outcome). The TOTAL effect captures direct + indirect + any
non-mediated pathways the intervention triggers. Even when the
chain's three edges all HELD, the total effect may be null,
positive, or negative depending on what other channels DDQN
operates through.

This bridge tests the cross-env consistency of the total effect
*independently* — not as a chain edge. Its verdict speaks to the
practical question "is the bias-clip mechanism's net policy
benefit consistent across environments?" — separate from "does
the bias-clip mechanism exist?"."""
from __future__ import annotations

import math

import polars as pl

from corroborate.analyses.panel.cross_env_probability_of_improvement import (
    CrossEnvProbabilityOfImprovementResult,
)
from corroborate.bridge.bridge import Direction, Tier, claim_bridge
from corroborate.bridge.verdict import RefutationClass, Verdict

from experiments.findings.ddqn._arms import (
    DDQN_ARM, INTERVENTION, VANILLA_ARM,
)
from experiments.findings.hasselt_clean._scope import (
    CANONICAL_DORMANCY_SCOPE,
    PREMISE_ACTIVE_PER_STRATUM,
)


@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_raw_mean',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=(
        CANONICAL_DORMANCY_SCOPE
        & PREMISE_ACTIVE_PER_STRATUM
        & (pl.col('bootstrap_fraction').median().over(['corpus', 'gamma']) > 0.5)
    ),
    predicted_direction='a_gt_b',
)
def ddqn_helps_outcome__consistently_cross_env(
    cross_env_probability_of_improvement: CrossEnvProbabilityOfImprovementResult,
    *,
    source: str = 'eval_best_burst_raw_mean',
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    stratify_by: tuple[str, ...] = ('env_name', 'gamma'),
    min_seeds_per_arm: int = 5,
    min_strata: int = 5,
    p_threshold_held: float = 0.05,
    p_threshold_pi: float = 0.15,
    p_xy_substantive_delta: float = 0.02,
) -> tuple[Verdict, RefutationClass | None]:
    """Cross-env consistency of DDQN's total effect on outcome via
    Mann-Whitney `P(DDQN > vanilla)` aggregated across strata.

    NOT a chain edge — Hasselt's theorem doesn't predict that
    the bias-clip mechanism nets out positive at outcome. Tests
    the empirical claim "DDQN improves outcome consistently
    across (env, γ) strata where the chain's premise and link
    are broadly active".

    Why P(X > Y) not Cohen-d sign-test (Agarwal et al. 2021,
    NeurIPS Outstanding Paper):
      - Scale-invariant: ordinal comparison handles Acrobot
        returns ~-75, LunarLander ~-78–+77, success-rate envs
        ~0–1 on the same [0, 1] probability scale.
      - Saturation-neutral: when both arms hit ceiling,
        per-stratum P_xy ≈ 0.5 — contributes neutrally to mean
        without an explicit saturation guard.
      - Magnitude-aware: uses per-stratum (P_xy − 0.5)
        deviations, not just direction (which the previous
        `cross_env_consistency_binomial` sign-test discards).

    Inference: exact sign-permutation on (P_xy − 0.5) — exact at
    any n (the 13-stratum hasselt_clean panel falls in the n<20
    regime where bootstrap-percentile CIs have known coverage
    error; permutation is rigorous). Bootstrap CI on the result
    is descriptive only at this n; reported for diagnostics."""
    del (
        source, treatment_arm, baseline_arm,
        stratify_by, min_seeds_per_arm,
    )
    if cross_env_probability_of_improvement.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT, None
    p = cross_env_probability_of_improvement.p_permutation
    if math.isnan(p):
        return Verdict.POWER_INSUFFICIENT, None
    p_xy_mean = cross_env_probability_of_improvement.p_xy_mean
    substantive_positive = p_xy_mean >= 0.5 + p_xy_substantive_delta
    substantive_negative = p_xy_mean <= 0.5 - p_xy_substantive_delta
    if p <= p_threshold_held and substantive_positive:
        return Verdict.HELD, None
    if p <= p_threshold_pi:
        return Verdict.POWER_INSUFFICIENT, None
    if substantive_negative:
        return Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP
    if not substantive_positive and not substantive_negative:
        return Verdict.NO_EFFECT, RefutationClass.NULL_EFFECT
    return Verdict.POWER_INSUFFICIENT, None


BRIDGES = (ddqn_helps_outcome__consistently_cross_env,)


__all__ = [
    'BRIDGES',
    'ddqn_helps_outcome__consistently_cross_env',
]
