"""Pool-based intervention bridges — pedagogically preserved.

The original Hasselt-chain B3/B4 used `stratified_arm_diff_pooled`
(DerSimonian-Laird random-effects pool of Cohen's d). Both fired
NO_EFFECT/NULL_EFFECT under the framework's PI-based discipline
on the canonical 9-env panel, despite B3's pooled d=-1.90 being
huge — the prediction interval bracketed zero under I²=0.97.

The substantive issue: **random-effects pooling assumes
exchangeable strata**, which RL environments aren't. The chain
edge's claim is consistency across heterogeneous envs, not
population-average extrapolation. The main `chain.py` uses
`cross_env_consistency_binomial` (sign-test) for B3/B4 — the
right claim shape.

This file preserves the pool-based attempt for pedagogy. The
verdicts here are stable (REFUTED at cluster level) and the
substantive content is the framework-honest "this is the wrong
claim shape" story."""
from __future__ import annotations

from corroborate.analyses.panel.stratified_arm_diff_pooled import (
    StratifiedArmDiffPooledResult,
)
from corroborate.bridge.bridge import Direction, Tier, claim_bridge
from corroborate.bridge.verdict import RefutationClass, Verdict
import polars as pl

from experiments.findings.ddqn._arms import (
    DDQN_ARM, INTERVENTION, VANILLA_ARM,
)
from experiments.findings.hasselt_clean._scope import (
    CANONICAL_DORMANCY_SCOPE,
    PREMISE_ACTIVE_PER_STRATUM,
)


@claim_bridge(
    source=INTERVENTION,
    target='jensen_gap',
    direction=Direction.INVERSE,
    tier=Tier.INTERVENTIONAL,
    scope=CANONICAL_DORMANCY_SCOPE & PREMISE_ACTIVE_PER_STRATUM,
    predicted_direction='a_lt_b',
)
def intervention_reduces_bias__pool_inadequate(
    stratified_arm_diff_pooled: StratifiedArmDiffPooledResult,
    *,
    source: str = 'jensen_gap',
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    stratify_by: tuple[str, ...] = ('env_name', 'gamma'),
    min_seeds_per_arm: int = 5,
) -> tuple[Verdict, RefutationClass | None]:
    """Mech edge via random-effects pool — pedagogically preserved.

    Fires NO_EFFECT/NULL_EFFECT on the canonical-dormancy panel.
    Not because DDQN has no effect (pooled d=-1.90, 9/9 envs in
    predicted direction), but because the random-effects PI
    brackets zero under I²=0.97. The pool refuses extrapolation
    to an 11th env from a population that's structurally
    non-exchangeable."""
    del (
        source, treatment_arm, baseline_arm,
        stratify_by, min_seeds_per_arm,
    )
    return stratified_arm_diff_pooled.verdict, None


@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_raw_mean',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=(
        CANONICAL_DORMANCY_SCOPE
        & PREMISE_ACTIVE_PER_STRATUM
        & (pl.col('bootstrap_fraction').median().over(['corpus']) > 0.5)
    ),
    predicted_direction='a_gt_b',
)
def intervention_helps_outcome__pool_inadequate(
    stratified_arm_diff_pooled: StratifiedArmDiffPooledResult,
    *,
    source: str = 'eval_best_burst_raw_mean',
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    stratify_by: tuple[str, ...] = ('env_name', 'gamma'),
    min_seeds_per_arm: int = 5,
) -> tuple[Verdict, RefutationClass | None]:
    """Outcome edge via random-effects pool — pedagogically
    preserved. Fires NO_EFFECT on the canonical-dormancy panel
    (pooled d=+0.47, p=0.17, I²=0.97). Here the outcome's
    heterogeneity is genuine (5 envs help, 4 harm/null); the
    pool returns null reasonably."""
    del (
        source, treatment_arm, baseline_arm,
        stratify_by, min_seeds_per_arm,
    )
    return stratified_arm_diff_pooled.verdict, None


BRIDGES = (
    intervention_reduces_bias__pool_inadequate,
    intervention_helps_outcome__pool_inadequate,
)


__all__ = [
    'BRIDGES',
    'intervention_reduces_bias__pool_inadequate',
    'intervention_helps_outcome__pool_inadequate',
]
