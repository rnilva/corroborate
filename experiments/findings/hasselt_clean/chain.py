"""Hasselt's chain as a directed walk on the post-eval graph.

Four primary bridges form the connected chain
  `jensen_dormancy_gap → jensen_gap → eval_best_burst_raw_mean`
plus two `do(DDQN)` attacks on the downstream nodes:

  ┌──────────────────────────┐                       ┌──────────────────────────┐
  │   jensen_dormancy_gap    │──B1──►  jensen_gap   │──B2──►  eval_best_burst   │
  └──────────────────────────┘   ▲                   ▲                          │
                                  B3                  B4                         │
                              do(DDQN) ─────────────► do(DDQN) ──────────────────┘

B1, B2 are ASSOCIATIONAL within-cell tests (vanilla-only): the
substrate's theorem (`σ_Q × √(2 log K) ≥ V_jens`) and the
bias→outcome mediator link, respectively. B3, B4 are
INTERVENTIONAL per-stratum tests of the DDQN intervention's bite
on the chain's downstream nodes, scoped on per-cell premise
activation.

Two sibling bridges (B3', B4') replicate B3, B4 with per-stratum
(env-level) conditioning rather than per-cell — robustness check
against the per-cell selection bias that can arise when an env's
surviving premise-active cells are a small fraction of the
stratum (the framework's `min_seeds_per_arm` floor handles this
implicitly; the sibling makes the robustness explicit).

The Finding `finding_hasselt_chain.py` AND-composes all six
bridges. The chain's edges in the post-eval graph form a
connected walk (validatable via `corroborate.graph.causal.is_walk`)
through which the framework's monotone composition propagates
underdetermination: any edge POWER_INSUFFICIENT or NO_EFFECT
walks the cluster verdict to UNDERPOWERED or REFUTED."""
from __future__ import annotations

import math

import polars as pl

from corroborate.analyses.panel.stratified_arm_diff_pooled import (
    StratifiedArmDiffPooledResult,
)
from corroborate.analyses.spearman.partial_spearman import (
    PartialSpearmanResult,
)
from corroborate.bridge.bridge import Direction, Tier, claim_bridge
from corroborate.bridge.verdict import RefutationClass, Verdict

from experiments.findings.ddqn._arms import (
    DDQN_ARM, INTERVENTION, VANILLA_ARM,
)
from experiments.findings.hasselt_clean._scope import (
    CANONICAL_DORMANCY_SCOPE,
    LINK_ACTIVE_PER_CELL,
    PREMISE_ACTIVE_PER_CELL,
    PREMISE_ACTIVE_PER_STRATUM,
    VANILLA_ONLY,
)


# ============================================================
# B1: Theorem edge — Hasselt's σ-floor predicts observed bias
# ============================================================

@claim_bridge(
    source='jensen_dormancy_gap',
    target='jensen_gap',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=CANONICAL_DORMANCY_SCOPE & VANILLA_ONLY,
    predicted_direction='a_lt_b',
)
def hasselt_floor_predicts_observed_bias__vanilla(
    partial_spearman: PartialSpearmanResult,
    *,
    x: str = 'jensen_dormancy_gap',
    y: str = 'jensen_gap',
    conditioning: tuple[str, ...] = (),
    stratify_by: str = 'env_name',
    min_stratum_size: int = 30,
    rho_threshold_held: float = -0.3,
    p_threshold_held: float = 0.05,
    null_threshold: float = 0.1,
    sign_flip_threshold: float = 0.3,
) -> tuple[Verdict, RefutationClass | None]:
    """Theorem edge. Under vanilla baseline, the Hasselt
    structural floor encoded as `jensen_dormancy_gap = max(0,
    σ_Q × √(2 log K) − observed_bias)` is by construction
    inversely related to `jensen_gap`: cells at the saturated
    floor (dormancy_gap = 0) carry the Jensen-bias the theorem
    describes; cells far below it have near-zero observed bias.

    Tests this as a stratified partial-Spearman across the
    canonical-dormancy panel."""
    del x, y, conditioning, stratify_by, min_stratum_size
    rho = partial_spearman.rho_pooled
    p = partial_spearman.p_value
    if math.isnan(rho) or math.isnan(p):
        return Verdict.POWER_INSUFFICIENT, None
    if rho <= rho_threshold_held and p <= p_threshold_held:
        return Verdict.HELD, None
    if rho >= sign_flip_threshold:
        return Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP
    if abs(rho) <= null_threshold:
        return Verdict.NO_EFFECT, RefutationClass.NULL_EFFECT
    return Verdict.POWER_INSUFFICIENT, None


# ============================================================
# B2: Link edge — observed bias predicts worse outcome (vanilla)
# ============================================================

@claim_bridge(
    source='jensen_gap',
    target='eval_best_burst_raw_mean',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=CANONICAL_DORMANCY_SCOPE & VANILLA_ONLY,
    predicted_direction='a_lt_b',
)
def bias_predicts_worse_outcome__vanilla(
    partial_spearman: PartialSpearmanResult,
    *,
    x: str = 'jensen_gap',
    y: str = 'eval_best_burst_raw_mean',
    conditioning: tuple[str, ...] = (),
    stratify_by: str = 'env_name',
    min_stratum_size: int = 30,
    rho_threshold_held: float = -0.3,
    p_threshold_held: float = 0.05,
    null_threshold: float = 0.1,
    sign_flip_threshold: float = 0.3,
) -> tuple[Verdict, RefutationClass | None]:
    """Link edge. Under vanilla baseline, higher observed bias
    predicts lower outcome. This is the empirical mediator-link
    test — whether the bias-reduction-could-help premise has
    bite at the link layer.

    History note: this link is known to be env-conditional
    (FINDINGS.md revisions 10-11). Pooling across the 5-env
    dormancy panel may produce a noisy estimate."""
    del x, y, conditioning, stratify_by, min_stratum_size
    rho = partial_spearman.rho_pooled
    p = partial_spearman.p_value
    if math.isnan(rho) or math.isnan(p):
        return Verdict.POWER_INSUFFICIENT, None
    if rho <= rho_threshold_held and p <= p_threshold_held:
        return Verdict.HELD, None
    if rho >= sign_flip_threshold:
        return Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP
    if abs(rho) <= null_threshold:
        return Verdict.NO_EFFECT, RefutationClass.NULL_EFFECT
    return Verdict.POWER_INSUFFICIENT, None


# ============================================================
# B3: Mech edge (PER-CELL) — DDQN reduces bias where premise active
# ============================================================

@claim_bridge(
    source=INTERVENTION,
    target='jensen_gap',
    direction=Direction.INVERSE,
    tier=Tier.INTERVENTIONAL,
    scope=CANONICAL_DORMANCY_SCOPE & PREMISE_ACTIVE_PER_CELL,
    predicted_direction='a_lt_b',
)
def intervention_reduces_bias__premise_active_per_cell(
    stratified_arm_diff_pooled: StratifiedArmDiffPooledResult,
    *,
    source: str = 'jensen_gap',
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    stratify_by: tuple[str, ...] = ('env_name',),
    min_seeds_per_arm: int = 5,
) -> tuple[Verdict, RefutationClass | None]:
    """Mechanism edge, per-cell premise conditioning. Where
    `jensen_dormancy_gap == 0` (Hasselt premise locally
    saturated), DDQN reduces `jensen_gap` relative to vanilla.
    Stratified by env, pooled via independent-samples Cohen's d.

    Per-cell conditioning is theorem-aligned (the σ-floor is a
    per-cell quantity). Sibling B3' replicates with per-stratum
    conditioning for robustness."""
    del (
        source, treatment_arm, baseline_arm,
        stratify_by, min_seeds_per_arm,
    )
    return stratified_arm_diff_pooled.verdict, None


# ============================================================
# B3': Mech edge (PER-STRATUM) — robustness sibling of B3
# ============================================================

@claim_bridge(
    source=INTERVENTION,
    target='jensen_gap',
    direction=Direction.INVERSE,
    tier=Tier.INTERVENTIONAL,
    scope=CANONICAL_DORMANCY_SCOPE & PREMISE_ACTIVE_PER_STRATUM,
    predicted_direction='a_lt_b',
)
def intervention_reduces_bias__premise_active_per_stratum(
    stratified_arm_diff_pooled: StratifiedArmDiffPooledResult,
    *,
    source: str = 'jensen_gap',
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    stratify_by: tuple[str, ...] = ('env_name',),
    min_seeds_per_arm: int = 5,
) -> tuple[Verdict, RefutationClass | None]:
    """Mechanism edge, per-stratum premise conditioning. Includes
    all cells (active + dormant) from envs where the per-env
    median `jensen_dormancy_gap` is zero (premise broadly
    active). Avoids per-cell selection bias.

    Robustness sibling for B3. If B3 and B3' agree → robust
    under both conditioning shapes; if they diverge → the
    per-cell selection bias is material."""
    del (
        source, treatment_arm, baseline_arm,
        stratify_by, min_seeds_per_arm,
    )
    return stratified_arm_diff_pooled.verdict, None


# ============================================================
# B4: Outcome edge (PER-CELL) — DDQN helps outcome where chain holds
# ============================================================

@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_raw_mean',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=(
        CANONICAL_DORMANCY_SCOPE
        & PREMISE_ACTIVE_PER_CELL
        & LINK_ACTIVE_PER_CELL
    ),
    predicted_direction='a_gt_b',
)
def intervention_helps_outcome__chain_holds_per_cell(
    stratified_arm_diff_pooled: StratifiedArmDiffPooledResult,
    *,
    source: str = 'eval_best_burst_raw_mean',
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    stratify_by: tuple[str, ...] = ('env_name',),
    min_seeds_per_arm: int = 5,
) -> tuple[Verdict, RefutationClass | None]:
    """Outcome edge, per-cell conditioning on premise + link
    activation. Where both upstream conditions hold (premise
    saturated AND bootstrap_fraction > 0.5), DDQN improves
    outcome relative to vanilla."""
    del (
        source, treatment_arm, baseline_arm,
        stratify_by, min_seeds_per_arm,
    )
    return stratified_arm_diff_pooled.verdict, None


# ============================================================
# B4': Outcome edge (PER-STRATUM) — robustness sibling of B4
# ============================================================

@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_raw_mean',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=(
        CANONICAL_DORMANCY_SCOPE
        & PREMISE_ACTIVE_PER_STRATUM
        & (pl.col('bootstrap_fraction').median().over(['env_name']) > 0.5)
    ),
    predicted_direction='a_gt_b',
)
def intervention_helps_outcome__chain_holds_per_stratum(
    stratified_arm_diff_pooled: StratifiedArmDiffPooledResult,
    *,
    source: str = 'eval_best_burst_raw_mean',
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    stratify_by: tuple[str, ...] = ('env_name',),
    min_seeds_per_arm: int = 5,
) -> tuple[Verdict, RefutationClass | None]:
    """Outcome edge, per-stratum conditioning. Includes all cells
    from envs where the per-env median premise + link are both
    active.

    Robustness sibling for B4."""
    del (
        source, treatment_arm, baseline_arm,
        stratify_by, min_seeds_per_arm,
    )
    return stratified_arm_diff_pooled.verdict, None


BRIDGES = (
    hasselt_floor_predicts_observed_bias__vanilla,
    bias_predicts_worse_outcome__vanilla,
    intervention_reduces_bias__premise_active_per_cell,
    intervention_reduces_bias__premise_active_per_stratum,
    intervention_helps_outcome__chain_holds_per_cell,
    intervention_helps_outcome__chain_holds_per_stratum,
)


__all__ = [
    'BRIDGES',
    'hasselt_floor_predicts_observed_bias__vanilla',
    'bias_predicts_worse_outcome__vanilla',
    'intervention_reduces_bias__premise_active_per_cell',
    'intervention_reduces_bias__premise_active_per_stratum',
    'intervention_helps_outcome__chain_holds_per_cell',
    'intervention_helps_outcome__chain_holds_per_stratum',
]
