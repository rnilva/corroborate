"""Hasselt's chain as a directed walk on the post-eval graph.

Four bridges form the connected chain
  `jensen_dormancy_gap → jensen_gap → eval_best_burst_raw_mean`
with `do(DDQN)` attacks on the two downstream nodes:

  ┌──────────────────────────┐                       ┌──────────────────────────┐
  │   jensen_dormancy_gap    │──B1──►  jensen_gap   │──B2──►  eval_best_burst   │
  └──────────────────────────┘   ▲                   ▲                          │
                                  B3                  B4                         │
                              do(DDQN) ─────────────► do(DDQN) ──────────────────┘

B1, B2 are ASSOCIATIONAL within-cell tests (vanilla-only): the
substrate's theorem (`σ_Q × √(2 log K) ≥ V_jens`) and the
bias→outcome mediator link, respectively. B3, B4 are
INTERVENTIONAL per-stratum tests of DDQN's bite on the chain's
downstream nodes.

**Per-stratum scope is the principled choice for the
intervention edges.** Per-cell conditioning on
`jensen_dormancy_gap == 0` (premise activation) is a
*post-treatment* scope filter: the DDQN intervention *itself*
changes which cells satisfy `gap == 0` (DDQN reduces observed
bias → more cells fall below the σ-floor → more cells are
dormant). Conditioning on a post-treatment variable
(equivalent to a collider in the chain's DAG) introduces
M-bias: the surviving "premise-active" subset under DDQN is a
DDQN-resistant cohort, not a comparable cell population.

Acrobot γ=0.999 surfaces this directly: under per-cell scope,
DDQN's per-arm jensen_gap reads HIGHER than vanilla's (15.9 vs
12.6) — but this is the selection effect of comparing
"DDQN-active" (DDQN failed to push below floor) against
"vanilla-active" (typical high-bias cells). Under per-stratum
scope (env median premise-active → include all cells from the
env), the same data shows DDQN's net effect ≈ 0 — the honest
answer: at Acrobot γ=0.999 (solved by both arms, V_eb≈-76 =
solved ceiling), Hasselt's mech has no failure mode left to
clip.

The Finding `finding_hasselt_chain_explicit.py` AND-composes
these four bridges; the chain's edges in the post-eval graph
form a connected walk (validatable via
`corroborate.graph.causal.is_walk`)."""
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
    (FINDINGS.md revisions 10-11). Pooling across the canonical
    dormancy panel may produce a modest pooled estimate."""
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
# B3: Mech edge — DDQN reduces bias (per-stratum)
# ============================================================
#
# Per-stratum premise activation (`env median jdg == 0`)
# avoids the post-treatment-conditioning bias of per-cell
# `jdg == 0`. DDQN's intervention itself shifts which cells
# fall below the σ-floor; the per-cell scope would select a
# DDQN-resistant cohort. The per-stratum filter keeps all
# cells from envs where the premise is broadly active, then
# pools the per-arm contrast across them.

@claim_bridge(
    source=INTERVENTION,
    target='jensen_gap',
    direction=Direction.INVERSE,
    tier=Tier.INTERVENTIONAL,
    scope=CANONICAL_DORMANCY_SCOPE & PREMISE_ACTIVE_PER_STRATUM,
    predicted_direction='a_lt_b',
)
def intervention_reduces_bias__premise_active(
    stratified_arm_diff_pooled: StratifiedArmDiffPooledResult,
    *,
    source: str = 'jensen_gap',
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    stratify_by: tuple[str, ...] = ('env_name',),
    min_seeds_per_arm: int = 5,
) -> tuple[Verdict, RefutationClass | None]:
    """Mechanism edge. Where the Hasselt premise is broadly
    active at the env level (median `jensen_dormancy_gap == 0`
    across the env's cells), DDQN reduces observed `jensen_gap`
    relative to vanilla. Stratified by env, pooled via
    independent-samples Cohen's d under DL random-effects.

    Per-stratum (rather than per-cell) conditioning is the
    principled choice: DDQN's intervention itself affects which
    cells satisfy `gap == 0`, so per-cell premise scope
    introduces post-treatment selection bias (chain-internal
    collider). Per-stratum filter keeps all cells from envs
    where the substrate's dormancy regime is broadly active."""
    del (
        source, treatment_arm, baseline_arm,
        stratify_by, min_seeds_per_arm,
    )
    return stratified_arm_diff_pooled.verdict, None


# ============================================================
# B4: Outcome edge — DDQN helps outcome (per-stratum)
# ============================================================

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
def intervention_helps_outcome__chain_holds(
    stratified_arm_diff_pooled: StratifiedArmDiffPooledResult,
    *,
    source: str = 'eval_best_burst_raw_mean',
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    stratify_by: tuple[str, ...] = ('env_name',),
    min_seeds_per_arm: int = 5,
) -> tuple[Verdict, RefutationClass | None]:
    """Outcome edge. Where both upstream conditions hold at the
    env level — premise broadly active (`median jdg == 0`) AND
    link broadly active (`median bootstrap_fraction > 0.5`) —
    DDQN improves outcome relative to vanilla.

    Both conditions are per-stratum (env-median) rather than
    per-cell, avoiding the post-treatment scope bias the
    per-cell formulation would introduce."""
    del (
        source, treatment_arm, baseline_arm,
        stratify_by, min_seeds_per_arm,
    )
    return stratified_arm_diff_pooled.verdict, None


BRIDGES = (
    hasselt_floor_predicts_observed_bias__vanilla,
    bias_predicts_worse_outcome__vanilla,
    intervention_reduces_bias__premise_active,
    intervention_helps_outcome__chain_holds,
)


__all__ = [
    'BRIDGES',
    'hasselt_floor_predicts_observed_bias__vanilla',
    'bias_predicts_worse_outcome__vanilla',
    'intervention_reduces_bias__premise_active',
    'intervention_helps_outcome__chain_holds',
]
