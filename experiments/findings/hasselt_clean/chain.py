"""Hasselt's chain as a directed walk on the post-eval graph,
using cross-env consistency for the intervention edges.

  ┌──────────────────────────┐                       ┌──────────────────────────┐
  │   jensen_dormancy_gap    │──B1──►  jensen_gap   │──B2──►  eval_best_burst   │
  └──────────────────────────┘   ▲                   ▲                          │
                                  B3                  B4                         │
                              do(DDQN) ─────────────► do(DDQN) ──────────────────┘

B1, B2 are ASSOCIATIONAL within-cell tests (vanilla-only): the
substrate's theorem (`σ_Q × √(2 log K) ≥ V_jens`) and the
bias→outcome mediator link. B3, B4 are INTERVENTIONAL
cross-env consistency tests via per-env Cohen's d sign-test.

**Why cross-env consistency, not random-effects pool.** The
chain edge's claim — "DDQN ∧ ¬dormant → bias↓" — is a
*directional consistency* claim across envs, not a
*population-average extrapolation* claim. Random-effects
pooling (DerSimonian-Laird) assumes the strata are exchangeable
draws from a population with `g_i ~ N(μ, τ²)`. RL environments
aren't exchangeable: they differ in network class (CNN vs MLP),
Q-magnitude (Asterix d=-8.9 vs Acrobot d=-0.01 — 800× scale
range), reward sparsity. The pool's prediction interval
correctly refuses extrapolation under this heterogeneity but
buries the substantive cross-env-directional-consistency claim
under a NO_EFFECT verdict.

The cross-env consistency primitive
(`cross_env_consistency_binomial`) tests the directional claim
directly: count envs in the predicted direction; binomial
sign-test. Doesn't require exchangeability; doesn't extrapolate
to a population; tests exactly the conditional claim the
substrate author actually wants.

(The pool-based bridges are preserved at
`experiments/findings/hasselt_clean/_failed_pool/` for the
pedagogical "what doesn't work and why" story.)"""
from __future__ import annotations

import math

from corroborate.analyses.panel.cross_env_consistency_binomial import (
    CrossEnvConsistencyBinomialResult,
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
# B2: Link edge — observed bias predicts outcome (vanilla)
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
    predicts lower outcome — the bias-reduction-could-help
    premise at the link layer."""
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
# B3: Mech edge — DDQN reduces bias consistently across envs
# ============================================================

@claim_bridge(
    source=INTERVENTION,
    target='jensen_gap',
    direction=Direction.INVERSE,
    tier=Tier.INTERVENTIONAL,
    scope=CANONICAL_DORMANCY_SCOPE & PREMISE_ACTIVE_PER_STRATUM,
    predicted_direction='a_lt_b',
)
def ddqn_reduces_bias__consistently_cross_env(
    cross_env_consistency_binomial: CrossEnvConsistencyBinomialResult,
    *,
    source: str = 'jensen_gap',
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    stratify_by: tuple[str, ...] = ('env_name', 'gamma'),
    null_floor: float = 0.0,
    min_seeds_per_arm: int = 5,
    min_strata: int = 5,
    p_threshold_held: float = 0.05,
    p_threshold_pi: float = 0.15,
) -> tuple[Verdict, RefutationClass | None]:
    """Mechanism edge as a cross-env consistency claim:
    *"DDQN ∧ ¬dormant → bias reduced"* tested as
    *"in every env where premise is broadly active, DDQN's
    Cohen's d on `jensen_gap` is negative"*.

    Binomial sign-test on the per-env d panel. HELD when the
    sign-test p ≤ 0.05 (under default null_floor=0.0, n=9 envs
    in predicted direction gives p ≈ 0.002).

    Cross-env consistency doesn't require env-exchangeability —
    it tests directionality, not extrapolation. The right tool
    for a conditional claim across heterogeneous strata."""
    del (
        source, treatment_arm, baseline_arm,
        stratify_by, null_floor, min_seeds_per_arm,
    )
    if cross_env_consistency_binomial.n_strata_total < min_strata:
        return Verdict.POWER_INSUFFICIENT, None
    p = cross_env_consistency_binomial.p_value
    if math.isnan(p):
        return Verdict.POWER_INSUFFICIENT, None
    n_total = cross_env_consistency_binomial.n_strata_total
    n_signed = cross_env_consistency_binomial.n_signed_predicted
    if p <= p_threshold_held:
        return Verdict.HELD, None
    if p <= p_threshold_pi:
        return Verdict.POWER_INSUFFICIENT, None
    if n_total > 0 and (n_total - n_signed) / n_total >= 0.70:
        return Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP
    if n_total > 0 and n_signed / n_total <= 0.60:
        return Verdict.NO_EFFECT, RefutationClass.NULL_EFFECT
    return Verdict.POWER_INSUFFICIENT, None


# ============================================================
# B2-late30: Link edge — bias predicts late-window outcome (vanilla)
# Sibling of `bias_predicts_worse_outcome__vanilla` under the
# late30 per-run scalar. Identical decision logic; only the
# outcome column differs. See REPORT.md §3.4-bis for the
# dual-metric framing.
# ============================================================

@claim_bridge(
    source='jensen_gap',
    target='eval_late_burst_raw_mean',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=CANONICAL_DORMANCY_SCOPE & VANILLA_ONLY,
    predicted_direction='a_lt_b',
)
def bias_predicts_worse_outcome__vanilla__late30(
    partial_spearman: PartialSpearmanResult,
    *,
    x: str = 'jensen_gap',
    y: str = 'eval_late_burst_raw_mean',
    conditioning: tuple[str, ...] = (),
    stratify_by: str = 'env_name',
    min_stratum_size: int = 30,
    rho_threshold_held: float = -0.3,
    p_threshold_held: float = 0.05,
    null_threshold: float = 0.1,
    sign_flip_threshold: float = 0.3,
) -> tuple[Verdict, RefutationClass | None]:
    """Late-window sibling of `bias_predicts_worse_outcome__vanilla`.

    Uses `eval_late_burst_raw_mean` (last 30% of bursts) as the
    outcome column instead of `eval_best_burst_raw_mean` (peak
    burst). The peak version aligns with the DDQN-paper reporting
    protocol; this late30 version aligns with the DDQN-paper
    stability narrative ('reducing overestimations can significantly
    benefit the stability of learning') and Agarwal-2021's
    late-window aggregate. Both verdicts are reported; their
    disagreement is the methodological finding."""
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


BRIDGES = (
    hasselt_floor_predicts_observed_bias__vanilla,
    bias_predicts_worse_outcome__vanilla,
    bias_predicts_worse_outcome__vanilla__late30,
    ddqn_reduces_bias__consistently_cross_env,
)


__all__ = [
    'BRIDGES',
    'hasselt_floor_predicts_observed_bias__vanilla',
    'bias_predicts_worse_outcome__vanilla',
    'bias_predicts_worse_outcome__vanilla__late30',
    'ddqn_reduces_bias__consistently_cross_env',
]
