"""Three-conditions bridges — declarative tests of necessary
conditions for DDQN's outcome benefit.

Each bridge scopes to a specific intervention-state slice and
tests whether the predicted condition-direction holds:

- **C1**: scope = FR γ=0.999 × MLP[64,64] × no-shaping.
  Direction.INVERSE on `jensen_gap` (DDQN reduces bias). HELD
  when DDQN's per-k stratum cohen_d is negative.

- **C2**: scope = MountainCar γ=0.999 × LINEAR FA × no-shaping.
  predicted_direction='null' on `jensen_gap` — FA-capped vanilla
  can't develop max-bias for DDQN to reduce. HELD when stratum
  cohen_d is in the null band.

- **C3**: scope = FourRooms γ=0.999 × MLP[64,64] × SHAPED.
  predicted_direction='null' on `eval_best_burst_mean` — shaping
  adds dense policy signal that overrides Q-noise. DDQN still
  cuts Q-bias but no longer translates to outcome. HELD when
  cohen_d in null band (or significantly negative — clip-wedge
  hurts).

The cluster Finding asserts all three jointly required. Each
bridge tests ONE condition's intervention; the others are held
fixed by the scope filter."""
from __future__ import annotations

import math

import polars as pl

from corroborate.analyses.stratified_arm_diff_pooled import (
    StratifiedArmDiffPooledResult,
)
from corroborate.bridge.bridge import Direction, Tier, claim_bridge
from corroborate.bridge.predicates import finite
from corroborate.bridge.verdict import RefutationClass, Verdict

from experiments.findings.ddqn_three_conditions._arms import INTERVENTION


# === Condition 1 — Q-bias exists at high γ × all k ===
#
# DDQN reduces jens substantially across all k strata of FR
# γ=0.999. Per-stratum Cohen's d should be uniformly negative
# (treatment_arm DDQN's jens < baseline_arm vanilla's jens).
#
# Empirical (FR γ=0.999, k=1-4 × MLP[64,64], n=30 each):
# Δ_jens at k=1=-8.68, k=2=-21.33, k=3=-30.09, k=4=-41.16.
# Cohen's d should be large-negative at every k stratum.


@claim_bridge(
    source=INTERVENTION,
    target='jensen_gap',
    direction=Direction.INVERSE,
    tier=Tier.INTERVENTIONAL,
    scope=(
        (pl.col('env_name') == 'FourRooms-misc')
        & (pl.col('gamma') == 0.999)
        & (pl.col('shaping_kind') == 'none')
        & (pl.col('fa_kind') == 'mlp_deep')
        & finite(pl.col('jensen_gap'))
    ),
    predicted_direction='a_lt_b',
)
def condition_1__q_bias_exists_under_high_gamma_and_K(
    stratified_arm_diff_pooled: StratifiedArmDiffPooledResult,
    *,
    stratify_by: tuple[str, ...] = ('k_eff',),
    min_strata: int = 3,
    min_vanilla_predictor: float = 0.5,
    per_stratum_d_threshold: float = -0.5,
) -> tuple[Verdict, RefutationClass | None]:
    """**Condition 1 (observational K-scaling, not σ-bound test)**:
    DDQN reduces `jensen_gap` uniformly across k_eff strata at
    FR γ=0.999 MLP[64,64].

    Stratified by `k_eff = native_actions × action_duplicate_k`
    (= K, the discrete action count). On FR (native=4), k_eff ∈
    {4, 8, 12, 16} for action_duplicate_k=1-4. Per-stratum
    Cohen's d on `jensen_gap` is substantially negative across
    all k_eff strata.

    Caveat (review surfaced 2026-05-15): the module-level prose
    names the `σ × √(2 ln K) × 1/(1−γ)` Hasselt bound, but this
    bridge tests ONLY that DDQN reduces jens uniformly across
    K_eff. The σ factor is not measured (Q-magnitude SD across
    cells is a substrate-level diagnostic, not a per-cell
    measurable in this hypothesis). The empirical pattern is
    consistent with the Hasselt bound but is also consistent
    with any monotone-in-K reduction; the bound's load-bearing
    σ factor is unverified.

    Direction.INVERSE encodes the Hasselt prediction;
    `predicted_direction='a_lt_b'` means treatment-arm jens <
    baseline-arm jens (DDQN reduces). Empirical readings in
    `findings_two_types_of_bias` (memory)."""
    del stratify_by, min_vanilla_predictor
    if stratified_arm_diff_pooled.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT, None
    all_below = True
    any_above = False
    n_valid = 0
    for s in stratified_arm_diff_pooled.per_stratum:
        d = s.cohen_d
        if math.isnan(d):
            continue
        n_valid += 1
        if d > per_stratum_d_threshold:
            all_below = False
        if d > 0.3:
            any_above = True
    if n_valid < min_strata:
        return Verdict.POWER_INSUFFICIENT, None
    if any_above:
        return Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP
    if all_below:
        return Verdict.HELD, None
    return Verdict.NO_EFFECT, None


# === Condition 2 — FA-capacity caps Type 1 (in linear FA) ===
#
# Scope = MountainCar γ=0.999 LINEAR FA. The scope filter selects
# ONLY linear-FA cells; the bridge tests that DDQN's effect on
# jens is essentially null in this regime (vanilla Q is already
# FA-capped; max-bias can't compound).
#
# Empirical (MC γ=0.999 linear FA): vanilla jens=128.28, DDQN
# jens=128.21, Δ=-0.08 (Cohen's d near zero).


@claim_bridge(
    source=INTERVENTION,
    target='jensen_gap',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=(
        (pl.col('env_name') == 'MountainCar-v0')
        & (pl.col('gamma') == 0.999)
        & (pl.col('shaping_kind') == 'none')
        & (pl.col('fa_kind') == 'linear')
        & finite(pl.col('jensen_gap'))
    ),
    predicted_direction='null',
)
def condition_2__fa_capacity_caps_type_1_in_linear_fa(
    stratified_arm_diff_pooled: StratifiedArmDiffPooledResult,
    *,
    stratify_by: tuple[str, ...] = ('env_name',),
    min_strata: int = 3,
    min_vanilla_predictor: float = float('-inf'),
    null_d_threshold: float = 0.2,
) -> tuple[Verdict, RefutationClass | None]:
    """**Condition 2 (UNDERPOWERED at current scope)**: With
    linear FA, vanilla's Q is FA-capped before max-bias
    compounds → DDQN's bias-correction has nothing to reduce.

    Currently scoped to MC γ=0.999 linear FA only — ONE stratum,
    which the DL-pooled primitive correctly flags as
    POWER_INSUFFICIENT. The bridge body delegates to the
    primitive's verdict honestly rather than overriding it.

    A single-stratum null cannot distinguish "FA caps Type 1
    universally" from "MC linear FA happens to have small
    σ_action". To upgrade this to HELD, the scope needs ≥3
    strata varying linear-FA across envs (e.g., FR γ=0.999
    linear FA, Acrobot γ=0.999 linear FA, MC γ=0.999 linear FA)
    — particularly a sparse-positive env where C1 is known to
    fire, to test that linear FA STILL caps Type 1 there.

    Caveat (review surfaced 2026-05-15): linear FA on MC
    produces σ_VAN ≈ σ_DDQN ≈ 128 (both FA-capped) so the null
    Δ_jens here is *mechanical from the regime*, not a clean
    test of the FA-capacity intervention's causal effect on
    Type 1. Substantive claim needs the
    sparse-positive-linear-FA counter-test."""
    del stratify_by, min_vanilla_predictor
    # Honest delegation: when n_strata < min_strata=3, the
    # primitive's POWER_INSUFFICIENT verdict is correct. Don't
    # smuggle a single-cell null into HELD.
    if stratified_arm_diff_pooled.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT, None
    # Multi-stratum case (NOT achievable on current corpus, but
    # documents the upgrade path): every stratum's |d| in null
    # band, no significant wrong-sign or right-sign refutation.
    any_significant = False
    any_wrong_sign = False
    for s in stratified_arm_diff_pooled.per_stratum:
        d, se = s.cohen_d, s.cohen_se
        if math.isnan(d) or math.isnan(se):
            continue
        ci_lo = d - 1.96 * se
        ci_hi = d + 1.96 * se
        if ci_hi < -null_d_threshold:
            any_significant = True
        if ci_lo > +null_d_threshold:
            any_wrong_sign = True
    if any_wrong_sign:
        return Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP
    if any_significant:
        return Verdict.NO_EFFECT, None
    return Verdict.HELD, None


# === Condition 3 — Shaping decouples bias from outcome ===
#
# Scope = FR γ=0.999 × MLP[64,64] × SHAPED (PotentialReward).
# Tests that under reward shaping, DDQN's effect on
# `eval_best_burst_mean` is null (or sign-flipped) even though
# DDQN still actively reduces jens.
#
# Empirical (FR γ=0.999 MLP[64,64] SHAPED, n=30 each):
# Vanilla out = 61.46, DDQN out = 60.97, Δ_out = -0.49
# Cohen's d should be near zero or slightly negative.


@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_raw_mean',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=(
        (pl.col('env_name') == 'FourRooms-misc')
        & (pl.col('gamma') == 0.999)
        & (pl.col('fa_kind') == 'mlp_deep')
        & (pl.col('shaping_kind') == 'potential_manhattan')
        & finite(pl.col('eval_best_burst_raw_mean'))
    ),
    predicted_direction='null',
)
def condition_3__shaping_decouples_mech_from_outcome(
    stratified_arm_diff_pooled: StratifiedArmDiffPooledResult,
    *,
    stratify_by: tuple[str, ...] = ('env_name',),
    min_strata: int = 3,
    min_vanilla_predictor: float = float('-inf'),
    null_d_threshold: float = 0.3,
) -> tuple[Verdict, RefutationClass | None]:
    """**Condition 3 (UNDERPOWERED at current scope)**: Reward
    shaping adds dense policy signal that decouples DDQN's bias-
    reduction from outcome translation.

    Currently scoped to FR γ=0.999 MLP[64,64] shaped — ONE
    stratum. The DL-pooled primitive correctly returns
    POWER_INSUFFICIENT; bridge body now delegates honestly
    rather than overriding.

    Alternative explanations the current scope does NOT rule out
    (review surfaced 2026-05-15):
    1. **Ceiling**: shaping makes both arms learn → both saturate
       near goal-success ceiling → Δ_out shrinks because of
       saturation, not policy-signal decoupling.
    2. **Reward-scale unit**: `eval_best_burst_raw_mean` on a
       shaped wrapper integrates the modified reward; Cohen's d
       magnitude is unit-bound to the shaping potential.
    3. **The "policy gradient overrides Q-noise in argmax"
       mechanism is unmeasured** — it would need probe-level
       visibility into the argmax decision (Q vs Φ-gradient
       contribution).

    Upgrade requires: multiple shaping conditions, OR shaping ×
    multiple sparse-positive envs, plus a control that
    distinguishes "ceiling saturation" from "argmax override"."""
    del stratify_by, min_vanilla_predictor
    # Honest delegation: single-stratum scope returns
    # POWER_INSUFFICIENT. Don't smuggle into HELD.
    if stratified_arm_diff_pooled.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT, None
    # Multi-stratum: HELD iff no stratum's CI fully excludes the
    # null toward the active-mechanism direction.
    any_positive_significant = False
    for s in stratified_arm_diff_pooled.per_stratum:
        d, se = s.cohen_d, s.cohen_se
        if math.isnan(d) or math.isnan(se):
            continue
        ci_lo = d - 1.96 * se
        if ci_lo > +null_d_threshold:
            any_positive_significant = True
    if any_positive_significant:
        return Verdict.NO_EFFECT, None
    return Verdict.HELD, None
