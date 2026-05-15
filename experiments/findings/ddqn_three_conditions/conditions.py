"""Three observational bridges scoped to the cells where the
three-conditions framework predicts a specific outcome.

Each bridge makes a SCOPED OBSERVATIONAL claim about what
happens on its specific (env, γ, FA, shaping) slice. The
substantive theoretical framework (in memory entries
`findings_two_types_of_bias`,
`findings_shaping_decouples_bias_from_outcome`) says these
three regimes are illustrative of the K/FA/shaping factors that
shape DDQN's mech-link translation, but the bridges here do NOT
claim universal necessity — just that the observed direction on
each scope matches the prediction.

- **C1**: multi-stratum (k_eff ∈ {4,8,12,16}) on FR γ=0.999
  MLP[64,64] no-shaping. Tests that DDQN's `jensen_gap`
  reduction is uniform across K_eff via DL-pooled
  `stratified_arm_diff_pooled`. HELD legitimately on multi-
  stratum panel.

- **C2**: single-cell observation on MountainCar γ=0.999 ×
  LINEAR FA. Tests that DDQN's Δ_jens on this specific cell is
  within a null band via `arm_mean_diff` (single-stratum
  Welch's t — the substrate-discipline-correct primitive for
  single-cell observations per `findings_within_stratum_primitives`).
  Does NOT claim universal "FA-capacity caps Type 1" — just
  observes that on THIS cell, DDQN doesn't reduce jens
  appreciably.

- **C3**: single-cell observation on FourRooms γ=0.999 MLP ×
  SHAPED. Same `arm_mean_diff` shape: tests that on THIS cell,
  DDQN doesn't significantly improve outcome.

The cluster Finding asserts a WITHIN-SCOPE CONSISTENCY claim:
three observations, each in the predicted direction. Not a
universal-necessity claim — that would require multi-env
counter-tests per the review note in
`finding_three_conditions.py:BLOCKED_ON`."""
from __future__ import annotations

import math

import polars as pl

from corroborate.analyses.arm_mean_diff import ArmMeanDiffResult
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
def condition_2__no_appreciable_jens_reduction_under_mc_linear_fa(
    arm_mean_diff: ArmMeanDiffResult,
    *,
    null_d_threshold: float = 0.3,
    p_threshold: float = 0.05,
) -> tuple[Verdict, RefutationClass | None]:
    """**Observational claim (scoped to MC γ=0.999 linear FA)**:
    DDQN does NOT appreciably reduce `jensen_gap` on this cell.
    `|standardized_effect|` within null band AND Welch's t-test
    fails to reject `mean_diff = 0`.

    This is a SINGLE-CELL observation consistent with the
    Type-2-dominated regime predicted by the two-types
    decomposition (`findings_two_types_of_bias`). It does NOT
    test universal "FA capacity caps Type 1" — that would
    require multi-env linear-FA counter-tests, particularly on
    a sparse-positive env where C1's mechanism is active. See
    `BLOCKED_ON` in `finding_three_conditions.py`.

    Empirical pre-author (MC γ=0.999 linear FA, n=60 per arm):
    vanilla jens=128.28, DDQN jens=128.21, Δ=-0.08
    (standardized d ≈ -0.001). The within-scope null
    observation is consistent with — but does not prove — the
    FA-cap hypothesis. The substrate-discipline-correct
    primitive for single-cell tests is `arm_mean_diff` (true
    independent-samples Welch's t, no smuggled pairing per
    `findings_within_stratum_primitives`)."""
    if math.isnan(arm_mean_diff.mean_diff):
        return Verdict.POWER_INSUFFICIENT, None
    d = arm_mean_diff.standardized_effect
    p = arm_mean_diff.mean_diff_p_value
    if math.isnan(d) or math.isnan(p):
        return Verdict.POWER_INSUFFICIENT, None
    # HELD = null observation: small standardized effect AND
    # Welch's t-test fails to reject zero.
    if abs(d) < null_d_threshold and p > p_threshold:
        return Verdict.HELD, None
    # Non-null effect of DDQN on jens at MC linear FA — refutes
    # the within-scope null observation.
    if d < -null_d_threshold and p < p_threshold:
        # DDQN reduces jens significantly — Type 1 NOT capped.
        return Verdict.NO_EFFECT, None
    if d > +null_d_threshold and p < p_threshold:
        return Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP
    # CI spans the threshold band.
    return Verdict.POWER_INSUFFICIENT, None


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
def condition_3__no_outcome_benefit_under_fr_shaped(
    arm_mean_diff: ArmMeanDiffResult,
    *,
    null_d_threshold: float = 0.3,
    p_threshold: float = 0.05,
) -> tuple[Verdict, RefutationClass | None]:
    """**Observational claim (scoped to FR γ=0.999 MLP shaped)**:
    DDQN does NOT significantly improve `eval_best_burst_raw_mean`
    on this cell. `standardized_effect` not significantly
    positive (Welch's t-test fails to reject in the
    Direction.DIRECT sense).

    Empirical pre-author (FR γ=0.999 MLP shaped, n=30 per arm):
    vanilla raw=81.8, DDQN raw=81.1, Δ=-0.72 (slight negative,
    consistent with clip-wedge harming under shaping at fixed
    reward polarity).

    Alternative explanations the current single-cell scope does
    NOT rule out:
    1. **Ceiling**: shaping makes both arms learn; both saturate;
       Δ shrinks because of saturation, not signal-decoupling.
    2. **Reward-scale unit**: shaped raw return integrates the
       modified reward; Cohen's d is unit-bound to the potential.
    3. **The "policy gradient overrides Q-noise" mechanism is
       unmeasured** — it would need argmax-decomposition probes.

    The within-scope null observation is consistent with — but
    does not prove — the policy-signal-decoupling hypothesis.
    See `BLOCKED_ON` for upgrade path."""
    if math.isnan(arm_mean_diff.mean_diff):
        return Verdict.POWER_INSUFFICIENT, None
    d = arm_mean_diff.standardized_effect
    p = arm_mean_diff.mean_diff_p_value
    if math.isnan(d) or math.isnan(p):
        return Verdict.POWER_INSUFFICIENT, None
    # HELD = no significantly-positive effect (the null
    # prediction direction).
    if d < +null_d_threshold or p > p_threshold:
        return Verdict.HELD, None
    # DDQN's outcome benefit translates significantly under
    # shaping → refutes the within-scope null observation.
    return Verdict.NO_EFFECT, None
