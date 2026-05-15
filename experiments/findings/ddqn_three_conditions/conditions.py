"""Within-FR K-scaling of DDQN's bias reduction.

Single observational bridge: at FourRooms γ=0.999 × MLP[64,64]
× no-shaping, DDQN reduces `jensen_gap` uniformly across
k_eff ∈ {4, 8, 12, 16}. Multi-stratum HELD via
`stratified_arm_diff_pooled` (4 strata, n=30 per arm per
stratum).

**What this is NOT**:
- NOT a test of the Hasselt σ × √(2 ln K) × 1/(1−γ) bound —
  the σ factor is unmeasured.
- NOT a test of "DDQN's outcome benefit requires Q-bias"; the
  bridge tests mech reduction (`jensen_gap`), not outcome
  translation.
- NOT a generalization to other envs / γ / FA — see the
  retracted-bridges note in `finding_three_conditions.py` for
  why the cross-env C2/C3 claims were withdrawn.

The substrate-corroborated framework (memo entries
`findings_two_types_of_bias`,
`findings_shaping_decouples_bias_from_outcome`) is broader;
this bridge isolates the one piece that has multi-stratum
power on the current corpus."""
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


# === C1 — DDQN reduces jens uniformly across K_eff at FR γ=0.999 ===
#
# Multi-stratum within FR. Each k_eff ∈ {4, 8, 12, 16} (= 4 ×
# action_duplicate_k for FR's native 4 actions) contributes one
# Cohen's d on `jensen_gap`. HELD iff all four strata's d are
# substantially negative.


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
def ddqn_reduces_jens_uniformly_across_k_at_fr_high_gamma(
    stratified_arm_diff_pooled: StratifiedArmDiffPooledResult,
    *,
    stratify_by: tuple[str, ...] = ('k_eff',),
    min_strata: int = 3,
    min_vanilla_predictor: float = 0.5,
    per_stratum_d_threshold: float = -0.5,
) -> tuple[Verdict, RefutationClass | None]:
    """Per-k_eff Cohen's d on `jensen_gap` is uniformly < -0.5
    at FourRooms γ=0.999 MLP[64,64] no-shaping across
    k_eff ∈ {4, 8, 12, 16}.

    Empirical readings in `findings_two_types_of_bias` (memory).
    Direction.INVERSE encodes the Hasselt mech prediction;
    `predicted_direction='a_lt_b'` means treatment-arm jens <
    baseline-arm jens (DDQN reduces).

    Caveat: this is the within-FR K-scaling claim only. The
    σ × √(2 ln K) Hasselt bound's load-bearing σ factor is
    unmeasured; the empirical pattern is consistent with the
    bound but also consistent with any monotone-in-K reduction.

    Verdict: HELD iff every admitted stratum's Cohen's d is
    below `per_stratum_d_threshold` and no stratum shows a
    wrong-sign refutation."""
    del stratify_by, min_vanilla_predictor
    if stratified_arm_diff_pooled.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT, None
    all_below = True
    any_wrong_sign = False
    n_valid = 0
    for s in stratified_arm_diff_pooled.per_stratum:
        d = s.cohen_d
        if math.isnan(d):
            continue
        n_valid += 1
        if d > per_stratum_d_threshold:
            all_below = False
        if d > 0.3:
            any_wrong_sign = True
    if n_valid < min_strata:
        return Verdict.POWER_INSUFFICIENT, None
    if any_wrong_sign:
        return Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP
    if all_below:
        return Verdict.HELD, None
    return Verdict.NO_EFFECT, None
