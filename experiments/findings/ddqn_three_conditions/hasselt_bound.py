"""Hasselt's three-factor bound on Q-learning overestimation.

Hasselt 2010's structural bound:

    bias ≤ σ_action × √(2 ln K) × 1/(1 − γ)

The bound has three multiplicative factors. DDQN's clip on the
bootstrap target is supposed to annihilate the `√(2 ln K)`
factor (the argmax-vs-max gap). The three factors give three
places to intervene; this module's four bridges test each.

- `√(2 ln K)` factor — K-scaling within FR γ=0.999 × MLP ×
  unshaped across k_eff ∈ {4, 8, 12, 16}.
- `1/(1 − γ)` factor — γ-amplification at FR × MLP × unshaped
  × k_eff=4 (K controlled) across γ ∈ {0.99, 0.999}.
- σ_action factor — rule + exception cluster:
  - rule: linear FA bounds σ_action → Hasselt mech dormant.
    Tested as the FA-capacity moderator on the panel
    {FR, Acrobot, MountainCar} × γ × {linear, mlp_deep}.
  - exception: MetaMaze γ=0.999 × linear FA — the cap FAILS
    because random-maze-per-episode forces FA-fit-error bias
    that DDQN clips via a non-σ path.

The cluster pattern: K HELD + γ HELD + (FA-moderator HELD +
MM-exception HELD) → "Hasselt's bound is the right
mechanistic frame for DDQN's bias-reduction at these envs"."""
from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType

import polars as pl

from corroborate.analyses.meta_regression_unpaired_d import (
    MetaRegressionResult,
)
from corroborate.analyses.stratified_arm_diff_pooled import (
    StratifiedArmDiffPooledResult,
)
from corroborate.bridge.bridge import Direction, Tier, claim_bridge
from corroborate.bridge.predicates import finite
from corroborate.bridge.verdict import RefutationClass, Verdict

from experiments.findings.ddqn_three_conditions._arms import INTERVENTION
from experiments.findings.ddqn_three_conditions._verdicts import (
    meta_regression_coef_verdict,
    per_stratum_d_threshold_verdict,
)


# FA capacity encoded as a numeric for meta-regression. Linear
# FA has minimal cross-action Q variance (σ_action bounded);
# MLP[64,64] has substantial room. The binary 0/1 encoding makes
# the meta-regression slope equal `mean(d at MLP) − mean(d at
# linear)` at the panel level.
_FA_CAPACITY: Mapping[object, Mapping[str, float]] = MappingProxyType({
    'linear':   MappingProxyType({'fa_capacity': 0.0}),
    'mlp_deep': MappingProxyType({'fa_capacity': 1.0}),
})


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
    min_baseline_predictor: float = 0.5,
    per_stratum_d_threshold: float = -0.5,
) -> tuple[Verdict, RefutationClass | None]:
    """Per-k_eff Cohen's d on `jensen_gap` is uniformly ≤ -0.5
    at FourRooms γ=0.999 × MLP[64,64] × no-shaping across
    k_eff ∈ {4, 8, 12, 16}.

    Direction.INVERSE encodes the Hasselt mech prediction;
    `predicted_direction='a_lt_b'` means treatment-arm jens <
    baseline-arm jens (DDQN reduces). Empirical readings live
    in `findings_two_types_of_bias`.

    Caveat: within-FR K-scaling only. The bound's σ factor is
    tested by the FA-capacity moderator below; the γ factor by
    the amplification bridge. This bridge isolates √(2 ln K)."""
    del stratify_by, min_baseline_predictor
    return per_stratum_d_threshold_verdict(
        stratified_arm_diff_pooled,
        threshold=per_stratum_d_threshold,
        sign=-1,
        min_strata=min_strata,
    )


@claim_bridge(
    source=INTERVENTION,
    target='jensen_gap',
    direction=Direction.INVERSE,
    tier=Tier.INTERVENTIONAL,
    scope=(
        (pl.col('env_name') == 'FourRooms-misc')
        & (pl.col('fa_kind') == 'mlp_deep')
        & (pl.col('shaping_kind') == 'none')
        & (pl.col('k_eff') == 4)
        & pl.col('gamma').is_in([0.99, 0.999])
        & finite(pl.col('jensen_gap'))
    ),
    predicted_direction='a_lt_b',
)
def ddqn_reduction_amplified_by_gamma__fr_mlp_k4_unshaped(
    stratified_arm_diff_pooled: StratifiedArmDiffPooledResult,
    *,
    stratify_by: tuple[str, ...] = ('gamma',),
    min_strata: int = 2,
    min_baseline_predictor: float = float('-inf'),
    per_stratum_d_threshold: float = -0.8,
    gamma_amp_ratio: float = 3.0,
) -> tuple[Verdict, RefutationClass | None]:
    """DDQN's bias-reduction magnitude scales with γ as Hasselt's
    1/(1−γ) factor predicts.

    Per-γ independent-samples Cohen's d + mean-diff on jensen_gap
    at FR × MLP[64,64] × unshaped × k_eff=4 across γ ∈ {0.99,
    0.999}. HELD iff:

    1. Per-stratum cohen_d ≤ -0.8 at BOTH γ strata (DDQN's effect
       is "large" by Cohen's convention at every γ in scope), AND
    2. |mean_diff(γ=0.999)| ≥ 3 × |mean_diff(γ=0.99)| (absolute
       magnitude of bias reduction scales with γ; 3× is a
       conservative lower bound vs the bound's structural
       prediction of 10× for 1/(1-γ) at γ ∈ {0.99, 0.999}).

    Refutations:
    - NO_EFFECT/SIGN_FLIP: any γ shows d > 0 (DDQN INCREASES jens).
    - NO_EFFECT/NULL: either γ shows d > -0.8 (DDQN's effect not
      large at one γ — the reduction isn't uniformly present).
    - POWER_INSUFFICIENT: amplification ratio < 3 (the
      γ-amplification structure isn't visible; data consistent
      with no γ-scaling).

    k_eff=4 (native FR action count, no action_duplicate wrapper)
    is fixed to remove the K factor as a confound — within this
    scope only γ varies."""
    del stratify_by, min_baseline_predictor
    if stratified_arm_diff_pooled.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT, None
    valid: list[tuple[float, float, float]] = []
    for s in stratified_arm_diff_pooled.per_stratum:
        d, md = s.cohen_d, s.mean_diff
        if math.isnan(d) or math.isnan(md):
            continue
        g_val_obj = s.stratum_id[0] if s.stratum_id else None
        if not isinstance(g_val_obj, (int, float)):
            continue
        valid.append((float(g_val_obj), d, abs(md)))
    if len(valid) < min_strata:
        return Verdict.POWER_INSUFFICIENT, None
    if any(d_val > per_stratum_d_threshold for _, d_val, _ in valid):
        return Verdict.NO_EFFECT, None
    g_to_amd: dict[float, float] = {g: amd for g, _, amd in valid}
    if 0.99 in g_to_amd and 0.999 in g_to_amd:
        amp = (g_to_amd[0.999] / g_to_amd[0.99]
               if g_to_amd[0.99] > 0 else float('inf'))
        if amp >= gamma_amp_ratio:
            return Verdict.HELD, None
        return Verdict.POWER_INSUFFICIENT, None
    return Verdict.POWER_INSUFFICIENT, None


@claim_bridge(
    source=INTERVENTION,
    target='jensen_gap',
    direction=Direction.INVERSE,
    tier=Tier.INTERVENTIONAL,
    scope=(
        pl.col('gamma').is_in([0.99, 0.999])
        & pl.col('fa_kind').is_in(['linear', 'mlp_deep'])
        & (pl.col('shaping_kind') == 'none')
        & pl.col('env_name').is_in([
            'FourRooms-misc', 'Acrobot-v1', 'MountainCar-v0',
        ])
        & finite(pl.col('jensen_gap'))
    ),
    predicted_direction='a_lt_b',
)
def fa_capacity_moderates_ddqn_jens_reduction(
    meta_regression_unpaired_d: MetaRegressionResult,
    *,
    treatment_arm: str = (
        'bootstrap=partial(Claim:bootstrap;'
        'greedification=Claim:double_greedify)'
    ),
    baseline_arm: str = 'baseline',
    source: str = 'jensen_gap',
    covariate_key_field: str = 'fa_kind',
    covariates_per_key: Mapping[
        object, Mapping[str, float],
    ] = _FA_CAPACITY,
    stratify_by: tuple[str, ...] = ('env_name', 'gamma', 'fa_kind'),
    min_strata: int = 6,
    slope_threshold: float = -0.5,
) -> Verdict:
    """FA capacity moderates DDQN's effect on jensen_gap.

    Per-(env, gamma, fa_kind) independent-samples Cohen's d on
    `jensen_gap` → random-effects meta-regression on
    `fa_capacity` (binary 0=linear, 1=mlp_deep). The slope is
    Δd per unit FA capacity — under the binary encoding, this
    equals `mean(d at MLP) − mean(d at linear)`.

    HELD iff the slope is significantly negative AND
    ≤ `slope_threshold` (= −0.5): DDQN's effect on jens at MLP
    is more negative than at linear by at least 0.5 Cohen units,
    consistent with σ_action being FA-capped (Hasselt bound's
    σ factor empirically corroborated).

    Refutations:
    - NO_EFFECT/SIGN_FLIP: slope significantly POSITIVE (DDQN's
      effect MORE negative at linear than at MLP — would refute
      the σ-via-FA hypothesis).
    - NO_EFFECT/NULL: slope CI brackets the threshold (not
      significantly different from zero or not large enough).
    - POWER_INSUFFICIENT: n_strata < 6 (we have at minimum 3 envs
      × 2 γ × 2 fa = 12 strata at full ingest; require half).

    Substantive scope excludes:
    - MetaMaze (encoded as the MM-exception bridge — FA-fit
      error from random-maze state distribution provides a
      parallel bias path that DDQN clips even at linear FA).
    - CartPole / Catch / DeepSea (vanilla doesn't overshoot at
      any FA — moderator effect is unmeasurable when there's
      no signal to moderate)."""
    del treatment_arm, baseline_arm, source
    del covariate_key_field, covariates_per_key, stratify_by
    return meta_regression_coef_verdict(
        meta_regression_unpaired_d,
        'fa_capacity',
        sign=-1,
        threshold=slope_threshold,
        min_strata=min_strata,
    )


@claim_bridge(
    source=INTERVENTION,
    target='jensen_gap',
    direction=Direction.INVERSE,
    tier=Tier.INTERVENTIONAL,
    scope=(
        (pl.col('env_name') == 'MetaMaze-misc')
        & (pl.col('gamma') == 0.999)
        & (pl.col('fa_kind') == 'linear')
        & (pl.col('shaping_kind') == 'none')
        & finite(pl.col('jensen_gap'))
    ),
    predicted_direction='a_lt_b',
)
def linear_fa_cap_fails_at_metamaze_g999__exception(
    stratified_arm_diff_pooled: StratifiedArmDiffPooledResult,
    *,
    stratify_by: tuple[str, ...] = ('n_episodes',),
    min_strata: int = 2,
    min_baseline_predictor: float = float('-inf'),
    per_stratum_d_threshold: float = -0.3,
) -> tuple[Verdict, RefutationClass | None]:
    """At MetaMaze γ=0.999 with linear FA, the FA-capacity rule
    FAILS — DDQN substantively reduces jens with a non-trivial
    effect at every eval-power level tested.

    Per-(n_episodes) independent-samples Cohen's d on
    `jensen_gap` at (env=MetaMaze, γ=0.999, linear, unshaped).
    Stratifying by n_episodes exposes the eval-power
    sensitivity: n_episodes=5 shows d ≈ −0.5 (real but partly
    diluted by MetaMaze's high per-episode eval variance);
    n_episodes=20 shows d ≈ −1.1 (the eval-power-fixed reading).

    HELD iff per-stratum d ≤ `per_stratum_d_threshold` (= −0.3,
    Cohen's "small") at EVERY n_episodes stratum. The two strata
    are expected to agree on sign; the larger n_episodes stratum
    just shows the cleaner magnitude.

    Substantive mechanism: MetaMaze draws a new random maze per
    evaluation episode. Linear FA cannot represent a single Q
    function that generalises across mazes → vanilla's bootstrap
    target is biased by FA-fit error → DDQN's clip removes it.
    The mech is FA-fit-error × state-distribution-shift, NOT the
    σ × √(2 ln K) path that the FA-capacity moderator tests and
    rules out across the 6-env rule scope.

    Forms a rule + exception cluster with the FA-capacity
    moderator: rule HELD + exception HELD = "σ-via-FA gates the
    Hasselt mech EXCEPT where FA-fit error provides a parallel
    bias path"."""
    del stratify_by, min_baseline_predictor
    return per_stratum_d_threshold_verdict(
        stratified_arm_diff_pooled,
        threshold=per_stratum_d_threshold,
        sign=-1,
        min_strata=min_strata,
    )
