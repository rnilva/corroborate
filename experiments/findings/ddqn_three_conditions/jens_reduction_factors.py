"""Empirical characterization of DDQN's effect on `jensen_gap`,
viewed through structural moderators that Hasselt 2010's bound
suggests are worth looking at:

    bias ≤ σ_action × √(2 ln K) × 1/(1 − γ)

The bridges here are NOT a test of Hasselt's bound. Hasselt's
formula is a LENS — it tells us WHERE to look (action count,
discount factor, function-approximator capacity). What we find
on each axis is the empirical content.

Bridges:

- **k_eff** (`ddqn_reduces_jens_uniformly_across_k_at_fr_high_gamma`):
  DDQN reduces jens at every k_eff ∈ {4, 8, 12, 16} at the
  FR γ=0.999 × MLP × unshaped reference cell. The reduction is
  uniform across the k-sweep — DDQN doesn't "give up" as k_eff
  grows. Practical reading: action duplication doesn't break
  DDQN's mechanism at this scope.

- **γ** (`ddqn_reduction_amplified_by_gamma__fr_mlp_k4_unshaped`):
  DDQN's jens reduction magnitude grows with γ (≥ 3× from
  γ=0.99 to γ=0.999). Practical reading: DDQN matters more at
  higher discount factors — relevant for any practitioner
  choosing γ → 1.

- **FA capacity** (`fa_capacity_moderates_ddqn_jens_reduction`):
  Across (env, γ, fa_kind) strata, DDQN's effect is more
  negative at MLP than at linear FA. Practical reading: linear
  FA caps DDQN's value-add. When constrained to linear FA
  (sample efficiency, interpretability), DDQN matters less.

- **FA-capacity exception** (`linear_fa_cap_fails_at_metamaze_*`):
  At MetaMaze γ=0.999 × linear, DDQN substantially reduces
  jens despite linear FA — an empirical anomaly worth flagging.
  We don't claim to know why; MM's random-maze-per-episode
  structure is the natural mechanism story but isn't tested.

- **σ_action sibling** (`sigma_action_predicts_ddqn_jens_reduction`):
  Is baseline σ_action (within-state across-action Q SD) a
  useful predictor of DDQN's standardized effect? Empirically:
  slope = +2.84, p=0.143 — σ_action does NOT cleanly predict
  DDQN's relative effectiveness. There's heterogeneity (I²=0.96)
  that σ alone doesn't explain. Caveat: at low σ, between-seed
  SD is also small, so Cohen's d gets inflated — the
  standardized test is confounded by this.

The empirically-anchored frame for interpretation is the
Type 1 / Type 2 decomposition from `findings_two_types_of_bias`
(DDQN reduces Type 1 when Type 1 has FA × γ headroom to
develop). Hasselt 2010's bound is one theoretical instance of
the Type 1 path.

What this module DOES claim:
- DDQN reduces jens in a structured way on these axes at these
  envs. The empirical characterization is the corroborated
  content.

What this module does NOT claim:
- That any specific theoretical factor (K, γ, σ, FA) is THE
  mechanism. The empirical correlations are real; the causal
  attribution is open.
- Generalisation beyond the envs in scope."""
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
    """DDQN reduces jens at every k_eff ∈ {4, 8, 12, 16} at the
    FR γ=0.999 × MLP[64,64] × no-shaping reference cell.

    Per-k_eff Cohen's d ≤ -0.5 at every stratum.

    Practical reading: DDQN's mechanism is active and uniform
    across action_duplicate manipulation at this scope. DDQN
    doesn't fail at high k_eff.

    Caveat: `k_eff = native_K × action_duplicate_k` doesn't
    cleanly identify Hasselt's K factor (action_duplicate
    creates correlated identical-effect actions, not iid
    K-armed-max draws). The empirical observation here is about
    DDQN's behavior under action duplication — whatever the
    underlying mechanism.

    `min_baseline_predictor=0.5` excludes strata where vanilla
    has no jens to reduce (pre-registered noise-floor)."""
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
    """DDQN's jens reduction magnitude grows monotonically with
    γ at FR × MLP × unshaped × k_eff=4.

    Per-γ Cohen's d ≤ -0.8 at BOTH γ ∈ {0.99, 0.999}, AND
    |mean_diff(γ=0.999)| ≥ 3 × |mean_diff(γ=0.99)|.

    Practical reading: DDQN matters more at higher discount
    factors. Empirical amplification from γ=0.99 to γ=0.999 is
    ~47× — practitioners choosing γ → 1 should expect DDQN's
    bias-reduction effect to be substantial in absolute terms.

    k_eff=4 (native FR action count, no action_duplicate) is
    fixed to hold the k-axis constant.

    Refutations:
    - NO_EFFECT/SIGN_FLIP: any γ shows d > 0.
    - NO_EFFECT/NULL: either γ shows d > -0.8.
    - POWER_INSUFFICIENT: amplification ratio < 3.

    Caveat: at FR γ=0.999, vanilla never finds the goal (MC ≈
    0.005 throughout, Q grows to ~100; see
    `findings_q_explosion_direct_evidence`). So the γ-axis effect
    here likely mixes Hasselt-style bootstrap-chain amplification
    with vanilla-degeneracy-at-γ→1. Whatever the mix, DDQN
    empirically reduces jens substantially more at γ=0.999."""
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
    """FA capacity moderates DDQN's effect on jens.

    Per-(env, gamma, fa_kind) Cohen's d → random-effects
    meta-regression on binary `fa_capacity` (0=linear,
    1=mlp_deep). Slope ≤ -0.5 AND significant → HELD.

    Practical reading: DDQN's bias-reduction effect is stronger
    at MLP than at linear FA. When constrained to linear FA
    (sample efficiency, interpretability), DDQN's value-add on
    jens is roughly 1 Cohen unit smaller per the empirical fit.

    Refutations:
    - NO_EFFECT/SIGN_FLIP: slope significantly POSITIVE.
    - NO_EFFECT/NULL: slope CI brackets the threshold.
    - POWER_INSUFFICIENT: n_strata < 6.

    Caveat: the binary `fa_capacity` proxy doesn't tell us WHY
    DDQN matters less at linear. Two natural mechanism stories:
    (a) linear FA caps σ_action (within-state across-action Q
    SD), bounding Hasselt's bias path; (b) linear FA caps Q
    absolute magnitude, so vanilla never develops substantial
    bias for DDQN to reduce regardless of σ. The sibling test
    `sigma_action_predicts_ddqn_jens_reduction` characterizes
    σ_action directly — it does NOT cleanly predict DDQN's
    standardized effect, suggesting story (b) does more of the
    work. The empirical observation (DDQN matters less at
    linear) is robust; the mechanism is open.

    Scope:
    - {FR, Acrobot, MountainCar}: pre-registered inclusion
      criterion (envs where vanilla MLP develops substantive
      bias). MetaMaze excluded — has a documented exception
      sibling. Bsuite light envs (CartPole, Catch, DeepSea)
      excluded — vanilla doesn't overshoot at any FA, so the
      moderator effect is unmeasurable."""
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
def sigma_action_predicts_ddqn_jens_reduction(
    meta_regression_unpaired_d: MetaRegressionResult,
    *,
    treatment_arm: str = (
        'bootstrap=partial(Claim:bootstrap;'
        'greedification=Claim:double_greedify)'
    ),
    baseline_arm: str = 'baseline',
    source: str = 'jensen_gap',
    continuous_covariate: str = 'q_action_std_late',
    continuous_covariate_arm: str = 'baseline',
    stratify_by: tuple[str, ...] = ('env_name', 'gamma', 'fa_kind'),
    min_strata: int = 6,
    slope_threshold: float = -2.0,
) -> Verdict:
    """Is baseline σ_action a useful predictor of DDQN's effect
    on jens?

    Per-(env, γ, fa_kind) Cohen's d → random-effects
    meta-regression on the continuous per-stratum baseline mean
    of `q_action_std_late` (within-state across-action Q SD,
    averaged over the late 50% of training). σ_action is the
    quantity Hasselt 2010's bound is most sensitive to; this
    bridge characterizes whether it predicts DDQN's empirical
    effect.

    HELD iff slope ≤ `slope_threshold` (= -2.0) AND significant:
    DDQN's standardized effect becomes more negative as σ grows.

    Empirical state on this corpus (n_strata=12; σ range
    0.005-0.36):

        slope = +2.84  CI = [-1.14, +6.82]  p = 0.143

    Verdict: POWER_INSUFFICIENT. σ_action does not cleanly
    predict DDQN's relative effectiveness on this panel. The
    point estimate is positive (opposite-sign from what a naive
    Hasselt extrapolation would suggest), with substantial
    between-stratum heterogeneity (I²=0.96).

    Practical reading: knowing baseline σ_action is NOT
    sufficient to predict how big DDQN's effect will be at a
    new (env, γ, fa) cell. Other factors — likely Q absolute
    magnitude and FA capacity for representing it — matter
    more.

    CAVEATS on the test design:
    - Dependent variable is Cohen's d (standardized by between-
      seed SD). At low σ, the SD is also small, which inflates
      d. The standardized test confounds "σ predicts effect"
      with "σ predicts SD". The proper Hasselt-bound test (does
      |raw mean_diff| stay UNDER σ × √(2 ln K) × 1/(1−γ)?) IS
      satisfied at every stratum on this panel — but that's a
      bound-holding test, not what this bridge does.
    - At the smallest observed σ (0.005), Cohen's d is heavily
      inflated by tiny between-seed SD; this is partly why the
      intercept is significantly negative (-0.96, p=0.017).

    Refutations (when reached):
    - HELD: σ_action useful predictor of DDQN's effect.
    - NO_EFFECT (sig slope ≥ +2.0): σ_action is anti-correlated
      with DDQN's standardized effect at this scope.

    Sibling to `fa_capacity_moderates_ddqn_jens_reduction`.
    NOT in either cluster Finding."""
    del treatment_arm, baseline_arm, source
    del continuous_covariate, continuous_covariate_arm, stratify_by
    return meta_regression_coef_verdict(
        meta_regression_unpaired_d,
        'q_action_std_late',
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
    """At MetaMaze γ=0.999 × linear FA, DDQN substantially reduces
    jens — an empirical anomaly relative to the FA-capacity
    moderator's "linear FA caps DDQN's effect" rule.

    Per-(n_episodes) Cohen's d on `jensen_gap` at (MM, γ=0.999,
    linear, unshaped). n_episodes=5 shows d ≈ -0.5; n_episodes=20
    (eval-power fixed) shows d ≈ -1.1. HELD iff per-stratum d ≤
    -0.3 at every n_episodes stratum.

    Practical reading: at MM × γ=0.999, DDQN matters even with
    linear FA — a notable break from the FA-capacity moderator's
    pattern. If you're running on MM (or maze envs that redraw
    geometry per episode) and choosing FA, this empirical
    observation says DDQN still earns its keep at linear FA.

    Caveat: we don't know WHY. MM redraws a random maze per
    evaluation episode — a natural mechanism story is that
    linear FA can't generalize across maze instances, so
    vanilla's bootstrap target carries FA-fit error that DDQN
    happens to clip. But we don't test this. Per
    `findings_two_types_of_bias`, MM's intermediate T1/T2 ratio
    (0.21-0.71) also leaves room for Type 1 contribution. We
    document the empirical anomaly; the mechanism is open.

    No other env with per-episode state-distribution shift has
    been tested at linear FA — we don't know whether this is an
    MM-specific anomaly or a general feature of non-stationary
    envs."""
    del stratify_by, min_baseline_predictor
    return per_stratum_d_threshold_verdict(
        stratified_arm_diff_pooled,
        threshold=per_stratum_d_threshold,
        sign=-1,
        min_strata=min_strata,
    )
