"""Three scalings of DDQN's bias-reduction, inspired by but NOT
identified to Hasselt 2010's bound.

Hasselt 2010's structural bound:

    bias ≤ σ_action × √(2 ln K) × 1/(1 − γ)

motivates three axes to interrogate DDQN's effect on
`jensen_gap`: action count, discount factor, function-
approximator capacity. Each bridge below tests a SCALING on its
axis. None identify the bridge's variable with Hasselt's
specific factor — the SCALINGS are corroborated; the factor-
by-factor ATTRIBUTION is not.

What the bridges actually test:

- **K-axis scaling** (`ddqn_reduces_jens_uniformly_across_k_*`):
  monotone-in-`k_eff` at FR γ=0.999 × MLP × unshaped across
  k_eff ∈ {4, 8, 12, 16}. `k_eff = native_K × action_duplicate_k`
  is a CONFOUNDED K-proxy: action_duplicate creates K
  identical-effect actions (perfectly correlated estimators),
  not iid K-armed-max draws. The √(2 ln K)-iid derivation does
  NOT apply to k_eff produced this way. Cannot discriminate
  Hasselt's K factor from state-distribution / exploration
  effects of action duplication.

- **γ-axis scaling** (`ddqn_reduction_amplified_by_gamma_*`):
  monotone-in-γ at FR × MLP × unshaped × k_eff=4 across γ ∈
  {0.99, 0.999}. Threshold ≥ 3× amplification. The structural
  prediction is 10×; the threshold is loose enough to admit
  any monotone-in-γ scaling. Cannot discriminate Hasselt's
  1/(1−γ) factor from the alternative in
  `findings_q_explosion_direct_evidence` (vanilla degeneracy
  at γ→1 — at FR γ=0.999 vanilla never finds goal, Q grows
  19,520× MC because no observational anchor exists).

- **FA-axis moderator** (`fa_capacity_moderates_*`): per-(env,
  γ, fa_kind) Cohen's d ↦ binary `fa_capacity` (0=linear,
  1=mlp_deep), random-effects meta-regression slope. The
  empirical pattern (Δd more negative at MLP than linear) is
  consistent with σ_action being FA-capped (Hasselt-σ
  interpretation). It is ALSO consistent with the Type 1 /
  Type 2 framing of `findings_two_types_of_bias`: linear FA
  truncates Q before the max-amplifier has headroom, so Type 1
  is small regardless of σ_action. This bridge tests FA-
  capacity moderation; it does NOT discriminate σ_action from
  FA-truncation. A continuous regression on `q_action_std_late`
  (already in REQUIRED_MEASURABLES) would; this binary form
  does not.

- **FA-axis exception** (`linear_fa_cap_fails_at_metamaze_*`):
  at MetaMaze γ=0.999 × linear, DDQN substantially reduces
  jens — an empirical anomaly relative to the FA-moderator
  rule. The mechanism story ("random-maze state-distribution
  shift → FA-fit error") is ASSERTED from env structure but
  NOT empirically discriminated from alternatives (Type 1
  contribution at MM's intermediate T1/T2 = 0.21 per
  `findings_two_types_of_bias`; FR-style late-divergence under
  sufficient |Q| magnitude). The bridge documents the
  anomaly; it does not corroborate the proposed mechanism.

The empirically-corroborated frame here is the Type 1 / Type 2
decomposition of `findings_two_types_of_bias`. Hasselt's bound
is the cleanest theoretical inspiration; this cluster's bridges
corroborate three scalings consistent with — but not
identified to — that bound."""
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

    Empirical content: DDQN's bias-reduction is monotone-in-k_eff
    at this scope. Direction.INVERSE encodes the predicted
    direction; `predicted_direction='a_lt_b'` means treatment-arm
    jens < baseline-arm jens. See `findings_two_types_of_bias`.

    What this bridge does NOT claim:
    - That k_eff identifies Hasselt's K factor.
      `k_eff = native_K × action_duplicate_k` is a confounded
      K-proxy: action_duplicate creates K identical-effect
      actions (perfectly correlated estimators), not iid
      K-armed-max draws. Hasselt's √(2 ln K)-iid-Gaussian-max
      derivation does NOT apply at k_eff. The empirical pattern
      is also consistent with state-distribution and exploration
      effects from action duplication (longer effective horizon,
      sparser per-action visit counts). The bridge corroborates
      the SCALING, not the Hasselt-K attribution.

    To identify the Hasselt-K factor, a natural-K bridge across
    envs with native |A| variation (MetaMaze=4, MC=3, Acrobot=3,
    Asterix=5, PacMan=5) is required; this corpus does not yet
    carry that panel.

    `min_baseline_predictor=0.5` excludes strata where vanilla's
    jens is below the noise floor (Type 1 has no headroom to
    develop). Pre-registered as a noise-floor exclusion, not
    post-hoc tuning."""
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
    """DDQN's bias-reduction magnitude scales monotonically with
    γ at FR × MLP[64,64] × unshaped × k_eff=4.

    Per-γ independent-samples Cohen's d + mean-diff on jensen_gap
    across γ ∈ {0.99, 0.999}. HELD iff:

    1. Per-stratum cohen_d ≤ -0.8 at BOTH γ strata (large Cohen's
       d at every γ in scope), AND
    2. |mean_diff(γ=0.999)| ≥ 3 × |mean_diff(γ=0.99)| (absolute
       magnitude of bias reduction grows with γ).

    Refutations:
    - NO_EFFECT/SIGN_FLIP: any γ shows d > 0 (DDQN INCREASES jens).
    - NO_EFFECT/NULL: either γ shows d > -0.8.
    - POWER_INSUFFICIENT: amplification ratio < 3.

    k_eff=4 (native FR action count, no action_duplicate wrapper)
    is fixed to hold the k-axis constant.

    What this bridge does NOT claim:
    - That the empirical amplification identifies Hasselt's
      `1/(1−γ)` factor specifically. The structural prediction
      is 10× from γ=0.99→0.999; the ≥ 3× threshold is loose
      enough to admit any monotone-in-γ scaling. Cannot
      discriminate Hasselt-bound amplification from the
      alternative mechanism documented in
      `findings_q_explosion_direct_evidence`: at FR γ=0.999
      vanilla never finds the goal (MC ≈ 0.005 throughout) and
      Q grows to 19,520× MC because no observational anchor
      exists — a degenerate dynamic, not bootstrap-chain
      amplification. Both alternatives predict γ-monotone Q
      growth at this scope.

    To discriminate, the threshold would need tightening to ≥ 8×
    (within 1.25× of the structural prediction) and a third
    γ-stratum (0.995) testing linearity-in-1/(1−γ). Both are
    follow-up work."""
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
    is more negative than at linear by at least 0.5 Cohen units.

    Refutations:
    - NO_EFFECT/SIGN_FLIP: slope significantly POSITIVE (DDQN's
      effect MORE negative at linear than at MLP).
    - NO_EFFECT/NULL: slope CI brackets the threshold.
    - POWER_INSUFFICIENT: n_strata < 6 (we have at minimum 3 envs
      × 2 γ × 2 fa = 12 strata at full ingest; require half).

    What this bridge does NOT claim:
    - That FA capacity identifies Hasselt's σ_action factor.
      The empirical pattern (Δd more negative at MLP than linear)
      is consistent with σ_action being FA-capped at linear —
      the Hasselt-σ interpretation. It is ALSO consistent with
      Type-1-truncation at linear FA: per
      `findings_two_types_of_bias`, linear FA truncates Q
      before the max-bias amplifier has headroom, so Type 1 is
      small regardless of within-state across-action σ. The
      binary `fa_capacity` proxy CANNOT discriminate these.
    - Anything about MetaMaze (encoded as the MM-exception
      bridge — see `linear_fa_cap_fails_at_metamaze_g999__exception`).
    - Anything about CartPole / Catch / DeepSea (excluded
      because vanilla doesn't overshoot at any FA — moderator
      effect is unmeasurable when there's no signal to
      moderate).

    The sibling discriminator bridge
    `sigma_action_predicts_ddqn_jens_reduction` (same scope, same
    stratify_by) tests the σ_action attribution directly via
    continuous meta-regression on `q_action_std_late`. Empirical
    state on this corpus: slope = +2.84, p=0.143 — sign-flipped
    from Hasselt's prediction (NS at n_strata=12). The σ_action
    attribution is NOT corroborated; the FA-cap effect this
    bridge HELDs on goes through a non-σ path (likely Type-1
    FA-truncation per `findings_two_types_of_bias`).

    The scope restriction to {FR, Acrobot, MountainCar} is a
    pre-registered inclusion criterion ("envs where vanilla
    MLP develops substantive bias") — not a post-hoc filter on
    moderator results. MetaMaze + the bsuite light envs are
    excluded for separately-documented mechanism reasons."""
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
    """Discriminating σ_action test: does baseline σ_action
    (within-state across-action Q SD) predict DDQN's effect on
    jensen_gap as Hasselt's σ factor implies?

    Per-(env, gamma, fa_kind) independent-samples Cohen's d on
    `jensen_gap` → random-effects meta-regression on the
    continuous per-stratum baseline mean of `q_action_std_late`
    (the proper σ_action measure per
    `findings_sigma_K_scaling_corroborated`).

    HELD iff slope ≤ `slope_threshold` (= −2.0) AND significant:
    DDQN's effect on jens becomes more negative as baseline
    σ_action grows, the directional prediction of Hasselt's
    bound. This is the test that the binary
    `fa_capacity_moderates_ddqn_jens_reduction` bridge cannot
    perform — that bridge HELDs on the FA-capacity binary, but
    the binary cannot discriminate σ_action capping from
    Type-1 FA-truncation.

    Empirical state on this corpus (12 strata, 3 envs × 2 γ × 2
    fa, baseline σ_action ranges 0.005–0.36):

        slope = +2.84  CI = [-1.14, +6.82]  p = 0.143
        intercept = -0.96  p = 0.017

    The slope is **sign-flipped** from the Hasselt prediction
    (Hasselt: σ↑ → d_jens↓; observed: σ↑ → d_jens↑, NS). The
    intercept is significantly negative — DDQN reduces jens
    even at σ=0 — which is inconsistent with the σ-mediator
    story (no σ → no Hasselt bias → no DDQN reduction
    expected).

    Verdict: POWER_INSUFFICIENT at this n_strata. The point
    estimate already favors REFUTATION of the Hasselt-σ
    mediator hypothesis (sign-flipped slope + nonzero
    intercept); the data do not reject the null at α=0.05.

    Refutations (when reached):
    - NO_EFFECT (sig POSITIVE slope ≥ +2.0): σ_action is NOT
      the Hasselt mediator; the FA-cap effect goes through a
      non-σ path (likely Type-1 FA-truncation per
      `findings_two_types_of_bias`).
    - HELD: σ_action is the Hasselt mediator at this scope.

    Sibling to `fa_capacity_moderates_ddqn_jens_reduction`:
    that bridge tests the FA-CAPACITY moderator (HELD), this
    bridge tests the σ_ACTION moderator (POW_INSUF / point
    estimate sign-flipped). Together they encode "FA capacity
    matters, σ_action does not" — the FA-cap effect is NOT
    mediated by σ_action; Hasselt's σ_action attribution is
    NOT corroborated by the proper continuous measure on this
    corpus.

    NOT in either cluster Finding. Stands as the standalone
    σ-discriminator bridge that documents why the cluster's
    `fa_capacity_moderates_*` bridge HELDs without identifying
    Hasselt's σ factor."""
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
    """At MetaMaze γ=0.999 with linear FA, the FA-capacity rule
    FAILS — DDQN substantively reduces jens with a non-trivial
    effect at every eval-power level tested.

    Per-(n_episodes) independent-samples Cohen's d on
    `jensen_gap` at (env=MetaMaze, γ=0.999, linear, unshaped).
    n_episodes=5 shows d ≈ −0.5; n_episodes=20 (eval-power
    fixed) shows d ≈ −1.1.

    HELD iff per-stratum d ≤ `per_stratum_d_threshold` (= −0.3,
    Cohen's "small") at EVERY n_episodes stratum.

    What this bridge does NOT claim:
    - That the FA-fit-error mechanism is empirically
      corroborated. The mechanism story (MetaMaze redraws a
      random maze per evaluation episode → linear FA cannot
      generalize across mazes → vanilla's bootstrap target is
      biased by FA-fit error → DDQN clips it via a non-σ path)
      is ASSERTED from env structure but NOT empirically
      discriminated from alternatives. Specifically: per
      `findings_two_types_of_bias`, MM has an intermediate
      T1/T2 ratio (0.21-0.71), so Type 1 still contributes at
      MM linear; an FR-style late-divergence at sufficient |Q|
      under linear is also possible. The bridge documents the
      empirical anomaly; the proposed mechanism is the cleanest
      narrative but is not measured.
    - That this exception generalises to other non-stationary
      envs. No other env with per-episode state-distribution
      shift (random-init Catch, randomized DeepSea, etc.) has
      been tested at linear FA. The rule + exception cluster
      with `fa_capacity_moderates_ddqn_jens_reduction` is
      currently a 1-env exception, not a generalizable
      structure.

    To corroborate the FA-fit-error story, a probe at other
    random-init / non-stationary envs at linear FA is required.
    Follow-up work."""
    del stratify_by, min_baseline_predictor
    return per_stratum_d_threshold_verdict(
        stratified_arm_diff_pooled,
        threshold=per_stratum_d_threshold,
        sign=-1,
        min_strata=min_strata,
    )
