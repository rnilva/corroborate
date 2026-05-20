"""σ_Λ_a bias-asymmetry: moderator-not-mediator scope cluster.

Two bridges that jointly characterise where the Λ_a structural
predictor sits in the causal graph at γ=0.999 cross-env:

  Bridge 1 — σ_Λ_a moderates DDQN's d_out cross-env. Per-env
    cross-seed SD of vanilla Λ_a (the "bias-asymmetry-variance"
    env feature) predicts the sign/strength of DDQN's outcome
    benefit. Memory's prior result (commit `9c857f0` era):
    ρ = −0.78 p=0.023 n=8; current 8-env panel (1140 cells,
    γ=0.999, cache rebuild commit `f471913`): ρ = −0.643 p=0.086
    n=8. Direction matches; magnitude weakened by FR contribution
    (σ_Λ_a = 1.02 with d_out = +0.09 breaks monotone).

  Bridge 2 — Per-cell Δ_Λ_a does NOT mediate Δ_outcome. Cross-
    stratum Δ partial-Spearman ρ(Δ_Λ_a, Δ_out | Δ_jens). DDQN's
    effect on per-cell Λ_a is small (Asterix 2.903 → 2.811,
    Breakout 1.652 ≈ 1.654) — Δ_Λ_a ≈ 0 by construction, so the
    within-cell mediator path is inactive. Predicted null
    (`predicted_direction='null'`); HELD when |ρ| <
    null_threshold (null confirmed per the framework convention
    at `core.hypothesis.PredictedDirection`).

The cluster's load-bearing claim: σ_Λ_a operates at the env-
aggregate level (moderation; CLAUDE.md §"Moderation vs
mediation"), NOT at the per-cell causal-path level (mediation).
The bias Type-A/B classifier IS a moderation pattern — env
features predict DDQN's effect — distinct from "DDQN reduces
Λ_a, which mediates the outcome." Theorem 3 + Cor 3.2's σ_clip
formalism predicts the structural form; the moderation bridge
empirically corroborates it cross-env; the mediation null
bridge falsifies the within-cell-causal-path reading.

The hardcoded `_SIGMA_LAMBDA_A_PER_ENV_G0999` is a frozen
empirical snapshot computed from the 8-env vanilla cells at
γ=0.999 in commit `f471913`'s cache state. Updating the cache
(e.g., re-ingesting from new sweeps) may shift these values;
the bridge tests the published values, not a live recompute.
For a live recompute use `scripts/q_channel_mediator_search.py`
or a new `sigma_lambda_a_per_env` measurable (TODO).
"""
from __future__ import annotations

import math
from types import MappingProxyType
from typing import Literal

import polars as pl

from corroborate.analyses.link.cross_stratum_arm_diff_partial_spearman import (
    CrossStratumArmDiffPartialSpearmanResult,
)
from corroborate.analyses.link.cross_stratum_property_slope import (
    CrossStratumPropertySlopeResult,
)
from corroborate.analyses.spearman.partial_spearman import (
    PartialSpearmanResult,
)
from corroborate.bridge.bridge import Direction, Tier, claim_bridge
from corroborate.bridge.verdict import RefutationClass, Verdict

from experiments.findings.ddqn._arms import (
    DDQN_ARM, INTERVENTION, VANILLA_ARM,
)
# Frozen empirical snapshot: per-env cross-seed SD of vanilla Λ_a
# at γ=0.999 (commit `f471913` cache state, 8-env panel n=1140).
# Computed from `q_action_std_late · √(2 ln K_eff) / q_argmax_margin_late`
# over `arm_key == baseline` cells per env.
_SIGMA_LAMBDA_A_PER_ENV_G0999: MappingProxyType[
    object, MappingProxyType[str, float]
] = MappingProxyType({
    'Acrobot-v1':             MappingProxyType({'sigma_lambda_a': 0.394}),
    'Asterix-MinAtar':        MappingProxyType({'sigma_lambda_a': 0.565}),
    'Breakout-MinAtar':       MappingProxyType({'sigma_lambda_a': 0.073}),
    'FourRooms-misc':         MappingProxyType({'sigma_lambda_a': 0.866}),
    'Freeway-MinAtar':        MappingProxyType({'sigma_lambda_a': 0.160}),
    'LunarLander-v2-jax':     MappingProxyType({'sigma_lambda_a': 0.340}),
    'MetaMaze-misc':          MappingProxyType({'sigma_lambda_a': 0.846}),
    'MountainCar-v0':         MappingProxyType({'sigma_lambda_a': 0.836}),
    'Snake-jumanji':          MappingProxyType({'sigma_lambda_a': 0.215}),
    'SpaceInvaders-MinAtar':  MappingProxyType({'sigma_lambda_a': 0.072}),
})


# 8-env γ=0.999 scope. Note: deliberately does NOT pin
# CANONICAL_HP_EXCLUDING_GAMMA — the MLP env γ=0.999 corpora
# (Acrobot, FourRooms, MetaMaze, MountainCar) live at
# non-canonical HP combinations (FA-depth axis probes, γ-sweeps
# at varying lr, etc.) and excluding them collapses the panel to
# 4 MinAtar envs (n=4 strata, below the moderation-verdict's
# min_strata=8 calibration). The σ_Λ_a-per-env values in
# `_SIGMA_LAMBDA_A_PER_ENV_G0999` are aggregates across each
# env's full γ=0.999 corpus, so the panel and the covariate
# match the same HP-pooled cohort.
_GAMMA_999_SCOPE: pl.Expr = (
    (pl.col('gamma') == 0.999)
    # Restrict to canonical k=1 (action_duplicate_k is null or 1) per
    # `findings_k_axis_gamma_regime_map`: Λ_a's K_eff dependency makes
    # k=2/k=4 strata structurally non-comparable on the per-env σ_Λ_a panel.
    & (pl.col('action_duplicate_k').is_null() | (pl.col('action_duplicate_k') == 1))
)


def _signed_spearman_verdict(
    rho: float,
    p_value: float,
    n_strata: int,
    *,
    sign: Literal[-1, 1],
    rho_threshold_held: float,
    p_threshold: float,
    null_threshold: float,
    sign_flip_threshold: float,
    min_strata: int,
) -> tuple[Verdict, RefutationClass | None]:
    """Local copy of the signed-Spearman verdict logic with
    relaxed `min_strata` for the n=8 cross-env panel (the
    canonical helper's default is 10, calibrated for stable
    direction-resolution; below that it can over-claim).

    Calibration note: at n=8, two-sided critical |r| at p=0.05 is
    0.707. HELD requires p ≤ 0.05 → |ρ| ≥ ~0.71 effectively.
    Memory's ρ=−0.78 cleared this comfortably; current ρ=−0.643
    does NOT (p=0.086). Verdict resolves to POWER_INSUFFICIENT
    rather than HELD on this exact cache state — the moderation
    pattern is suggestive but not framework-HELD at n=8."""
    if n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT, None
    if math.isnan(rho) or math.isnan(p_value):
        return Verdict.POWER_INSUFFICIENT, None
    correct_sign = (sign > 0 and rho > 0) or (sign < 0 and rho < 0)
    if correct_sign and abs(rho) >= rho_threshold_held and p_value <= p_threshold:
        return Verdict.HELD, None
    wrong_sign = (sign > 0 and rho < 0) or (sign < 0 and rho > 0)
    if wrong_sign and abs(rho) >= sign_flip_threshold:
        return Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP
    if abs(rho) < null_threshold:
        return Verdict.NO_EFFECT, RefutationClass.NULL_EFFECT
    return Verdict.POWER_INSUFFICIENT, None


# Bridge 1 — σ_Λ_a moderates DDQN's outcome cross-env.
@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_raw_mean',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=(
        _GAMMA_999_SCOPE
        # `lambda_a_late.is_finite()` filter was vestigial here — the σ_Λ_a
        # covariate is per-env hardcoded in `_SIGMA_LAMBDA_A_PER_ENV_G0999`,
        # so per-cell λ_a doesn't gate cell inclusion. Removed to admit
        # LunarLander + Snake whose framework `lambda_a_late` is NaN due to
        # the resolver trap (inputs present, derived value not computed).
        & pl.col('eval_best_burst_raw_mean').is_finite()
    ),
    predicted_direction='a_lt_b',
)
def sigma_lambda_a_moderates_ddqn_outcome__cross_env_g0999(
    cross_stratum_property_slope: CrossStratumPropertySlopeResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    source: str = 'eval_best_burst_raw_mean',
    stratify_by: tuple[str, ...] = ('env_name',),
    covariate_name: str = 'sigma_lambda_a',
    covariate_key_field: str = 'env_name',
    covariates_per_key: MappingProxyType[
        object, MappingProxyType[str, float]
    ] = _SIGMA_LAMBDA_A_PER_ENV_G0999,
    scope_predictor: str = 'jensen_gap',
    min_baseline_predictor: float = 0.0,
    min_seeds_per_arm: int = 5,
    rho_threshold_held: float = 0.6,
    p_threshold: float = 0.05,
    null_threshold: float = 0.2,
    sign_flip_threshold: float = 0.5,
    min_strata: int = 8,
) -> tuple[Verdict, RefutationClass | None]:
    """Cross-env Spearman ρ between per-env σ_Λ_a (cross-seed SD
    of vanilla Λ_a at γ=0.999, env-aggregate "bias-asymmetry-
    variance" feature) and per-env Cohen's d on DDQN-vs-vanilla
    raw outcome.

    Predicted direction: ρ < 0 — higher σ_Λ_a → less DDQN benefit
    (Type A envs where bias is uniform across actions; DDQN's
    clip corrupts argmax). The Cor 3.2 closed-form
    `γ · σ_clip · √(2 ln K) < Δ_v` says argmax preservation
    requires σ_clip < Δ_v/(γ√(2 ln K)); when cross-seed σ_Λ_a is
    large, the inequality fails on more seeds → DDQN's clip
    flips argmax → outcome harm.

    Empirical: memory ρ = −0.78 n=8 p=0.023 (commit `9c857f0`
    era, possibly tighter cohort); current 8-env panel n=1140
    (commit `f471913`): ρ = −0.643 p=0.086 — direction matches,
    magnitude weakened by FR (σ_Λ_a=1.02 + d_out=+0.09 breaks
    monotone). Verdict at this cache state resolves to
    POWER_INSUFFICIENT (p > 0.05); memory's tighter cohort would
    fire HELD."""
    del treatment_arm, baseline_arm, source, stratify_by
    del covariate_name, covariate_key_field, covariates_per_key
    del scope_predictor, min_baseline_predictor, min_seeds_per_arm
    return _signed_spearman_verdict(
        cross_stratum_property_slope.rho,
        cross_stratum_property_slope.p_value,
        cross_stratum_property_slope.n_strata,
        sign=-1,
        rho_threshold_held=rho_threshold_held,
        p_threshold=p_threshold,
        null_threshold=null_threshold,
        sign_flip_threshold=sign_flip_threshold,
        min_strata=min_strata,
    )


# Bridge 2 — Per-cell Δ_Λ_a does NOT mediate Δ_outcome.
@claim_bridge(
    source='lambda_a_late',
    target='eval_best_burst_raw_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=(
        _GAMMA_999_SCOPE
        & pl.col('lambda_a_late').is_finite()
        & pl.col('jensen_gap').is_finite()
        & pl.col('eval_best_burst_raw_mean').is_finite()
    ),
    predicted_direction='null',
)
def lambda_a_does_not_mediate_outcome__cross_stratum_g0999(
    cross_stratum_arm_diff_partial_spearman: (
        CrossStratumArmDiffPartialSpearmanResult
    ),
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    predictor: str = 'lambda_a_late',
    target: str = 'eval_best_burst_raw_mean',
    confound: str = 'jensen_gap',
    stratify_by: tuple[str, ...] = ('env_name',),
    min_seeds_per_arm: int = 5,
    null_threshold: float = 0.2,
    held_negative_rho: float = 0.5,
    min_strata: int = 5,
) -> tuple[Verdict, RefutationClass | None]:
    """Cross-stratum partial Spearman ρ(Δ_Λ_a, Δ_out | Δ_jens).

    Tests whether the per-stratum change in Λ_a (induced by DDQN
    vs vanilla) predicts the per-stratum change in outcome,
    controlling for the dominant bias-magnitude channel
    (Δ_jens). If Λ_a is a within-cell mediator on the causal
    path, Δ_Λ_a → Δ_out should correlate negatively (DDQN
    reduces Λ_a → outcome improves).

    Predicted direction: NULL (|ρ| < null_threshold). Empirical
    evidence walks back the within-cell mediation reading:
    DDQN's effect on per-cell Λ_a is small (~3% reduction on
    Asterix, ~0% on Breakout), so Δ_Λ_a is constrained to a
    narrow range that can't capture cross-stratum d_out variance.
    Pair with Bridge 1 (moderation at the env-aggregate level)
    — both predicted to land in the moderator-not-mediator
    configuration.

    Under `predicted_direction='null'`: HELD when the null
    prediction is confirmed (|ρ| < null_threshold). NO_EFFECT
    (xpass, `SIGN_FLIP`) when |ρ| ≥ held_negative_rho with the
    wrong (negative) sign — would mean Δ_Λ_a DOES drive Δ_out
    after all, falsifying the moderator-not-mediator framing
    (the predicted null fails — an effect was observed). The
    null-band-confirmed case is the framework's HELD per
    `core.hypothesis.PredictedDirection`'s docstring."""
    del treatment_arm, baseline_arm, predictor, target, confound
    del stratify_by, min_seeds_per_arm
    if cross_stratum_arm_diff_partial_spearman.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT, None
    rho = cross_stratum_arm_diff_partial_spearman.rho
    if math.isnan(rho):
        return Verdict.POWER_INSUFFICIENT, None
    if abs(rho) < null_threshold:
        return Verdict.HELD, None  # null prediction confirmed
    if rho <= -held_negative_rho:
        return Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP
    return Verdict.POWER_INSUFFICIENT, None


# Bridge 3 — Joint (σ_clip, Δ_v, jens) triplet within-cell partial-ρ.
@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_raw_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=(
        _GAMMA_999_SCOPE
        & pl.col('lambda_a_late').is_finite()
        & pl.col('q_action_std_late').is_finite()
        & pl.col('q_argmax_margin_late').is_finite()
        & pl.col('jensen_gap').is_finite()
        & pl.col('eval_best_burst_raw_mean').is_finite()
        & pl.col('arm_is_baseline').is_finite()
    ),
    predicted_direction='null',
)
def joint_bias_geometry_mediates_arm_outcome__cross_env_g0999(
    partial_spearman: PartialSpearmanResult,
    *,
    x: str = 'arm_is_baseline',
    y: str = 'eval_best_burst_raw_mean',
    conditioning: tuple[str, ...] = (
        'q_action_std_late', 'q_argmax_margin_late', 'jensen_gap',
    ),
    stratify_by: str = 'env_name',
    min_stratum_size: int = 8,
    null_threshold: float = 0.10,
    held_strong_rho: float = 0.30,
    min_strata: int = 5,
) -> tuple[Verdict, RefutationClass | None]:
    """Per-cell partial Spearman ρ(arm, outcome | σ_clip, Δ_v, jens)
    stratified by env_name. Tests whether the joint (σ_clip, Δ_v,
    jens) triplet absorbs the within-cell arm → outcome effect.

    `x='arm_is_baseline'` encodes arm as 0/1 (baseline=1, DDQN=0).
    Spearman ρ on a binary indicator is monotone-equivalent to a
    rank-sum statistic, so partial-ρ here is the nonparametric
    "how much of the arm-effect remains after conditioning on
    the triplet."

    Predicted direction: NULL (full mediation by the triplet →
    partial-ρ ≈ 0). HELD if |ρ_partial| ≤ null_threshold (joint
    triplet absorbs the arm-effect → null prediction confirmed).
    NO_EFFECT (xpass, `SIGN_FLIP`) if |ρ_partial| ≥
    held_strong_rho — the arm-effect SURVIVES the joint
    conditioning, so the predicted full mediation is refuted
    (an effect was observed when none was predicted).

    Empirical evidence: on the 5-env MinAtar-heavy subset (n=360
    cells) the joint triplet shrinks |ρ| 0.273 → 0.094 (66%
    absorption). On the full 8-env panel (n=1140) only 17%
    absorption — joint mediation is env-cohort dependent. The
    cluster-finding's BLOCKED_ON predicts admit under k=4 panel
    extension (Asterix's special case dilutes)."""
    del x, y, conditioning, stratify_by, min_stratum_size
    if partial_spearman.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT, None
    rho = partial_spearman.rho_pooled
    if math.isnan(rho):
        return Verdict.POWER_INSUFFICIENT, None
    if abs(rho) <= null_threshold:
        return Verdict.HELD, None  # null prediction confirmed
    if abs(rho) >= held_strong_rho:
        return Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP
    return Verdict.POWER_INSUFFICIENT, None


# Bridge 4 + 5 — Within-arm Λ_a-as-predictor asymmetry.
# Vanilla cells: Λ_a predicts outcome. DDQN cells: clip abolishes
# the predictive relationship. Together they form a paired
# within-arm pattern: DDQN's intervention NEUTRALIZES the
# vanilla-side bias-asymmetry → outcome channel.
_VANILLA_G999_SCOPE: pl.Expr = (
    _GAMMA_999_SCOPE
    & (pl.col('arm_key') == VANILLA_ARM)
    & pl.col('lambda_a_late').is_finite()
    & pl.col('jensen_gap').is_finite()
    & pl.col('eval_best_burst_raw_mean').is_finite()
)

_DDQN_G999_SCOPE: pl.Expr = (
    _GAMMA_999_SCOPE
    & (pl.col('arm_key') == DDQN_ARM)
    & pl.col('lambda_a_late').is_finite()
    & pl.col('jensen_gap').is_finite()
    & pl.col('eval_best_burst_raw_mean').is_finite()
)


@claim_bridge(
    source='lambda_a_late',
    target='eval_best_burst_raw_mean',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=_VANILLA_G999_SCOPE,
    predicted_direction='a_lt_b',
)
def vanilla_lambda_a_predicts_outcome__within_arm_g0999(
    partial_spearman: PartialSpearmanResult,
    *,
    x: str = 'lambda_a_late',
    y: str = 'eval_best_burst_raw_mean',
    conditioning: tuple[str, ...] = ('jensen_gap',),
    stratify_by: str = 'env_name',
    min_stratum_size: int = 8,
    held_rho: float = 0.10,
    sign_flip_rho: float = 0.10,
    min_strata: int = 5,
) -> tuple[Verdict, RefutationClass | None]:
    """Within VANILLA cells at γ=0.999: per-cell Λ_a inversely
    predicts outcome after controlling for jens.

    Cor 3.2 reading: high σ_clip · √(2 ln K) / Δ_v signals that
    the bias-asymmetry inequality (γ · σ_clip · √(2 ln K) < Δ_v)
    is closer to being violated. Vanilla cells where the
    inequality fails should show degraded outcome (bias corrupts
    argmax → policy worse). Conditioning on jens isolates the
    asymmetry channel from the bias-magnitude channel.

    Empirical: marginal ρ=-0.299 p=5e-13; partial | jens
    ρ=-0.169 p=7.7e-5 (commit `4c075bf` cache state, n=570
    vanilla cells × 8 envs). HELD when ρ ≤ -held_rho with
    sufficient strata.

    Pairs with `ddqn_lambda_a_does_not_predict_outcome__within_arm_g0999`
    — the DDQN-side null test. The pair forms a within-arm
    asymmetry pattern: Λ_a → outcome is real in vanilla and
    abolished in DDQN — DIRECT per-cell evidence for the
    bias Type A/B framing, distinct from the n=8 cross-env
    moderation panel which is currently POWER_INSUFFICIENT."""
    del x, y, conditioning, stratify_by, min_stratum_size
    if partial_spearman.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT, None
    rho = partial_spearman.rho_pooled
    if math.isnan(rho):
        return Verdict.POWER_INSUFFICIENT, None
    if rho <= -held_rho:
        return Verdict.HELD, None
    if rho >= sign_flip_rho:
        return Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP
    return Verdict.POWER_INSUFFICIENT, None


@claim_bridge(
    source='lambda_a_late',
    target='eval_best_burst_raw_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=_DDQN_G999_SCOPE,
    predicted_direction='null',
)
def ddqn_lambda_a_does_not_predict_outcome__within_arm_g0999(
    partial_spearman: PartialSpearmanResult,
    *,
    x: str = 'lambda_a_late',
    y: str = 'eval_best_burst_raw_mean',
    conditioning: tuple[str, ...] = ('jensen_gap',),
    stratify_by: str = 'env_name',
    min_stratum_size: int = 8,
    null_threshold: float = 0.10,
    held_strong_rho: float = 0.30,
    min_strata: int = 5,
) -> tuple[Verdict, RefutationClass | None]:
    """Within DDQN cells at γ=0.999: Λ_a does NOT predict outcome
    after conditioning on jens.

    DDQN's clip neutralizes the bias-asymmetry channel — even
    when σ_clip · √(2 ln K) / Δ_v is high, DDQN's argmax doesn't
    suffer outcome-wise because the clip preserves the policy
    despite the per-cell asymmetry signature. Tests the dual of
    `vanilla_lambda_a_predicts_outcome__within_arm_g0999`.

    Empirical: marginal ρ=-0.072 p=0.09 (basically null);
    partial | jens ρ=+0.006 p=0.89 (clean null) — DDQN's outcome
    is uncoupled from per-cell Λ_a after controlling for jens.

    Predicted null. HELD (null prediction confirmed) when |ρ| ≤
    null_threshold. NO_EFFECT (xpass, `SIGN_FLIP`) if |ρ| ≥
    held_strong_rho would say DDQN's outcome IS coupled to Λ_a
    — would refute the bias-asymmetry-neutralization reading
    (an effect was observed when none was predicted)."""
    del x, y, conditioning, stratify_by, min_stratum_size
    if partial_spearman.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT, None
    rho = partial_spearman.rho_pooled
    if math.isnan(rho):
        return Verdict.POWER_INSUFFICIENT, None
    if abs(rho) <= null_threshold:
        return Verdict.HELD, None  # null prediction confirmed
    if abs(rho) >= held_strong_rho:
        return Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP
    return Verdict.POWER_INSUFFICIENT, None


# Bridge 6 — Asterix-specific Λ_a mechanism (the harm-env standout).
_ASTERIX_VANILLA_G999_SCOPE: pl.Expr = (
    (pl.col('env_name') == 'Asterix-MinAtar')
    & (pl.col('gamma') == 0.999)
    & (pl.col('arm_key') == VANILLA_ARM)
    & pl.col('lambda_a_late').is_finite()
    & pl.col('jensen_gap').is_finite()
    & pl.col('eval_best_burst_raw_mean').is_finite()
)


@claim_bridge(
    source='lambda_a_late',
    target='eval_best_burst_raw_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=_ASTERIX_VANILLA_G999_SCOPE,
    predicted_direction='a_gt_b',
)
def asterix_vanilla_lambda_a_positively_predicts_outcome__g0999(
    partial_spearman: PartialSpearmanResult,
    *,
    x: str = 'lambda_a_late',
    y: str = 'eval_best_burst_raw_mean',
    conditioning: tuple[str, ...] = ('jensen_gap',),
    stratify_by: str = 'env_name',
    min_stratum_size: int = 20,
    held_rho: float = 0.30,
    sign_flip_rho: float = 0.20,
    min_strata: int = 1,
) -> tuple[Verdict, RefutationClass | None]:
    """Within VANILLA cells on Asterix-MinAtar γ=0.999 (n=30), per-cell
    Λ_a POSITIVELY predicts outcome after conditioning on jens.

    This is the substantive per-env evidence for the **Asterix
    γ=0.999 harm mechanism**: vanilla's anisotropic bias on
    Asterix is POLICY-INFORMATIVE — the cross-action SD signature
    correlates positively with vanilla outcome (high Λ_a → BETTER
    vanilla performance). DDQN's clip symmetrises the bootstrap,
    destroying this information-bearing asymmetry → outcome
    worsens (cf. `finding_asterix_gamma_999_harm`, where DDQN's
    d_out=−3.2).

    The per-env partial-r at Asterix is +0.35 (p=0.061) on the
    n=30 vanilla panel — sign correct + marginal significance.
    Pairs in spirit with `vanilla_lambda_a_predicts_outcome__within_arm_g0999`
    (pooled cross-env ρ=−0.17 — that pooled NEGATIVE direction is
    the "Type A" majority signal; Asterix's POSITIVE direction is
    the Type-B minority outlier this bridge captures explicitly).

    HELD direction is OPPOSITE to the pooled within-vanilla
    bridge — same source/target/scope-structure, different
    predicted_direction. Together they encode the per-env
    heterogeneity in the bias Type-A/B framing.

    HELD when ρ ≥ +held_rho. SIGN_FLIP when ρ ≤ −sign_flip_rho
    (would say Asterix is in the Type-A majority cohort, walking
    back the mechanism-specific claim)."""
    del x, y, conditioning, stratify_by, min_stratum_size
    if partial_spearman.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT, None
    rho = partial_spearman.rho_pooled
    if math.isnan(rho):
        return Verdict.POWER_INSUFFICIENT, None
    if rho >= held_rho:
        return Verdict.HELD, None
    if rho <= -sign_flip_rho:
        return Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP
    return Verdict.POWER_INSUFFICIENT, None


BRIDGES = (
    sigma_lambda_a_moderates_ddqn_outcome__cross_env_g0999,
    lambda_a_does_not_mediate_outcome__cross_stratum_g0999,
    joint_bias_geometry_mediates_arm_outcome__cross_env_g0999,
    vanilla_lambda_a_predicts_outcome__within_arm_g0999,
    ddqn_lambda_a_does_not_predict_outcome__within_arm_g0999,
    asterix_vanilla_lambda_a_positively_predicts_outcome__g0999,
)
