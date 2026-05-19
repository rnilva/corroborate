"""Synthetic bias Type-A/B controlled-substrate bridges (v2).

v1 → v2: see `src/corroborate_rl/corroborate_rl/synthetic_bias_typeb.py`
module docstring + `/tmp/synthetic_env_roast.md`. v1's bridges
tested ρ(reward_variance_scale, d_out) and ρ(reward_sparsity,
d_out) on a 12-env panel where the underlying env was a bandit
in a tuxedo (action-independent transitions). v2 replaces the
env shape entirely; bridges here test the v2 axes.

The v2 panel has TWO structural axes:
- L = n_states ∈ {16, 64} (FA-capacity)
- α = anisotropy_alpha ∈ {-0.5, 0, +0.5} (Type-A/B)

Plus γ ∈ {0.95, 0.99, 0.999} as a substrate axis. 6 envs × 3
γ × 2 arms × 12 seeds = 432 cells.

The substantive predictions (P1 — anisotropy_alpha drives DDQN's
sign; P2 — γ amplifies the Type-A/B split; P3 — L amplifies the
split) are pre-registered in
`docs/PRE_REGISTRATION_synthetic_bias_typeb.md`.
"""
from __future__ import annotations

import math
from types import MappingProxyType
from typing import Literal

import polars as pl

from corroborate.analyses.link.cross_stratum_property_slope import (
    CrossStratumPropertySlopeResult,
)
from corroborate.bridge.bridge import Direction, Tier, claim_bridge
from corroborate.bridge.verdict import RefutationClass, Verdict

from experiments.findings.ddqn._arms import (
    DDQN_ARM, INTERVENTION, VANILLA_ARM,
)


# v2 synthetic env naming convention:
#   "TypeBChainV2-K{K}-L{n_states}-alpha{anisotropy_alpha}-synthetic"
# γ is a substrate axis (set via YAML), not baked into the env
# name. The (env_name, gamma) pair identifies a cell stratum.
#
# Hardcoded structural covariates mirror chain_depth's
# `_LOG_HORIZON_PER_GAMMA` pattern. These are STRUCTURAL —
# derivable from the env name itself — not data-fitted.
def _build_synthetic_covariates() -> MappingProxyType[
    object, MappingProxyType[str, float]
]:
    """Per-env structural covariates: (n_states, anisotropy_alpha).

    Keyed by env name. The values encode the env's STRUCTURAL
    design parameters (set at registration time in
    `env_catalogue._register_synthetic_bias_typeb_panel`), NOT
    empirical observables. Hardcoding is appropriate because the
    env IS its structural config — there's no drift between cache
    state and these values."""
    out: dict[object, MappingProxyType[str, float]] = {}
    for n_states in (16, 64):
        for alpha in (-0.5, 0.0, 0.5):
            name = (
                f"TypeBChainV2-K4-L{n_states}"
                f"-alpha{alpha}-synthetic"
            )
            out[name] = MappingProxyType({
                'n_states': float(n_states),
                'anisotropy_alpha': float(alpha),
                # Composite "Type-B-ness" score: positive α (best
                # action noisy) × log(n_states) (FA capacity
                # binding). Both ingredients amplify the
                # type-B harm prediction.
                'type_b_score': float(alpha) * math.log(float(n_states)),
            })
    return MappingProxyType(out)


_SYNTHETIC_COVARIATES: MappingProxyType[
    object, MappingProxyType[str, float]
] = _build_synthetic_covariates()


# Scope: v2 synthetic envs. γ NOT pinned here because the sweep
# spans γ ∈ {0.95, 0.99, 0.999} and individual bridges decide
# whether to pin γ or pool.
_SYNTHETIC_TYPEB_V2_SCOPE: pl.Expr = (
    pl.col('env_name').str.starts_with('TypeBChainV2-K4-')
    & pl.col('env_name').str.ends_with('-synthetic')
)

# γ=0.999 sub-scope for the "gamma amplifies Type-B" bridge.
_SYNTHETIC_TYPEB_V2_G999_SCOPE: pl.Expr = (
    _SYNTHETIC_TYPEB_V2_SCOPE & (pl.col('gamma') == 0.999)
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
    """Sign-aware verdict for cross-stratum Spearman slopes.

    Local copy with `min_strata` defaulting to ≤ panel size for
    the small synthetic panel."""
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


# P1 — DDQN harms under positive anisotropy_alpha (Type-B regime).
# Pooled across γ; bridges below test γ-amplification + L-axis
# as separate questions.
@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_raw_mean',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=(
        _SYNTHETIC_TYPEB_V2_SCOPE
        & pl.col('eval_best_burst_raw_mean').is_finite()
    ),
    predicted_direction='a_lt_b',
)
def ddqn_harms_under_positive_alpha__synthetic_typeb_v2(
    cross_stratum_property_slope: CrossStratumPropertySlopeResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    source: str = 'eval_best_burst_raw_mean',
    stratify_by: tuple[str, ...] = ('env_name', 'gamma'),
    covariate_name: str = 'anisotropy_alpha',
    covariate_key_field: str = 'env_name',
    covariates_per_key: MappingProxyType[
        object, MappingProxyType[str, float]
    ] = _SYNTHETIC_COVARIATES,
    scope_predictor: str = 'eval_best_burst_raw_mean',
    min_baseline_predictor: float = -1e9,  # no gate; synthetic always active
    min_seeds_per_arm: int = 5,
    rho_threshold_held: float = 0.4,
    p_threshold: float = 0.05,
    null_threshold: float = 0.2,
    sign_flip_threshold: float = 0.4,
    min_strata: int = 10,
) -> tuple[Verdict, RefutationClass | None]:
    """P1 — Cross-stratum Spearman ρ between `anisotropy_alpha`
    (the env-structural Type-A/B axis) and DDQN-vs-vanilla
    Cohen's d on outcome, stratified by (env_name, γ).

    Predicted ρ ≤ −0.4 with p ≤ 0.05 — positive α (best-action
    noisy = Type-B) → DDQN harms (clip removes policy-informative
    noise asymmetry); negative α (best-action quiet = Type-A) →
    DDQN helps (clip denoises non-best actions' max-bias). The
    canonical bias Type-A/B test on the v2 controlled substrate.

    Stratification by (env_name, γ) gives 6 envs × 3 γ = 18
    strata before missing-cell pruning, well above
    `min_strata=10`.

    HELD: ρ ≤ −0.4 with p ≤ 0.05.
    NO_EFFECT (SIGN_FLIP): ρ ≥ +0.4 — DDQN HELPS more on Type-B
    envs (would refute the central α-as-bias-shape interpretation).
    NO_EFFECT (NULL_EFFECT): |ρ| < 0.2 — α doesn't predict
    DDQN's effect direction; controlled substrate fails to
    reproduce the natural-env Asterix Type-B mechanism."""
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


# P2 — γ AMPLIFIES the Type-A/B split. At γ=0.999 (highest chain
# amplification), ρ(α, d_out) should be MORE negative than at
# γ=0.95. This tests the bias-amplification chain story: longer
# effective horizon → larger Var_a[Q*] disparity between Type-A
# and Type-B regimes.
@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_raw_mean',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=(
        _SYNTHETIC_TYPEB_V2_G999_SCOPE
        & pl.col('eval_best_burst_raw_mean').is_finite()
    ),
    predicted_direction='a_lt_b',
)
def ddqn_harm_amplified_at_g999__synthetic_typeb_v2(
    cross_stratum_property_slope: CrossStratumPropertySlopeResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    source: str = 'eval_best_burst_raw_mean',
    stratify_by: tuple[str, ...] = ('env_name',),
    covariate_name: str = 'anisotropy_alpha',
    covariate_key_field: str = 'env_name',
    covariates_per_key: MappingProxyType[
        object, MappingProxyType[str, float]
    ] = _SYNTHETIC_COVARIATES,
    scope_predictor: str = 'eval_best_burst_raw_mean',
    min_baseline_predictor: float = -1e9,
    min_seeds_per_arm: int = 5,
    rho_threshold_held: float = 0.5,
    p_threshold: float = 0.1,
    null_threshold: float = 0.2,
    sign_flip_threshold: float = 0.5,
    min_strata: int = 5,
) -> tuple[Verdict, RefutationClass | None]:
    """P2 — At γ=0.999 (load-bearing chain-amplification regime),
    ρ(anisotropy_alpha, d_out) ≤ −0.5 across the 6 v2 envs. The
    same shape as the natural-env Asterix γ=0.999 harm finding
    (memory: `findings_asterix_g999_harm_is_optimization_dynamics`).

    Predicted ρ ≤ −0.5 with p ≤ 0.1 — same direction as P1 but
    stronger at the γ=0.999 corner. The "γ amplifies Type-B"
    pattern is the load-bearing prediction this v2 design is
    built to test (γ-axis was a v1 punt; addressing critic
    recommendation #4).

    `min_strata=5` is set lower than P1's `10` because this
    bridge pins γ=0.999 (only 6 strata total at this corner).
    The smaller-n threshold trades formal-significance for
    direction-detection at the load-bearing γ corner.
    """
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


# P3 — FA-capacity (L = n_states) amplifies the Type-A/B split.
# At higher L, FA is more capacity-bound; the within-arm bias
# asymmetry that DDQN's clip removes is more policy-informative.
# Predicts ρ(type_b_score, d_out) ≤ −0.4 where type_b_score =
# α × log(L) — joint effect of both axes.
@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_raw_mean',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=(
        _SYNTHETIC_TYPEB_V2_SCOPE
        & pl.col('eval_best_burst_raw_mean').is_finite()
    ),
    predicted_direction='a_lt_b',
)
def ddqn_harm_scales_with_type_b_score__synthetic_typeb_v2(
    cross_stratum_property_slope: CrossStratumPropertySlopeResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    source: str = 'eval_best_burst_raw_mean',
    stratify_by: tuple[str, ...] = ('env_name', 'gamma'),
    covariate_name: str = 'type_b_score',
    covariate_key_field: str = 'env_name',
    covariates_per_key: MappingProxyType[
        object, MappingProxyType[str, float]
    ] = _SYNTHETIC_COVARIATES,
    scope_predictor: str = 'eval_best_burst_raw_mean',
    min_baseline_predictor: float = -1e9,
    min_seeds_per_arm: int = 5,
    rho_threshold_held: float = 0.4,
    p_threshold: float = 0.05,
    null_threshold: float = 0.2,
    sign_flip_threshold: float = 0.4,
    min_strata: int = 10,
) -> tuple[Verdict, RefutationClass | None]:
    """P3 — Cross-stratum Spearman ρ between `type_b_score = α ×
    log(L)` and DDQN-vs-vanilla Cohen's d on outcome.

    Predicted ρ ≤ −0.4 with p ≤ 0.05 — the composite Type-B
    score (positive α × log(L) — both axes pushing toward
    Type-B) should predict DDQN harm more cleanly than α alone
    (P1) because L modulates the FA-capacity that makes the
    bias asymmetry policy-informative.

    The substantive disambiguation against P1 is: if P1 HELDs
    but P3 NULLs, α alone carries the signal and L doesn't
    modulate it (the FA-capacity axis isn't load-bearing in
    this synthetic substrate). If P3 HELDs more decisively
    than P1, L genuinely amplifies the Type-B signal."""
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


# N1 — n_states ALONE doesn't predict d_out at fixed α=0
# (isotropic noise). Tests that FA-capacity is a MODULATOR not
# a DRIVER: without the anisotropy axis, more capacity shouldn't
# create an arm-difference direction.
_SYNTHETIC_TYPEB_V2_ALPHA0_SCOPE: pl.Expr = (
    _SYNTHETIC_TYPEB_V2_SCOPE
    & pl.col('env_name').str.contains('alpha0.0-synthetic')
)


@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_raw_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=(
        _SYNTHETIC_TYPEB_V2_ALPHA0_SCOPE
        & pl.col('eval_best_burst_raw_mean').is_finite()
    ),
    predicted_direction='null',
)
def n_states_alone_does_not_predict_dout__synthetic_typeb_v2(
    cross_stratum_property_slope: CrossStratumPropertySlopeResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    source: str = 'eval_best_burst_raw_mean',
    stratify_by: tuple[str, ...] = ('env_name', 'gamma'),
    covariate_name: str = 'n_states',
    covariate_key_field: str = 'env_name',
    covariates_per_key: MappingProxyType[
        object, MappingProxyType[str, float]
    ] = _SYNTHETIC_COVARIATES,
    scope_predictor: str = 'eval_best_burst_raw_mean',
    min_baseline_predictor: float = -1e9,
    min_seeds_per_arm: int = 5,
    null_threshold: float = 0.3,
    effect_observed_threshold: float = 0.6,
    min_strata: int = 4,
) -> tuple[Verdict, RefutationClass | None]:
    """N1 — Cross-stratum Spearman ρ(n_states, d_out) at α=0
    (isotropic-noise envs only): 2 envs × 3 γ = 6 strata.

    Predicted null: at α=0, neither arm has policy-informative
    bias asymmetry to preserve / destroy; n_states is a pure
    FA-capacity stress with no Type-A/B character. ρ should be
    small (|ρ| ≤ 0.3); if effect observed (|ρ| ≥ 0.6), capacity
    is doing independent work — substantive walk-back of the
    P3 interpretation.

    `min_strata=4` is lower than P1/P3 because this is a 6-
    stratum sub-panel by construction. POWER_INSUFFICIENT if
    fewer than 4 strata admit Cohen's d (e.g. all seeds
    converged identically at one α=0 env)."""
    del treatment_arm, baseline_arm, source, stratify_by
    del covariate_name, covariate_key_field, covariates_per_key
    del scope_predictor, min_baseline_predictor, min_seeds_per_arm
    if cross_stratum_property_slope.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT, None
    rho = cross_stratum_property_slope.rho
    if math.isnan(rho):
        return Verdict.POWER_INSUFFICIENT, None
    if abs(rho) <= null_threshold:
        return Verdict.HELD, None
    if abs(rho) >= effect_observed_threshold:
        return Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP
    return Verdict.POWER_INSUFFICIENT, None


BRIDGES = (
    ddqn_harms_under_positive_alpha__synthetic_typeb_v2,
    ddqn_harm_amplified_at_g999__synthetic_typeb_v2,
    ddqn_harm_scales_with_type_b_score__synthetic_typeb_v2,
    n_states_alone_does_not_predict_dout__synthetic_typeb_v2,
)
