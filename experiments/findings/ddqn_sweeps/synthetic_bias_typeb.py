"""Synthetic bias Type-A/B controlled-substrate bridges.

Tests the pre-registered predictions from
`docs/PRE_REGISTRATION_synthetic_bias_typeb.md` against the
synthetic K=4 chain MDP sweep (12 envs spanning rvs × sparsity ×
noise; `experiments/configs/synthetic_bias_typeb_sweep.yaml`).

The synthetic envs have CLEAN structural Var_a[Q*] = (rvs × sp)²
by construction. Bridges here test whether DDQN's outcome-effect
direction tracks this controlled feature, addressing the n=1
limitation of the natural-env Type-A/B panel.

Pre-registered predictions (P1 + P4 enforced here; P2/P3 require
σ_Λ_a-per-env which is a sweep-output observable we'll add a
companion Finding for once cells materialise; N1 sparsity-null
will be tested when data is in)."""
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


# Synthetic env naming convention:
#   "TypeBChain-K{K}-rvs{rvs}-sp{sp}-ns{ns}-synthetic"
# Hardcoded covariates per env, mirroring chain_depth's
# `_LOG_HORIZON_PER_GAMMA` pattern. These are STRUCTURAL —
# derivable from the env name itself — not data-fitted.
def _build_synthetic_covariates() -> MappingProxyType[
    object, MappingProxyType[str, float]
]:
    """Pair (rvs, sparsity, noise) per env name. The values
    encode the env's STRUCTURAL design parameters (set at
    registration time in `env_catalogue._register_synthetic_bias_typeb_panel`),
    NOT empirical observables. Hardcoding is appropriate
    because the env IS its structural config — there's no
    drift between cache state and these values."""
    out: dict[object, MappingProxyType[str, float]] = {}
    for rvs in (0.2, 1.0, 3.0):
        for sp in (0.2, 1.0):
            for ns in (0.1, 0.5):
                name = (
                    f"TypeBChain-K4-rvs{rvs}-sp{sp}-ns{ns}-synthetic"
                )
                # True Var_a[Q*] per Cor 3.2-substrate is ∝
                # (rvs × sp)²; the structural cross-action SD
                # is the natural covariate.
                struct_sigma = float(rvs * sp)
                out[name] = MappingProxyType({
                    'reward_variance_scale': float(rvs),
                    'reward_sparsity': float(sp),
                    'reward_noise_scale': float(ns),
                    'structural_sigma_a': struct_sigma,
                })
    return MappingProxyType(out)


_SYNTHETIC_COVARIATES: MappingProxyType[
    object, MappingProxyType[str, float]
] = _build_synthetic_covariates()


# Scope: the 12 synthetic envs at γ=0.99 canonical (sweep default).
_SYNTHETIC_TYPEB_SCOPE: pl.Expr = (
    pl.col('env_name').str.starts_with('TypeBChain-K4-')
    & pl.col('env_name').str.ends_with('-synthetic')
    & (pl.col('gamma') == 0.99)
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

    Local copy with `min_strata` defaulting to 10 (matching the
    canonical helper's calibration); at n_strata=12 here the
    panel comfortably clears that gate."""
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


# P1 — DDQN harms under high reward_variance_scale.
@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_raw_mean',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=(
        _SYNTHETIC_TYPEB_SCOPE
        & pl.col('eval_best_burst_raw_mean').is_finite()
    ),
    predicted_direction='a_lt_b',
)
def ddqn_harms_under_high_rvs__synthetic_typeb(
    cross_stratum_property_slope: CrossStratumPropertySlopeResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    source: str = 'eval_best_burst_raw_mean',
    stratify_by: tuple[str, ...] = ('env_name',),
    covariate_name: str = 'reward_variance_scale',
    covariate_key_field: str = 'env_name',
    covariates_per_key: MappingProxyType[
        object, MappingProxyType[str, float]
    ] = _SYNTHETIC_COVARIATES,
    scope_predictor: str = 'eval_best_burst_raw_mean',
    min_baseline_predictor: float = -1e9,  # no gate; synthetic envs are always "active"
    min_seeds_per_arm: int = 5,
    rho_threshold_held: float = 0.5,
    p_threshold: float = 0.05,
    null_threshold: float = 0.2,
    sign_flip_threshold: float = 0.5,
    min_strata: int = 10,
) -> tuple[Verdict, RefutationClass | None]:
    """P1 — Cross-env Spearman ρ between reward_variance_scale
    (the env-structural cross-action Q-mean spread) and DDQN-vs-
    vanilla Cohen's d on outcome.

    Predicted ρ ≤ −0.5 with p ≤ 0.05 (cross-action-variance
    inflates Var_a[Q*]; on high-rvs envs, vanilla's bias
    asymmetry is policy-informative; DDQN's symmetric clip
    destroys it → outcome harm). The canonical bias Type-B
    test on controlled substrate.

    HELD: ρ ≤ −0.5 with p ≤ 0.05 — bias Type-A/B framework-typed
    cross-env, n=12 controlled synthetic envs.
    NO_EFFECT (SIGN_FLIP): ρ ≥ +0.5 — DDQN HELPS more on
    high-rvs envs (would refute the central interpretation).
    NO_EFFECT (NULL_EFFECT): |ρ| < 0.2 — rvs doesn't drive
    DDQN's outcome direction (controlled substrate's null on
    the Var_a[Q*] mediator)."""
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


# N1 — Sparsity ALONE does not predict d_out.
@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_raw_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=(
        _SYNTHETIC_TYPEB_SCOPE
        & pl.col('eval_best_burst_raw_mean').is_finite()
    ),
    predicted_direction='null',
)
def sparsity_alone_does_not_predict_dout__synthetic_typeb(
    cross_stratum_property_slope: CrossStratumPropertySlopeResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    source: str = 'eval_best_burst_raw_mean',
    stratify_by: tuple[str, ...] = ('env_name',),
    covariate_name: str = 'reward_sparsity',
    covariate_key_field: str = 'env_name',
    covariates_per_key: MappingProxyType[
        object, MappingProxyType[str, float]
    ] = _SYNTHETIC_COVARIATES,
    scope_predictor: str = 'eval_best_burst_raw_mean',
    min_baseline_predictor: float = -1e9,
    min_seeds_per_arm: int = 5,
    null_threshold: float = 0.2,
    effect_observed_threshold: float = 0.5,
    min_strata: int = 10,
) -> tuple[Verdict, RefutationClass | None]:
    """N1 — Cross-env Spearman ρ(reward_sparsity, d_out) on the
    synthetic Type-B panel.

    Predicted null: sparsity's effect on d_out is ≤ |0.2|
    cross-env. The substantive interpretation is that
    `reward_variance_scale` (P1) is the load-bearing driver of
    DDQN's effect direction; sparsity is a modulator of the
    magnitude of Var_a[Q*] but not the direction of DDQN's
    effect at fixed rvs.

    HELD when |ρ| ≤ null_threshold (sparsity-as-sole-feature
    doesn't classify d_out). NO_EFFECT when |ρ| ≥ effect_observed
    — the null prediction fails, sparsity DOES have substantial
    independent effect (would walk back the "Var_a[Q*] is
    sufficient" reading)."""
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
    ddqn_harms_under_high_rvs__synthetic_typeb,
    sparsity_alone_does_not_predict_dout__synthetic_typeb,
)
