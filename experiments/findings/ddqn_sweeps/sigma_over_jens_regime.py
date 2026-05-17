"""σ_VAN / jens_VAN at γ=0.999 as a regime discriminator for DDQN's
outcome sign.

Theory (`findings_sigma_over_jens_regime_discriminator.md`): DDQN's
clip is a precision trade-off whose net sign depends on whether
vanilla's bias is uniform-across-actions (Type A) or asymmetric
across-actions (Type B). The σ_VAN / jens_VAN ratio measures
"action-discrimination per unit overestimation":

  - Low σ/jens (≤ 0.02) → bias scale ≫ action variance → uniform
    overestimation → Type A → DDQN's clip introduces argmax noise
    amplified by 1/(1−γ) → outcome HARM (when env is learnable).
  - High σ/jens (≥ 0.05) → action variance comparable to bias →
    asymmetric overestimation → Type B → DDQN's clip restores
    representation budget → outcome HELP.

This module encodes three bridges that operationalize the theory:

1. `ddqn_outcome_scales_with_sigma_over_jens__gamma_999_xenv`
   — cross-env Spearman ρ between σ/jens (per env) and per-env
   Cohen's d on outcome. Predicts ρ > 0 across envs.

2. `ddqn_harms_asterix_gamma_999` — single-env CI test on Asterix
   γ=0.999 (Type A + learnable). Predicts d_out CI fully below 0.

3. `ddqn_helps_breakout_gamma_999` — single-env CI test on Breakout
   γ=0.999 (Type B / FA-truncation collapse). Predicts d_out CI
   fully above 0.

Findings layer aggregates them per `HYPOTHESIS_AS_GRAPH.md`.

σ/jens values per env at γ=0.999, vanilla arm, from the canonical
ddqn cache + ddqn_three_conditions cache snapshot (2026-05-17):

  env             | σ_VAN  | jens_VAN | σ/jens  | regime predicted
  Asterix-MinAtar |  4.338 |  280.35  | 0.0155  | A (uniform)
  Breakout-MinAtar|  2.068 |   33.97  | 0.0609  | B / mixed
  Acrobot-v1      |  0.279 |   74.96  | 0.0037  | A
  MountainCar-v0  |  0.141 |   98.49  | 0.0014  | A
  FourRooms-misc  |  0.146 |   27.78  | 0.0052  | A
  MetaMaze-misc   |  0.179 |   10.77  | 0.0166  | A

Freeway-MinAtar and SpaceInvaders-MinAtar at γ=0.999 are pending
(running `minatar_gamma_sweep_k1` blocks). When they land, the
covariate dict here gets extended and the cross-env bridge re-fires
with denser strata.
"""
from __future__ import annotations

import math
from types import MappingProxyType

import polars as pl

from corroborate.analyses.cross_stratum_property_slope import (
    CrossStratumPropertySlopeResult,
)
from corroborate.analyses.stratified_arm_diff_pooled import (
    StratifiedArmDiffPooledResult,
)
from corroborate.bridge.bridge import Direction, Tier, claim_bridge
from corroborate.bridge.verdict import RefutationClass, Verdict

from experiments.findings.ddqn._arms import (
    DDQN_ARM, INTERVENTION, VANILLA_ARM,
)
from experiments.findings.ddqn._verdicts import (
    cross_stratum_signed_spearman_verdict,
)


# Per-env σ_VAN / jens_VAN measured on γ=0.999 vanilla cells (k=1,
# canonical-shape HPs). Snapshot 2026-05-17. Envs without γ=0.999
# data yet (Freeway, SpaceInvaders, Snake, PacMan, SlidingTile,
# CartPole-saturated) are absent here; the analysis primitive's
# `covariates_per_key` lookup drops missing keys.
_SIGMA_OVER_JENS_PER_ENV: MappingProxyType[object, MappingProxyType[str, float]] = (
    MappingProxyType({
        'Asterix-MinAtar':  MappingProxyType({'sigma_over_jens': 0.0155}),
        'Breakout-MinAtar': MappingProxyType({'sigma_over_jens': 0.0609}),
        'Acrobot-v1':       MappingProxyType({'sigma_over_jens': 0.0037}),
        'MountainCar-v0':   MappingProxyType({'sigma_over_jens': 0.0014}),
        'FourRooms-misc':   MappingProxyType({'sigma_over_jens': 0.0052}),
        'MetaMaze-misc':    MappingProxyType({'sigma_over_jens': 0.0166}),
    })
)


# Bridge 1 — Cross-env discriminator (the universal claim).
@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_raw_mean',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=(
        (pl.col('gamma') == 0.999)
        & pl.col('eval_best_burst_raw_mean').is_finite()
        & pl.col('jensen_gap').is_finite()
        & ((pl.col('n_step') == 1) | pl.col('n_step').is_null())
        & pl.col('action_duplicate_k').is_null()
        & (pl.col('wrappers') == '()')
        & pl.col('env_name').is_in(tuple(_SIGMA_OVER_JENS_PER_ENV.keys()))
    ),
    predicted_direction='a_gt_b',
)
def ddqn_outcome_scales_with_sigma_over_jens__gamma_999_xenv(
    cross_stratum_property_slope: CrossStratumPropertySlopeResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    source: str = 'eval_best_burst_raw_mean',
    stratify_by: tuple[str, ...] = ('env_name',),
    covariate_name: str = 'sigma_over_jens',
    covariate_key_field: str = 'env_name',
    covariates_per_key: MappingProxyType[object, MappingProxyType[str, float]] = (
        _SIGMA_OVER_JENS_PER_ENV
    ),
    scope_predictor: str = 'jensen_gap',
    min_baseline_predictor: float = 0.5,
    min_seeds_per_arm: int = 5,
    rho_threshold_held: float = 0.5,
    p_threshold: float = 0.10,
    null_threshold: float = 0.2,
    sign_flip_threshold: float = 0.5,
    min_strata: int = 6,
) -> tuple[Verdict, RefutationClass | None]:
    """Cross-env Spearman ρ between σ_VAN/jens_VAN (per env) and
    per-env Cohen's d on raw outcome.

    **Predicted direction**: ρ > 0 — higher σ/jens (Type B,
    asymmetric overestimation) → larger DDQN outcome benefit;
    lower σ/jens (Type A, uniform) → smaller or negative benefit.

    **Calibrated for n_strata=6** (current cache; k=1 γ=0.999 has
    Asterix + Breakout, plus pre-existing γ=0.999 cells for the
    4 vector envs). Thresholds relaxed vs canonical chain-depth
    bridge: rho_threshold_held=0.5 (sign expectation only),
    p_threshold=0.10 (one-sided since direction is predicted).

    Once Freeway and SpaceInvaders k=1 γ=0.999 land, n_strata=8 →
    rerun fires with denser data."""
    del treatment_arm, baseline_arm, source, stratify_by
    del covariate_name, covariate_key_field, covariates_per_key
    del scope_predictor, min_baseline_predictor, min_seeds_per_arm
    return cross_stratum_signed_spearman_verdict(
        cross_stratum_property_slope,
        sign=1,
        rho_threshold_held=rho_threshold_held,
        p_threshold=p_threshold,
        null_threshold=null_threshold,
        sign_flip_threshold=sign_flip_threshold,
        min_strata=min_strata,
    )


# Bridge 2 — Per-env: Asterix γ=0.999 d_out CI fully below 0.
@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_raw_mean',
    direction=Direction.INVERSE,
    tier=Tier.INTERVENTIONAL,
    scope=(
        (pl.col('env_name') == 'Asterix-MinAtar')
        & (pl.col('gamma') == 0.999)
        & pl.col('eval_best_burst_raw_mean').is_finite()
        & pl.col('jensen_gap').is_finite()
        & ((pl.col('n_step') == 1) | pl.col('n_step').is_null())
        & pl.col('action_duplicate_k').is_null()
        & (pl.col('wrappers') == '()')
    ),
    predicted_direction='a_lt_b',
)
def ddqn_harms_asterix_gamma_999(
    stratified_arm_diff_pooled: StratifiedArmDiffPooledResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    source: str = 'eval_best_burst_raw_mean',
    stratify_by: tuple[str, ...] = ('env_name',),
    scope_predictor: str = 'jensen_gap',
    min_baseline_predictor: float = 50.0,
    min_seeds_per_arm: int = 5,
    harm_floor: float = -0.4,
    min_strata: int = 1,
) -> tuple[Verdict, RefutationClass | None]:
    """Asterix γ=0.999 d_out CI should be FULLY BELOW
    `harm_floor` (default −0.4). Predicts the Type-A + learnable
    case: DDQN's clip corrupts argmax at extreme γ → outcome harm.

    HELD if d_env CI_high ≤ harm_floor (d significantly < -0.4).
    NO_EFFECT (SIGN_FLIP) if d_env CI_low ≥ +harm_floor (DDQN
    helps instead). POWER_INSUFFICIENT otherwise."""
    del treatment_arm, baseline_arm, source, stratify_by
    del scope_predictor, min_baseline_predictor, min_seeds_per_arm
    if stratified_arm_diff_pooled.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT, None
    for s in stratified_arm_diff_pooled.per_stratum:
        d, se = s.cohen_d, s.cohen_se
        if math.isnan(d) or math.isnan(se):
            continue
        ci_lo = d - 1.96 * se
        ci_hi = d + 1.96 * se
        if ci_hi <= harm_floor:
            return Verdict.HELD, None
        if ci_lo >= -harm_floor:
            return Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP
        # CI spans the harm_floor → underpowered
    return Verdict.POWER_INSUFFICIENT, None


# Bridge 3 — Per-env: Breakout γ=0.999 d_out CI fully above 0.
@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_raw_mean',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=(
        (pl.col('env_name') == 'Breakout-MinAtar')
        & (pl.col('gamma') == 0.999)
        & pl.col('eval_best_burst_raw_mean').is_finite()
        & pl.col('jensen_gap').is_finite()
        & ((pl.col('n_step') == 1) | pl.col('n_step').is_null())
        & pl.col('action_duplicate_k').is_null()
        & (pl.col('wrappers') == '()')
    ),
    predicted_direction='a_gt_b',
)
def ddqn_helps_breakout_gamma_999(
    stratified_arm_diff_pooled: StratifiedArmDiffPooledResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    source: str = 'eval_best_burst_raw_mean',
    stratify_by: tuple[str, ...] = ('env_name',),
    scope_predictor: str = 'jensen_gap',
    min_baseline_predictor: float = 5.0,
    min_seeds_per_arm: int = 5,
    help_floor: float = 0.4,
    min_strata: int = 1,
) -> tuple[Verdict, RefutationClass | None]:
    """Breakout γ=0.999 d_out CI should be FULLY ABOVE `help_floor`
    (default +0.4). Predicts the Type-B / FA-truncation case:
    vanilla collapses (FA can't represent precision at long
    horizon), DDQN rescues.

    HELD if d_env CI_low ≥ help_floor. NO_EFFECT (SIGN_FLIP) if
    CI_high ≤ -help_floor. POWER_INSUFFICIENT otherwise."""
    del treatment_arm, baseline_arm, source, stratify_by
    del scope_predictor, min_baseline_predictor, min_seeds_per_arm
    if stratified_arm_diff_pooled.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT, None
    for s in stratified_arm_diff_pooled.per_stratum:
        d, se = s.cohen_d, s.cohen_se
        if math.isnan(d) or math.isnan(se):
            continue
        ci_lo = d - 1.96 * se
        ci_hi = d + 1.96 * se
        if ci_lo >= help_floor:
            return Verdict.HELD, None
        if ci_hi <= -help_floor:
            return Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP
    return Verdict.POWER_INSUFFICIENT, None


BRIDGES = (
    ddqn_outcome_scales_with_sigma_over_jens__gamma_999_xenv,
    ddqn_harms_asterix_gamma_999,
    ddqn_helps_breakout_gamma_999,
)
