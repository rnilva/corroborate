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

from corroborate.analyses.link.cross_stratum_property_slope import (
    CrossStratumPropertySlopeResult,
)
from corroborate.analyses.panel.stratified_arm_diff_pooled import (
    StratifiedArmDiffPooledResult,
)
from corroborate.bridge.bridge import Direction, Tier, claim_bridge
from corroborate.bridge.verdict import RefutationClass, Verdict

from experiments.findings.ddqn._arms import (
    DDQN_ARM, INTERVENTION, VANILLA_ARM,
)
from experiments.findings.ddqn._scope import CANONICAL_HP_EXCLUDING_GAMMA
from experiments.findings.ddqn._verdicts import (
    cross_stratum_signed_spearman_verdict,
)


# File-level scope shared by every bridge in this module: γ=0.999
# + canonical-shape HPs + Q-MC coupled. See
# `clip_argmax_harm_mechanism._GAMMA_999_LEARNABLE_CANONICAL_SCOPE`
# for the rationale (this is identical to that constant — could
# be hoisted to a shared file if a third module wants it).
_GAMMA_999_LEARNABLE_CANONICAL_SCOPE: pl.Expr = (
    (pl.col('gamma') == 0.999)
    & CANONICAL_HP_EXCLUDING_GAMMA
    & pl.col('q_mc_burst_correlation_late').is_finite()
    & (pl.col('q_mc_burst_correlation_late') >= 0.3)
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
        _GAMMA_999_LEARNABLE_CANONICAL_SCOPE
        & pl.col('eval_best_burst_raw_mean').is_finite()
        & pl.col('jensen_gap').is_finite()
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
    min_strata: int = 5,
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
        & _GAMMA_999_LEARNABLE_CANONICAL_SCOPE
        & pl.col('eval_best_burst_raw_mean').is_finite()
        & pl.col('jensen_gap').is_finite()
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
    return _ci_test(stratified_arm_diff_pooled, floor=harm_floor,
                    direction=-1, min_strata=min_strata)


# Bridge 3 — Per-env: Breakout γ=0.999 d_out CI fully above 0.
@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_raw_mean',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=(
        (pl.col('env_name') == 'Breakout-MinAtar')
        & _GAMMA_999_LEARNABLE_CANONICAL_SCOPE
        & pl.col('eval_best_burst_raw_mean').is_finite()
        & pl.col('jensen_gap').is_finite()
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
    return _ci_test(stratified_arm_diff_pooled, floor=help_floor,
                    direction=1, min_strata=min_strata)


def _ci_test(
    pooled: StratifiedArmDiffPooledResult,
    *,
    floor: float,
    direction: int,
    min_strata: int = 1,
) -> tuple[Verdict, RefutationClass | None]:
    """Per-env CI gate. `direction=+1` ↔ help (CI_low ≥ +floor);
    `direction=−1` ↔ harm (CI_high ≤ −|floor|). Mirror floor used
    for sign-flip detection."""
    if pooled.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT, None
    abs_floor = abs(floor)
    for s in pooled.per_stratum:
        d, se = s.cohen_d, s.cohen_se
        if math.isnan(d) or math.isnan(se):
            continue
        ci_lo = d - 1.96 * se
        ci_hi = d + 1.96 * se
        if direction > 0:
            if ci_lo >= abs_floor:
                return Verdict.HELD, None
            if ci_hi <= -abs_floor:
                return Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP
        else:
            if ci_hi <= -abs_floor:
                return Verdict.HELD, None
            if ci_lo >= abs_floor:
                return Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP
    return Verdict.POWER_INSUFFICIENT, None


# ============ Per-burst outcome variants ============
#
# The earlier bridges (1, 2, 3) use `eval_best_burst_raw_mean` —
# the PEAK burst raw return. Two additional outcome shapes encode
# different aspects of the γ=0.999 dynamic:
#
# - `eval_late_burst_raw_mean`: mean of the LAST 25% of bursts.
#   Tests "the mechanism bites where Q has grown most" — at
#   γ=0.999 Q grows monotonically across training, so late-burst
#   captures the peak-mechanism state. Asterix d_out sharpens
#   from -0.80 (best) to -1.07 (late).
#
# - `eval_full_auc_raw_mean`: trajectory-averaged outcome (whole
#   training). Tests the cumulative effect; less sensitive to
#   timing artifacts.


# Bridge 4 — cross-env late-burst.
@claim_bridge(
    source=INTERVENTION,
    target='eval_late_burst_raw_mean',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=(
        _GAMMA_999_LEARNABLE_CANONICAL_SCOPE
        & pl.col('eval_late_burst_raw_mean').is_finite()
        & pl.col('jensen_gap').is_finite()
        & pl.col('env_name').is_in(tuple(_SIGMA_OVER_JENS_PER_ENV.keys()))
    ),
    predicted_direction='a_gt_b',
)
def ddqn_outcome_scales_with_sigma_over_jens__gamma_999_xenv__late_burst(
    cross_stratum_property_slope: CrossStratumPropertySlopeResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    source: str = 'eval_late_burst_raw_mean',
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
    min_strata: int = 5,
) -> tuple[Verdict, RefutationClass | None]:
    """Late-burst variant of bridge 1. Tests the same cross-env
    Spearman ρ on `eval_late_burst_raw_mean` — where Q-explosion
    has progressed most. Predicted ρ stronger than best-burst form
    (Asterix harm sharpens -0.80 → -1.07 between metrics)."""
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


# Bridge 5 — cross-env full-AUC.
@claim_bridge(
    source=INTERVENTION,
    target='eval_full_auc_raw_mean',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=(
        _GAMMA_999_LEARNABLE_CANONICAL_SCOPE
        & pl.col('eval_full_auc_raw_mean').is_finite()
        & pl.col('jensen_gap').is_finite()
        & pl.col('env_name').is_in(tuple(_SIGMA_OVER_JENS_PER_ENV.keys()))
    ),
    predicted_direction='a_gt_b',
)
def ddqn_outcome_scales_with_sigma_over_jens__gamma_999_xenv__full_auc(
    cross_stratum_property_slope: CrossStratumPropertySlopeResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    source: str = 'eval_full_auc_raw_mean',
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
    min_strata: int = 5,
) -> tuple[Verdict, RefutationClass | None]:
    """Full-AUC variant of bridge 1. Trajectory-averaged outcome.
    Less sensitive to timing artifacts than best-burst."""
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


# Bridge 6 — Asterix late-burst harm.
@claim_bridge(
    source=INTERVENTION,
    target='eval_late_burst_raw_mean',
    direction=Direction.INVERSE,
    tier=Tier.INTERVENTIONAL,
    scope=(
        (pl.col('env_name') == 'Asterix-MinAtar')
        & _GAMMA_999_LEARNABLE_CANONICAL_SCOPE
        & pl.col('eval_late_burst_raw_mean').is_finite()
        & pl.col('jensen_gap').is_finite()
    ),
    predicted_direction='a_lt_b',
)
def ddqn_harms_asterix_gamma_999__late_burst(
    stratified_arm_diff_pooled: StratifiedArmDiffPooledResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    source: str = 'eval_late_burst_raw_mean',
    stratify_by: tuple[str, ...] = ('env_name',),
    scope_predictor: str = 'jensen_gap',
    min_baseline_predictor: float = 50.0,
    min_seeds_per_arm: int = 5,
    harm_floor: float = -0.5,
    min_strata: int = 1,
) -> tuple[Verdict, RefutationClass | None]:
    """Late-burst variant of Asterix harm bridge. Asterix d_out
    sharpens to −1.07 in late-burst (vs −0.80 best-burst), so the
    floor is set tighter at −0.5."""
    del treatment_arm, baseline_arm, source, stratify_by
    del scope_predictor, min_baseline_predictor, min_seeds_per_arm
    return _ci_test(stratified_arm_diff_pooled, floor=harm_floor,
                    direction=-1, min_strata=min_strata)


# Bridge 7 — Asterix full-AUC harm.
@claim_bridge(
    source=INTERVENTION,
    target='eval_full_auc_raw_mean',
    direction=Direction.INVERSE,
    tier=Tier.INTERVENTIONAL,
    scope=(
        (pl.col('env_name') == 'Asterix-MinAtar')
        & _GAMMA_999_LEARNABLE_CANONICAL_SCOPE
        & pl.col('eval_full_auc_raw_mean').is_finite()
        & pl.col('jensen_gap').is_finite()
    ),
    predicted_direction='a_lt_b',
)
def ddqn_harms_asterix_gamma_999__full_auc(
    stratified_arm_diff_pooled: StratifiedArmDiffPooledResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    source: str = 'eval_full_auc_raw_mean',
    stratify_by: tuple[str, ...] = ('env_name',),
    scope_predictor: str = 'jensen_gap',
    min_baseline_predictor: float = 50.0,
    min_seeds_per_arm: int = 5,
    harm_floor: float = -0.5,
    min_strata: int = 1,
) -> tuple[Verdict, RefutationClass | None]:
    """Full-AUC variant of Asterix harm bridge. Asterix d_out
    is −1.08 across the AUC. floor = −0.5."""
    del treatment_arm, baseline_arm, source, stratify_by
    del scope_predictor, min_baseline_predictor, min_seeds_per_arm
    return _ci_test(stratified_arm_diff_pooled, floor=harm_floor,
                    direction=-1, min_strata=min_strata)


# Bridge 8 — Breakout late-burst help.
@claim_bridge(
    source=INTERVENTION,
    target='eval_late_burst_raw_mean',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=(
        (pl.col('env_name') == 'Breakout-MinAtar')
        & _GAMMA_999_LEARNABLE_CANONICAL_SCOPE
        & pl.col('eval_late_burst_raw_mean').is_finite()
        & pl.col('jensen_gap').is_finite()
    ),
    predicted_direction='a_gt_b',
)
def ddqn_helps_breakout_gamma_999__late_burst(
    stratified_arm_diff_pooled: StratifiedArmDiffPooledResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    source: str = 'eval_late_burst_raw_mean',
    stratify_by: tuple[str, ...] = ('env_name',),
    scope_predictor: str = 'jensen_gap',
    min_baseline_predictor: float = 5.0,
    min_seeds_per_arm: int = 5,
    help_floor: float = 0.4,
    min_strata: int = 1,
) -> tuple[Verdict, RefutationClass | None]:
    """Late-burst variant of Breakout help bridge. d_out = +0.67
    in late-burst — close to best-burst's +0.66."""
    del treatment_arm, baseline_arm, source, stratify_by
    del scope_predictor, min_baseline_predictor, min_seeds_per_arm
    return _ci_test(stratified_arm_diff_pooled, floor=help_floor,
                    direction=1, min_strata=min_strata)


# Bridge 9 — Breakout full-AUC help.
@claim_bridge(
    source=INTERVENTION,
    target='eval_full_auc_raw_mean',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=(
        (pl.col('env_name') == 'Breakout-MinAtar')
        & _GAMMA_999_LEARNABLE_CANONICAL_SCOPE
        & pl.col('eval_full_auc_raw_mean').is_finite()
        & pl.col('jensen_gap').is_finite()
    ),
    predicted_direction='a_gt_b',
)
def ddqn_helps_breakout_gamma_999__full_auc(
    stratified_arm_diff_pooled: StratifiedArmDiffPooledResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    source: str = 'eval_full_auc_raw_mean',
    stratify_by: tuple[str, ...] = ('env_name',),
    scope_predictor: str = 'jensen_gap',
    min_baseline_predictor: float = 5.0,
    min_seeds_per_arm: int = 5,
    help_floor: float = 0.3,
    min_strata: int = 1,
) -> tuple[Verdict, RefutationClass | None]:
    """Full-AUC variant of Breakout help bridge. d_out = +0.42
    in full-AUC — weakest of Breakout's outcome metrics; floor
    set lower at +0.3."""
    del treatment_arm, baseline_arm, source, stratify_by
    del scope_predictor, min_baseline_predictor, min_seeds_per_arm
    return _ci_test(stratified_arm_diff_pooled, floor=help_floor,
                    direction=1, min_strata=min_strata)


BRIDGES = (
    ddqn_outcome_scales_with_sigma_over_jens__gamma_999_xenv,
    ddqn_outcome_scales_with_sigma_over_jens__gamma_999_xenv__late_burst,
    ddqn_outcome_scales_with_sigma_over_jens__gamma_999_xenv__full_auc,
    ddqn_harms_asterix_gamma_999,
    ddqn_harms_asterix_gamma_999__late_burst,
    ddqn_harms_asterix_gamma_999__full_auc,
    ddqn_helps_breakout_gamma_999,
    ddqn_helps_breakout_gamma_999__late_burst,
    ddqn_helps_breakout_gamma_999__full_auc,
)
