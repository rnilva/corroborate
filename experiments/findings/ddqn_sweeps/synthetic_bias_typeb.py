"""Synthetic bias Type-A/B controlled-substrate bridges (v3).

v1 → v2 → v3 evolution lives in
`src/corroborate_rl/corroborate_rl/synthetic_bias_typeb.py`. v1
was scrapped (action-independent transitions; bandit in a tuxedo);
v2 was scrapped (per-step reward-noise α conflated with the
Q-target-side Var_a[V*(s')] that Cor 3.2's σ_clip actually
concerns; under-powered n_seeds=12; over-parameterized FA at
L=64). v3 replaces the env entirely. See `/tmp/synthetic_v2_roast.md`
for the v2 critique and the five recommendations v3 addresses.

The v3 panel has TWO structural axes:

- L = n_states ∈ {32, 1024} (FA-capacity axis; with hidden=[16]
  the L=1024 corner aliases 4096 Q-values into 16-dim hidden →
  genuine FA-binding).
- β = beta ∈ {0.0, 0.5, 0.9} (Type-A/B axis on the Q-target side;
  per-block payoff shape `(peak, peak·β, peak·β², peak·β³)`).

Plus γ ∈ {0.95, 0.99, 0.999} as a substrate axis. 6 envs × 3 γ ×
2 arms × 27 seeds = 972 cells (≤ 1000 budget).

The substantive predictions are pre-registered in
`docs/PRE_REGISTRATION_synthetic_bias_typeb.md`. The verdict
helper is the shared substrate primitive
`cross_stratum_signed_spearman_verdict` (calibrated for
n_strata≥10).
"""
from __future__ import annotations

import math
from types import MappingProxyType

import polars as pl

from corroborate.analyses.link.cross_stratum_property_slope import (
    CrossStratumPropertySlopeResult,
)
from corroborate.bridge.bridge import Direction, Tier, claim_bridge
from corroborate.bridge.verdict import RefutationClass, Verdict

from experiments.findings.ddqn._arms import (
    DDQN_ARM, INTERVENTION, VANILLA_ARM,
)
from experiments.findings.ddqn._verdicts import (
    cross_stratum_signed_spearman_verdict,
)


# v3 synthetic env naming convention:
#   "TypeBChainV3-K{K}-L{n_states}-beta{beta}-synthetic"
# γ is a substrate axis (set via YAML), not baked into the env
# name. The (env_name, gamma) pair identifies a cell stratum.
#
# Hardcoded structural covariates mirror chain_depth's
# `_LOG_HORIZON_PER_GAMMA` pattern. These are STRUCTURAL —
# derivable from the env name itself — not data-fitted.
def _build_synthetic_covariates() -> MappingProxyType[
    object, MappingProxyType[str, float]
]:
    """Per-env structural covariates for the v3 panel.

    Keys are env names; values are immutable dicts of the env's
    STRUCTURAL design parameters baked at registration time in
    `env_catalogue._register_synthetic_bias_typeb_panel`. The
    covariates exposed here are:

    - `beta`: the Type-A/B axis (per-block payoff geometric ratio).
    - `n_states`: chain length L, the FA-capacity axis.
    - `argmax_margin`: closed-form best-vs-second-best margin
      `peak_value × (1 - β)`. At peak_value=1.0 this is
      monotone-decreasing in β; the inverse of the Type-A/B knob.
    - `log_n_states`: `log(L)`, the FA-capacity axis on a scale
      that linearizes the bottleneck-pressure intuition.
    """
    out: dict[object, MappingProxyType[str, float]] = {}
    peak_value = 1.0  # pinned across v3 panel
    for n_states in (32, 1024):
        for beta in (0.0, 0.5, 0.9):
            name = (
                f"TypeBChainV3-K4-L{n_states}"
                f"-beta{beta}-synthetic"
            )
            out[name] = MappingProxyType({
                'beta': float(beta),
                'n_states': float(n_states),
                'argmax_margin': peak_value * (1.0 - float(beta)),
                'log_n_states': math.log(float(n_states)),
            })
    return MappingProxyType(out)


_SYNTHETIC_COVARIATES: MappingProxyType[
    object, MappingProxyType[str, float]
] = _build_synthetic_covariates()


# Scope: v3 synthetic envs. γ NOT pinned here because the sweep
# spans γ ∈ {0.95, 0.99, 0.999} and individual bridges decide
# whether to pin γ or pool.
_SYNTHETIC_TYPEB_V3_SCOPE: pl.Expr = (
    pl.col('env_name').str.starts_with('TypeBChainV3-K4-')
    & pl.col('env_name').str.ends_with('-synthetic')
)

# γ=0.999 sub-scope for the "gamma amplifies Type-B" diagnostic.
_SYNTHETIC_TYPEB_V3_G999_SCOPE: pl.Expr = (
    _SYNTHETIC_TYPEB_V3_SCOPE & (pl.col('gamma') == 0.999)
)

# β=0 sub-scope for the FA-capacity-alone-doesn't-drive-d_out
# null prediction (N1).
_SYNTHETIC_TYPEB_V3_BETA0_SCOPE: pl.Expr = (
    _SYNTHETIC_TYPEB_V3_SCOPE
    & pl.col('env_name').str.contains('beta0.0-synthetic')
)


# ============ PRIMARY prediction (P1) ============
#
# P1 — DDQN's outcome benefit decreases as β grows (Type-A → Type-B).
# Cross-stratum Spearman ρ between `beta` (the env-structural
# Type-A/B axis) and DDQN-vs-vanilla Cohen's d on outcome,
# stratified by (env_name, gamma). 6 envs × 3 γ = 18 strata; well
# above min_strata=10 calibration of the verdict helper.
#
# This is the LOAD-BEARING bridge — the v3 design's primary
# substantive prediction. The PRE_REGISTRATION doc defines its
# REFUTATION criterion (a specific data shape that retracts the
# claim that synthetic substrate enables causal env-feature
# identification of the Asterix Type-B mechanism).
@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_raw_mean',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=(
        _SYNTHETIC_TYPEB_V3_SCOPE
        & pl.col('eval_best_burst_raw_mean').is_finite()
    ),
    predicted_direction='a_lt_b',
)
def ddqn_harms_under_high_beta__synthetic_typeb_v3(
    cross_stratum_property_slope: CrossStratumPropertySlopeResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    source: str = 'eval_best_burst_raw_mean',
    stratify_by: tuple[str, ...] = ('env_name', 'gamma'),
    covariate_name: str = 'beta',
    covariate_key_field: str = 'env_name',
    covariates_per_key: MappingProxyType[
        object, MappingProxyType[str, float]
    ] = _SYNTHETIC_COVARIATES,
    scope_predictor: str = 'eval_best_burst_raw_mean',
    min_baseline_predictor: float = -1e9,  # no gate; synthetic always active
    min_seeds_per_arm: int = 5,
    rho_threshold_held: float = 0.5,
    p_threshold: float = 0.05,
    null_threshold: float = 0.2,
    sign_flip_threshold: float = 0.5,
    min_strata: int = 10,
) -> tuple[Verdict, RefutationClass | None]:
    """P1 (PRIMARY) — Cross-stratum Spearman ρ between `beta` (the
    Q-target-side anisotropy primitive) and DDQN-vs-vanilla
    Cohen's d on outcome, stratified by (env_name, γ).

    **HELD criterion**: ρ ≤ −0.5 AND p ≤ 0.05 on n_strata ≥ 10.
    Direction: higher β → more graded payoff shape → smaller
    argmax-margin Δ_v = peak·(1-β) → knife-edge regime where
    DDQN's clip introduces argmax-corrupting asymmetry → DDQN
    HARMS more.

    **REFUTATION criterion** (the load-bearing pre-registered
    walk-back-as-retraction; see PRE_REGISTRATION doc §REFUTATION):
    NO_EFFECT-NULL fires (|ρ| < 0.2) AND the panel is adequately
    powered (n_strata ≥ 15). In that case the synthetic-substrate
    paradigm DOES NOT reproduce the natural-env Asterix Type-B
    mechanism. NOT a publishable walk-back; the substrate-author
    must retract the claim that v3 enables causal env-feature
    identification.

    **SIGN_FLIP** (ρ ≥ +0.5): DDQN HELPS more as β grows. Walks
    back the Cor 3.2 σ_clip → argmax-corruption mechanism: more
    graded payoffs make DDQN's clip MORE effective. Suggests
    distinct mechanism (e.g., FA-residual smoothness or replay-
    distribution coupling) is dominant in synthetic substrate."""
    del treatment_arm, baseline_arm, source, stratify_by
    del covariate_name, covariate_key_field, covariates_per_key
    del scope_predictor, min_baseline_predictor, min_seeds_per_arm
    return cross_stratum_signed_spearman_verdict(
        cross_stratum_property_slope,
        sign=-1,
        rho_threshold_held=rho_threshold_held,
        p_threshold=p_threshold,
        null_threshold=null_threshold,
        sign_flip_threshold=sign_flip_threshold,
        min_strata=min_strata,
    )


# ============ DIAGNOSTIC: argmax margin (inverse parameterization
# of P1; corroborates the mechanism interpretation) ============
#
# D1 — DDQN's outcome benefit INCREASES with argmax_margin. This
# is `peak_value · (1 - β)`, the closed-form knife-edge width.
# Predicting direct (a > b under HIGH argmax_margin) tests the
# SAME slope as P1 with the opposite sign convention — IF the
# mechanism is "DDQN's clip corrupts narrow argmax", D1 must HELD
# whenever P1 HELDs (and the rank correlation is exactly inverted
# since the covariate is monotone in -β).
#
# Why register both: D1 is a sanity check on the mechanism story.
# P1 measures the β knob; D1 measures the DOWNSTREAM knife-edge
# margin. Joint HELD is corroborating. P1 HELD + D1 NULL would
# indicate the β knob has a non-knife-edge mechanism (substantive
# finding).
@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_raw_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=(
        _SYNTHETIC_TYPEB_V3_SCOPE
        & pl.col('eval_best_burst_raw_mean').is_finite()
    ),
    predicted_direction='a_gt_b',
)
def ddqn_helps_when_argmax_margin_wide__synthetic_typeb_v3(
    cross_stratum_property_slope: CrossStratumPropertySlopeResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    source: str = 'eval_best_burst_raw_mean',
    stratify_by: tuple[str, ...] = ('env_name', 'gamma'),
    covariate_name: str = 'argmax_margin',
    covariate_key_field: str = 'env_name',
    covariates_per_key: MappingProxyType[
        object, MappingProxyType[str, float]
    ] = _SYNTHETIC_COVARIATES,
    scope_predictor: str = 'eval_best_burst_raw_mean',
    min_baseline_predictor: float = -1e9,
    min_seeds_per_arm: int = 5,
    rho_threshold_held: float = 0.5,
    p_threshold: float = 0.05,
    null_threshold: float = 0.2,
    sign_flip_threshold: float = 0.5,
    min_strata: int = 10,
) -> tuple[Verdict, RefutationClass | None]:
    """D1 (DIAGNOSTIC) — Cross-stratum Spearman ρ between
    `argmax_margin = peak_value · (1 - β)` and DDQN-vs-vanilla
    Cohen's d on outcome, stratified by (env_name, γ).

    **HELD criterion**: ρ ≥ +0.5 AND p ≤ 0.05 on n_strata ≥ 10.
    Direction: wider knife-edge margin → less argmax-corruption
    risk from DDQN's clip → DDQN's optimism-bias correction
    dominates → DDQN HELPS more.

    **Joint with P1**: P1 HELD + D1 HELD ⇒ the β → argmax-margin
    → DDQN-effect chain is corroborated. P1 HELD + D1 NULL ⇒ β
    matters but argmax-margin isn't the mediator (substantive
    open question for which mechanism dominates).

    Note: this is structurally equivalent to P1 with opposite
    sign because `argmax_margin = peak · (1 - β)` is monotone-
    decreasing in β. The ρ should be EXACTLY the negative of P1's
    ρ (rank-equivalent transform). The duplication serves as a
    redundancy check on the verdict-helper's sign handling."""
    del treatment_arm, baseline_arm, source, stratify_by
    del covariate_name, covariate_key_field, covariates_per_key
    del scope_predictor, min_baseline_predictor, min_seeds_per_arm
    return cross_stratum_signed_spearman_verdict(
        cross_stratum_property_slope,
        sign=+1,
        rho_threshold_held=rho_threshold_held,
        p_threshold=p_threshold,
        null_threshold=null_threshold,
        sign_flip_threshold=sign_flip_threshold,
        min_strata=min_strata,
    )


# ============ DIAGNOSTIC: γ amplification ============
#
# D2 — At γ=0.999 (the load-bearing chain-amplification regime
# where natural-env Asterix harm appears), ρ(β, d_out) should be
# MORE negative than the pooled P1. The 1/(1-γ)=1000× horizon
# magnifies the FA-residual contribution to Q* approximation
# error → knife-edge argmax-corruption from DDQN's clip is
# amplified.
#
# n_strata=6 at γ=0.999 alone is below the verdict helper's
# min_strata=10 calibration band → will fire POWER_INSUFFICIENT
# even under signal. Documented honestly; the diagnostic value
# is in the rank-ordering of effect sizes between γ levels (not
# a HELD verdict at this sub-scope). Surfaces direction
# informally; full P1 pool carries the formal verdict.
@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_raw_mean',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=(
        _SYNTHETIC_TYPEB_V3_G999_SCOPE
        & pl.col('eval_best_burst_raw_mean').is_finite()
    ),
    predicted_direction='a_lt_b',
)
def ddqn_harm_amplified_at_g999__synthetic_typeb_v3(
    cross_stratum_property_slope: CrossStratumPropertySlopeResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    source: str = 'eval_best_burst_raw_mean',
    stratify_by: tuple[str, ...] = ('env_name',),
    covariate_name: str = 'beta',
    covariate_key_field: str = 'env_name',
    covariates_per_key: MappingProxyType[
        object, MappingProxyType[str, float]
    ] = _SYNTHETIC_COVARIATES,
    scope_predictor: str = 'eval_best_burst_raw_mean',
    min_baseline_predictor: float = -1e9,
    min_seeds_per_arm: int = 5,
    rho_threshold_held: float = 0.6,
    p_threshold: float = 0.1,
    null_threshold: float = 0.2,
    sign_flip_threshold: float = 0.5,
    min_strata: int = 10,
) -> tuple[Verdict, RefutationClass | None]:
    """D2 (DIAGNOSTIC) — At γ=0.999, ρ(β, d_out) ≤ −0.6 with
    p ≤ 0.1 across n_strata = 6 envs. The same shape as the
    natural-env Asterix γ=0.999 harm finding (memory
    `findings_asterix_g999_harm_is_optimization_dynamics`).

    **STRUCTURAL POWER_INSUFFICIENT**: n_strata=6 < min_strata=10
    → this bridge fires POWER_INSUFFICIENT at the formal verdict
    layer. Its diagnostic value is in the OBSERVED ρ direction
    + magnitude (surfaced in the analysis result, not the
    verdict) compared to P1's pooled ρ. If observed |ρ_γ=0.999| >
    |ρ_pooled| with consistent sign, the γ-amplification
    direction is corroborated even when the formal verdict can't
    fire HELD at this sub-scope."""
    del treatment_arm, baseline_arm, source, stratify_by
    del covariate_name, covariate_key_field, covariates_per_key
    del scope_predictor, min_baseline_predictor, min_seeds_per_arm
    return cross_stratum_signed_spearman_verdict(
        cross_stratum_property_slope,
        sign=-1,
        rho_threshold_held=rho_threshold_held,
        p_threshold=p_threshold,
        null_threshold=null_threshold,
        sign_flip_threshold=sign_flip_threshold,
        min_strata=min_strata,
    )


# ============ N1 — adequately-powered FALSIFIABLE null ============
#
# N1 — FA-capacity ALONE doesn't drive d_out's sign at β=0
# (peaked Type-A). At Type-A, the argmax-margin is wide (Δ_v =
# peak_value), the σ/Δ regime is benign (2%), and the FA-binding
# only modulates HOW MUCH the bias correction helps — not whether
# it helps. L should not flip the sign of d_out.
#
# v2's N1 was unfalsifiable: min_strata=4 against 6 strata, with
# null_threshold=0.3 → ~70% noise admission. v3 lifts the bar:
# - stratify_by=(env_name, gamma) → 2 L × 3 γ = 6 strata at β=0;
#   STILL too small. v3 fixes this by REQUIRING n_strata ≥
#   min_strata calibrated against the panel's per-stratum Cohen's
#   d SE.
#
# Per-stratum d SE at n_seeds=27 ≈ sqrt(4/27) ≈ 0.385. For ρ
# under H0 with n_strata=6 strata, SE ≈ 1/sqrt(n-1) ≈ 0.45 → the
# null band |ρ|<0.3 has type-I-error rate ≈ 0.56, NOT 0.05.
#
# v3 N1 imposes:
# - min_strata ≥ 6 STRICT.
# - null_threshold = 0.30 with the understanding that NULL is
#   declared only if ρ is within ±0.30 AND p>0.30 (i.e., the
#   panel is NOT marginally significant in either direction).
# - effect_observed_threshold = 0.70 (sign-flip refutation).
#
# At n_strata=6, |r|_crit at p=0.05 is 0.829; |r|_crit at p=0.30
# is 0.34. So null_threshold=0.30 + p_floor=0.30 means "data
# rules out marginal signal" — a stronger null commitment than
# v2's noise-permissive |ρ|≤0.3 alone.
@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_raw_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=(
        _SYNTHETIC_TYPEB_V3_BETA0_SCOPE
        & pl.col('eval_best_burst_raw_mean').is_finite()
    ),
    predicted_direction='null',
)
def n_states_alone_does_not_drive_dout__synthetic_typeb_v3(
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
    null_threshold_rho: float = 0.30,
    null_threshold_p: float = 0.30,
    effect_observed_threshold: float = 0.70,
    min_strata: int = 6,
) -> tuple[Verdict, RefutationClass | None]:
    """N1 — At β=0 (Type-A peaked, wide argmax-margin), L doesn't
    drive d_out direction across the 6 sub-strata (2 L × 3 γ).

    **HELD-as-null criterion** (predicted_direction='null'):
        |ρ| < null_threshold_rho (0.30) AND
        p > null_threshold_p (0.30)
    The data must rule out a marginal slope in either direction.
    At n_strata=6, |r|_crit at p=0.30 is 0.34; null_threshold_rho
    < this, so "satisfies-null" implies "not even marginally
    detectable".

    **SIGN_FLIP / SIGN_DETECTED refutation**: |ρ| ≥ 0.70 →
    capacity has an independent effect at β=0. Walks back the
    "FA-binding modulates β's mechanism only" interpretation:
    capacity might drive d_out directly via a non-β channel
    (substantive open question; framework refuses to silently
    absorb).

    **POWER_INSUFFICIENT**: n_strata < 6 OR |ρ| in the
    [null_threshold_rho, effect_observed_threshold) middle band.
    HONEST UNDERPOWER — neither HELD nor refuted; the design
    can't disambiguate at this sub-panel size. v2's N1 collapsed
    POWER_INSUFFICIENT into HELD via permissive |ρ|≤0.3 → ~70%
    type-I; v3 N1 keeps the band explicit."""
    del treatment_arm, baseline_arm, source, stratify_by
    del covariate_name, covariate_key_field, covariates_per_key
    del scope_predictor, min_baseline_predictor, min_seeds_per_arm
    if cross_stratum_property_slope.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT, None
    rho = cross_stratum_property_slope.rho
    p_value = cross_stratum_property_slope.p_value
    if math.isnan(rho) or math.isnan(p_value):
        return Verdict.POWER_INSUFFICIENT, None
    if abs(rho) < null_threshold_rho and p_value > null_threshold_p:
        return Verdict.HELD, None
    if abs(rho) >= effect_observed_threshold:
        return Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP
    return Verdict.POWER_INSUFFICIENT, None


BRIDGES = (
    ddqn_harms_under_high_beta__synthetic_typeb_v3,
    ddqn_helps_when_argmax_margin_wide__synthetic_typeb_v3,
    ddqn_harm_amplified_at_g999__synthetic_typeb_v3,
    n_states_alone_does_not_drive_dout__synthetic_typeb_v3,
)
