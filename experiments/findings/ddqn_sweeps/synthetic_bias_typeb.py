"""Synthetic bias Type-A/B controlled-substrate bridges (v3.2).

v1 → v2 → v3 → v3.1 → v3.2 evolution lives in
`src/corroborate_rl/corroborate_rl/synthetic_bias_typeb.py`.

- **v1** scrapped: action-independent transitions; bandit in a tuxedo.
- **v2** scrapped: per-step reward-noise α conflated with the
  Q-target-side Var_a[V*(s')] that Cor 3.2's σ_clip actually
  concerns; under-powered n_seeds=12; over-parameterized FA at L=64.
- **v3** scrapped pre-launch (`/tmp/synthetic_v3_review.md`):
  state-baked deterministic payoff `mu_state(s) = peak·β^(s mod K)`
  had `Var_a[V*(s'_a)] = 0` identically (modular periodicity);
  Q* had only K=4 distinct values across L=1024 states (no FA
  capacity binding). v3 was the SAME conceptual error as v2 (per-
  step reward variance vs Q-target-side variance), just relocated.
- **v3.1**: RANDOM per-state payoffs
  `mu_state[s] = peak · (1 - spread + spread · U_s)` with
  `U_s ~ U(0, 1)` seeded by `payoff_seed`. Value iteration
  confirms `Var_a[V*(s'_a)] > 0` at every spread > 0 and Q* has
  ~L distinct entries (no modular collapse). The L axis now
  GENUINELY binds FA capacity.
- **v3.2** (pre-launch iteration on `/tmp/synthetic_v3_1_review.md`):
  n_seeds 8 → 16 to address attenuation; drop the L=32 envs to
  recover cell budget; tighten REFUTATION null_threshold 0.20 →
  0.15 to match the attenuation arithmetic (false-REFUTATION rate
  drops ~30-43% → ~15% at true ρ=−0.5). The N1 bridge (L-axis
  null at spread=0) becomes structurally degenerate at single-L
  and is REMOVED from the pre-registered set.

The v3.2 panel has TWO structural axes (L axis collapsed to a
single value):

- `payoff_spread` ∈ {0.0, 0.25, 0.5, 0.75, 1.0}: the v3.1
  anisotropy axis. spread=0 is the degenerate isotropic case
  (Var_a[V*]=0, DDQN expected NULL); spread=1 is maximum
  anisotropy.
- `payoff_seed` ∈ {0, 1, 2}: cross-realisation averaging. Multiple
  envs at each spread with different `payoff_seed` smooth
  cross-env over the seed-specific topology of the random
  per-state payoff vector.

Plus γ ∈ {0.95, 0.99, 0.999} as a substrate axis. 15 envs × 3 γ ×
2 arms × 16 seeds = 1440 cells (≤ 1500 budget).

The substantive predictions are pre-registered in
`docs/PRE_REGISTRATION_synthetic_bias_typeb.md`. The verdict
helper is the shared substrate primitive
`cross_stratum_signed_spearman_verdict` (calibrated for
n_strata≥10).

**Substantive scope (v3.2 narrowing)**: The v3.1 reviewer
verified via VI that σ_clip and Δ_v both scale linearly with
`payoff_spread`, so σ/Δ stays ratio-invariant at ~0.466 across
all spread > 0. v3.2 tests the mechanism in the
**FA-residual-heavy regime (σ/Δ ≈ 47%)**, NOT the Asterix 1-3%
knife-edge regime. A HELD verdict supports the mechanism shape
at FA-heavy σ/Δ; it does NOT establish that Asterix's
knife-edge mechanism is the same. The knife-edge regime test
remains open (would require a v4 with decoupled σ_clip / Δ_v
knobs).
"""
from __future__ import annotations

from types import MappingProxyType

import math

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


# v3.1+ synthetic env naming convention (preserved across v3.1 → v3.2;
# only the registered subset narrows in v3.2):
#   "TypeBChainV31-K{K}-L{n_states}-spread{spread}-seed{seed}-synthetic"
# γ is a substrate axis (set via YAML), not baked into the env
# name. The (env_name, gamma) pair identifies a cell stratum.
#
# Hardcoded structural covariates mirror chain_depth's
# `_LOG_HORIZON_PER_GAMMA` pattern. These are STRUCTURAL —
# derivable from the env name itself — not data-fitted.
# v3.2 narrowing: L axis collapsed to 1024 only.
_N_STATES_VALUES: tuple[int, ...] = (1024,)
_SPREAD_VALUES: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
_PAYOFF_SEEDS: tuple[int, ...] = (0, 1, 2)


def _build_synthetic_covariates() -> MappingProxyType[
    object, MappingProxyType[str, float]
]:
    """Per-env structural covariates for the v3.2 panel.

    Keys are env names; values are immutable dicts of the env's
    STRUCTURAL design parameters baked at registration time in
    `env_catalogue._register_synthetic_bias_typeb_panel`. The
    covariates exposed here are:

    - `payoff_spread`: the v3.1 anisotropy axis. Per-state payoff
      range scaling factor. spread=0 → all states peak_value;
      spread=1 → states uniform on [0, peak_value]. Var_a[V*(s')]
      scales monotonically with spread (verified by VI).
    - `n_states`: chain length L. v3.2 carries the L=1024-only
      panel; the covariate is constant across the panel and
      provided as metadata, not a Spearman covariate (the L-axis
      N1 bridge was removed in v3.2).
    - `log_n_states`: `log(L)`. Constant across v3.2 panel.
    - `payoff_seed`: the cross-realisation index. NOT used as a
      causal covariate; carried for diagnostic purposes only.
    - `expected_v_var_proxy`: closed-form upper bound on
      Var_a[V*(s'_a)] at this spread. For U(0, 1) draws with the
      shape `peak · (1 - spread + spread · U)`, the variance of
      the per-state mu_state is `(peak · spread)² / 12`. The
      Var_a[V*] at the optimal policy is approximately this scaled
      by `(1 - γ)^-2` for short-horizon credit assignment — used
      here as a diagnostic shape variable, NOT an exact predictor.
    """
    out: dict[object, MappingProxyType[str, float]] = {}
    peak_value = 1.0
    for n_states in _N_STATES_VALUES:
        for spread in _SPREAD_VALUES:
            for payoff_seed in _PAYOFF_SEEDS:
                name = (
                    f"TypeBChainV31-K4-L{n_states}"
                    f"-spread{spread}"
                    f"-seed{payoff_seed}-synthetic"
                )
                # Var of U(0,1) is 1/12; rescale by (peak·spread)².
                mu_var = (peak_value * spread) ** 2 / 12.0
                out[name] = MappingProxyType({
                    'payoff_spread': float(spread),
                    'n_states': float(n_states),
                    'log_n_states': math.log(float(n_states)),
                    'payoff_seed': float(payoff_seed),
                    'expected_v_var_proxy': mu_var,
                })
    return MappingProxyType(out)


_SYNTHETIC_COVARIATES: MappingProxyType[
    object, MappingProxyType[str, float]
] = _build_synthetic_covariates()


# Scope: v3.1+ synthetic envs (L=1024 only in v3.2). γ NOT pinned
# here because the sweep spans γ ∈ {0.95, 0.99, 0.999} and
# individual bridges decide whether to pin γ or pool.
_SYNTHETIC_TYPEB_V31_SCOPE: pl.Expr = (
    pl.col('env_name').str.starts_with('TypeBChainV31-K4-')
    & pl.col('env_name').str.ends_with('-synthetic')
)

# γ=0.999 sub-scope for the "gamma amplifies anisotropy" diagnostic.
_SYNTHETIC_TYPEB_V31_G999_SCOPE: pl.Expr = (
    _SYNTHETIC_TYPEB_V31_SCOPE & (pl.col('gamma') == 0.999)
)


# ============ PRIMARY prediction (P1) ============
#
# P1 — DDQN's outcome benefit decreases as payoff_spread grows.
# Cross-stratum Spearman ρ between `payoff_spread` (the v3.1
# anisotropy axis) and DDQN-vs-vanilla Cohen's d on outcome,
# stratified by (env_name, gamma). 15 envs × 3 γ = 45 strata; well
# above min_strata=10 calibration of the verdict helper.
#
# This is the LOAD-BEARING bridge — the v3.2 design's primary
# substantive prediction. The PRE_REGISTRATION doc defines its
# REFUTATION criterion (a specific data shape that retracts the
# claim that synthetic substrate enables causal env-feature
# identification of the Asterix Type-B mechanism, at the FA-residual-
# heavy σ/Δ ≈ 47% regime v3.2 tests).
#
# **Predicted mechanism**: at high payoff_spread, Var_a[V*(s'_a)]
# is non-trivial → vanilla's max-of-K bootstrap picks up
# policy-informative noise from successor V* heterogeneity →
# DDQN's clip introduces argmax-asymmetry that corrupts the
# knife-edge selection in chain-amplified regimes → DDQN HARMS.
#
# **v3.2 change**: null_threshold tightened from 0.20 → 0.15. At
# n_seeds=16, attenuation analysis (see `/tmp/synthetic_v3_1_review.md`
# §"n_seeds=8: power calculation") gives reliability r → 0.32 and
# attenuation factor √r → 0.57: a true ρ=−0.5 lands at observed
# ρ ≈ −0.29 with SE ≈ 0.13. P(|ρ| < 0.15) ≈ 15% at the new
# threshold (was 43% at the v3.1 0.20 threshold). The REFUTATION
# clause remains binding; the false-REFUTATION rate drops to a
# defensible level.
@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_raw_mean',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=(
        _SYNTHETIC_TYPEB_V31_SCOPE
        & pl.col('eval_best_burst_raw_mean').is_finite()
    ),
    predicted_direction='a_lt_b',
)
def ddqn_harms_under_high_spread__synthetic_typeb_v32(
    cross_stratum_property_slope: CrossStratumPropertySlopeResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    source: str = 'eval_best_burst_raw_mean',
    stratify_by: tuple[str, ...] = ('env_name', 'gamma'),
    covariate_name: str = 'payoff_spread',
    covariate_key_field: str = 'env_name',
    covariates_per_key: MappingProxyType[
        object, MappingProxyType[str, float]
    ] = _SYNTHETIC_COVARIATES,
    scope_predictor: str = 'eval_best_burst_raw_mean',
    min_baseline_predictor: float = -1e9,  # no gate; synthetic always active
    min_seeds_per_arm: int = 5,
    rho_threshold_held: float = 0.5,
    p_threshold: float = 0.05,
    null_threshold: float = 0.15,
    sign_flip_threshold: float = 0.5,
    min_strata: int = 10,
) -> tuple[Verdict, RefutationClass | None]:
    """P1 (PRIMARY) — Cross-stratum Spearman ρ between
    `payoff_spread` (the v3.1 anisotropy axis) and DDQN-vs-vanilla
    Cohen's d on outcome, stratified by (env_name, γ).

    **HELD criterion**: ρ ≤ −0.5 AND p ≤ 0.05 on n_strata ≥ 10.
    Direction: higher payoff_spread → higher Var_a[V*(s'_a)] →
    more max-of-K bootstrap bias → DDQN's clip corrupts the
    chain-amplified argmax → DDQN HARMS more.

    **REFUTATION criterion** (the load-bearing pre-registered
    walk-back-as-retraction; see PRE_REGISTRATION doc §REFUTATION):
    NO_EFFECT-NULL fires (|ρ| < 0.15 — tightened from v3.1's 0.20
    to match the v3.2 attenuation arithmetic at n_seeds=16) AND
    the panel is adequately powered (n_strata ≥ 15). In that case
    the v3.2 synthetic substrate DOES NOT reproduce the natural-env
    Asterix Type-B mechanism AT THE FA-RESIDUAL-HEAVY σ/Δ ≈ 47%
    REGIME v3.2 tests. NOT a publishable walk-back; the
    substrate-author must retract the claim that v3.2 enables
    causal env-feature identification at this σ/Δ regime.

    **SIGN_FLIP** (ρ ≥ +0.5): DDQN HELPS more as spread grows.
    Walks back the σ_clip → argmax-corruption mechanism: more
    Q-target heterogeneity makes DDQN's clip MORE effective at
    reducing optimism bias. Suggests distinct mechanism (e.g.,
    overestimation bias dominates margin-corruption) is the
    natural-env Asterix story."""
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


# ============ DIAGNOSTIC: γ amplification ============
#
# D2 — At γ=0.999 (the load-bearing chain-amplification regime
# where natural-env Asterix harm appears), ρ(payoff_spread, d_out)
# should be MORE negative than the pooled P1. The 1/(1-γ)=1000×
# horizon magnifies the per-state V* heterogeneity contribution
# → knife-edge argmax-corruption from DDQN's clip is amplified.
#
# n_strata at γ=0.999 in v3.2 = 5 spread × 3 payoff_seeds = 15
# envs at a single γ level. Borderline at the verdict helper's
# min_strata=15 calibration — STRUCTURAL POWER_INSUFFICIENT for
# very weak effects.
#
# **PRE-REGISTERED VERDICT: power_insufficient** (the v3 reviewer's
# discipline issue #1: the bridge was documented as STRUCTURAL
# POWER_INSUFFICIENT but pre-registered as HELD in the YAML. v3.1+
# fixes this — the predicted verdict matches the structural
# diagnostic.)
@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_raw_mean',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=(
        _SYNTHETIC_TYPEB_V31_G999_SCOPE
        & pl.col('eval_best_burst_raw_mean').is_finite()
    ),
    predicted_direction='a_lt_b',
)
def ddqn_harm_amplified_at_g999__synthetic_typeb_v32(
    cross_stratum_property_slope: CrossStratumPropertySlopeResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    source: str = 'eval_best_burst_raw_mean',
    stratify_by: tuple[str, ...] = ('env_name',),
    covariate_name: str = 'payoff_spread',
    covariate_key_field: str = 'env_name',
    covariates_per_key: MappingProxyType[
        object, MappingProxyType[str, float]
    ] = _SYNTHETIC_COVARIATES,
    scope_predictor: str = 'eval_best_burst_raw_mean',
    min_baseline_predictor: float = -1e9,
    min_seeds_per_arm: int = 5,
    rho_threshold_held: float = 0.6,
    p_threshold: float = 0.1,
    null_threshold: float = 0.15,
    sign_flip_threshold: float = 0.5,
    min_strata: int = 15,
) -> tuple[Verdict, RefutationClass | None]:
    """D2 (DIAGNOSTIC) — At γ=0.999, ρ(payoff_spread, d_out) ≤ −0.6
    with p ≤ 0.1 across n_strata = 15 envs at the single γ level.
    The same shape as the natural-env Asterix γ=0.999 harm finding
    (memory `findings_asterix_g999_harm_is_optimization_dynamics`).

    **STRUCTURAL POWER_INSUFFICIENT**: The verdict helper's
    `min_strata=15` floor catches the boundary case where the
    γ=0.999 sub-panel has 15 envs (5 spread × 3 payoff_seeds) at
    L=1024-only in v3.2. v3.2 pre-registers the PREDICTED verdict
    as POWER_INSUFFICIENT (matching the structural diagnostic),
    inheriting v3.1's fix to v3's discipline issue #1.

    Diagnostic value is in the OBSERVED ρ direction + magnitude
    (surfaced in the analysis result, not the verdict) compared to
    P1's pooled ρ. If observed |ρ_γ=0.999| > |ρ_pooled| with
    consistent sign, the γ-amplification direction is corroborated
    even when the formal verdict can't fire HELD at this
    sub-scope."""
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


# ============ v3 → v3.1 → v3.2 bridge surgery: N1 dropped ============
#
# v3.1 carried N1 (`n_states_alone_does_not_drive_dout__synthetic_typeb_v31`)
# as an adequately-powered FALSIFIABLE null at payoff_spread=0:
# "L doesn't drive d_out direction at the isotropic baseline". v3.2
# drops the L=32 envs from the registered panel (see the YAML
# rationale + `_register_synthetic_bias_typeb_panel`'s v3.2 doc).
# With L=1024 the only registered value, the L-axis Spearman has a
# single covariate value at spread=0 → ρ is undefined. N1 was
# REMOVED from the pre-registered bridge set and from the Finding's
# BRIDGES tuple. The L-as-modulator question is structurally
# unaddressable at this panel; a future v4 with a richer L axis
# would restore it.


BRIDGES = (
    ddqn_harms_under_high_spread__synthetic_typeb_v32,
    ddqn_harm_amplified_at_g999__synthetic_typeb_v32,
)
