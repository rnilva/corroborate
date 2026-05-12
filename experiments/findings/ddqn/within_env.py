"""Within-env do() probes — γ on FourRooms + MetaMaze.

- `ddqn_benefit_scales_with_effective_horizon__fourrooms` (CLAIM 5):
  FR γ-sweep, per-γ stratum-Cohen's d on outcome, Pearson r against
  γ tests chain-depth scaling. AWAITING DATA (γ=0.999 FR cells
  absent post-rebuild).
- `metamaze_link_steeper_at_high_gamma` (CLAIM 24): on MetaMaze n_γ=2
  ({0.99, 0.999}), Δ_outcome should AMPLIFY at high γ if chain-depth
  is the lever. Currently REFUTED — was paired-Δ init-correlation.
"""
from __future__ import annotations

import math

import polars as pl

from corroborate.analyses.stratum_effect_panel import StratumEffectPanel
from corroborate.bridge.bridge import Direction, Tier, claim_bridge
from corroborate.bridge.verdict import Verdict
from corroborate.stats import MetaRegressionResult

from experiments.findings.ddqn._arms import (
    DDQN_ARM, INTERVENTION, VANILLA_ARM,
)
from experiments.findings.ddqn._scope import (
    G1_VANILLA_CONFIG_PREMISE_ACTIVE,
    VANILLA_JENS_NOISE_FLOOR,
)
from experiments.findings.ddqn._verdicts import (
    meta_regression_coefficient_verdict,
)


# Per-γ effective_horizon on FourRooms (empirical means at each γ
# on the current ddqn cache after `gamma_sweep_fourrooms` ingest
# 2026-05-12). Pinned for CLAIM 5's multi-stratum random-effects
# meta-regression on `effective_horizon` slope across γ-strata.
_FOURROOMS_EFFECTIVE_HORIZON_PER_GAMMA: dict[object, dict[str, float]] = {
    0.99: {'effective_horizon': 37.3},
    0.995: {'effective_horizon': 80.6},
    0.999: {'effective_horizon': 235.6},
}


# CLAIM 5 — within-env do(γ) on FourRooms.
@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_mean',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=(
        (pl.col('env_name') == 'FourRooms-misc')
        & pl.col('gamma').is_in([0.99, 0.995, 0.999])
        & ((pl.col('n_step') == 1) | pl.col('n_step').is_null())
        & pl.col('action_duplicate_k').is_null()
        & (pl.col('reward_scale').is_null() | (pl.col('reward_scale') == 1.0))
        & pl.col('target_sync.tau').is_null()
    ),
    predicted_direction='a_gt_b',
)
def ddqn_benefit_scales_with_effective_horizon__fourrooms(
    meta_regression_unpaired_d: MetaRegressionResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    source: str = 'eval_best_burst_mean',
    stratify_by: tuple[str, ...] = (
        'gamma', 'total_steps', 'reward_scale',
    ),
    covariate_key_field: str = 'gamma',
    covariates_per_key: dict[object, dict[str, float]] = (
        _FOURROOMS_EFFECTIVE_HORIZON_PER_GAMMA
    ),
    scope_predictor: str = 'jensen_gap',
    min_vanilla_predictor: float = VANILLA_JENS_NOISE_FLOOR,
    slope_threshold: float = 0.01,
    min_strata: int = 3,
) -> Verdict:
    """Within-FR do(γ) chain-depth scaling probe. Per-(γ, config)
    independent-samples Cohen's d → random-effects meta-regression
    on `effective_horizon` (env-derived from γ). HELD when β_eff_h
    ≥ `slope_threshold` AND significant.

    Post-roast issue 7 refactor (2026-05-12): replaced
    `stratum_id_scaling_verdict` (Pearson r on per-γ cohen_d
    panel) with the multi-stratum meta-regression shape used by
    CLAIM 19. The previous form inherited the n=3 envs Pearson r
    brittleness (`findings_n3_pearson_brittle`) — at n_strata=2
    (current cache γ=0.99 only), Pearson r is degenerate; even at
    n=3 a 1-SE perturbation could swing r between +1 and -1. The
    meta-regression form expands the panel via within-γ config
    replicates (`(γ, total_steps, reward_scale)` strata), giving
    proper SE on the slope coefficient.

    `slope_threshold=0.01` is the substrate-meaningful magnitude
    (calibrated like CLAIM 19): observed eff_h range across FR's
    γ values ≈ 42 units (27.6 at γ=0.99 → ~70 at γ=0.999);
    threshold 0.01 corresponds to |Δd| ≥ 0.42 across the span —
    Cohen's "small effect" magnitude.

    Cache has only γ=0.99 FR cells in three sub-corpora →
    n_strata ≤ 3, covariate is constant across all strata →
    meta-regression unidentified → POW_INSUF. Once γ=0.999 FR
    cells land, the multi-stratum form has between-γ variation
    AND within-γ replicates → proper test of the chain-depth
    amplification claim documented in
    `findings_gamma_sweep_three_regimes.md`."""
    del treatment_arm, baseline_arm, source, stratify_by
    del covariate_key_field, covariates_per_key
    del scope_predictor, min_vanilla_predictor
    return meta_regression_coefficient_verdict(
        meta_regression_unpaired_d,
        'effective_horizon',
        sign=1,
        threshold=slope_threshold,
        min_strata=min_strata,
    )


# CLAIM 24 — Within-MetaMaze do(γ): link slope steepens?
_METAMAZE_GAMMA_SCOPE = (
    (pl.col('env_name') == 'MetaMaze-misc')
    & pl.col('gamma').is_in([0.99, 0.999])
    & G1_VANILLA_CONFIG_PREMISE_ACTIVE
    & ((pl.col('n_step') == 1) | pl.col('n_step').is_null())
    & pl.col('action_duplicate_k').is_null()
    & (pl.col('reward_scale').is_null() | (pl.col('reward_scale') == 1.0))
    & pl.col('target_sync.tau').is_null()
)


def _metamaze_amplification_verdict(
    panel: StratumEffectPanel,
    *,
    high_gamma: float,
    low_gamma: float,
    high_floor: float,
    amplification_ratio_min: float,
) -> Verdict:
    """Shared decision logic for the two γ-amplification sibling
    bridges (mean + median). HELD when high-γ stratum Δ_o ≥
    `high_floor` AND high-γ ≥ `amplification_ratio_min` × low-γ
    Δ_o (or low-γ ≤ 0 trivially)."""
    if panel.n_strata < 2:
        return Verdict.POWER_INSUFFICIENT
    deltas_outcome = panel.deltas.get('eval_best_burst_mean', ())
    high_delta: float | None = None
    low_delta: float | None = None
    for stratum, delta in zip(panel.strata, deltas_outcome, strict=True):
        gamma_val = stratum[0]
        if (
            isinstance(gamma_val, (int, float))
            and not math.isnan(float(gamma_val))
        ):
            if math.isclose(float(gamma_val), high_gamma, rel_tol=1e-6):
                high_delta = delta
            elif math.isclose(float(gamma_val), low_gamma, rel_tol=1e-6):
                low_delta = delta
    if (
        high_delta is None or low_delta is None
        or math.isnan(high_delta) or math.isnan(low_delta)
    ):
        return Verdict.POWER_INSUFFICIENT
    if high_delta < high_floor:
        return Verdict.NO_EFFECT
    if low_delta <= 0:
        return Verdict.HELD
    if (high_delta / low_delta) >= amplification_ratio_min:
        return Verdict.HELD
    return Verdict.NO_EFFECT


@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=_METAMAZE_GAMMA_SCOPE,
    predicted_direction='a_gt_b',
)
def metamaze_link_steeper_at_high_gamma(
    stratum_effect_panel: StratumEffectPanel,
    *,
    measurables: tuple[str, ...] = ('eval_best_burst_mean',),
    stratify_by: tuple[str, ...] = ('gamma',),
    min_seeds_per_arm: int = 10,
    high_gamma: float = 0.999,
    low_gamma: float = 0.99,
    high_floor: float = 0.5,
    amplification_ratio_min: float = 1.5,
) -> Verdict:
    """Within-MetaMaze do(γ): n_γ=2 amplification test. HELD when
    high-γ Δ_o ≥ `high_floor` AND high-γ ≥ `amplification_ratio_min`
    × low-γ Δ_o (or low-γ ≤ 0). Currently REFUTED on postfix corpora
    — paired-Δ +2.55 was init-correlation, not amplification. NB:
    `eval_best_burst_mean` is itself a mean over seeds; per
    `findings_metamaze_gamma_link.md`, the high-γ Δ is outlier-driven
    (median Δ ≈ +2.55) — a true median-aggregated variant would need
    StratumEffectPanel to expose median deltas, which it currently
    doesn't."""
    del measurables, stratify_by, min_seeds_per_arm
    return _metamaze_amplification_verdict(
        stratum_effect_panel,
        high_gamma=high_gamma, low_gamma=low_gamma,
        high_floor=high_floor,
        amplification_ratio_min=amplification_ratio_min,
    )


BRIDGES = (
    ddqn_benefit_scales_with_effective_horizon__fourrooms,
    metamaze_link_steeper_at_high_gamma,
)
