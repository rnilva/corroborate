"""Hasselt bias-correction chain: jensen_gap → outcome.

Link side (mech-activation half lives in `dqn_bridges.py` on a
separate corpus). 8 bridges:

- `acrobot_per_burst_link_active__gamma_0999`: per-burst link r on
  Acrobot γ=0.999 (paired_link_per_burst → phase_link_consistency).
- `reach_link_{backdoor_ate_negative,placebo_refuted,rcc_robust}`:
  DoWhy backdoor + refutations on the per-burst Δ_jens panel,
  REACH-cohort scope.
- `extreme_q_divergence_attenuates_link__{binary,placebo_refuted,
  rcc_robust}`: DoWhy backdoor + refutations on the link-attenuation
  binary contrast (q_div > 1000 vs in-band cells).
- `fourrooms_action_dim_link_active__inflated`: within-FourRooms
  panel_regress of Δ_outcome on Δ_jens across `action_duplicate_k`.

All sourced from `jensen_gap` (the predictor); tier ASSOCIATIONAL
since the data is observational on the per-burst predictor even
when derived from a do() contrast."""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import numpy.typing as npt
import polars as pl

from corroborate.analyses.link_attenuation_dowhy import LinkAttenuationDowhyResult
from corroborate.analyses.paired_delta_link_dowhy import PairedDeltaLinkDowhyResult
from corroborate.analyses.paired_link_per_burst import (
    PerBurstLinkResult, phase_link_consistency,
)
from corroborate.analyses.stratum_effect_panel import (
    StratumEffectPanel, panel_regress,
)
from corroborate.bridge.bridge import Direction, Tier, claim_bridge
from corroborate.bridge.predicates import finite, finite_ge
from corroborate.bridge.verdict import Verdict
from corroborate.measurables import Measurable

from experiments.findings.ddqn._arms import DDQN_ARM, VANILLA_ARM
from experiments.findings.ddqn._common import (
    JENSEN_BIAS_PER_BURST_MEAN, MC_RETURN_PER_BURST_MEAN,
)
from experiments.findings.ddqn._scope import (
    DDQN_RELEVANT_SCOPE, G1_VANILLA_CONFIG_PREMISE_ACTIVE, REACH_ENVS_FOUR,
)
from experiments.findings.ddqn._verdicts import (
    dowhy_backdoor_verdict, dowhy_placebo_verdict, dowhy_rcc_verdict,
)


@claim_bridge(
    source='jensen_gap',
    target='mc_return',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=(
        (pl.col('env_name') == 'Acrobot-v1')
        & (pl.col('corpus') == 'l2_x_gamma_acrobot')
        & finite_ge('effective_horizon', 80.0)
        & (pl.col('optimizer.inner.weight_decay') == 0.0001)
    ),
)
def acrobot_per_burst_link_active__gamma_0999(
    paired_link_per_burst: PerBurstLinkResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    target: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ] = MC_RETURN_PER_BURST_MEAN,
    predictor: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ] = JENSEN_BIAS_PER_BURST_MEAN,
    env_name: str = 'Acrobot-v1',
    consistency_floor: float = 0.7,
) -> Verdict:
    """Per-burst r(Δ_jens, Δ_out) significantly negative in at
    least `consistency_floor` of bursts on Acrobot γ=0.999."""
    del treatment_arm, baseline_arm, target, predictor
    plc = phase_link_consistency(paired_link_per_burst, env_name=env_name)
    if math.isnan(plc):
        return Verdict.POWER_INSUFFICIENT
    if plc >= consistency_floor:
        return Verdict.HELD
    if plc >= consistency_floor * 0.5:
        return Verdict.POWER_INSUFFICIENT
    return Verdict.NO_EFFECT


@claim_bridge(
    source='jensen_gap',
    target='eval_best_burst_mean',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=DDQN_RELEVANT_SCOPE,
)
def reach_link_backdoor_ate_negative(
    paired_delta_link_dowhy: PairedDeltaLinkDowhyResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    link_target: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ] = MC_RETURN_PER_BURST_MEAN,
    link_predictor: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ] = JENSEN_BIAS_PER_BURST_MEAN,
    env_filter: tuple[str, ...] = REACH_ENVS_FOUR,
    ate_ceiling: float = -0.1,
) -> Verdict:
    """DoWhy backdoor ATE on per-(env, burst, seed) Δ panel across
    REACH cohort. HELD when identified AND ATE ≤ ceiling."""
    del treatment_arm, baseline_arm, link_predictor, link_target, env_filter
    return dowhy_backdoor_verdict(
        paired_delta_link_dowhy.backdoor, ate_ceiling=ate_ceiling,
    )


@claim_bridge(
    source='jensen_gap',
    target='eval_best_burst_mean',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=DDQN_RELEVANT_SCOPE,
)
def reach_link_placebo_refuted(
    paired_delta_link_dowhy: PairedDeltaLinkDowhyResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    link_target: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ] = MC_RETURN_PER_BURST_MEAN,
    link_predictor: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ] = JENSEN_BIAS_PER_BURST_MEAN,
    env_filter: tuple[str, ...] = REACH_ENVS_FOUR,
    placebo_max_ratio: float = 0.2,
) -> Verdict:
    """Placebo refutation: random treatment shrinks ATE to ~zero.
    HELD when |placebo / real| < `placebo_max_ratio`."""
    del treatment_arm, baseline_arm, link_predictor, link_target, env_filter
    return dowhy_placebo_verdict(
        paired_delta_link_dowhy.placebo, max_ratio=placebo_max_ratio,
    )


@claim_bridge(
    source='jensen_gap',
    target='eval_best_burst_mean',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=DDQN_RELEVANT_SCOPE,
)
def reach_link_rcc_robust(
    paired_delta_link_dowhy: PairedDeltaLinkDowhyResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    link_target: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ] = MC_RETURN_PER_BURST_MEAN,
    link_predictor: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ] = JENSEN_BIAS_PER_BURST_MEAN,
    env_filter: tuple[str, ...] = REACH_ENVS_FOUR,
    rcc_max_drift_ratio: float = 0.1,
) -> Verdict:
    """RCC refutation: noise covariate added to adjustment set
    leaves ATE within `rcc_max_drift_ratio` of real."""
    del treatment_arm, baseline_arm, link_predictor, link_target, env_filter
    return dowhy_rcc_verdict(
        paired_delta_link_dowhy.random_common_cause,
        max_drift_ratio=rcc_max_drift_ratio,
    )


@claim_bridge(
    source='jensen_gap',
    target='eval_best_burst_mean',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=(
        (pl.col('env_name') == 'FourRooms-misc')
        & pl.col('action_duplicate_k').is_not_null()
        & G1_VANILLA_CONFIG_PREMISE_ACTIVE
    ),
    predicted_direction='a_lt_b',
)
def fourrooms_action_dim_link_active__inflated(
    stratum_effect_panel: StratumEffectPanel,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    measurables: tuple[str, ...] = ('jensen_gap', 'eval_best_burst_mean'),
    stratify_by: tuple[str, ...] = ('action_duplicate_k',),
    min_seeds_per_arm: int = 5,
    x: str = 'jensen_gap',
    y: str = 'eval_best_burst_mean',
    slope_max: float = -0.05,
    r_squared_floor: float = 0.7,
    min_strata: int = 3,
) -> Verdict:
    """Within-FourRooms chain-amplifier link via `action_duplicate_k`
    panel (k ∈ {1,2,3,4}). Per-k Δ_outcome regressed on per-k
    Δ_jens. HELD when slope ≤ slope_max AND clean fit (R² ≥ floor)
    AND n_strata ≥ min_strata."""
    del treatment_arm, baseline_arm, measurables, stratify_by, min_seeds_per_arm
    result = panel_regress(stratum_effect_panel, x=x, y=y)
    if result.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT
    if math.isnan(result.slope):
        return Verdict.POWER_INSUFFICIENT
    if result.slope > slope_max:
        return Verdict.NO_EFFECT
    if result.r_squared < r_squared_floor:
        return Verdict.NO_EFFECT
    return Verdict.HELD


# Extreme Q-div trio: upper-bound scope companion to dormancy
# refutation (CLAIM 2). Bridges share the same scope predicate
# `_EXTREME_Q_DIV_SCOPE` so cluster identity is structurally
# unified by extent_hash.
_EXTREME_Q_DIV_SCOPE = (
    (pl.col('total_steps') == 1_000_000)
    & finite('q_divergence_score')
    & (
        pl.col('q_network.channels').is_null()
        | (pl.col('q_network.channels') != '(32,64)')
    )
)


@claim_bridge(
    source='jensen_gap',
    target='eval_best_burst_mean',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=_EXTREME_Q_DIV_SCOPE,
)
def extreme_q_divergence_attenuates_link__binary(
    link_attenuation_dowhy: LinkAttenuationDowhyResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    attenuator: str = 'q_divergence_score',
    binary_threshold: float = 1000.0,
    link_target: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ] = MC_RETURN_PER_BURST_MEAN,
    link_predictor: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ] = JENSEN_BIAS_PER_BURST_MEAN,
    ate_ceiling: float = -0.10,
    dedupe_strategy: str = 'mean',
) -> Verdict:
    """Binary contrast: cells with `q_divergence_score > 1000`
    have link strength attenuated by ≥ 0.10 vs in-band cells, after
    env-family backdoor adjustment. HELD when ATE ≤ -0.10.
    `zero_guard` handles RNG-dependent machine-epsilon signs."""
    del treatment_arm, baseline_arm, attenuator, binary_threshold
    del link_target, link_predictor, dedupe_strategy
    return dowhy_backdoor_verdict(
        link_attenuation_dowhy.backdoor,
        ate_ceiling=ate_ceiling, zero_guard=True,
    )


@claim_bridge(
    source='jensen_gap',
    target='eval_best_burst_mean',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=_EXTREME_Q_DIV_SCOPE,
)
def extreme_q_divergence_attenuates_link__placebo_refuted(
    link_attenuation_dowhy: LinkAttenuationDowhyResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    attenuator: str = 'q_divergence_score',
    binary_threshold: float = 1000.0,
    link_target: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ] = MC_RETURN_PER_BURST_MEAN,
    link_predictor: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ] = JENSEN_BIAS_PER_BURST_MEAN,
    placebo_max_ratio: float = 0.2,
    dedupe_strategy: str = 'mean',
) -> Verdict:
    """Placebo refutation on the binary above-1000 ATE."""
    del treatment_arm, baseline_arm, attenuator, binary_threshold
    del link_target, link_predictor, dedupe_strategy
    return dowhy_placebo_verdict(
        link_attenuation_dowhy.placebo, max_ratio=placebo_max_ratio,
    )


@claim_bridge(
    source='jensen_gap',
    target='eval_best_burst_mean',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=_EXTREME_Q_DIV_SCOPE,
)
def extreme_q_divergence_attenuates_link__rcc_robust(
    link_attenuation_dowhy: LinkAttenuationDowhyResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    attenuator: str = 'q_divergence_score',
    binary_threshold: float = 1000.0,
    link_target: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ] = MC_RETURN_PER_BURST_MEAN,
    link_predictor: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ] = JENSEN_BIAS_PER_BURST_MEAN,
    rcc_max_drift_ratio: float = 0.15,
    dedupe_strategy: str = 'mean',
) -> Verdict:
    """RCC refutation on the binary above-1000 ATE."""
    del treatment_arm, baseline_arm, attenuator, binary_threshold
    del link_target, link_predictor, dedupe_strategy
    return dowhy_rcc_verdict(
        link_attenuation_dowhy.random_common_cause,
        max_drift_ratio=rcc_max_drift_ratio,
    )


BRIDGES = (
    acrobot_per_burst_link_active__gamma_0999,
    reach_link_backdoor_ate_negative,
    reach_link_placebo_refuted,
    reach_link_rcc_robust,
    fourrooms_action_dim_link_active__inflated,
    extreme_q_divergence_attenuates_link__binary,
    extreme_q_divergence_attenuates_link__placebo_refuted,
    extreme_q_divergence_attenuates_link__rcc_robust,
)
