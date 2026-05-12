"""Hasselt bias-correction chain: jensen_gap → outcome.

Link side (mech-activation half lives in `dqn_bridges.py` on a
separate corpus). 4 bridges:

- `acrobot_per_burst_link_active__gamma_0999`: per-burst link r on
  Acrobot γ=0.999 (paired_link_per_burst → phase_link_consistency).
- `reach_link_dowhy_corroborated`: composite DoWhy backdoor +
  placebo + RCC on the per-burst Δ_jens panel, REACH-cohort scope.
- `extreme_q_divergence_attenuates_link__dowhy_corroborated`:
  composite DoWhy backdoor + placebo + RCC on the
  link-attenuation binary contrast (env-mean q_div > 1000 vs
  in-band envs). "Link" = bias-drops→outcome-rises causal arrow;
  tested via per-(env, burst) stratum-Δ outcome under mech-active
  conditioning (vanilla mean jens > 0.05 + DDQN's structural
  bias-reduction tendency means Δ_outcome here proxies the
  mech→outcome link, not just a marginal effect).
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

from corroborate.analyses.paired_link_per_burst import (
    PerBurstLinkResult, phase_link_consistency,
)
from corroborate.analyses.stratum_delta_link_dowhy import (
    StratumDeltaLinkDowhyResult,
)
from corroborate.analyses.stratum_effect_panel import (
    StratumEffectPanel, panel_regress,
)
from corroborate.analyses.stratum_outcome_attenuation_dowhy import (
    StratumOutcomeAttenuationDowhyResult,
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
from experiments.findings.ddqn._verdicts import dowhy_trio_verdict


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
def reach_link_dowhy_corroborated(
    stratum_delta_link_dowhy: StratumDeltaLinkDowhyResult,
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
    min_vanilla_predictor: float = 0.05,
    ate_ceiling: float = -0.1,
    placebo_max_ratio: float = 0.2,
    rcc_max_drift_ratio: float = 0.1,
) -> Verdict:
    """DoWhy backdoor + placebo + RCC trio on per-(env, burst)
    **stratum-level** Δ panel across REACH cohort
    (Acrobot/FourRooms/MountainCar/MetaMaze). Phase-3 refactor
    (2026-05-12): replaced `paired_delta_link_dowhy` (per-(env,
    burst, seed) seed-paired rows) with `stratum_delta_link_dowhy`
    (per-(env, burst) independent-samples rows; seeds pooled within
    each arm). Mech conditioning built in via
    `min_vanilla_predictor=0.05` — strata where vanilla mean jens <
    0.05 (G1 dormant) never reach DoWhy.

    HELD iff backdoor identified the predicted-negative ATE AND
    placebo shrunk it to ~zero AND RCC left it near-stable. The
    three checks always travel together."""
    del treatment_arm, baseline_arm, link_predictor, link_target
    del env_filter, min_vanilla_predictor
    return dowhy_trio_verdict(
        stratum_delta_link_dowhy,
        ate_ceiling=ate_ceiling,
        placebo_max_ratio=placebo_max_ratio,
        rcc_max_drift_ratio=rcc_max_drift_ratio,
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


@claim_bridge(
    source='jensen_gap',
    target='eval_best_burst_mean',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=(
        (pl.col('total_steps') == 1_000_000)
        & finite('q_divergence_score')
        & (
            pl.col('q_network.channels').is_null()
            | (pl.col('q_network.channels') != '(32,64)')
        )
    ),
)
def extreme_q_divergence_attenuates_link__dowhy_corroborated(
    stratum_outcome_attenuation_dowhy: StratumOutcomeAttenuationDowhyResult,
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
    min_vanilla_predictor: float = 0.05,
    ate_ceiling: float = -0.10,
    placebo_max_ratio: float = 0.2,
    rcc_max_drift_ratio: float = 0.15,
) -> Verdict:
    """Binary contrast: envs with mean `q_divergence_score > 1000`
    have a weaker bias→outcome **link** vs in-band envs.

    Phase-4 refactor (2026-05-12): replaced `link_attenuation_dowhy`
    (within-(env, burst) Pearson r computed from seed-paired Δs as
    outcome) with `stratum_outcome_attenuation_dowhy` (per-(env,
    burst) independent-samples Δ_outcome). "Link" is the causal
    claim "bias drops → outcome rises," not the seed-pairing
    mechanic. Mech conditioning via `min_vanilla_predictor=0.05`
    ensures premise active per stratum; DDQN's structural tendency
    to reduce bias means Δ_outcome attenuation in high-q_div
    strata corresponds to a weakened mech→outcome link, not just
    a marginal Δ_outcome effect.

    DoWhy backdoor + placebo + RCC trio under env one-hot
    adjustment. `zero_guard=True` on backdoor handles RNG-
    dependent machine-epsilon signs."""
    del treatment_arm, baseline_arm, attenuator, binary_threshold
    del link_target, link_predictor, min_vanilla_predictor
    return dowhy_trio_verdict(
        stratum_outcome_attenuation_dowhy,
        ate_ceiling=ate_ceiling,
        placebo_max_ratio=placebo_max_ratio,
        rcc_max_drift_ratio=rcc_max_drift_ratio,
        zero_guard=True,
    )


BRIDGES = (
    acrobot_per_burst_link_active__gamma_0999,
    reach_link_dowhy_corroborated,
    fourrooms_action_dim_link_active__inflated,
    extreme_q_divergence_attenuates_link__dowhy_corroborated,
)
