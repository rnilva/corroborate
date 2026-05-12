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
from corroborate.analyses.stratum_link_moderation_dowhy import (
    StratumLinkModerationDowhyResult,
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
        & (pl.col('gamma') == 0.999)
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
    least `consistency_floor` of bursts on Acrobot γ=0.999.

    Phase-5 audit (2026-05-12): kept as-is. Per-burst within-cell
    seed pairing IS structural — computing a Pearson r at a single
    (env, burst) stratum requires multiple observations within
    that stratum, and seed-paired Δs are the only way to get
    per-seed scalars to correlate. There's no independent-samples
    analog at within-stratum grain. The user's "replace seed-paired
    analyses" directive applied to cross-stratum pseudo-
    replication (Phases 1, 3, 4) — not within-stratum estimation.

    AWAITING DATA: scope gates on `corpus == 'l2_x_gamma_acrobot'`
    which isn't in the current universal cache (similar to CLAIM 5
    and Polyak bridges). Per `findings_l2_acrobot_goldilocks.md`,
    per-burst r ranges ≈ -0.93 to -0.998 with plc=1.0 on that
    corpus — when reintegrated, bridge fires HELD."""
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
        ate_threshold=ate_ceiling,
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
    stratum_link_moderation_dowhy: StratumLinkModerationDowhyResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    attenuator: str = 'q_divergence_score',
    binary_threshold: float = 1.0,
    link_target: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ] = MC_RETURN_PER_BURST_MEAN,
    link_predictor: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ] = JENSEN_BIAS_PER_BURST_MEAN,
    min_vanilla_predictor: float = 0.05,
    interaction_ate_floor: float = 0.10,
    placebo_max_ratio: float = 0.2,
    rcc_max_drift_ratio: float = 0.15,
) -> Verdict:
    """Binary contrast: envs with mean `q_divergence_score > 1.0`
    (Q above the Bellman bound, the "Q-divergent" semantic per
    `findings_q_div_threshold_too_loose.md`) have a weaker
    bias→outcome **link** vs in-band envs.

    Phase-4b refactor (2026-05-12, post-roast): replaced
    `stratum_outcome_attenuation_dowhy` (outcome-only, didn't see
    Δ_jens → faith claim that outcome-attenuation = link-attenuation)
    with `stratum_link_moderation_dowhy` — the proper mediation-
    aware test. The interaction coefficient
    `Δ_predictor × 1[env above q_div threshold]` IS the link
    moderation: β_int > 0 means above-threshold envs have a
    less-negative slope of Δ_outcome on Δ_jens (link weakened).
    Independent-samples per (env, burst) stratum, no seed pairing.

    Identification: binary_attenuator is env-determined and would
    be colinear with env one-hot. Resolution: only the
    interaction term enters as the causal target; env-dummies +
    Δ_predictor adjust. The interaction's within-env variation
    (Δ_predictor changes across bursts within each env) makes
    β_int identifiable.

    HELD when interaction β ≥ `interaction_ate_floor` AND placebo
    refutes AND RCC stable. Mech conditioning via
    `min_vanilla_predictor=0.05`.

    AWAITING DATA: the current cache's max q_divergence_score is
    1.05 (one CartPole cell). The pre-rebuild sync=10k MinAtar
    corpora that produced Q-explosion regimes (q_div ≫ 1) aren't
    in the universal cache. Bridge fires POW_INSUF until those
    corpora are reintegrated."""
    del treatment_arm, baseline_arm, attenuator, binary_threshold
    del link_target, link_predictor, min_vanilla_predictor
    return dowhy_trio_verdict(
        stratum_link_moderation_dowhy,
        ate_threshold=interaction_ate_floor,
        sign=1,
        placebo_max_ratio=placebo_max_ratio,
        rcc_max_drift_ratio=rcc_max_drift_ratio,
    )


BRIDGES = (
    acrobot_per_burst_link_active__gamma_0999,
    reach_link_dowhy_corroborated,
    fourrooms_action_dim_link_active__inflated,
    extreme_q_divergence_attenuates_link__dowhy_corroborated,
)
