"""Hasselt bias-correction chain: jensen_gap → outcome.

Link side (mech-activation half lives in `dqn_bridges.py` on a
separate corpus). 8 bridges across 4 causal claims (the two
DoWhy claims each split into 3 bridges per CLAUDE.md's
cluster-shaped causal claims principle):

- `acrobot_per_burst_link_active__gamma_0999`: per-burst link
  binomial test on Acrobot γ=0.999.
- `reach_link_{backdoor_ate_negative, placebo_refuted, rcc_robust}`:
  DoWhy backdoor + placebo + RCC on the per-(env, burst)
  stratum-Δ panel, REACH-cohort scope. Three logically distinct
  robustness questions; Finding-level cluster verdict
  AND-aggregates.
- `extreme_q_div_link_{interaction_positive, placebo_refuted,
  rcc_robust}`: same three-bridge cluster on the
  Δ_predictor × 1[env above q_div threshold] interaction term
  (link moderation by extreme Q-divergence).
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
    PerBurstLinkResult,
    phase_link_consistency_binomial_p,
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
    JENSEN_BIAS_PER_BURST_MEAN, MC_RETURN_RAW_PER_BURST_MEAN,
)
from experiments.findings.ddqn._scope import (
    DDQN_RELEVANT_SCOPE, G1_VANILLA_CONFIG_PREMISE_ACTIVE,
    REACH_ENVS_FOUR, VANILLA_JENS_NOISE_FLOOR,
)
from experiments.findings.ddqn._verdicts import (
    dowhy_backdoor_verdict,
    dowhy_placebo_verdict,
    dowhy_rcc_verdict,
)


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
    ] = MC_RETURN_RAW_PER_BURST_MEAN,
    predictor: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ] = JENSEN_BIAS_PER_BURST_MEAN,
    env_name: str = 'Acrobot-v1',
    significance: float = 0.05,
    expected_sign: int = -1,
    binomial_alpha: float = 0.05,
    min_bursts: int = 6,
) -> Verdict:
    """Per-burst r(Δ_jens, Δ_out) significantly negative in
    more bursts than chance on Acrobot γ=0.999. Binomial test
    against H0 "fraction-of-significant-sign-match bursts is
    at chance rate".

    Phase-5 audit (2026-05-12): per-burst within-cell seed
    pairing IS structural — computing a Pearson r at a single
    (env, burst) stratum requires multiple observations within
    that stratum, and seed-paired Δs are the only way to get
    per-seed scalars to correlate.

    Post-roast (issue 5): replaced bare-fraction threshold
    (`plc ≥ 0.7 → HELD`) with binomial-test gate. Under H0 (no
    link), each per-burst sign-match-and-significant test has
    Type-I rate `significance/2`; the count of such bursts ~
    Binomial(n_bursts, significance/2). HELD when the observed
    count is significantly above chance via
    `phase_link_consistency_binomial_p`.

    `expected_sign=-1` for the bias-correction story: r < 0
    means more bias reduction → more outcome gain (per the
    panel's negative-r-when-active convention for source=jens,
    target=outcome).

    AWAITING DATA: scope gates on Acrobot γ=0.999 (the
    `l2_x_gamma_acrobot` corpus) which isn't in the current
    universal cache (similar to CLAIM 5 and Polyak bridges). Per
    `findings_l2_acrobot_goldilocks.md`, per-burst r ranges
    ≈ -0.93 to -0.998 with plc=1.0 on that corpus — when
    reintegrated, the binomial p-value will be vanishingly small
    and the bridge fires HELD."""
    del treatment_arm, baseline_arm, target, predictor
    p_value, _n_signif, n_total = phase_link_consistency_binomial_p(
        paired_link_per_burst,
        env_name=env_name,
        significance=significance,
        expected_sign=expected_sign,
    )
    if n_total < min_bursts:
        return Verdict.POWER_INSUFFICIENT
    if math.isnan(p_value):
        return Verdict.POWER_INSUFFICIENT
    if p_value < binomial_alpha:
        return Verdict.HELD
    # Distinguish "tested, null not rejected" from "barely below
    # rejection" — POW_INSUF only when borderline (p between
    # α and 2α).
    if p_value < binomial_alpha * 2:
        return Verdict.POWER_INSUFFICIENT
    return Verdict.NO_EFFECT


# REACH causal trio. Post-roast issue 6 (2026-05-12): the
# Phase-B trio collapse into one composite bridge was wrong per
# CLAUDE.md's cluster-shaped causal claims principle. The three
# checks test logically distinct robustness questions (backdoor =
# adjustment-identified ATE; placebo = instrument validity; RCC =
# omitted-confound sensitivity) and should be authored as
# separate bridges. The Finding's cluster verdict
# (`finding_reach_bias_link.BRIDGES`) handles AND-aggregation at
# the graph level — SUPPORTED iff all three HELD per
# `composed_verdict`.
@claim_bridge(
    source='jensen_gap',
    target='eval_best_burst_raw_mean',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=DDQN_RELEVANT_SCOPE,
)
def reach_link_backdoor_ate_negative(
    stratum_delta_link_dowhy: StratumDeltaLinkDowhyResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    link_target: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ] = MC_RETURN_RAW_PER_BURST_MEAN,
    link_predictor: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ] = JENSEN_BIAS_PER_BURST_MEAN,
    env_filter: tuple[str, ...] = REACH_ENVS_FOUR,
    min_vanilla_predictor: float = VANILLA_JENS_NOISE_FLOOR,
    ate_ceiling: float = -0.1,
) -> Verdict:
    """REACH stratum-Δ link: DoWhy backdoor ATE of Δ_jens on
    Δ_outcome is predicted-negative (more bias-reduction → more
    outcome-gain) under env-adjustment. HELD when identified ∧
    ATE ≤ `ate_ceiling`."""
    del treatment_arm, baseline_arm, link_predictor, link_target
    del env_filter, min_vanilla_predictor
    return dowhy_backdoor_verdict(
        stratum_delta_link_dowhy.backdoor,
        ate_threshold=ate_ceiling, sign=-1,
    )


@claim_bridge(
    source='jensen_gap',
    target='eval_best_burst_raw_mean',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=DDQN_RELEVANT_SCOPE,
)
def reach_link_placebo_refuted(
    stratum_delta_link_dowhy: StratumDeltaLinkDowhyResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    link_target: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ] = MC_RETURN_RAW_PER_BURST_MEAN,
    link_predictor: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ] = JENSEN_BIAS_PER_BURST_MEAN,
    env_filter: tuple[str, ...] = REACH_ENVS_FOUR,
    min_vanilla_predictor: float = VANILLA_JENS_NOISE_FLOOR,
    placebo_max_ratio: float = 0.2,
) -> Verdict:
    """REACH stratum-Δ link: placebo refutation — random
    treatment shrinks ATE near zero. HELD when |placebo/real| <
    `placebo_max_ratio`."""
    del treatment_arm, baseline_arm, link_predictor, link_target
    del env_filter, min_vanilla_predictor
    return dowhy_placebo_verdict(
        stratum_delta_link_dowhy.placebo,
        max_ratio=placebo_max_ratio,
    )


@claim_bridge(
    source='jensen_gap',
    target='eval_best_burst_raw_mean',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=DDQN_RELEVANT_SCOPE,
)
def reach_link_rcc_robust(
    stratum_delta_link_dowhy: StratumDeltaLinkDowhyResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    link_target: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ] = MC_RETURN_RAW_PER_BURST_MEAN,
    link_predictor: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ] = JENSEN_BIAS_PER_BURST_MEAN,
    env_filter: tuple[str, ...] = REACH_ENVS_FOUR,
    min_vanilla_predictor: float = VANILLA_JENS_NOISE_FLOOR,
    rcc_max_drift_ratio: float = 0.1,
) -> Verdict:
    """REACH stratum-Δ link: random-common-cause refutation —
    synthetic confounder leaves ATE near-stable. HELD when drift
    < `rcc_max_drift_ratio`."""
    del treatment_arm, baseline_arm, link_predictor, link_target
    del env_filter, min_vanilla_predictor
    return dowhy_rcc_verdict(
        stratum_delta_link_dowhy.random_common_cause,
        max_drift_ratio=rcc_max_drift_ratio,
    )


@claim_bridge(
    source='jensen_gap',
    target='eval_best_burst_raw_mean',
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


# Extreme-Q-div link-moderation trio. Same cluster-shape
# discipline as the REACH trio: three separate bridges testing
# three logically distinct robustness questions on the interaction
# coefficient β_int (link attenuation by the q_div > 1.0 binary
# moderator). Finding-level cluster verdict handles AND-aggregation.
# AWAITING DATA: cache's max q_divergence_score is 1.05; pre-
# rebuild sync=10k MinAtar corpora (q_div ≫ 1) aren't in the
# universal cache. All three bridges fire POW_INSUF until those
# corpora reintegrate.
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
    target='eval_best_burst_raw_mean',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=_EXTREME_Q_DIV_SCOPE,
)
def extreme_q_div_link_interaction_positive(
    stratum_link_moderation_dowhy: StratumLinkModerationDowhyResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    attenuator: str = 'q_divergence_score',
    binary_threshold: float = 1.0,
    link_target: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ] = MC_RETURN_RAW_PER_BURST_MEAN,
    link_predictor: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ] = JENSEN_BIAS_PER_BURST_MEAN,
    min_vanilla_predictor: float = VANILLA_JENS_NOISE_FLOOR,
    interaction_ate_floor: float = 0.10,
) -> Verdict:
    """Interaction `Δ_predictor × 1[env above q_div threshold]`
    has positive ATE on Δ_target — above-threshold envs have a
    less-negative link slope (link attenuated). HELD when β_int
    ≥ `interaction_ate_floor` AND identified."""
    del treatment_arm, baseline_arm, attenuator, binary_threshold
    del link_target, link_predictor, min_vanilla_predictor
    return dowhy_backdoor_verdict(
        stratum_link_moderation_dowhy.backdoor,
        ate_threshold=interaction_ate_floor, sign=1,
    )


@claim_bridge(
    source='jensen_gap',
    target='eval_best_burst_raw_mean',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=_EXTREME_Q_DIV_SCOPE,
)
def extreme_q_div_link_placebo_refuted(
    stratum_link_moderation_dowhy: StratumLinkModerationDowhyResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    attenuator: str = 'q_divergence_score',
    binary_threshold: float = 1.0,
    link_target: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ] = MC_RETURN_RAW_PER_BURST_MEAN,
    link_predictor: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ] = JENSEN_BIAS_PER_BURST_MEAN,
    min_vanilla_predictor: float = VANILLA_JENS_NOISE_FLOOR,
    placebo_max_ratio: float = 0.2,
) -> Verdict:
    """Placebo refutation on the interaction-term ATE: random
    treatment shrinks ATE near zero. HELD when |placebo/real| <
    `placebo_max_ratio`."""
    del treatment_arm, baseline_arm, attenuator, binary_threshold
    del link_target, link_predictor, min_vanilla_predictor
    return dowhy_placebo_verdict(
        stratum_link_moderation_dowhy.placebo,
        max_ratio=placebo_max_ratio,
    )


@claim_bridge(
    source='jensen_gap',
    target='eval_best_burst_raw_mean',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=_EXTREME_Q_DIV_SCOPE,
)
def extreme_q_div_link_rcc_robust(
    stratum_link_moderation_dowhy: StratumLinkModerationDowhyResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    attenuator: str = 'q_divergence_score',
    binary_threshold: float = 1.0,
    link_target: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ] = MC_RETURN_RAW_PER_BURST_MEAN,
    link_predictor: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ] = JENSEN_BIAS_PER_BURST_MEAN,
    min_vanilla_predictor: float = VANILLA_JENS_NOISE_FLOOR,
    rcc_max_drift_ratio: float = 0.15,
) -> Verdict:
    """RCC refutation on the interaction-term ATE: synthetic
    confounder leaves it near-stable. HELD when drift <
    `rcc_max_drift_ratio`."""
    del treatment_arm, baseline_arm, attenuator, binary_threshold
    del link_target, link_predictor, min_vanilla_predictor
    return dowhy_rcc_verdict(
        stratum_link_moderation_dowhy.random_common_cause,
        max_drift_ratio=rcc_max_drift_ratio,
    )


BRIDGES = (
    acrobot_per_burst_link_active__gamma_0999,
    reach_link_backdoor_ate_negative,
    reach_link_placebo_refuted,
    reach_link_rcc_robust,
    fourrooms_action_dim_link_active__inflated,
    extreme_q_div_link_interaction_positive,
    extreme_q_div_link_placebo_refuted,
    extreme_q_div_link_rcc_robust,
)
