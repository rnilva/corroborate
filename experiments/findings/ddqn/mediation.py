"""Mediator probes — what carries Δ_outcome after Δ_jens.

10 bridges across 4 candidate mediators:

- `q_divergence_shadowed_by_jens` / `argmax_entropy_shadowed_by_jens`
  (CLAIM 23): null-form bridges. q_div = jens × per-env-constant
  mathematically; argmaxH co-varies via shared Q-distribution.
  Partial-Spearman | Δ_jens should collapse them.
- `eff_h_mediates_g_link__{goal,survival}_envs` (CLAIM 12): polarity-
  stratified eff_h mediator. GOAL polarity ρ_partial ≤ -0.3; SURVIVAL
  ρ_partial ≥ +0.3 (polarity-tautology sign).
- `target_staleness_late_mediates_outcome__minatar_intermediate_sync`
  (CLAIM 13): partial-Spearman | jens for staleness as non-eff_h
  mediator. Within-cell breaks at scope (ρ≈-0.07).
- `cross_config_staleness_slope_negative__survive` (CLAIM 21): cross-
  config descriptive ρ on (Δ_stale, Δ_outcome) for SURVIVE polarity;
  ρ=-0.9 (n=5).
- `staleness_amplifies_ddqn_outcome__sparse_goal_polyak` (CLAIM 15)
  and `staleness_does_not_amplify_ddqn_outcome__survival_polyak`
  (CLAIM 15b): Polyak-do(τ) causal corroborations. AWAITING DATA.
- `effh_predicts_link_power__reach_envs` (CLAIM 19): per-burst meta-
  regression of Δ_outcome on env-mean eff_h for REACH polarity.
- `argmax_entropy_predicts_link_power__survive_envs` (CLAIM 20):
  STARTING-POINT SURVIVE companion to CLAIM 19. n=5 small."""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import numpy.typing as npt
import polars as pl

from corroborate.analyses.paired_continuous_do_dowhy import (
    PairedContinuousDoResult,
)
from corroborate.analyses.stratified_partial_spearman import (
    StratifiedPartialSpearmanResult,
)
from corroborate.analyses.stratum_effect_panel import StratumEffectPanel
from corroborate.bridge.bridge import Direction, Tier, claim_bridge
from corroborate.bridge.predicates import (
    finite, finite_gt, finite_lt,
)
from corroborate.bridge.verdict import RefutationClass, Verdict
from corroborate.measurables import Measurable
from corroborate.stats import MetaRegressionResult

from experiments.findings.ddqn._arms import DDQN_ARM, VANILLA_ARM
from experiments.findings.ddqn._common import MC_RETURN_PER_BURST_MEAN
from experiments.findings.ddqn._scope import (
    DDQN_RELEVANT_SCOPE, VANILLA_CONFIG_Q_BOUNDED,
)
from experiments.findings.ddqn._verdicts import (
    partial_spearman_null_verdict, partial_spearman_signed_verdict,
)


# CLAIM 23 — Δ_jens shadow tests.
@claim_bridge(
    source='q_divergence_score',
    target='eval_best_burst_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=DDQN_RELEVANT_SCOPE,
    predicted_direction='null',
)
def q_divergence_shadowed_by_jens(
    stratified_partial_spearman: StratifiedPartialSpearmanResult,
    *,
    x: str = 'q_divergence_score',
    y: str = 'eval_best_burst_mean',
    conditioning: str = 'jensen_gap',
    stratify_by: str = 'env_name',
    min_stratum_size: int = 5,
    null_max_abs_rho: float = 0.2,
    min_strata: int = 2,
) -> Verdict:
    """`ρ_partial(qdiv, outcome | jens)` env-stratified Fisher-z
    pooled. HELD (null confirmed) when |ρ| < `null_max_abs_rho`."""
    del x, y, conditioning, stratify_by, min_stratum_size
    return partial_spearman_null_verdict(
        stratified_partial_spearman,
        max_abs_rho=null_max_abs_rho, min_strata=min_strata,
    )


@claim_bridge(
    source='argmax_entropy_late',
    target='eval_best_burst_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=DDQN_RELEVANT_SCOPE,
    predicted_direction='null',
)
def argmax_entropy_shadowed_by_jens(
    stratified_partial_spearman: StratifiedPartialSpearmanResult,
    *,
    x: str = 'argmax_entropy_late',
    y: str = 'eval_best_burst_mean',
    conditioning: str = 'jensen_gap',
    stratify_by: str = 'env_name',
    min_stratum_size: int = 5,
    null_max_abs_rho: float = 0.2,
    min_strata: int = 2,
) -> Verdict:
    """`ρ_partial(argmax_entropy_late, outcome | jens)`. HELD
    (null confirmed) when |ρ| < `null_max_abs_rho`."""
    del x, y, conditioning, stratify_by, min_stratum_size
    return partial_spearman_null_verdict(
        stratified_partial_spearman,
        max_abs_rho=null_max_abs_rho, min_strata=min_strata,
    )


# CLAIM 12 — env-polarity moderates the eff_h mediator sign.
@claim_bridge(
    source='effective_horizon',
    target='eval_best_burst_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    pair_by=('env_name', 'corpus', 'gamma', 'total_steps', 'sync_period', 'seed'),
    scope=(
        finite_lt('env_reward_polarity', -0.3)
        & (
            pl.col('q_divergence_score').is_nan()
            | finite_lt('q_divergence_score', 1000.0)
        )
    ),
    predicted_direction='a_lt_b',
)
def eff_h_mediates_g_link__goal_envs(
    stratified_partial_spearman: StratifiedPartialSpearmanResult,
    *,
    x: str = 'effective_horizon',
    y: str = 'eval_best_burst_mean',
    conditioning: str = 'jensen_gap',
    stratify_by: str = 'env_name',
    min_stratum_size: int = 5,
    magnitude_threshold: float = 0.3,
    min_strata: int = 2,
) -> Verdict:
    """GOAL-polarity (`polarity < -0.3`). HELD when ρ ≤ -threshold."""
    del x, y, conditioning, stratify_by, min_stratum_size
    return partial_spearman_signed_verdict(
        stratified_partial_spearman,
        threshold=magnitude_threshold, sign=-1, min_strata=min_strata,
    )


@claim_bridge(
    source='effective_horizon',
    target='eval_best_burst_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    pair_by=('env_name', 'corpus', 'gamma', 'total_steps', 'sync_period', 'seed'),
    scope=(
        finite_gt('env_reward_polarity', 0.3)
        & (
            pl.col('q_divergence_score').is_nan()
            | finite_lt('q_divergence_score', 1000.0)
        )
    ),
    predicted_direction='a_gt_b',
)
def eff_h_mediates_g_link__survival_envs(
    stratified_partial_spearman: StratifiedPartialSpearmanResult,
    *,
    x: str = 'effective_horizon',
    y: str = 'eval_best_burst_mean',
    conditioning: str = 'jensen_gap',
    stratify_by: str = 'env_name',
    min_stratum_size: int = 5,
    magnitude_threshold: float = 0.3,
    min_strata: int = 2,
) -> Verdict:
    """SURVIVAL-polarity (`polarity > +0.3`). HELD when ρ ≥
    threshold (polarity-tautology sign)."""
    del x, y, conditioning, stratify_by, min_stratum_size
    return partial_spearman_signed_verdict(
        stratified_partial_spearman,
        threshold=magnitude_threshold, sign=+1, min_strata=min_strata,
    )


# CLAIM 13 — target_staleness_late mediator chain.
@claim_bridge(
    source='target_staleness_late',
    target='eval_best_burst_mean',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=(
        pl.col('corpus').is_in(
            ['asterix_postfix_chunk10',
             'survive_sync_intermediate_minatar_postfix'],
        )
        & finite('target_staleness_late')
        & finite('jensen_gap')
        & finite('eval_best_burst_mean')
        & VANILLA_CONFIG_Q_BOUNDED
    ),
    predicted_direction='a_lt_b',
)
def target_staleness_late_mediates_outcome__minatar_intermediate_sync(
    stratified_partial_spearman: StratifiedPartialSpearmanResult,
    *,
    x: str = 'target_staleness_late',
    y: str = 'eval_best_burst_mean',
    conditioning: str = 'jensen_gap',
    stratify_by: str = 'env_name',
    min_stratum_size: int = 10,
    magnitude_threshold: float = 0.2,
    min_strata: int = 2,
) -> Verdict:
    """`ρ_partial(staleness_late, outcome | jens)` env-stratified on
    MinAtar intermediate-sync SURVIVE cohort. HELD when ρ ≤
    -magnitude_threshold (predicted negative)."""
    del x, y, conditioning, stratify_by, min_stratum_size
    return partial_spearman_signed_verdict(
        stratified_partial_spearman,
        threshold=magnitude_threshold, sign=-1, min_strata=min_strata,
    )


# CLAIM 21 — Polarity-stratified cross-config staleness slope.
@claim_bridge(
    source='target_staleness_late',
    target='eval_best_burst_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=(
        finite('q_divergence_score') & finite_lt('q_divergence_score', 1.0)
        & finite('jensen_dormancy_gap') & finite_lt('jensen_dormancy_gap', 0.05)
        & finite('env_reward_polarity')
        & finite_gt('env_reward_polarity', 0.3)
        & finite('target_staleness_late')
        & ((pl.col('n_step') == 1) | pl.col('n_step').is_null())
        & pl.col('action_duplicate_k').is_null()
        & (pl.col('reward_scale').is_null() | (pl.col('reward_scale') == 1.0))
        & pl.col('target_sync.tau').is_null()
    ),
    predicted_direction='a_lt_b',
)
def cross_config_staleness_slope_negative__survive(
    stratum_effect_panel: StratumEffectPanel,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    measurables: tuple[str, ...] = (
        'target_staleness_late', 'eval_best_burst_mean',
    ),
    stratify_by: tuple[str, ...] = (
        'env_name', 'sync_period', 'total_steps', 'corpus',
    ),
    min_seeds_per_arm: int = 5,
    rho_threshold: float = -0.5,
    p_threshold: float = 0.1,
    min_strata: int = 3,
) -> tuple[Verdict, RefutationClass | None]:
    """Cross-config Spearman on per-config (Δ_stale, Δ_outcome) for
    SURVIVE polarity. HELD when ρ ≤ threshold AND p ≤ p_threshold
    AND n ≥ min_strata. Empirical n=5: ρ=-0.90, p=0.037."""
    del treatment_arm, baseline_arm, measurables, stratify_by, min_seeds_per_arm
    panel = stratum_effect_panel
    if panel.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT, None
    d_pred = panel.deltas.get('target_staleness_late', ())
    d_target = panel.deltas.get('eval_best_burst_mean', ())
    valid = [
        (p_, t_) for p_, t_ in zip(d_pred, d_target, strict=True)
        if not (math.isnan(p_) or math.isnan(t_))
    ]
    if len(valid) < min_strata:
        return Verdict.POWER_INSUFFICIENT, None
    from scipy.stats import spearmanr as _spearmanr
    pred_arr = np.asarray([p_ for p_, _ in valid], dtype=np.float64)
    target_arr = np.asarray([t_ for _, t_ in valid], dtype=np.float64)
    rho_v, p_v = _spearmanr(pred_arr, target_arr)
    rho = float(rho_v)
    p = float(p_v)
    if math.isnan(rho) or math.isnan(p):
        return Verdict.POWER_INSUFFICIENT, None
    if rho <= rho_threshold and p <= p_threshold:
        return Verdict.HELD, None
    if rho > 0.0:
        return Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP
    return Verdict.NO_EFFECT, RefutationClass.NULL_EFFECT


# CLAIM 15 / 15b — Polyak-do(τ) corroboration of staleness causality.
_POLYAK_GOAL_SCOPE = (
    finite('target_sync.tau')
    & (pl.col('target_sync.tau') > 0)
    & finite_lt('env_reward_polarity', -0.5)
    & finite('q_divergence_score')
    & finite_lt('q_divergence_score', 100.0)
    & finite('target_staleness_late')
    & finite('eval_best_burst_mean')
    & finite_gt('q_late_mean', 0.0)
)

_POLYAK_SURVIVE_SCOPE = (
    finite('target_sync.tau')
    & (pl.col('target_sync.tau') > 0)
    & finite_gt('env_reward_polarity', 0.3)
    & finite('target_staleness_late')
    & finite('eval_best_burst_mean')
)


@claim_bridge(
    source='target_staleness_late',
    target='eval_best_burst_mean',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    pair_by=(
        'env_name', 'gamma', 'sync_period',
        'total_steps', 'seed', 'target_sync.tau',
    ),
    scope=_POLYAK_GOAL_SCOPE,
    predicted_direction='a_lt_b',
)
def staleness_amplifies_ddqn_outcome__sparse_goal_polyak(
    paired_continuous_do_dowhy: PairedContinuousDoResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    treatment_var: str = 'target_staleness_late',
    treatment_var_arm: str = 'baseline',
    outcome: str = 'eval_best_burst_mean',
    ate_threshold: float = 1.0,
    refutation_drift_threshold: float = 5.0,
    n_pairs_floor: int = 30,
) -> Verdict:
    """Polyak-do(τ) on GOAL polarity: per-pair baseline target
    staleness causally amplifies DDQN's outcome benefit. HELD if
    identified ∧ ATE > threshold ∧ refutations clean. Historical:
    FR n=120, ATE≈+5. AWAITING DATA (polyak_tau_intervention
    absent post-rebuild)."""
    del treatment_arm, baseline_arm, treatment_var, treatment_var_arm, outcome
    result = paired_continuous_do_dowhy
    if not result.backdoor.identified:
        return Verdict.POWER_INSUFFICIENT
    if result.n_pairs < n_pairs_floor:
        return Verdict.POWER_INSUFFICIENT
    if math.isnan(result.backdoor.ate):
        return Verdict.POWER_INSUFFICIENT
    if result.backdoor.ate <= ate_threshold:
        return Verdict.NO_EFFECT
    if (
        not math.isnan(result.placebo.refuted_ate)
        and abs(result.placebo.refuted_ate) > refutation_drift_threshold
    ):
        return Verdict.POWER_INSUFFICIENT
    if (
        not math.isnan(result.random_common_cause.drift)
        and result.random_common_cause.drift > refutation_drift_threshold
    ):
        return Verdict.POWER_INSUFFICIENT
    return Verdict.HELD


@claim_bridge(
    source='target_staleness_late',
    target='eval_best_burst_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    pair_by=(
        'env_name', 'gamma', 'sync_period',
        'total_steps', 'seed', 'target_sync.tau',
    ),
    scope=_POLYAK_SURVIVE_SCOPE,
    predicted_direction='null',
)
def staleness_does_not_amplify_ddqn_outcome__survival_polyak(
    paired_continuous_do_dowhy: PairedContinuousDoResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    treatment_var: str = 'target_staleness_late',
    treatment_var_arm: str = 'baseline',
    outcome: str = 'eval_best_burst_mean',
    null_band: float = 5.0,
    n_pairs_floor: int = 30,
) -> Verdict:
    """SURVIVAL-polarity companion. Null-form HELD when |ATE| <
    null_band AND identified AND n ≥ floor. Staleness mediation
    chain BREAKS on SURVIVE polarity. AWAITING DATA."""
    del treatment_arm, baseline_arm, treatment_var, treatment_var_arm, outcome
    result = paired_continuous_do_dowhy
    if not result.backdoor.identified:
        return Verdict.POWER_INSUFFICIENT
    if result.n_pairs < n_pairs_floor:
        return Verdict.POWER_INSUFFICIENT
    if math.isnan(result.backdoor.ate):
        return Verdict.POWER_INSUFFICIENT
    if abs(result.backdoor.ate) < null_band:
        return Verdict.HELD
    return Verdict.NO_EFFECT


# CLAIM 19 / 20 — cross-env link-power predictors.
_LINK_POWER_SCOPE_COMMON = (
    finite('q_divergence_score') & finite_lt('q_divergence_score', 1.0)
    & finite_gt('bootstrap_fraction', 0.5)
    & finite('jensen_dormancy_gap') & finite_lt('jensen_dormancy_gap', 0.05)
    & finite('env_reward_polarity')
    & ((pl.col('n_step') == 1) | pl.col('n_step').is_null())
    & pl.col('action_duplicate_k').is_null()
    & (pl.col('reward_scale').is_null() | (pl.col('reward_scale') == 1.0))
    & pl.col('target_sync.tau').is_null()
)


@claim_bridge(
    source='effective_horizon',
    target='eval_best_burst_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    pair_by=('seed', 'total_steps', 'eval_every'),
    scope=(_LINK_POWER_SCOPE_COMMON & finite_lt('env_reward_polarity', -0.3)),
    predicted_direction='a_gt_b',
)
def effh_predicts_link_power__reach_envs(
    meta_regression_per_burst: MetaRegressionResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    source: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ] = MC_RETURN_PER_BURST_MEAN,
    covariates: tuple[str, ...] = ('effective_horizon',),
    dedupe_strategy: str = 'mean',
    slope_threshold: float = 0.005,
) -> Verdict:
    """Per-(env, burst) meta-regression of Δ_outcome on env-mean
    effective_horizon, REACH polarity. HELD when β ≥ threshold AND
    significant. Currently NO_EFFECT: β=-0.0046, p=0.041 — opposite
    direction (per-burst slope flips vs env-mean aggregate due to
    phase-structure inversion)."""
    del treatment_arm, baseline_arm, source, covariates, dedupe_strategy
    coef = next(
        (c for c in meta_regression_per_burst.coefficients
         if c.name == 'effective_horizon'),
        None,
    )
    if coef is None:
        return Verdict.POWER_INSUFFICIENT
    if not coef.is_significant:
        return Verdict.POWER_INSUFFICIENT
    if coef.coefficient >= slope_threshold:
        return Verdict.HELD
    return Verdict.NO_EFFECT


@claim_bridge(
    source='argmax_entropy_late',
    target='eval_best_burst_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    pair_by=('seed', 'total_steps', 'eval_every'),
    scope=(_LINK_POWER_SCOPE_COMMON & finite_gt('env_reward_polarity', 0.3)),
    predicted_direction='a_gt_b',
)
def argmax_entropy_predicts_link_power__survive_envs(
    meta_regression_per_burst: MetaRegressionResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    source: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ] = MC_RETURN_PER_BURST_MEAN,
    covariates: tuple[str, ...] = ('argmax_entropy_late',),
    dedupe_strategy: str = 'mean',
    slope_threshold: float = 0.5,
) -> Verdict:
    """STARTING-POINT SURVIVE companion to CLAIM 19. Per-env paired-
    g regressed on env-mean argmax_entropy_late. HELD when β ≥
    threshold AND significant. Caveats: argmaxH is mostly env-
    structural (van↔dd Pearson +0.95); n=5 small."""
    del treatment_arm, baseline_arm, source, covariates, dedupe_strategy
    coef = next(
        (c for c in meta_regression_per_burst.coefficients
         if c.name == 'argmax_entropy_late'),
        None,
    )
    if coef is None:
        return Verdict.POWER_INSUFFICIENT
    if not coef.is_significant:
        return Verdict.POWER_INSUFFICIENT
    if coef.coefficient >= slope_threshold:
        return Verdict.HELD
    return Verdict.NO_EFFECT


BRIDGES = (
    q_divergence_shadowed_by_jens,
    argmax_entropy_shadowed_by_jens,
    eff_h_mediates_g_link__goal_envs,
    eff_h_mediates_g_link__survival_envs,
    target_staleness_late_mediates_outcome__minatar_intermediate_sync,
    cross_config_staleness_slope_negative__survive,
    staleness_amplifies_ddqn_outcome__sparse_goal_polyak,
    staleness_does_not_amplify_ddqn_outcome__survival_polyak,
    effh_predicts_link_power__reach_envs,
    argmax_entropy_predicts_link_power__survive_envs,
)
