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
- `effh_predicts_link_power__reach_envs` (CLAIM 19): per-env paired-g
  meta-regression of Δ_outcome on env-mean eff_h for env-level REACH
  cohort (Acrobot/FourRooms/MountainCar). n=3 underpowered for now.
- `argmax_entropy_link_power_null__survive_envs` (CLAIM 20):
  predicted-NULL companion (post-fix); env-level SURVIVE cohort
  (Asterix/Breakout/CartPole) gives r≈+0.03 — argmaxH does NOT
  predict link power, confirming the null."""
from __future__ import annotations

import math

import numpy as np
import polars as pl

from corroborate.analyses.paired_continuous_do_dowhy import (
    PairedContinuousDoResult,
)
from corroborate.analyses.stratified_arm_diff_pooled import (
    StratifiedArmDiffPooledResult,
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

from experiments.findings.ddqn._arms import DDQN_ARM, VANILLA_ARM
from experiments.findings.ddqn._scope import (
    DDQN_RELEVANT_SCOPE, LINK_POWER_REACH_ENVS, LINK_POWER_SURVIVE_ENVS,
    VANILLA_CONFIG_Q_BOUNDED,
)
from experiments.findings.ddqn._verdicts import (
    env_covariate_pearson_verdict,
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
        pl.col('env_name').is_in(['Asterix-MinAtar', 'Breakout-MinAtar'])
        & pl.col('sync_period').is_in([500, 1500, 3000])
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


# CLAIM 19 / 20 — cross-env link-power predictors. Phase-1 refactor
# (2026-05-12): replaced `meta_regression_paired_g` (per-env Hedges'
# g via seed-pairing — pseudo-replicates seeds as iid Δ-samples
# within env) with `stratified_arm_diff_pooled` (independent-samples
# Cohen's d per env — strata are the unit of inference, not cells).
# Mech conditioning via `scope_predictor='jensen_gap'`,
# `min_vanilla_predictor=0.05` — only strata where vanilla actually
# has bias > 0.05 (premise active) contribute. The bridge body
# extracts the per-env Cohen's d panel and computes Pearson r on the
# env-level covariate via `env_covariate_pearson_verdict`.
_LINK_POWER_BASE_SCOPE = (
    finite('q_divergence_score') & finite_lt('q_divergence_score', 1.0)
    & finite_gt('bootstrap_fraction', 0.5)
    & ((pl.col('n_step') == 1) | pl.col('n_step').is_null())
    & pl.col('action_duplicate_k').is_null()
    & (pl.col('reward_scale').is_null() | (pl.col('reward_scale') == 1.0))
    & pl.col('target_sync.tau').is_null()
)


# Env-level covariate values (empirical per-env means on the current
# ddqn cache, pinned). Scalar form for
# `env_covariate_pearson_verdict`. Update when data shifts.
_EFFECTIVE_HORIZON_PER_ENV: dict[str, float] = {
    'Acrobot-v1': 48.9,
    'FourRooms-misc': 27.6,
    'MountainCar-v0': 62.7,
}


_ARGMAX_ENTROPY_LATE_PER_ENV: dict[str, float] = {
    'Asterix-MinAtar': 1.18,
    'Breakout-MinAtar': 0.84,
    'CartPole-v1': 0.69,
}


@claim_bridge(
    source='effective_horizon',
    target='eval_best_burst_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=(_LINK_POWER_BASE_SCOPE & pl.col('env_name').is_in(LINK_POWER_REACH_ENVS)),
    predicted_direction='a_gt_b',
)
def effh_predicts_link_power__reach_envs(
    stratified_arm_diff_pooled: StratifiedArmDiffPooledResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    source: str = 'eval_best_burst_mean',
    stratify_by: tuple[str, ...] = ('env_name',),
    scope_predictor: str = 'jensen_gap',
    min_vanilla_predictor: float = 0.05,
    env_covariate: dict[str, float] = _EFFECTIVE_HORIZON_PER_ENV,
    r_threshold: float = 0.7,
    min_envs: int = 3,
) -> Verdict:
    """Cross-env link-power: env-level Pearson r between per-env
    independent-samples Cohen's d (DDQN vs vanilla on
    eval_best_burst_mean) and env-mean effective_horizon. REACH
    cohort (Acrobot/FourRooms/MountainCar). Mech-conditioning:
    only strata with vanilla mean jensen_gap > 0.05 (premise
    active) contribute. HELD when r ≥ `r_threshold` AND
    significant. n=3 underpowered for now; directionally r=+0.85
    on cache → POW_INSUF until more REACH envs ingest."""
    del treatment_arm, baseline_arm, source, stratify_by
    del scope_predictor, min_vanilla_predictor
    return env_covariate_pearson_verdict(
        stratified_arm_diff_pooled.per_stratum,
        env_covariate,
        sign=1,
        threshold=r_threshold,
        min_envs=min_envs,
    )


@claim_bridge(
    source='argmax_entropy_late',
    target='eval_best_burst_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=(_LINK_POWER_BASE_SCOPE & pl.col('env_name').is_in(LINK_POWER_SURVIVE_ENVS)),
    predicted_direction='null',
)
def argmax_entropy_link_power_null__survive_envs(
    stratified_arm_diff_pooled: StratifiedArmDiffPooledResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    source: str = 'eval_best_burst_mean',
    stratify_by: tuple[str, ...] = ('env_name',),
    scope_predictor: str = 'jensen_gap',
    min_vanilla_predictor: float = 0.05,
    env_covariate: dict[str, float] = _ARGMAX_ENTROPY_LATE_PER_ENV,
    null_r_ceiling: float = 0.3,
    min_envs: int = 3,
) -> Verdict:
    """Predicted-NULL form. Env-level SURVIVE cohort
    (Asterix/Breakout/CartPole). Mech-conditioning via
    `min_vanilla_predictor=0.05`. HELD when |r| < `null_r_ceiling`
    AND non-significant (null confirmed); NO_EFFECT when slope is
    significantly nonzero (null refuted). Memory's n=5 starting-
    point HELD (Pearson +0.91 in the SIGNED form) was a different
    claim shape on a different cache snapshot; the current null
    framing is the post-fix companion to CLAIM 19."""
    del treatment_arm, baseline_arm, source, stratify_by
    del scope_predictor, min_vanilla_predictor
    return env_covariate_pearson_verdict(
        stratified_arm_diff_pooled.per_stratum,
        env_covariate,
        sign=0,
        threshold=null_r_ceiling,
        min_envs=min_envs,
    )


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
    argmax_entropy_link_power_null__survive_envs,
)
