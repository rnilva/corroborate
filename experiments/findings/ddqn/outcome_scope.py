"""Cross-env scope of the outcome benefit.

Three bridges encoding the three-channel architecture:

- `ddqn_refuted_when_dormancy_fires` (CLAIM 2): necessary-scope
  refutation. On dormant-at-best-burst cells (Hasselt premise
  inactive), Δ_outcome should be ≈ 0.
- `ddqn_helps_under_three_gate_scope__cross_env` (CLAIM 26b): the
  cross-env panel HELD bridge under [G1 ∧ G2 ∧ G3] conjunction.
- `clip_wedge_predicts_outcome__polarity_moderated__dormant_scope`
  (CLAIM 3): sufficient-condition Channel-B complement on dormant
  cells via stratified partial Spearman."""
from __future__ import annotations

import math

import polars as pl

from corroborate.analyses.stratified_arm_diff_pooled import (
    StratifiedArmDiffPooledResult,
)
from corroborate.analyses.stratified_partial_spearman import (
    StratifiedPartialSpearmanResult,
)
from corroborate.bridge.bridge import Direction, Tier, claim_bridge
from corroborate.bridge.verdict import RefutationClass, Verdict

from experiments.findings.ddqn._arms import INTERVENTION


# CLAIM 2 — Necessary-scope dormancy refutation.
@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_mean',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=(
        pl.col('jensen_dormancy_gap_at_best_burst').is_finite()
        & (pl.col('jensen_dormancy_gap_at_best_burst') >= 0.05)
        & pl.col('eval_best_burst_mean').is_finite()
    ),
    predicted_direction='null',
)
def ddqn_refuted_when_dormancy_fires(
    stratified_arm_diff_pooled: StratifiedArmDiffPooledResult,
    *,
    null_ceiling: float = 0.2,
    min_strata: int = 2,
    stratify_by: tuple[str, ...] = ('env_name',),
    min_vanilla_predictor: float = float('-inf'),
) -> tuple[Verdict, RefutationClass | None]:
    """On dormant-at-best-burst cells (σ_Q × √(2 log K) − (Q − MC)
    > 0.05), Δ_outcome should be ≈ 0. Per-env Cohen's d. HELD if
    all per-env CIs in ±`null_ceiling`; INVARIANT_VIOLATION if any
    CI fully > +ceiling; NO_EFFECT (SIGN_FLIP) if any CI fully <
    −ceiling; else POW_INSUF."""
    del stratify_by, min_vanilla_predictor
    if stratified_arm_diff_pooled.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT, None
    any_above = False
    any_below = False
    any_spans = False
    n_envs_valid = 0
    for s in stratified_arm_diff_pooled.per_stratum:
        d_env = s.cohen_d
        se_env = s.cohen_se
        if math.isnan(d_env) or math.isnan(se_env):
            continue
        n_envs_valid += 1
        ci_lo = d_env - 1.96 * se_env
        ci_hi = d_env + 1.96 * se_env
        if ci_lo > null_ceiling:
            any_above = True
        elif ci_hi < -null_ceiling:
            any_below = True
        elif not (ci_lo >= -null_ceiling and ci_hi <= null_ceiling):
            any_spans = True
    if n_envs_valid < min_strata:
        return Verdict.POWER_INSUFFICIENT, None
    if any_above:
        return Verdict.INVARIANT_VIOLATION, None
    if any_below:
        return Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP
    if not any_spans:
        return Verdict.HELD, None
    return Verdict.POWER_INSUFFICIENT, None


# CLAIM 26b — three-gate scope conjunction predicts outcome benefit.
@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_mean',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=(
        # ARM-SYMMETRIC predicates only. Stratum-level filter is
        # applied INSIDE the primitive via `min_vanilla_predictor`.
        pl.col('eval_best_burst_mean').is_finite()
        & pl.col('jensen_gap').is_finite()
        & pl.col('n_actions').is_finite() & (pl.col('n_actions') >= 3)
        & ((pl.col('n_step') == 1) | pl.col('n_step').is_null())
        & pl.col('action_duplicate_k').is_null()
        & ~((pl.col('env_name') == 'MetaMaze-misc')
            & (pl.col('gamma') == 0.999))
        & (pl.col('env_name') != 'CartPole-v1')
        & (pl.col('env_name') != 'SlidingTilePuzzle-jumanji')
    ),
    predicted_direction='a_gt_b',
)
def ddqn_helps_under_three_gate_scope__cross_env(
    stratified_arm_diff_pooled: StratifiedArmDiffPooledResult,
    *,
    threshold_d: float = 0.05,
    alpha: float = 0.05,
    min_strata: int = 4,
    stratify_by: tuple[str, ...] = (
        'env_name', 'sync_period', 'gamma', 'total_steps',
    ),
    min_vanilla_predictor: float = 2.0,
) -> Verdict:
    """Cross-config independent-samples Cohen's d (Hedges 1981),
    DerSimonian-Laird random-effects pool over strata that pass the
    stratum-level G1 filter (`mean(vanilla jens) > min_vanilla_
    predictor`). Stratify by `(env, sync, γ, total_steps)`. HELD
    when pooled d > threshold, p < α, n_strata ≥ min_strata."""
    if stratified_arm_diff_pooled.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT
    d = stratified_arm_diff_pooled.pooled_d
    p = stratified_arm_diff_pooled.pooled_p_value
    if math.isnan(d) or math.isnan(p):
        return Verdict.NO_EFFECT
    if d < 0.0:
        return Verdict.NO_EFFECT
    significant = p < alpha
    above = d >= threshold_d
    if significant and above:
        return Verdict.HELD
    if above or significant:
        return Verdict.POWER_INSUFFICIENT
    return Verdict.NO_EFFECT


# CLAIM 3 — Q-clip channel sufficient-condition on dormant scope.
@claim_bridge(
    source='clip_wedge_polarity_aligned',
    target='eval_best_burst_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=(
        pl.col('jensen_gap').is_finite() & (pl.col('jensen_gap') < 0.1)
        & pl.col('clip_wedge_polarity_aligned').is_finite()
        & pl.col('env_reward_polarity').is_finite()
        & (pl.col('env_reward_polarity').abs() > 0.3)
        & pl.col('eval_best_burst_mean').is_finite()
    ),
    predicted_direction='a_gt_b',
)
def clip_wedge_predicts_outcome__polarity_moderated__dormant_scope(
    stratified_partial_spearman: StratifiedPartialSpearmanResult,
    *,
    x: str = 'clip_wedge_polarity_aligned',
    y: str = 'eval_best_burst_mean',
    conditioning: str = 'jensen_gap',
    stratify_by: str = 'env_name',
    min_stratum_size: int = 5,
    min_rho: float = 0.2,
    min_strata: int = 2,
) -> Verdict:
    """Sufficient-condition complement to CLAIM 26b: on dormant cells
    (`jensen_gap < 0.1`), the polarity-aligned clip wedge predicts
    outcome benefit after partialling out residual jens.
    `clip_wedge_polarity_aligned = ddqn_bootstrap_gap × sign(
    env_reward_polarity)` folds polarity-moderation into a single
    predictor. HELD if pooled partial-r ≥ `min_rho`."""
    del x, y, conditioning, stratify_by, min_stratum_size
    if stratified_partial_spearman.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT
    rho = stratified_partial_spearman.rho_pooled
    if math.isnan(rho):
        return Verdict.POWER_INSUFFICIENT
    if rho >= min_rho:
        return Verdict.HELD
    if rho <= -min_rho:
        return Verdict.NO_EFFECT
    return Verdict.POWER_INSUFFICIENT


BRIDGES = (
    ddqn_refuted_when_dormancy_fires,
    ddqn_helps_under_three_gate_scope__cross_env,
    clip_wedge_predicts_outcome__polarity_moderated__dormant_scope,
)
