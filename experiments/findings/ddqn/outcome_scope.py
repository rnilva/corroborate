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
from types import MappingProxyType

import polars as pl

from corroborate.analyses.cross_stratum_property_slope import (
    CrossStratumPropertySlopeResult,
)
from corroborate.analyses.stratified_arm_diff_pooled import (
    StratifiedArmDiffPooledResult,
)
from corroborate.analyses.stratified_partial_spearman import (
    StratifiedPartialSpearmanResult,
)
from corroborate.bridge.bridge import Direction, Tier, claim_bridge
from corroborate.bridge.verdict import RefutationClass, Verdict

from experiments.findings.ddqn._arms import (
    DDQN_ARM, INTERVENTION, VANILLA_ARM,
)
from experiments.findings.ddqn._verdicts import (
    cross_stratum_signed_spearman_verdict,
)


# CLAIM 2 — Necessary-scope dormancy refutation.
@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_raw_mean',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=(
        pl.col('jensen_dormancy_gap_at_best_burst').is_finite()
        & (pl.col('jensen_dormancy_gap_at_best_burst') >= 0.05)
        & pl.col('eval_best_burst_raw_mean').is_finite()
    ),
    predicted_direction='null',
)
def ddqn_refuted_when_dormancy_fires(
    stratified_arm_diff_pooled: StratifiedArmDiffPooledResult,
    *,
    null_ceiling: float = 0.2,
    min_strata: int = 2,
    stratify_by: tuple[str, ...] = ('env_name',),
    min_baseline_predictor: float = float('-inf'),
) -> tuple[Verdict, RefutationClass | None]:
    """On dormant-at-best-burst cells (σ_Q × √(2 log K) − (Q − MC)
    > 0.05), Δ_outcome should be ≈ 0. Per-env Cohen's d. HELD if
    all per-env CIs in ±`null_ceiling`; INVARIANT_VIOLATION if any
    CI fully > +ceiling; NO_EFFECT (SIGN_FLIP) if any CI fully <
    −ceiling; else POW_INSUF."""
    del stratify_by, min_baseline_predictor
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
    target='eval_best_burst_raw_mean',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=(
        # ARM-SYMMETRIC predicates only. Stratum-level filter is
        # applied INSIDE the primitive via `min_baseline_predictor`.
        pl.col('eval_best_burst_raw_mean').is_finite()
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
    min_baseline_predictor: float = 2.0,
) -> Verdict:
    """Cross-config independent-samples Cohen's d (Hedges 1981),
    DerSimonian-Laird random-effects pool over strata that pass the
    stratum-level G1 filter (`mean(vanilla jens) > min_vanilla_
    predictor`). Stratify by `(env, sync, γ, total_steps)`. HELD
    when pooled d > threshold, p < α, n_strata ≥ min_strata.

    Target is `eval_best_burst_raw_mean` (γ-invariant), per
    `findings_units_bug.md`. γ-discounted return distorts cross-env
    Cohen's d because different envs have different episode lengths
    → γ-discount scales differently per env. Diagnostic 2026-05-14:
    on the discounted target d=+0.39 p=0.078 (POWER_INSUFFICIENT,
    SpaceInvaders shows spurious d=-0.44 sign-flip); on raw target
    d=+0.46 p=0.006 (HELD, SI sign-flip vanishes to d=+0.11)."""
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
    target='eval_best_burst_raw_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=(
        pl.col('jensen_gap').is_finite() & (pl.col('jensen_gap') < 0.1)
        & pl.col('clip_wedge_polarity_aligned').is_finite()
        & pl.col('env_reward_polarity').is_finite()
        & (pl.col('env_reward_polarity').abs() > 0.3)
        & pl.col('eval_best_burst_raw_mean').is_finite()
    ),
    predicted_direction='a_gt_b',
)
def clip_wedge_predicts_outcome__polarity_moderated__dormant_scope(
    stratified_partial_spearman: StratifiedPartialSpearmanResult,
    *,
    x: str = 'clip_wedge_polarity_aligned',
    y: str = 'eval_best_burst_raw_mean',
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


# CLAIM 34 — 26b ∧ inter-state Q autocorr along eval trajectories.
# Replaces the q_autocorr_late (training-batch-mean) scope with
# q_trajectory_autocorr_late (proper inter-state Q correlation
# along actual eval trajectory). Tests whether trajectory-Q
# smoothness identifies the homogeneous-effect sub-scope on
# outcome.
@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_raw_mean',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=(
        # 26b's arm-symmetric structural scope (verbatim — preserve
        # the env-specific exclusions calibrated for G2 / G3 gates).
        pl.col('eval_best_burst_raw_mean').is_finite()
        & pl.col('jensen_gap').is_finite()
        & pl.col('n_actions').is_finite() & (pl.col('n_actions') >= 3)
        & ((pl.col('n_step') == 1) | pl.col('n_step').is_null())
        & pl.col('action_duplicate_k').is_null()
        & ~((pl.col('env_name') == 'MetaMaze-misc')
            & (pl.col('gamma') == 0.999))
        & (pl.col('env_name') != 'CartPole-v1')
        & (pl.col('env_name') != 'SlidingTilePuzzle-jumanji')
        # Inter-state α (axis i of the unified-degeneracy theory):
        # lag-1 autocorr of `predicted_q_per_step` along the
        # actual eval trajectory. Threshold 0.5 admits cells where
        # Q is reasonably smooth across consecutive trajectory
        # states (under-smooth → policy decisions don't propagate;
        # too-smooth → no discrimination). NB: this is NOT
        # `q_autocorr_late` which is the training-batch-mean
        # autocorr (confounded by network nonlinearity).
        & pl.col('q_trajectory_autocorr_late').is_finite()
        & (pl.col('q_trajectory_autocorr_late') > 0.5)
    ),
    predicted_direction='a_gt_b',
)
def ddqn_helps_under_three_gate_scope_AND_trajectory_smooth__cross_env(
    stratified_arm_diff_pooled: StratifiedArmDiffPooledResult,
    *,
    threshold_d: float = 0.05,
    alpha: float = 0.05,
    min_strata: int = 4,
    stratify_by: tuple[str, ...] = (
        'env_name', 'sync_period', 'gamma', 'total_steps',
    ),
    min_baseline_predictor: float = 2.0,
) -> Verdict:
    """26b's structural scope + inter-state Q trajectory smoothness
    (`q_trajectory_autocorr_late > 0.5`). DL pool of per-stratum
    independent-samples Cohen's d on RAW outcome.

    **What this tests:** whether spatial smoothness of Q ALONG
    ACTUAL TRAJECTORIES identifies the homogeneous-effect
    sub-scope of 26b. This is the proper empirical proxy for the
    FA-coherence theory's axis (i) — Q smoothness on the agent's
    visited states under the function approximator.

    **What this replaces:** an earlier version used
    `q_autocorr_late > 0.5`, which is the training-batch-mean
    autocorr — a confound of network nonlinearity (linear FA has
    HIGHER training-batch autocorr than deep MLP on most envs,
    OPPOSITE of theory expectation) + replay-buffer stability +
    convergence. The training-batch quantity does NOT measure
    "Q spatial smoothness along trajectories", which is what the
    theory's axis (i) is about. The empirical finding under
    `q_autocorr_late` (I² 0.66 → 0.01) is a real reduction in
    cross-stratum heterogeneity but the mechanistic attribution
    to FA-coherence was unjustified.

    `q_trajectory_autocorr_late` is computed from
    `predicted_q_per_step` along the actual eval trajectory →
    inter-state α in the substrate-author sense."""
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


# Per-env vanilla reward_nonzero_frac, computed post-ingest
# (2026-05-13) from canonical-config bare-env cells on the unified
# cache. Density tracks env's intrinsic per-step reward
# informativeness: sparse-terminal → ~0; dense per-step → ~1.
#
# Densities aggregated over (sync_period, total_steps) variants at
# γ=0.99 vanilla, wrappers=() filter. Multiple values per env
# (e.g. Asterix has sync∈{500,1000,1500}) are condensed to a single
# mean; the cross-env meta-regression is robust to small within-env
# variation because env-level density spans 0 to 1.
_REWARD_DENSITY_PER_ENV: MappingProxyType[object, MappingProxyType[str, float]] = (
    MappingProxyType({
        'Acrobot-v1':            MappingProxyType({'reward_density_vanilla': 0.991}),
        'Asterix-MinAtar':       MappingProxyType({'reward_density_vanilla': 0.022}),
        'Breakout-MinAtar':      MappingProxyType({'reward_density_vanilla': 0.072}),
        'FourRooms-misc':        MappingProxyType({'reward_density_vanilla': 0.001}),
        'MetaMaze-misc':         MappingProxyType({'reward_density_vanilla': 0.015}),
        'MountainCar-v0':        MappingProxyType({'reward_density_vanilla': 1.000}),
        'PacMan-jumanji':        MappingProxyType({'reward_density_vanilla': 0.420}),
        'Snake-jumanji':         MappingProxyType({'reward_density_vanilla': 0.0000}),
        'SpaceInvaders-MinAtar': MappingProxyType({'reward_density_vanilla': 0.108}),
    })
)


# CLAIM 35 — Link-layer: DDQN's OUTCOME benefit scales INVERSELY
# with reward density across envs. Meta-regression within 26b's
# scope on raw outcome.
@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_raw_mean',
    direction=Direction.INVERSE,
    tier=Tier.INTERVENTIONAL,
    scope=(
        pl.col('eval_best_burst_raw_mean').is_finite()
        & pl.col('jensen_gap').is_finite()
        & pl.col('n_actions').is_finite() & (pl.col('n_actions') >= 3)
        & ((pl.col('n_step') == 1) | pl.col('n_step').is_null())
        & pl.col('action_duplicate_k').is_null()
        & ~((pl.col('env_name') == 'MetaMaze-misc')
            & (pl.col('gamma') == 0.999))
        & (pl.col('env_name') != 'CartPole-v1')
        & (pl.col('env_name') != 'SlidingTilePuzzle-jumanji')
        & (pl.col('wrappers') == '()')
        & pl.col('env_name').is_in(tuple(_REWARD_DENSITY_PER_ENV.keys()))
    ),
    predicted_direction='a_gt_b',
)
def ddqn_link_outcome_scales_inversely_with_reward_density__cross_env(
    cross_stratum_property_slope: CrossStratumPropertySlopeResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    source: str = 'eval_best_burst_raw_mean',
    stratify_by: tuple[str, ...] = ('env_name',),
    covariate_name: str = 'reward_density_vanilla',
    covariate_key_field: str = 'env_name',
    covariates_per_key: MappingProxyType[object, MappingProxyType[str, float]] = (
        _REWARD_DENSITY_PER_ENV
    ),
    scope_predictor: str = 'jensen_gap',
    min_baseline_predictor: float = 2.0,
    min_seeds_per_arm: int = 5,
    rho_threshold_held: float = 0.6,
    p_threshold: float = 0.05,
    null_threshold: float = 0.2,
    sign_flip_threshold: float = 0.5,
    min_strata: int = 10,
) -> tuple[Verdict, RefutationClass | None]:
    """Cross-env Spearman ρ between `reward_density_vanilla` and
    per-env Cohen's d on raw outcome.

    **Predicted direction**: ρ < 0 — denser per-step reward →
    smaller DDQN outcome benefit. The theory's axis (iii) operates
    at the LINK layer: vanilla learns adequately when reward
    signal is rich; DDQN's de-biasing has less to translate.

    **Calibrated for n_strata≥10**. At canonical the bridge admits
    ≤7 envs after G1 gating (`min_baseline_predictor=2.0`) — below
    the resolution band of either Spearman or meta-regression at
    canonical's env count. Fires POWER_INSUFFICIENT honestly.

    Small-n sibling of `meta_regression_unpaired_d` on the same
    panel; neither resolves canonical-scope at n≤10. The
    pre-canonical meta-regression diagnostic (HP-mixed pool, n=15)
    gave β=−0.66 p=0.016 but doesn't transfer to canonical
    (see `findings_canonical_scope_reverification`).

    Verdict matrix (per `cross_stratum_signed_spearman_verdict`):
      HELD                  : ρ ≤ −0.6 AND p ≤ 0.05 (binding gate)
      NO_EFFECT (SIGN_FLIP) : ρ ≥ +0.5 (decisive wrong-direction)
      NO_EFFECT (NULL_EFFECT) : |ρ| < 0.2 (calibrated for n≥10)
      POWER_INSUFFICIENT    : in-between, or n_strata < 10"""
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


BRIDGES = (
    ddqn_refuted_when_dormancy_fires,
    ddqn_helps_under_three_gate_scope__cross_env,
    ddqn_helps_under_three_gate_scope_AND_trajectory_smooth__cross_env,
    ddqn_link_outcome_scales_inversely_with_reward_density__cross_env,
    clip_wedge_predicts_outcome__polarity_moderated__dormant_scope,
)
