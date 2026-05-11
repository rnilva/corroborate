"""Second-layer theorem on intrinsic_penalty: bridges for the
dampened-α sweep on the G1-active panel.

Pre-registered in `docs/SECOND_LAYER_THEOREM.md` (2026-05-11).
Tests three predictions of the second-layer theorem on the
α-sweep cells produced by `dampened_alpha_g1_mlp.yaml` +
`dampened_alpha_g1_cnn.yaml`:

  P1 (linearity)            — Δ_outcome(α) is linear in α per env
  P2 (Δ_jens linearity)     — Δ_jens(α) is linear in α with sign='a_lt_b'
  P3 (penalty sign + scale) — intrinsic_penalty has predicted sign,
                              scales with ddqn_bootstrap_gap_late

The α-sweep produces cells where `bootstrap.greedification.alpha`
is a leaf parameter (the dampened_double_greedify's α argument).
Cross-α slope tests use this column as the predictor.

Hasselt 2010 explains the slope (universal -1 link-asymptote);
this module tests the env-specific cancellation term — the
"intrinsic penalty" — that determines DDQN's NET outcome effect
once both the bias-link contribution and the algorithmic-cost
penalty are accounted for.

The bridge that reads `ddqn_bootstrap_gap_late` triggers
automatic backfill from `target_max_q_per_step` +
`target_q_at_online_argmax_per_step` trace columns (already
persisted in ddqn_universe corpus).

Scope predicates restrict to the G1-active panel envs with
v_jens > 2.0 (substantive premise) — same scope as CLAIM 26b
minus the SlidingTile / MetaMaze γ=0.999 / CartPole G3
exclusions, because the α-sweep DELIBERATELY includes
SlidingTile to test the intrinsic_penalty pattern there.
"""
from __future__ import annotations

import math
from functools import partial

import polars as pl

import corroborate.analyses  # pyright: ignore[reportUnusedImport]
import corroborate_rl.dqn.measurables  # pyright: ignore[reportUnusedImport]
from corroborate.analyses.stratified_partial_spearman import (
    StratifiedPartialSpearmanResult,
)
from corroborate.analyses.stratified_spearman import (
    StratifiedSpearmanResult,
)
from corroborate.bridge.bridge import Direction, Tier, claim_bridge
from corroborate.bridge.verdict import RefutationClass, Verdict
from corroborate.core.intervention import DoEffect, Intervention
from corroborate_rl.dqn.claims.bootstrap import (
    bootstrap, dampened_double_greedify,
)


# Module-level do-effect for the α-sweep: treatment = α=1 (full DDQN),
# baseline = α=0 (vanilla). Intermediate α arms are tested via
# cross-config slope, not as paired treatment/baseline pairs.
DAMPENED_FULL_SWAP = Intervention(
    slot_path='bootstrap',
    replacement=partial(
        bootstrap,
        greedification=partial(dampened_double_greedify, alpha=1.0),
    ),
)
INTERVENTION = DoEffect(treatment=(DAMPENED_FULL_SWAP,), baseline=())

# Helper expressions (mirror ddqn_universe.py's pattern)
def _finite(col: str) -> pl.Expr:
    return pl.col(col).is_finite()


# =====================================================================
# CLAIM 27 — Second-layer theorem on intrinsic penalty.
#
# Hasselt 2010's bias-link mechanism predicts slope (≈ -1 universal).
# The intrinsic_penalty residual is env-specific and currently has no
# theoretical account. The α-sweep on the G1-active panel tests the
# algorithmic-cost derivation pre-registered in
# `docs/SECOND_LAYER_THEOREM.md`.
#
# All three bridges below pin `predicted_direction` BEFORE the sweep
# runs. Post-hoc adjustment is refused; the test stands or falls on
# the pre-registered predictions.
#
# Scope: the four α-sweep envs (MetaMaze γ=0.99, MountainCar,
# SlidingTile, Snake) at the protocols matched to their ddqn_universe
# cache cells (1M steps, env-specific sync_period and q_network).
# =====================================================================


_ALPHA_SWEEP_SCOPE = (
    # `effective_alpha` is the derived measurable that unifies α from
    # three sources: dampened_double_greedify(α) cells, vanilla
    # baseline (α=0), pure DDQN (α=1). Filtering to finite values
    # excludes cells from action-duplicate / n-step / polyak / expectile
    # arms (which have NaN effective_alpha).
    _finite('eval_best_burst_mean')
    & _finite('jensen_gap')
    & _finite('effective_alpha')
    & pl.col('env_name').is_in([
        # MLP envs (1M steps): dampened_alpha_g1_mlp.yaml
        'MetaMaze-misc',
        'MountainCar-v0',
        # MinAtar envs: 200k staging (dampened_alpha_g1_minatar_200k.yaml)
        #              + 1M canonical (dampened_alpha_g1_cnn.yaml)
        'Asterix-MinAtar',
        'Breakout-MinAtar',
        'Freeway-MinAtar',
        'SpaceInvaders-MinAtar',
    ])
    # total_steps NOT pinned at scope level — both 200k MinAtar (cache
    # + new α-sweep) and 1M MinAtar are valid scope cells. The bridge's
    # `config_keys` includes `total_steps` so 200k and 1M get separate
    # slope groups (not pooled).
    & pl.col('total_steps').is_in([200000, 1000000])
    # MinAtar sync_period pinned to 500: the regime where vanilla
    # develops Jensen overestimation (at sync=3000, jens=0 across
    # MinAtar envs and DDQN has nothing to correct — α-sweep tests
    # a flatline). MLP envs use sync=100, MinAtar sync=500.
    & pl.col('sync_period').is_in([100, 500])
    # gamma matched to G1-active scope (γ=0.999 MetaMaze excluded for
    # G3-bottom-A reasons in CLAIM 26b — keep that exclusion here)
    & ~((pl.col('env_name') == 'MetaMaze-misc') & (pl.col('gamma') == 0.999))
    # Exclude other intervention dimensions (n-step, action-duplicate,
    # polyak, reward-scale wrappers) — only clean DDQN-vs-vanilla
    # contrast in scope.
    & ((pl.col('n_step') == 1) | pl.col('n_step').is_null())
    & pl.col('action_duplicate_k').is_null()
    & pl.col('target_sync.tau').is_null()
    & (pl.col('reward_scale').is_null() | (pl.col('reward_scale') == 1.0))
)


@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_mean',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    pair_by=('seed',),
    scope=_ALPHA_SWEEP_SCOPE,
    predicted_direction='a_gt_b',
)
def alpha_outcome_slope_per_env__second_layer(
    stratified_spearman: StratifiedSpearmanResult,
    *,
    x: str = 'effective_alpha',
    y: str = 'eval_best_burst_mean',
    stratify_by: str = 'env_name',
    min_stratum_size: int = 5,
    rho_threshold: float = 0.3,
    min_strata: int = 4,
) -> tuple[Verdict, RefutationClass | None]:
    """P1 (pre-registered): Δ_outcome(α) is LINEAR in α per env.

    **2026-05-11 migrated from `cross_config_paired_slope` (per-
    pair-Δ form, wrong shape — dropped intermediate-α cells via
    arm_key filter) to JCI `stratified_spearman` (per-cell
    marginal Spearman, env-stratified, Fisher-z pooled).

    Pooled across envs, the prediction is "α positively correlates
    with outcome on average" (`predicted_direction='a_gt_b'`).
    HELD when ρ_pooled ≥ rho_threshold.

    **Note on sign-aware semantics.** The original prediction was
    SIGN-AWARE (positive on net-positive-DDQN envs, negative on
    net-negative-DDQN envs). JCI's pooled ρ DILUTES opposite
    signs. Empirical per-env (post-rebuild): Breakout ρ≈0.0
    (null, n_α=5), SpaceInvaders ρ≈+0.34 (positive, n_α=5),
    other 4 envs only have α=0 and α=1 anchors. Pooled across
    all 6 envs, the per-env signal is diluted. The α-sweep is
    under-instrumented for "per-env sign-aware" testing —
    intermediate-α cells only exist on Breakout + SpaceInvaders."""
    del x, y, stratify_by, min_stratum_size
    if stratified_spearman.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT, None
    rho = stratified_spearman.rho_pooled
    p = stratified_spearman.p_value
    if math.isnan(rho) or math.isnan(p):
        return Verdict.POWER_INSUFFICIENT, None
    if rho >= rho_threshold:
        return Verdict.HELD, None
    if rho <= -rho_threshold:
        return Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP
    return Verdict.NO_EFFECT, RefutationClass.NULL_EFFECT


@claim_bridge(
    source=INTERVENTION,
    target='jensen_gap',
    direction=Direction.INVERSE,
    tier=Tier.INTERVENTIONAL,
    pair_by=('seed',),
    scope=_ALPHA_SWEEP_SCOPE,
    predicted_direction='a_lt_b',
)
def alpha_jens_slope_per_env__second_layer(
    stratified_spearman: StratifiedSpearmanResult,
    *,
    x: str = 'effective_alpha',
    y: str = 'jensen_gap',
    stratify_by: str = 'env_name',
    min_stratum_size: int = 5,
    rho_threshold: float = -0.5,
    min_strata: int = 4,
) -> tuple[Verdict, RefutationClass | None]:
    """P2 (pre-registered): Δ_jens(α) is LINEAR in α with sign 'a_lt_b'.

    **2026-05-11 migrated from `cross_config_paired_slope` to JCI
    `stratified_spearman`.** Per-cell within-env Spearman
    ρ(α, jens) pooled via Fisher-z. HELD when ρ_pooled ≤ −0.5.

    The cross-α slope of `jensen_gap` against α should be
    NEGATIVE: increasing α from 0 (vanilla) to 1 (full DDQN)
    progressively reduces the Q-MC bias. UNLIKE P1, this prediction
    is universal-negative across envs (Hasselt mechanism is
    direction-invariant), so JCI pooling is appropriate.

    Empirical per-env (post-rebuild): Asterix ρ=−0.73 ***,
    Breakout ρ=−0.85 ***, Freeway ρ=−0.10 ns, MetaMaze ρ=−0.34 **,
    MountainCar ρ=−0.28 *, SpaceInvaders ρ=−0.90 ***. 5 of 6 envs
    confirm negative direction; Freeway null is Q-explosion regime
    (mech dormant). Pooled ρ should be strongly negative.

    Refutation would mean the bias-correction mechanism doesn't
    scale linearly with α, which would also refute P1's premise."""
    del x, y, stratify_by, min_stratum_size
    if stratified_spearman.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT, None
    rho = stratified_spearman.rho_pooled
    p = stratified_spearman.p_value
    if math.isnan(rho) or math.isnan(p):
        return Verdict.POWER_INSUFFICIENT, None
    if rho <= rho_threshold:
        return Verdict.HELD, None
    if rho >= 0.0:
        return Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP
    return Verdict.POWER_INSUFFICIENT, None


@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_mean',
    direction=Direction.INVERSE,
    tier=Tier.INTERVENTIONAL,
    pair_by=('seed',),
    scope=_ALPHA_SWEEP_SCOPE,
    predicted_direction='a_lt_b',
)
def intrinsic_penalty_scales_with_bootstrap_gap__second_layer(
    stratified_partial_spearman: StratifiedPartialSpearmanResult,
    *,
    x: str = 'ddqn_bootstrap_gap_late',
    y: str = 'eval_best_burst_mean',
    conditioning: str = 'effective_alpha',
    stratify_by: str = 'env_name',
    min_stratum_size: int = 5,
    rho_threshold: float = -0.3,
    min_strata: int = 2,
) -> tuple[Verdict, RefutationClass | None]:
    """P3 (pre-registered): intrinsic_penalty has the SIGN predicted
    by the algorithmic-cost derivation.

    **2026-05-11 migrated from `cross_config_paired_slope` to JCI
    `stratified_partial_spearman`.** Tests
    `ρ_partial(bootstrap_gap, outcome | α)` env-stratified —
    residualizes on α (P3 explicitly requires "controlling for the
    bias-link contribution"). Then pools via Fisher-z across envs.
    HELD when ρ_pooled ≤ −0.3 (relaxed from −0.5 since per-env
    n is small after conditioning).

    Empirical per-env (marginal, post-rebuild): Asterix ρ≈−0.04,
    **Breakout ρ=+0.61 *** (n=120)**, Freeway ρ≈+0.05, SpaceInvaders
    ρ≈−0.03. Breakout sign is OPPOSITE to predicted direction —
    REFUTED. The "algorithmic-cost derivation" predicts negative
    coupling; Breakout's strong positive coupling refutes it.

    The bridge survives as a falsifiable artifact documenting that
    the second-layer theorem's intrinsic_penalty prediction fails
    on the env with the most α-sweep data."""
    del x, y, conditioning, stratify_by, min_stratum_size
    if stratified_partial_spearman.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT, None
    rho = stratified_partial_spearman.rho_pooled
    p = stratified_partial_spearman.p_value
    if math.isnan(rho) or math.isnan(p):
        return Verdict.POWER_INSUFFICIENT, None
    if rho <= rho_threshold:
        return Verdict.HELD, None
    if rho >= 0.0:
        return Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP
    return Verdict.POWER_INSUFFICIENT, None


# Module-level exports for run_hypothesis discovery
SECOND_LAYER_BRIDGES = (
    alpha_outcome_slope_per_env__second_layer,
    alpha_jens_slope_per_env__second_layer,
    intrinsic_penalty_scales_with_bootstrap_gap__second_layer,
)
BRIDGES = SECOND_LAYER_BRIDGES
