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
    # Core finite-value filters.
    _finite('eval_best_burst_mean')
    & _finite('jensen_gap')
    & _finite('effective_alpha')
    # **Endogenous sweep-coverage predicate (replaces env_name allow-
    # list).** Keep cells in envs that have ≥ 3 distinct
    # `effective_alpha` values — this is what "actually has α-sweep
    # data" means in the corpus. Envs with only α=0 / α=1 endpoints
    # contribute 2-point binary contrasts that dilute "α-linearity"
    # testing; the window predicate filters them out without naming
    # specific envs. Per `feedback_endogenous_scope_predicates.md`:
    # scope on data properties, not env names.
    & (pl.col('effective_alpha').n_unique().over('env_name') >= 3)
    # Config-level pins (NOT env names — these are intervention/
    # protocol axes). Same configuration as where the dampened-α
    # sweep ran, so the α-cells and endpoint-cells are protocol-
    # matched within env.
    & pl.col('total_steps').is_in([200000, 1000000])
    & pl.col('sync_period').is_in([100, 500])
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
    min_strata: int = 2,
) -> tuple[Verdict, RefutationClass | None]:
    """P1 (pre-registered): Δ_outcome(α) is LINEAR in α per env.

    **2026-05-11 migrated from `cross_config_paired_slope` (per-
    pair-Δ form, wrong shape — dropped intermediate-α cells via
    arm_key filter) to JCI `stratified_spearman` (per-cell
    marginal Spearman, env-stratified, Fisher-z pooled).

    Pooled across envs, the prediction is "α positively correlates
    with outcome on average" (`predicted_direction='a_gt_b'`).
    HELD when ρ_pooled ≥ rho_threshold.

    **Scope tightened 2026-05-11** via endogenous predicate
    `effective_alpha.n_unique().over('env_name') >= 3` — only
    envs with actual α-sweep coverage (≥3 distinct α levels)
    enter scope. The remaining 2 strata (Breakout + SpaceInvaders)
    are the only envs in the corpus with intermediate-α cells.

    Empirical post-rebuild (n=360, 2 strata): ρ_pooled=+0.166,
    p=0.0016 — significant tiny coupling but below the 0.3
    threshold. Sign-aware per-env: Breakout ρ≈0.0 (null),
    SpaceInvaders ρ≈+0.34 (positive). The pre-registered
    "linearity per env with sign matching env's Δ_outcome sign"
    prediction is NOT confirmable at pooled level — JCI dilutes
    opposite signs. To corroborate the sign-aware prediction
    would need a different primitive (per-env panel) or
    additional α-sweep envs."""
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
    min_strata: int = 2,
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

    **Scope tightened 2026-05-11** via the same endogenous
    predicate as P1 — only envs with ≥3 distinct α levels.

    Empirical post-rebuild (n=360, 2 strata): ρ_pooled=**−0.880**,
    p≈0, HELD. Both Breakout and SpaceInvaders strongly confirm
    the negative α→jens slope.

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
