"""CLAIM 7 — DDQN's under-learning rescue on FourRooms.

DDQN dominates vanilla's reward-scale response curve at rs ∈
[0.03, 0.3] on FR. Peak Δ ≈ +0.50 native at rs=0.3. Hasselt-floor
reading: DDQN's reduced ε lets it learn at smaller σ_Q. NOT a
√(log|A|) law — rescue regime is FourRooms-specific.

- `__fourrooms_rs_0p1` (CLAIM 7): primary HELD bridge at rs=0.1.
- `__fourrooms_rs_0p3` (CLAIM 7b): rescue-regime peak at rs=0.3.
- `__acrobot_rs_0p1` / `__cartpole_rs_0p1` (CLAIM 7c/7d): null
  bridges — rescue is FR-specific.
- `__fourrooms_rs_0p1` argmax_entropy (CLAIM 7e): rescue mechanism
  probe — DDQN's policy actually SHARPENS (REFUTED via SIGN_FLIP).
- `__fourrooms_rs_1p0` argmax_entropy (CLAIM 7f): standard regime
  null — argmax-sharpening is reward-scale-invariant (REFUTED via
  SIGN_FLIP)."""
from __future__ import annotations

import math

import polars as pl

from corroborate.analyses.arm_mean_diff import ArmMeanDiffResult
from corroborate.bridge.bridge import Direction, Tier, claim_bridge
from corroborate.bridge.verdict import RefutationClass, Verdict

from experiments.findings.ddqn._arms import INTERVENTION
from experiments.findings.ddqn._verdicts import (
    native_diff_ci_verdict, native_diff_null_verdict, rescue_threshold,
)


@claim_bridge(
    source=INTERVENTION,
    target='outcome_native',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=(
        (pl.col('env_name') == 'FourRooms-misc')
        & (pl.col('reward_scale') == 0.1)
    ),
    predicted_direction='a_gt_b',
)
def ddqn_rescues_underlearning_vanilla__fourrooms_rs_0p1(
    arm_mean_diff: ArmMeanDiffResult,
    *,
    threshold_diff: float = rescue_threshold(),
) -> Verdict:
    """Pearl rung-2 do(arm=ddqn) on FR rs=0.1: DDQN's native mean_diff
    closes ≥ 0.5 × (0.8 − 0.1) = 0.35 of the failure-to-optimal gap.
    HELD when widened-CI lower ≥ threshold. Empirical: md=+0.638,
    CI=[+0.594, +0.682]."""
    return native_diff_ci_verdict(
        arm_mean_diff.mean_diff, arm_mean_diff.mean_diff_se,
        threshold_diff,
    )


# CLAIM 7b — rescue-regime peak at rs=0.3.
@claim_bridge(
    source=INTERVENTION,
    target='outcome_native',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=(
        (pl.col('env_name') == 'FourRooms-misc')
        & (pl.col('reward_scale') == 0.3)
    ),
    predicted_direction='a_gt_b',
)
def ddqn_dominates_vanilla_response_curve__fourrooms_rs_0p3(
    arm_mean_diff: ArmMeanDiffResult,
    *,
    threshold_diff: float = rescue_threshold(),
) -> Verdict:
    """Sibling of CLAIM 7 at the rescue-regime peak (rs=0.3). Same
    +0.35 threshold. Currently REFUTED on postfix corpus: md=+0.259
    CI=[+0.169, +0.349] — rs=0.3 sits on upper edge where vanilla
    recovers."""
    return native_diff_ci_verdict(
        arm_mean_diff.mean_diff, arm_mean_diff.mean_diff_se,
        threshold_diff,
    )


# CLAIM 7c / 7d — rescue is FR-specific.
@claim_bridge(
    source=INTERVENTION,
    target='outcome_native',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=(
        (pl.col('env_name') == 'Acrobot-v1')
        & (pl.col('reward_scale') == 0.1)
        & ((pl.col('n_step') == 1) | pl.col('n_step').is_null())
    ),
    predicted_direction='null',
)
def ddqn_does_not_rescue__acrobot_rs_0p1(
    arm_mean_diff: ArmMeanDiffResult,
    *,
    null_ceiling: float = 0.2,
) -> Verdict:
    """Acrobot rs=0.1 null bridge: rescue does NOT activate (FR-
    specific). HELD when native CI ⊂ ±null_ceiling."""
    return native_diff_null_verdict(
        arm_mean_diff.mean_diff, arm_mean_diff.mean_diff_se,
        null_ceiling,
    )


@claim_bridge(
    source=INTERVENTION,
    target='outcome_native',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=(
        (pl.col('env_name') == 'CartPole-v1')
        & (pl.col('reward_scale') == 0.1)
        & ((pl.col('n_step') == 1) | pl.col('n_step').is_null())
    ),
    predicted_direction='null',
)
def ddqn_does_not_rescue__cartpole_rs_0p1(
    arm_mean_diff: ArmMeanDiffResult,
    *,
    null_ceiling: float = 0.2,
) -> Verdict:
    """CartPole rs=0.1 sister to Acrobot null bridge."""
    return native_diff_null_verdict(
        arm_mean_diff.mean_diff, arm_mean_diff.mean_diff_se,
        null_ceiling,
    )


# CLAIM 7e / 7f — rescue mechanism is action-selection-level.
@claim_bridge(
    source=INTERVENTION,
    target='argmax_entropy_late',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=(
        (pl.col('env_name') == 'FourRooms-misc')
        & (pl.col('reward_scale') == 0.1)
        & ((pl.col('n_step') == 1) | pl.col('n_step').is_null())
    ),
    predicted_direction='a_gt_b',
)
def ddqn_increases_argmax_entropy__fourrooms_rs_0p1(
    arm_mean_diff: ArmMeanDiffResult,
    *,
    threshold_diff: float = 0.05,
) -> tuple[Verdict, RefutationClass | None]:
    """FR rs=0.1 rescue: predicted DDQN entropy higher than vanilla's.
    REFUTED via SIGN_FLIP — actually SUBSTANTIALLY LOWER (mean_diff=
    -0.232 nats, p=2.6e-12). Rescue is "policy sharpens after
    learning unblocks", not "exploration maintained"."""
    diff = arm_mean_diff.mean_diff
    p = arm_mean_diff.mean_diff_p_value
    if math.isnan(diff) or math.isnan(p):
        return Verdict.POWER_INSUFFICIENT, None
    if diff < 0.0:
        return Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP
    significant = p < 0.05
    above_threshold = diff >= threshold_diff
    if significant and above_threshold:
        return Verdict.HELD, None
    if above_threshold or significant:
        return Verdict.POWER_INSUFFICIENT, RefutationClass.UNDERPOWERED
    return Verdict.NO_EFFECT, RefutationClass.NULL_EFFECT


@claim_bridge(
    source=INTERVENTION,
    target='argmax_entropy_late',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=(
        (pl.col('env_name') == 'FourRooms-misc')
        & (pl.col('reward_scale') == 1.0)
        & ((pl.col('n_step') == 1) | pl.col('n_step').is_null())
    ),
    predicted_direction='null',
)
def ddqn_entropy_matches_vanilla__fourrooms_rs_1p0(
    arm_mean_diff: ArmMeanDiffResult,
    *,
    null_ceiling: float = 0.05,
) -> tuple[Verdict, RefutationClass | None]:
    """FR rs=1.0 standard regime: argmax-sharpening is reward-scale-
    invariant — DDQN STILL has lower argmaxH than vanilla (mean_diff=
    -0.099). REFUTED via SIGN_FLIP."""
    diff = arm_mean_diff.mean_diff
    p = arm_mean_diff.mean_diff_p_value
    if math.isnan(diff) or math.isnan(p):
        return Verdict.POWER_INSUFFICIENT, None
    is_small = abs(diff) <= null_ceiling
    is_ns = p > 0.05
    if is_small and is_ns:
        return Verdict.HELD, None
    if is_small or is_ns:
        return Verdict.POWER_INSUFFICIENT, RefutationClass.UNDERPOWERED
    return Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP


BRIDGES = (
    ddqn_rescues_underlearning_vanilla__fourrooms_rs_0p1,
    ddqn_dominates_vanilla_response_curve__fourrooms_rs_0p3,
    ddqn_does_not_rescue__acrobot_rs_0p1,
    ddqn_does_not_rescue__cartpole_rs_0p1,
    ddqn_increases_argmax_entropy__fourrooms_rs_0p1,
    ddqn_entropy_matches_vanilla__fourrooms_rs_1p0,
)
