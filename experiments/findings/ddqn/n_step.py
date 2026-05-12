"""CLAIM 9 — n-step falsification of bootstrap-bias-compounding.

Pearl rung-2 negative-prediction probe. As n-step replaces
bootstrap with MC backup, DDQN's advantage should COLLAPSE
monotonically. Historically Δ→0: n=1 +0.087 → n=10 +0.005 (ns).
See `findings_nstep_falsification.md`.

Encoded as a PAIR:
- `ddqn_helps_at_full_bootstrap__fourrooms_n1`: HELD positive at
  full bootstrap.
- `ddqn_null_under_monte_carlo__fourrooms_n10`: HELD null at near-MC.

Verdict directionality is inverted between the two by design — the
theorem predicts smallness at n=10."""
from __future__ import annotations

import math

import polars as pl

from corroborate.analyses.arm_mean_diff import ArmMeanDiffResult
from corroborate.bridge.bridge import Direction, Tier, claim_bridge
from corroborate.bridge.verdict import Verdict

from experiments.findings.ddqn._arms import INTERVENTION


@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_mean',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=(
        (pl.col('env_name') == 'FourRooms-misc')
        & (pl.col('n_step') == 1)
    ),
)
def ddqn_helps_at_full_bootstrap__fourrooms_n1(
    arm_mean_diff: ArmMeanDiffResult,
    *,
    threshold_diff: float = 0.05,
) -> Verdict:
    """At n=1, DDQN's FR outcome benefit ≥ +0.05 with p<0.05.
    Positive baseline of the falsification curve. HELD when both
    hold. Independent-samples (Welch's t) — same-seed DDQN/vanilla
    cells diverge from step 1."""
    diff = arm_mean_diff.mean_diff
    p = arm_mean_diff.mean_diff_p_value
    if math.isnan(diff) or math.isnan(p):
        return Verdict.NO_EFFECT
    if diff < 0.0:
        return Verdict.NO_EFFECT
    significant = p < 0.05
    above_threshold = diff >= threshold_diff
    if significant and above_threshold:
        return Verdict.HELD
    if above_threshold or significant:
        return Verdict.POWER_INSUFFICIENT
    return Verdict.NO_EFFECT


@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_mean',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=(
        (pl.col('env_name') == 'FourRooms-misc')
        & (pl.col('n_step') == 10)
    ),
)
def ddqn_null_under_monte_carlo__fourrooms_n10(
    arm_mean_diff: ArmMeanDiffResult,
    *,
    null_ceiling: float = 0.02,
) -> Verdict:
    """At n=10 (near-MC, bootstrap ≈ 0), DDQN's FR outcome benefit
    ≤ +0.02 AND p > 0.05. HELD when small (the PREDICTED outcome of
    the falsification probe). Verdict mapping is inverted vs the
    n=1 bridge by design. AWAITING DATA: nstep_lambda_fourrooms
    sweep absent post-rebuild."""
    diff = arm_mean_diff.mean_diff
    p = arm_mean_diff.mean_diff_p_value
    if math.isnan(diff) or math.isnan(p):
        return Verdict.POWER_INSUFFICIENT
    is_small = abs(diff) <= null_ceiling
    is_ns = p > 0.05
    if is_small and is_ns:
        return Verdict.HELD
    if is_small or is_ns:
        return Verdict.POWER_INSUFFICIENT
    return Verdict.NO_EFFECT


BRIDGES = (
    ddqn_helps_at_full_bootstrap__fourrooms_n1,
    ddqn_null_under_monte_carlo__fourrooms_n10,
)
