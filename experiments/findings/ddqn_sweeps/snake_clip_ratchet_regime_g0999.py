"""Snake γ=0.999 cross-γ confirmatory pre-registration.

Pre-registered 2026-05-18 (commit hash documented at fold-in time)
BEFORE the T3a panel-extension sweep
(`g0999_panel_extension_jumanji.yaml`) lands Snake γ=0.999 cells.

Background. `finding_snake_clip_ratchet_regime` SUPPORTED Snake's
CLIP-RATCHET regime at γ=0.99. Bridge 2 was substituted post-hoc
(`q_late_mean` d=+0.20 NS → `q_action_std_late` d=+0.65 sig)
because the bimodal seed distribution makes mean Q a low-power
test statistic. The σ_Q form is substantively cleaner but the
substitution is post-hoc. THIS module pre-registers the same
3-bridge cluster at γ=0.999 — the σ_Q-inflation bridge now lands
as a genuinely-pre-registered confirmatory test.

Predicted (γ=0.999 vs γ=0.99 at canonical CNN HPs, n=30):

  Bridge 1 (PC `arm — q_max_temporal_cv_late` edge present):
    HELD if Snake's clip-ratchet failure is γ-portable. DRIFT to
    POWER_INSUFFICIENT or absent edge → CLIP-RATCHET is γ=0.99
    specific (4-regime classifier is regime-detector at the γ
    level, not env level).

  Bridge 2 (σ_Q d ≥ +0.4 sig at α=0.05): HELD if cross-action SD
    inflation is γ-portable. This is the load-bearing
    pre-registered bridge — if it lands HELD at γ=0.999 the
    post-hoc substitution at γ=0.99 has a confirmatory cell.

  Bridge 3 (PC `arm ⫫ outcome` marginal): HELD if bimodal seed
    distribution dominates outcome (Cohen's d on outcome NS).
    DRIFT to non-marginal → outcome is no longer bimodal at
    γ=0.999 (e.g., DDQN harm dominates or DDQN help emerges).

Walk-back conditions. If 2+ bridges DRIFT to non-HELD at γ=0.999,
the CLIP-RATCHET regime is γ-dependent at the env level — Snake
reclassifies into one of the existing 3 bins (likely Q-EXPLODED
given γ→1 typically pushes vanilla into Q-explosion territory).
This would be a framework-mechanic-caught walk-back of T2a's
SUPPORTED claim and a paper-grade demonstration of the DRIFT
mechanic on a high-stakes prediction.
"""
from __future__ import annotations

import math

import polars as pl

from corroborate.analyses.pc_discovery import PCDiscoveryResult
from corroborate.analyses.panel.stratified_arm_diff_pooled import (
    StratifiedArmDiffPooledResult,
)
from corroborate.bridge.bridge import Direction, Tier, claim_bridge
from corroborate.bridge.verdict import RefutationClass, Verdict

from experiments.findings.ddqn._arms import (
    DDQN_ARM, INTERVENTION, VANILLA_ARM,
)


_SNAKE_G0999_SCOPE: pl.Expr = (
    (pl.col('env_name') == 'Snake-jumanji')
    & (pl.col('gamma') == 0.999)
    & (pl.col('total_steps') == 1_000_000)
    & (pl.col('q_network.hidden') == '(64)')
)


_ARM_INDICATOR: dict[str, tuple[str, str]] = {
    'arm_ddqn_indicator': ('arm_key', DDQN_ARM),
}


_NODES: tuple[str, ...] = (
    'arm_ddqn_indicator',
    'jensen_gap',
    'q_late_mean',
    'q_action_std_late',
    'q_argmax_margin_late',
    'q_trajectory_autocorr_late',
    'q_max_temporal_cv_late',
    'eval_best_burst_raw_mean',
)


# ============ Bridge 1: arm — q_max_temporal_cv edge in PC skeleton ============

@claim_bridge(
    source=INTERVENTION,
    target='q_max_temporal_cv_late',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=(
        _SNAKE_G0999_SCOPE
        & pl.col('q_max_temporal_cv_late').is_finite()
    ),
    predicted_direction='a_gt_b',
)
def snake_g0999_arm_drives_temporal_cv(
    pc_discovery: PCDiscoveryResult,
    *,
    nodes: tuple[str, ...] = _NODES,
    alpha: float = 0.05,
    max_conditioning: int = 2,
    indicators: dict[str, tuple[str, str]] = _ARM_INDICATOR,
    min_cells: int = 30,
    arm_node: str = 'arm_ddqn_indicator',
    temporal_cv_node: str = 'q_max_temporal_cv_late',
) -> Verdict:
    """Pre-registered: PC skeleton at Snake γ=0.999 contains
    `arm — q_max_temporal_cv_late` (the CLIP-RATCHET temporal
    signature is γ-portable). HELD if signature replicates from
    γ=0.99."""
    del nodes, alpha, max_conditioning, indicators
    if pc_discovery.n_cells < min_cells:
        return Verdict.POWER_INSUFFICIENT
    if pc_discovery.is_in_skeleton(arm_node, temporal_cv_node):
        return Verdict.HELD
    return Verdict.POWER_INSUFFICIENT


# ============ Bridge 2: σ_Q inflation (load-bearing confirmatory) ============

@claim_bridge(
    source=INTERVENTION,
    target='q_action_std_late',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=(
        _SNAKE_G0999_SCOPE
        & pl.col('q_action_std_late').is_finite()
    ),
    predicted_direction='a_gt_b',
)
def snake_g0999_arm_inflates_action_std(
    stratified_arm_diff_pooled: StratifiedArmDiffPooledResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    source: str = 'q_action_std_late',
    stratify_by: tuple[str, ...] = ('env_name',),
    scope_predictor: str = 'jensen_gap',
    min_baseline_predictor: float = 0.0,
    min_seeds_per_arm: int = 5,
    inflate_floor: float = 0.4,
    alpha: float = 0.05,
    min_strata: int = 1,
) -> tuple[Verdict, RefutationClass | None]:
    """Pre-registered confirmatory test: DDQN inflates cross-action
    SD of Q at Snake γ=0.999, d ≥ +0.4 sig at α=0.05.

    Load-bearing: γ=0.99 finding's Bridge 2 was substituted
    post-hoc (`q_late_mean` → `q_action_std_late`). This γ=0.999
    bridge is genuinely pre-registered before data lands. HELD →
    σ_Q form has confirmatory evidence. DRIFT to NS or SIGN_FLIP →
    γ=0.99 finding's regime claim walks back."""
    del (
        treatment_arm, baseline_arm, source, stratify_by,
        scope_predictor, min_baseline_predictor, min_seeds_per_arm,
    )
    res = stratified_arm_diff_pooled
    if res.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT, None
    if res.n_strata == 1 and math.isnan(res.pooled_d):
        s = res.per_stratum[0]
        if math.isnan(s.cohen_d) or math.isnan(s.cohen_se) or s.cohen_se <= 0:
            return Verdict.POWER_INSUFFICIENT, None
        d = s.cohen_d
        from scipy.stats import norm
        z = d / s.cohen_se
        p = 2.0 * (1.0 - float(norm.cdf(abs(z))))
    else:
        d = res.pooled_d
        p = res.pooled_p_value
        if math.isnan(d) or math.isnan(p):
            return Verdict.POWER_INSUFFICIENT, None
    if d >= inflate_floor and p < alpha:
        return Verdict.HELD, None
    if d <= -inflate_floor and p < alpha:
        return Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP
    return Verdict.POWER_INSUFFICIENT, None


# ============ Bridge 3: arm ⫫ outcome marginal ============

@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_raw_mean',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=(
        _SNAKE_G0999_SCOPE
        & pl.col('eval_best_burst_raw_mean').is_finite()
    ),
    predicted_direction='null',
)
def snake_g0999_arm_outcome_marginal_independent(
    pc_discovery: PCDiscoveryResult,
    *,
    nodes: tuple[str, ...] = _NODES,
    alpha: float = 0.05,
    max_conditioning: int = 2,
    indicators: dict[str, tuple[str, str]] = _ARM_INDICATOR,
    min_cells: int = 30,
    arm_node: str = 'arm_ddqn_indicator',
    outcome_node: str = 'eval_best_burst_raw_mean',
) -> Verdict:
    """Pre-registered: bimodal seed distribution dominates outcome
    at Snake γ=0.999 (PC finds `arm ⫫ outcome` marginally). HELD →
    bimodal pattern is γ-portable. DRIFT to non-marginal → outcome
    is no longer bimodal (e.g., DDQN harm dominates at γ=0.999)."""
    del nodes, alpha, max_conditioning, indicators
    if pc_discovery.n_cells < min_cells:
        return Verdict.POWER_INSUFFICIENT
    if pc_discovery.is_marginally_independent(arm_node, outcome_node):
        return Verdict.HELD
    return Verdict.POWER_INSUFFICIENT


BRIDGES = (
    snake_g0999_arm_drives_temporal_cv,
    snake_g0999_arm_inflates_action_std,
    snake_g0999_arm_outcome_marginal_independent,
)
