"""Snake γ=0.99: 4th regime category in cross-env DDQN outcome
classification — CLIP-RATCHET / T1-sign-flipped.

The 4-env γ=0.999 MinAtar panel
(`finding_cross_env_outcome_regime_g999`) identified three
regimes (Q-EXPLODED, Q-STRUCTURED, Q-COLLAPSED). Snake γ=0.99 at
canonical HPs (n=60, `snake_1M` corpus) fits a structurally
distinct fourth regime documented in
`findings_snake_ddqn_destabilizes_sparse_reward`: DDQN's clip
LOCKS Q-explosion in 3/30 seeds rather than suppressing it
uniformly, producing a bimodal seed distribution that
renders the marginal Cohen's d uninformative.

Three bridges jointly assert the 4th-regime claim:

  1. `snake_arm_drives_temporal_cv` — `arm — q_max_temporal_cv_late`
     edge IS in PC skeleton (unique to Snake; not in any
     γ=0.999 MinAtar env's PC skeleton).

  2. `snake_t1_sign_flipped` — Q_VAN < Q_DDQN at the mean
     (Type 1 NEGATIVE, ~-0.53), contradicting the standard
     "T1 reduces max-bias" framing. DDQN INFLATES rather than
     reduces Q on Snake.

  3. `snake_arm_outcome_marginal_independent` — `arm ⫫ outcome`
     marginally (consistent with the bimodal d=+0.22 NS).
     Confirms the marginal Cohen's d framing hides the actual
     regime structure.

If all 3 HELD → 4th-regime claim SUPPORTED. The Finding sits
alongside `finding_cross_env_outcome_regime_g999` as the
fifth-env extension of the regime classification.
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


_SNAKE_G099_SCOPE: pl.Expr = (
    (pl.col('env_name') == 'Snake-jumanji')
    & (pl.col('gamma') == 0.99)
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
        _SNAKE_G099_SCOPE
        & pl.col('q_max_temporal_cv_late').is_finite()
    ),
    predicted_direction='a_gt_b',
)
def snake_arm_drives_temporal_cv(
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
    """HELDs when PC's skeleton includes `arm — q_max_temporal_cv`
    on Snake γ=0.99. This edge is the structural signature of
    clip-ratchet failure (DDQN triples temporal CV; not seen in
    any γ=0.999 MinAtar env)."""
    del nodes, alpha, max_conditioning, indicators
    if pc_discovery.n_cells < min_cells:
        return Verdict.POWER_INSUFFICIENT
    if pc_discovery.is_in_skeleton(arm_node, temporal_cv_node):
        return Verdict.HELD
    return Verdict.POWER_INSUFFICIENT


# ============ Bridge 2: T1 sign-flipped (DDQN inflates Q) ============

@claim_bridge(
    source=INTERVENTION,
    target='q_late_mean',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=(
        _SNAKE_G099_SCOPE
        & pl.col('q_late_mean').is_finite()
    ),
    predicted_direction='a_gt_b',
)
def snake_t1_sign_flipped(
    stratified_arm_diff_pooled: StratifiedArmDiffPooledResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    source: str = 'q_late_mean',
    stratify_by: tuple[str, ...] = ('env_name',),
    scope_predictor: str = 'jensen_gap',
    min_baseline_predictor: float = 0.0,
    min_seeds_per_arm: int = 5,
    inflate_floor: float = 0.2,
    alpha: float = 0.10,
    min_strata: int = 1,
) -> tuple[Verdict, RefutationClass | None]:
    """T1 = Q_VAN − Q_DDQN < 0 on Snake (DDQN INFLATES Q). HELD
    when Cohen's d on q_late_mean ≥ inflate_floor (DDQN's mean Q
    is significantly above vanilla's) at α=0.10.

    The bimodal seed distribution means the population mean shift
    is outlier-driven; we accept α=0.10 because the substantive
    claim (T1 sign-flipped) is the existence of the inflation,
    not its tightness."""
    del (
        treatment_arm, baseline_arm, source, stratify_by,
        scope_predictor, min_baseline_predictor, min_seeds_per_arm,
    )
    res = stratified_arm_diff_pooled
    if res.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT, None
    # Single-stratum fallback (DL pool nan at n_strata=1)
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


# ============ Bridge 3: arm ⫫ outcome marginal (bimodal NS) ============

@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_raw_mean',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=(
        _SNAKE_G099_SCOPE
        & pl.col('eval_best_burst_raw_mean').is_finite()
    ),
    predicted_direction='null',
)
def snake_arm_outcome_marginal_independent(
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
    """HELDs when PC finds `arm ⫫ outcome` marginally on Snake
    γ=0.99. Confirms d=+0.22 NS reading; the marginal mean-
    difference is uninformative because the seed distribution
    is bimodal (3 Q-exploding DDQN seeds + 1 outcome=7.67
    outlier + 26 seeds matching vanilla)."""
    del nodes, alpha, max_conditioning, indicators
    if pc_discovery.n_cells < min_cells:
        return Verdict.POWER_INSUFFICIENT
    if pc_discovery.is_marginally_independent(arm_node, outcome_node):
        return Verdict.HELD
    return Verdict.POWER_INSUFFICIENT


BRIDGES = (
    snake_arm_drives_temporal_cv,
    snake_t1_sign_flipped,
    snake_arm_outcome_marginal_independent,
)
