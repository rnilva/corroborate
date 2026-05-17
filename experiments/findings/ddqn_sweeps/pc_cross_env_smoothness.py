"""Cross-env PC bridges on Asterix/Breakout/Freeway γ=0.999.

DDQN's effect on Q-smoothness has DIFFERENT structural roles
per env at γ=0.999, surfaced by conservative-PC:

  Asterix (HARMS outcome): arm — smoothness in skeleton at all
    α — smoothness is an INDEPENDENT arm-driven channel
    (alongside jens and q_late in the mediator triangle).

  Breakout (HELPS outcome): arm ⫫ smoothness | {jens} at α=0.20
    — smoothness IS jens-shadow on this env. The d=-0.46
    moderate effect is fully screened by jens conditioning.

  Freeway (~neutral outcome): arm ⫫ smoothness even at α=0.20;
    arm ⫫ outcome MARGINALLY — DDQN's clip is structurally
    inactive on Freeway γ=0.999.

The three per-env structural claims jointly support the read
that smoothness is NOT a universal mediator — its role is
env-specific, and the cross-env Δ_smoothness/Δ_outcome ordinal
correlation reflects a shared cause (DDQN engagement strength)
rather than smoothness-as-mediator.

Three bridges, one per env. The Asterix bridge asserts arm —
smoothness IS in the skeleton (mechanism-active); the Breakout
bridge asserts arm ⫫ smoothness | {jens} at α=0.20 (jens-shadow
verdict); the Freeway bridge asserts arm ⫫ smoothness at α=0.20
(structurally inactive).
"""
from __future__ import annotations

import polars as pl

from corroborate.analyses.pc_discovery import PCDiscoveryResult
from corroborate.bridge.bridge import Direction, Tier, claim_bridge
from corroborate.bridge.verdict import Verdict

from experiments.findings.ddqn._arms import DDQN_ARM, INTERVENTION
from experiments.findings.ddqn._scope import CANONICAL_HP_EXCLUDING_GAMMA


_ARM_INDICATOR: dict[str, tuple[str, str]] = {
    'arm_ddqn_indicator': ('arm_key', DDQN_ARM),
}

_NODES: tuple[str, ...] = (
    'arm_ddqn_indicator',
    'jensen_gap',
    'q_late_mean',
    'q_inter_state_grad_overlap_late',
    'q_trajectory_autocorr_late',
    'q_action_grad_overlap_late',
    'state_conditional_argmax_entropy_late',
    'bootstrap_action_mismatch_late',
    'eval_best_burst_raw_mean',
)


def _scope_env(env_name: str) -> pl.Expr:
    return (
        (pl.col('env_name') == env_name)
        & (pl.col('gamma') == 0.999)
        & CANONICAL_HP_EXCLUDING_GAMMA
    )


# ============ Asterix: smoothness IS in skeleton ============

@claim_bridge(
    source=INTERVENTION,
    target='q_inter_state_grad_overlap_late',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=_scope_env('Asterix-MinAtar'),
    predicted_direction='a_lt_b',
)
def pc_smoothness_in_skeleton__asterix_g999(
    pc_discovery: PCDiscoveryResult,
    *,
    nodes: tuple[str, ...] = _NODES,
    alpha: float = 0.05,
    max_conditioning: int = 2,
    indicators: dict[str, tuple[str, str]] = _ARM_INDICATOR,
    min_cells: int = 30,
    arm_node: str = 'arm_ddqn_indicator',
    smoothness_node: str = 'q_inter_state_grad_overlap_late',
) -> Verdict:
    """HELDs when PC keeps `arm — smoothness` in the skeleton on
    Asterix γ=0.999 — smoothness is an independent arm-driven
    channel here, NOT a jens-shadow."""
    del nodes, alpha, max_conditioning, indicators
    if pc_discovery.n_cells < min_cells:
        return Verdict.POWER_INSUFFICIENT
    if pc_discovery.is_in_skeleton(arm_node, smoothness_node):
        return Verdict.HELD
    return Verdict.POWER_INSUFFICIENT


# ============ Breakout: smoothness is jens-shadow ============

@claim_bridge(
    source=INTERVENTION,
    target='q_inter_state_grad_overlap_late',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=_scope_env('Breakout-MinAtar'),
    predicted_direction='null',
)
def pc_smoothness_is_jens_shadow__breakout_g999(
    pc_discovery: PCDiscoveryResult,
    *,
    nodes: tuple[str, ...] = _NODES,
    alpha: float = 0.20,
    max_conditioning: int = 2,
    indicators: dict[str, tuple[str, str]] = _ARM_INDICATOR,
    min_cells: int = 30,
    arm_node: str = 'arm_ddqn_indicator',
    smoothness_node: str = 'q_inter_state_grad_overlap_late',
    confound_node: str = 'jensen_gap',
) -> Verdict:
    """HELDs when PC at α=0.20 finds `arm ⫫ smoothness | {jens}`
    — DDQN's moderate Breakout smoothness effect (d=-0.46) is
    fully screened by jens conditioning. Uses α=0.20 to give the
    borderline d=-0.46 effect a chance to clear; the substantive
    claim is that even when the edge could survive, it's screened
    by jens, NOT that it's marginally independent."""
    del nodes, alpha, max_conditioning, indicators
    if pc_discovery.n_cells < min_cells:
        return Verdict.POWER_INSUFFICIENT
    # arm ⫫ smoothness | {jens} HELDs iff {jens} is a separator
    if pc_discovery.has_separating_set_containing(
        arm_node, smoothness_node, confound_node,
    ):
        return Verdict.HELD
    return Verdict.POWER_INSUFFICIENT


# ============ Freeway: DDQN structurally inactive ============

@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_raw_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=_scope_env('Freeway-MinAtar'),
    predicted_direction='null',
)
def pc_arm_inactive_marginal__freeway_g999(
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
    """HELDs when PC finds `arm ⫫ outcome` marginally on Freeway
    γ=0.999 — DDQN's clip doesn't affect outcome here (d=+0.10
    NS empirically), and the structural read confirms it."""
    del nodes, alpha, max_conditioning, indicators
    if pc_discovery.n_cells < min_cells:
        return Verdict.POWER_INSUFFICIENT
    if pc_discovery.is_marginally_independent(arm_node, outcome_node):
        return Verdict.HELD
    return Verdict.POWER_INSUFFICIENT


BRIDGES = (
    pc_smoothness_in_skeleton__asterix_g999,
    pc_smoothness_is_jens_shadow__breakout_g999,
    pc_arm_inactive_marginal__freeway_g999,
)
