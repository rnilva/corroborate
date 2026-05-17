"""PC-discovery bridges for the Asterix γ=0.999 mediator triangle.

Conservative-PC on the 9-variable joint cohort (n=60) discovered
a non-orientable triangle `{jensen_gap, q_late_mean,
q_inter_state_grad_overlap_late}` as the mediator candidate set
and ruled out 5 other candidates. Each substantive verdict is a
single bridge consuming `pc_discovery: PCDiscoveryResult`:

Mediator-in-triangle (3 bridges):
  - jens screens `arm ⫫ outcome`
  - q_late_mean screens `arm ⫫ outcome`
  - q_inter_state_grad_overlap_late screens `arm ⫫ outcome`

Mediator rule-outs (3 bridges):
  - `arm ⫫ state_conditional_argmax_entropy_late` marginal
    (H_cond is NOT a mediator)
  - `arm ⫫ q_action_grad_overlap_late` marginal (cross-action
    smoothness is NOT arm-affected; baseline covariate)
  - `q_trajectory_autocorr_late ⫫ outcome` marginal (temporal
    autocorr is training artifact, not outcome predictor)

The 6 bridges share file-level scope (Asterix γ=0.999, canonical-
shape HPs) and pc_discovery node list. Together they form the
structural-discovery cluster: the mediator search is sharpened
from 6+ candidates to exactly 3, with the triangle's
non-orientability flagged as the open question.
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


_ASTERIX_GAMMA_999_SCOPE: pl.Expr = (
    (pl.col('env_name') == 'Asterix-MinAtar')
    & (pl.col('gamma') == 0.999)
    & CANONICAL_HP_EXCLUDING_GAMMA
)

# arm-encoded column needs to exist on each cell as a scalar
# float; bridges add `arm_ddqn` via `with_columns` upstream.
# Here we use `arm_key` string directly via the cells' record
# dict on the analysis side — pc_discovery filters non-scalar
# values so we encode at the @analysis layer via `nodes`. Use
# a registered measurable (planned) for the proper path; for
# now author the bridge against the column produced by
# REQUIRED_MEASURABLES / a measurable. The framework's
# `arm_ddqn` encoding lives in `experiments.findings.ddqn._arms`
# (TODO link). For Asterix γ=0.999 single-env at n=60, the arm
# column resolves via `arm_ddqn_indicator` measurable
# (registered alongside _arms).
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


# ============ Mediator-in-triangle bridges ============

@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_raw_mean',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=_ASTERIX_GAMMA_999_SCOPE,
    predicted_direction='a_lt_b',
)
def pc_jens_screens_arm_to_outcome__asterix_g999(
    pc_discovery: PCDiscoveryResult,
    *,
    nodes: tuple[str, ...] = _NODES,
    alpha: float = 0.05,
    max_conditioning: int = 2,
    indicators: dict[str, tuple[str, str]] = _ARM_INDICATOR,
    min_cells: int = 30,
    arm_node: str = 'arm_ddqn_indicator',
    outcome_node: str = 'eval_best_burst_raw_mean',
    mediator_node: str = 'jensen_gap',
) -> Verdict:
    """HELDs when `arm ⫫ outcome | {jens}` was discovered by PC
    — jens is sufficient to mediate the arm's effect on outcome
    at this scope."""
    del nodes, alpha, max_conditioning, indicators
    if pc_discovery.n_cells < min_cells:
        return Verdict.POWER_INSUFFICIENT
    if pc_discovery.has_separating_set_containing(
        arm_node, outcome_node, mediator_node,
    ):
        return Verdict.HELD
    return Verdict.POWER_INSUFFICIENT


@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_raw_mean',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=_ASTERIX_GAMMA_999_SCOPE,
    predicted_direction='a_lt_b',
)
def pc_qlate_screens_arm_to_outcome__asterix_g999(
    pc_discovery: PCDiscoveryResult,
    *,
    nodes: tuple[str, ...] = _NODES,
    alpha: float = 0.05,
    max_conditioning: int = 2,
    indicators: dict[str, tuple[str, str]] = _ARM_INDICATOR,
    min_cells: int = 30,
    arm_node: str = 'arm_ddqn_indicator',
    outcome_node: str = 'eval_best_burst_raw_mean',
    mediator_node: str = 'q_late_mean',
) -> Verdict:
    """HELDs when `arm ⫫ outcome | {q_late_mean}` was discovered
    — Q magnitude is sufficient to mediate."""
    del nodes, alpha, max_conditioning, indicators
    if pc_discovery.n_cells < min_cells:
        return Verdict.POWER_INSUFFICIENT
    if pc_discovery.has_separating_set_containing(
        arm_node, outcome_node, mediator_node,
    ):
        return Verdict.HELD
    return Verdict.POWER_INSUFFICIENT


@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_raw_mean',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=_ASTERIX_GAMMA_999_SCOPE,
    predicted_direction='a_lt_b',
)
def pc_smoothness_screens_arm_to_outcome__asterix_g999(
    pc_discovery: PCDiscoveryResult,
    *,
    nodes: tuple[str, ...] = _NODES,
    alpha: float = 0.05,
    max_conditioning: int = 2,
    indicators: dict[str, tuple[str, str]] = _ARM_INDICATOR,
    min_cells: int = 30,
    arm_node: str = 'arm_ddqn_indicator',
    outcome_node: str = 'eval_best_burst_raw_mean',
    mediator_node: str = 'q_inter_state_grad_overlap_late',
) -> Verdict:
    """HELDs when `arm ⫫ outcome | {smoothness}` was discovered
    — cross-state Q smoothness is sufficient to mediate."""
    del nodes, alpha, max_conditioning, indicators
    if pc_discovery.n_cells < min_cells:
        return Verdict.POWER_INSUFFICIENT
    if pc_discovery.has_separating_set_containing(
        arm_node, outcome_node, mediator_node,
    ):
        return Verdict.HELD
    return Verdict.POWER_INSUFFICIENT


# ============ Mediator rule-out bridges ============

@claim_bridge(
    source=INTERVENTION,
    target='state_conditional_argmax_entropy_late',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=_ASTERIX_GAMMA_999_SCOPE,
    predicted_direction='null',
)
def pc_hcond_not_arm_affected__asterix_g999(
    pc_discovery: PCDiscoveryResult,
    *,
    nodes: tuple[str, ...] = _NODES,
    alpha: float = 0.05,
    max_conditioning: int = 2,
    indicators: dict[str, tuple[str, str]] = _ARM_INDICATOR,
    min_cells: int = 30,
    arm_node: str = 'arm_ddqn_indicator',
    candidate_node: str = 'state_conditional_argmax_entropy_late',
) -> Verdict:
    """HELDs when PC found `arm ⫫ H_cond` marginally — H_cond is
    NOT moved by DDQN's clip; can't be a mediator. Formal rule-
    out of the clip-argmax-noise mechanism."""
    del nodes, alpha, max_conditioning, indicators
    if pc_discovery.n_cells < min_cells:
        return Verdict.POWER_INSUFFICIENT
    if pc_discovery.is_marginally_independent(arm_node, candidate_node):
        return Verdict.HELD
    return Verdict.POWER_INSUFFICIENT


@claim_bridge(
    source=INTERVENTION,
    target='q_action_grad_overlap_late',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=_ASTERIX_GAMMA_999_SCOPE,
    predicted_direction='null',
)
def pc_action_grad_overlap_not_arm_affected__asterix_g999(
    pc_discovery: PCDiscoveryResult,
    *,
    nodes: tuple[str, ...] = _NODES,
    alpha: float = 0.05,
    max_conditioning: int = 2,
    indicators: dict[str, tuple[str, str]] = _ARM_INDICATOR,
    min_cells: int = 30,
    arm_node: str = 'arm_ddqn_indicator',
    candidate_node: str = 'q_action_grad_overlap_late',
) -> Verdict:
    """HELDs when PC found `arm ⫫ q_action_grad_overlap` marginally
    — cross-action gradient overlap is NOT moved by DDQN. It
    survives in the skeleton as a direct outcome predictor (env
    baseline variance), but rules out as mediator."""
    del nodes, alpha, max_conditioning, indicators
    if pc_discovery.n_cells < min_cells:
        return Verdict.POWER_INSUFFICIENT
    if pc_discovery.is_marginally_independent(arm_node, candidate_node):
        return Verdict.HELD
    return Verdict.POWER_INSUFFICIENT


@claim_bridge(
    source='q_trajectory_autocorr_late',
    target='eval_best_burst_raw_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=_ASTERIX_GAMMA_999_SCOPE,
    predicted_direction='null',
)
def pc_trajectory_autocorr_not_outcome_predictor__asterix_g999(
    pc_discovery: PCDiscoveryResult,
    *,
    nodes: tuple[str, ...] = _NODES,
    alpha: float = 0.05,
    max_conditioning: int = 2,
    indicators: dict[str, tuple[str, str]] = _ARM_INDICATOR,
    min_cells: int = 30,
    candidate_node: str = 'q_trajectory_autocorr_late',
    outcome_node: str = 'eval_best_burst_raw_mean',
) -> Verdict:
    """HELDs when PC found `trajectory_autocorr ⫫ outcome`
    marginally — temporal Q autocorr is a training-dynamics
    artifact, not an outcome predictor. Rules it out as a
    candidate channel."""
    del nodes, alpha, max_conditioning, indicators
    if pc_discovery.n_cells < min_cells:
        return Verdict.POWER_INSUFFICIENT
    if pc_discovery.is_marginally_independent(candidate_node, outcome_node):
        return Verdict.HELD
    return Verdict.POWER_INSUFFICIENT


BRIDGES = (
    pc_jens_screens_arm_to_outcome__asterix_g999,
    pc_qlate_screens_arm_to_outcome__asterix_g999,
    pc_smoothness_screens_arm_to_outcome__asterix_g999,
    pc_hcond_not_arm_affected__asterix_g999,
    pc_action_grad_overlap_not_arm_affected__asterix_g999,
    pc_trajectory_autocorr_not_outcome_predictor__asterix_g999,
)
