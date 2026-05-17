"""Conservative-PC sharpens the Asterix γ=0.999 mediator search.

Cluster Finding aggregating 6 PC-discovery bridges into one
substantive claim: at Asterix γ=0.999 (canonical-shape HPs,
n=60), conservative-PC at α=0.05, max_conditioning=2 narrows the
mediator candidate set from 6+ to exactly 3 and rules out 3
common candidates with formal independence verdicts.

Mediator-in-triangle bridges (3 HELDs expected):
  - `arm ⫫ outcome | {jensen_gap}` HELDs
  - `arm ⫫ outcome | {q_late_mean}` HELDs
  - `arm ⫫ outcome | {q_inter_state_grad_overlap_late}` HELDs

Mediator rule-out bridges (3 HELDs expected):
  - `arm ⫫ state_conditional_argmax_entropy_late` marginal HELDs
  - `arm ⫫ q_action_grad_overlap_late` marginal HELDs
  - `q_trajectory_autocorr_late ⫫ outcome` marginal HELDs

All 6 HELD → SUPPORTED. The composed verdict carries the
substantive claim: PC has formally narrowed the mediator search.

The non-orientability of the triangle is the open question —
captured in the parent finding (`q_smoothness_harm_mechanism` +
related), unblocked by the multi-env panel that k=2 / k=4 will
provide.
"""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn_sweeps.pc_mediator_triangle_asterix import (
    pc_action_grad_overlap_not_arm_affected__asterix_g999,
    pc_hcond_not_arm_affected__asterix_g999,
    pc_jens_screens_arm_to_outcome__asterix_g999,
    pc_qlate_screens_arm_to_outcome__asterix_g999,
    pc_smoothness_screens_arm_to_outcome__asterix_g999,
    pc_trajectory_autocorr_not_outcome_predictor__asterix_g999,
)


EXPECTED: ClusterVerdict = ClusterVerdict.SUPPORTED


BLOCKED_ON: str | None = None


BRIDGES: tuple[Bridge, ...] = (
    pc_jens_screens_arm_to_outcome__asterix_g999,
    pc_qlate_screens_arm_to_outcome__asterix_g999,
    pc_smoothness_screens_arm_to_outcome__asterix_g999,
    pc_hcond_not_arm_affected__asterix_g999,
    pc_action_grad_overlap_not_arm_affected__asterix_g999,
    pc_trajectory_autocorr_not_outcome_predictor__asterix_g999,
)
