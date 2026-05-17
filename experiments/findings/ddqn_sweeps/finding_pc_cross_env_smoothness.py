"""Cross-env PC reveals env-specific structural role of Q-smoothness.

Three per-env PC bridges at γ=0.999 jointly assert the cross-env
structural diversity:

  Asterix: arm — smoothness IS in skeleton — independent channel.
  Breakout: arm ⫫ smoothness | {jens} — jens-shadow.
  Freeway: arm ⫫ outcome marginal — DDQN structurally inactive.

All 3 HELD → SUPPORTED. The composed verdict carries the
substantive cross-env claim: smoothness has DIFFERENT structural
roles per env, refining the single-env
`findings_q_smoothness_is_jens_shadow` (which was correct for
Breakout, wrong for Asterix where smoothness is genuinely
arm-driven independently).

The substantive implication: smoothness is NOT a universal
mediator. On Asterix its co-movement with outcome reflects
shared causation from DDQN's clip engagement, not a smoothness
→ outcome path.
"""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn_sweeps.pc_cross_env_smoothness import (
    pc_arm_inactive_marginal__freeway_g999,
    pc_smoothness_in_skeleton__asterix_g999,
    pc_smoothness_is_jens_shadow__breakout_g999,
)


EXPECTED: ClusterVerdict = ClusterVerdict.SUPPORTED


BLOCKED_ON: str | None = None


BRIDGES: tuple[Bridge, ...] = (
    pc_smoothness_in_skeleton__asterix_g999,
    pc_smoothness_is_jens_shadow__breakout_g999,
    pc_arm_inactive_marginal__freeway_g999,
)
