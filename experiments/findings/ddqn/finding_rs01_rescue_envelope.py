"""Rescue at rs=0.1 is FourRooms-specific — positive HELD on FR
+ null HELDs on Acrobot/CartPole — DDQN rescue does NOT
generalize beyond the underlearning regime of FourRooms.

Hand-roll #3 — asymmetric envelope across DIFFERENT scopes
(per-env). Bridges share `(do(DDQN), outcome_native)` but not
extent; they form an envelope, not a cluster. Same Finding
Protocol surface as REACH and MetaMaze — `composed_verdict`
doesn't care whether bridges happen to share extent.

Expected SUPPORTED. Postfix cache currently has the two null
sides at POWER_INSUFFICIENT (CIs span the ±0.2 ceiling), so the
envelope walks UNDERPOWERED — honest drift signal."""
from __future__ import annotations

import sys

from corroborate.bridge.bridge import Bridge
from corroborate.core.finding import run_finding
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn.rs_rescue import (
    ddqn_does_not_rescue__acrobot_rs_0p1,
    ddqn_does_not_rescue__cartpole_rs_0p1,
    ddqn_rescues_underlearning_vanilla__fourrooms_rs_0p1,
)


EXPECTED: ClusterVerdict = ClusterVerdict.SUPPORTED


BRIDGES: tuple[Bridge, ...] = (
    ddqn_rescues_underlearning_vanilla__fourrooms_rs_0p1,
    ddqn_does_not_rescue__acrobot_rs_0p1,
    ddqn_does_not_rescue__cartpole_rs_0p1,
)


if __name__ == '__main__':
    run_finding(sys.modules[__name__])
