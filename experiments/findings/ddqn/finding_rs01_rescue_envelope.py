"""Rescue at rs=0.1 is FourRooms-specific — positive HELD on FR
+ null HELDs on Acrobot/CartPole — DDQN rescue does NOT
generalize beyond the underlearning regime of FourRooms.

Hand-roll #3 — asymmetric envelope across DIFFERENT scopes
(per-env). Bridges share `(do(DDQN), outcome_native)` but not
extent; they form an envelope, not a cluster.

Theoretical claim is SUPPORTED (positive HELD on FR + null HELDs
on Acrobot/CartPole). `EXPECTED` is pinned to the *current
empirical state* (UNDERPOWERED) so drift fires only on state
change. `BLOCKED_ON` documents the gap: postfix cache has the
two null sides at POWER_INSUFFICIENT (CIs span the ±0.2 ceiling
at n=60); landing them at HELD needs more cells."""
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


EXPECTED: ClusterVerdict = ClusterVerdict.UNDERPOWERED


BLOCKED_ON: str = (
    'Acrobot/CartPole rs=0.1 null sides are POWER_INSUFFICIENT '
    '(CIs span ±0.2 at n=60). Theoretical claim asserts SUPPORTED '
    'envelope; data needs ~n=240 per env to land nulls at HELD. '
    'When the cache gains those cells, this finding will drift '
    'from UNDERPOWERED → SUPPORTED — author update: flip EXPECTED '
    'and clear BLOCKED_ON.'
)


BRIDGES: tuple[Bridge, ...] = (
    ddqn_rescues_underlearning_vanilla__fourrooms_rs_0p1,
    ddqn_does_not_rescue__acrobot_rs_0p1,
    ddqn_does_not_rescue__cartpole_rs_0p1,
)


if __name__ == '__main__':
    run_finding(sys.modules[__name__])
