"""Policy-structure-channel: DDQN's outcome benefit on
{Breakout, PacMan} is mediated by argmax_entropy_late, NOT by
jensen_gap reduction.

DoWhy backdoor + refutation per-env at canonical 1M:

  Breakout-MinAtar:
    ATE_marg = +17.67, ATE | entropy = +8.42 → 52% absorbed
    ATE | jens = +45.29 (sign-FLIPPED collider behavior — jens is
        NOT a clean mediator; it's confounded by entropy)
    placebo ATE = 0.000 ✓, RCC drift = 0.022 ✓
  PacMan-jumanji:
    ATE_marg = +165.67, ATE | entropy = +56.54 → **66% absorbed**
    ATE | jens = +156.98 (only 5% absorbed — jens is NOT the
        mediator on PacMan)
    placebo ATE = 0.000 ✓, RCC drift = 2.80 (1.7% of ATE) ✓

The non-Hasselt mechanism: DDQN modifies the argmax distribution
across states (vanilla DQN's overestimation bias collapses the
policy onto fewer favorite actions; DDQN preserves diversity
on Breakout, expands on PacMan). This is a POLICY-STRUCTURE
mechanism, not a Q-value-correction mechanism.

The collider behavior of jens on Breakout (sign-flipped slope
when conditioning) confirms jens is downstream of treatment +
outcome-correlated, not a true mediator. Memory:
`findings_ddqn_mediator_heterogeneity` documents both channels.

EXPECTED: SUPPORTED. Both envs show clean entropy mediation with
refutation passes. The policy-structure mechanism is a SECOND
load-bearing DDQN mechanism orthogonal to Hasselt's bias
reduction.

Reproducer: `scripts/per_env_dowhy_mediation.py`."""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict


EXPECTED: ClusterVerdict = ClusterVerdict.SUPPORTED
BLOCKED_ON: str | None = 'bridge implementations deferred — DoWhy per-env mediation primitive needs scope wrapping'
BRIDGES: tuple[Bridge, ...] = ()
