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

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn_sweeps.rs_rescue import (
    ddqn_does_not_rescue__acrobot_rs_0p1,
    ddqn_does_not_rescue__cartpole_rs_0p1,
    ddqn_rescues_underlearning_vanilla__fourrooms_rs_0p1,
)


EXPECTED: ClusterVerdict = ClusterVerdict.UNDERPOWERED


BLOCKED_ON: str | None = (
    '2026-05-19 partial-DRIFT: ingest of rs01_followup_acrobot_cartpole '
    '(n=90/arm, restored from cloud) brought CartPole to n=120 → '
    'CartPole rs=0.1 null HELD (md=+0.077, CI ⊂ ±0.2). Acrobot still '
    'POWER_INSUFFICIENT at n=120 (md=+0.094, CI slightly exceeds '
    '±0.2 ceiling). Envelope needs Acrobot rs=0.1 to land HELD before '
    'flipping to SUPPORTED — one more n=30 batch should suffice given '
    'current SE. When it lands, flip EXPECTED and clear BLOCKED_ON.'
)


BRIDGES: tuple[Bridge, ...] = (
    ddqn_rescues_underlearning_vanilla__fourrooms_rs_0p1,
    ddqn_does_not_rescue__acrobot_rs_0p1,
    ddqn_does_not_rescue__cartpole_rs_0p1,
)
