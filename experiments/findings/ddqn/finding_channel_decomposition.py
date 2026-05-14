"""DDQN's outcome benefit operates through a Q-shape mediator
channel (action-std) at per-burst within-cell granularity.

The substantive channel decomposition at canonical:

1. **bg-channel** (algorithmic clip): `bg_per_burst_link_to_outcome`
   fires HELD as predicted-NULL (pooled ρ ≈ -0.16, below |0.2|
   null band). The bg → outcome per-burst link is env-specific
   (cancels in pool, strong negative on Acrobot, mixed elsewhere).
2. **Q-shape channel** (action-spread): `q_action_std_per_burst_
   link_to_outcome` fires HELD as predicted-POSITIVE (pooled ρ =
   +0.249 across 10 envs). DDQN's effect on within-state Q-spread
   consistently predicts per-burst outcome.

Companion (NOT in this Finding's cluster — empirically below
threshold): `q_argmax_margin_per_burst_link_to_outcome` fires
NO_EFFECT (ρ=+0.157, below +0.2 marginal threshold). The
empirical mediator search (`scripts/q_channel_mediator_search.py`)
shows q_argmax_margin's role is at the Q→MC RESIDUAL level
(reduces partial ρ(q, mc | bg) from +0.58 to +0.23 — single
strongest residual reducer) — i.e., it mediates the Q→MC channel
conditional on bg, but doesn't directly correlate with outcome
at the marginal per-burst granularity tested here.

Per-env variation (from mediator-search script): Q-shape mediation
strength differs substantially across envs — Breakout-MinAtar
shows huge Q→MC residual reduction (-0.62) after conditioning on
bg + Q-shape; PacMan/SI/MetaMaze moderate (-0.22 to -0.52);
Asterix/Freeway/SlidingTile small (-0.06 to -0.16). Documented in
memory `findings_q_shape_env_class_stratification`.

**Caveat — Q-MC tautology**: q_action_std scales with Q
magnitude on positive-return envs; Q estimates MC return so
high-Q-spread cells trivially have high MC. Per-env data
partially contradicts the naive tautology (Asterix SURVIVE shows
ρ=-0.32, MountainCar GOAL shows ρ=+0.73), so the signal isn't
purely tautological, but partial contribution is plausible. The
substantively clean form needs `ρ(q_action_std, mc | q_per_burst)`
— the per-burst partial primitive doesn't support conditioning
yet (`project_per_burst_partial_primitive` deferred). This
Finding's SUPPORTED verdict reflects the marginal empirical
signal at canonical; the partial-conditioned form may differ.

This Finding documents the positive Q-shape channel claim. The
bg-channel runs through `finding_three_gate_scope_outcome_held`
(pooled HELD at canonical) and `bg_per_burst_link_to_outcome`
(per-burst predicted-null HELD — env-specific). The
cross-env / cross-HP outcome translation is REFUTED (see
`finding_reach_bias_link`, `finding_hp_variance_outcome_refuted`).
The per-burst within-cell Q-shape channel is the substantive
local mediator that survives canonical scope."""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn.bias_correction import (
    bg_per_burst_link_to_outcome,
)
from experiments.findings.ddqn.q_shape_mediation import (
    q_action_std_per_burst_link_to_outcome,
)


EXPECTED: ClusterVerdict = ClusterVerdict.SUPPORTED


BLOCKED_ON: str | None = None


BRIDGES: tuple[Bridge, ...] = (
    bg_per_burst_link_to_outcome,
    q_action_std_per_burst_link_to_outcome,
)
