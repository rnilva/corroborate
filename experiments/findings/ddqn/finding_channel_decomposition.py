"""Q-shape "channel" at canonical is largely Q-MC tautology
artifact — REFUTED after partial conditioning on Q-magnitude.

Two-bridge cluster on `q_action_std_per_burst → mc_return_raw_
per_burst_mean` documents the substantive resolution of the
Q-shape channel claim:

1. `q_action_std_per_burst_link_to_outcome` (marginal):
   ρ_pooled = +0.249 → HELD (above +0.2 threshold)
2. `q_action_std_per_burst_link_to_outcome__partial_q` (partial
   conditioned on q_per_burst): ρ_partial = below +0.2 threshold,
   p=0.0006 → **NO_EFFECT**

Cluster verdict: **REFUTED**. The marginal HELD was largely
driven by the Q-IS-MC structural coupling on positive-return
envs (Q estimates MC return, so Q-spread cells trivially have
high MC). After partialling Q-magnitude, the Q-SHAPE residual
doesn't surface above the substantive threshold.

**Implication for the "two channels" framing at canonical**:
the channel decomposition reduces to ONE substantive channel
(bg / algorithmic clip — `finding_three_gate_scope_outcome_
held` SUPPORTED) plus Q-magnitude acting through structural
Q-MC coupling (which is part of the bg→outcome chain anyway,
not an independent mediator). The pre-canonical "Q-channel
beyond bg" finding survives only as a residual that the
mediator-search script (`scripts/q_channel_mediator_search.py`)
explains via within-cell Q-shape measures — but those measures
don't add substantive predictive value to outcome beyond
Q-magnitude itself.

Companion (NOT in this Finding's cluster — different extent,
weaker marginal signal): `q_argmax_margin_per_burst_link_to_
outcome` fires NO_EFFECT marginally (ρ=+0.157). Its role lives
at the Q→MC residual conditioning level per the mediator-search
script (reduces partial ρ(q, mc | bg) by 0.35), not as a direct
outcome predictor.

Per-env Q-shape mediation strength (from mediator-search) varies
across envs — documented in memory
`findings_q_shape_env_class_stratification`. The variation is
Q-magnitude-variance-based, not polarity-based, and after this
partial-conditioning test the "differs by env" claim is also
shown to be largely tautology-driven.

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

from experiments.findings.ddqn.q_shape_mediation import (
    q_action_std_per_burst_link_to_outcome,
    q_action_std_per_burst_link_to_outcome__partial_q,
)


EXPECTED: ClusterVerdict = ClusterVerdict.REFUTED


BLOCKED_ON: str | None = None


BRIDGES: tuple[Bridge, ...] = (
    q_action_std_per_burst_link_to_outcome,
    q_action_std_per_burst_link_to_outcome__partial_q,
)
