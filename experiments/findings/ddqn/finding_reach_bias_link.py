"""REACH bias-correction link is causally corroborated — DoWhy
backdoor + placebo + RCC trio admits on
jensen_gap → eval_best_burst_mean across FourRooms / Acrobot /
MountainCar / MetaMaze under DDQN_RELEVANT_SCOPE.

Hand-roll #1. The framework discovers this Finding through the
parent's `Hypothesis.FINDINGS` tuple; no reverse pointer to the
parent — the import graph stays one-way. Composite-bridge form
(post-trio-collapse 2026-05-12); three sub-checks live inside
`reach_link_dowhy_corroborated`'s `PairedDeltaLinkDowhyResult`
return value."""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn.bias_correction import (
    reach_link_dowhy_corroborated,
)


EXPECTED: ClusterVerdict = ClusterVerdict.SUPPORTED


BLOCKED_ON: str | None = None


BRIDGES: tuple[Bridge, ...] = (
    reach_link_dowhy_corroborated,
)
