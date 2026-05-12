"""REACH bias-correction link is causally corroborated — DoWhy
backdoor + placebo + RCC trio admits on
jensen_gap → eval_best_burst_mean across FourRooms / Acrobot /
MountainCar / MetaMaze under DDQN_RELEVANT_SCOPE.

Hand-roll #1. The framework discovers this Finding through the
parent's `Hypothesis.FINDINGS` tuple; no reverse pointer to the
parent — the import graph stays one-way. Three-bridge cluster
form (post-roast issue 6 reversion 2026-05-12): the trio
collapse into one composite bridge was wrong per CLAUDE.md's
cluster-shaped causal claims principle. The three bridges test
logically distinct robustness questions (adjustment-identified
ATE; instrument validity via placebo; omitted-confound
sensitivity via RCC); the Finding's cluster verdict handles
AND-aggregation at the graph level via `composed_verdict`."""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn.bias_correction import (
    reach_link_backdoor_ate_negative,
    reach_link_placebo_refuted,
    reach_link_rcc_robust,
)


EXPECTED: ClusterVerdict = ClusterVerdict.SUPPORTED


BLOCKED_ON: str | None = None


BRIDGES: tuple[Bridge, ...] = (
    reach_link_backdoor_ate_negative,
    reach_link_placebo_refuted,
    reach_link_rcc_robust,
)
