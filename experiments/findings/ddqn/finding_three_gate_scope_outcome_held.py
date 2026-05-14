"""DDQN's outcome benefit is HELD under the three-gate scope
conjunction at canonical.

The substrate-level positive claim the corpus supports at canonical:
when the SCOPE conditions hold (G1: bias premise active, G2:
argmax-vulnerable env, G3: chain-deep enough), DDQN reduces
overestimation AND that reduction translates to higher raw outcome.
Pooled Cohen's d = +0.46, p = 0.006, CI = [+0.13, +0.78] across
7 admitted envs at canonical (commit 8fd695c, after γ-discount→raw
target fix).

The trajectory-smoothness sibling
(`q_trajectory_autocorr_late > 0.5`) adds a fourth gate testing the
unified-degeneracy theory's axis (i) — Q-spatial-smoothness along
the agent's visited trajectory. Both bridges fire HELD at canonical.

This is the load-bearing positive outcome claim. Outside the
three-gate scope, the cross-env / cross-HP dose-response is NULL
or REFUTED (see `finding_reach_bias_link` REFUTED on the
unscoped cross-env arm-diff form). Per memory
`findings_outcome_translation_refuted_cross_scope`, the
practitioner-facing "DDQN universally improves on harder envs"
claim is REFUTED across both env-variance and HP-variance pools.
The properly-scoped form is what holds."""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn.outcome_scope import (
    ddqn_helps_under_three_gate_scope__cross_env,
    ddqn_helps_under_three_gate_scope_AND_trajectory_smooth__cross_env,
)


EXPECTED: ClusterVerdict = ClusterVerdict.SUPPORTED


BLOCKED_ON: str | None = None


BRIDGES: tuple[Bridge, ...] = (
    ddqn_helps_under_three_gate_scope__cross_env,
    ddqn_helps_under_three_gate_scope_AND_trajectory_smooth__cross_env,
)
