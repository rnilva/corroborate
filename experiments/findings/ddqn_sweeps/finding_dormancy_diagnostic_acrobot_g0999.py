"""Dormancy diagnostic at Acrobot γ=0.999 — compensates the Mech bridge.

This Finding pairs with `finding_jens_reduction_consistency_g0999`
to provide a clean two-sided story for the DDQN-cuts-jens claim:

  - Mech bridge (`ddqn_reduces_jens_consistently__canonical_g0999`):
    DDQN reduces jens at 9/10 canonical γ=0.999 envs, binomial
    p = 0.011 → HELD. Acrobot γ=0.999 is the 9/10 outlier
    (d_jens = +0.10 — near zero).

  - Diagnostic bridge (`dormancy_gates_jens_at_acrobot_g0999`):
    At Acrobot γ=0.999, within-arm partial-ρ(dormancy, jens |
    arm) = −0.664 p = 2.1e-9 → HELD. The Jensen-dormancy
    measurable correctly indexes per-cell bias presence: cells
    with high dormancy (Q below σ_Q-based floor) genuinely
    have low observed jens.

The diagnostic bridge "compensates" the mech bridge in this
sense: it justifies, with framework-typed evidence, that the
+0.10 outlier at Acrobot γ=0.999 is a per-cell mech-dormancy
artifact (per CLAUDE.md: dormant-mech cells are UNTESTABLE not
NULL) rather than a "DDQN-fails-at-this-env" mechanism. The
consistency bridge's noise-tolerance is substantively grounded.

Single bridge, single env. The diagnostic generalises naturally
to cross-env consistency once `jensen_dormancy_gap` is backfilled
across the canonical pool (currently 3 of 10 envs have it; would
need traces restore + recompute on 7 more corpora).
"""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn_sweeps.dormancy_diagnostic import (
    dormancy_gates_jens_at_acrobot_g0999,
)


EXPECTED: ClusterVerdict = ClusterVerdict.SUPPORTED


BLOCKED_ON: str | None = None


BRIDGES: tuple[Bridge, ...] = (
    dormancy_gates_jens_at_acrobot_g0999,
)
