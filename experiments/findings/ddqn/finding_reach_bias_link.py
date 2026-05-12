"""REACH bias-correction link is causally corroborated — DoWhy
refutation triple admits on jensen_gap → eval_best_burst_mean
across FourRooms / Acrobot / MountainCar / MetaMaze under
DDQN_RELEVANT_SCOPE.

Hand-roll #1. The framework discovers this Finding through the
parent's `Hypothesis.FINDINGS` tuple; no reverse pointer to the
parent — the import graph stays one-way."""
from __future__ import annotations

import sys

from corroborate.bridge.bridge import Bridge
from corroborate.core.finding import run_finding
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn.bias_correction import (
    reach_link_backdoor_ate_negative,
    reach_link_placebo_refuted,
    reach_link_rcc_robust,
)


EXPECTED: ClusterVerdict = ClusterVerdict.SUPPORTED


BRIDGES: tuple[Bridge, ...] = (
    reach_link_backdoor_ate_negative,
    reach_link_placebo_refuted,
    reach_link_rcc_robust,
)


if __name__ == '__main__':
    run_finding(sys.modules[__name__])
