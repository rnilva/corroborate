"""MetaMaze γ-amplification REFUTED — predicted high-γ ↑ DDQN
benefit; both mean + median sibling walks return refuted on the
(do(DDQN), eval_best_burst_mean) cluster under
_METAMAZE_GAMMA_SCOPE on the postfix corpus.

Hand-roll #2 — stress-tests the Finding shape under
`EXPECTED=REFUTED`. Same Protocol surface as the SUPPORTED case,
opposite expected verdict; validates the surface is symmetric
under negation. Substantive: see
`findings_metamaze_gamma_link.md`."""
from __future__ import annotations

import sys

from corroborate.bridge.bridge import Bridge
from corroborate.core.finding import run_finding
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn.within_env import (
    metamaze_link_steeper_at_high_gamma,
    metamaze_link_steeper_at_high_gamma__median,
)


EXPECTED: ClusterVerdict = ClusterVerdict.REFUTED


BLOCKED_ON: str = ''


BRIDGES: tuple[Bridge, ...] = (
    metamaze_link_steeper_at_high_gamma,
    metamaze_link_steeper_at_high_gamma__median,
)


if __name__ == '__main__':
    run_finding(sys.modules[__name__])
