"""MetaMaze γ-amplification REFUTED — predicted high-γ ↑ DDQN
benefit; refuted on the (do(DDQN), eval_best_burst_mean) cluster
under _METAMAZE_GAMMA_SCOPE on the postfix corpus.

Hand-roll #2 — stress-tests the Finding shape under
`EXPECTED=REFUTED`. Same Protocol surface as the SUPPORTED case,
opposite expected verdict; validates the surface is symmetric
under negation. Substantive: see
`findings_metamaze_gamma_link.md`."""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn.within_env import (
    metamaze_link_steeper_at_high_gamma,
)


# REFUTED on the merged ddqn_sweeps cache (post 2026-05-18
# consolidation): the cache grew with origin/main's new corpora,
# unblocking the MetaMaze γ=0.999 cells that the bridge needs. The
# bridge now fires and refutes the substrate's "MetaMaze γ-
# amplification" prediction. The original BLOCKED_ON gap is closed.
EXPECTED: ClusterVerdict = ClusterVerdict.REFUTED


BLOCKED_ON: str | None = None


BRIDGES: tuple[Bridge, ...] = (
    metamaze_link_steeper_at_high_gamma,
)
