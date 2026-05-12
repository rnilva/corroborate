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


EXPECTED: ClusterVerdict = ClusterVerdict.REFUTED


BLOCKED_ON: str | None = None


BRIDGES: tuple[Bridge, ...] = (
    metamaze_link_steeper_at_high_gamma,
)
