"""The geometric-series argmax-accumulation γ-scaling claim
holds at the k=2 action_duplicate corpus but does NOT generalise
to the broader k=1+k=2 panel — UNDERPOWERED on the cross-K extent.

Surface claim originally tested:
    "At higher γ, σ_Λa drifts more during training before
    converging. The open limitation acknowledged in v9 (parallel
    to §9.3's Robbins-Monro gap for Theorem 1) is γ-amplified,
    consistent with the bias-amplification mechanism Λ_m governs."

Empirical content (cross-K panel, 4 envs × 2-3 γ × 2 K = 840 cells):

    env           K   γ=0.95 growth  γ=0.999 growth
    Asterix       2   1.11           1.16           (≈ flat)
    Asterix       5   ~1.7           ~2.3           (γ-scales)
    Breakout      2   0.96           1.03           (≈ flat)
    Breakout      3   ~1.3           ~2.4           (γ-scales)
    Freeway       3   2.31           1.08           (INVERSE)
    SI            4   1.25           1.04           (INVERSE)

The k=2 action_duplicate corpus showed clean γ-scaling
(ρ ≈ +0.7 at n=180 Breakout+Asterix Phase 2). The broader
k=1+k=2 panel Fisher-z pool is POWER_INSUFFICIENT (p=0.35) —
the γ-scaling does NOT cleanly generalise: Freeway and SI on the
k=1 corpus show INVERSE-correlation; Breakout k=1 is essentially
flat. The k=2-only signal was an action-duplicate-amplified
effect that doesn't survive aggregation across K-values.

Honest reading:
- The geometric-series accumulation gap IS γ-amplified WITHIN
  the k=2 corpus (action_duplicate inflates Λ_a magnitudes,
  making the early→converged drift sensitive to γ).
- Without action_duplicate (k=1), envs with naturally small σ
  (Breakout, Freeway, SI) don't show the same γ-trajectory.
- The Theorem 3 open limitation (geometric-series gap) is
  real but its γ-scaling pattern is action-count-dependent.

This Finding's UNDERPOWERED verdict honestly reflects the
cross-K aggregation NOT supporting the original k=2-only claim.
A k=2-scoped sibling bridge would HELD; the broader-panel
verdict is UNDERPOWERED.

Methodology cross-refs:
- `findings_theorem3_a4a_empirical_test` — pilot k=2 ad-hoc analysis.
- THEORY note §6.1's Status section — open limitation framing
  parallel to §9.3's Robbins-Monro gap."""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.theorem3.bridges import (
    geometric_gap_scales_with_gamma__minatar_gamma_sweep,
)


EXPECTED: ClusterVerdict = ClusterVerdict.UNDERPOWERED


BLOCKED_ON_INFO = (
    "The k=2 action_duplicate-only panel showed strong γ-scaling "
    "(ρ ~ +0.7 at the 2-env n=180 Breakout+Asterix scope). On the "
    "broader k=1+k=2 panel (4 envs × 2-3 γ × 2 K-values = 840 cells), "
    "Fisher-z-pooled Spearman ρ is POWER_INSUFFICIENT (p=0.35) — the "
    "geometric-series γ-scaling does NOT cleanly generalise across "
    "action_duplicate interventions. Without action_duplicate (k=1), "
    "Breakout's growth ratio is flat (~1.0) and SI/Freeway "
    "INVERSE-correlate with γ. The k=2-only signal was an "
    "action-duplicate-amplified effect, not an env-general one."
)


BLOCKED_ON: str | None = None


BRIDGES: tuple[Bridge, ...] = (
    geometric_gap_scales_with_gamma__minatar_gamma_sweep,
)
