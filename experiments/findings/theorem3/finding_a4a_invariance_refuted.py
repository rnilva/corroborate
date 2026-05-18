"""Theorem 3's (A4'a) cross-γ invariance is REFUTED on Breakout —
the magnitude-alignment assumption holds in absolute terms at
γ ≤ 0.99 (CV 4-6%) but degrades 3× at γ=0.999 (CV ~14%).

Surface claim originally tested:
    "Per-burst σ_Λa in the converged tail is γ-invariant. The
    empirical signature σ_Λa^env operating at any γ is a
    calibrated proxy for one-step σ_clip under (A4'a)."

What the panel showed (Breakout × baseline × n=30 cells/γ):

    γ        mean tail_cv   trajectory regime
    0.95     5.6%           stable converged tail
    0.99     3.8%           stable converged tail
    0.999    13.6%          peaks-then-restabilises, less stable

Spearman ρ(γ, q_lambda_a_tail_cv) cross-cell = +0.42 (p ≈ 2e-5)
— significantly positive, refuting the null (|ρ| < 0.3)
prediction. The ad-hoc analysis (`findings_theorem3_a4a_empirical
_test`) read CV under 20% in absolute terms as "(A4'a) holds";
the framework's surface-cross-γ test catches a real subtlety
that absolute-magnitude reading missed.

Empirical content (refined): (A4'a) magnitude alignment is
approximately right at γ ≤ 0.99 but increasingly marginal at
γ=0.999 — exactly the regime the canonical Theorem 3 n=8 panel
operates in. The geometric-series accumulation gap (sibling
Finding `finding_geometric_gap_scales_with_gamma`) HELDs at
ρ=+0.95 confirming the open limitation is γ-amplified.

Implication for Theorem 3's empirical signature:
- The σ_Λa^env → d_out cross-env Spearman result (`findings_
  theorem3_sigma_clip_validation`) measures at γ=0.999 across
  envs. The within-env tail-stability is weakest there → the
  proxy is noisier than the ad-hoc read suggested.
- The "calibrated proxy" claim in the Status section of THEORY
  §6.1 should be qualified: (A4'a) holds APPROXIMATELY at the
  canonical regime, but tail stability degrades with γ.
- The cross-env n=8 result still stands (Spearman ρ=−0.778
  p=0.023) — it's a rank correlation, robust to within-cell
  CV variation. The directional finding survives. The
  magnitude calibration is the part that needs qualifying.

Methodology cross-refs:
- `findings_theorem3_a4a_empirical_test` — pilot ad-hoc analysis
  that DID NOT catch the γ-correlation (read absolute CV instead).
- `findings_theorem3_sigma_clip_validation` — n=8 panel
  Spearman ρ result; not invalidated, but anchor noisier at γ=0.999.
- THEORY note §6.1 (committed `b416432`) — (A4'a) statement +
  open limitation framing.

What this Finding shows the framework caught:
- The cross-γ surface test surfaces a real signal (γ→tail_cv
  correlation) that the per-γ-mean absolute-CV reading missed.
- The "(A4'a) HOLDS" verdict from ad-hoc analysis was too strong
  — the framework's typed bridge surfaces honest drift."""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.theorem3.bridges import (
    a4a_tail_cv_invariant_across_gamma__breakout,
)


EXPECTED: ClusterVerdict = ClusterVerdict.REFUTED


BLOCKED_ON: str | None = None


BRIDGES: tuple[Bridge, ...] = (
    a4a_tail_cv_invariant_across_gamma__breakout,
)
