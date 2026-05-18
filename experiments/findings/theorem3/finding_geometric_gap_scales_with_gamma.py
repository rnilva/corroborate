"""The geometric-series argmax-accumulation gap (Theorem 3's open
limitation, THEORY §6.1) is empirically visible and γ-scaled.

Surface claim under test:
    "At higher γ, σ_Λa drifts more during training before
    converging. The open limitation acknowledged in v9 (parallel
    to §9.3's Robbins-Monro gap for Theorem 1) is not γ-invariant
    — it's γ-amplified, consistent with the bias-amplification
    mechanism Λ_m governs."

Breakout γ ∈ {0.95, 0.99, 0.999} × baseline arm × n=30 cells/γ,
with `q_lambda_a_growth_ratio` (= tail_mean / init_mean) per cell:

    γ        growth_ratio   reading
    0.95     2.4×           moderate accumulation
    0.99     4.6×           larger; converged state drifted further
    0.999    4.6×           similar; γ→1 doesn't further amplify

Spearman ρ(γ, growth_ratio) across cells predicted POSITIVE
(ρ > +0.3) — confirms γ-scaling of the geometric-series open
limitation.

This is NOT a refutation of Theorem 3. The growth ratio
quantifies how far the converged-iterate σ_clip drifts from the
one-step bootstrap σ_clip — exactly the bridge that (A4'a)'s
sibling Finding documents as approximately calibrated within the
converged tail. What this Finding shows: the TRAINING TRAJECTORY
of σ_clip is non-stationary, so substrate authors using σ_Λa^env
as an empirical proxy should anchor their measurement in the
converged tail (not at any intermediate checkpoint).

Implication: the σ_Λa^env → d_out cross-env Spearman result in
`findings_theorem3_sigma_clip_validation` is robust to γ
because both sides operate at converged-iterate state. A
re-measurement at any other checkpoint would scale.

Methodology cross-refs:
- `findings_theorem3_a4a_empirical_test` — pilot ad-hoc analysis
  (3-arm γ-sweep documented init→converged growth at 2.4×–4.6×).
- THEORY note §6.1's Status section — open limitation framing
  parallel to §9.3's Robbins-Monro gap."""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.theorem3.bridges import (
    geometric_gap_scales_with_gamma__minatar_gamma_sweep,
)


EXPECTED: ClusterVerdict = ClusterVerdict.SUPPORTED


BLOCKED_ON: str | None = None


BRIDGES: tuple[Bridge, ...] = (
    geometric_gap_scales_with_gamma__minatar_gamma_sweep,
)
