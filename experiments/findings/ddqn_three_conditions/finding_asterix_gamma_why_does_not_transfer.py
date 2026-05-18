"""WALK-BACK: the "FR chain does not transfer to Asterix"
prediction is REFUTED. The chain's input stage (γ → self_ref) IS
active at Asterix, contrary to my "no anchor failure" framing.

Authored prediction:
- `gamma_self_ref_null_at_asterix` — Asterix vanilla outcome is
  invariant across γ ∈ {0.95, 0.99, 0.999} (=22), so we expected
  ρ(γ, bootstrap_self_reference_fraction) ≈ 0. The intuition:
  "no anchor failure means self_ref doesn't grow with γ".
- `gamma_jens_residual_at_asterix_after_fr_mediators` — predicted
  substantial residual partial ρ ≥ +0.5 after partialling out
  the FR mediator set (a different mechanism class for Asterix).

Empirical (post 2026-05-17 backfill, n=180 baseline cells across
3 γ values):
- gamma_self_ref_null_at_asterix:                       NO_EFFECT (p=0)
- gamma_jens_residual_at_asterix_after_fr_mediators:    POWER_INSUFFICIENT (p=0.175)

→ 1 REFUTED + 1 underpowered → REFUTED.

The substantive lesson. `bootstrap_self_reference_fraction` is
NOT specifically an "anchor failure indicator". The measurable
captures `|γ × Q_target| / (|γ × Q_target| + |reward| + ε)` —
how dominant the bootstrap term is in the TD update. As γ → 1,
|Q| grows at any env regardless of whether reward signal is
accessible. At Asterix γ=0.999, |Q| reaches 436 (still reward-
accessible — vanilla outcome=22 throughout) and the bootstrap
term mechanically dominates the small reward term. self_ref
grows with γ at Asterix the same way it does at FR/SI.

The "anchor failure" framing was FR-specific (MC ≈ 0 makes the
denominator tiny). The measurable itself is a generic
Q-magnitude-dominance index.

Implication: the {self_ref, σ_action} mediator set may transfer
to Asterix's γ → jens path too — the framework just couldn't
detect it at n=180 (the joint partial test is underpowered with
3 γ levels). The regime split between SI (DDQN helps +2.18) and
Asterix (DDQN harms -0.76) is on the OUTCOME side (whether the
clip preserves or corrupts the working argmax), NOT on the
mediator side.

What this Finding DOES claim now (post walk-back):
- The "Asterix has a different γ→jens mechanism from FR" claim
  is REFUTED on this evidence. The FR-chain's input stage
  (γ → self_ref) operates the same way at Asterix.
- We cannot decide whether the FULL FR chain mediates Asterix's
  γ → jens path; the residual test is underpowered.

Open question: the OUTCOME-side mechanism for the regime split.
The mediators may be the same; the question is why DDQN's
reduction of those same mediators helps SI policy but hurts
Asterix policy. Likely: Asterix's working argmax depends on the
high-magnitude Q ridge that DDQN's clip dismantles
(`finding_asterix_g999_pc_mediator_triangle` in ddqn_sweeps
documents the q_late + smoothness signatures specific to Asterix).

What this Finding does NOT claim:
- That the FR chain definitively mediates Asterix's γ → jens
  (the joint partial is underpowered).
- What the actual regime-split mechanism is.

Related: `finding_si_gamma_why_transfers_from_fr` (SI HELDs the
chain), `finding_asterix_g999_pc_mediator_triangle` (Asterix-
specific mediator triangle, in ddqn_sweeps)."""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn_three_conditions.minatar_gamma_why_transfer import (
    gamma_jens_residual_at_asterix_after_fr_mediators,
    gamma_self_ref_null_at_asterix,
)


EXPECTED: ClusterVerdict = ClusterVerdict.REFUTED


BLOCKED_ON: str | None = None


BRIDGES: tuple[Bridge, ...] = (
    gamma_self_ref_null_at_asterix,
    gamma_jens_residual_at_asterix_after_fr_mediators,
)
