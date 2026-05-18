"""Transfer test: does the FR γ-WHY mediation chain replicate at
SpaceInvaders γ ∈ {0.95, 0.99, 0.999}?

The FR γ-WHY finding (`finding_sigma_action_completes_chain`)
established that γ amplifies vanilla jens at FR via two joint
mediators — bootstrap_self_reference_fraction (anchor failure)
+ q_action_std_late (per-state Q variance). Joint partial
ρ(γ, jens | self_ref + σ_action) at FR is ~+0.06 NS (92%
reduction from marginal +0.78). The chain mediates fully.

SI γ=0.999 is the natural FR-analogue in the MinAtar 4-env
panel — vanilla outcome drops 101→74 going γ=0.99→γ=0.999
(partial anchor failure), DDQN rescues outcome to 105 (d_out
+2.18, biggest help). The Q-STRUCTURED regime (jens γ-scaling
133×, comparable to FR γ-scaling) suggests the same anchor-
failure-driven Q growth might be operational.

This Finding composes 3 SI-scoped chain bridges into a transfer
verdict:

1. `gamma_predicts_q_self_reference_at_si` — within SI baseline
   cells, ρ(γ, self_ref) ≥ +0.5. If γ doesn't grow self_ref at
   SI, the FR anchor-failure mechanism is not operational here.

2. `q_self_reference_predicts_jens_at_si` — within SI baseline,
   ρ(self_ref, jens) ≥ +0.5. self_ref → jens link transfers iff
   this HELDs.

3. `gamma_jens_jointly_mediated_by_self_ref_and_sigma_action_at_si` —
   joint partial ρ(γ, jens | self_ref + σ_action) ≤ 0.3 (null).
   If FR chain operates at SI, joint partial here is null.

EXPECTED: SUPPORTED (FR chain transfers to SI, the Q-STRUCTURED
sparse-positive analogue). REFUTED would mean SI's
high-γ-jens-driven outcome harm runs through a different
mediator set than FR's anchor-failure path.

Substantive implication if SUPPORTED:
The FR γ-WHY mechanism is class-portable (Q-STRUCTURED +
partial-anchor-failure envs), not env-specific. The chain
operates wherever vanilla has BOTH (a) reduced reward access
at high γ AND (b) per-state Q variance growth — which is the
generalisable mechanism, not "FR-only".

Companion Finding (sibling refutation): see
`finding_asterix_gamma_why_does_not_transfer` (in the same
`minatar_gamma_why_transfer.py` module). Asterix γ=0.999 vanilla
outcome STAYS at 22 across γ — no anchor failure — FR chain
predicted to refute there. The two findings together discriminate
where the FR mechanism applies vs where a different mechanism
(Q-EXPLODED, working-policy-disrupted-by-clip) is at work."""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn_three_conditions.minatar_gamma_why_transfer import (
    gamma_jens_jointly_mediated_by_self_ref_and_sigma_action_at_si,
    gamma_predicts_q_self_reference_at_si,
    q_self_reference_predicts_jens_at_si,
)


EXPECTED: ClusterVerdict = ClusterVerdict.SUPPORTED


BLOCKED_ON: str | None = None


# Empirical (post 2026-05-17 backfill of SI γ ∈ {0.95, 0.99, 0.999}
# with bootstrap_self_reference_fraction):
# - gamma_predicts_q_self_reference_at_si:                  HELD  (p≈0)
# - q_self_reference_predicts_jens_at_si:                   HELD  (p≈0)
# - gamma_jens_jointly_mediated_by_self_ref_and_sigma_action_at_si:
#     HELD null  (p=0.123, joint partial small)
# → 3/3 HELD → SUPPORTED.
#
# The FR γ-WHY chain transfers cleanly to SI. Joint partial after
# {self_ref, σ_action} is small AND p≥0.05 (null prediction
# confirmed). The mediator set characterized at FR — Q-self-
# reference fraction (bootstrap term dominance) plus per-state Q
# variance — is the same mechanism that drives SI's γ→jens path.


BRIDGES: tuple[Bridge, ...] = (
    gamma_predicts_q_self_reference_at_si,
    q_self_reference_predicts_jens_at_si,
    gamma_jens_jointly_mediated_by_self_ref_and_sigma_action_at_si,
)
