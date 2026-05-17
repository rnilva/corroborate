"""γ-WHY mediation chain — Q self-reference is a STRONG but
PARTIAL mediator. The FULL-mediation hypothesis is refuted.

The companion to `finding_gamma_amplification_anchor_gated`
(circumstantial co-occurrence: γ-amp + vanilla outcome
collapse). This Finding tests the mediation chain via the
directly-measured Q-explosion signature
`bootstrap_self_reference_fraction`.

Three bridges at FR × MLP × unshaped × baseline cells (n=120):

1. `gamma_predicts_q_self_reference_at_fr_mlp` — HELD.
   ρ(γ, self_ref) = +0.87. γ↑ → more bootstrap-target
   self-reference.

2. `q_self_reference_predicts_jens_at_fr_mlp` — HELD.
   ρ(self_ref, jens) = +0.93. More self-reference → larger
   bias.

3. `gamma_jens_mediated_by_q_self_reference_at_fr_mlp` — NO_EFFECT.
   ρ(γ, jens | self_ref) = +0.33 (p=0.0003). NOT null after
   conditioning — γ has a residual direct effect on jens
   beyond what self-reference explains.

**The composed empirical reading**: self-reference is a STRONG
mediator but NOT a complete one. The marginal correlation
ρ(γ, jens) = +0.87 drops to +0.33 after partialling out
self-reference — a 62% reduction. So self-reference carries
roughly two-thirds of γ's effect on jens at this scope, but the
other third remains. Candidate mechanisms for the residual:
- Hasselt's `σ × √(2 ln K)` per-step max-bias: not captured by
  self_ref_frac, which only measures the r-vs-γQ ratio at the
  target.
- Bootstrap-chain length `1/(1−γ)`: longer effective horizon
  → more chained max-bias accumulation, independent of whether
  the target is self-referential.
- Some Q-magnitude property correlated with γ.

What this Finding CLAIMS (matching the empirical reading):
- The full-mediation hypothesis is refuted. Self-reference is
  PARTIAL.

What this Finding does NOT claim:
- That self-reference is unimportant — stages 1 & 2 HELD with
  ρ ≈ 0.87 and 0.93. The mediator IS load-bearing.
- That the residual 33% effect IS Hasselt's `1/(1−γ)` or
  `σ × √(2 ln K)` — these are candidate stories not directly
  tested. To discriminate, additional mediator bridges (e.g.,
  per-cell Q magnitude, bootstrap chain length proxy) are
  needed.
- Cross-env generalization. At Acrobot, vanilla anchors fine
  (per `finding_gamma_amplification_anchor_gated`); the chain
  at Acrobot likely doesn't run the same way.

The honest scientific reading is more interesting than full
mediation would have been: DDQN's effect at FR γ=0.999 is
SIMULTANEOUSLY (a) "we save vanilla from self-referential
collapse" AND (b) "we reduce some other γ-amplified bias path".
Both contribute.

Companion: `finding_gamma_amplification_anchor_gated` (cross-env
discriminator via vanilla outcome) — corroborates the
self-reference path; doesn't constrain the residual path."""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn_three_conditions.jens_reduction_factors import (
    gamma_jens_mediated_by_q_self_reference_at_fr_mlp,
    gamma_predicts_q_self_reference_at_fr_mlp,
    q_self_reference_predicts_jens_at_fr_mlp,
)


EXPECTED: ClusterVerdict = ClusterVerdict.REFUTED


BLOCKED_ON: str | None = None


BRIDGES: tuple[Bridge, ...] = (
    gamma_predicts_q_self_reference_at_fr_mlp,
    q_self_reference_predicts_jens_at_fr_mlp,
    gamma_jens_mediated_by_q_self_reference_at_fr_mlp,
)
