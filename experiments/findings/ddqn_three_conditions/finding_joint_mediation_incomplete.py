"""Joint mediation of γ → jens by {self_ref, q_late} is INCOMPLETE
at FR baseline.

The explicit refutation of joint full-mediation. Companion to
the two upstream Findings:
- `finding_gamma_jens_via_q_self_reference` (refuted full
  mediation by self_ref alone — residual +0.33)
- `finding_q_magnitude_residual_mediator` (supported q_late as
  a residual co-mediator)

This Finding asks: do self_ref AND q_late TOGETHER fully mediate
γ → jens? Single bridge:

`gamma_jens_jointly_mediated_by_self_ref_and_q_late_at_fr_mlp`
   — predicted null: multi-Z partial Spearman
     ρ(γ, jens | self_ref, q_late) ≈ 0.

Empirical at FR × MLP × unshaped × baseline (n=120):
    ρ_partial = +0.22, p = 0.014

Verdict: NO_EFFECT (significant residual). Joint full-mediation
is REFUTED — even with both mediators conditioned, ~28% of γ's
effect on jens remains unexplained.

What this Finding CLAIMS (matching the empirical reading):
- Joint full-mediation by {self_ref, q_late} is refuted.

What this Finding does NOT claim:
- That self_ref or q_late are unimportant — they account for
  ~72% of γ's effect together (correlation drops 0.78 → 0.22).
- That the residual 28% is any specific other mechanism. The
  candidates remain untested:
  - Hasselt's σ × √(2 ln K) per-step max-bias scaling
  - Bootstrap-chain length `1/(1−γ)` — γ amplifies the chain
    itself even after value-magnitude and anchor are partialled.
  - Interaction effects between mediators not captured by
    linear (partial-rank) regression.

The mediation budget at FR baseline (n=120):

  Mediator set                 Residual ρ(γ, jens | ·)   Reduction
  (none, marginal)              +0.78                     0%
  {self_ref}                    +0.33                     58%
  {q_late}                      +0.56                     28%
  {self_ref, q_late}            +0.22                     72%
  {self_ref, q_late, ?}         ?                         ?

The final row is the future research question. To close the
chain, an additional mediator measurable would need to capture
the residual γ-driven Hasselt-style or chain-length effect."""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn_three_conditions.jens_reduction_factors import (
    gamma_jens_jointly_mediated_by_self_ref_and_q_late_at_fr_mlp,
)


EXPECTED: ClusterVerdict = ClusterVerdict.REFUTED


BLOCKED_ON: str | None = None


BRIDGES: tuple[Bridge, ...] = (
    gamma_jens_jointly_mediated_by_self_ref_and_q_late_at_fr_mlp,
)
