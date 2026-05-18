"""Q magnitude is a residual co-mediator alongside self-reference
at FR × MLP × unshaped × baseline.

Companion to `finding_gamma_jens_via_q_self_reference` (which
refuted full mediation via self-reference alone — residual partial
ρ = +0.33). This Finding asks: what's in the residual?

Two bridges test Q magnitude (`q_late_mean`) as a co-mediator:

1. `gamma_predicts_q_late_residual_at_fr_mlp`
   — Stage 1: ρ(γ, q_late | self_ref) ≥ +0.3 with p < 0.05.
     γ predicts Q magnitude even after partialling out the
     self-reference fraction. Empirical: +0.57 (p<1e-11).

2. `q_late_predicts_jens_residual_at_fr_mlp`
   — Stage 2: ρ(q_late, jens | self_ref) ≥ +0.2 with p < 0.05.
     Q magnitude predicts jens even after partialling out
     self-reference. Empirical: +0.26 (p=0.004). The threshold
     is smaller here (0.2 vs 0.3) — the residual is smaller after
     partialling out the strongest mediator.

If both HELD, Q magnitude is empirically a second mediator path:
γ → q_late → jens, alongside γ → self_ref → jens.

**The mediation budget at FR baseline** (n=120):
- Marginal ρ(γ, jens) = +0.78
- Partial ρ(γ, jens | self_ref) = +0.33 (58% reduction)
- Partial ρ(γ, jens | self_ref + q_late) = +0.22 (72% reduction)

So self_ref + q_late together account for ~72% of γ's effect on
jens; ~28% remains. The residual is NOT explained by these two
mediators. Candidate residual mechanisms:
- Hasselt's σ × √(2 ln K) per-step max-bias scaling
- Bootstrap-chain length 1/(1−γ) — γ amplifies the chain itself,
  independent of value-magnitude or anchor.
- Some interaction effect.

What this Finding CLAIMS:
- Q magnitude is a SECOND empirically-supported mediator at FR
  baseline, alongside self-reference. Both bridges HELD.

What this Finding does NOT claim:
- That q_late + self_ref are JOINTLY sufficient mediators
  (companion measurement: ρ(γ, jens | self_ref + q_late) = +0.22,
  p=0.014 — still a significant residual after both are partialled
  out). This is recorded in the cluster's narrative; not authored
  as a separate bridge yet (would need a multi-Z partial Spearman
  primitive, currently not in the framework's analysis surface).
- Cross-env generalization. Q-magnitude dynamics at Acrobot are
  different — vanilla anchors fine, q_late stays bounded.

Together with `finding_gamma_jens_via_q_self_reference`, the
empirical reading is: at FR γ=0.999 vanilla, γ amplifies jens via
AT LEAST two structurally distinct paths — (a) self-reference
fraction (the bootstrap target's anchor failure) AND (b) Q
magnitude growth — plus a smaller residual ~28% unaccounted for."""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn_three_conditions.jens_reduction_factors import (
    gamma_predicts_q_late_residual_at_fr_mlp,
    q_late_predicts_jens_residual_at_fr_mlp,
)


EXPECTED: ClusterVerdict = ClusterVerdict.SUPPORTED


BLOCKED_ON: str | None = None


# Scope note 2026-05-18. The chain bridges in this Finding
# exclude the `fr_warmup_intervention` corpus from the panel
# (see `finding_sigma_action_completes_chain` for rationale).
# The canonical-corpus panel (n=120) shows both stages HELD.


BRIDGES: tuple[Bridge, ...] = (
    gamma_predicts_q_late_residual_at_fr_mlp,
    q_late_predicts_jens_residual_at_fr_mlp,
)
