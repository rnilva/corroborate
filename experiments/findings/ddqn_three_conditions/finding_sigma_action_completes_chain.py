"""σ_action completes the γ → jens mediation chain at FR baseline.

The clean two-mediator chain. After three rounds of investigation
(self-ref alone → 33% residual; q_late as co-mediator → 22%
joint residual; joint with bg_magnitude → ρ=0.97 jens-twin
methodological problem), σ_action emerges as the upstream
residual mediator that closes the chain without confounding.

Empirical at FR × MLP × unshaped × baseline (n=120):

  Stage 1: ρ(γ, σ_action | self_ref)         = +0.59  p<1e-12
  Stage 2: ρ(σ_action, jens | self_ref)      = +0.48  p<1e-7
  Stage 3: ρ(γ, jens | self_ref + σ_action)  = +0.06  p=0.50

All three bridges HELD. The 2-mediator chain {self_ref + σ_action}
accounts for ~92% of γ → jens correlation at this scope (drops
0.78 → 0.06, NS).

Why σ_action and not q_late or bg_magnitude:
- `bootstrap_gap_magnitude` is ρ=0.97 with jens — essentially a
  jens-twin. Conditioning on it removes jens's own variance, not
  upstream-mediated variance. Methodologically unclean.
- `q_late_mean` is co-mediator (supported by
  `finding_q_magnitude_residual_mediator`) but with {self_ref +
  q_late} together still leaving +0.22 residual (refuted by
  `finding_joint_mediation_incomplete`).
- `q_action_std_late` (σ_action) is a per-state across-action Q
  variance measure — structurally upstream of jens, not a
  shadow.

What this Finding CLAIMS:
- Joint full-mediation of γ → jens by {self_ref + σ_action} is
  empirically corroborated at FR × MLP × unshaped × baseline.

What this Finding does NOT claim:
- That σ_action is the UNIQUE upstream of jens. σ_action and
  q_late_mean are highly correlated (ρ=0.93) — they capture
  overlapping Q-growth dimensions. The pair {self_ref,
  q_late} also reaches ρ=0.22 residual; {self_ref, σ_action}
  reaches +0.06. The empirical fit difference is small; the
  causal-vs-correlated distinction needs intervention-based
  evidence to resolve definitively.
- Cross-env generalisation. Different envs have different
  Q-trajectory dynamics; this chain is FR-baseline-specific.
- That the chain is identified IN THE CAUSAL SENSE — partial
  Spearman tests conditional independence, not causal direction.
  The chain interpretation is mechanistic-narrative, supported
  by the empirical mediation structure.

The mediation budget at FR × MLP × unshaped × baseline (n=120):

  Mediator set                       Residual ρ(γ, jens | ·)   Reduction
  (marginal)                          +0.78                     0%
  {self_ref}                          +0.33                     58%
  {q_late}                            +0.56                     28%
  {self_ref, q_late}                  +0.22                     72%
  {self_ref, σ_action}                +0.06   *NS*              92%

So the substantive picture: γ amplifies vanilla jens at FR via
TWO pathways:
1. **Anchor failure** (self_ref): ε-greedy rarely encounters
   reward at long horizons; bootstrap targets become γ × Q with
   no observational grounding.
2. **Per-state Q variance growth** (σ_action): within-state
   across-action Q SD grows with γ (longer chain, more
   per-action max-bias accumulation); higher σ → more jens.

Together they close the chain. DDQN's clip on the bootstrap
target plausibly attacks both: it removes the max-over-actions
bias (reducing σ_action's effect) AND it stabilizes the
bootstrap chain in a way that makes anchor-finding more likely
(reducing self_ref's effect)."""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn_three_conditions.jens_reduction_factors import (
    gamma_jens_jointly_mediated_by_self_ref_and_sigma_action_at_fr_mlp,
    gamma_predicts_sigma_action_residual_at_fr_mlp,
    sigma_action_predicts_jens_residual_at_fr_mlp,
)


EXPECTED: ClusterVerdict = ClusterVerdict.SUPPORTED


BLOCKED_ON: str | None = None


BRIDGES: tuple[Bridge, ...] = (
    gamma_predicts_sigma_action_residual_at_fr_mlp,
    sigma_action_predicts_jens_residual_at_fr_mlp,
    gamma_jens_jointly_mediated_by_self_ref_and_sigma_action_at_fr_mlp,
)
