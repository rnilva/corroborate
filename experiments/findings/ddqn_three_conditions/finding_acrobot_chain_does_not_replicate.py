"""The FR γ → jens mediation chain does NOT replicate at Acrobot
baseline — the cross-env regime contrast.

At FR × MLP × unshaped × baseline (per
`finding_sigma_action_completes_chain`), γ → jens is strongly
amplified (marginal ρ = +0.78) and the {self_ref, σ_action}
chain closes it cleanly (partial ρ = +0.06, NS, 92% reduction).

At Acrobot × MLP × unshaped × baseline (n=120), the SAME chain
test gives a structurally different picture:

  Marginal ρ(γ, jens)              = +0.19  (vs FR's +0.78)
  ρ(γ, q_late_mean)                = -0.72  (vs FR's +0.79)
  ρ(γ, self_ref) / ρ(γ, σ_action)  = +0.72 / +0.63  (similar)
  ρ(self_ref, q_late_mean)         = -1.00  (perfectly anti-corr)

Two qualitative cross-env differences this Finding records:

1. **γ → jens marginal is small at Acrobot.** Below Cohen's
   ±0.3 band; barely statistically significant despite n=120.
   Vanilla anchors at Acrobot (dense per-step reward), so
   the Q-explosion / anchor-failure mechanism that drove FR's
   +0.78 doesn't fire here.

2. **q_late_mean is sign-flipped vs FR.** ρ(γ, q_late) at
   Acrobot is -0.72 (vs FR's +0.79). At Acrobot's
   dense-negative-reward regime, higher γ → less negative Q
   (vanilla finds the goal in fewer steps; accumulated penalty
   is smaller in magnitude). At FR's sparse-positive-reward,
   higher γ → more positive Q (unbounded overestimation
   without an anchor). Same axis, opposite Q-growth regimes.

Two bridges in `jens_reduction_factors.py`:

- `gamma_jens_marginal_small_at_acrobot_mlp` — Spearman
  ρ(γ, jens) at Acrobot baseline, predicted small but
  detectable positive (ρ ≥ +0.1 AND sig). HELDs iff effect is
  present but well below FR's +0.78. Empirical: +0.19.
- `q_late_sign_flipped_with_gamma_at_acrobot_mlp` — Spearman
  ρ(γ, q_late) at Acrobot baseline, predicted strongly negative.
  HELDs at ρ ≤ -0.5 AND p < 0.05.

**What this Finding claims**:
- The FR mediation chain is **regime-specific** — it requires
  the sparse-single-terminal × γ→1 regime where vanilla can't
  anchor. The chain mechanism doesn't generalize to dense-
  reward envs by virtue of the same env-feature axis.
- The same manipulation (γ) drives Q in **structurally
  opposite directions** at FR vs Acrobot (positive overshoot
  vs negative-magnitude shrink). The Q-growth regime is the
  load-bearing axis.

**What this Finding does NOT claim**:
- That DDQN is irrelevant at Acrobot. DDQN's effect on
  jens at Acrobot is small but real (per existing
  bridges in this module). It's the γ → jens AMPLIFICATION
  that doesn't replicate — DDQN's mechanism itself can still
  operate, just on a smaller substrate.
- That Acrobot has zero mediation structure. The marginal is
  weak (+0.19), so any single mediator trivially collapses
  the partial to NS. To test mediation at Acrobot, you'd need
  a regime where γ has substantive jens-amplification — which
  this scope doesn't provide.

Companion: `finding_sigma_action_completes_chain` documents the
FR-baseline chain. The pair encodes the regime-specificity
explicitly via supported-at-FR / not-replicated-at-Acrobot
bridges."""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn_three_conditions.jens_reduction_factors import (
    gamma_jens_marginal_small_at_acrobot_mlp,
    q_late_sign_flipped_with_gamma_at_acrobot_mlp,
)


EXPECTED: ClusterVerdict = ClusterVerdict.SUPPORTED


BLOCKED_ON: str | None = None


BRIDGES: tuple[Bridge, ...] = (
    gamma_jens_marginal_small_at_acrobot_mlp,
    q_late_sign_flipped_with_gamma_at_acrobot_mlp,
)
