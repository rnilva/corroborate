"""WHY does γ amplify DDQN's reduction of `jensen_gap` at FR?

Two candidate mechanisms — the same surface pattern, different
underlying paths:

- **Story A (Hasselt 1/(1−γ) amplification)**: vanilla's per-
  step max-bias compounds over the longer effective horizon as
  γ → 1. The amplification is a generic property of any env
  with substantive per-step bias.
- **Story B (vanilla-degeneracy at γ → 1)**: at sparse-single-
  terminal envs and high γ, vanilla can't find the goal at all.
  MC ≈ 0 means no observational anchor for Q, so Q grows
  unbounded. DDQN's clip keeps Q anchored regardless. The
  amplification is contingent on env structure that produces
  degeneracy.

The cluster composes three observational (WHAT) bridges into a
WHY claim — γ-amplification at FR is paired with vanilla
anchor failure at γ=0.999, and the amplification does not
replicate at envs where vanilla's anchor is preserved.

Bridges:

1. `ddqn_reduction_amplified_by_gamma__fr_mlp_k4_unshaped`
   — the surface observation: 47× amplification at FR × MLP ×
   k=4 from γ=0.99 to 0.999. HELD.

2. `vanilla_anchor_collapses_with_gamma_at_fr_mlp`
   — within-FR anchor observation: vanilla's eval outcome
   collapses from 1.0 at γ=0.99 (finds goal every episode) to
   0.19 at γ=0.999 (~42% of cells score 0). ρ(γ, outcome)
   strongly negative at FR baseline arm.

3. `vanilla_anchor_preserved_with_gamma_at_acrobot_mlp`
   — cross-env discriminator: at Acrobot × MLP × unshaped,
   vanilla outcome stays at -79.2 (γ=0.99) vs -73.8 (γ=0.999).
   Both finite — vanilla anchors at both γ. ρ(γ, outcome) ≈
   +0.34 (small POSITIVE — γ↑ slightly improves vanilla because
   the negative-step reward has more effective horizon to
   optimize).

The opposite-sign γ-effect is the discriminator: at FR (sparse-
single-terminal-positive) γ↑ collapses vanilla; at Acrobot
(dense-negative-reward) γ↑ slightly helps vanilla. Same γ
manipulation, opposite signs — driven by env reward structure,
not by a generic Hasselt amplification.

The composed reading: γ-amplification of DDQN's jens reduction
at FR IS PAIRED WITH vanilla anchor collapse at FR γ=0.999 AND
does NOT replicate at Acrobot where vanilla's anchor is
preserved. Story B (anchor-failure-gated) is more consistent
with this evidence than Story A (Hasselt's 1/(1−γ) generic
amplification).

What this Finding CLAIMS:
- The empirical co-occurrence (γ-amp + anchor-collapse at FR;
  no γ-amp + anchor-preserved at Acrobot) supports the
  anchor-failure interpretation over the
  generic-Hasselt-amplification interpretation at this corpus's
  scope.

What this Finding does NOT claim:
- That Hasselt's 1/(1−γ) factor has zero contribution — only
  that it's not the load-bearing path on this evidence.
- That MountainCar / other envs would follow the same pattern.
  Acrobot is the only non-FR env in this Finding's discriminator.
- That FR is "broken" at γ=0.999 — just that vanilla is mostly
  degenerate there. The pattern is a feature of (env structure
  × γ), not a sweep failure.

Related bridges that would strengthen the claim (follow-up):
- Add an Acrobot γ-amplification observational bridge that HELDs
  on "no 3× amplification at Acrobot" — directly contrasts with
  bridge (1).
- Author a within-cell partial-Spearman bridge: ρ(γ, jens |
  vanilla_outcome) — does γ have RESIDUAL predictive power on
  jens after partialling out anchor failure? If ρ_partial ≈ 0,
  anchor failure fully mediates. If ρ_partial substantial,
  Hasselt's amplification path also contributes."""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn_three_conditions.jens_reduction_factors import (
    ddqn_reduction_amplified_by_gamma__fr_mlp_k4_unshaped,
    vanilla_anchor_collapses_with_gamma_at_fr_mlp,
    vanilla_anchor_preserved_with_gamma_at_acrobot_mlp,
)


EXPECTED: ClusterVerdict = ClusterVerdict.UNDERPOWERED


BLOCKED_ON: str | None = None


# Walk-back 2026-05-18. After tightening the FR chain bridges to
# `total_steps == 200000` (proper single-horizon scope), the
# Acrobot cross-env discriminator bridge `vanilla_anchor_preserved_
# with_gamma_at_acrobot_mlp` dropped from ρ=+0.34 sig (pooled
# 200k + 1M cells, n=240) to ρ=+0.14 NS (200k only, n=180). The
# pooled-horizon evidence was inflated by horizon-effect.
#
# Single-horizon honest read: vanilla outcome at Acrobot γ=0.99
# (-78.4) vs γ=0.999 (-75.9) is roughly flat — consistent with
# "anchor preserved" (vanilla doesn't dramatically collapse) but
# can't statistically reject ρ=0 at n=180.
#
# The two FR-leg bridges (γ-amplification of jens, vanilla anchor
# collapse at FR) remain HELD. The cross-env discriminator that
# would let us claim "FR's amplification is anchor-failure-gated"
# instead of "Hasselt 1/(1−γ) generic amplification" is now too
# weak to land at threshold. UNDERPOWERED matches the empirical
# state — Finding can flip back to SUPPORTED if we get more
# Acrobot baseline cells at 200k canonical (current n=90/arm
# per γ at canonical mlp_deep × unshaped).


BRIDGES: tuple[Bridge, ...] = (
    ddqn_reduction_amplified_by_gamma__fr_mlp_k4_unshaped,
    vanilla_anchor_collapses_with_gamma_at_fr_mlp,
    vanilla_anchor_preserved_with_gamma_at_acrobot_mlp,
)
