"""DDQN's outcome help at FR γ=0.999 is NOT proportional to bias reduction.

Surface claim that this Finding refutes:
    "DDQN reduces jens (Q-overestimation) → smaller bias →
    better policy → better outcome. The bigger the bias
    reduction, the bigger the outcome benefit."

Empirical pattern at FR × MLP[64,64] × unshaped × γ=0.999 across
k_eff ∈ {4, 8, 12, 16}:

    k_eff   Δ_jens_mean   Δ_out_mean   n
    4       -8.76         +0.82        180
    8       -33.66        +0.81         30
    12      -59.56        +0.75         30
    16      -87.20        +0.50         30

Δ_jens grows ~10× monotonically as k_eff goes 4→16. Δ_out is
approximately flat (0.82 → 0.50, slight DECREASE). Across all
~120 seed × k_eff pairs, ρ(Δ_jens, Δ_out) marginal = +0.075 (NS).

The Finding's three bridges decompose this pattern:

1. `ddqn_jens_reduction_amplified_by_k_eff_at_fr_g999_mlp` —
   the mechanism HELD: per-k_eff d_jens ≤ -0.5 everywhere AND
   |mean_diff(k=16)| ≥ 3 × |mean_diff(k=4)|. DDQN's clip bites
   harder at higher k_eff, as Hasselt's √(2 ln K) factor predicts.

2. `ddqn_outcome_help_constant_across_k_eff_at_fr_g999_mlp` —
   the outcome HELD on direction but NOT on amplification:
   per-k_eff d_out ≥ +0.3 everywhere AND ratio max/min |Δ_out|
   ≤ 3 (empirically ~1.6). DDQN helps every k_eff; benefit is
   flat across the k_eff axis.

3. `jens_reduction_does_not_predict_outcome_at_fr_g999_mlp` —
   the link is null: marginal ρ(Δ_jens, Δ_out) ≤ 0.2 across
   seed × k_eff pairs. Mechanism magnitude does not predict
   outcome magnitude.

Composed reading: **DDQN's value at FR γ=0.999 is BINARY, not
GRADED.** The argmax-rescue threshold matters; bias-reduction
magnitude beyond the threshold is incidental. Once vanilla's
Q-explosion would corrupt the argmax, DDQN's clip prevents the
corruption — but the size of that prevention (more or less Q
suppression) doesn't translate to a graded outcome improvement.

Substantive implications:
- The Hasselt-2016 narrative "bigger bias reduction → bigger
  policy improvement" holds in DIRECTION but not in MAGNITUDE at
  this scope.
- Outcome benefit cap is set by policy-rescue payoff (whether
  the argmax is corrupted at all), not by mechanism strength.
- For finding regimes where DDQN's outcome help DOES scale with
  bias reduction, the substrate-author needs a discriminator that
  separates "always-needed argmax rescue" (binary) from
  "more-rescue-helps" (graded). Reward density, terminal-reward
  rarity, or σ_action margin to the rescue threshold are
  candidate axes.

What this Finding does NOT claim:
- That bias reduction is causally irrelevant — only that its
  magnitude doesn't translate at THIS scope (FR × MLP × γ=0.999
  × unshaped × k_eff ∈ {4..16}).
- That a different env or γ regime would show the same null. The
  cross-env mediator-heterogeneity Finding (memory:
  `findings_ddqn_mediator_heterogeneity`) already documents that
  different envs use different channels.
- That a finer-grained outcome metric (e.g., recovery time, or
  per-burst trajectory shape) wouldn't track Δ_jens magnitude.
  `eval_best_burst_raw_mean` is the framework's default; a
  scale-sensitive outcome could behave differently.

Related findings:
- `finding_gamma_amplification_anchor_gated` — at FR γ=0.999,
  vanilla Q-explodes and DDQN's mechanism fires hardest. This
  Finding picks up where that one leaves off: yes mech fires
  hardest, no outcome doesn't track magnitude.
- `findings_two_translation_regimes` (memory) — the two-regime
  framing already established that single-number outcome metrics
  collapse phase structure. This Finding adds: even at the cell
  level, mech magnitude does not scale outcome magnitude at
  Regime 1 (FR-style)."""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn_three_conditions.jens_reduction_factors import (
    ddqn_jens_reduction_amplified_by_k_eff_at_fr_g999_mlp,
    ddqn_outcome_help_constant_across_k_eff_at_fr_g999_mlp,
    jens_reduction_does_not_predict_outcome_at_fr_g999_mlp,
)


# REFUTED on the merged cache post 2026-05-18 consolidation. The
# third bridge (`jens_reduction_does_not_predict_outcome_at_fr_
# g999_mlp`) migrated from the off-limits seed-paired-Δ
# `partial_spearman_paired` (pre-consolidation, reported ρ ≈ +0.075
# NS) to the canonical per-cell partial Spearman primitive
# (ρ(jensen_gap, eval_best_burst_raw_mean | q_late_mean) ≈ -0.45,
# n=300, p<0.001 → NO_EFFECT). The canonical RL-substrate primitive
# surfaces a STRONG negative link that the per-pair-Δ form did not.
# The "no jens→outcome link at FR γ=0.999" claim does NOT survive
# the canonical primitive. Methodology follow-up is open (see
# module docstring) — substrate authors should decide whether the
# original decoupling-via-binary-rescue framing needs revision OR
# whether a different mediation question (e.g. partial-stratified
# by k_eff levels) is the right substrate-honest test.
EXPECTED: ClusterVerdict = ClusterVerdict.REFUTED


BLOCKED_ON: str | None = None


BRIDGES: tuple[Bridge, ...] = (
    ddqn_jens_reduction_amplified_by_k_eff_at_fr_g999_mlp,
    ddqn_outcome_help_constant_across_k_eff_at_fr_g999_mlp,
    jens_reduction_does_not_predict_outcome_at_fr_g999_mlp,
)
