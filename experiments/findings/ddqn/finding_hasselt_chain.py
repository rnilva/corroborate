"""Hasselt causal chain: arm → bootstrap_gap → jens → outcome,
plus the MC_disc ↔ MC_raw coupling anchor.

Decomposes the literature's compound claim "DDQN reduces Q
overestimation, which causes higher return" into four
falsifiable edges, using the framework's three-tiered edge-
conditioning primitives:

- Stage 0 (`mc_disc_raw_coupled__per_env_jci`):
  outcome-measurement tautology baseline. Per-env Spearman ρ
  between the γ-discounted `eval_best_burst_mean` and the raw
  `eval_best_burst_raw_mean`. Quantifies whether switching
  outcome targets escapes the `jens = Q − MC` algebraic
  identity. Pooled ρ = +0.61 → HELD at threshold 0.5 (corpus
  retains residual tautology even with raw outcome).

- Stage 1 (`algorithm_reduces_bootstrap_gap_magnitude`):
  algorithmic intervention magnitude. Tests whether DDQN's
  decoupling produces networks with smaller per-step argmax-
  disagreement than vanilla's. Per-env Cohen's d, DL-pooled.
  Empirical: cohen_d ≈ −0.6, 9/11 envs negative → expected HELD.

- Stage 2 (`bootstrap_gap_predicts_jens__theorem`):
  direct theorem corroboration. Per-step bias source
  (`bootstrap_gap_magnitude`) integrates to end-state bias
  (`jensen_gap`) along the bootstrap chain — Hasselt's theorem's
  direct empirical prediction. JCI Spearman, env-stratified.
  Empirical: ρ = +0.51, p < 10⁻¹⁰ → expected HELD.

- Stage 3 (`intervention_outcome_link_null__mech_conditioned`):
  link conditioned on mech edge. After partialling out the mech
  edge (jens), does intervention magnitude have residual effect
  on outcome? Per Hasselt's mediation story, the residual
  SHOULD be null (full mediation through jens). JCI partial
  Spearman with z='jensen_gap'. Empirical: ρ_partial = +0.046
  → expected HELD (null confirmed).

The composed verdict uses `composed_verdict` from
`corroborate.graph.causal` — SUPPORTED iff every edge admits.

**Reading the verdict pattern:**

- Stage 1 HELD + Stage 2 HELD + Stage 3 HELD → "full Hasselt
  mediation": intervention reduces algorithmic-disagreement,
  algorithmic-disagreement predicts bias-state, intervention's
  outcome effect is fully mediated by bias-state.
- Stage 1 HELD + Stage 2 HELD + Stage 3 NO_EFFECT → "direct
  effect persists": intervention has outcome effect that
  bypasses bias-state. Refutes pure-mediation Hasselt reading.
- Stage 1 or Stage 2 REFUTED → chain breaks at intervention
  or theorem-prediction level.

**Caveat: ambiguity at Stage 3.** A null residual at Stage 3 is
consistent with BOTH (a) full mediation as Hasselt predicts AND
(b) no link at all. The current bridge cluster includes the
companion `mediation_link_null__jci_partial_clip` test on the
stratum panel, which gives ρ ≈ −0.61 (with vanilla-outcome
control) — a sign-flipped signal, consistent with (b). So the
chain's Stage 3 null is likely "no link" rather than "full
mediation" on this corpus."""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn.bias_correction import (
    algorithm_reduces_bootstrap_gap_magnitude,
    bootstrap_gap_predicts_jens__theorem,
    intervention_outcome_link_null__mech_conditioned,
    mc_disc_raw_coupled__per_env_jci,
)


EXPECTED: ClusterVerdict = ClusterVerdict.SUPPORTED


BLOCKED_ON: str | None = None


BRIDGES: tuple[Bridge, ...] = (
    mc_disc_raw_coupled__per_env_jci,
    algorithm_reduces_bootstrap_gap_magnitude,
    bootstrap_gap_predicts_jens__theorem,
    intervention_outcome_link_null__mech_conditioned,
)
