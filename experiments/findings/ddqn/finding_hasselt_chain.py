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


# Canonical-scope state post-2026-05-14 backfill (`mc_return`
# discounted + Q-shape measurables backfilled via `--ingest-all`):
# Stage 1 (`algorithm_reduces_bootstrap_gap_magnitude`) drifted
# from HELD to POW_INSUF. Per-env Cohen's d is wildly
# heterogeneous (I²=0.94): Asterix -2.6 / SlidingTile -1.3 /
# Freeway -0.7 / MountainCar -0.4 (DDQN reduces bg) but Snake
# +0.5 / PacMan +0.8 / SpaceInvaders +2.6 (DDQN INCREASES bg).
# Pooled d = -0.08, p=0.85, CI=[-0.90, +0.74] — null.
#
# This matches memory `findings_clip_to_trained_q_propagation`:
# the per-step downward clip is deterministic, but trained-Q
# comparisons can flip across envs (finite-training residual).
# The substrate-level "DDQN reduces bg universally" claim is
# heterogeneous at canonical; pooled-d is null.
#
# Chain consequence: Stage 1 POW_INSUF → chain UNDERPOWERED at
# the Finding level. Stages 0 / 2 / 3 still HELD. The chain
# CAN be re-supported if Stage 1 is re-authored per-env (with
# heterogeneity-aware verdict) or replaced with a more direct
# mech-magnitude bridge that doesn't pool across heterogeneous
# envs.
EXPECTED: ClusterVerdict = ClusterVerdict.UNDERPOWERED


BLOCKED_ON: str | None = (
    'algorithm_reduces_bootstrap_gap_magnitude is POW_INSUF due to '
    'I²=0.94 cross-env heterogeneity (DDQN reduces bg on Asterix/'
    'SlidingTile/Freeway/MountainCar; INCREASES bg on Snake/PacMan/'
    'SpaceInvaders/Breakout). Pooled-d null. Need either '
    'heterogeneity-aware verdict on this bridge OR a per-env '
    'stratified shape to re-support the chain. Substantive '
    'finding, not data deficiency.'
)


BRIDGES: tuple[Bridge, ...] = (
    mc_disc_raw_coupled__per_env_jci,
    algorithm_reduces_bootstrap_gap_magnitude,
    bootstrap_gap_predicts_jens__theorem,
    intervention_outcome_link_null__mech_conditioned,
)
