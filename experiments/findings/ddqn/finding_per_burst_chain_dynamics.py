"""Per-burst chain dynamics: `bg_per_burst → entropy_per_burst → outcome_per_burst`.

Tests the bias-correction → policy → outcome chain at per-burst
granularity (windowed training-step measurables) instead of
cell-level (full-trajectory or late-50% reductions). Three
bridges authored on the per-(cell, burst) unfolded panel via
`per_burst_jci_spearman`:

- `bg_per_burst_predicts_entropy_per_burst` (predicted positive)
- `entropy_per_burst_predicts_outcome_per_burst` (predicted null)
- `bg_per_burst_link_to_outcome` (predicted null)

Why this exists: the cell-level chain analysis suggested a
polarity-moderated `bg → entropy → outcome` story
(`finding_polarity_conditional_chain`). The per-burst
diagnostic showed env-by-burst heterogeneity that the cell-
level aggregation hides: Acrobot (REACH-polarity) has POSITIVE
ρ(entropy_pb, mc_pb) at every burst (opposite of the polarity-
hypothesis), SpaceInvaders (SURVIVE-polarity) has NEGATIVE.
Per-burst JCI Spearman pools over (env, burst) rows preserving
this heterogeneity for the pool.

Expected pool verdicts (from diagnostic):
- bg→entropy_pb pooled: positive but smaller than cell-level
  (averaged over the per-burst env heterogeneity)
- entropy_pb→mc_pb pooled: near zero (sign-mixed by env cancels)
- bg→mc_pb pooled: near zero

The bridges' predicted directions encode this — null where the
diagnostic suggested cancellation. If verdicts come back null,
the polarity-conditional cell-level chain WAS an aggregation
artifact. If bg→entropy_pb fires HELD, the intervention's
effect on entropy is real at per-burst granularity too (which
the diagnostic supports: most envs have positive per-burst
ρ(bg_pb, ent_pb))."""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn.bias_correction import (
    bg_per_burst_link_to_outcome,
    bg_per_burst_predicts_entropy_per_burst,
    entropy_per_burst_predicts_outcome_per_burst,
)


# Empirical result on rebuilt cache (2026-05-13, n=4730 cells):
#   bg_per_burst → entropy_per_burst: HELD (positive, predicted)
#   entropy_per_burst → outcome_per_burst: HELD (null-pooled,
#     consistent with sign-mixed by env)
#   bg_per_burst → outcome_per_burst: NO_EFFECT (predicted null,
#     but pool has a REAL signal — mostly driven by Acrobot's
#     strong NEGATIVE ρ ≈ −0.8 per burst, opposite of Hasselt
#     direction. The null prediction was empirically wrong.)
# Cluster verdict = REFUTED because the third bridge violates
# its predicted_direction. Substantively the per-burst chain
# DOES corroborate the bg → entropy step, but the chain breaks
# at the outcome step in an env-specific (not polarity-clean)
# way. EXPECTED pins the empirical REFUTED state.
EXPECTED: ClusterVerdict = ClusterVerdict.REFUTED


BLOCKED_ON: str | None = None


BRIDGES: tuple[Bridge, ...] = (
    bg_per_burst_predicts_entropy_per_burst,
    entropy_per_burst_predicts_outcome_per_burst,
    bg_per_burst_link_to_outcome,
)
