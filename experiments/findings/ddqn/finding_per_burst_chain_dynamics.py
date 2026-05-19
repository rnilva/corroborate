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
#   entropy_per_burst → outcome_per_burst: NO_EFFECT — the null
#     prediction failed: pooled |ρ| ≥ 0.2 (an effect WAS observed
#     when none was predicted, the xpass case per
#     `core.hypothesis.PredictedDirection`). Per-env heterogeneity
#     (Acrobot strong negative ρ ≈ −0.8 vs sign-mixed cohort)
#     drives a pooled magnitude that exceeds the null band.
#   bg_per_burst → outcome_per_burst: NO_EFFECT — same xpass
#     failure mode at this cohort.
# Cluster verdict: REFUTED per the framework convention. With
# the null-prediction bridges' bodies returning HELD only when
# |ρ| < null_threshold (per `partial_spearman_null_verdict`),
# the cluster admits only when each member's prediction is
# confirmed; the two NO_EFFECT members propagate to REFUTED.
# Pre-919f73f-revert, the cluster fired REFUTED for the same
# reason; the 919f73f band-aid temporarily made NO_EFFECT under
# null admit-equivalent, which we've now reverted to the
# framework's documented intent.
EXPECTED: ClusterVerdict = ClusterVerdict.REFUTED


BLOCKED_ON: str | None = None


BRIDGES: tuple[Bridge, ...] = (
    bg_per_burst_predicts_entropy_per_burst,
    entropy_per_burst_predicts_outcome_per_burst,
    bg_per_burst_link_to_outcome,
)
