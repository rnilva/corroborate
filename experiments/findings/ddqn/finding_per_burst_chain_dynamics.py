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
#   bg_per_burst → outcome_per_burst: NO_EFFECT (NULL_EFFECT).
#     The pooled ρ is near zero — the bridge's NULL prediction
#     is corroborated AT THE POOLED LEVEL even though per-env
#     reveals heterogeneity (Acrobot strong NEGATIVE ρ ≈ −0.8,
#     mostly cancelled when pooled with sign-mixed other envs).
# Cluster verdict = SUPPORTED post commit `919f73f` (framework
# now correctly stamps NO_EFFECT (NULL_EFFECT) under
# predicted_direction='null' as corroboration). Pre-fix the
# cluster fired REFUTED because the null-prediction
# admit-equivalence wasn't recognised — EXPECTED was pinned
# REFUTED to match the buggy stamping. The literal-bridge-verdict
# reading is: all 3 bridges admit their respective predictions
# (HELD, HELD, NO_EFFECT under null prediction) → SUPPORTED.
# The per-env heterogeneity Acrobot caveat is a separate
# question that a future per-env bridge could test specifically.
EXPECTED: ClusterVerdict = ClusterVerdict.SUPPORTED


BLOCKED_ON: str | None = None


BRIDGES: tuple[Bridge, ...] = (
    bg_per_burst_predicts_entropy_per_burst,
    entropy_per_burst_predicts_outcome_per_burst,
    bg_per_burst_link_to_outcome,
)
