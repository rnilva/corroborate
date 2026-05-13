"""Polarity-moderated bias-correction → outcome chain.

The bias-correction → outcome relationship is polarity-conditional:
positive in SURVIVE envs (reward accumulates with episode length),
null/negative in REACH envs (reward from reaching a goal). The
chain decomposes as:

    bg → argmax_entropy_late → outcome

with the FIRST edge polarity-blind (positive in both) and the
SECOND edge sign-flipped (positive in SURVIVE, negative in REACH).

This Finding's bridges form a disjoint-scope cluster pattern
(HYPOTHESIS_AS_GRAPH.md §3b): same edge identity authored on
polarity-disjoint scopes, with sign-flipped HELD verdicts. The
contrast IS the stratification justification — we author both
halves and the framework's cluster machinery surfaces the
disjoint-scope shape.

Empirical (2026-05-13, DDQN_RELEVANT_SCOPE):

| edge | SURVIVE ρ | REACH ρ |
|---|---|---|
| bg → entropy | +0.30 (HELD) | +0.42 (HELD) |  polarity-blind
| entropy → outcome | +0.36 (HELD pos) | −0.09 (HELD neg) | flipped
| bg → outcome | +0.19 (HELD pos) | −0.03 (HELD null) | flipped

The chain holds in SURVIVE (full mediation via entropy). In
REACH, bg still drives entropy but entropy is anti-correlated
with outcome. Both halves admit; the SUPPORTED verdict on this
Finding means "the polarity-moderation pattern is empirically
corroborated" — a stronger claim than "the link fires in one
half".

**Corrected mechanism (2026-05-13)**: the polarity-moderation is
NOT "entropy moderates by polarity" — entropy is a downstream
symptom. The load-bearing structural fact is that DDQN's
bootstrap target `target_q[argmax_online] ≤ max_a target_q` is
always-downward by construction. In positive-Q envs (SURVIVE)
this clip reduces vanilla's upward overestimation (helps); in
negative-Q envs (REACH) the same downward clip pushes Q more
negative (often hurts, overshooting truth). Per-burst PC on
FourRooms shows entropy → mc is FULLY MEDIATED by bg (residual
+0.036 after conditioning on bg_pb). See
`findings_ddqn_reward_sign_conditional.md`.

Sibling to `finding_hasselt_chain` (the polarity-blind variant).
That Finding's Stage 3 (bg → outcome | jens, env) fires null
because of polarity cancellation; this Finding decomposes the
cancellation."""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn.bias_correction import (
    intervention_predicts_policy_decisiveness__mc_free,
    policy_decisiveness_helps_outcome__survive,
    policy_decisiveness_hurts_outcome__reach,
)


# Canonical-scope re-verification (2026-05-13,
# `findings_canonical_scope_reverification.md`):
# At canonical scope (1M total_steps, env-class-default configs,
# no HP sweeps), the cluster fires REFUTED. The polarity-
# conditional entropy→outcome moderation that appeared in the
# unscoped pool (cross-config HP-mixed cells) doesn't survive
# at converged-canonical: both `policy_decisiveness_helps_outcome__survive`
# and `_hurts_outcome__reach` fire NO_EFFECT, and the bg→outcome
# polarity-disjoint cluster bridges in `bias_correction.py` both
# fire NO_EFFECT too. The earlier SUPPORTED came from under-
# converged cells contributing noise that matched the polarity-
# moderation hypothesis spuriously.
EXPECTED: ClusterVerdict = ClusterVerdict.REFUTED


BLOCKED_ON: str | None = None


BRIDGES: tuple[Bridge, ...] = (
    # Polarity-blind step: bg → entropy
    intervention_predicts_policy_decisiveness__mc_free,
    # Polarity-moderated step (the contrast): entropy → outcome
    policy_decisiveness_helps_outcome__survive,
    policy_decisiveness_hurts_outcome__reach,
)


# Note: the transitive bg → outcome edges (`bg_link_to_outcome__
# survive` and `bg_link_to_outcome_null__reach`) are NOT in this
# Finding's BRIDGES — they're authored on the same disjoint
# polarity scopes but tested empirically to demonstrate the
# transitivity. Under G1 gating, `bg_link_to_outcome__survive`
# fires NO_EFFECT (G1 filters out SURVIVE configs where vanilla's
# jens is below the noise floor; the SURVIVE-polarity outcome
# signal lives mostly in the G1-INACTIVE configs the filter
# removes). The contrast that justifies polarity stratification
# is the entropy → outcome sign flip — those two bridges form the
# disjoint-scope cluster.
