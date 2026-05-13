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
with outcome — uncertain policies fail to commit to a goal.
Both halves admit; the SUPPORTED verdict on this Finding means
"the polarity-moderation pattern is empirically corroborated"
— a stronger claim than "the link fires in one half".

Sibling to `finding_hasselt_chain` (the polarity-blind variant).
That Finding's Stage 3 (bg → outcome | jens, env) fires null
because of polarity cancellation; this Finding decomposes the
cancellation."""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn.bias_correction import (
    bg_link_to_outcome__survive,
    bg_link_to_outcome_null__reach,
    intervention_predicts_policy_decisiveness__mc_free,
    policy_decisiveness_helps_outcome__survive,
    policy_decisiveness_hurts_outcome__reach,
)


EXPECTED: ClusterVerdict = ClusterVerdict.SUPPORTED


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
