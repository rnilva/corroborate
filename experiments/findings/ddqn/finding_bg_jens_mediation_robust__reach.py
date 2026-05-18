"""Scope-cluster Finding: bg→jens→outcome mediation survives
BOTH rank-based AND linear identifications on REACH-polarity envs.

CLAUDE.md §"Mediation recipe" prescribes `partial_spearman` as
the canonical mediation primitive (rank-based, multicollinearity-
robust). The salvaged `mediation_dowhy` provides a
typed-linearity-diagnostic counterpart — its `linearity_status ==
RELIABLE` corroborates that the linear-mediation assumption is
ALSO defensible at this scope (i.e., direct/total agree on sign +
indirect_proportion in [0, 1] → no multicollinearity-induced
sign-flip).

The cluster pairs the canonical bridge with its linearity
diagnostic. SUPPORTED iff both admit (the framework's
`composed_verdict` semantics):

  bg_outcome_fully_mediated_by_jens__reach_envs (HELD as null):
    ρ_partial(bg, outcome | jens) ≈ 0 — rank-based "chain
    closes" reading. EMPIRICAL: ρ ≈ −0.10, p ≈ 0.23,
    n_strata=4.

  bg_outcome_mediation_linearity_holds__reach_envs (HELD):
    `mediation_dowhy.linearity_status == RELIABLE` — linear
    decomposition gives a coherent same-sign answer in [0, 1].

Cluster shape: the substrate's methodological lesson (CLAUDE.md
§"Mediation recipe") is that ROBUST mediation findings agree
across rank-based AND linear methods. A bridge claiming
mediation via partial_spearman alone is exposed to the failure
mode where partial_spearman would correctly say "mediates" but
linear decomposition fails — corroboration narrows the
methodology-robustness gap.

REFUTED here would mean: the canonical bridge HELDs (rank-based
mediation supported) BUT linearity is broken — the v10 FR γ-WHY
failure mode applies at REACH scope. The Finding-level REFUTED
documents the methodological lesson concretely: "rank-based
partial Spearman is the trustworthy primitive at REACH; linear
DoWhy's magnitudes can't be read at face value here."

EMPIRICAL state pin: the linearity sibling has not yet been
evaluated on the canonical cache (sibling bridge just added);
EXPECTED is pinned to the pre-evaluation state with BLOCKED_ON.
First evaluation will reveal whether linearity holds on REACH;
this Finding's verdict tracks that result."""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn.polarity_conditional_mediation import (
    bg_outcome_fully_mediated_by_jens__reach_envs,
    bg_outcome_mediation_linearity_holds__reach_envs,
)


# SUPPORTED on canonical ddqn cache: linear DoWhy decomposition
# on REACH scope returns total_ate=-3282.6, direct_ate=-940.5
# (same sign as total → no multicollinearity flip), indirect
# proportion = 0.71 (in [0, 1] → no suppression artifact).
# `linearity_status == RELIABLE`. Paired with the canonical
# `bg_outcome_fully_mediated_by_jens__reach_envs` (HELD as null
# at ρ≈-0.10): mediation survives BOTH rank-based and linear
# identifications. Substrate-level evidence that the
# mediation-recipe lesson's RELIABLE case applies to REACH.
EXPECTED: ClusterVerdict = ClusterVerdict.SUPPORTED


BLOCKED_ON: str | None = None


BRIDGES: tuple[Bridge, ...] = (
    bg_outcome_fully_mediated_by_jens__reach_envs,
    bg_outcome_mediation_linearity_holds__reach_envs,
)
