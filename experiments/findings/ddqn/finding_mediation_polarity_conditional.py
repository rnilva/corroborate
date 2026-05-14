"""Polarity-conditional mediation cluster — REFUTED at canonical.

The hypothesis: DDQN's outcome benefit is mediated through jens
ONLY on REACH-polarity envs; on SURVIVE-polarity envs no
mediated link exists. Three polarity-disjoint bridges test the
substantive disambiguation of the global pooled
`intervention_outcome_link_null__mech_conditioned` ambiguity
("full mediation" vs "no link"):

  bg_outcome_link_held_negative__reach_envs:
    Marginal ρ(bg, outcome) on REACH (pol<-0.3) — predicted
    NEGATIVE & HELD (DDQN reduces bg → outcome improves).
    EMPIRICAL: ρ_pool = -0.177, p=0.014, n_strata=5 → NO_EFFECT
    (sign correct, magnitude below +0.2 threshold).

  bg_outcome_fully_mediated_by_jens__reach_envs:
    Partial ρ(bg, outcome | jens) on REACH — predicted NULL
    (full mediation through jens).
    EMPIRICAL: ρ_partial = -0.143, p=0.051 → HELD as null.

  bg_outcome_link_null__survive_envs:
    Marginal ρ(bg, outcome) on SURVIVE — predicted NULL
    (no link to mediate; DDQN's clip propagation flips per
    `findings_clip_to_trained_q_propagation`).
    EMPIRICAL: ρ_pool = +0.038, p=0.61 → HELD as null.

Cluster verdict: REFUTED. Bridge 1's NO_EFFECT propagates to
the Finding-level refutation. The pattern from per-env smoke
data (MountainCar marginal -0.73, MetaMaze -0.26, Freeway -0.24)
doesn't survive Fisher-z pooling at the cohort level — the
Acrobot-MetaMaze-FourRooms-MountainCar-Snake mixture washes out
to -0.18, just below the +0.2 substantive threshold.

**Honest disambiguation conclusion**: at canonical's REACH-
cohort n (5 envs), we CAN'T cleanly establish the marginal
bg→outcome link as HELD. Without that, the partial-null doesn't
disambiguate full-mediation from no-link. The "full mediation
on REACH" hypothesis is plausible per per-env data but
underpowered at the pool level.

The Finding documents this honestly as REFUTED. The substantive
claim "DDQN's outcome benefit on REACH is jens-mediated" remains
plausible but unverified at canonical scope; would need more
REACH-polarity envs to clear the pool threshold. Memory:
`findings_canonical_scope_reverification` already notes
outcome-translation claims don't survive canonical pooled
scope — this Finding adds the polarity-conditional rescue
attempt that also doesn't survive."""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn.polarity_conditional_mediation import (
    bg_outcome_fully_mediated_by_jens__reach_envs,
    bg_outcome_link_held_negative__reach_envs,
    bg_outcome_link_null__survive_envs,
)


EXPECTED: ClusterVerdict = ClusterVerdict.REFUTED


BLOCKED_ON: str | None = None


BRIDGES: tuple[Bridge, ...] = (
    bg_outcome_link_held_negative__reach_envs,
    bg_outcome_fully_mediated_by_jens__reach_envs,
    bg_outcome_link_null__survive_envs,
)
