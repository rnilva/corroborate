"""Scope-cluster Finding: staleness→jens→outcome mediation
survives BOTH rank-based AND linear identifications on MinAtar
intermediate-sync envs.

Sibling pair to `finding_bg_jens_mediation_robust__reach` for
the second canonical mediation claim in the substrate:

  target_staleness_late_mediates_outcome__minatar_intermediate_sync
    (HELD as signed-negative):
    ρ_partial(staleness, outcome | jens) ≤ −0.2 per-env on
    Asterix-MinAtar + Breakout-MinAtar across sync ∈
    {500, 1500, 3000}. Predicted-negative.

  target_staleness_late_linearity_holds__minatar_intermediate_sync
    (HELD): `mediation_dowhy.linearity_status == RELIABLE` on the
    same scope.

The cluster's contribution is the same as the bg/jens pair:
robust mediation findings agree across rank-based AND linear
identifications. The pair documents that the staleness mediation
result is robust to the v10 multicollinearity failure mode (or
flags REFUTED if the failure mode IS active on this scope, in
which case the rank-based answer stands alone).

EMPIRICAL state pin: linearity sibling just added; first
evaluation determines admission. SUPPORTED is the predicted
state under the substrate's mediation-recipe lesson; BLOCKED_ON
tracks the pending evaluation."""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn.mediation import (
    target_staleness_late_linearity_holds__minatar_intermediate_sync,
    target_staleness_late_mediates_outcome__minatar_intermediate_sync,
)


# EMPTY_EXTENT on the canonical ddqn cache: the MinAtar
# intermediate-sync scope (sync_period ∈ {500, 1500, 3000} on
# Asterix/Breakout-MinAtar) is empirically empty here. Cells
# live in the `minatar_1M` corpus; ingest into the cache (or
# co-ingest as sibling under the ddqn hypothesis) to surface
# the cluster's substantive verdict.
EXPECTED: ClusterVerdict = ClusterVerdict.EMPTY_EXTENT


BLOCKED_ON: str | None = (
    'minatar_1M corpus not co-ingested into the canonical ddqn '
    'cache — Asterix/Breakout-MinAtar at sync_period ∈ '
    '{500, 1500, 3000} cells are absent. Ingest via '
    '`python -m corroborate catalogue experiments/data '
    'experiments/probes/` to locate the corpus, then sweep-merge '
    'into the ddqn cache scope. Predicted SUPPORTED under the '
    'mediation-recipe lesson (rank-based + linear should both '
    'admit when collinearity is mild) once cells land.'
)


BRIDGES: tuple[Bridge, ...] = (
    target_staleness_late_mediates_outcome__minatar_intermediate_sync,
    target_staleness_late_linearity_holds__minatar_intermediate_sync,
)
