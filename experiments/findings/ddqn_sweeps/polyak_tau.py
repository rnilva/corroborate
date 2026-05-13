"""Polyak-do(τ) bridges — exposed under ddqn_sweeps so they can fire
on cells with target_sync.tau > 0 (excluded by ddqn canonical scope).

The bridge functions themselves still live in
`experiments.findings.ddqn.mediation` (they share helper imports and
docstring context with the other staleness bridges); only their
exposure under `BRIDGES` moves here."""
from __future__ import annotations

from experiments.findings.ddqn.mediation import (
    staleness_amplifies_ddqn_outcome__sparse_goal_polyak,
    staleness_does_not_amplify_ddqn_outcome__survival_polyak,
)


BRIDGES = (
    staleness_amplifies_ddqn_outcome__sparse_goal_polyak,
    staleness_does_not_amplify_ddqn_outcome__survival_polyak,
)
