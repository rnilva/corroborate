"""Hasselt-clean: the DDQN bias-clip claim authored as an
explicit directed walk on the framework's causal-graph layer,
with cross-env consistency sign-tests on the intervention
edges (the principled tool for heterogeneous-stratum claims).

Companion to `experiments.findings.ddqn`. Where the original
`ddqn/bias_correction.py` cluster uses `jensen_dormancy_gap` as
a *scope predicate* on the mech bridge (premise activation
filters cells), this hypothesis authors premise activation as a
*first-class upstream edge* of the chain `jensen_dormancy_gap →
jensen_gap → eval_best_burst_raw_mean`, with `do(DDQN)` attacks
on the two downstream nodes.

The Finding `finding_hasselt_chain_explicit.py` AND-composes
the four bridges; the chain's edges form a connected walk
validatable via `corroborate.graph.causal.is_walk`.

Subdirectory `_failed_pool/` preserves the original
random-effects pool attempt for B3/B4 — pedagogically anchored
to the lesson that cross-env pooling requires exchangeability
RL envs structurally lack. Its bridges fire NO_EFFECT under
the framework's PI-based discipline; the Finding there pins
REFUTED for drift-tracking honesty."""
from __future__ import annotations

import corroborate.analyses  # pyright: ignore[reportUnusedImport]  # populate registry
import corroborate_rl.dqn.measurables  # pyright: ignore[reportUnusedImport]  # populate measurable registry

from experiments.findings.ddqn._arms import INTERVENTION as INTERVENTION
from experiments.findings.ddqn._common import CLAIM as CLAIM
from experiments.findings.hasselt_clean._scope import (
    JDG_AVAILABLE_ENVS as JDG_AVAILABLE_ENVS,
    MODULE_SCOPE as MODULE_SCOPE,
)
from experiments.findings.hasselt_clean.chain import BRIDGES as _CHAIN_BRIDGES
from experiments.findings.hasselt_clean.outcome_consistency import (
    BRIDGES as _OUTCOME_BRIDGES,
)
from experiments.findings.hasselt_clean._failed_pool.chain_pool import (
    BRIDGES as _FAILED_POOL_BRIDGES,
)
from experiments.findings.hasselt_clean import (
    finding_ddqn_outcome_consistency,
    finding_hasselt_chain_explicit,
)
from experiments.findings.hasselt_clean._failed_pool import (
    finding_chain_pool_inadequate,
)


BRIDGES = (*_CHAIN_BRIDGES, *_OUTCOME_BRIDGES, *_FAILED_POOL_BRIDGES)


FINDINGS = (
    finding_hasselt_chain_explicit,
    finding_ddqn_outcome_consistency,
    finding_chain_pool_inadequate,
)


REQUIRED_MEASURABLES: tuple[str, ...] = (
    'jensen_dormancy_gap',
    'jensen_gap',
    'eval_best_burst_raw_mean',
    'bootstrap_fraction',
    'arm_is_baseline',
)
