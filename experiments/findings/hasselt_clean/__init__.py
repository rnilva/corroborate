"""Hasselt-clean: the DDQN bias-clip claim authored as an
explicit directed walk on the framework's causal-graph layer.

Companion to `experiments.findings.ddqn`. Where the original
`ddqn/bias_correction.py` cluster uses `jensen_dormancy_gap` as
a *scope predicate* on the mech bridge (i.e., premise activation
filters which cells the intervention bridge operates on), this
hypothesis authors premise activation as a *first-class upstream
edge* of the chain `jensen_dormancy_gap → jensen_gap →
eval_best_burst_raw_mean`, with `do(DDQN)` attacks on the two
downstream nodes.

The structural difference is the principled form of "theoretical
bounds as upstream edges" the framework supports but did not
previously demonstrate in a substrate hypothesis. The Finding
`finding_hasselt_chain_explicit.py` AND-composes the six bridges;
the chain edges form a connected walk validatable via
`corroborate.graph.causal.is_walk`."""
from __future__ import annotations

import corroborate.analyses  # pyright: ignore[reportUnusedImport]  # populate registry
import corroborate_rl.dqn.measurables  # pyright: ignore[reportUnusedImport]  # populate measurable registry

from experiments.findings.ddqn._arms import INTERVENTION as INTERVENTION
from experiments.findings.ddqn._common import CLAIM as CLAIM
from experiments.findings.hasselt_clean._scope import (
    JDG_AVAILABLE_ENVS as JDG_AVAILABLE_ENVS,
    MODULE_SCOPE as MODULE_SCOPE,
)
from experiments.findings.hasselt_clean.chain import BRIDGES as BRIDGES
from experiments.findings.hasselt_clean import finding_hasselt_chain_explicit


FINDINGS = (
    finding_hasselt_chain_explicit,
)


REQUIRED_MEASURABLES: tuple[str, ...] = (
    'jensen_dormancy_gap',
    'jensen_gap',
    'eval_best_burst_raw_mean',
    'bootstrap_fraction',
    'arm_is_baseline',
)
