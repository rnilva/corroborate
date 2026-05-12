"""DDQN measurement graph — bridges organized by CLAIM.

Migrated from the single-file `ddqn_universe.py` (deleted
2026-05-12). Each claim file holds the bridges that share a
theoretical unit; the four private files (`_arms`, `_scope`,
`_verdicts`, `_common`) hold sub-module-shared constants.

The hypothesis runner reads three module-level names from this
package: `CLAIM` (outermost claim for endogeneity gating),
`MODULE_SCOPE` (AND-combined into every bridge's scope), and
`BRIDGES` (the closure of bridge declarations evaluated against
the per-module cache `experiments/data/cache/ddqn.parquet`)."""
from __future__ import annotations

import corroborate.analyses  # pyright: ignore[reportUnusedImport]  # populate registry
import corroborate_rl.dqn.measurables  # pyright: ignore[reportUnusedImport]  # populate measurable registry

from experiments.findings.ddqn._arms import INTERVENTION as INTERVENTION
from experiments.findings.ddqn._common import CLAIM as CLAIM
from experiments.findings.ddqn._scope import MODULE_SCOPE as MODULE_SCOPE
from experiments.findings.ddqn.adaptive_controller import BRIDGES as _ADAPTIVE
from experiments.findings.ddqn.bias_correction import BRIDGES as _BIAS_CORRECTION
from experiments.findings.ddqn.mediation import BRIDGES as _MEDIATION
from experiments.findings.ddqn.n_step import BRIDGES as _N_STEP
from experiments.findings.ddqn.outcome_scope import BRIDGES as _OUTCOME_SCOPE
from experiments.findings.ddqn.rs_rescue import BRIDGES as _RS_RESCUE
from experiments.findings.ddqn.within_env import BRIDGES as _WITHIN_ENV


BRIDGES = (
    *_OUTCOME_SCOPE,
    *_ADAPTIVE,
    *_WITHIN_ENV,
    *_RS_RESCUE,
    *_N_STEP,
    *_BIAS_CORRECTION,
    *_MEDIATION,
)


# Findings — imported AFTER BRIDGES so finding-modules' lookup of
# this package (`HYPOTHESIS.BRIDGES`) resolves to the fully-built
# tuple. Cluster-shaped claims that this hypothesis's post-eval
# graph carries; the framework's renderer + Finding consumer
# iterate this tuple.
from experiments.findings.ddqn import (  # noqa: E402
    finding_metamaze_gamma_amplification as _FINDING_METAMAZE,
    finding_reach_bias_link as _FINDING_REACH,
    finding_rs01_rescue_envelope as _FINDING_RS01,
)


FINDINGS = (
    _FINDING_REACH,
    _FINDING_METAMAZE,
    _FINDING_RS01,
)
