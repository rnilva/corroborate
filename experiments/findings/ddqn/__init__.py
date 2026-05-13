"""DDQN measurement graph — bridges organized by CLAIM.

Migrated from the single-file `ddqn_universe.py` (deleted
2026-05-12). Each claim file holds the bridges that share a
theoretical unit; the four private files (`_arms`, `_scope`,
`_verdicts`, `_common`) hold sub-module-shared constants.

The hypothesis runner reads four module-level names from this
package: `CLAIM` (outermost claim for endogeneity gating),
`MODULE_SCOPE` (AND-combined into every bridge's scope),
`BRIDGES` (the closure of bridge declarations evaluated against
the per-module cache `experiments/data/cache/ddqn.parquet`), and
`FINDINGS` (cluster-shaped claims authored against this
hypothesis's post-eval graph)."""
from __future__ import annotations

import corroborate.analyses  # pyright: ignore[reportUnusedImport]  # populate registry
import corroborate_rl.dqn.measurables  # pyright: ignore[reportUnusedImport]  # populate measurable registry

from experiments.findings.ddqn import (
    finding_hasselt_chain,
    finding_metamaze_gamma_amplification,
    finding_polarity_conditional_chain,
    finding_reach_bias_link,
    finding_rs01_rescue_envelope,
)
from experiments.findings.ddqn._arms import INTERVENTION as INTERVENTION
from experiments.findings.ddqn._common import CLAIM as CLAIM
from experiments.findings.ddqn._scope import MODULE_SCOPE as MODULE_SCOPE
from experiments.findings.ddqn.bias_correction import BRIDGES as _BIAS_CORRECTION
from experiments.findings.ddqn.mediation import BRIDGES as _MEDIATION
from experiments.findings.ddqn.n_step import BRIDGES as _N_STEP
from experiments.findings.ddqn.outcome_scope import BRIDGES as _OUTCOME_SCOPE
from experiments.findings.ddqn.rs_rescue import BRIDGES as _RS_RESCUE
from experiments.findings.ddqn.within_env import BRIDGES as _WITHIN_ENV


BRIDGES = (
    *_OUTCOME_SCOPE,
    *_WITHIN_ENV,
    *_RS_RESCUE,
    *_N_STEP,
    *_BIAS_CORRECTION,
    *_MEDIATION,
)


FINDINGS = (
    finding_hasselt_chain,
    finding_polarity_conditional_chain,
    finding_reach_bias_link,
    finding_metamaze_gamma_amplification,
    finding_rs01_rescue_envelope,
)
