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
    finding_per_burst_chain_dynamics,
    finding_polarity_conditional_chain,
    finding_reach_bias_link,
)
from experiments.findings.ddqn._arms import INTERVENTION as INTERVENTION
from experiments.findings.ddqn._common import CLAIM as CLAIM
from experiments.findings.ddqn._scope import MODULE_SCOPE as MODULE_SCOPE
from experiments.findings.ddqn.bias_correction import BRIDGES as _BIAS_CORRECTION
from experiments.findings.ddqn.mediation import BRIDGES as _MEDIATION
from experiments.findings.ddqn.outcome_scope import BRIDGES as _OUTCOME_SCOPE
from experiments.findings.ddqn.within_env import BRIDGES as _WITHIN_ENV


# HP-sweep bridges (n_step, reward-scale rescue) moved to sibling
# module `experiments.findings.ddqn_sweeps`. The canonical module
# scope (`MODULE_SCOPE = ~bsuite & DDQN_CANONICAL_REGIME`) admits
# zero cells for HP-sweep bridges, so they belong in a separate
# hypothesis with a relaxed scope universe.
BRIDGES = (
    *_OUTCOME_SCOPE,
    *_WITHIN_ENV,
    *_BIAS_CORRECTION,
    *_MEDIATION,
)


FINDINGS = (
    finding_hasselt_chain,
    finding_polarity_conditional_chain,
    finding_per_burst_chain_dynamics,
    finding_reach_bias_link,
)


# Pre-populate measurables that have no bridge consumer yet but
# are needed for the per-burst two-channel decomposition
# (`findings_ddqn_reward_sign_conditional.md`). Validated against
# the @measurable registry at `_validate_hypothesis`.
REQUIRED_MEASURABLES: tuple[str, ...] = (
    'q_per_burst',
    # Q-channel mediator candidates (no bridge consumes yet;
    # `scripts/q_channel_mediator_search.py` tests which one
    # explains the partial ρ(q, mc | bg) ≈ +0.55 residual.
    'q_action_std_late',
    'q_argmax_margin_late',
    'argmax_persistence_late',
    'q_max_temporal_cv_late',
    'q_mc_calibration_pearson',
    # Per-burst variants for within-cell mediator testing
    # (`findings_two_channel_cross_corpus.md` walk-back).
    'q_argmax_margin_per_burst',
    'q_action_std_per_burst',
)
