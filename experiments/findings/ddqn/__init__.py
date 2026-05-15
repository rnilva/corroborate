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
    finding_channel_decomposition,
    finding_cross_env_mediation,
    finding_hasselt_chain,
    finding_mediation_polarity_conditional,
    finding_per_burst_chain_dynamics,
    finding_polarity_conditional_chain,
    finding_reach_bias_link,
    finding_tautology_baseline_chain,
    finding_three_gate_scope_outcome_held,
)
from experiments.findings.ddqn._arms import INTERVENTION as INTERVENTION
from experiments.findings.ddqn._common import CLAIM as CLAIM
from experiments.findings.ddqn._scope import MODULE_SCOPE as MODULE_SCOPE
from experiments.findings.ddqn.bias_correction import BRIDGES as _BIAS_CORRECTION
from experiments.findings.ddqn.bias_correction_xenv import (
    bias_correction_dose_response__xenv_arm_diff,
    bias_correction_dose_response__xenv_arm_diff_loo_robust,
)
from experiments.findings.ddqn.mediation import BRIDGES as _MEDIATION
from experiments.findings.ddqn.outcome_scope import BRIDGES as _OUTCOME_SCOPE
from experiments.findings.ddqn.polarity_conditional_mediation import (
    BRIDGES as _POLARITY_MEDIATION,
)
from experiments.findings.ddqn.cross_env_mediation import (
    ddqn_outcome_scales_with_bg_frac_active__xenv,
    ddqn_outcome_scales_with_jens_reduction__xenv,
    ddqn_outcome_scales_with_jens_reduction__xenv_loo_robust,
)
from experiments.findings.ddqn.q_shape_mediation import BRIDGES as _Q_SHAPE
from experiments.findings.ddqn.q_suppression_translation import (
    ddqn_q_suppression_tracks_outcome_translation__xenv,
    ddqn_q_suppression_tracks_outcome_translation__xenv_loo_robust,
)
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
    bias_correction_dose_response__xenv_arm_diff,
    bias_correction_dose_response__xenv_arm_diff_loo_robust,
    *_MEDIATION,
    *_POLARITY_MEDIATION,
    *_Q_SHAPE,
    ddqn_outcome_scales_with_jens_reduction__xenv,
    ddqn_outcome_scales_with_jens_reduction__xenv_loo_robust,
    ddqn_outcome_scales_with_bg_frac_active__xenv,
    ddqn_q_suppression_tracks_outcome_translation__xenv,
    ddqn_q_suppression_tracks_outcome_translation__xenv_loo_robust,
)


FINDINGS = (
    finding_hasselt_chain,
    finding_polarity_conditional_chain,
    finding_per_burst_chain_dynamics,
    finding_reach_bias_link,
    finding_three_gate_scope_outcome_held,
    finding_channel_decomposition,
    finding_tautology_baseline_chain,
    finding_mediation_polarity_conditional,
    finding_cross_env_mediation,
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
    # Alternative bg aggregations — the mean reduction
    # (`bootstrap_gap_magnitude`) hides arm differences by
    # averaging mostly-zeros from convergence. `frac_active`
    # (rate of online/target argmax disagreement) and `q99`
    # (tail magnitude) recover signal. See memory
    # `findings_bg_not_causally_manipulated_at_canonical`.
    'bootstrap_gap_frac_active',
    'bootstrap_gap_q99',
    # State-conditional argmax measurables (2026-05-15).
    # Distinguishes state-differentiated policy (high MI) from
    # Q-flat noise (low MI + high marginal entropy) — algorithm-
    # agnostic decomposition, currently consumed by the DDQN
    # policy-channel claim per memory
    # `findings_ddqn_mediator_heterogeneity`. Requires state_hash
    # on each env: `bucket_hash` (vector envs) or
    # `image_downsample_hash` (MinAtar). Envs without a registered
    # state_hash return NaN (sentinel).
    'state_conditional_argmax_entropy_late',
    'mutual_info_state_argmax_late',
    # Env-level disc-raw alignment for outcome-translation scope.
    # Where r > 0.7, raw vs disc outcome are interchangeable; below
    # that, the choice matters and bridges must commit.
    'env_disc_raw_alignment',
    # Per-burst-window outcome measurables (2026-05-15) — additions
    # to the canonical `eval_best_burst_mean` family. `best-burst`
    # picks a single peak; `full_auc` integrates the trajectory;
    # `late_burst` reads the convergence-region endpoint. The
    # MetaMaze canonical-verify + Acrobot k=16 findings showed
    # best-burst structurally hides translation in slow-converging
    # / late-collapse-prone envs (memories
    # `findings_metamaze_translates_after_eval_power` and
    # `findings_per_burst_acrobot_k_sweep`).
    'eval_full_auc_mean',
    'eval_full_auc_raw_mean',
    'eval_late_burst_mean',
    'eval_late_burst_raw_mean',
)
