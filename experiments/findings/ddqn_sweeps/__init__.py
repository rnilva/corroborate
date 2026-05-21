"""Companion hypothesis module for DDQN: HP-sweep bridges.

The parent `experiments.findings.ddqn` module pins all HPs to the
canonical config (see `_scope.py:DDQN_CANONICAL_REGIME`). Bridges
that *intentionally* vary HPs — n-step, reward-scale, Polyak-τ,
γ-sweep, action-duplicate, network-depth — would have empty extents
there. They live here instead.

`MODULE_SCOPE` is loose: just `~bsuite` exclusion. Each bridge in
this module opts INTO its specific HP-sweep regime via its own
`scope=` parameter (e.g., `reward_scale == 0.1` for the rs-rescue
bridges, `n_step == 10` for the MC-backup falsification, etc.).

The framework's "scope universe is file-level" axiom is honored by
keeping this module separate from the canonical-pinned `ddqn`
module — verdict landscapes from the two are interpretable against
their respective universes."""
from __future__ import annotations

import corroborate.analyses  # pyright: ignore[reportUnusedImport]  # populate registry
import corroborate_rl.dqn.measurables  # pyright: ignore[reportUnusedImport]  # populate measurable registry

import polars as pl

from experiments.findings.ddqn_sweeps import (
    finding_action_duplicate_fa_mechanism,
    finding_asterix_g999_pc_mediator_triangle,
    finding_asterix_g999_smoothness_harm_chain,
    finding_asterix_lambda_a_mechanism,
    finding_cross_env_outcome_regime_g999,
    finding_lambda_a_mediation,
    finding_lambda_a_within_arm_asymmetry,
    finding_pc_cross_env_smoothness,
    finding_snake_clip_ratchet_regime,
    finding_snake_clip_ratchet_regime_g0999,
    finding_synthetic_bias_typeb,
    finding_asterix_gamma_999_harm,
    finding_breakout_gamma_999_help_underpowered,
    finding_ddqn_clip_argmax_harm_chain,
    finding_hp_variance_outcome_refuted,
    finding_metamaze_gamma_amplification,
    finding_rs01_rescue_envelope,
    finding_sigma_over_jens_regime_discriminator,
)
from experiments.findings.ddqn._arms import INTERVENTION as INTERVENTION
from experiments.findings.ddqn._common import CLAIM as CLAIM
from experiments.findings.ddqn_sweeps.bias_correction_hp_variance import (
    BRIDGES as _BIAS_CORRECTION_HP,
)
from experiments.findings.ddqn_sweeps.chain_depth import BRIDGES as _CHAIN_DEPTH
from experiments.findings.ddqn_sweeps.eff_h_intervention import (
    BRIDGES as _EFF_H_INTERVENTION,
)
from experiments.findings.ddqn_sweeps.n_step import BRIDGES as _N_STEP
from experiments.findings.ddqn_sweeps.polyak_tau import BRIDGES as _POLYAK_TAU
from experiments.findings.ddqn_sweeps.rs_rescue import BRIDGES as _RS_RESCUE
from experiments.findings.ddqn_sweeps.within_env_sweeps import BRIDGES as _WITHIN_ENV_SWEEPS
from experiments.findings.ddqn_sweeps.dense_eval_acrobot_k_scaling import (
    ddqn_full_auc_helps_at_acrobot_k16_dense,
    ddqn_full_auc_null_at_acrobot_k4_dense,
)
from experiments.findings.ddqn_sweeps.sigma_over_jens_regime import (
    BRIDGES as _SIGMA_OVER_JENS_REGIME,
)
from experiments.findings.ddqn_sweeps.clip_argmax_harm_mechanism import (
    BRIDGES as _CLIP_ARGMAX_HARM_MECHANISM,
)
from experiments.findings.ddqn_sweeps.q_smoothness_harm_mechanism import (
    BRIDGES as _Q_SMOOTHNESS_HARM_MECHANISM,
)
from experiments.findings.ddqn_sweeps.pc_mediator_triangle_asterix import (
    BRIDGES as _PC_MEDIATOR_TRIANGLE,
)
from experiments.findings.ddqn_sweeps.pc_cross_env_smoothness import (
    BRIDGES as _PC_CROSS_ENV_SMOOTHNESS,
)
from experiments.findings.ddqn_sweeps.action_duplicate_fa_mechanism import (
    BRIDGES as _ACTION_DUPLICATE_FA_MECHANISM,
)
from experiments.findings.ddqn_sweeps.lambda_a_mediation import (
    BRIDGES as _LAMBDA_A_MEDIATION,
)
from experiments.findings.ddqn_sweeps.synthetic_bias_typeb import (
    BRIDGES as _SYNTHETIC_BIAS_TYPEB,
)
from experiments.findings.ddqn_sweeps.outcome_regime_g999_cross_env import (
    BRIDGES as _OUTCOME_REGIME_G999_CROSS_ENV,
)
from experiments.findings.ddqn_sweeps.snake_clip_ratchet_regime import (
    BRIDGES as _SNAKE_CLIP_RATCHET_REGIME,
)
from experiments.findings.ddqn_sweeps.snake_clip_ratchet_regime_g0999 import (
    BRIDGES as _SNAKE_CLIP_RATCHET_REGIME_G0999,
)
from experiments.findings.ddqn_sweeps.jens_reduction_consistency import (
    ddqn_reduces_jens_consistently__canonical_g0999,
)
from experiments.findings.ddqn_sweeps.dormancy_diagnostic import (
    dormancy_gates_jens_at_acrobot_g0999,
)
from experiments.findings.ddqn_sweeps.loop_channel_consistency import (
    ddqn_outcome_opposes_loop_rate__canonical_g0999,
)
from experiments.findings.ddqn_sweeps.smoothness_alignment_consistency import (
    ddqn_outcome_aligns_with_q_smoothness__canonical_g0999,
)
from experiments.findings.ddqn_sweeps import (
    finding_jens_reduction_consistency_g0999,
)
from experiments.findings.ddqn_sweeps import (
    finding_dormancy_diagnostic_acrobot_g0999,
)
from experiments.findings.ddqn_sweeps import (
    finding_loop_channel_consistency_g0999,
)
from experiments.findings.ddqn_sweeps import (
    finding_smoothness_alignment_consistency_g0999,
)


# Loose module scope: only bsuite exclusion. HP-sweep bridges set
# their own scope= predicates per intervention.
MODULE_SCOPE: pl.Expr = ~pl.col('env_name').str.ends_with('-bsuite')


BRIDGES = (
    *_BIAS_CORRECTION_HP,
    *_CHAIN_DEPTH,
    *_EFF_H_INTERVENTION,
    *_N_STEP,
    *_POLYAK_TAU,
    *_RS_RESCUE,
    *_WITHIN_ENV_SWEEPS,
    *_SIGMA_OVER_JENS_REGIME,
    *_CLIP_ARGMAX_HARM_MECHANISM,
    *_Q_SMOOTHNESS_HARM_MECHANISM,
    *_PC_MEDIATOR_TRIANGLE,
    *_PC_CROSS_ENV_SMOOTHNESS,
    *_ACTION_DUPLICATE_FA_MECHANISM,
    *_LAMBDA_A_MEDIATION,
    *_SYNTHETIC_BIAS_TYPEB,
    *_OUTCOME_REGIME_G999_CROSS_ENV,
    *_SNAKE_CLIP_RATCHET_REGIME,
    *_SNAKE_CLIP_RATCHET_REGIME_G0999,
    ddqn_full_auc_helps_at_acrobot_k16_dense,
    ddqn_full_auc_null_at_acrobot_k4_dense,
    ddqn_reduces_jens_consistently__canonical_g0999,
    dormancy_gates_jens_at_acrobot_g0999,
    ddqn_outcome_opposes_loop_rate__canonical_g0999,
    ddqn_outcome_aligns_with_q_smoothness__canonical_g0999,
)


FINDINGS = (
    finding_hp_variance_outcome_refuted,
    finding_metamaze_gamma_amplification,
    finding_rs01_rescue_envelope,
    finding_sigma_over_jens_regime_discriminator,
    finding_asterix_gamma_999_harm,
    finding_breakout_gamma_999_help_underpowered,
    finding_ddqn_clip_argmax_harm_chain,
    finding_asterix_g999_smoothness_harm_chain,
    finding_asterix_g999_pc_mediator_triangle,
    finding_pc_cross_env_smoothness,
    finding_action_duplicate_fa_mechanism,
    finding_cross_env_outcome_regime_g999,
    finding_lambda_a_mediation,
    finding_lambda_a_within_arm_asymmetry,
    finding_asterix_lambda_a_mechanism,
    finding_snake_clip_ratchet_regime,
    finding_snake_clip_ratchet_regime_g0999,
    finding_synthetic_bias_typeb,
    finding_jens_reduction_consistency_g0999,
    finding_dormancy_diagnostic_acrobot_g0999,
    finding_loop_channel_consistency_g0999,
    finding_smoothness_alignment_consistency_g0999,
)


# Measurables consumed by PC-discovery `nodes=(...)` kwargs only
# (NOT referenced by any bridge's `source`/`target`/`scope` polars
# expression). Bridge-referenced measurables auto-backfill via
# the scope walker; this escape hatch covers names that appear
# exclusively inside @analysis kwargs.
REQUIRED_MEASURABLES: tuple[str, ...] = (
    'q_action_std_late',
    'q_argmax_margin_late',
    'q_trajectory_autocorr_late',
    'lambda_a_late',
    'arm_is_baseline',
)
