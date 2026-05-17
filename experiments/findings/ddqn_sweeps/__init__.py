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
    finding_asterix_g999_pc_mediator_triangle,
    finding_asterix_g999_smoothness_harm_chain,
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
    ddqn_full_auc_helps_at_acrobot_k16_dense,
    ddqn_full_auc_null_at_acrobot_k4_dense,
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
)
