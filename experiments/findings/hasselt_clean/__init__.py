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
    # late30 sibling — last 30% (ceil(n/3)) of bursts' raw-return mean.
    # Used by the late30-sibling bridges in `chain.py` / `outcome_consistency.py`
    # so each outcome-using bridge has a verdict under BOTH the peak
    # metric (DDQN-paper-aligned) AND the late-window metric
    # (Agarwal-aligned, stability-narrative). See §3.4-bis of the
    # g099_mediation report.
    'eval_late_burst_raw_mean',
    'bootstrap_fraction',
    'arm_is_baseline',
    # Per-burst MC return projections — staged for exploration of
    # outcome-trajectory dynamics (CV-of-seeds across bursts vs
    # CV-of-episodes within burst). `mc_return__mean_axis_-1` is
    # the discounted per-burst mean; `mc_return__std_axis_-1` the
    # per-burst σ across the K eval episodes (env stochasticity at
    # fixed policy); `mc_return_episode_cv_per_burst` packages
    # σ_eps / |μ_eps| per burst, the trajectory analogue of the
    # scalar `outcome_episode_sigma`. No bridge consumes these yet
    # — staged via REQUIRED_MEASURABLES per the framework escape
    # hatch (CLAUDE.md "Optional REQUIRED_MEASURABLES attribute").
    'mc_return__mean_axis_-1',
    'mc_return__std_axis_-1',
    'mc_return_episode_cv_per_burst',
    'outcome_episode_sigma',
    # Per-burst non-tautological mediator candidates — required for
    # `dynamic_partial_spearman` mediation trajectories. `bg` is the
    # canonical Bellman wedge; `mean_per_state_cumulative_bias` is the
    # per-state Bellman residual; the two disagree-rate proxies surface
    # the rate vs magnitude decomposition (cf.
    # `findings_per_burst_mediation_trajectory`).
    'bootstrap_gap_magnitude_per_burst',
    'mean_per_state_cumulative_bias_per_burst',
    'bootstrap_disagree_rate_per_burst',
    'greedy_match_per_burst',
    # Pure (truly non-tautological wrt mc_return) per-burst candidates
    # surfaced by the earlier per-env mediator audit (see
    # `experiments/figures/g099_per_env_mediator_and_outcome.png`):
    # state_repeat_rate is the canonical mediator at Freeway / Asterix /
    # PacMan; argmax_entropy at Breakout / FourRooms; n_unique +
    # entropy for state-coverage breadth.
    'argmax_entropy_per_burst',
    'state_hash_n_unique_per_burst',
    'state_hash_entropy_per_burst',
    'state_repeat_rate_window64_per_burst',
    # REDQ-style normalized bias (Chen et al. 2021): bias / |E[Q^π]|.
    # Scale-invariant form of the cumulative bias mediator — pair
    # with `mean_per_state_cumulative_bias_*` to disentangle absolute
    # bias from relative bias in cross-env aggregation. Both versions
    # read `mc_return_from_step` directly → tautological with
    # mc_return-based outcomes, diagnostic-only.
    'normalized_bias_redq_late',
    'normalized_bias_redq_per_burst',
    # Additional non-tautological per-burst candidates surfaced by the
    # registered-measurables audit. Q-side (Bellman-tautological wrt
    # the bias premise but axis-distinct): q_argmax_margin, q_action_std,
    # q_autocorr, q_lambda_a. State/policy-side: state_conditional
    # _argmax_entropy. Bellman residual: bootstrap_disagree_gap_conditional.
    # PC will test whether the expanded set still has pstate_bias
    # as a sufficient single-Z separator.
    'q_argmax_margin_per_burst',
    'q_action_std_per_burst',
    'q_autocorr_per_burst',
    'q_lambda_a_per_burst',
    'state_conditional_argmax_entropy_per_burst',
    'bootstrap_disagree_gap_conditional_per_burst',
)
