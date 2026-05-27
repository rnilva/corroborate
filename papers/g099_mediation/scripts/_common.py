"""Shared utilities for figure generation in papers/g099_mediation.

Defines:
  - The CLEAN (tautology-corrected) per-burst mediator set
  - Panel loading helper scoped to canonical γ=0.99 corpora
  - Status enum + dsep% helpers
"""
from __future__ import annotations
import polars as pl
import numpy as np

from corroborate.data.panel import Panel
from experiments.findings.hasselt_clean._scope import CANONICAL_G099_CORPORA


# Tautology-clean per-burst mediator candidates.
# EXCLUDES: mean_per_state_cumulative_bias_per_burst (reads mc_return_from_step),
#           jensen_dormancy_gap_per_burst (reads mc_return), mc_*_per_burst.
CLEAN_MEDIATORS = (
    # Bellman residual side (don't read MC)
    ('bg_magnitude',    'bootstrap_gap_magnitude_per_burst'),
    ('bg_disagree',     'bootstrap_disagree_rate_per_burst'),
    ('bg_disagree_cond','bootstrap_disagree_gap_conditional_per_burst'),
    ('greedy_match',    'greedy_match_per_burst'),
    # Action/policy side
    ('argmax_ent',      'argmax_entropy_per_burst'),
    ('state_cond_ent',  'state_conditional_argmax_entropy_per_burst'),
    # Q-shape (CNN-only — populated at MinAtar + LL)
    ('q_argmax_margin', 'q_argmax_margin_per_burst'),
    ('q_action_std',    'q_action_std_per_burst'),
    ('q_autocorr',      'q_autocorr_per_burst'),
    ('q_lambda_a',      'q_lambda_a_per_burst'),
    # State-coverage
    ('state_n_unq',     'state_hash_n_unique_per_burst'),
    ('state_ent',       'state_hash_entropy_per_burst'),
    ('state_repeat',    'state_repeat_rate_window64_per_burst'),
)
TAUTOLOGICAL_BLOCKLIST = frozenset({
    'mean_per_state_cumulative_bias_per_burst',
    'mean_per_state_cumulative_bias_late',
    'jensen_gap', 'jensen_dormancy_gap',
    'jensen_dormancy_gap_per_burst', 'jensen_bias_per_eps',
    'mc_return__mean_axis_-1', 'mc_return__std_axis_-1',
    'mc_return_episode_cv_per_burst', 'mc_cv_per_burst',
    'mc_variance_per_burst', 'log_mc_cv_per_burst', 'log_mc_variance_per_burst',
})
STATUS_RANK = {
    'CONSISTENT_DIRECTION': 0,
    'WEAK_TIME_VARYING': 1,
    'SIGN_FLIP_DETECTED': 2,
    'UNDERPOWERED_BURSTS': 3,
}


def load_g099_canonical_panel() -> pl.DataFrame:
    """Load hasselt_clean cache scoped to γ=0.99 canonical corpora."""
    panel = Panel.from_cache('experiments.findings.hasselt_clean')
    panel = panel.narrow(
        (pl.col('gamma') == 0.99) & pl.col('corpus').is_in(CANONICAL_G099_CORPORA)
    )
    return panel.cells


def absorb(rho_marg: float, rho_part: float) -> float:
    """Per-stratum absorption %: 1 - |partial| / |marginal|."""
    if (np.isnan(rho_marg) or np.isnan(rho_part) or abs(rho_marg) < 1e-9):
        return float('nan')
    return (1 - abs(rho_part) / abs(rho_marg)) * 100
