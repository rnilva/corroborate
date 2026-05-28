"""Shared foundation for the 5-layer DDQN γ=0.99 paper scripts.

Defines:
  - Canonical panel loader (`load_g099_canonical_panel`)
  - Canonical env order and display labels
  - Outcome metric column names
  - Canonical (clean) mediator column name (Bellman residual, no MC-leak)
  - Shared figure styling (arm colors, env-axis sort)
"""
from __future__ import annotations
from collections.abc import Mapping

import polars as pl

from corroborate.data.panel import Panel
from experiments.findings.ddqn._arms import DDQN_ARM, VANILLA_ARM
from experiments.findings.hasselt_clean._scope import CANONICAL_G099_CORPORA

__all__ = [
    'load_g099_canonical_panel',
    'ENV_ORDER', 'env_label',
    'MECH_BIAS_COL', 'MECH_DORMANCY_COL',
    'OUTCOME_LATE_COL', 'OUTCOME_PEAK_COL', 'OUTCOME_PER_BURST_COL',
    'MEDIATOR_PER_BURST_COL', 'MEDIATOR_PER_BURST_LABEL',
    'TREATMENT_ARM', 'BASELINE_ARM',
    'ARM_COLOR', 'COLOR_HELPS', 'COLOR_HARMS', 'COLOR_NULL', 'COLOR_UNDERPOWERED',
]

# Framework-canonical arm-key strings (derived from the DDQN intervention).
TREATMENT_ARM: str = DDQN_ARM       # Double DQN
BASELINE_ARM: str = VANILLA_ARM     # Vanilla DQN


# ─── canonical panel ──────────────────────────────────────────────
def load_g099_canonical_panel() -> pl.DataFrame:
    """Load hasselt_clean cache scoped to γ=0.99 canonical corpora."""
    panel = Panel.from_cache('experiments.findings.hasselt_clean')
    panel = panel.narrow(
        (pl.col('gamma') == 0.99) & pl.col('corpus').is_in(CANONICAL_G099_CORPORA)
    )
    return panel.cells


# ─── env catalogue ─────────────────────────────────────────────────
# Canonical order used for all per-env figures. MinAtar + Jumanji
# grouped together; MLP-state envs at the end.
ENV_ORDER: tuple[str, ...] = (
    'Asterix-MinAtar',
    'Breakout-MinAtar',
    'Freeway-MinAtar',
    'SpaceInvaders-MinAtar',
    'PacMan-jumanji',
    'Snake-jumanji',
    'MetaMaze-misc',
    'FourRooms-misc',
    'CartPole-v1',
    'Acrobot-v1',
    'MountainCar-v0',
    'LunarLander-v2-jax',
)


_ENV_LABEL: Mapping[str, str] = {
    'Asterix-MinAtar': 'Asterix',
    'Breakout-MinAtar': 'Breakout',
    'Freeway-MinAtar': 'Freeway',
    'SpaceInvaders-MinAtar': 'SpaceInvaders',
    'PacMan-jumanji': 'PacMan',
    'Snake-jumanji': 'Snake',
    'MetaMaze-misc': 'MetaMaze',
    'FourRooms-misc': 'FourRooms',
    'CartPole-v1': 'CartPole',
    'Acrobot-v1': 'Acrobot',
    'MountainCar-v0': 'MountainCar',
    'LunarLander-v2-jax': 'LunarLander',
}


def env_label(env_name: str) -> str:
    """Short display label for an env_name."""
    return _ENV_LABEL.get(env_name, env_name)


# ─── canonical measurable columns ──────────────────────────────────
# Mech layer: the FRAMEWORK's canonical bias measure (cell-level scalar).
# Clamped to max(0, mean(Q − MC)) — 0 may mean unbiased OR mech-dormant;
# pair with `jensen_dormancy_gap` to distinguish.
MECH_BIAS_COL: str = 'jensen_gap'
MECH_DORMANCY_COL: str = 'jensen_dormancy_gap'

# Outcome layer: late-30%-of-bursts raw mean (NOT γ-discounted).
OUTCOME_LATE_COL: str = 'eval_late_burst_raw_mean'
# Outcome layer (alternative): best-burst raw mean (peak-of-training).
OUTCOME_PEAK_COL: str = 'eval_best_burst_raw_mean'

# Mediation layer: per-burst outcome trajectory (γ-discounted return).
OUTCOME_PER_BURST_COL: str = 'mc_return__mean_axis_-1'

# Mediation layer: canonical Bellman-residual mediator (does NOT read
# mc_return; avoids the soft tautology that bias = Q − MC has with
# mc_return-based outcomes).
MEDIATOR_PER_BURST_COL: str = 'bootstrap_gap_magnitude_per_burst'
MEDIATOR_PER_BURST_LABEL: str = 'bootstrap_gap_magnitude (per burst)'


# ─── styling ──────────────────────────────────────────────────────
ARM_COLOR: Mapping[str, str] = {
    'V': '#1f77b4',   # vanilla DQN
    'D': '#d62728',   # double DQN
}
COLOR_HELPS = '#1a8536'
COLOR_HARMS = '#a23'
COLOR_NULL = '#888'
COLOR_UNDERPOWERED = '#d4ad28'
