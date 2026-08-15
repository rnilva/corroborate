"""Shared foundation for the γ=0.999 DDQN-harm case-study scripts.

Two data surfaces, both checked into the repo (no cloud round-trip):

  1. The binary V-vs-D γ=0.999 panel — `experiments/data/cache/
     hasselt_clean_gpanel.parquet`, the framework's canonical
     graph-panel cache. Carries 8 envs × {vanilla, DDQN} × ~30 seeds
     with the full measurable set. Loaded via `load_g999_panel`.
     This is the γ=0.999 analogue of `hasselt_clean.parquet` (the
     γ=0.99 paper's single cache).

  2. The 5-point α (clip-strength) dose-response panel — frozen to
     `papers/g999_harm/data/alpha_dose_cells.csv`. The dampened-
     greedify intervention arms (α ∈ {0.25, 0.5, 0.75}) live only in
     cloud-evicted corpora, so the per-cell scalars are frozen here
     for offline reproducibility. Loaded via `load_alpha_dose_cells`.

Defines: panel loaders, canonical env order + display labels, arm-key
constants, outcome / mediator column names, shared figure styling.
"""
from __future__ import annotations

import os

# Force CPU before the arms import pulls JAX — these scripts only read
# cached scalars, never compile a kernel, so a GPU alloc is wasted (and
# OOMs noisily on a shared device).
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

from collections.abc import Mapping
from pathlib import Path

import polars as pl

from experiments.findings.hasselt_clean._arms import DDQN_ARM, VANILLA_ARM

__all__ = [
    'load_g999_panel', 'load_alpha_dose_cells',
    'ENV_ORDER', 'env_label',
    'TREATMENT_ARM', 'BASELINE_ARM',
    'GAMMA', 'OUTCOME_PEAK_COL', 'OUTCOME_LATE_COL',
    'OUTCOME_RAW_EPISODES_COL', 'MECH_BIAS_COL', 'REDQ_BIAS_COL',
    'ALPHA_EVAL_COL',
    'ARM_COLOR', 'ARM_LABEL', 'COLOR_HELPS', 'COLOR_HARMS', 'COLOR_NULL',
]

# Framework-canonical arm-key strings (derived from the DDQN intervention).
TREATMENT_ARM: str = DDQN_ARM       # Double DQN  (α = 1.0)
BASELINE_ARM: str = VANILLA_ARM     # Vanilla DQN (α = 0.0)

GAMMA: float = 0.999

_PAPER_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _PAPER_DIR.parents[1]
_GPANEL_CACHE = _REPO_ROOT / 'experiments/data/cache/hasselt_clean_gpanel.parquet'
_ALPHA_CSV = _PAPER_DIR / 'data/alpha_dose_cells.csv'


# ─── data surfaces ─────────────────────────────────────────────────
def load_g999_panel() -> pl.DataFrame:
    """The binary V-vs-D γ=0.999 panel from the gpanel cache.

    Filters `gamma == 0.999`; retains the two canonical arms
    (`baseline`, `double_greedify`). 8 envs are present at γ=0.999 in
    the cache (the MLP-state envs CartPole/Acrobot/MountainCar and the
    3M Jumanji Snake corpus are γ=0.99-only / cloud-blocked there).

    Single source of truth: rebuild via `corroborate hypothesis
    experiments.findings.hasselt_clean --ingest <corpus>` into the
    gpanel cache — never hand-edit the parquet.
    """
    cells = pl.read_parquet(_GPANEL_CACHE)
    return cells.filter(pl.col('gamma').cast(pl.Float64).round(4) == GAMMA)


def load_alpha_dose_cells() -> pl.DataFrame:
    """The frozen 5-point α dose-response panel (Asterix γ=0.999).

    75 cells: α ∈ {0.0, 0.25, 0.5, 0.75, 1.0} × 15 seeds. α=0 is
    vanilla (`canonical_n_eps20_ckpt` baseline), α=1 is full DDQN, the
    intermediate arms are `dampened_double_greedify(α)`. Frozen from
    the cloud-evicted dampened-α corpora; see the module docstring.
    """
    return pl.read_csv(_ALPHA_CSV)


# ─── env catalogue (γ=0.999 panel order) ──────────────────────────
# Asterix first — it is the sole harm env at γ=0.999 and the spine of
# the case study. MinAtar grouped, then Jumanji, then misc/MLP.
ENV_ORDER: tuple[str, ...] = (
    'Asterix-MinAtar',
    'Breakout-MinAtar',
    'Freeway-MinAtar',
    'SpaceInvaders-MinAtar',
    'PacMan-jumanji',
    'MetaMaze-misc',
    'FourRooms-misc',
    'LunarLander-v2-jax',
)

_ENV_LABEL: Mapping[str, str] = {
    'Asterix-MinAtar': 'Asterix',
    'Breakout-MinAtar': 'Breakout',
    'Freeway-MinAtar': 'Freeway',
    'SpaceInvaders-MinAtar': 'SpaceInvaders',
    'PacMan-jumanji': 'PacMan',
    'MetaMaze-misc': 'MetaMaze',
    'FourRooms-misc': 'FourRooms',
    'LunarLander-v2-jax': 'LunarLander',
}


def env_label(env_name: str) -> str:
    """Short display label for an env_name."""
    return _ENV_LABEL.get(env_name, env_name)


# ─── canonical measurable columns ──────────────────────────────────
# Outcome: best-burst raw (undiscounted) mean — peak-of-training.
OUTCOME_PEAK_COL: str = 'eval_best_burst_raw_mean'
# Outcome: late-30%-of-bursts raw mean — steady-state.
OUTCOME_LATE_COL: str = 'eval_late_burst_raw_mean'
# Outcome: per-burst raw episode lists (learning curves).
OUTCOME_RAW_EPISODES_COL: str = 'mc_return_raw_episodes'

# Mech: framework-canonical Jensen bias (clamped max(0, mean(Q − MC))).
MECH_BIAS_COL: str = 'jensen_gap'
# Mech: REDQ relative bias (Q − MC)/|MC| (Chen 2021), late-burst.
REDQ_BIAS_COL: str = 'normalized_bias_redq_late'

# Alpha dose: the discounted best-burst eval the dose-response plots
# (matches the y-axis of the original figure).
ALPHA_EVAL_COL: str = 'eval_best_burst_mean'


# ─── styling ──────────────────────────────────────────────────────
ARM_COLOR: Mapping[str, str] = {
    BASELINE_ARM: '#2166ac',   # vanilla DQN  (blue)
    TREATMENT_ARM: '#b2182b',  # double DQN   (red)
}
ARM_LABEL: Mapping[str, str] = {
    BASELINE_ARM: 'vanilla',
    TREATMENT_ARM: 'DDQN',
}
COLOR_HELPS = '#1a7d3a'
COLOR_HARMS = '#b2182b'
COLOR_NULL = '#888'
