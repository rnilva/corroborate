"""Per-env solve thresholds — what does it mean for an agent to
have "solved" each env in the catalogue.

**Critical units note.** The framework's eval (`rl/dqn/eval.py:60`)
computes the *discounted* Monte-Carlo return: `mc_return =
Σ_t γ^t r_t`. Literature solve thresholds (gymnasium docs, Y&T
2019, Osband 2019) are typically *raw* episodic return. The
threshold values pinned below have been **converted to discounted
units at γ=0.99** so they align with the framework's eval.
Conversions:

- For envs with constant per-step reward (CartPole = +1/step,
  Acrobot/MountainCar = -1/step), the formula is exact:
    raw_return = R * L for episode length L.
    discounted_return = R * (1 - γ^L) / (1 - γ).
  Setting raw_return at solve threshold T (= length-T-equivalent)
  gives discounted threshold = R * (1 - γ^T) / (1 - γ).
- For sparse-terminal-reward envs (Catch, DeepSea, MNISTBandit),
  the discount factor scales the terminal reward by γ^L. Adjust
  thresholds slightly down from the score-0.5 raw value.
- For variable-per-step-reward envs (MinAtar), the conversion is
  approximate: discounted ≈ raw × (1 - γ^L_avg) / L_avg / (1 - γ),
  computed at typical episode length. Documented as approximate.

Convergence is env-specific: a slope-based "training plateaued"
test is too soft. Instead we pin a per-env *outcome* threshold
sourced from one of three tiers:

- **literature** — converted from gymnasium / bsuite / paper-
  canonical raw thresholds via the discount-factor formula.
- **derived** — chosen as a fraction of a literature DQN baseline
  (MinAtar envs use 50% of Young & Tian 2019), then converted.
- **sample_relative** — defined relative to the corpus.
- **absent** — no defensible threshold.

The `is_solved(env_name, outcome_value, *, table=...)` predicate
maps an outcome scalar to a bool; cells where the agent's outcome
(in discounted units) ≥ threshold count as solved.

Threshold semantics: applied to whichever outcome path the caller
specifies. The framework's `eval_final_mean` and
`eval_best_burst_mean` both record discounted return."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal


type ThresholdConfidence = Literal[
    'literature', 'derived', 'sample_relative', 'absent',
]


@dataclass(frozen=True, slots=True)
class SolveThreshold:
    """One env's solve threshold + provenance.

    `threshold` — outcome value at/above which the cell counts as
      solved. None when confidence is 'absent' (no defensible value).
    `source` — short citation key documenting where the threshold
      came from (e.g., 'gymnasium-docs', 'osband-2019',
      'young-tian-2019-50pct').
    `confidence` — quality tier of the threshold (see module
      docstring).
    `outcome_path_assumed` — which outcome path the threshold is
      written for. Mostly `'eval_final_mean'` for stable
      convergence; using a softer outcome path with the same
      threshold is permissive (catastrophic-forgetting envs would
      pass best_burst but fail final_mean)."""
    env_name: str
    threshold: float | None
    source: str
    confidence: ThresholdConfidence
    outcome_path_assumed: str = 'eval_final_mean'


# ============ The table ============

# Tier 1 (literature, converted to discounted at γ=0.99).
#
# Classics — exact conversion (constant per-step reward):
#   CartPole-v1: raw 475 → 100 * (1 - 0.99^475) ≈ 99.16
#   Acrobot-v1: raw -100 → -100 * (1 - 0.99^100) ≈ -63.40
#   MountainCar-v0: raw -110 → -100 * (1 - 0.99^110) ≈ -67.33
#
# bsuite — sparse terminal reward; γ^L attenuation small for short
# episodes. Conservative slight downward adjustment from 0.5.
# DiscountingChain has its own internal discount; threshold left
# at the env's near-max raw value since the framework's discount
# compounds with the env's.

_TIER_1: tuple[SolveThreshold, ...] = (
    SolveThreshold(
        'CartPole-v1', threshold=99.0,
        source='gymnasium-docs-475-discounted-gamma-0.99',
        confidence='literature',
    ),
    SolveThreshold(
        'Acrobot-v1', threshold=-63.4,
        source='gymnasium-docs-(-100)-discounted-gamma-0.99',
        confidence='literature',
    ),
    SolveThreshold(
        'MountainCar-v0', threshold=-67.3,
        source='gymnasium-docs-(-110)-discounted-gamma-0.99',
        confidence='literature',
    ),
    SolveThreshold(
        'Catch-bsuite', threshold=0.45,
        source='osband-2019-score-0.5-discounted-approx',
        confidence='literature',
    ),
    SolveThreshold(
        'DeepSea-bsuite', threshold=0.45,
        source='osband-2019-score-0.5-discounted-approx',
        confidence='literature',
    ),
    SolveThreshold(
        'DiscountingChain-bsuite', threshold=1.0,
        source='osband-2019-near-max-1.1-env-internal-discount',
        confidence='literature',
    ),
    SolveThreshold(
        'MemoryChain-bsuite', threshold=0.45,
        source='osband-2019-score-0.5-discounted-approx',
        confidence='literature',
    ),
    SolveThreshold(
        'MNISTBandit-bsuite', threshold=0.5,
        source='osband-2019-score-0.5-bandit-no-discount',
        confidence='literature',
    ),
    SolveThreshold(
        'UmbrellaChain-bsuite', threshold=0.45,
        source='osband-2019-score-0.5-discounted-approx',
        confidence='literature',
    ),
)

# Tier 2 (derived, converted to discounted at γ=0.99).
#
# Y&T 2019 reported DQN raw episodic returns for MinAtar at 10M
# training steps. We pin 50% of baseline as the "decent" threshold
# (raw), then convert to discounted via:
#   discounted ≈ raw × (1 - γ^L_avg) / L_avg / (1 - γ)
# at typical MinAtar episode length L_avg ≈ 500. The
# (1-γ^500)/(500*(1-γ)) factor at γ=0.99 is ≈ 0.199. So:
#   Asterix raw 6.8 → discounted ≈ 1.35
#   Breakout raw 6.2 → discounted ≈ 1.23
#   Freeway raw 12.9 → discounted ≈ 2.57
#   SpaceInvaders raw 3.7 → discounted ≈ 0.74
#
# These conversions are *approximate* — variable-per-step-reward
# envs don't have an exact raw→discounted formula without knowing
# when in the episode rewards land.

_TIER_2: tuple[SolveThreshold, ...] = (
    SolveThreshold(
        'Asterix-MinAtar', threshold=1.35,
        source='young-tian-2019-50pct-discounted-approx',
        confidence='derived',
    ),
    SolveThreshold(
        'Breakout-MinAtar', threshold=1.23,
        source='young-tian-2019-50pct-discounted-approx',
        confidence='derived',
    ),
    SolveThreshold(
        'Freeway-MinAtar', threshold=2.57,
        source='young-tian-2019-50pct-discounted-approx',
        confidence='derived',
    ),
    SolveThreshold(
        'SpaceInvaders-MinAtar', threshold=0.74,
        source='young-tian-2019-50pct-discounted-approx',
        confidence='derived',
    ),
)

# Tier 3 (absent). Misc envs without canonical literature
# thresholds. Listed explicitly so consumers know the env was
# *considered* and judged unthresholdable, vs being missing by
# accident. Excluded from converged-only analyses by default.

_TIER_3: tuple[SolveThreshold, ...] = (
    SolveThreshold(
        'BernoulliBandit-misc', threshold=None,
        source='no-canonical-criterion',
        confidence='absent',
    ),
    SolveThreshold(
        'GaussianBandit-misc', threshold=None,
        source='no-canonical-criterion',
        confidence='absent',
    ),
    SolveThreshold(
        'FourRooms-misc', threshold=None,
        source='no-canonical-criterion',
        confidence='absent',
    ),
    SolveThreshold(
        'MetaMaze-misc', threshold=None,
        source='no-canonical-criterion',
        confidence='absent',
    ),
    SolveThreshold(
        'Pong-misc', threshold=None,
        source='no-canonical-criterion',
        confidence='absent',
    ),
)


SOLVE_THRESHOLDS: Mapping[str, SolveThreshold] = {
    t.env_name: t for t in (*_TIER_1, *_TIER_2, *_TIER_3)
}
"""Frozen registry of per-env solve thresholds. Consumers should
read by `env_name`. 18 envs total: 9 literature + 4 derived + 5
absent."""


# ============ Predicate ============

def is_solved(
    env_name: str, outcome_value: float,
    *,
    table: Mapping[str, SolveThreshold] = SOLVE_THRESHOLDS,
) -> bool | None:
    """Did this cell solve the env, given the threshold table?

    Returns:
    - `True` if `outcome_value >= threshold`.
    - `False` if `outcome_value < threshold`.
    - `None` when the env's threshold is `'absent'` (caller decides
      what to do — exclude / treat as unknown).
    - `KeyError` when `env_name` isn't in the table — loud failure
      so consumers can't silently mis-classify."""
    if env_name not in table:
        raise KeyError(
            f"env_name {env_name!r} not in SOLVE_THRESHOLDS. "
            f"Registered envs: {sorted(table)!r}",
        )
    spec = table[env_name]
    if spec.threshold is None:
        return None
    return outcome_value >= spec.threshold


def envs_with_threshold(
    table: Mapping[str, SolveThreshold] = SOLVE_THRESHOLDS,
) -> tuple[str, ...]:
    """Env names where a defensible threshold exists (confidence
    is 'literature' or 'derived'). Excludes `'absent'` and
    `'sample_relative'` envs by default."""
    return tuple(
        sorted(
            name for name, t in table.items()
            if t.confidence in ('literature', 'derived')
        )
    )
