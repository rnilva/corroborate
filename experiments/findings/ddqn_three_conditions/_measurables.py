"""Hypothesis-local derived measurables.

Three substrate-derived columns the condition bridges scope on.
Each reads existing per-cell scalar fields and returns a
categorical / integer leaf — the framework's persistence layer
admits `str | int | float | bool` as `MeasurementLeaf`.

Lives in the hypothesis module (not the substrate's canonical
`corroborate_rl.dqn.measurables`) because the stratification
categories are hypothesis-specific: `shaping_kind` is meaningful
only when potential-based shaping is a study axis, `fa_kind` is
the linear-vs-deep distinction this hypothesis tests, `k_eff` is
the action-multiplier × native-action product.

Registered via `@measurable` so the framework's `--ingest`
pipeline computes them automatically when ingesting raw corpora
into the hypothesis cache. No manual corpus-joiner needed."""
from __future__ import annotations

from collections.abc import Mapping

from corroborate.measurables import measurable


# Native discrete action counts per env. Used by `k_eff`.
_NATIVE_ACTIONS: dict[str, int] = {
    'FourRooms-misc': 4,
    'Acrobot-v1': 3,
    'MountainCar-v0': 3,
    'MetaMaze-misc': 4,
    'CartPole-v1': 2,
    'Asterix-MinAtar': 5,
    'Breakout-MinAtar': 3,
    'Freeway-MinAtar': 3,
    'SpaceInvaders-MinAtar': 4,
    'PacMan-jumanji': 5,
    'SlidingTilePuzzle-jumanji': 4,
    'Snake-jumanji': 4,
}


@measurable(reads=('wrappers',))
def shaping_kind(record: Mapping[str, object]) -> str:
    """Categorical: which kind of reward shaping is active.

    Returns `'potential_manhattan'` when the `wrappers` field
    contains a `PotentialReward` dataclass; `'none'` otherwise.
    Used as a scope axis for Condition 3 (policy-signal-strength
    decouples bias from outcome translation under shaping)."""
    w = record.get('wrappers')
    if w is None:
        return 'none'
    s = str(w)
    if 'PotentialReward' in s:
        return 'potential_manhattan'
    return 'none'


@measurable(reads=('q_network.hidden',))
def fa_kind(record: Mapping[str, object]) -> str:
    """Categorical: function-approximator capacity class.

    Returns `'linear'` when the `q_network.hidden` field encodes
    an empty hidden tuple `()` (linear FA); `'mlp_deep'`
    otherwise (any non-empty MLP). Used as a scope axis for
    Condition 2 (FA-capacity gates Type 1 manifestation)."""
    h = record.get('q_network.hidden')
    if h is None:
        return 'mlp_deep'
    if str(h) == '()':
        return 'linear'
    return 'mlp_deep'


@measurable(reads=('env_name', 'action_duplicate_k'))
def k_eff(record: Mapping[str, object]) -> int:
    """Integer: effective discrete-action count after
    action_duplicate wrapping.

    Computed as `native_actions(env_name) × action_duplicate_k`.
    Returns the native count when `action_duplicate_k` is None
    (no wrapper) or missing from the catalogue. Used as a scope
    axis for Condition 1 (Q-bias scales with K)."""
    env = record.get('env_name')
    if not isinstance(env, str):
        return 0
    native = _NATIVE_ACTIONS.get(env, 0)
    k_raw = record.get('action_duplicate_k')
    if k_raw is None:
        return native
    try:
        k = int(float(k_raw))  # action_duplicate_k arrives as float
    except (TypeError, ValueError):
        return native
    return native * k
