"""§5 candidate-mediator features for DDQN — typed `Measurable`s.

Eight scalars projected from the per-cell DQN trajectory. These are
the candidate-mediator set referenced in PAPER §5.1: within-env
Pearson against `outcome.eval_final_mean` plus the arm-conditional
diagnostic produces the three-regime mediator taxonomy (TD-
convergence / action-margin / stay-greedy).

Six are typed `Measurable[Mapping[str, np.ndarray], float]` —
declared via `@measurable(reads=...)` so they (a) appear in the
measurable registry, (b) carry their leaf-key dependencies as
`reads` (consumed by future `Bridge.transitive_reads` for the
redundancy primitive), (c) are introspectable by `walk_paths`-
shape consumers.

Two are plain functions that don't fit the `Measurable[Mapping,
T]` mold:

- `fill_ratio_late(record, *, capacity)` — typed Measurable but
  needs an extra `capacity` kwarg the framework's auto-resolver
  doesn't fill (capacity isn't a record key OR a measurable). Use
  by direct call: `fill_ratio_late(record, capacity=10_000)`. The
  @measurable wrapper still registers + tracks `reads`.
- `epsilon_late(*, eps_init, eps_final, anneal_steps, total_steps)`
  — closed-form from HP leaves, doesn't take a record at all.
  Plain function; not a Measurable because the type bound
  `Mapping[str, ...]` doesn't apply.

These are RL-substrate-specific (they read DQN-trace keys), so
they live under `corroborate.rl.dqn`. Generic post-hoc reductions
(`mean_window`, `growth_window`) live in `corroborate.reductions`;
the mediators here are *named compositions* using DQN-specific
keys."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import numpy.typing as npt

from corroborate.measurable import Measurable, measurable


def _mean_window(
    arr: npt.NDArray[np.float64], lo: float, hi: float,
) -> float:
    """Mean over the fractional window [lo, hi] of `arr`'s leading
    axis. Mirrors `corroborate.reductions.mean_window` but inlined
    here to keep the per-cell numpy hot path direct."""
    n = arr.shape[0]
    if n == 0:
        return float('nan')
    i_lo = int(lo * n)
    i_hi = max(int(hi * n), i_lo + 1)
    return float(np.mean(arr[i_lo:i_hi]))


def _record_array(
    record: Mapping[str, object], key: str,
) -> npt.NDArray[np.float64] | None:
    """Pull `record[key]` as a float64 numpy array. Returns None
    if the key is absent (so callers can NaN-propagate)."""
    if key not in record:
        return None
    return np.asarray(record[key], dtype=np.float64)


# ============ Six record→float Measurables ============

@measurable(reads=('online_max_q_per_step', 'online_min_q_per_step'))
def q_gap_late(record: Mapping[str, object]) -> float:
    """Mean of (online_max_q − online_min_q) over the last 50% of
    training. §5.1's action-margin mediator — sharp policy
    preferences correlate with reward in Breakout / MNISTBandit /
    SpaceInvaders."""
    max_q = _record_array(record, 'online_max_q_per_step')
    min_q = _record_array(record, 'online_min_q_per_step')
    if max_q is None or min_q is None:
        return float('nan')
    return _mean_window(max_q - min_q, 0.5, 1.0)


@measurable(reads=('online_max_q_per_step', 'online_min_q_per_step'))
def q_gap_growth(record: Mapping[str, object]) -> float:
    """(late_half_mean − early_half_mean) of the q_gap signal.
    Captures whether action-margin widens as training progresses
    (positive growth → emergent policy sharpness)."""
    max_q = _record_array(record, 'online_max_q_per_step')
    min_q = _record_array(record, 'online_min_q_per_step')
    if max_q is None or min_q is None:
        return float('nan')
    gap = max_q - min_q
    early = _mean_window(gap, 0.0, 0.5)
    late = _mean_window(gap, 0.5, 1.0)
    return float(late - early)


@measurable(reads=('online_max_q_per_step',))
def q_max_growth(record: Mapping[str, object]) -> float:
    """late_quarter / max(|early_quarter|, 1e-9) of online_max_q.
    §5.1's value-curve growth — vanilla DQN's Jensen bias
    typically pushes this above 1; DDQN attenuates."""
    arr = _record_array(record, 'online_max_q_per_step')
    if arr is None:
        return float('nan')
    early = _mean_window(arr, 0.0, 0.25)
    late = _mean_window(arr, 0.75, 1.0)
    return float(late / max(abs(early), 1e-9))


@measurable(reads=('online_mean_q_per_step', 'online_max_q_per_step'))
def v_vs_max_delta_late(record: Mapping[str, object]) -> float:
    """Mean of |q_mean − q_max| over the late 50%. DDQN's mechanism
    signature: vanilla's overestimation widens the action-Q
    distribution (large delta); DDQN's decoupled selection narrows
    it. The Hasselt 2010 Jensen-bias proxy at the per-step level."""
    mean_q = _record_array(record, 'online_mean_q_per_step')
    max_q = _record_array(record, 'online_max_q_per_step')
    if mean_q is None or max_q is None:
        return float('nan')
    return _mean_window(np.abs(mean_q - max_q), 0.5, 1.0)


@measurable(reads=('td_error',))
def td_residual_late(record: Mapping[str, object]) -> float:
    """Mean of |TD residual| over the late 50%. §5.1's TD-
    convergence mediator — Acrobot's r=+0.84 vs GaussianBandit's
    r=−0.81 (sign-flip across regimes) is the canonical motivation
    for per-env PC in §6."""
    arr = _record_array(record, 'td_error')
    if arr is None:
        return float('nan')
    return _mean_window(arr, 0.5, 1.0)


@measurable(reads=('online_argmax_per_step', 'target_argmax_per_step'))
def greedy_match_late(record: Mapping[str, object]) -> float:
    """Mean of (online_argmax == target_argmax) over the late 50%.
    §5.1's stay-greedy mediator. DDQN's slot swap explicitly
    decouples these argmaxes; large match ⇒ DDQN's mechanism is
    *inactive* on this cell (the two estimators agree, so vanilla
    and DDQN reduce to each other)."""
    online = _record_array(record, 'online_argmax_per_step')
    target = _record_array(record, 'target_argmax_per_step')
    if online is None or target is None:
        return float('nan')
    match = (online == target).astype(np.float64)
    return _mean_window(match, 0.5, 1.0)


# ============ Two with extra config (not auto-resolvable) ============

@measurable(reads=('buf_size',))
def fill_ratio_late(
    record: Mapping[str, object], *, capacity: int,
) -> float:
    """Mean of buf_size / capacity over the late 50%. §5.1's
    coverage mediator — when buf < capacity throughout training
    the agent is still in the early-replay regime; when full,
    bootstrapping operates on a stationary buffer.

    NOTE: `capacity` is an extra kwarg the framework's auto-resolver
    does NOT fill — it's neither a record key nor a registered
    measurable. Callers must pass it directly:
    `fill_ratio_late(record, capacity=10_000)`. The Measurable
    wrapper still tracks `reads=('buf_size',)` for downstream
    redundancy / reads-set fingerprinting."""
    if capacity <= 0:
        return float('nan')
    arr = _record_array(record, 'buf_size')
    if arr is None:
        return float('nan')
    return _mean_window(arr / float(capacity), 0.5, 1.0)


def epsilon_late(
    *, eps_init: float, eps_final: float,
    anneal_steps: int, total_steps: int,
) -> float:
    """Mean of the linear-ε schedule value over the late 50% of
    training. Closed-form from leaves — no record needed since the
    schedule is deterministic in `step`. Linear schedule: ε(step) =
    eps_init + (eps_final − eps_init) · min(step/anneal_steps, 1).

    NOT a Measurable: the type bound `Mapping[str, ...]` doesn't
    apply (no record arg). Plain function; consumers call it with
    schedule HPs from the configurational leaves of a RunRow."""
    if anneal_steps <= 0 or total_steps <= 0:
        return float('nan')
    lo = total_steps // 2
    if lo >= total_steps:
        return float('nan')
    steps = np.arange(lo, total_steps, dtype=np.float64)
    progress = np.minimum(steps / float(anneal_steps), 1.0)
    eps = eps_init + (eps_final - eps_init) * progress
    return float(np.mean(eps))


# Convenience export — the six record-only Measurables for
# consumers that want to iterate. fill_ratio_late and epsilon_late
# omitted because they need extra kwargs.
RECORD_ONLY_MEDIATORS: tuple[Measurable[Mapping[str, object], float], ...] = (
    q_gap_late, q_gap_growth, q_max_growth,
    v_vs_max_delta_late, td_residual_late, greedy_match_late,
)
