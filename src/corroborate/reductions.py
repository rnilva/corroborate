"""Reductions — Measurable[R, T] factories that compose by value.

The post-hoc analytical primitive. Three layers in this module:

1. *Lifting record keys to measurables.* `from_key('q_max')`
   returns a `Measurable[Mapping[str, jax.Array], jax.Array]` that
   reads `record['q_max']`. The leaf primitive — every other
   reduction is built over `from_key`-derived measurables.

2. *Time-axis reductions.* `max_abs(of)`, `mean_window(of, lo, hi)`,
   `growth_window(of, early, late)` — each takes an existing
   `Measurable[R, jax.Array]` and returns a new
   `Measurable[R, float]`. `reads` propagates: a reduction inherits
   the leaf-key set from its operand.

3. *Outcome projections.* The schema-row `primary_outcome_summary`
   is just a measurable applied to a per-step record. Step 4's
   `late_window_mean(record_key='ep_return', fraction=0.1)` is
   `mean_window(from_key('ep_return'), 0.9, 1.0)` — same primitive
   used at a different framing level.

Composition is by value. `max_abs(from_key('q_max'))` is a
`Measurable[R, float]` whose `name` is `'q_max__max_abs'` and
whose `reads` is `('q_max',)`. No name-keyed registry, no
`inspect.signature` injection — typed end-to-end."""
from __future__ import annotations

from collections.abc import Mapping

import jax.numpy as jnp

from corroborate.measurable import Measurable


# ============ Leaf: lift a record key to a Measurable ============

def from_key(
    key: str,
) -> Measurable[Mapping[str, jnp.ndarray], jnp.ndarray]:
    """Read `record[key]` as a `jax.Array`. The leaf primitive
    that lifts a record-keyed value into typed `Measurable` space.

    Parameterized by `key` rather than declared as a `@measurable`
    function because the latter would close over a static name —
    factories take parameters."""
    def fn(record: Mapping[str, jnp.ndarray]) -> jnp.ndarray:
        return record[key]
    return Measurable(fn=fn, name=key, reads=(key,))


# ============ Time-axis reductions ============

def max_abs[R: Mapping[str, object]](
    of: Measurable[R, jnp.ndarray],
) -> Measurable[R, float]:
    """Max of `|·|` over the operand array. Returns scalar.

    Used by Banach-contraction invariants: `bounded(max_abs(
    from_key('max_q')), threshold=1e3)` asserts |Q| stays
    bounded — Q-divergence (deadly-triad signature) trips the
    INVARIANT_VIOLATION verdict."""
    name = f'{of.name}__max_abs'

    def fn(record: R) -> float:
        return float(jnp.max(jnp.abs(of(record))))
    return Measurable(fn=fn, name=name, reads=of.reads)


def mean_window[R: Mapping[str, object]](
    of: Measurable[R, jnp.ndarray],
    lo: float,
    hi: float,
) -> Measurable[R, float]:
    """Mean of operand over fractional window `[lo, hi]` of its
    first axis. `mean_window(_, 0.9, 1.0)` is the late-10% mean
    — the canonical outcome projection.

    Bounds: `0.0 <= lo < hi <= 1.0`."""
    if not (0.0 <= lo < hi <= 1.0):
        raise ValueError(
            f'mean_window: need 0 ≤ lo < hi ≤ 1; got [{lo}, {hi}]',
        )
    label = f'mean_{int(round(lo * 100))}_{int(round(hi * 100))}'
    name = f'{of.name}__{label}'

    def fn(record: R) -> float:
        arr = of(record)
        n = int(arr.shape[0])
        i_lo = int(lo * n)
        i_hi = int(hi * n)
        # Guard the corner cases where n is tiny: ensure at
        # least one element falls in the window.
        if i_hi <= i_lo:
            i_hi = i_lo + 1
        return float(jnp.mean(arr[i_lo:i_hi]))
    return Measurable(fn=fn, name=name, reads=of.reads)


def unique_count[R: Mapping[str, object]](
    of: Measurable[R, jnp.ndarray],
) -> Measurable[R, int]:
    """Count of distinct values in the operand array (flattened).

    Used by coverage invariants — `at_least(unique_count(action),
    threshold=2)` is the necessary-condition test that ε-greedy
    explored more than one action over the trajectory."""
    name = f'{of.name}__unique_count'

    def fn(record: R) -> int:
        arr = of(record)
        return int(jnp.unique(arr.flatten()).shape[0])
    return Measurable(fn=fn, name=name, reads=of.reads)


def disagreement_rate[R: Mapping[str, object]](
    of_a: Measurable[R, jnp.ndarray],
    of_b: Measurable[R, jnp.ndarray],
) -> Measurable[R, float]:
    """Element-wise mean of `of_a != of_b`.

    Multi-input reduction. Used by the Hasselt-independence
    invariant: comparing `online_argmax` and `target_argmax` per
    transition tells us how often the two estimators pick
    different actions on the same state. DDQN's bias-correction
    leans on this disagreement being non-trivial; rate ≈ 0 means
    online ≡ target (so DDQN reduces to vanilla)."""
    name = f'disagreement[{of_a.name},{of_b.name}]'
    reads = tuple(dict.fromkeys(of_a.reads + of_b.reads))

    def fn(record: R) -> float:
        a = of_a(record)
        b = of_b(record)
        return float(jnp.mean(a != b))
    return Measurable(fn=fn, name=name, reads=reads)


def growth_window[R: Mapping[str, object]](
    of: Measurable[R, jnp.ndarray],
    *,
    early: tuple[float, float] = (0.0, 0.25),
    late: tuple[float, float] = (0.75, 1.0),
) -> Measurable[R, float]:
    """Ratio of late-window mean over early-window mean. Late /
    max(|early|, 1e-9).

    Geometric-decay invariants want this < 1 (e.g. Bellman
    residual decays under contraction). Overestimation-bound
    invariants want this < some threshold > 1 (e.g. max_q's drift
    under vanilla DQN's Jensen bias is bounded). The threshold
    is the caller's call, parameterised on `bounded`."""
    early_m = mean_window(of, *early)
    late_m = mean_window(of, *late)
    name = (f'{of.name}__growth_'
            f'{int(round(early[0] * 100))}_{int(round(late[1] * 100))}')

    def fn(record: R) -> float:
        e = early_m(record)
        l = late_m(record)
        return l / max(abs(e), 1e-9)
    return Measurable(fn=fn, name=name, reads=of.reads)


# ============ Outcome projections (schema-row helpers) ============

def late_window_mean(
    key: str, fraction: float = 0.1,
) -> Measurable[Mapping[str, jnp.ndarray], float]:
    """Schema-row outcome projection: mean over the last `fraction`
    of `record[key]`. Convenience wrapper around `mean_window`.

    NOTE: for episode-return outcomes specifically, prefer
    `late_episode_return_mean` — `ep_return` is per-step
    cumulative-within-episode (a sawtooth under standard logging)
    and a plain mean over a sawtooth is NOT the per-episode
    return. `late_window_mean` is correct for genuinely-per-step
    quantities (loss, td_error, max_q)."""
    if not (0.0 < fraction <= 1.0):
        raise ValueError(
            f'late_window_mean: need 0 < fraction ≤ 1; got {fraction}',
        )
    return mean_window(from_key(key), 1.0 - fraction, 1.0)


def late_episode_return_mean(
    return_key: str = 'ep_return',
    done_key: str = 'done',
    fraction: float = 0.1,
) -> Measurable[Mapping[str, jnp.ndarray], float]:
    """Mean of episode returns terminating in the late
    `fraction` of the trajectory. Avoids the sawtooth pitfall:
    plain `late_window_mean('ep_return', fraction)` averages a
    cumulative-within-episode signal that mixes mid-episode
    partial sums with end-of-episode totals — NOT the per-episode
    return.

    `record[return_key]` is the cumulative-within-episode value
    at each step (so on `done==1` it's the final episode return).
    `record[done_key]` is the binary done flag. This reduction
    masks to (`step in [1-fraction, 1.0)` ∧ `done==1`) then means
    the surviving values — i.e. mean of episode-end returns
    whose terminal step lies in the late window.

    Returns 0.0 if no episode ended in the late window."""
    if not (0.0 < fraction <= 1.0):
        raise ValueError(
            f'late_episode_return_mean: need 0 < fraction ≤ 1; '
            f'got {fraction}',
        )
    name = (
        f'{return_key}_at_done__late_window_mean_'
        f'{int(round((1.0 - fraction) * 100))}_100'
    )

    def fn(record: Mapping[str, jnp.ndarray]) -> float:
        ep_return = record[return_key]
        done = record[done_key]
        n = int(ep_return.shape[0])
        cutoff = int((1.0 - fraction) * n)
        time_mask = jnp.arange(n) >= cutoff
        ep_end_mask = done > 0.5
        keep_mask = time_mask & ep_end_mask
        n_kept = int(jnp.sum(keep_mask))
        if n_kept == 0:
            return 0.0
        masked = jnp.where(keep_mask, ep_return, 0.0)
        return float(jnp.sum(masked) / n_kept)

    return Measurable(fn=fn, name=name, reads=(return_key, done_key))
