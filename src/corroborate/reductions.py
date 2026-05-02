"""Reductions — Measurable[R, T] factories that compose by value.

The post-hoc analytical primitive. Three layers in this module:

1. *Lifting record keys to measurables.* `from_key('q_max')`
   returns a `Measurable[Mapping[str, numpy array], numpy array]` that
   reads `record['q_max']`. The leaf primitive — every other
   reduction is built over `from_key`-derived measurables.

2. *Time-axis reductions.* `max_abs(of)`, `mean_window(of, lo, hi)`,
   `growth_window(of, early, late)` — each takes an existing
   `Measurable[R, numpy array]` and returns a new
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

import numpy as np

from corroborate.measurable import Measurable


# ============ Leaf: lift a record key to a Measurable ============

def from_key(
    key: str,
) -> Measurable[Mapping[str, np.ndarray], np.ndarray]:
    """Read `record[key]` as a `numpy array`. The leaf primitive
    that lifts a record-keyed value into typed `Measurable` space.

    Parameterized by `key` rather than declared as a `@measurable`
    function because the latter would close over a static name —
    factories take parameters."""
    def fn(record: Mapping[str, np.ndarray]) -> np.ndarray:
        return record[key]
    return Measurable(fn=fn, name=key, reads=(key,))


# ============ Time-axis reductions ============

def max_abs[R: Mapping[str, object]](
    of: Measurable[R, np.ndarray],
) -> Measurable[R, float]:
    """Max of `|·|` over the operand array. Returns scalar.

    Used by Banach-contraction invariants: `bounded(max_abs(
    from_key('max_q')), threshold=1e3)` asserts |Q| stays
    bounded — Q-divergence (deadly-triad signature) trips the
    INVARIANT_VIOLATION verdict."""
    name = f'{of.name}__max_abs'

    def fn(record: R) -> float:
        return float(np.max(np.abs(of(record))))
    return Measurable(fn=fn, name=name, reads=of.reads)


def mean_window[R: Mapping[str, object]](
    of: Measurable[R, np.ndarray],
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
        return float(np.mean(arr[i_lo:i_hi]))
    return Measurable(fn=fn, name=name, reads=of.reads)


def growth_window[R: Mapping[str, object]](
    of: Measurable[R, np.ndarray],
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
) -> Measurable[Mapping[str, np.ndarray], float]:
    """Schema-row outcome projection: mean over the last `fraction`
    of `record[key]`. Convenience wrapper around `mean_window`.

    NOTE: when the trajectory carries a cumulative-within-episode
    sawtooth (e.g. RL's ep_return signal that resets on episode
    terminations), a plain `late_window_mean` averages over the
    sawtooth — NOT the per-episode quantity. Use
    `masked_window_mean(value_key, mask_key, fraction)` to filter
    to mask-positive entries (e.g. terminal steps) before
    averaging. `late_window_mean` is correct for genuinely-
    per-step quantities (loss, td_error, max_q)."""
    if not (0.0 < fraction <= 1.0):
        raise ValueError(
            f'late_window_mean: need 0 < fraction ≤ 1; got {fraction}',
        )
    return mean_window(from_key(key), 1.0 - fraction, 1.0)


def masked_window_mean(
    value_key: str,
    mask_key: str,
    fraction: float = 0.1,
) -> Measurable[Mapping[str, np.ndarray], float]:
    """Mean of `record[value_key]` over entries where (`step in
    late `fraction` of trajectory` ∧ `record[mask_key] > 0.5`).

    Generic mechanic: take a window of the trajectory's last
    `fraction`, restrict to indices whose mask flag is set, mean
    the surviving values. Substrate-neutral.

    Use case: in RL, `record['ep_return']` is per-step cumulative
    return that *resets on done* (a sawtooth); the per-episode
    return appears on terminal steps where `record['done'] > 0.5`.
    `masked_window_mean('ep_return', 'done', 0.1)` averages the
    last 10% of episode-end returns. For non-RL substrates, mask
    is whatever binary indicator the experiment defines.

    Returns NaN if no element survives the mask in the window —
    `0.0` would collide with a legitimate `value_key` of zero
    (e.g. RL envs with reward range crossing zero). Downstream
    consumers must handle NaN explicitly."""
    if not (0.0 < fraction <= 1.0):
        raise ValueError(
            f'masked_window_mean: need 0 < fraction ≤ 1; '
            f'got {fraction}',
        )
    name = (
        f'{value_key}_masked_by_{mask_key}__late_window_mean_'
        f'{int(round((1.0 - fraction) * 100))}_100'
    )

    def fn(record: Mapping[str, np.ndarray]) -> float:
        values = record[value_key]
        mask = record[mask_key]
        n = int(values.shape[0])
        cutoff = int((1.0 - fraction) * n)
        time_mask = np.arange(n) >= cutoff
        keep_mask = time_mask & (mask > 0.5)
        n_kept = int(np.sum(keep_mask))
        if n_kept == 0:
            return float('nan')
        masked = np.where(keep_mask, values, 0.0)
        return float(np.sum(masked) / n_kept)

    return Measurable(fn=fn, name=name, reads=(value_key, mask_key))


# ============ Peak-aware windows ============
#
# `mean_window(of, lo, hi)` uses STATIC fractional bounds — same
# window for every cell. Useful when "the late half of training"
# is a meaningful slice. But when each cell has a per-cell peak
# point (e.g. `eval_best_burst_step`), a static window
# averages over different parts of the trajectory across cells.
# The peak-aware primitives below take `peak_idx_key`: a record
# key whose value is the per-cell peak in array-index units. The
# caller is responsible for ensuring the index is array-aligned
# (typically by precomputing `peak_step * n // total_steps` and
# inserting the result into the record before invocation).


def mean_peak_window[R: Mapping[str, object]](
    of: Measurable[R, np.ndarray],
    peak_idx_key: str,
    *,
    pre_frac: float = 0.5,
) -> Measurable[R, float]:
    """Mean of `of` over the window `[peak_idx * (1 - pre_frac),
    peak_idx]` — a *pre-peak* slice whose width scales with the
    cell's own peak position.

    `peak_idx_key` is a record key whose value is an integer-typed
    per-cell peak point in array-index units. `pre_frac` controls
    window width as a fraction of `peak_idx`. Default 0.5 means
    the window spans the late half of the cell's pre-peak
    trajectory — analogous to `mean_window(_, 0.5, 1.0)` but
    truncated at the peak.

    Returns NaN when `peak_idx <= 1` (insufficient pre-peak data)
    or when the window contains zero elements."""
    if not (0.0 < pre_frac <= 1.0):
        raise ValueError(
            f'mean_peak_window: need 0 < pre_frac ≤ 1; got {pre_frac}',
        )
    name = f'{of.name}__peak_pre{int(round(pre_frac * 100))}'
    reads = tuple(dict.fromkeys(of.reads + (peak_idx_key,)))

    def fn(record: R) -> float:
        peak_v = record.get(peak_idx_key)
        if not isinstance(peak_v, (int, float)):
            return float('nan')
        peak_idx = int(peak_v)
        if peak_idx <= 1:
            return float('nan')
        arr = of(record)
        n = int(len(arr))
        peak_idx = min(peak_idx, n)
        lo = max(0, int(peak_idx - pre_frac * peak_idx))
        if peak_idx <= lo:
            return float('nan')
        return float(np.mean(arr[lo:peak_idx]))
    return Measurable(fn=fn, name=name, reads=reads)


def peak_centered_window[R: Mapping[str, object]](
    of: Measurable[R, np.ndarray],
    peak_idx_key: str,
    *,
    half_width_frac: float = 0.125,
) -> Measurable[R, float]:
    """Mean of `of` over a window centered at the cell's peak —
    `[peak_idx - h, peak_idx + h]` where `h = half_width_frac * n`
    (so window width ≈ 2 * half_width_frac of the trajectory).
    Default `half_width_frac=0.125` → window of width ≈ 25%
    centered at the peak.

    Returns NaN if the resulting window has fewer than 2 valid
    elements (degenerate-window case)."""
    if not (0.0 < half_width_frac <= 0.5):
        raise ValueError(
            f'peak_centered_window: need 0 < half_width_frac ≤ 0.5; '
            f'got {half_width_frac}',
        )
    name = f'{of.name}__peak_centered{int(round(half_width_frac * 200))}'
    reads = tuple(dict.fromkeys(of.reads + (peak_idx_key,)))

    def fn(record: R) -> float:
        peak_v = record.get(peak_idx_key)
        if not isinstance(peak_v, (int, float)):
            return float('nan')
        peak_idx = int(peak_v)
        arr = of(record)
        n = int(len(arr))
        if n < 2 or peak_idx < 0 or peak_idx > n:
            return float('nan')
        h = max(1, int(half_width_frac * n))
        lo = max(0, peak_idx - h)
        hi = min(n, peak_idx + h)
        if hi - lo < 2:
            return float('nan')
        return float(np.mean(arr[lo:hi]))
    return Measurable(fn=fn, name=name, reads=reads)
