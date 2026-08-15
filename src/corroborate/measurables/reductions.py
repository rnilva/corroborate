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
   `late_window_mean(record_key='<key>', fraction=0.1)` is
   `mean_window(from_key('<key>'), 0.9, 1.0)` — same primitive
   used at a different framing level.

Composition is by value. `max_abs(from_key('<key>'))` is a
`Measurable[R, float]` whose `name` is `'<key>__max_abs'` and
whose `reads` is `('<key>',)`. No name-keyed registry, no
`inspect.signature` injection — typed end-to-end."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Protocol, cast

import numpy as np
import numpy.typing as npt

from corroborate.measurables.measurable import Measurable


type _AxisOp = Literal['mean', 'var', 'std', 'max', 'min', 'sum']


# ============ Reduction Protocol — typed factory contract ============
#
# Every factory that lifts `Measurable[R, T_in]` to
# `Measurable[R, T_out]` via parametric arguments satisfies this
# Protocol structurally. `max_abs`, `mean_window`, `growth_window`,
# `mean_peak_window`, `peak_centered_window`, `reduce_axis`,
# `slice_axis`, `log_safe`, `cv_safe` — all conform without
# inheritance (Python's structural Protocol matching).
#
# The Protocol formalises the implementation-author contract: new
# reductions written outside this module (e.g. for a non-RL
# substrate's domain-specific shape) get type-checked against this
# shape. Deviation from `(of: Measurable[R, T_in], *params) ->
# Measurable[R, T_out]` fails pyright at the factory's call site.
#
# `from_key`, `late_window_mean`, `masked_window_mean` are NOT
# Reductions — they take primitive args (`key: str`, `fraction:
# float`) rather than another Measurable. They are leaf-Measurable
# factories; reductions lift existing measurables.

class Reduction[
    R: Mapping[str, object], T_in, T_out, **P,
](Protocol):
    """Typed factory contract: lift `Measurable[R, T_in]` to
    `Measurable[R, T_out]` via parametric arguments.

    Substrate-authoring contract for "what is a reduction". Every
    factory in this module that takes a `Measurable` as its first
    positional argument and returns a `Measurable` conforms
    structurally — no explicit subclassing needed.

    Example:
        my_reduce: Reduction[
            Mapping[str, object],
            npt.NDArray[np.floating],
            npt.NDArray[np.floating],
            [int],
        ] = reduce_axis  # type-checks via structural conformance
    """
    def __call__(
        self,
        of: Measurable[R, T_in],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Measurable[R, T_out]: ...


# ============ Leaf: lift a record key to a Measurable ============

def from_key(
    key: str,
) -> Measurable[Mapping[str, object], npt.NDArray[np.floating]]:
    """Read `record[key]` as a `numpy.ndarray`. The leaf primitive
    that lifts a record-keyed value into typed `Measurable` space.

    Coerces via `np.asarray` so polars-shape inputs (e.g. a 2-D
    `mc_return` column surfacing as `list[list[float]]` in a row
    dict) become proper ndarrays before downstream factories
    apply their `.shape` / `.mean(axis=...)` / etc. calls. Without
    this coercion every caller would have to repeat the
    `np.asarray(record['key'], dtype=np.float64)` boilerplate.

    Raises `KeyError` if `key` is absent from the record — the
    cache builder catches this upstream and stores None.

    Parameterized by `key` rather than declared as a `@measurable`
    function because the latter would close over a static name —
    factories take parameters."""
    def fn(record: Mapping[str, object]) -> npt.NDArray[np.floating]:
        v = record[key]
        return np.asarray(v)
    return Measurable(fn=fn, name=key, reads=(key,))


# ============ Time-axis reductions ============

def max_abs[R: Mapping[str, object]](
    of: Measurable[R, npt.NDArray[np.floating]],
) -> Measurable[R, float]:
    """Max of `|·|` over the operand array. Returns scalar.

    Use: bound a record-key's L∞ norm via `bounded(max_abs(
    from_key('<key>')), threshold=...)`. Magnitude excursion
    above the threshold trips the INVARIANT_VIOLATION verdict."""
    name = f'{of.name}__max_abs'

    def fn(record: R) -> float:
        # max |x| = max(max(x), -min(x)). The reduction-then-abs path
        # preserves dtype through numpy's stubs; `np.abs(arr)` itself
        # returns dtype[Any] for arbitrary inputs.
        arr = of(record)
        return max(float(abs(np.max(arr))), float(abs(np.min(arr))))
    return Measurable(
        fn=fn, name=name, reads=of.reads,
        compose_of=(cast(
            'Measurable[Mapping[str, object], object]', of,
        ),),
    )


def mean_window[R: Mapping[str, object]](
    of: Measurable[R, npt.NDArray[np.floating]],
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
        # 0-d / scalar inputs (e.g. a null trace cell that
        # `from_key`'s `np.asarray` decoded as a 0-d ndarray) have
        # no window to take a mean over; NaN-propagate. Subsumes
        # the substrate's prior `_mean_window` helper, which
        # guarded the same case before being deleted.
        if arr.ndim == 0:
            return float('nan')
        n = len(arr)
        if n == 0:
            return float('nan')
        i_lo = int(lo * n)
        i_hi = int(hi * n)
        # Guard the corner cases where n is tiny: ensure at
        # least one element falls in the window.
        if i_hi <= i_lo:
            i_hi = i_lo + 1
        return float(np.mean(arr[i_lo:i_hi]))
    return Measurable(
        fn=fn, name=name, reads=of.reads,
        compose_of=(cast(
            'Measurable[Mapping[str, object], object]', of,
        ),),
    )


def growth_window[R: Mapping[str, object]](
    of: Measurable[R, npt.NDArray[np.floating]],
    *,
    early: tuple[float, float] = (0.0, 0.25),
    late: tuple[float, float] = (0.75, 1.0),
) -> Measurable[R, float]:
    """Ratio of late-window mean over early-window mean. Late /
    max(|early|, 1e-9).

    Geometric-decay invariants want this < 1 (the operand
    decays late vs early). Bounded-drift invariants want this
    < some threshold > 1. The threshold is the caller's call,
    parameterised on `bounded`."""
    early_m = mean_window(of, *early)
    late_m = mean_window(of, *late)
    name = (f'{of.name}__growth_'
            f'{int(round(early[0] * 100))}_{int(round(late[1] * 100))}')

    def fn(record: R) -> float:
        e = early_m(record)
        l = late_m(record)
        return l / max(abs(e), 1e-9)
    return Measurable(
        fn=fn, name=name, reads=of.reads,
        compose_of=(
            cast(
                'Measurable[Mapping[str, object], object]', early_m,
            ),
            cast(
                'Measurable[Mapping[str, object], object]', late_m,
            ),
        ),
    )


# ============ Outcome projections (schema-row helpers) ============

def late_window_mean(
    key: str, fraction: float = 0.1,
) -> Measurable[Mapping[str, object], float]:
    """Schema-row outcome projection: mean over the last `fraction`
    of `record[key]`. Convenience wrapper around `mean_window`.

    NOTE: when the trajectory carries a cumulative-within-segment
    sawtooth (a per-step accumulator that resets on segment
    boundaries), a plain `late_window_mean` averages over the
    sawtooth — NOT the per-segment quantity. Use
    `masked_window_mean(value_key, mask_key, fraction)` to filter
    to mask-positive entries (e.g. segment-boundary steps) before
    averaging. `late_window_mean` is correct for genuinely
    per-step quantities."""
    if not (0.0 < fraction <= 1.0):
        raise ValueError(
            f'late_window_mean: need 0 < fraction ≤ 1; got {fraction}',
        )
    return mean_window(from_key(key), 1.0 - fraction, 1.0)


def masked_window_mean(
    value_key: str,
    mask_key: str,
    fraction: float = 0.1,
) -> Measurable[Mapping[str, npt.NDArray[np.floating]], float]:
    """Mean of `record[value_key]` over entries where (`step in
    late `fraction` of trajectory` ∧ `record[mask_key] > 0.5`).

    Generic mechanic: take a window of the trajectory's last
    `fraction`, restrict to indices whose mask flag is set, mean
    the surviving values. Substrate-neutral.

    Use case: when a per-step record key is a cumulative
    accumulator that resets on segment boundaries, the per-
    segment-end value appears at boundary indices marked by a
    binary `mask_key`. `masked_window_mean(value_key, mask_key,
    0.1)` averages the last 10% of boundary-marked values. The
    binary indicator is whatever the implementation defines.

    Returns NaN if no element survives the mask in the window —
    `0.0` would collide with a legitimate `value_key` of zero.
    Downstream consumers must handle NaN explicitly."""
    if not (0.0 < fraction <= 1.0):
        raise ValueError(
            f'masked_window_mean: need 0 < fraction ≤ 1; '
            f'got {fraction}',
        )
    name = (
        f'{value_key}_masked_by_{mask_key}__late_window_mean_'
        f'{int(round((1.0 - fraction) * 100))}_100'
    )

    def fn(record: Mapping[str, npt.NDArray[np.floating]]) -> float:
        values = record[value_key]
        mask = record[mask_key]
        n = len(values)
        cutoff = int((1.0 - fraction) * n)
        time_mask = np.arange(n) >= cutoff
        keep_mask = time_mask & (mask > 0.5)
        n_kept = int(np.sum(keep_mask))
        if n_kept == 0:
            return float('nan')
        # `np.sum(values, where=keep_mask)` selects which positions
        # contribute — typed `floating[Any]` and (unlike
        # `values * keep_mask`) safe against NaN at False-mask
        # positions, which would otherwise propagate via NaN×0=NaN.
        return float(np.sum(values, where=keep_mask) / n_kept)

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
    of: Measurable[R, npt.NDArray[np.floating]],
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
    return Measurable(
        fn=fn, name=name, reads=reads,
        compose_of=(cast(
            'Measurable[Mapping[str, object], object]', of,
        ),),
    )


def peak_centered_window[R: Mapping[str, object]](
    of: Measurable[R, npt.NDArray[np.floating]],
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
    return Measurable(
        fn=fn, name=name, reads=reads,
        compose_of=(cast(
            'Measurable[Mapping[str, object], object]', of,
        ),),
    )


# ============ Axis-aware reductions for N-D operands ============
#
# `mean_window` / `growth_window` / `peak_*` assume a 1-D operand
# and reduce along its leading axis. When an operand is 2-D
# (e.g. `mc_return` shape `(n_outer, n_inner)` — eval-checkpoint
# index × replicates per checkpoint), authors typically want to
# (a) collapse the inner axis to per-outer scalars, then (b)
# window or otherwise reduce along the leading axis.
#
# `reduce_axis` and `slice_axis` are the substrate-blind primitives
# for that shape: each collapses or slices a single axis. The
# composition `mean_window(reduce_axis(from_key('X'), op='mean'),
# 0.0, 0.25)` reads as "first-quarter of per-outer means on X" —
# a pattern previously written as a hand-rolled `@measurable`.
#
# Auto-generated names embed the axis + op for cache predictability:
#   `X__mean_axis_-1`, `X__var_axis_-1`,
#   `X__slice_axis_0_25_75`.


def reduce_axis[R: Mapping[str, object]](
    of: Measurable[R, npt.NDArray[np.floating]],
    *,
    axis: int = -1,
    op: _AxisOp = 'mean',
) -> Measurable[R, npt.NDArray[np.floating]]:
    """Collapse `axis` of `of`'s output via `op`. Returns an
    array one dimension lower than the input. `axis=-1` (default)
    is the inner-axis collapse — the typical "reduce-replicates-
    keep-checkpoints" shape (e.g. `mc_return` shape `(n_bursts,
    n_episodes)` → per-burst scalars shape `(n_bursts,)`).

    `op` supports `'mean'` / `'var'` / `'std'` / `'max'` /
    `'min'` / `'sum'`. Other ops belong in a hand-written
    `@measurable` (the framework primitives stay narrow).

    Empty operand passes through unchanged so downstream factories
    can NaN-propagate. The factory itself doesn't catch missing
    leaf reads — the cache builder catches `KeyError` from
    `from_key` upstream and stores None on cells where the leaf
    isn't present."""
    name = f'{of.name}__{op}_axis_{axis}'

    def fn(record: R) -> npt.NDArray[np.floating]:
        arr = of(record)
        if arr.size == 0:
            return arr
        # `ndarray.mean(axis=...)` and friends are typed `Any` in
        # numpy >= 2.x stubs type axis-reductions precisely; no
        # boundary cast needed (size==0 already short-circuited above).
        if op == 'mean':
            return arr.mean(axis=axis)
        if op == 'var':
            return arr.var(axis=axis)
        if op == 'std':
            return arr.std(axis=axis)
        if op == 'max':
            return arr.max(axis=axis)
        if op == 'min':
            return arr.min(axis=axis)
        return arr.sum(axis=axis)
    return Measurable(
        fn=fn, name=name, reads=of.reads,
        compose_of=(cast(
            'Measurable[Mapping[str, object], object]', of,
        ),),
    )


type _ArgOp = Literal['argmax', 'argmin']


def select_at[R: Mapping[str, object]](
    values: Measurable[R, npt.NDArray[np.floating]],
    indicator: Measurable[R, npt.NDArray[np.floating]],
    *,
    op: _ArgOp = 'argmax',
    axis: int = -1,
) -> Measurable[R, float]:
    """Reduce `values` to scalar by indexing at the `argmax`/`argmin`
    of `indicator` along `axis`. Both measurables must produce
    arrays whose `axis`-dimension matches.

    The "select-at-argmax" pattern surfaces whenever an implementation
    needs "the value of X at the time-step / burst where Y peaked"
    — e.g., "the mechanism state at the burst where outcome was
    best", "the std of Q at the burst where bias was largest".
    Generic composition of two per-axis arrays at a typed call
    site — no domain-specific code needed.

    Returns NaN when either operand is empty, axis-mismatched, or
    the indicator is all-NaN (no valid argmax). NaN-propagation
    matches the rest of the reduction module.

    Note: not a `Reduction` (unary Protocol) — `select_at` is a
    binary combinator. The two arms are tied through the shared
    record `R` (read once, project twice), so `reads` is the
    UNION of both operands' read sets.
    """
    name = f'{values.name}__{op}_{indicator.name}_axis_{axis}'

    def fn(record: R) -> float:
        vals = values(record)
        ind = indicator(record)
        if vals.size == 0 or ind.size == 0:
            return float('nan')
        if vals.shape[axis] != ind.shape[axis]:
            return float('nan')
        # `nanargmax/min` returns 0 for all-NaN, but with a runtime
        # warning. Guard against all-NaN explicitly so the caller
        # gets NaN-propagated, not silent index-0 fallback.
        finite_mask = np.isfinite(ind)
        if not finite_mask.any():
            return float('nan')
        try:
            if op == 'argmax':
                idx = int(np.nanargmax(ind))
            else:
                idx = int(np.nanargmin(ind))
        except ValueError:  # all-NaN slice
            return float('nan')
        # `np.take` honours `axis`; for axis=-1 on 1-D, equivalent
        # to indexing.
        val = np.take(vals, idx, axis=axis)
        # If `values` was N-D and we picked along axis, val may be
        # an array (e.g., (n_bursts, n_episodes) values + per-burst
        # indicator → val shape (n_episodes,)). For scalar return,
        # require the result to be 0-D; otherwise NaN.
        v = np.asarray(val)
        if v.ndim != 0:
            return float('nan')
        return float(v)

    return Measurable(
        fn=fn, name=name,
        reads=tuple(sorted(set(values.reads) | set(indicator.reads))),
        compose_of=(
            cast('Measurable[Mapping[str, object], object]', values),
            cast('Measurable[Mapping[str, object], object]', indicator),
        ),
    )


def slice_axis[R: Mapping[str, object]](
    of: Measurable[R, npt.NDArray[np.floating]],
    *,
    axis: int = 0,
    lo: float = 0.0,
    hi: float = 1.0,
) -> Measurable[R, npt.NDArray[np.floating]]:
    """Take the fractional `[lo, hi]` window along `axis` of
    `of`'s output, returning the slice (same N-D as input).
    Sibling of `mean_window` that keeps the slice instead of
    averaging it — useful when downstream wants a non-mean
    reduction over the windowed slice.

    Bounds: `0.0 <= lo < hi <= 1.0`."""
    if not (0.0 <= lo < hi <= 1.0):
        raise ValueError(
            f'slice_axis: need 0 ≤ lo < hi ≤ 1; got [{lo}, {hi}]',
        )
    label = f'slice_axis_{axis}_{int(round(lo * 100))}_{int(round(hi * 100))}'
    name = f'{of.name}__{label}'

    def fn(record: R) -> npt.NDArray[np.floating]:
        arr = of(record)
        if arr.size == 0:
            return arr
        n = np.size(arr, axis)
        i_lo = int(lo * n)
        i_hi = max(int(hi * n), i_lo + 1)
        idx: list[slice | int] = [slice(None)] * arr.ndim
        idx[axis] = slice(i_lo, i_hi)
        return arr[tuple(idx)]
    return Measurable(
        fn=fn, name=name, reads=of.reads,
        compose_of=(cast(
            'Measurable[Mapping[str, object], object]', of,
        ),),
    )


# ============ Element-wise unary lifts ============


def log_safe[R: Mapping[str, object]](
    of: Measurable[R, npt.NDArray[np.floating]],
) -> Measurable[R, npt.NDArray[np.floating]]:
    """Element-wise `log(x)` with NaN on non-positive values.
    Same NaN-honest discipline as the hand-written `log_mc_variance_
    per_burst`: zero or negative entries (e.g. deterministic-
    converged bursts with mc_return variance = 0) become NaN, so
    averaging across cells doesn't leak `-∞` outliers via a 1e-9
    epsilon shortcut."""
    name = f'{of.name}__log'

    def fn(record: R) -> npt.NDArray[np.floating]:
        arr = np.asarray(of(record), dtype=np.float64)
        out = np.full(arr.shape, np.nan, dtype=np.float64)
        mask = arr > 0
        out[mask] = np.log(arr[mask])
        return out
    return Measurable(
        fn=fn, name=name, reads=of.reads,
        compose_of=(cast(
            'Measurable[Mapping[str, object], object]', of,
        ),),
    )


def cv_safe[R: Mapping[str, object]](
    of: Measurable[R, npt.NDArray[np.floating]],
    *,
    axis: int = -1,
) -> Measurable[R, npt.NDArray[np.floating]]:
    """Coefficient of variation: `std(arr, axis) / |mean(arr,
    axis)|`. Reduces `axis`; output has one less dimension. NaN
    where mean is zero (degenerate; CV undefined). Composes with
    the rest of the toolkit — feeding `cv_safe` into
    `mean_window` gives e.g. mean-CV-late."""
    mean_m = reduce_axis(of, axis=axis, op='mean')
    std_m = reduce_axis(of, axis=axis, op='std')
    name = f'{of.name}__cv_axis_{axis}'

    def fn(record: R) -> npt.NDArray[np.floating]:
        m = np.asarray(mean_m(record), dtype=np.float64)
        s = np.asarray(std_m(record), dtype=np.float64)
        out = np.full(m.shape, np.nan, dtype=np.float64)
        mask = np.abs(m) > 0
        out[mask] = s[mask] / np.abs(m[mask])
        return out
    return Measurable(
        fn=fn, name=name, reads=of.reads,
        compose_of=(
            cast(
                'Measurable[Mapping[str, object], object]', mean_m,
            ),
            cast(
                'Measurable[Mapping[str, object], object]', std_m,
            ),
        ),
    )



