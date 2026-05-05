"""Polars predicate helpers — NaN-safe scope expressions.

Polars deviates from IEEE-754 in two places that bridge authors
trip on when authoring `Bridge.scope=` predicates:

1. `pl.col(x) > threshold` returns True for `NaN`, admitting the
   row. IEEE-754 says all NaN comparisons are False; polars's
   filter context promotes `null > threshold` to True. Forgetting
   the `is_finite()` guard silently lets non-finite cells leak
   through bridge scopes — a verdict-shifting bug.

2. `pl.col(x).mean()` propagates NaN — a single NaN in the column
   makes the aggregate NaN. Combined with `.over(partition)` for
   stratified means, one missing cell poisons the whole partition.
   `.fill_nan(None)` flips NaN to null, which polars aggregates
   skip natively.

These helpers wrap both pitfalls so bridge authors compose
predicates without remembering the guards. Substrate-neutral:
callers pick partition keys appropriate to their substrate (RL
uses `'env_name'`; other substrates use their own).

Mirrors the framework's typed-discipline: every helper returns a
`pl.Expr` that composes via `&` / `|` with other scope predicates.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import polars as pl


type _Source = str | pl.Expr
type _Partition = str | Sequence[str]
type _Op = Literal['mean', 'median', 'min', 'max', 'sum']


def _coerce(source: _Source) -> pl.Expr:
    """Accept either a column name (`str`) or a `pl.Expr`. Bridge
    scope authors typically pass a column name as a string."""
    return pl.col(source) if isinstance(source, str) else source


def finite(source: _Source) -> pl.Expr:
    """`source.is_finite()` — True iff the value is neither NaN
    nor infinite. Use as a standalone predicate or compose with
    other expressions via `&` / `|`."""
    return _coerce(source).is_finite()


def finite_gt(source: _Source, threshold: float) -> pl.Expr:
    """NaN-safe `source > threshold`. Without the `is_finite()`
    guard, polars returns True for `NaN > threshold` (admits
    the row), letting non-finite cells leak through bridge
    scopes. Returns False on NaN/inf; True on finite values
    strictly greater than `threshold`."""
    col = _coerce(source)
    return col.is_finite() & (col > threshold)


def finite_lt(source: _Source, threshold: float) -> pl.Expr:
    """NaN-safe `source < threshold`. See `finite_gt` for
    rationale."""
    col = _coerce(source)
    return col.is_finite() & (col < threshold)


def finite_ge(source: _Source, threshold: float) -> pl.Expr:
    """NaN-safe `source >= threshold`."""
    col = _coerce(source)
    return col.is_finite() & (col >= threshold)


def finite_le(source: _Source, threshold: float) -> pl.Expr:
    """NaN-safe `source <= threshold`."""
    col = _coerce(source)
    return col.is_finite() & (col <= threshold)


def finite_between(
    source: _Source, low: float, high: float,
) -> pl.Expr:
    """NaN-safe `low <= source <= high`."""
    col = _coerce(source)
    return col.is_finite() & (col >= low) & (col <= high)


def partition_aggregate(
    source: _Source,
    *,
    by: _Partition,
    op: _Op = 'mean',
) -> pl.Expr:
    """NaN-safe partition aggregate of `source` over `by`.

    Returns a per-row `pl.Expr` where each row carries its
    partition's aggregate value (mean / median / min / max / sum).
    NaN values in `source` are treated as null and excluded from
    the aggregate, so a single NaN doesn't poison the partition's
    aggregate as `pl.col(x).mean()` would.

    `by` is a single column name or a sequence of names for
    multi-key partitioning. The framework stays substrate-neutral
    — callers pick partition keys appropriate to their substrate
    (RL substrate uses `'env_name'`; other substrates use their
    own).

    Example:
        scope = (
            (pl.col('total_steps') >= 1_000_000)
            & (partition_aggregate(
                'q_divergence_score',
                by=['env_name', 'total_steps'],
                op='mean',
              ) > 1.0)
        )

    **Sharp edge — over() vs other filter predicates:** polars's
    `.over(by)` aggregates over the input dataframe *before* other
    predicates in the same `df.filter(...)` expression resolve.
    If a sibling predicate restricts to a sub-cohort (e.g.,
    `lr == 0.001`), `partition_aggregate` does NOT see that
    restriction — its mean spans all values of `lr` for matching
    `(env, capacity, ...)` groups. Symptom: the helper's
    partition-mean comes back surprisingly high (or low), and
    `> threshold` filters drop cells you expected to keep.

    **Two workarounds:**

    1. Add the restricting columns to `by`. If a bridge filters
       `lr == 0.001` and partition-aggregates Q-divergence, set
       `by=['optimizer.inner.lr', 'env_name', 'replay.capacity']`
       — each (lr, env, capacity) gets its own mean, so the lr
       restriction is honored automatically.
    2. Apply the restricting filter in a separate `.filter(...)`
       call upstream (e.g., chain `df.filter(...).filter(...)`).
       Polars evaluates filters sequentially, so the second
       filter's `over()` sees the first filter's output. Bridge
       authors pinned to `Bridge.scope: pl.Expr` (single
       expression) should prefer (1).

    Both work; (1) is more explicit and survives scope expression
    composition.
    """
    col = _coerce(source).fill_nan(None)
    match op:
        case 'mean':
            agg = col.mean()
        case 'median':
            agg = col.median()
        case 'min':
            agg = col.min()
        case 'max':
            agg = col.max()
        case 'sum':
            agg = col.sum()
    partition = [by] if isinstance(by, str) else list(by)
    return agg.over(partition)


__all__ = [
    'finite',
    'finite_between',
    'finite_ge',
    'finite_gt',
    'finite_le',
    'finite_lt',
    'partition_aggregate',
]
