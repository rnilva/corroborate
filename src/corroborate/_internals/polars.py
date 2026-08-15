"""Polars boundary — typed wrappers over polars' `Any`-leaky surface.

Polars uses `Any` for heterogeneous cell values
(`DataFrame.to_dicts() -> list[dict[str, Any]]`,
`iter_rows(named=True) -> Iterator[dict[str, Any]]`,
`Series.item() -> Any`). The framework's `Mapping[str, object]` /
typed-scalar discipline forbids casual `Any`. This module narrows
the few polars surfaces the framework consumes; downstream callers
narrow per-field via `_narrow.is_*` TypeIs predicates and
`require_*` accessors.

Same rationale as `_json_boundary`. Each function laundering an
`Any`-typed return is the framework's ONE allowed escape hatch for
that polars surface — the runtime invariant (heterogeneous cell
values) cannot be expressed in polars' static contract.

Module name is underscore-prefixed to signal **internal use
only**. External users should import polars directly."""
from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence

import polars as pl


def to_dicts(df: pl.DataFrame) -> Sequence[Mapping[str, object]]:
    """Decode a polars DataFrame to a `Sequence[Mapping[str,
    object]]`. `Sequence` (covariant) and `Mapping` (covariant in
    value) widen polars' `list[dict[str, Any]]` cleanly — the
    framework's value-side narrowing happens via TypeIs predicates
    downstream."""
    return df.to_dicts()


def iter_dicts(df: pl.DataFrame) -> Iterator[Mapping[str, object]]:
    """Streaming variant of `to_dicts` — iterate per-row dicts
    without materialising the full list.

    Wraps `df.iter_rows(named=True)`, which is typed
    `Iterator[dict[str, Any]]`. `Iterator` is covariant and
    `Mapping` is covariant in value, so the widen lands without
    a `pyright: ignore`."""
    return df.iter_rows(named=True)


def scalar_int(lf: pl.LazyFrame, expr: pl.Expr) -> int:
    """Materialise a scalar-int column from a LazyFrame via
    `lf.select(expr).collect().item()`.

    Polars' `.item()` returns `Any` (cell type is heterogeneous in
    general); narrowing via isinstance (with explicit `bool`
    exclusion since `bool ⊂ int` in Python) is the typed exit.
    Raises `TypeError` if the materialised cell isn't an int."""
    # `Any` source: polars' `Series.item() -> Any`. Laundered
    # immediately by the isinstance guard below.
    v = lf.select(expr).collect().item()
    if isinstance(v, bool) or not isinstance(v, int):
        raise TypeError(
            f'expected int scalar from polars LazyFrame, got '
            f'{type(v).__name__}',
        )
    return v


def series_std_float(series: pl.Series) -> float:
    """Std of a float-typed polars Series, narrowed to `float`.

    Polars' `Series.std()` is typed `float | timedelta | None`
    because polars admits time-typed columns (where std is a
    duration). Callers that already filtered to float columns
    (e.g. via `dtype.is_float()`) get a typed exit here without
    repeating the timedelta narrow at every site. `None` is
    returned by polars on empty / all-null series; the helper
    coerces to `0.0` to match the `or 0.0` idiom callers used
    inline before this boundary existed."""
    v = series.std()
    if v is None:
        return 0.0
    if not isinstance(v, (int, float)):
        raise TypeError(
            f"expected float from polars Series.std(), got "
            f"{type(v).__name__} (column dtype must be float)",
        )
    return float(v)


def is_list_of_list(df: pl.DataFrame, col: str) -> bool:
    """True iff `col` is a 2-D `List(List(...))` polars column.

    `pl.List.inner` is typed `PolarsDataType` (a private union alias
    `DataType | DataTypeClass`); accessing `.inner` on the base
    `DataType` (what `df.schema[c]` returns) raises
    `reportAttributeAccessIssue`. The narrowing via two nested
    `isinstance` checks lives in this boundary so callers consume a
    typed `bool` without the attribute-access leak."""
    dtype = df.schema[col]
    if not isinstance(dtype, pl.List):
        return False
    return isinstance(dtype.inner, pl.List)
