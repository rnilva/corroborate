"""Shared Polars filtering semantics for bridge and deferred scopes."""
from __future__ import annotations

import polars as pl


def filter_cells(df: pl.DataFrame, expr: pl.Expr) -> pl.DataFrame:
    """Filter cells, resolving or null-padding referenced columns.

    Registered measurable columns are computed on demand. Other missing
    columns are filled with null, so their predicate excludes rows rather
    than raising. Repeated references are deduplicated before materializing
    missing columns.
    """
    referenced = list(dict.fromkeys(expr.meta.root_names()))
    missing = [column for column in referenced if column not in df.columns]
    if not missing:
        return df.filter(expr)

    from corroborate.measurables import compute_missing_columns

    df = compute_missing_columns(df, missing)
    truly_missing = [column for column in missing if column not in df.columns]
    if truly_missing:
        df = df.with_columns(
            [pl.lit(None).alias(column) for column in truly_missing],
        )
    return df.filter(expr)


__all__ = ['filter_cells']
