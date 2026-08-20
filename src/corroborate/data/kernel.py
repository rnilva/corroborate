"""Shared per-stratum kernel functions.

The kernel extracts the analytical core of the canonical
per-stratum primitives so that @analysis adapters and
`Panel.derive` delegate to ONE implementation. Without this,
the aggregator semantics could drift between `Panel.derive`
and `cross_stratum_property_slope._derive_per_stratum_covariate`
(both compute per-(env, arm) aggregates) — and any new
implementation-author primitive would reinvent the wheel.

The kernel takes a `pl.DataFrame` + structured spec and returns
typed results. `cells_to_dataframe` is the conversion boundary
that adapts the legacy `Iterable[Mapping]` cells input to the
DataFrame-canonical analysis surface.

Functions in this module are framework-internal stable — the
analysis primitives are the public surface, not the kernel. But
the kernel is exposed as a single source of truth that
implementation-author analysis-authoring can call directly when none
of the canonical analyses fit."""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Literal

import polars as pl


def per_stratum_aggregate(
    cells: pl.DataFrame,
    *,
    column: str,
    aggregator: Literal['mean', 'std', 'median'],
    stratify_by: tuple[str, ...],
    cell_filter: pl.Expr | None = None,
    min_n: int = 1,
) -> Mapping[tuple[object, ...], float]:
    """Per-stratum scalar aggregate of `column` via `aggregator`,
    optionally narrowed by `cell_filter`, keyed by the `stratify_by`
    columns. Returns `{stratum_id: aggregate_value}`.

    Cells with non-finite `column` value are dropped before
    aggregation. Strata with fewer than `min_n` surviving cells
    are skipped (SD undefined at n<2; implementation-author chooses
    a higher floor for power-sensitive uses).

    Shared kernel for `Panel.derive` and
    `cross_stratum_property_slope._derive_per_stratum_covariate`
    (Phase 2 migration target). Both call sites converge here so
    aggregator semantics can't drift across the
    Iterable[Mapping] ↔ Panel surface."""
    if column not in cells.columns or not stratify_by:
        return {}
    if cells.height == 0:
        return {}
    if cell_filter is not None:
        cells = cells.filter(cell_filter)
    if cells.height == 0:
        return {}
    # Drop non-finite column values.
    if cells.schema[column].is_float():
        cells = cells.filter(pl.col(column).is_finite())
    else:
        cells = cells.filter(pl.col(column).is_not_null())
    if cells.height == 0:
        return {}
    if aggregator == 'mean':
        agg_expr = pl.col(column).mean()
    elif aggregator == 'std':
        # Sample SD (ddof=1) — convention matches the prior
        # `_derive_per_stratum_covariate` impl + numpy's
        # ddof=1 default for std.
        agg_expr = pl.col(column).std(ddof=1)
    else:
        agg_expr = pl.col(column).median()
    grouped = (
        cells.group_by(list(stratify_by))
        .agg([agg_expr.alias('_agg'), pl.len().alias('_n')])
    )
    out: dict[tuple[object, ...], float] = {}
    rows_obj: list[dict[str, object]] = [
        dict(r) for r in grouped.iter_rows(named=True)
    ]
    for row in rows_obj:
        n_raw = row['_n']
        if not isinstance(n_raw, (int, float)):
            continue
        n = int(n_raw)
        if n < min_n:
            continue
        v = row['_agg']
        if not isinstance(v, (int, float)):
            continue
        v_f = float(v)
        if not math.isfinite(v_f):
            continue
        stratum_id: tuple[object, ...] = tuple(
            row[k] for k in stratify_by
        )
        out[stratum_id] = v_f
    return out


def cells_to_dataframe(
    cells: pl.DataFrame | Iterable[Mapping[str, object]],
) -> pl.DataFrame:
    """Materialise rows into the polars DataFrame every @analysis
    primitive takes. A DataFrame passes through unchanged.

    This is the ENTRY-BOUNDARY conversion, called once by whoever
    holds rows — the bridge evaluator (its gate rows), tests built
    from `RunRow.as_dict()`, ad-hoc dict lists — never inside the
    primitives themselves: analyses are plain-DataFrame functions
    a polars user can call with no framework context.

    The two-arm input type IS the polymorphic boundary CLAUDE.md's
    "no `object` parameters" rule permits — the body narrows
    immediately via `isinstance(cells, pl.DataFrame)` so the
    `Iterable[Mapping[str, object]]` branch carries the
    fully-typed shape through.

    Uses `pl.from_dicts(..., infer_schema_length=None)` for the
    Iterable[Mapping] branch — scans the full input for type
    inference. `pl.DataFrame(cells)` blows up on real-world
    heterogeneous schemas where row N introduces a column type
    the first few rows didn't carry (cache cells from diagonal-
    relaxed-concat'd corpora trip this regularly)."""
    if isinstance(cells, pl.DataFrame):
        return cells
    cells_list: list[Mapping[str, object]] = list(cells)
    if not cells_list:
        return pl.DataFrame()
    # Skip the `[dict(c) for c in cells_list]` copy when cells
    # are already concrete dicts — polars' `from_dicts` accepts
    # any Mapping. The copy was insurance against frozen-dict
    # subtypes but the cost was real on 8k-cell caches.
    return pl.from_dicts(cells_list, infer_schema_length=None)


__all__ = [
    'cells_to_dataframe',
    'per_stratum_aggregate',
]
