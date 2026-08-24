"""Shared per-stratum kernel functions.

The kernel extracts the analytical core of the canonical
per-stratum primitives so that @analysis adapters and
`Panel.derive` delegate to ONE implementation. Without this,
the aggregator semantics could drift between `Panel.derive`
and `cross_stratum_property_slope._derive_per_stratum_covariate`
(both compute per-(env, arm) aggregates) — and any new
implementation-author primitive would reinvent the wheel.

The kernel takes a `pl.DataFrame` + structured spec and returns
typed results. `to_dataframe` is the conversion boundary
that adapts the legacy `Iterable[Mapping]` data input to the
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
    data: pl.DataFrame,
    *,
    column: str,
    aggregator: Literal['mean', 'std', 'median'],
    stratify_by: tuple[str, ...],
    row_filter: pl.Expr | None = None,
    min_n: int = 1,
) -> Mapping[tuple[object, ...], float]:
    """Per-stratum scalar aggregate of `column` via `aggregator`,
    optionally narrowed by `row_filter`, keyed by the `stratify_by`
    columns. Returns `{stratum_id: aggregate_value}`.

    Rows with non-finite `column` value are dropped before
    aggregation. Strata with fewer than `min_n` surviving data
    are skipped (SD undefined at n<2; implementation-author chooses
    a higher floor for power-sensitive uses).

    Shared kernel for `Panel.derive` and
    `cross_stratum_property_slope._derive_per_stratum_covariate`
    (Phase 2 migration target). Both call sites converge here so
    aggregator semantics can't drift across the
    Iterable[Mapping] ↔ Panel surface."""
    if column not in data.columns or not stratify_by:
        return {}
    if data.height == 0:
        return {}
    if row_filter is not None:
        data = data.filter(row_filter)
    if data.height == 0:
        return {}
    # Drop non-finite column values.
    if data.schema[column].is_float():
        data = data.filter(pl.col(column).is_finite())
    else:
        data = data.filter(pl.col(column).is_not_null())
    if data.height == 0:
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
        data.group_by(list(stratify_by))
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


def to_dataframe(
    data: pl.DataFrame | Iterable[Mapping[str, object]],
) -> pl.DataFrame:
    """Materialise rows into the polars DataFrame every @analysis
    primitive takes. A DataFrame passes through unchanged.

    This is the ENTRY-BOUNDARY conversion, called once by whoever
    holds rows — the bridge evaluator (its gate rows), tests built
    from `RunRow.as_dict()`, ad-hoc dict lists. The `data` INPUT
    of a primitive is never converted inside the primitive —
    analyses are plain-DataFrame functions a polars user can call
    with no framework context; in-tree delegates that DERIVE new
    row collections hand them to rows-level cores rather than
    encoding a frame the callee would immediately decode.

    The two-arm input type IS the polymorphic boundary CLAUDE.md's
    "no `object` parameters" rule permits — the body narrows
    immediately via `isinstance(data, pl.DataFrame)` so the
    `Iterable[Mapping[str, object]]` branch carries the
    fully-typed shape through.

    Uses `pl.from_dicts(..., infer_schema_length=None)` for the
    Iterable[Mapping] branch — scans the full input for type
    inference. `pl.DataFrame(data)` blows up on real-world
    heterogeneous schemas where row N introduces a column type
    the first few rows didn't carry (cache data from diagonal-
    relaxed-concat'd corpora trip this regularly)."""
    if isinstance(data, pl.DataFrame):
        return data
    rows_list: list[Mapping[str, object]] = list(data)
    if not rows_list:
        return pl.DataFrame()
    _require_consistent_kinds(rows_list)
    # Skip the `[dict(c) for c in rows_list]` copy when data
    # are already concrete dicts — polars' `from_dicts` accepts
    # any Mapping. The copy was insurance against frozen-dict
    # subtypes but the cost was real on 8k-row caches.
    return pl.from_dicts(rows_list, infer_schema_length=None)


def _scalar_kind(value: object) -> str | None:
    """Coarse category for the mixed-kind check: numeric
    (int/float — polars widens losslessly), bool, str. None and
    nested values return None — absence isn't evidence, and
    nested shapes are polars' own to accept or reject."""
    if value is None or isinstance(value, (list, dict)):
        return None
    if isinstance(value, bool):
        return 'bool'
    if isinstance(value, (int, float)):
        return 'numeric'
    if isinstance(value, str):
        return 'str'
    return None


def _require_consistent_kinds(
    rows: list[Mapping[str, object]],
) -> None:
    """polars' full-scan inference resolves a column holding both
    numbers and text by silently stringifying the numbers — after
    which numeric readers see NaN for every row and a verdict can
    quietly degrade. A column mixing scalar categories across
    rows is a producer error; refuse it by name instead."""
    kinds: dict[str, str] = {}
    for row in rows:
        for key, value in row.items():
            kind = _scalar_kind(value)
            if kind is None:
                continue
            earlier = kinds.setdefault(key, kind)
            if earlier != kind:
                raise ValueError(
                    f'to_dataframe: column {key!r} mixes '
                    f'{earlier} and {kind} values across rows — '
                    f'polars would silently unify these to '
                    f'strings; repair the rows instead',
                )


__all__ = [
    'to_dataframe',
    'per_stratum_aggregate',
]
