"""Persistence — parquet + zarr round-trip for schema rows.

Most row types write to a flat columnar parquet via `pl.DataFrame
([r.as_dict() for r in rows]).write_parquet(path)`. No JSON
wrapping, no struct columns: each typed-provenance field becomes
its own typed column, and each `measurements` entry becomes its
own typed column at top level. Querying HPs at the dataframe
level just works (`df.filter(pl.col('optimizer.inner.lr') < 1e-3)`).

Heterogeneous measurement keys across rows: parquet requires every
column to have one type per file, but different cells/comparisons
can carry different paths. Polars handles missing columns by null-
padding when constructing the DataFrame from a list-of-dicts —
rows that don't carry a path get null in that column, which
`from_row_dict` skips on read.

`TraceRow` has a SECOND backend: `arrays` (multi-dim numpy arrays)
goes to a zarr store keyed by `{cell_id}/{array_name}`. Parquet
nested-list columns work technically for 2-D+ data but suffer from
Python list materialisation overhead and opaque queries. Zarr is
the right fit. The `write_tracerows(rows, parquet_path, *,
zarr_path=None)` signature lets the caller persist scalars+1-D to
parquet and arrays to zarr in one call.

`apply_trace_reductions(traces, add, drop)` is the polars-expr
post-trace hook — authors declare reductions as polars exprs +
an explicit drop list, applied in-memory before persisting. Operates
on the parquet side (leaves); array-side reductions belong in
plain numpy in the consumer."""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import polars as pl

from corroborate._polars_boundary import to_dicts as _to_dicts
from corroborate.schema import (
    ArrayLeaf,
    ComparisonRow,
    RunRow,
    TraceRow,
)


# ============ RunRow ============

def write_runrows(rows: Iterable[RunRow], path: Path) -> None:
    """Write RunRows to a flat columnar parquet. Round-trip pair:
    `read_runrows`."""
    records = [r.as_dict() for r in rows]
    pl.DataFrame(records).write_parquet(path)


def read_runrows(path: Path) -> list[RunRow]:
    df = pl.read_parquet(path)
    return [RunRow.from_row_dict(d) for d in _to_dicts(df)]


# ============ ComparisonRow ============

def write_comparisonrows(rows: Iterable[ComparisonRow], path: Path) -> None:
    records = [r.as_dict() for r in rows]
    pl.DataFrame(records).write_parquet(path)


def read_comparisonrows(path: Path) -> list[ComparisonRow]:
    df = pl.read_parquet(path)
    return [ComparisonRow.from_row_dict(d) for d in _to_dicts(df)]


# ============ TraceRow ============

# Trace persistence: scalars + 1-D series go to parquet (flat
# columnar, polars-queryable). Multi-dim arrays go to zarr keyed
# by `{cell_id}/{array_name}`. Both stores join on cell_id.

def write_tracerows(
    rows: Iterable[TraceRow],
    parquet_path: Path,
    *,
    zarr_path: Path | None = None,
) -> None:
    """Write TraceRows: scalars + 1-D series → parquet; multi-dim
    arrays → zarr (if `zarr_path` is given).

    Each row's `leaves` mapping flattens to top-level parquet
    columns. Each row's `arrays` mapping is split out: for each
    `(name, ndarray)` pair, writes
    `zarr_path/{cell_id}/{name}` as a zarr array with zstd
    compression. If `zarr_path is None` and any row has non-empty
    arrays, raises ValueError — silent data loss is the wrong
    default.

    Round-trip pair: `read_tracerows`."""
    rows_list = list(rows)
    has_arrays = any(r.arrays for r in rows_list)
    if has_arrays and zarr_path is None:
        raise ValueError(
            'TraceRow.arrays present but zarr_path not provided. '
            'Pass zarr_path to persist multi-dim arrays.',
        )

    # Parquet side: scalars + 1-D series.
    records = [r.as_dict() for r in rows_list]
    pl.DataFrame(records).write_parquet(parquet_path)

    # Zarr side: multi-dim arrays per cell.
    if zarr_path is None or not has_arrays:
        return
    import zarr  # type: ignore[reportMissingTypeStubs]
    root = zarr.open_group(  # type: ignore[reportUnknownMemberType]
        str(zarr_path), mode='a',
    )
    for r in rows_list:
        if not r.arrays:
            continue
        # Overwrite-on-conflict: re-running a sweep should refresh
        # the zarr group for that cell, not append-and-corrupt.
        if r.id in root:
            del root[r.id]  # type: ignore[reportUnknownMemberType]
        grp = root.create_group(r.id)  # type: ignore[reportUnknownMemberType]
        for name, arr in r.arrays.items():
            np_arr = np.asarray(arr)
            zarr_arr = grp.create_array(  # type: ignore[reportUnknownMemberType]
                name=name, shape=np_arr.shape, dtype=np_arr.dtype,
            )
            zarr_arr[:] = np_arr  # type: ignore[reportUnknownMemberType]


def read_tracerows(
    parquet_path: Path,
    *,
    zarr_path: Path | None = None,
) -> list[TraceRow]:
    """Read TraceRows: scalars + 1-D series from parquet; multi-dim
    arrays from zarr (if `zarr_path` is given and the cell has a
    group there).

    For each row in parquet, looks up `zarr_path/{cell_id}/` and
    materialises every array as numpy via `arr[:]`. Lazy-load is
    NOT yet exposed; callers that don't want arrays in memory
    should call without `zarr_path`."""
    df = pl.read_parquet(parquet_path)
    parquet_dicts = list(_to_dicts(df))

    arrays_by_id: dict[str, Mapping[str, ArrayLeaf]] = {}
    if zarr_path is not None and Path(zarr_path).exists():
        import zarr  # type: ignore[reportMissingTypeStubs]
        root = zarr.open_group(  # type: ignore[reportUnknownMemberType]
            str(zarr_path), mode='r',
        )
        for cell_id in root:  # type: ignore[reportUnknownVariableType]
            cell_grp = root[cell_id]  # type: ignore[reportUnknownMemberType]
            arrays: dict[str, ArrayLeaf] = {}
            for arr_name in cell_grp:  # type: ignore[reportUnknownVariableType]
                arrays[arr_name] = np.asarray(  # type: ignore[reportUnknownArgumentType]
                    cell_grp[arr_name][:]  # type: ignore[reportUnknownMemberType]
                )
            arrays_by_id[str(cell_id)] = arrays

    out: list[TraceRow] = []
    for d in parquet_dicts:
        cell_id = d.get('id')
        cell_arrays = (
            arrays_by_id.get(str(cell_id)) if cell_id is not None else None
        )
        out.append(TraceRow.from_row_dict(d, arrays=cell_arrays))
    return out


# ============ Polars-expr post-trace reductions ============

def apply_trace_reductions(
    traces: Sequence[TraceRow],
    *,
    add: Sequence[pl.Expr] = (),
    drop: Sequence[str] = (),
) -> list[TraceRow]:
    """Apply polars exprs to a batch of TraceRows; optionally
    drop named source columns after.

    Use case: a sweep produces 3-D record arrays (e.g.
    `online_q_values` shape `(steps, batch, n_actions)`) that
    dominate trace-store disk usage. Authors who want only
    derived summaries (e.g. per-step max-Q, online-target
    correlation) declare the reductions as polars exprs +
    explicitly drop the source 3-D arrays. The reduced traces
    are dramatically smaller; the same exprs work post-hoc on
    persisted full traces, so the analysis intent is portable.

    `add`: polars exprs that produce new columns (one per expr).
    Each expr's output becomes a new leaf in the trace.

    `drop`: source column names to remove AFTER computing `add`.
    Explicit so authors signal "I'm willing to discard the raw
    data in exchange for these reductions."

    Empty `add` + empty `drop` returns `traces` unchanged.

    Authors operate on TraceRow leaves at the polars-list level:
    a 3-D leaf is `List(List(List(Float64)))`; `list.eval(
    pl.element().list.max())` collapses one inner dim, etc."""
    if not add and not drop:
        return list(traces)
    df = pl.DataFrame([t.as_dict() for t in traces])
    if add:
        df = df.with_columns(*add)
    if drop:
        df = df.drop(*drop)
    return [TraceRow.from_row_dict(d) for d in _to_dicts(df)]
