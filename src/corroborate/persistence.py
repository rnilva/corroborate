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

from collections.abc import Iterable, Iterator, Mapping, Sequence
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
    from zarr.codecs import (  # type: ignore[reportMissingTypeStubs]
        BloscCodec,
    )
    # Blosc/zstd@3 — fast + small. Per-array shuffle helps for
    # smooth float arrays (Q-values, td-errors), neutral for ints.
    compressor = BloscCodec(  # type: ignore[reportUnknownMemberType]
        cname='zstd', clevel=3, shuffle='shuffle',
    )
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
                compressors=compressor,
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


# ============ Dtype tightening for trace stores ============

def tighten_trace_dtypes(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Cast `List(Float64)` columns to `List(Float32)` and
    `List(Int64)` columns to `List(Int32)`. No-op for any
    other dtype.

    Applied at trace-store write / merge time. The framework's
    sweep emits per-step series via `arr.tolist()`, which upcasts
    JAX float32 → Python float (= float64 in polars) and JAX
    int32 → Python int (= int64 in polars). The original JAX
    arrays were narrower; the upcast is a round-trip waste. This
    helper undoes it, halving the per-step series storage size at
    write time with zero information loss.

    For the §3 corpus this is a ~13% on-disk reduction (1.60 GB
    → 1.39 GB) plus *faster* writes (less data to compress).
    Larger savings stack with int range narrowing — int columns
    whose true range fits in int8 or int16 could be tightened
    further; this helper sticks to int64 → int32 because the
    range is universally safe."""
    schema = lf.collect_schema()
    casts: list[pl.Expr] = []
    for name, dt in schema.items():
        if dt == pl.List(pl.Float64):
            casts.append(pl.col(name).cast(pl.List(pl.Float32)))
        elif dt == pl.List(pl.Int64):
            casts.append(pl.col(name).cast(pl.List(pl.Int32)))
    if casts:
        return lf.with_columns(casts)
    return lf


# ============ Streaming reader (memory-bounded) ============

def iter_trace_records(
    parquet_path: Path,
    *,
    columns: Sequence[str] | None = None,
    batch_size: int = 32,
    zarr_path: Path | None = None,
) -> Iterator[Mapping[str, object]]:
    """Stream per-cell trace records from a parquet without
    full-corpus materialization. Yields one dict per cell; the
    dict is a polars row dict (1-D `List` columns surface as
    Python `list[float]` / `list[int]`; scalars surface as their
    native type).

    `columns` — optional projection. Only the named columns are
    read; saves substantial bandwidth + memory when the consumer
    only needs a subset of per-step series. `'id'` is always
    included even if not in the projection.

    `batch_size` — slice size in rows. Smaller → lower peak
    memory; larger → fewer parquet re-opens. The default 32
    bounds memory at ~`batch_size × per-row-size` even when the
    corpus is large.

    `zarr_path` — when provided, attaches the per-cell zarr
    arrays to each yielded record under their original keys
    (numpy ndarrays). Skipped by default to keep the streaming
    path memory-bounded; pass it only when the consumer actually
    reads multi-dim arrays.

    Memory-bounded alternative to `read_tracerows` for post-hoc
    per-cell projections (mediator computation, fact extraction
    across the corpus, ...) where TraceRow's typed shape isn't
    needed. Drops the ~30× Python-object overhead of
    `to_dicts(df)` materialisation by reading slice-by-slice.

    Round-trip pair: produced by `write_tracerows`."""
    proj: list[str] | None
    if columns is not None:
        proj = list(columns)
        if 'id' not in proj:
            proj = ['id', *proj]
    else:
        proj = None

    arrays_by_id: dict[str, Mapping[str, np.ndarray]] = {}
    if zarr_path is not None and Path(zarr_path).exists():
        import zarr  # type: ignore[reportMissingTypeStubs]
        root = zarr.open_group(  # type: ignore[reportUnknownMemberType]
            str(zarr_path), mode='r',
        )
        for cell_id in root:  # type: ignore[reportUnknownVariableType]
            cell_grp = root[cell_id]  # type: ignore[reportUnknownMemberType]
            arrays: dict[str, np.ndarray] = {}
            for arr_name in cell_grp:  # type: ignore[reportUnknownVariableType]
                arrays[arr_name] = np.asarray(  # type: ignore[reportUnknownArgumentType]
                    cell_grp[arr_name][:]  # type: ignore[reportUnknownMemberType]
                )
            arrays_by_id[str(cell_id)] = arrays

    n_rows = pl.scan_parquet(parquet_path).select(
        pl.len(),
    ).collect().item()
    if not isinstance(n_rows, int):
        raise TypeError(
            f'expected int row count from parquet, got {type(n_rows)}',
        )

    for start in range(0, n_rows, batch_size):
        lf = pl.scan_parquet(parquet_path)
        if proj is not None:
            lf = lf.select(proj)
        df = lf.slice(start, batch_size).collect()
        for row in df.iter_rows(named=True):
            cell_id = row.get('id')
            if (zarr_path is not None
                    and isinstance(cell_id, str)
                    and cell_id in arrays_by_id):
                # Attach arrays under their original keys, alongside
                # the parquet-side fields.
                merged: dict[str, object] = dict(row)
                merged.update(arrays_by_id[cell_id])
                yield merged
            else:
                yield row


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
