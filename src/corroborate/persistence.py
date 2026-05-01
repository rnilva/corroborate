"""Persistence — parquet round-trip for schema rows.

Every row type writes to a flat columnar parquet via `pl.DataFrame
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

`TraceRow` carries multi-dim arrays in `leaves` as nested-list
columns. Polars infers narrow dtype from numpy arrays at write
time (`List(Float32)` / `List(Int32)` / `List(Array(<scalar>,
shape=N))`); at read time the streaming reader
(`iter_trace_records`) keeps memory bounded by yielding row
slices instead of materialising the full corpus.

`apply_trace_reductions(traces, add, drop)` is the polars-expr
post-trace hook — authors declare reductions as polars exprs +
an explicit drop list, applied in-memory before persisting."""
from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path

import polars as pl

import pyarrow as pa
import pyarrow.parquet as pq

from corroborate._polars_boundary import to_dicts as _to_dicts
from corroborate.schema import (
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

def write_tracerows(
    rows: Iterable[TraceRow],
    parquet_path: Path,
) -> None:
    """Write TraceRows to a flat columnar parquet. Each row's
    `leaves` mapping flattens to top-level columns; multi-dim
    arrays land in nested-list columns (polars infers narrow
    dtype from numpy at write time). Round-trip pair:
    `read_tracerows`."""
    rows_list = list(rows)
    records = [r.as_dict() for r in rows_list]
    pl.DataFrame(records).write_parquet(parquet_path)


def read_tracerows(parquet_path: Path) -> list[TraceRow]:
    """Read TraceRows from parquet. Materialises the whole file;
    for memory-bounded streaming reads use `iter_trace_records`."""
    df = pl.read_parquet(parquet_path)
    return [TraceRow.from_row_dict(d) for d in _to_dicts(df)]


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
) -> Iterator[Mapping[str, object]]:
    """Stream per-cell trace records from a parquet without
    full-corpus materialization. Yields one dict per cell; the
    dict is a polars row dict (`List` columns surface as Python
    `list[float]` / `list[int]` / nested lists for 2-D+; scalars
    surface as their native type).

    `columns` — optional projection. Only the named columns are
    read; saves substantial bandwidth + memory when the consumer
    only needs a subset of per-step series. `'id'` is always
    included even if not in the projection.

    `batch_size` — slice size in rows. Smaller → lower peak
    memory; larger → fewer parquet re-opens. The default 32
    bounds memory at ~`batch_size × per-row-size` even when the
    corpus is large.

    Memory-bounded alternative to `read_tracerows` for post-hoc
    per-cell projections (measurable computation, fact extraction
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


# ============ Streaming concat across many per-arm parquets ============

def stream_concat_parquets(
    inputs: Sequence[Path | str], out: Path, *,
    type_widening: bool = True,
    compression: str = 'zstd',
    compression_level: int = 3,
) -> None:
    """Concatenate `inputs` to `out` via polars'
    `concat(how='diagonal_relaxed')` — null-pads missing columns
    across inputs AND auto-promotes types across schema
    differences (int→float when any input has float for the same
    field; list-of-int→list-of-float for nested lists; large_list
    and list handled identically).

    `inputs` accepts either local `Path`s or fsspec URI strings
    (e.g. `s3://bucket/path/file.parquet`); polars dispatches via
    fsspec for URI inputs. Mixing both in one call is allowed.

    `diagonal_relaxed` is necessary because per-arm parquets in
    a sweep can disagree on column SET (DDQN arms emit
    `invariant.at_most[jensen_dormancy_gap<=0].*` that vanilla
    arms don't). The strict `vertical_relaxed` errors on column-
    set mismatches; the merge primitive at the parquet boundary
    has to handle the realistic case where two arms authored
    different intervention_arms / different invariants.

    `type_widening=True` (default) uses `diagonal_relaxed`. Set
    False for strict diagonal concat that errors on type
    mismatches but still null-pads missing columns.

    Reads all inputs into memory before writing — polars'
    `sink_parquet` silently produces an empty file when
    `how='vertical_relaxed'` requires schema resolution it
    can't perform lazily. For corpus-scale (10s of GB) merges,
    use a chunked driver upstream rather than this primitive."""
    if not inputs:
        raise ValueError('stream_concat_parquets: no inputs')
    if out.exists():
        out.unlink()
    how = 'diagonal_relaxed' if type_widening else 'diagonal'
    # `glob=False`: arm-tag relpaths embed `wrap[<wrapper>(<args>)]`
    # which polars otherwise treats as glob character classes,
    # producing "expanded paths were empty" on S3 URIs. Each input
    # is a single concrete path, never a glob — so disable globbing.
    eager_frames = [pl.read_parquet(str(p), glob=False) for p in inputs]
    merged = pl.concat(eager_frames, how=how)
    merged.write_parquet(
        str(out),
        compression=compression, compression_level=compression_level,
    )
