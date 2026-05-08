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
an explicit drop list, applied in-memory before persisting.

`ComputationGraph` topology is persisted as a sidecar JSON file
(`graphs.json`) keyed by `Hypothesis.arm_key()` — one entry per
hypothesis-arm. The graph survives sweep-time provenance: post-
sweep consumers reconstruct the static call topology
(`@claim` invocation edges) without re-running the trace pass."""
from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Literal

import polars as pl

# Mirrors polars' private `ParquetCompression` alias (which sits in
# `polars._typing` — not part of the public surface). Inlining keeps
# our signature stable across polars internal renames and lets the
# checker reject typos at the call boundary.
ParquetCompression = Literal[
    'lz4', 'uncompressed', 'snappy', 'gzip', 'brotli', 'zstd',
]

from corroborate._internals.json import loads as _json_loads
from corroborate._internals.narrow import is_mapping_str_object
from corroborate._internals.polars import (
    iter_dicts as _iter_dicts,
    scalar_int as _scalar_int,
    to_dicts as _to_dicts,
)
from corroborate.graph.computation import (
    ComputationEdge,
    ComputationGraph,
)
from corroborate.graph import Graph
from corroborate.corpus.schema import (
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

    n_rows = _scalar_int(pl.scan_parquet(parquet_path), pl.len())

    for start in range(0, n_rows, batch_size):
        lf = pl.scan_parquet(parquet_path)
        if proj is not None:
            lf = lf.select(proj)
        df = lf.slice(start, batch_size).collect()
        yield from _iter_dicts(df)


# ============ Polars-expr post-trace reductions ============

def apply_trace_reductions(
    traces: Sequence[TraceRow],
    *,
    add: Sequence[pl.Expr] = (),
    drop: Sequence[str] = (),
) -> list[TraceRow]:
    """Apply polars exprs to a batch of TraceRows; optionally
    drop named source columns after.

    Use case: a sweep produces high-rank record arrays (e.g. a
    per-step `(steps, batch, action_dim)` tensor) that dominate
    trace-store disk usage. Authors who want only derived
    summaries (per-step reductions, pairwise correlations)
    declare the reductions as polars exprs + explicitly drop
    the source arrays. The reduced traces are dramatically
    smaller; the same exprs work post-hoc on persisted full
    traces, so the analysis intent is portable.

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

def atomic_write_parquet(df: pl.DataFrame, path: Path) -> None:
    """tmp+rename parquet write. Lives here so both runner-side
    cache writes and corpus-side measurements writes share one
    implementation; mirrors the pattern `stream_concat_parquets`
    uses internally for the merged output. C2 / I4 invariant
    (CACHE_BUILD.md / SWEEP_PERSISTENCY.md).

    A killed-mid-write process leaves no torn file at the
    consumer's path; consumers either see the pre-write state
    or the fully-written new state.

    **Concurrency**: the tmp file uses a unique suffix
    (`tempfile.mkstemp` in the destination directory). Two
    concurrent writes against the same `path` produce two
    different tmp inodes; whichever `replace`s last wins for
    the destination, but neither writer's content is silently
    truncated by the other — eliminating the fixed-`.partial`-
    suffix TOCTOU race where writer B unlinking writer A's
    in-progress partial would silently lose A's content. On
    failure the tmp file is unlinked (no `.partial`-flavored
    breadcrumbs left behind to grow into stale clutter)."""
    fd, tmp_str = tempfile.mkstemp(
        prefix=path.name + '.', suffix='.partial', dir=path.parent,
    )
    os.close(fd)  # polars opens its own writer
    tmp = Path(tmp_str)
    try:
        df.write_parquet(tmp)
        tmp.replace(path)
    except BaseException:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


def atomic_write_text(path: Path, content: str) -> None:
    """Same atomicity guarantee for text payloads (JSON sidecars,
    closure-hash manifests). Order matters at the call site:
    write the parquet ATOMICALLY first, THEN the sidecar — a
    half-updated state then has a stale sidecar pointing at a
    fresh parquet (drift detection on next run self-heals via
    column invalidation), rather than a fresh sidecar pointing
    at a torn parquet (consumer reads garbage).

    Concurrency: same unique-suffix tmp pattern as
    `atomic_write_parquet`."""
    fd, tmp_str = tempfile.mkstemp(
        prefix=path.name + '.', suffix='.partial', dir=path.parent,
    )
    tmp = Path(tmp_str)
    try:
        with os.fdopen(fd, 'w') as fh:
            _ = fh.write(content)
        tmp.replace(path)
    except BaseException:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


def _estimated_decompressed_bytes(path: Path | str) -> int:
    """Sum of `total_uncompressed_size` across the parquet's row
    groups. Falls back to 4× the on-disk size when metadata read
    fails (zstd typical compression ratio for trace data). Used
    by the adaptive chunk-sizer to keep peak merge RAM under a
    target budget rather than blindly trusting the static
    `chunk_size` default — which OOMs on multi-GB-decompressed
    trace shards."""
    try:
        import pyarrow.parquet as _pq
        meta = _pq.ParquetFile(str(path)).metadata
        total = 0
        for i in range(meta.num_row_groups):
            total += meta.row_group(i).total_byte_size
        if total > 0:
            return total
    except Exception:
        pass
    try:
        from pathlib import Path as _Path
        return _Path(path).stat().st_size * 4
    except Exception:
        return 0


def _adaptive_chunk_size(
    inputs: Sequence[Path | str],
    *,
    requested_chunk_size: int,
    target_ram_gb: float,
) -> int:
    """Lower the requested `chunk_size` when the per-file
    decompressed estimates suggest peak RAM would exceed
    `target_ram_gb`. Pessimistic toward the largest file: peak
    RAM during eager concat is roughly `chunk_size × max(
    decompressed_size)`; we solve for `chunk_size` that keeps it
    under the target.

    Never raises chunk_size above `requested_chunk_size` — the
    static default is a ceiling, the adaptive computation is a
    floor."""
    if not inputs:
        return requested_chunk_size
    sizes = [_estimated_decompressed_bytes(p) for p in inputs]
    max_size = max(sizes) if sizes else 0
    if max_size <= 0:
        return requested_chunk_size
    target_bytes = int(target_ram_gb * (1 << 30))
    safe = max(1, target_bytes // max_size)
    return min(requested_chunk_size, safe)


def stream_concat_parquets(
    inputs: Sequence[Path | str], out: Path, *,
    type_widening: bool = True,
    compression: ParquetCompression = 'zstd',
    compression_level: int = 3,
    chunk_size: int = 4,
    scratch_dir: Path | None = None,
    target_ram_gb: float = 8.0,
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
    a sweep can disagree on column SET (some arms emit invariant-
    bridge columns the others don't). The strict
    `vertical_relaxed` errors on column-set mismatches; the merge
    primitive at the parquet boundary has to handle the realistic
    case where two arms authored different intervention_arms /
    different invariants.

    `type_widening=True` (default) uses `diagonal_relaxed`. Set
    False for strict diagonal concat that errors on type
    mismatches but still null-pads missing columns.

    Memory: chunked across `chunk_size` inputs at a time. Each
    chunk is concatenated eagerly + written to a temp file; chunk
    files are then concatenated recursively. Peak memory is ~
    `chunk_size` decompressed inputs, NOT all inputs. Polars'
    `sink_parquet` is NOT used because it silently produces empty
    output for `how='diagonal_relaxed'` (schema resolution it
    can't perform lazily); chunked eager concat is the workaround.

    For 12 SpaceInvaders 1M trace shards (~580MB compressed each,
    ~5GB decompressed), `chunk_size=4` keeps peak RAM under ~20GB
    versus ~60GB for unchunked.

    `scratch_dir` controls where the per-chunk temp parquets
    land during the recursive merge. **Defaults to `out.parent`**
    (the output's directory) so the temp scratch shares the same
    filesystem as the final output and can't outgrow a tiny
    overlay/`/tmp` mount. Pass an explicit path to override (e.g.
    a fast SSD scratch). Containerized runs where `/tmp` is on a
    small overlay used to silently fail the merge with
    `ENOSPC` — the chunk outputs (~chunk_size × compressed input
    size) easily exceed a few-GB overlay; the new default puts
    them next to the actual output where disk space was already
    provisioned for the result."""
    if not inputs:
        raise ValueError('stream_concat_parquets: no inputs')
    if chunk_size < 1:
        raise ValueError(f'chunk_size must be ≥ 1, got {chunk_size}')
    # **OOM mitigation**: adaptive chunk-size based on per-file
    # decompressed-size estimates. Static `chunk_size=4` blindly
    # holds 4× max(decomp_size) in RAM during the eager concat;
    # for 5GB-decomp trace shards that's 20GB peak. The adaptive
    # path reads parquet metadata to estimate decomp sizes and
    # lowers `chunk_size` so peak stays under `target_ram_gb`.
    chunk_size = _adaptive_chunk_size(
        inputs,
        requested_chunk_size=chunk_size,
        target_ram_gb=target_ram_gb,
    )
    if out.exists():
        out.unlink()
    how = 'diagonal_relaxed' if type_widening else 'diagonal'
    # `glob=False`: arm-tag relpaths embed `wrap[<wrapper>(<args>)]`
    # which polars otherwise treats as glob character classes,
    # producing "expanded paths were empty" on S3 URIs. Each input
    # is a single concrete path, never a glob — so disable globbing.
    inputs_list = [str(p) for p in inputs]
    # Atomicity (invariant I4 in SWEEP_PERSISTENCY.md): write to a
    # `<out>.partial` sibling and rename to `out` only after a
    # successful write. A crashed mid-write process leaves no
    # `.partial` file at the consumer's path; consumers never see
    # torn parquets. Atomic rename on POSIX; on Windows
    # `Path.replace` provides equivalent semantics.
    out_partial = out.with_suffix(out.suffix + '.partial')
    if out_partial.exists():
        out_partial.unlink()
    if len(inputs_list) <= chunk_size:
        # Small case: load+concat+write directly. No temp files.
        eager_frames = [
            pl.read_parquet(p, glob=False) for p in inputs_list
        ]
        merged = pl.concat(eager_frames, how=how)
        merged.write_parquet(
            str(out_partial),
            compression=compression, compression_level=compression_level,
        )
        out_partial.replace(out)
        return

    # Large case: chunked recursive merge. Pass 1 — write
    # `chunk_size`-sized batches to temp files; pass 2 — recursive
    # `stream_concat_parquets` on the temp files (smaller; usually
    # fits the chunk_size threshold in one more pass).
    import tempfile
    import shutil
    effective_scratch = scratch_dir if scratch_dir is not None else out.parent
    effective_scratch.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(
        prefix='stream_concat_', dir=str(effective_scratch),
    ))
    try:
        chunk_outs: list[Path] = []
        for i in range(0, len(inputs_list), chunk_size):
            batch = inputs_list[i:i + chunk_size]
            chunk_path = tmp_dir / f'chunk_{i // chunk_size:04d}.parquet'
            eager_frames = [pl.read_parquet(p, glob=False) for p in batch]
            merged = pl.concat(eager_frames, how=how)
            merged.write_parquet(
                str(chunk_path),
                compression=compression,
                compression_level=compression_level,
            )
            del merged, eager_frames
            chunk_outs.append(chunk_path)
        # Recurse on the chunk files. Same chunk_size — at most
        # log_chunk_size(N) levels of recursion (small for any
        # realistic N). Recursive calls inherit `scratch_dir`
        # (None means each level recomputes its own `out.parent`,
        # which is correct when chunks are inside `tmp_dir`).
        stream_concat_parquets(
            chunk_outs, out,
            type_widening=type_widening,
            compression=compression,
            compression_level=compression_level,
            chunk_size=chunk_size,
            scratch_dir=scratch_dir,
            target_ram_gb=target_ram_gb,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ============ ComputationGraph sidecar (provenance) ============

def _graph_to_spec(g: ComputationGraph) -> dict[str, object]:
    """Serialise a `ComputationGraph` to a JSON-friendly mapping.
    Edges encode as `(source, target, reader_arg, source_path)`
    tuples (the same shape as `signature(g)`'s edge tuples) — the
    minimal-fingerprint representation."""
    return {
        'nodes': sorted(g.nodes),
        'edges': sorted(
            [
                e.source,
                e.target,
                e.metadata.reader_arg,
                e.metadata.source_path,
            ]
            for e in g.edges
        ),
    }


def _spec_to_graph(spec: Mapping[str, object]) -> ComputationGraph:
    """Inverse of `_graph_to_spec`. Reconstructs a
    `ComputationGraph` from the JSON-decoded mapping."""
    g: ComputationGraph = Graph()
    nodes = spec.get('nodes', [])
    if not isinstance(nodes, list):
        raise TypeError(
            f'graph spec `nodes` must be a list of strings, '
            f'got {type(nodes).__name__}',
        )
    for n in nodes:
        if not isinstance(n, str):
            raise TypeError(
                f'graph spec node must be str, got {type(n).__name__}',
            )
        g = g.with_node(n)
    edges = spec.get('edges', [])
    if not isinstance(edges, list):
        raise TypeError(
            f'graph spec `edges` must be a list, got {type(edges).__name__}',
        )
    for e in edges:
        if (
            not isinstance(e, list) or len(e) != 4
            or not all(isinstance(x, str) for x in e)
        ):
            raise TypeError(
                f'graph spec edge must be [source, target, reader_arg, '
                f'source_path] of strings, got {e!r}',
            )
        source, target, reader_arg, source_path = e
        g = g.with_edge(
            source, target,
            ComputationEdge(
                reader_arg=reader_arg,
                source_path=source_path,
            ),
        )
    return g


def write_graphs_sidecar(
    graphs: Mapping[str, ComputationGraph], path: Path,
) -> None:
    """Persist `graphs` keyed by `Hypothesis.arm_key()` (or any
    substrate-chosen identifier) as a JSON sidecar at `path`. Each
    entry stores the static call topology — the `@claim` invocation
    edges captured during the sweep's first abstract-trace pass.

    Round-trip pair: `read_graphs_sidecar(path)`. Use this once
    per corpus alongside `runs.parquet` / `traces.parquet` so
    post-hoc consumers can recover topology without re-running
    the sweep."""
    spec = {arm_key: _graph_to_spec(g) for arm_key, g in graphs.items()}
    path.write_text(json.dumps(spec, indent=2, sort_keys=True))


def read_graphs_sidecar(path: Path) -> dict[str, ComputationGraph]:
    """Inverse of `write_graphs_sidecar`. Returns an empty dict
    when the sidecar is absent — substrates with no graph capture
    omit it transparently."""
    if not path.exists():
        return {}
    raw = _json_loads(path.read_text())
    if not is_mapping_str_object(raw):
        raise TypeError(
            f'graphs sidecar at {path} must be a JSON object, '
            f'got {type(raw).__name__}',
        )
    out: dict[str, ComputationGraph] = {}
    for arm_key, spec in raw.items():
        if not is_mapping_str_object(spec):
            raise TypeError(
                f'graphs sidecar value for {arm_key!r} must be an '
                f'object, got {type(spec).__name__}',
            )
        out[arm_key] = _spec_to_graph(spec)
    return out
