"""Per-corpus measurement store — the Phase 1 layer from
CACHE_BUILD.md.

A `<corpus_dir>/measurements.parquet` file holds every measurable
ever computed for that corpus, keyed by `RunRow.id`. The matching
`<corpus_dir>/measurements.hashes.json` sidecar records the
closure hash per column so drift detection works the same way
as the per-hypothesis cache layer.

The two-level architecture this enables:
  - Per-hypothesis cache becomes a cheap **projection** over
    in-scope corpora's `measurements.parquet`s + a column subset.
  - Multiple hypotheses sharing corpora + measurables compute
    each measurable ONCE (in this layer), not once per hypothesis.

Phase 1 wires the new layer alongside the existing
`_load_one_corpus` path: the runner consults
`measurements.parquet` and skips the in-loop measurable
computation when all required columns are present + current.
Phase 2 (separate commit) gut-renovates `_ingest_and_compute` to
be a pure projection over per-corpus measurement stores.

API:
  - `build_measurements(corpus_dir, *, required, runs_df, traces_path)
    -> Path`: compute missing/drifted measurables, write
    measurements.parquet atomically. Idempotent.
  - `load_measurements(corpus_dir, *, columns) -> DataFrame`:
    pure read.
  - `current_signatures(corpus_dir) -> dict[str, str]`:
    return the closure-hash sidecar contents (empty dict if
    absent).
"""
from __future__ import annotations

import json
import sys
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Literal
from pathlib import Path

import polars as pl

if TYPE_CHECKING:
    import pyarrow.parquet

from corroborate._internals.json import loads as _json_loads
from corroborate._internals.narrow import is_mapping_str_object
from corroborate.corpus.persistence import (
    atomic_write_parquet,
    atomic_write_text,
)
from corroborate.measurables import (
    compute_missing_columns,
    get_registered,
    registered_names,
)


MEASUREMENTS_FILENAME = 'measurements.parquet'
DEFAULT_TRACE_BATCH_SIZE = 100
"""Default cells-per-batch for `compute_trace_measurables_streaming`.
Tunes peak RAM during measurable computation: higher = fewer
row-group reads but more cells × trace col size held at once.
100 keeps peak under ~1 GB even for 1M-step MinAtar traces with
6 per-step list cols (~10 MB per cell × 100 = 1 GB)."""
DEFAULT_TRACE_BYTE_BUDGET = 512 * 1024 ** 2
"""Default decompressed-bytes budget per streaming batch in
`compute_trace_measurables_streaming`. The load-bearing RAM bound
for heavy per-step trace columns (the cell-count `batch_size` is a
coarse secondary cap). A batch's summed per-column uncompressed
size stays under this; a single row group already over budget
routes to the per-cell lazy-scan fallback.

512 MiB decompressed × the polars/arrow inflation that list-typed
per-step trace columns incur (~10×) keeps peak RAM in the low
single-digit GB range. For 3M-step MinAtar traces re-chunked to
small row groups (snake_g099_canonical_3M_ckpt: ~183 MB/cell, 2
cells/group ≈ 365 MB), this yields one row group (~2 cells) per
batch. Lighter 1M-step traces (~10 MB/cell) pack many cells per
batch. A single row group already over budget routes to the
per-cell lazy-scan fallback. Comfortable on the 12-16 GB envelopes
the framework runs in."""
SIDECAR_FILENAME = 'measurements.hashes.json'


def _measurements_path(corpus_dir: Path) -> Path:
    return corpus_dir / MEASUREMENTS_FILENAME


def _sidecar_path(corpus_dir: Path) -> Path:
    return corpus_dir / SIDECAR_FILENAME


def current_signatures(corpus_dir: Path) -> dict[str, str]:
    """Read the closure-hash sidecar; return empty dict when
    absent or unparseable. Returns a fresh `dict` (not `Mapping`)
    so the caller can mutate without copying — the function
    builds a new dict each call regardless."""
    path = _sidecar_path(corpus_dir)
    if not path.exists():
        return {}
    try:
        raw = _json_loads(path.read_text())
    except (OSError, ValueError):
        return {}
    if not is_mapping_str_object(raw):
        return {}
    out: dict[str, str] = {}
    for k, v in raw.items():
        if isinstance(v, str):
            out[k] = v
    return out


def load_measurements(
    corpus_dir: Path,
    *,
    columns: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Pure read of the per-corpus measurement store. Returns
    an empty DataFrame when the file doesn't exist.

    `columns`, when given, is the projection to read — polars'
    column-projection pushdown means only the requested columns'
    pages are decompressed (cheap on the typical wide-but-sparse
    measurements table). `id` is always included.
    """
    path = _measurements_path(corpus_dir)
    if not path.exists():
        return pl.DataFrame()
    if columns is None:
        return pl.read_parquet(path)
    cols = ['id'] + [c for c in columns if c != 'id']
    # `pl.read_parquet` raises if a requested column is absent;
    # narrow to existing columns first.
    schema = pl.scan_parquet(path).collect_schema()
    available = set(schema.names())
    cols_present = [c for c in cols if c in available]
    if not cols_present:
        return pl.DataFrame()
    return pl.read_parquet(path, columns=cols_present)


def build_measurements(
    corpus_dir: Path,
    *,
    required: Sequence[str],
    runs_df: pl.DataFrame,
    traces_path: Path | None = None,
    measurable_signature_fn: Callable[[str], str | None] | None = None,
    force: frozenset[str] = frozenset(),
) -> Path:
    """Compute missing + drifted measurables for `corpus_dir`,
    write `measurements.parquet` atomically, update sidecar.

    `runs_df` is the cell-level DataFrame from `runs.parquet`
    (PLUS any joined trace columns the caller already attached).
    Required measurables that read trace columns expect those
    columns to be on `runs_df` already — the caller is
    responsible for the join (the runner does this via
    `_join_required_traces`).

    `traces_path` is informational only (logged in progress
    output); the actual trace columns must be on `runs_df`.

    `measurable_signature_fn` is the closure-hash function. The
    runner's `_measurable_signature` is the canonical
    implementation; we accept it as a parameter so callers can
    inject a stub in tests. Defaults to looking up the
    measurable in the registry and calling `.signature()`.

    Idempotency: if every required measurable is already in
    `measurements.parquet` with a matching closure hash, this is
    a no-op (no parquet rewrite). Drift detection drops drifted
    columns + recomputes them.

    `force`: names recomputed unconditionally — dropped from BOTH
    the existing store AND any matching `runs_df` stamp so they
    rebuild from the caller-joined reads even when their closure
    hash is sidecar-current (the input traces changed but the
    formula didn't — e.g. a re-eval at a new n_episodes). Dropping
    only the existing store is insufficient: a measurable the
    substrate stamped into `runs.parquet` survives on `runs_df` and
    `compute_missing_columns` skips it (present-and-non-null), so
    the stale stamp wins. The caller must guarantee the forced
    names' reads are on `runs_df`, else the rebuild null-pads.

    Returns the path to `measurements.parquet`.
    """
    if measurable_signature_fn is None:
        def _default_sig(name: str) -> str | None:
            m = get_registered(name)
            return None if m is None else m.signature()
        sig_fn: Callable[[str], str | None] = _default_sig
    else:
        sig_fn = measurable_signature_fn

    out_path = _measurements_path(corpus_dir)
    sidecar_path = _sidecar_path(corpus_dir)

    existing = load_measurements(corpus_dir)
    if 'id' not in runs_df.columns:
        raise ValueError(
            f'build_measurements({corpus_dir}): runs_df is missing '
            f'the `id` column — required as the per-cell key',
        )
    # Force-recompute, runs_df side. The existing-store force-drop
    # (drift loop below) handles `measurements.parquet`; this handles
    # a forced measurable the substrate STAMPED into `runs.parquet`
    # (`RunRow.measurements`). Such a stamp lands on `runs_df` →
    # `joined`, and `compute_missing_columns` SKIPS any column already
    # present-and-non-null in its input frame. So without this drop a
    # stale runs.parquet scalar (e.g. a re-eval at a new n_episodes:
    # the trace `mc_return` changed but the old `eval_best_burst_mean`
    # stamp didn't) silently wins over the caller-joined fresh trace
    # reads — `force` would be a no-op exactly as it was for the
    # existing-store side before this. Forced names are registered
    # measurables; their own reads (trace cols / config leaves) are
    # distinct column names and remain on `runs_df`, so the rebuild
    # recomputes from the fresh reads rather than null-padding.
    if force:
        runs_force_stamps = [
            c for c in runs_df.columns if c != 'id' and c in force
        ]
        if runs_force_stamps:
            runs_df = runs_df.drop(runs_force_stamps)
    # Defend against id-duplicate corruption: each `id` should
    # appear exactly once in the per-corpus store. A stale store
    # with duplicate ids (legacy from pre-Phase-1 cache builds, or
    # accumulated across multiple sweep merges without dedup)
    # causes the runs_df → existing left-join below to Cartesian-
    # multiply: with K duplicates per id, runs_df.height × K rows
    # come out. The next `compute_missing_columns` + write doubles
    # again on subsequent rebuilds. Detect + rebuild from scratch
    # when corruption is observed; the per-cell measurable
    # computation is the same shape, just the join input is now
    # sane.
    if (
        existing.height > 0
        and existing.height != existing['id'].n_unique()
    ):
        sys.stderr.write(
            f'measurements: WARNING — {out_path} carries '
            f'{existing.height} rows but only '
            f'{existing["id"].n_unique()} unique ids; rebuilding '
            f'from scratch.\n',
        )
        existing = pl.DataFrame()
    # **corpus-integrity invariant CI6**: row-level orphan eviction.
    # `existing` may contain rows whose `id` is no longer in
    # `runs_df['id']` — sweep extensions or partial reruns that
    # dropped cells from runs.parquet leave stale orphans in
    # measurements.parquet. Without this drop, every subsequent
    # rebuild keeps recomputing measurables for cells that no
    # longer exist scientifically. Reconcile to a strict subset.
    orphan_rows_dropped = False
    if existing.height > 0:
        runs_ids = set(runs_df['id'].to_list())
        existing_ids = set(existing['id'].to_list())
        orphan_ids = existing_ids - runs_ids
        if orphan_ids:
            sys.stderr.write(
                f'measurements: dropping {len(orphan_ids)} orphan '
                f'row(s) from {out_path} (id no longer in '
                f'runs.parquet)\n',
            )
            existing = existing.filter(pl.col('id').is_in(list(runs_ids)))
            orphan_rows_dropped = True
    stored_sigs = current_signatures(corpus_dir)

    # Drop drifted + orphan columns from the existing store. Same
    # two-axis logic as the runner's `_invalidate_drifted` (C4).
    drop_cols: list[str] = []
    if existing.height > 0:
        all_registered = set(registered_names())
        required_set = set(required)
        for col in existing.columns:
            if col == 'id':
                continue
            if col not in all_registered:
                # Non-measurable column shouldn't be in this store
                # at all — drop defensively.
                drop_cols.append(col)
                continue
            if col not in required_set:
                # Orphan: no longer required.
                drop_cols.append(col)
                continue
            if col in force:
                # Operator-forced recompute: drop so the value is
                # rebuilt from the (caller-joined) trace columns,
                # bypassing the sidecar-current check. Without this the
                # sidecar-current branch below keeps the stale value
                # even though the caller wants it recomputed (e.g. the
                # source traces changed but the measurable's CLOSURE
                # hash didn't — re-eval at a new n_episodes). The caller
                # (`recompute_corpus_measurables`) only forces names
                # whose reads it has verified are present, so the
                # rebuild can't null-pad.
                drop_cols.append(col)
                continue
            current_hash = sig_fn(col)
            stored = stored_sigs.get(col)
            if current_hash is None:
                # Anomalous: column passed `col in all_registered`
                # but the signature fn returned None. Either the
                # registry was mutated mid-loop (race) or the
                # measurable's `signature()` returned None (which
                # should never happen for the canonical
                # `_default_sig` — only via injected sig fns in
                # tests). Drop the column rather than silently
                # keep it: drift coverage requires a real current
                # hash, and the conservative move is to recompute.
                # Post-roast-#4 fix.
                if stored is not None:
                    drop_cols.append(col)
                continue
            if stored is not None and stored != current_hash:
                drop_cols.append(col)
        if drop_cols:
            existing = existing.drop(drop_cols)
            for c in drop_cols:
                stored_sigs.pop(c, None)

    # Compute. Pass the runs_df + (optional already-existing
    # measurements joined on id) to compute_missing_columns,
    # which:
    #   - Fills entirely-missing columns for every cell.
    #   - **Partial-nullity recompute**: for columns that ARE
    #     present in `joined` but contain any null cells,
    #     recomputes ONLY the null cells (`measurable.py`'s
    #     `existing_values` per-pending entry). Non-null cells
    #     pass through unchanged.
    #   - Returns `joined` unchanged when no pending work
    #     (every required column present + no nulls).
    #
    # Importantly we pass the FULL `required` list (not a
    # subset of "missing names") so the partial-nullity branch
    # fires on present-but-null columns. A naive
    # `[n for n in required if n not in existing.columns]`
    # filter would silently skip the recompute for nulls.
    to_compute_full = [
        n for n in required if get_registered(n) is not None
    ]
    overlap_dropped: list[str] = []
    if existing.height > 0:
        # Collision dedup (post-#1 roast fix): when `runs_df` ALREADY
        # carries a measurable column whose name matches one in
        # `existing` (Phase 3 substrate-side stamp via
        # `RunRow.measurements`), polars' default left-join would
        # produce a `<col>_right` suffix on the existing-store
        # version. The right-suffixed column is then orphaned at
        # `select(measurable_cols)` — the existing-store values
        # are silently dropped without any merge logic consulting
        # them.
        #
        # Substrate-stamped values are authoritative per Phase 3
        # **only when the substrate had the inputs to compute
        # them**. Trace-dependent measurables (`reads` references
        # per-step trace columns like `online_max_q_per_step`) are
        # stamped NaN at sweep time because traces don't exist yet
        # — runner adds them as a post-sweep reduction. For those,
        # runs_df's NaN stamp is NOT authoritative; the prior
        # per-corpus store's value (computed once traces were
        # restored) is. Drop only when the substrate had all
        # inputs — preserve existing for trace-dependent stamps
        # the substrate couldn't have computed.
        #
        # `unregistered_policy='runs_wins'`: post-CI6 orphan
        # eviction guarantees existing.columns are registered;
        # this branch is the defensive fallback for the rare
        # not-yet-evicted-orphan case. The runner has the
        # registry import.
        runs_cols = set(runs_df.columns)
        existing_meas_cols = set(existing.columns)
        # `_drop_from_runs` is calculated for completeness but
        # unused on this code path: build_measurements operates
        # by dropping the EXISTING-side cols + then the
        # complement is dropped from runs_df further down (the
        # `runs_dropped = [c for c in existing.columns if ...]`
        # construction post-overlap). Helper still returns both
        # for the Panel.from_corpus path.
        _drop_from_runs_unused, overlap_dropped_set = (
            resolve_runs_meas_collision(
                runs_cols=runs_cols,
                meas_cols=existing_meas_cols,
                unregistered_policy='runs_wins',
            )
        )
        del _drop_from_runs_unused
        overlap_dropped = list(overlap_dropped_set)
        if overlap_dropped:
            existing = existing.drop(overlap_dropped)
        # Where existing wins (kept above), runs_df's NaN-stamped
        # column is itself the collision target — drop from
        # runs_df so the join doesn't produce `<col>_right`. The
        # existing-store value flows through unsuffixed.
        runs_dropped = [
            c for c in existing.columns
            if c != 'id' and c in runs_df.columns
        ]
        if runs_dropped:
            runs_df = runs_df.drop(runs_dropped)
        # Bring forward still-current columns by joining on id. The
        # runs_df may have more rows than existing (new cells); left-
        # join keeps all runs cells. Cells in runs_df not yet in
        # existing carry NULL for measurable cols —
        # `compute_missing_columns`'s partial-nullity branch then
        # fills them while preserving existing non-null values.
        joined = runs_df.join(existing, on='id', how='left')
    else:
        joined = runs_df

    # Idempotent skip: if nothing drifted, no orphans, no new
    # cells, AND every required column is fully populated (no
    # partial nulls), then `compute_missing_columns` would be a
    # no-op — skip the rewrite. Computing this BEFORE the call
    # means we don't pay for `to_dicts()` on the no-work path.
    #
    # Post-roast-#2 fix: also require ID-set membership. Pre-fix,
    # a caller passing a runs_df with disjoint IDs but identical
    # row count and pre-populated measurable cols would skip the
    # rebuild — the persisted store would retain old cells' values
    # keyed under the OLD IDs. Production paths (Phase 2.1's
    # `_load_one_corpus`) always pass the corpus's own runs_df, so
    # this didn't fire — but the contract should explicitly reject
    # the disjoint-ID case rather than silently corrupt the store.
    def _has_missing_values(col: pl.Series) -> bool:
        """True iff `col` has null OR NaN entries. Mirrors the
        float-dtype-aware missing-mask used by `compute_missing_
        columns` itself; pre-fix the gate used only `is_null()`
        which silently passed all-NaN float columns through the
        idempotent-skip path, leaving stale-NaN measurables
        un-recomputed even when local inputs satisfy their reads
        (today's F3+F4 gotcha)."""
        if col.is_null().any():
            return True
        return col.dtype.is_float() and bool(col.is_nan().any())

    no_partial_nulls = all(
        n not in joined.columns
        or not _has_missing_values(joined[n])
        for n in to_compute_full
    )
    all_required_present = all(
        n in joined.columns for n in to_compute_full
    )
    ids_match = (
        existing.height > 0
        and existing.height == runs_df.height
        and set(existing['id'].to_list()) == set(runs_df['id'].to_list())
    )
    if (
        not drop_cols
        and not overlap_dropped
        and not orphan_rows_dropped
        and ids_match
        and all_required_present
        and no_partial_nulls
    ):
        return out_path

    # Skip-recompute for unsatisfiable measurables. A measurable
    # whose **transitive** record-key reads aren't all in
    # `joined.columns` will silently return NaN per cell (the
    # measurable itself NaN-propagates when its shared-source
    # `from_key` invocation raises `KeyError`), and
    # `compute_missing_columns`'s partial-null branch will then
    # OVERWRITE existing finite values in `joined` with the fresh
    # NaN — turning a previously-good per-corpus store into stale
    # NaN.
    #
    # `m.reads` alone isn't enough: a measurable like `effective_
    # horizon` declares `reads=('gamma',)` but takes a parameter-
    # injected `bootstrap_fraction` whose own `reads=('done',)`
    # depends on per-step trace data. `transitive_reads(name)`
    # walks the dependency closure to surface the true leaf-key
    # set the cell record needs to carry.
    from corroborate.measurables.measurable import transitive_reads
    to_compute_satisfied: list[str] = []
    for n in to_compute_full:
        m = get_registered(n)
        if m is None:
            continue
        leaf_reads = transitive_reads(n)
        if all(r in joined.columns for r in leaf_reads):
            to_compute_satisfied.append(n)
    enriched = compute_missing_columns(joined, to_compute_satisfied)

    # Project to id + measurable columns only (drop any joined
    # trace cols / raw record fields the caller passed in).
    #
    # **Stale-NaN sentinel guard**: drop registered measurables
    # that (a) we did NOT recompute this round (not in
    # `to_compute_satisfied`) AND (b) carry only null/NaN values.
    # These typically arrive on `runs_df` as sweep-time NaN
    # stamps (the substrate stamped them but couldn't compute —
    # injected dep missing at sweep time). Preserving them here
    # would persist a registered column + closure-hash for a
    # value the framework never actually computed, locking the
    # corpus into a permanent stuck-NaN that even
    # `--force-recompute` won't fix (the sidecar hash makes it
    # look "current"). Drop them so a future ingest with
    # restored traces can re-stamp them honestly via
    # `compute_missing_columns`' partial-nullity branch.
    to_compute_satisfied_set = set(to_compute_satisfied)
    registered = registered_names()
    measurable_cols: list[str] = []
    dropped_stale_nan: list[str] = []
    for c in enriched.columns:
        if c == 'id':
            measurable_cols.append(c)
            continue
        if c not in registered:
            continue
        if c in to_compute_satisfied_set:
            measurable_cols.append(c)
            continue
        # Registered + present + not recomputed this round.
        # Keep only if it carries any non-missing value (a
        # legitimate sweep-time stamp). Drop if all-missing —
        # that's the unsatisfiable-stamped-NaN case.
        if _has_missing_values(enriched[c]):
            col_data = enriched[c]
            n_null = int(col_data.is_null().sum() or 0)
            n_nan = (
                int(col_data.is_nan().sum() or 0)
                if col_data.dtype.is_float() else 0
            )
            if (n_null + n_nan) == enriched.height:
                dropped_stale_nan.append(c)
                continue
        measurable_cols.append(c)
    if dropped_stale_nan:
        sys.stderr.write(
            f'measurements: dropped {len(dropped_stale_nan)} stale-NaN '
            f'registered column(s) from {out_path} '
            f'(sweep-time stamps with no live recompute path): '
            f'{", ".join(sorted(dropped_stale_nan))}\n',
        )
    out_df = enriched.select(measurable_cols)

    # Skip pointless writes (post-roast-#3 fix): when no measurable
    # cols are computed AND no existing store needs to be updated,
    # writing an id-only parquet + empty sidecar wastes I/O and
    # creates artifacts the framework didn't need. The first build
    # with `required=()` against an empty corpus dir hits this.
    if len(measurable_cols) == 1 and not out_path.exists():
        return out_path

    # Sidecar: closure hashes for every column actually present.
    new_sigs: dict[str, str] = {}
    for col in out_df.columns:
        if col == 'id':
            continue
        sig = sig_fn(col)
        if sig is not None:
            new_sigs[col] = sig

    atomic_write_parquet(out_df, out_path)
    atomic_write_text(
        sidecar_path,
        json.dumps(new_sigs, indent=2, sort_keys=True),
    )
    n_measurable_cols = len(out_df.columns) - 1
    sys.stderr.write(
        f'measurements: wrote {out_df.height} cells × '
        f'{n_measurable_cols} measurable col'
        f'{"s" if n_measurable_cols != 1 else ""} to {out_path}\n',
    )
    return out_path


# ============ CACHE_ADDITIVITY.md Phase 2: --check mode ============


from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CorpusDriftReport:
    """Per-corpus drift summary: which required columns are
    missing from the per-corpus sidecar (never computed) vs.
    drifted (stored hash differs from current registry hash).
    A corpus is `is_clean` iff both lists are empty."""
    corpus_dir: Path
    drifted: tuple[str, ...]
    missing: tuple[str, ...]

    @property
    def is_clean(self) -> bool:
        return not self.drifted and not self.missing


@dataclass(frozen=True, slots=True)
class DriftReport:
    """**CACHE_ADDITIVITY.md CA5** report shape. Aggregates per-
    corpus drift / missing across the data root. Pure read, no
    compute, no walk of `runs.parquet`."""
    per_corpus: tuple[CorpusDriftReport, ...]

    @property
    def is_clean(self) -> bool:
        return all(c.is_clean for c in self.per_corpus)

    @property
    def n_corpora_drifted(self) -> int:
        return sum(1 for c in self.per_corpus if c.drifted)

    @property
    def n_corpora_missing(self) -> int:
        return sum(1 for c in self.per_corpus if c.missing)

    def affected_corpus_names(self) -> tuple[str, ...]:
        """Names of corpora that need refreshing (any drift OR
        missing column). Suitable for direct splat into
        `--ingest <names>`."""
        return tuple(
            c.corpus_dir.name
            for c in self.per_corpus if not c.is_clean
        )


def _row_group_uncompressed_bytes(
    pq_file: 'pyarrow.parquet.ParquetFile',
    rg_index: int,
    columns: Sequence[str],
) -> int:
    """Summed decompressed byte size of `columns` within row group
    `rg_index`, from parquet column-chunk metadata (no data read).

    Drives size-aware batching in `compute_trace_measurables_
    streaming`: per-step trace columns dominate the footprint, so
    budgeting batches by their decompressed bytes is the load-
    bearing RAM bound. Columns absent from the schema contribute 0.
    Falls back to the row group's `total_byte_size` when per-column
    metadata is unavailable (older parquet writers)."""
    rg = pq_file.metadata.row_group(rg_index)
    wanted = set(columns)
    total = 0
    saw_any = False
    for c in range(rg.num_columns):
        col = rg.column(c)
        # `path_in_schema` is the dotted leaf path; the top-level
        # column name is its first segment (list / struct children
        # share the parent's name, so all chunks of a list column
        # accumulate under the same wanted-name).
        name = col.path_in_schema.split('.', 1)[0]
        if name in wanted:
            saw_any = True
            total += int(col.total_uncompressed_size)
    if not saw_any:
        # No matching column chunks (unexpected) — be conservative
        # and report the whole row group's size so the batcher
        # doesn't under-budget.
        return int(rg.total_byte_size)
    return total


def compute_trace_measurables_streaming(
    runs_df: pl.DataFrame,
    traces_path: Path,
    *,
    measurable_reads: frozenset[str],
    required: Sequence[str],
    batch_size: int = DEFAULT_TRACE_BATCH_SIZE,
    byte_budget: int = DEFAULT_TRACE_BYTE_BUDGET,
) -> pl.DataFrame:
    """Row-group-streaming computation of trace-dependent
    measurables. Returns a small DataFrame with `id` + measurable
    scalar columns only — never holds full trace cols in memory.

    For each row group of `traces_path`:
      1. Read the cols in `measurable_reads` (intersected with
         what the trace actually carries) plus `id` for that
         row group only.
      2. Inner-join with `runs_df` rows whose ids are in the
         row group.
      3. Run `compute_missing_columns(joined, required)` on the
         small batch.
      4. Project to `id` + registered measurable cols.
      5. Concat into the scalar accumulator.

    Peak RAM = `batch_size` cells' trace cols ≈ batch_size × max
    decompressed col size. For 1M-step MinAtar traces with 6
    per-step list cols (~10 MB / cell), batch=100 → ~1 GB peak.

    The accumulator is small: only id + measurable scalars,
    grows linearly with cell count but ~O(KB per cell), well
    within RAM for any realistic corpus size.

    No-op when the trace file is absent or carries none of the
    requested cols — returns a `(0, 1)`-shaped frame with just
    an empty `id` column."""
    import pyarrow.parquet as _pq

    if not traces_path.exists():
        return pl.DataFrame({'id': []}, schema={'id': pl.Utf8})
    try:
        schema = pl.scan_parquet(traces_path).collect_schema()
    except pl.exceptions.ComputeError:
        return pl.DataFrame({'id': []}, schema={'id': pl.Utf8})
    available = set(schema.names())
    cols_to_load = ['id'] + sorted(measurable_reads & available)
    if len(cols_to_load) == 1:
        return pl.DataFrame({'id': []}, schema={'id': pl.Utf8})
    if 'id' not in runs_df.columns:
        raise ValueError(
            f'compute_trace_measurables_streaming({traces_path}): '
            f'runs_df is missing the `id` column',
        )
    # Drop any `required` measurable the substrate STAMPED into
    # runs.parquet (RunRow.measurements). The per-batch
    # `compute_missing_columns` below SKIPS a column already
    # present-and-non-null in its input frame, so a stale stamp
    # (e.g. an eval-derived measurable carried forward from a
    # re-eval at an OLD n_episodes) would shadow the fresh
    # trace-based recompute this function exists to perform. This
    # function's contract is "compute `required` FROM
    # `traces_path`" — the runs.parquet stamp is never
    # authoritative here. Mirrors the `build_measurements` force
    # path's runs_df drop; covers both the row-group loop and the
    # `_compute_trace_measurables_per_id` fallback (both consume
    # `runs_indexed`).
    runs_indexed = runs_df
    _required_stamped = [
        c for c in runs_indexed.columns if c != 'id' and c in set(required)
    ]
    if _required_stamped:
        runs_indexed = runs_indexed.drop(_required_stamped)
    pq_file = _pq.ParquetFile(str(traces_path))
    measurable_set = set(registered_names())
    n_per_rg = max(1, batch_size)
    n_row_groups = pq_file.num_row_groups
    if n_row_groups == 0:
        return pl.DataFrame({'id': []}, schema={'id': pl.Utf8})

    # **Size-aware batching (row-group-OOM root cause fix).** Row
    # groups are the framework's nominal streaming unit, but two
    # writer pathologies defeat naive count-based batching:
    #
    #   (1) A cloud sweep merger may write a SINGLE huge row group
    #       spanning every cell (snake_g099_canonical_3M_ckpt: 60
    #       cells in 1 RG ~= 12 GB uncompressed). Reading "by row
    #       group" then loads everything at once -> OOM.
    #   (2) Even after re-chunking into small row groups, a
    #       count-based batcher with `batch_size` >= n_cells
    #       recombines them into one batch -> same OOM.
    #
    # The fix budgets each batch by DECOMPRESSED BYTES of the
    # columns we actually load (not cell count): sum each row
    # group's per-column `total_uncompressed_size` for `cols_to_
    # load`. A batch never exceeds `byte_budget`; a single row
    # group already over budget routes to the per-cell lazy-scan
    # fallback (peak RAM bounded by ONE cell's trace footprint).
    rg_load_bytes = [
        _row_group_uncompressed_bytes(pq_file, i, cols_to_load)
        for i in range(n_row_groups)
    ]
    rg_rows_list = [
        pq_file.metadata.row_group(i).num_rows
        for i in range(n_row_groups)
    ]
    max_rg_rows = max(rg_rows_list)
    max_rg_bytes = max(rg_load_bytes)
    # Per-cell estimate from the heaviest row group (guards the
    # single-huge-RG case where per-RG bytes >> budget but per-cell
    # is fine — the per-cell scan then bounds RAM at one cell).
    per_cell_bytes_est = max(
        (b / max(1, r)) for b, r in zip(rg_load_bytes, rg_rows_list)
    )
    # Route to per-cell scan when EITHER a row group exceeds the
    # byte budget (its rows can't be read as one block) OR the
    # writer used rows-per-group exceeding the cell-count cap (the
    # single-huge-RG layout) AND that group is too big to read
    # whole. The per-cell scan reads one cell at a time, so its
    # peak is `per_cell_bytes_est` × the polars/arrow inflation.
    use_per_id_scan = max_rg_bytes > byte_budget or max_rg_rows > n_per_rg
    if use_per_id_scan:
        return _compute_trace_measurables_per_id(
            runs_indexed,
            traces_path,
            cols_to_load=cols_to_load,
            required=required,
            measurable_set=measurable_set,
        )
    del per_cell_bytes_est
    accumulators: list[pl.DataFrame] = []
    rg_indices = list(range(n_row_groups))
    # Group consecutive row-groups so each batch stays under BOTH
    # the cell-count cap (`n_per_rg`) and the decompressed-byte
    # budget. Byte budget is the load-bearing bound for heavy
    # per-step trace columns; the cell cap is a coarse secondary
    # limit retained for backward compatibility with small traces.
    rg_groups: list[list[int]] = []
    cur: list[int] = []
    cur_rows = 0
    cur_bytes = 0
    for idx in rg_indices:
        rg_rows = rg_rows_list[idx]
        rg_bytes = rg_load_bytes[idx]
        over_rows = cur and cur_rows + rg_rows > n_per_rg
        over_bytes = cur and cur_bytes + rg_bytes > byte_budget
        if over_rows or over_bytes:
            rg_groups.append(cur)
            cur = [idx]
            cur_rows = rg_rows
            cur_bytes = rg_bytes
        else:
            cur.append(idx)
            cur_rows += rg_rows
            cur_bytes += rg_bytes
    if cur:
        rg_groups.append(cur)
    for group in rg_groups:
        # `read_row_groups(...)` reads multiple consecutive row
        # groups in one pyarrow call, more efficient than per-rg
        # reads when the writer used many small row groups.
        table = pq_file.read_row_groups(group, columns=cols_to_load)
        batch_traces = pl.from_arrow(table)
        if not isinstance(batch_traces, pl.DataFrame):
            # `from_arrow` can return a Series for 1-col inputs;
            # we have at least 'id' + N cols so this shouldn't
            # fire, but guard defensively.
            continue
        batch_runs = runs_indexed.filter(
            pl.col('id').is_in(batch_traces['id'].to_list()),
        )
        if batch_runs.height == 0:
            continue
        # Drop trace col duplicates from runs_df (e.g. `id`-only
        # overlap is fine; substantive overlap means runs.parquet
        # carried a trace col which polars' join would suffix).
        overlap = [
            c for c in batch_traces.columns
            if c != 'id' and c in batch_runs.columns
        ]
        if overlap:
            batch_traces = batch_traces.drop(overlap)
        joined = batch_runs.join(batch_traces, on='id', how='left')
        enriched = compute_missing_columns(joined, list(required))
        # Project to id + measurable cols only (drops trace cols
        # before accumulation).
        keep = ['id'] + [
            c for c in enriched.columns
            if c != 'id' and c in measurable_set
        ]
        accumulators.append(enriched.select(keep))
        del table, batch_traces, joined, enriched
    if not accumulators:
        return pl.DataFrame({'id': []}, schema={'id': pl.Utf8})
    return pl.concat(accumulators, how='diagonal_relaxed')


def _compute_trace_measurables_per_id(
    runs_df: pl.DataFrame,
    traces_path: Path,
    *,
    cols_to_load: Sequence[str],
    required: Sequence[str],
    measurable_set: set[str],
) -> pl.DataFrame:
    """Per-cell lazy-scan fallback for `compute_trace_measurables_
    streaming` when row groups are unusable as a streaming axis
    (single huge RG = the cloud-write OOM case). Drives the scan
    by `runs_df['id']` order; each iteration holds one cell's
    trace columns in scope. Peak RAM = one cell × col size.

    Uses polars' lazy-scan + filter predicate-pushdown so the
    parquet reader only materialises the matching row(s). The
    predicate is an equality on `id`; modern parquet writers
    record per-row-group min/max so a sorted `id` column reads
    only the relevant RG — but even an unsorted file degrades to
    "scan all RGs, return one row" which is bounded by the
    LARGEST RG's footprint per cell, NOT the whole file.

    For traces written as one giant row group, the predicate
    can't skip pages but polars' streaming engine still emits
    row-at-a-time without materialising the entire column up
    front (the streaming sink contract). Concretely: a 1-RG / 60-
    cell / 30 GB trace file reads in ~5 s per cell via this path,
    peaking ~500 MB instead of OOMing.
    """
    if 'id' not in runs_df.columns:
        return pl.DataFrame({'id': []}, schema={'id': pl.Utf8})
    accumulators: list[pl.DataFrame] = []
    keep_cols = list(cols_to_load)
    runs_cols = set(runs_df.columns)
    overlap = [
        c for c in keep_cols
        if c != 'id' and c in runs_cols
    ]
    cell_ids: list[object] = runs_df['id'].to_list()
    for cid in cell_ids:
        if not isinstance(cid, str):
            continue
        # Lazy scan + filter pushdown: polars only materialises
        # rows matching the predicate. For a single-RG file the
        # predicate can't skip the RG metadata, but column
        # decoding short-circuits per-page when the row index
        # falls outside the matching set.
        try:
            cell_traces = (
                pl.scan_parquet(traces_path)
                .select(keep_cols)
                .filter(pl.col('id') == cid)
                .collect()
            )
        except pl.exceptions.ComputeError:
            continue
        if cell_traces.height == 0:
            continue
        cell_runs = runs_df.filter(pl.col('id') == cid)
        if cell_runs.height == 0:
            continue
        if overlap:
            cell_traces = cell_traces.drop(overlap)
        joined = cell_runs.join(cell_traces, on='id', how='left')
        enriched = compute_missing_columns(joined, list(required))
        keep = ['id'] + [
            c for c in enriched.columns
            if c != 'id' and c in measurable_set
        ]
        accumulators.append(enriched.select(keep))
        del cell_traces, cell_runs, joined, enriched
    if not accumulators:
        return pl.DataFrame({'id': []}, schema={'id': pl.Utf8})
    return pl.concat(accumulators, how='diagonal_relaxed')


def build_measurements_streaming(
    corpus_dir: Path,
    *,
    required: Sequence[str],
    runs_df: pl.DataFrame,
    traces_path: Path | None,
    measurable_reads: frozenset[str],
    measurable_signature_fn: Callable[[str], str | None] | None = None,
    batch_size: int = DEFAULT_TRACE_BATCH_SIZE,
) -> Path:
    """Streaming counterpart to ``build_measurements`` for corpora
    whose trace columns can't be fully materialised in RAM.

    The non-streaming ``build_measurements`` expects the caller to
    have already joined every required trace column onto ``runs_df``
    (the runner's ``_join_required_traces`` does this with a full
    ``pl.read_parquet``). For trace files written as one huge row
    group (e.g. snake_g099_canonical_3M_ckpt: 60 cells in 1 row
    group ~= 30 GB uncompressed), that full read OOMs before
    ``build_measurements`` is ever reached.

    This entry computes the **trace-dependent** measurables via
    ``compute_trace_measurables_streaming`` (row-group streaming
    with a per-cell lazy-scan fallback for single-RG files -> peak
    RAM bounded by ONE cell's trace footprint), joins the resulting
    small scalar / per-burst columns onto ``runs_df``, then
    delegates to ``build_measurements`` for persistence + the
    **pure** (non-trace) measurables. ``build_measurements``'s
    partial-nullity branch passes the already-computed trace
    columns through unchanged while filling the pure ones from
    ``runs_df`` proper.

    ``measurable_reads`` is the union of every required measurable's
    transitive record-key reads (the runner's ``trace_reads`` set
    restricted to measurable reads). Only its intersection with the
    trace file's schema is streamed; columns the trace doesn't carry
    are left for ``build_measurements`` to resolve from ``runs_df``.

    ``traces_path=None`` (or a missing file) degenerates to a plain
    ``build_measurements`` call -- no streaming needed.

    Returns the path to ``measurements.parquet`` (same contract as
    ``build_measurements``)."""
    if 'id' not in runs_df.columns:
        raise ValueError(
            f'build_measurements_streaming({corpus_dir}): runs_df is '
            f'missing the `id` column -- required as the per-cell key',
        )
    enriched_runs = runs_df
    if traces_path is not None and traces_path.exists():
        # Stream trace-measurable computation with bounded RAM. The
        # primitive itself picks row-group batching vs the per-cell
        # lazy-scan fallback based on the trace file's row-group
        # layout (single huge RG -> per-cell scan).
        trace_meas = compute_trace_measurables_streaming(
            runs_df,
            traces_path,
            measurable_reads=measurable_reads,
            required=required,
            batch_size=batch_size,
        )
        # ``trace_meas`` carries `id` + the measurable scalar / per-
        # burst columns only -- never the heavyweight trace inputs.
        # Join onto runs_df so ``build_measurements`` sees the
        # already-computed values (its partial-nullity branch keeps
        # them, computes the pure measurables, persists the store).
        meas_cols = [c for c in trace_meas.columns if c != 'id']
        if meas_cols:
            collide = [c for c in meas_cols if c in enriched_runs.columns]
            if collide:
                enriched_runs = enriched_runs.drop(collide)
            enriched_runs = enriched_runs.join(
                trace_meas, on='id', how='left',
            )
    return build_measurements(
        corpus_dir,
        required=required,
        runs_df=enriched_runs,
        traces_path=traces_path,
        measurable_signature_fn=measurable_signature_fn,
    )


@dataclass(frozen=True, slots=True)
class RecomputeResult:
    """Outcome of a `recompute_corpus_measurables` call on one
    corpus.

    Five disjoint name sets discriminate what happened per
    measurable:

    - `recomputed`: measurables whose values were freshly written
      to `measurements.parquet` on this call. Either missing from
      the sidecar before, or sidecar hash drifted from the current
      registry, OR present via `force=` / `recover_nan=` opt-in.
    - `already_current`: measurables whose sidecar hash already
      matched the current registry AND were not forced — no
      recompute needed.
    - `unsatisfiable`: measurables in the gap whose transitive
      record-key reads aren't available in `runs.parquet` or the
      local `traces.parquet`. Skipped to avoid silently overwriting
      finite per-corpus values with NaN. Re-ingest with cloud
      restore to materialise traces if these need filling.
    - `unregistered`: names passed in `required` that aren't in
      the framework's `@measurable` registry. Caller-side bug
      surface — these can never be computed regardless.
    - `recovered_nan`: measurables that were sidecar-current
      (hash matched) but had NaN values in
      `measurements.parquet` AND whose transitive reads are now
      satisfied locally — auto-recomputed when `recover_nan=True`
      was passed. Distinct from `recomputed` so callers can audit
      "what stale-NaN got fixed on this pass" separately from
      "what changed because the substrate updated."
    - `forced_recompute`: measurables that were sidecar-current
      but bypassed via the `force=` parameter (operator
      explicitly asked to recompute). Distinct from
      `recovered_nan` (which is auto-detection-driven, the
      `recover_nan=True` path) so the audit log can tell
      "operator-forced" from "framework-auto-recovered."

    `is_clean` mirrors `CorpusDriftReport.is_clean`: True iff
    nothing was recomputed AND nothing was unsatisfiable AND
    nothing was unregistered AND nothing was recovered from
    stale-NaN AND nothing was force-recomputed (i.e. the corpus
    was fully current on entry, with no operator overrides)."""
    corpus_dir: Path
    recomputed: tuple[str, ...]
    already_current: tuple[str, ...]
    unsatisfiable: tuple[str, ...]
    unregistered: tuple[str, ...]
    recovered_nan: tuple[str, ...] = ()
    forced_recompute: tuple[str, ...] = ()

    @property
    def is_clean(self) -> bool:
        return not (
            self.recomputed or self.unsatisfiable or self.unregistered
            or self.recovered_nan or self.forced_recompute
        )


def resolve_runs_meas_collision(
    *,
    runs_cols: set[str],
    meas_cols: set[str],
    unregistered_policy: Literal['runs_wins', 'meas_wins'],
) -> tuple[set[str], set[str]]:
    """Decide which side wins on (runs ∩ measurements) column
    collisions. Returns `(cols_to_drop_from_runs,
    cols_to_drop_from_meas)`.

    For each name in `meas_cols ∩ runs_cols` (excluding `id`):

    - If the measurable IS registered AND its `transitive_reads`
      are all in `runs_cols` → runs wins (the substrate had all
      inputs at sweep time + its stamp is authoritative). Drop
      from meas.
    - If the measurable IS registered AND its reads aren't all
      in `runs_cols` → meas wins (trace-dependent measurable;
      runs.parquet carries only a sweep-time NaN stamp; the
      measurements file was post-sweep computed when traces
      were available). Drop from runs.
    - If the measurable is NOT registered (or transitive_reads
      raises) → `unregistered_policy` decides:
        * `'runs_wins'` (build_measurements default):
          defensive fallback for CI6 post-orphan-eviction case;
          the runner has the registry import, non-registered
          overlap shouldn't happen in practice.
        * `'meas_wins'` (Panel.from_corpus default): for
          exploration entry points where the substrate may not
          have been imported. The measurements file exists
          because SOME prior runner stamped a value; trust it
          over the runs-side NaN.

    Single source of truth for the substrate-could-compute
    collision logic. Both `build_measurements` and
    `data.panel.Panel.from_corpus` delegate here."""
    from corroborate.measurables.measurable import transitive_reads
    drop_from_runs: set[str] = set()
    drop_from_meas: set[str] = set()
    for c in meas_cols & runs_cols:
        if c == 'id':
            continue
        m = get_registered(c)
        if m is None:
            if unregistered_policy == 'runs_wins':
                drop_from_meas.add(c)
            else:
                drop_from_runs.add(c)
            continue
        try:
            leaf_reads = transitive_reads(c)
        except KeyError:
            if unregistered_policy == 'runs_wins':
                drop_from_meas.add(c)
            else:
                drop_from_runs.add(c)
            continue
        if all(r in runs_cols for r in leaf_reads):
            # Substrate could compute → runs wins.
            drop_from_meas.add(c)
        else:
            # Trace-dependent → meas wins.
            drop_from_runs.add(c)
    return drop_from_runs, drop_from_meas


def check_recoverable_nan(
    corpus_dir: Path,
    *,
    required: Sequence[str],
    measurable_signature_fn: Callable[[str], str | None] | None = None,
) -> tuple[str, ...]:
    """Return names in `required` that are sidecar-current but
    carry NaN/null values in `measurements.parquet` AND whose
    transitive reads are now available from local
    `runs.parquet` ∪ `traces.parquet`.

    These are the "silent NaN, recoverable now" measurables — the
    sidecar hash matched at compute time (so `--recompute-
    measurables` skips them as `already_current`) but the actual
    cell values came out NaN because a transitive trace col was
    cloud-evicted when the measurable was originally computed. The
    standard recompute path doesn't surface them; this function
    does.

    Detection criteria, all must hold:

    1. Name is sidecar-current (`current_signatures[name]` matches
       the registry's live signature).
    2. The column is fully NaN/null in `measurements.parquet`
       (partial NaN is fine — those compute via the normal
       `compute_missing_columns` per-cell missing-mask).
    3. The transitive reads are a subset of
       `runs.parquet.columns ∪ traces.parquet.columns` (NOT
       counting the measurable's own column, which is by
       definition currently all-NaN).

    Used in tandem with `recompute_corpus_measurables(...,
    recover_nan=True)` — that path treats the returned names as
    forced gap and recomputes them on the next build pass.

    **Scope this primitive deliberately does NOT cover:**

    - **Partial-NaN columns.** A column with SOME finite cells
      and some stale-NaN cells (typical for partial trace
      eviction across a corpus) is not surfaced — the detection
      threshold is "all entries are null/NaN." Partial-NaN is
      handled by the standard `compute_missing_columns` per-cell
      missing-mask: pass the name via `force=` if the partial
      state should be rebuilt unconditionally.
    - **Measurables absent from `measurements.parquet`.** A
      required name with no column at all (no sidecar entry,
      never written) falls under the normal sidecar-mismatch gap
      path in `recompute_corpus_measurables`, not this primitive.
      "Recoverable-NaN" specifically means "column exists +
      sidecar says current + values came out NaN at compute
      time."
    - **Numerical-noise near-zero values.** Only literal null /
      NaN counts as missing. A column of finite 1e-20 values
      from a degenerate computation is NOT flagged.

    No-op when `runs.parquet` is missing or sidecar is empty —
    returns `()`."""
    if measurable_signature_fn is None:
        def _default_sig(name: str) -> str | None:
            m = get_registered(name)
            return None if m is None else m.signature()
        sig_fn: Callable[[str], str | None] = _default_sig
    else:
        sig_fn = measurable_signature_fn

    runs_path = corpus_dir / 'runs.parquet'
    if not runs_path.exists():
        return ()
    stored = current_signatures(corpus_dir)
    if not stored:
        return ()
    measurements_path = _measurements_path(corpus_dir)
    if not measurements_path.exists():
        return ()

    # Schema-level scan: figure out which `required` names have a
    # column written, are sidecar-current, and are fully null/NaN.
    meas_schema = pl.scan_parquet(measurements_path).collect_schema()
    meas_cols = set(meas_schema.names())
    candidates: list[str] = []
    for name in required:
        if name not in meas_cols:
            continue
        live = sig_fn(name)
        if live is None:
            continue
        if stored.get(name) != live:
            continue  # drift handled by normal gap path
        candidates.append(name)

    if not candidates:
        return ()

    # Cell-level NaN check: load just the candidate columns from
    # measurements.parquet and find names that are all-null or
    # all-NaN.
    meas = pl.read_parquet(measurements_path, columns=candidates)
    all_nan: list[str] = []
    for name in candidates:
        col = meas[name]
        n_null = int(col.is_null().sum())
        n_nan = int(col.is_nan().sum()) if col.dtype.is_float() else 0
        if n_null + n_nan == len(col):
            all_nan.append(name)
    if not all_nan:
        return ()

    # Transitive-reads availability: same gate as the recompute
    # path so we only surface recoverable cases.
    from corroborate.measurables.measurable import transitive_reads
    available: set[str] = set(
        pl.scan_parquet(runs_path).collect_schema().names()
    )
    traces_path = corpus_dir / 'traces.parquet'
    if traces_path.exists():
        try:
            available |= set(
                pl.scan_parquet(traces_path).collect_schema().names()
            )
        except pl.exceptions.ComputeError:
            pass

    recoverable: list[str] = []
    for name in all_nan:
        try:
            reads = transitive_reads(name)
        except KeyError:
            continue
        # `name` itself is NaN, so don't count its own column as
        # an available read.
        if (reads - {name}).issubset(available):
            recoverable.append(name)
    return tuple(recoverable)


def recompute_corpus_measurables(
    corpus_dir: Path,
    *,
    required: Sequence[str],
    measurable_signature_fn: Callable[[str], str | None] | None = None,
    force: frozenset[str] | None = None,
    recover_nan: bool = False,
) -> RecomputeResult:
    """Recompute the *gap* between `required` and the corpus's
    persisted `measurements.parquet` — strictly using LOCAL inputs
    (`runs.parquet` + local `traces.parquet`). Opt-in
    counterpart to the cloud-restore-driven recompute in
    `_load_one_corpus`: gives the operator a way to fill in
    newly-registered measurables without paying a cloud round-trip
    when the trace data already lives locally.

    Idempotent. Drift / missing detection uses the same closure-
    hash contract as `check_drift`. The recompute path filters the
    gap to satisfiable measurables (whose `transitive_reads` are
    all present in `runs.parquet` ∪ local trace columns) before
    handing off to `build_measurements`. Unsatisfiable measurables
    are reported, NOT computed — overwriting a finite per-corpus
    value with a fresh NaN (which is what would happen if we
    forced compute with the read missing) would be silent data
    loss.

    Cloud-evicted traces are NOT auto-restored on this path —
    that's the contract the docstring promises and what callers
    rely on for predictability. If the user wants restore-driven
    recompute, the existing `--ingest` flow already handles it
    via `_load_one_corpus`'s sidecar-current check.

    `force`: optional frozenset of measurable names to recompute
    unconditionally, bypassing the sidecar's "current" check.
    Empty frozenset is treated as "no forcing" (None semantics);
    pass actual names to force. Forced names that aren't in
    `required` are silently ignored (the recompute path can only
    operate on the required closure). Forced names land in
    `recovered_nan` of the result for auditability.

    `recover_nan`: when True, auto-detect names that are
    sidecar-current but fully-NaN in `measurements.parquet`
    (using `check_recoverable_nan`) and add them to the force
    set. Closes the silent-NaN gap where the sidecar says
    "computed" but the trace col was cloud-evicted at the time.

    No-op when `runs.parquet` is missing (returns an empty
    `RecomputeResult` with the corpus_dir field set). The caller
    typically logs this case and continues.
    """
    if measurable_signature_fn is None:
        def _default_sig(name: str) -> str | None:
            m = get_registered(name)
            return None if m is None else m.signature()
        sig_fn: Callable[[str], str | None] = _default_sig
    else:
        sig_fn = measurable_signature_fn

    runs_path = corpus_dir / 'runs.parquet'
    if not runs_path.exists():
        return RecomputeResult(
            corpus_dir=corpus_dir,
            recomputed=(),
            already_current=(),
            unsatisfiable=(),
            unregistered=tuple(required),
        )

    # Resolve the effective force-set: explicit `force` ∪
    # auto-detected stale-NaN names (when recover_nan=True).
    # Track origin separately for the audit fields.
    force_explicit: set[str] = set(force) if force else set()
    nan_detected: tuple[str, ...] = ()
    if recover_nan:
        nan_detected = check_recoverable_nan(
            corpus_dir,
            required=required,
            measurable_signature_fn=sig_fn,
        )
    force_from_nan: set[str] = set(nan_detected)
    # Intersect with `required`: forcing names outside the
    # required closure has no recompute pathway (build_measurements
    # only operates on `required`).
    required_set = set(required)
    force_explicit &= required_set
    force_from_nan &= required_set
    # A name in BOTH explicit and NaN-detected is attributed to
    # the explicit set (operator intent dominates auto-detection).
    force_from_nan -= force_explicit
    force_set = force_explicit | force_from_nan

    # 1. Classify each required name vs the current sidecar state.
    stored = current_signatures(corpus_dir)
    unregistered: list[str] = []
    already_current: list[str] = []
    gap: list[str] = []   # missing OR drifted OR forced
    for name in required:
        live = sig_fn(name)
        if live is None:
            unregistered.append(name)
            continue
        if name in force_set:
            gap.append(name)
            continue
        if stored.get(name) == live:
            already_current.append(name)
            continue
        gap.append(name)

    if not gap:
        return RecomputeResult(
            corpus_dir=corpus_dir,
            recomputed=(),
            already_current=tuple(already_current),
            unsatisfiable=(),
            unregistered=tuple(unregistered),
            recovered_nan=(),
            forced_recompute=(),
        )

    # 2. Walk the gap and decide which are satisfiable from local
    #    inputs only. Reads from runs.parquet schema first (cheap —
    #    metadata only), then unions with the local trace schema
    #    when present.
    from corroborate.measurables.measurable import transitive_reads
    runs_schema = pl.scan_parquet(runs_path).collect_schema()
    available: set[str] = set(runs_schema.names())
    traces_path = corpus_dir / 'traces.parquet'
    if traces_path.exists():
        try:
            traces_schema = pl.scan_parquet(traces_path).collect_schema()
            available |= set(traces_schema.names())
        except pl.exceptions.ComputeError:
            # Corrupt / truncated traces.parquet — treat as having
            # no trace cols. Mirrors `_join_required_traces`'s
            # defensive handling in the runner.
            pass

    satisfiable: list[str] = []
    unsatisfiable: list[str] = []
    for name in gap:
        try:
            reads = transitive_reads(name)
        except KeyError:
            # Registry mutated between sig_fn and transitive_reads
            # — defensive only; should not fire in single-thread
            # use.
            unsatisfiable.append(name)
            continue
        if reads.issubset(available):
            satisfiable.append(name)
        else:
            unsatisfiable.append(name)

    if not satisfiable:
        return RecomputeResult(
            corpus_dir=corpus_dir,
            recomputed=(),
            already_current=tuple(already_current),
            unsatisfiable=tuple(unsatisfiable),
            unregistered=tuple(unregistered),
            recovered_nan=(),
            forced_recompute=(),
        )

    # 3. Load runs + join the trace cols the satisfiable
    #    measurables actually need. Mirrors the runner's
    #    `_join_required_traces` but reduced to the satisfiable
    #    closure (avoids loading trace cols the gap doesn't read).
    gap_reads: set[str] = set()
    for name in satisfiable:
        gap_reads |= transitive_reads(name)
    runs_df = pl.read_parquet(runs_path)
    if traces_path.exists() and 'id' in runs_df.columns:
        try:
            traces_schema_names: set[str] = set(
                pl.scan_parquet(traces_path).collect_schema().names()
            )
        except pl.exceptions.ComputeError:
            traces_schema_names = set()
        cols_to_load = ['id'] + sorted(gap_reads & traces_schema_names)
        if len(cols_to_load) > 1:
            traces = pl.read_parquet(traces_path, columns=cols_to_load)
            # Drop collisions: any non-id trace col already on
            # runs_df. The runs-side value wins (substrate-stamped
            # at sweep time); polars' join would otherwise
            # `_right`-suffix it and break downstream reads.
            overlap = [
                c for c in traces.columns
                if c != 'id' and c in runs_df.columns
            ]
            if overlap:
                traces = traces.drop(overlap)
            runs_df = runs_df.join(traces, on='id', how='left')

    # 4. Force a build_measurements pass. `required` includes both
    #    the satisfiable gap + already-current names so the build's
    #    own drift / orphan logic preserves the existing columns
    #    that don't need recompute. (Passing only the satisfiable
    #    gap would mark all already-current cols as orphan and
    #    drop them.)
    build_measurements(
        corpus_dir,
        required=required,
        runs_df=runs_df,
        measurable_signature_fn=sig_fn,
        # Force-recompute the SATISFIABLE forced names even when the
        # sidecar says "current". `force` widened the gap (so their
        # reads got joined into runs_df above), but build_measurements
        # re-checks the sidecar — without passing force through it would
        # keep the stale value (the force no-op bug). Restrict to
        # `satisfiable` so unsatisfiable forced names stay PRESERVED
        # (forcing them would null-pad — their reads aren't available).
        force=frozenset(satisfiable) & force_set,
    )

    # Split recomputed into 3 disjoint audit slots:
    # - `recovered_nan`: auto-detected stale-NaN (recover_nan=True)
    # - `forced_recompute`: explicit operator force (force=...)
    # - `recomputed`: regular substrate-driven gap (missing or drifted)
    satisfiable_set = set(satisfiable)
    nan_in_satisfiable = satisfiable_set & force_from_nan
    forced_in_satisfiable = satisfiable_set & force_explicit
    return RecomputeResult(
        corpus_dir=corpus_dir,
        recomputed=tuple(
            n for n in satisfiable
            if n not in nan_in_satisfiable and n not in forced_in_satisfiable
        ),
        already_current=tuple(already_current),
        unsatisfiable=tuple(unsatisfiable),
        unregistered=tuple(unregistered),
        recovered_nan=tuple(n for n in satisfiable if n in nan_in_satisfiable),
        forced_recompute=tuple(
            n for n in satisfiable if n in forced_in_satisfiable
        ),
    )


def check_drift(
    root: Path,
    *,
    required: Sequence[str],
    measurable_signature_fn: Callable[[str], str | None] | None = None,
) -> DriftReport:
    """Walk subdirs of `root`, audit each corpus's
    `measurements.hashes.json` against the current registry's
    closure hashes for `required` measurables. Returns a
    `DriftReport`. Pure read — does NOT load `runs.parquet`,
    does NOT compute measurables, does NOT touch cloud.

    Skips:
    - Subdirs without `runs.parquet` (not a corpus).
    - Subdirs with `.in_progress` sentinel (sweep mid-flight,
      same convention as the runner).
    - `required` names that don't resolve in the current
      registry (`signature_fn` returns None) — those would be
      caller-side authoring bugs, not drift.
    """
    if measurable_signature_fn is None:
        def _default_sig(name: str) -> str | None:
            m = get_registered(name)
            return None if m is None else m.signature()
        sig_fn: Callable[[str], str | None] = _default_sig
    else:
        sig_fn = measurable_signature_fn

    if not root.is_dir():
        return DriftReport(per_corpus=())

    # Lazy import to avoid circular dep.
    from corroborate.corpus.integrity import is_in_progress

    per: list[CorpusDriftReport] = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        if not (sub / 'runs.parquet').exists():
            continue
        if is_in_progress(sub):
            continue
        stored = current_signatures(sub)
        drifted: list[str] = []
        missing: list[str] = []
        for name in required:
            current = sig_fn(name)
            if current is None:
                continue   # measurable unknown to registry; not drift
            stored_hash = stored.get(name)
            if stored_hash is None:
                missing.append(name)
            elif stored_hash != current:
                drifted.append(name)
        per.append(CorpusDriftReport(
            corpus_dir=sub,
            drifted=tuple(drifted),
            missing=tuple(missing),
        ))
    return DriftReport(per_corpus=tuple(per))


__all__ = [
    'MEASUREMENTS_FILENAME',
    'SIDECAR_FILENAME',
    'CorpusDriftReport',
    'DriftReport',
    'RecomputeResult',
    'build_measurements',
    'build_measurements_streaming',
    'check_drift',
    'check_recoverable_nan',
    'current_signatures',
    'load_measurements',
    'recompute_corpus_measurables',
]
