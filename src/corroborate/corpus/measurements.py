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
from pathlib import Path

import polars as pl

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
    # **CORPUS_INTEGRITY.md CI6**: row-level orphan eviction.
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
        from corroborate.measurables.measurable import transitive_reads
        runs_cols = set(runs_df.columns)
        overlap_dropped = []
        for c in existing.columns:
            if c == 'id' or c not in runs_cols:
                continue
            m = get_registered(c)
            if m is None:
                # Non-registered overlap (shouldn't happen post-CI6
                # orphan eviction); fall back to old "runs_df wins"
                # so the join doesn't produce `_right`-suffixed
                # garbage.
                overlap_dropped.append(c)
                continue
            # Substrate "could compute" iff its transitive leaf
            # reads are all in runs_df (pre-trace-join). Direct
            # reads alone miss param-injected dependencies (e.g.
            # `effective_horizon` directly reads `gamma`, but
            # transitively also needs `done` via injected
            # `bootstrap_fraction`).
            leaf_reads = transitive_reads(c)
            substrate_could_compute = all(r in runs_cols for r in leaf_reads)
            if substrate_could_compute:
                overlap_dropped.append(c)
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
    no_partial_nulls = all(
        n not in joined.columns
        or not joined[n].is_null().any()
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
    measurable_cols = [
        c for c in enriched.columns
        if c == 'id' or c in registered_names()
    ]
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


def compute_trace_measurables_streaming(
    runs_df: pl.DataFrame,
    traces_path: Path,
    *,
    measurable_reads: frozenset[str],
    required: Sequence[str],
    batch_size: int = DEFAULT_TRACE_BATCH_SIZE,
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
    runs_indexed = runs_df.set_sorted('id') if False else runs_df
    pq_file = _pq.ParquetFile(str(traces_path))
    accumulators: list[pl.DataFrame] = []
    measurable_set = set(registered_names())
    n_per_rg = max(1, batch_size)
    rg_indices = list(range(pq_file.num_row_groups))
    # Group consecutive row-groups so each batch carries ~batch_size
    # cells. Polars / pyarrow row-group counts vary by writer; the
    # outer loop chunks them to a stable batch size.
    rg_groups: list[list[int]] = []
    cur: list[int] = []
    cur_rows = 0
    for idx in rg_indices:
        rg_rows = pq_file.metadata.row_group(idx).num_rows
        if cur and cur_rows + rg_rows > n_per_rg:
            rg_groups.append(cur)
            cur = [idx]
            cur_rows = rg_rows
        else:
            cur.append(idx)
            cur_rows += rg_rows
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
            pl.col('id').is_in(batch_traces['id']),
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
    'build_measurements',
    'check_drift',
    'current_signatures',
    'load_measurements',
]
