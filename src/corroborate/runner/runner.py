"""Hypothesis-module runner — pytest-like dispatch for bridges files.

Each `experiments/findings/<name>.py` IS a hypothesis: it declares
the bridges that test the claim (`BRIDGES` tuple) plus, via its own
top-level imports, registers the substrate measurables required.
The runner imports the module, walks its bridges' measurable
dependencies, populates / extends the per-module cache by
computing missing measurables for each cell, and dispatches each
bridge through `claim_bridge.evaluate()`.

Cache lifecycle (pytest-like — incidental, not first-class):

- One cache file per module at
  `experiments/data/cache/<module>.parquet`. Decoupled from the
  module source so a copy of the bridges file can be tested
  without disturbing the canonical cache.
- Append-on-use: when a new corpus's cells flow through, missing
  measurables are computed and the cells get appended to the
  cache. Cell-level dedup by `id` (UUID) — same cell never gets
  re-ingested. Measurable-level dedup by column presence — a
  cell already enriched with the required measurables doesn't
  recompute them.
- `use_cache=False`: pure data → measurables → verdicts; no
  cache read or write.
- `write_cache=False`: read cache for speedup, but don't persist
  updates (useful for ad-hoc verdicts on a one-off input).
- `rebuild=True`: invalidate the cache before running.

Lazy raw-restore: when ingesting a corpus directory that's local-
archived (only `_remote.json` present, no `runs.parquet`), the
runner pulls raw from s3 unless `restore_from_cloud=False`. The
warning surface is loud when restore is unavailable and a corpus
is needed but missing.

Hypothesis surface — every bridges module / class satisfies the
`Hypothesis` Protocol (`corroborate.core.hypothesis.Hypothesis`)
by declaring `INTERVENTION: DoEffect`, `BRIDGES:
tuple[Bridge, ...]`, `__name__: str`. The cache file is keyed
off `h.__name__.split('.')[-1]` — there is no override surface;
if the cache lives in a non-default location, write a thin
script that calls `run(..., cache_path=...)` directly (the
kwarg exists for that purpose).

This module is library-only — no argparse, no `if __name__ ==
'__main__'`. The CLI thin-wrapper lives at
`scripts/run_hypothesis.py`."""
from __future__ import annotations

import importlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import polars as pl

from corroborate.bridge.analysis import get_registered as _get_analysis
from corroborate.bridge.bridge import (
    Bridge,
    BridgeEvaluation,
    evaluate,
    measurable_names_for_bridges,
)
from corroborate.core.hypothesis import Hypothesis
from corroborate.corpus.cloud import RemoteManifest
from corroborate.corpus.schema import LINEAGE_FIELDS
from corroborate.measurables import (
    compute_missing_columns,
    get_registered,
    transitive_reads,
)


# ============ Hypothesis validation + bridge auto-collection ============


def collect_bridges(
    namespace: Mapping[str, object],
) -> tuple[Bridge, ...]:
    """Auto-collect every module-level `Bridge` instance from a
    namespace. Substitute for hand-curating `BRIDGES` tuples.

    Usage at the bottom of a bridges file:

        BRIDGES = collect_bridges(globals())

    Replaces the explicit `BRIDGES = (*GROUP_A, *GROUP_B, ...)`
    boilerplate when the author just wants "every bridge in this
    file." Named subgroups (`NSTEP_INTERVENTION_BRIDGES`,
    `ACTION_DIM_BRIDGES`) stay as deliberately-curated tuples for
    partial-evaluation use cases — they're documentation, not the
    canonical-run set.

    Order is module-definition order (the order Python populates
    `globals()`); cross-bridge dependencies should be declared
    explicitly via per-bridge gates, not relied on through
    ordering."""
    return tuple(
        v for v in namespace.values() if isinstance(v, Bridge)
    )


def _validate_hypothesis(h: object) -> Hypothesis:
    """Narrow `h` to the framework's `Hypothesis` Protocol via
    `__instancecheck__` (`runtime_checkable`), then verify each
    `BRIDGES` element is a `Bridge`. Raises `TypeError` on shape
    errors. Both Python modules and class-based hypotheses
    satisfy the Protocol structurally as long as they expose
    `INTERVENTION: DoEffect`, `BRIDGES: tuple[Bridge, ...]`, and
    `__name__: str`."""
    if not isinstance(h, Hypothesis):
        raise TypeError(
            f'{type(h).__name__} does not satisfy the Hypothesis '
            f'Protocol: missing one of `INTERVENTION: DoEffect`, '
            f'`BRIDGES: tuple[Bridge, ...]`, `__name__: str` at the '
            f'module / class level.',
        )
    # `runtime_checkable.__instancecheck__` only validates attribute
    # *presence*, so a defensive element-type check defends against
    # malformed authoring (e.g. a non-Bridge slipped into the tuple
    # via a copy-paste mistake).
    for b in h.BRIDGES:
        if not isinstance(b, Bridge):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(
                f'{h.__name__}.BRIDGES contains non-Bridge: '
                f'{type(b).__name__}',
            )
    return h


def _default_cache_path(h: Hypothesis) -> Path:
    """Per-hypothesis cache file at
    `experiments/data/cache/<short>.parquet`. For modules, `<short>`
    is the last segment of the dotted path; for classes, it's the
    class's bare `__name__`."""
    short = h.__name__.split('.')[-1]
    return Path('experiments/data/cache') / f'{short}.parquet'


# ============ Measurable signature + manifest ============


def _measurable_signature(name: str) -> str | None:
    """Closure hash for a registered measurable, or None when the
    name isn't currently registered (e.g. a column already in the
    cache that doesn't belong to a current measurable — those are
    left untouched). Forwards to `Measurable.signature()`; the
    actual hash logic lives on the Measurable itself."""
    m = get_registered(name)
    return None if m is None else m.signature()


def _manifest_path(cache_path: Path) -> Path:
    """Manifest sidecar lives alongside the cache parquet."""
    return cache_path.with_suffix('.hashes.json')


def _read_manifest(path: Path) -> dict[str, str]:
    """Parse the sidecar JSON; tolerant of corruption / wrong shape
    (returns `{}` rather than raising) so a malformed manifest just
    triggers a full rebuild rather than aborting the runner."""
    if not path.exists():
        return {}
    # `json.loads` is typed `Any`; cast to `object` so the
    # isinstance below actually narrows.
    parsed = cast(object, json.loads(path.read_text()))
    if not isinstance(parsed, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in parsed.items():
        if isinstance(k, str) and isinstance(v, str):
            out[k] = v
    return out


def _write_manifest(path: Path, sigs: Mapping[str, str]) -> None:
    path.write_text(json.dumps(dict(sigs), indent=2, sort_keys=True))


def _invalidate_drifted(
    cache: pl.DataFrame,
    manifest: Mapping[str, str],
    required: Sequence[str],
) -> pl.DataFrame:
    """Drop columns whose stored signature doesn't match the
    current closure hash. The dropped columns then fall through
    `_compute_measurables`'s "missing column → fill" path so the
    user sees fresh values without a manual `--rebuild`.

    Loud warning lists what drifted so the user knows where the
    recompute time went."""
    drifted: list[str] = []
    for name in required:
        if name not in cache.columns:
            continue
        current = _measurable_signature(name)
        if current is None:
            continue
        stored = manifest.get(name)
        if stored is not None and stored != current:
            drifted.append(name)
    if not drifted:
        return cache
    print(
        f'runner: invalidating {len(drifted)} drifted measurable '
        f'column(s): {drifted}',
        file=sys.stderr,
    )
    return cache.drop(drifted)


# ============ Public surface ============


def run(
    h: Hypothesis | str,
    *,
    data: pl.DataFrame | Path | str | None = None,
    use_cache: bool = True,
    write_cache: bool = True,
    rebuild: bool = False,
    restore_from_cloud: bool = True,
    cache_path: Path | None = None,
) -> dict[str, BridgeEvaluation]:
    """Run a hypothesis's bridges on `data`, returning per-bridge
    verdicts.

    `h` may be:
    - a Python module satisfying the `Hypothesis` Protocol
      (`INTERVENTION: DoEffect` + `BRIDGES: tuple[Bridge, ...]`),
    - a class-based hypothesis (frozen dataclass with `ClassVar`
      fields of the same shape),
    - a string dotted module path (the CLI's input form);
      imported via `importlib.import_module` then validated.

    Cache lifecycle:

    - `use_cache=True` (default): read+write the per-hypothesis
      cache. Cells already in cache with all required measurables
      skip recomputation. New cells from `data` get measurables
      computed and appended.
    - `use_cache=False`: pure compute path; no cache read or write.
    - `write_cache=False` + `use_cache=True`: read cache, run, but
      don't persist updates.
    - `rebuild=True`: invalidate the per-hypothesis cache before
      running. Implies `use_cache=True`.

    `restore_from_cloud=True` (default): when ingesting a corpus
    directory whose `runs.parquet` is missing locally but has a
    `_remote.json` manifest, pull raw from s3. Set False to opt
    out + warn loudly on the missing data.

    `cache_path`: explicit override for the cache file. When None
    and `use_cache=True`, defaults to
    `experiments/data/cache/<short>.parquet` where `<short>` is
    the last segment of `h.__name__` (modules) or the class's
    `__name__`.

    `data` may be:
    - `None`: run on whatever's already in the cache.
    - a `pl.DataFrame`: use as-is.
    - a path to a `.parquet` file: read directly.
    - a path to a directory: walk its subdirs for per-corpus
      `runs.parquet` (with auto-restore), concat via
      `diagonal_relaxed`."""
    if isinstance(h, str):
        h = _validate_hypothesis(importlib.import_module(h))
    else:
        h = _validate_hypothesis(h)
    bridges = h.BRIDGES

    resolved_cache: Path | None = None
    if use_cache:
        resolved_cache = cache_path if cache_path is not None else _default_cache_path(h)
        resolved_cache.parent.mkdir(parents=True, exist_ok=True)
        if rebuild:
            resolved_cache.unlink(missing_ok=True)
            _manifest_path(resolved_cache).unlink(missing_ok=True)

    cells = _ingest_and_compute(
        bridges=bridges,
        data=data,
        cache_path=resolved_cache,
        write_cache=write_cache and use_cache,
        restore_from_cloud=restore_from_cloud,
    )

    if cells.height == 0:
        raise SystemExit(
            f'{h.__name__}: no cells available — pass --data to '
            f'ingest a corpus, or check the cache at {resolved_cache}',
        )

    out: dict[str, BridgeEvaluation] = {}
    for b in bridges:
        try:
            out[b.name] = evaluate(b, cells)
        except Exception as e:  # noqa: BLE001
            # An authoring bug in a bridge's `holds_when` body
            # (e.g. typo'd column name, malformed analysis call)
            # raises here. Print the error to stderr and skip
            # this bridge — DO NOT synthesise a fake verdict.
            # Conflating evaluation errors with the POWER_INSUFFICIENT
            # verdict would smuggle authoring bugs past the reader,
            # exactly what the framework's verdict layer refuses
            # (see verdict.Verdict docstring + CLAUDE.md §verdict).
            print(
                f'  [bridge {b.name!r} raised during evaluation: '
                f'{type(e).__name__}: {e}]',
                file=sys.stderr,
            )
    return out


# ============ Cache + ingest ============


def _ingest_and_compute(
    *,
    bridges: tuple[Bridge, ...],
    data: pl.DataFrame | Path | str | None,
    cache_path: Path | None,
    write_cache: bool,
    restore_from_cloud: bool,
) -> pl.DataFrame:
    """Load cache (if any), append new data after computing missing
    measurables, persist, return the merged DataFrame.

    Lifecycle:
    1. Read parquet + sidecar manifest (`<cache>.hashes.json`).
    2. Drop columns whose closure hash drifted vs. manifest — this
       is the "measurable formula changed" detection path.
    3. Existing cells fall through `_compute_measurables` to fill
       missing columns (drifted ones are now missing, plus any
       brand-new required measurables).
    4. New cells from `data` get measurables computed and merged.
    5. Persist parquet + updated manifest with current signatures.

    The manifest is written for all currently-required measurables
    that have a column — so on the next run, anything edited in
    the meantime gets caught by step 2."""
    required = sorted(measurable_names_for_bridges(bridges))
    manifest_path = (
        _manifest_path(cache_path) if cache_path is not None else None
    )
    stored_manifest = (
        _read_manifest(manifest_path) if manifest_path is not None else {}
    )

    cache = _load_cache(cache_path)
    cache = _invalidate_drifted(cache, stored_manifest, required)
    new_data = _load_data(
        data, restore_from_cloud=restore_from_cloud,
        required=required, bridges=bridges,
    )

    if new_data is None or new_data.height == 0:
        return _enrich_cache_in_place(
            cache, required, cache_path, manifest_path, write_cache,
        )

    new_subset = _dedup_against_cache(cache, new_data)
    enriched_new = _compute_measurables(new_subset, required)

    if cache.height == 0:
        merged = enriched_new
    else:
        cache_enriched = _compute_measurables(cache, required)
        merged = pl.concat(
            [cache_enriched, enriched_new], how='diagonal_relaxed',
        ) if enriched_new.height > 0 else cache_enriched

    if cache_path is not None and write_cache:
        merged.write_parquet(cache_path)
        if manifest_path is not None:
            _write_manifest(manifest_path, _signatures_for(required, merged))
    return merged


def _enrich_cache_in_place(
    cache: pl.DataFrame,
    required: Sequence[str],
    cache_path: Path | None,
    manifest_path: Path | None,
    write_cache: bool,
) -> pl.DataFrame:
    """When no new data is supplied, still pass the cache through
    `_compute_measurables` so that newly-added required measurables
    get filled in for existing cells. Persist the manifest if
    columns changed (added / drifted)."""
    if cache.height == 0:
        return cache
    enriched = _compute_measurables(cache, required)
    if (
        cache_path is not None
        and write_cache
        and enriched.columns != cache.columns
    ):
        enriched.write_parquet(cache_path)
        if manifest_path is not None:
            _write_manifest(
                manifest_path, _signatures_for(required, enriched),
            )
    return enriched


def _signatures_for(
    required: Sequence[str], df: pl.DataFrame,
) -> dict[str, str]:
    """Snapshot the current closure hash for every required
    measurable that's actually a column in `df`. Only registered
    names get a signature; unknown columns aren't tracked."""
    out: dict[str, str] = {}
    for name in required:
        if name not in df.columns:
            continue
        sig = _measurable_signature(name)
        if sig is not None:
            out[name] = sig
    return out


# Provenance / lineage tags that don't define a cell's scientific
# identity. Two cells differing ONLY by these are the same physical
# experiment surfaced under different bookkeeping (e.g. one corpus
# was assembled by merging another — same RunRow, different `corpus`
# tag). UUID isn't a scientific-identity column either: a fresh
# `uuid.uuid4()` gets minted at run time per cell, so independent
# re-runs of the same `(env, arm, seed, HPs)` get distinct ids
# despite being scientifically equivalent — content equality is the
# right check, not UUID equality.
_PROVENANCE_TAGS: frozenset[str] = LINEAGE_FIELDS | {'corpus'}


def _dedup_by_content(df: pl.DataFrame, *, source: str) -> pl.DataFrame:
    """Drop rows whose non-provenance columns are all equal — i.e.
    cells that differ only in `id`/`corpus`/`timestamp` / lineage
    tags. The merge artifacts (same physical run surfaced under two
    `corpus` tags) collapse to one row; truly distinct runs are
    preserved (their measurement columns differ).

    Polars' `unique(subset=...)` handles primitive columns natively;
    list/object columns get coerced via `hash` first so the equality
    check is value-based even on heterogeneous shapes."""
    if df.height == 0:
        return df
    content_cols = [c for c in df.columns if c not in _PROVENANCE_TAGS]
    if not content_cols:
        return df
    before = df.height
    try:
        deduped = df.unique(subset=content_cols, keep='first')
    except pl.exceptions.InvalidOperationError:
        # Some content columns may be list/object dtypes that
        # `unique` can't compare directly; hash them first.
        hash_expr = pl.struct(content_cols).hash().alias('_content_hash')
        deduped = (
            df.with_columns(hash_expr)
            .unique(subset=['_content_hash'], keep='first')
            .drop('_content_hash')
        )
    dropped = before - deduped.height
    if dropped:
        print(
            f'runner: deduped {dropped} content-identical cell(s) '
            f'from {source} (same scientific cell surfaced under '
            f'multiple {sorted(_PROVENANCE_TAGS & set(df.columns))} '
            f'tags)',
            file=sys.stderr,
        )
    return deduped


def _load_cache(path: Path | None) -> pl.DataFrame:
    if path is None or not path.exists():
        return pl.DataFrame()
    return _dedup_by_content(pl.read_parquet(path), source='cache')


def _dedup_against_cache(
    cache: pl.DataFrame, new_data: pl.DataFrame,
) -> pl.DataFrame:
    """Drop cells from `new_data` whose `id` is already in `cache`."""
    if cache.height == 0 or 'id' not in cache.columns:
        return new_data
    if 'id' not in new_data.columns:
        return new_data
    existing = set(cache['id'].to_list())
    if not existing:
        return new_data
    return new_data.filter(~pl.col('id').is_in(list(existing)))


def _compute_measurables(
    df: pl.DataFrame,
    required: Sequence[str],
) -> pl.DataFrame:
    """For each required measurable not yet in df.columns, compute
    per-cell and add as a column. Thin forwarder to
    `corroborate.measurable.compute_missing_columns` — the per-cell
    eval loop lives there as the single source of truth."""
    return compute_missing_columns(df, required)


# ============ Data loading ============


def _load_data(
    data: pl.DataFrame | Path | str | None,
    *,
    restore_from_cloud: bool,
    required: Sequence[str],
    bridges: tuple[Bridge, ...],
) -> pl.DataFrame | None:
    """Resolve data into a DataFrame, with auto-restore on missing-
    raw corpora when given a directory."""
    if data is None:
        return None
    if isinstance(data, pl.DataFrame):
        return data
    p = Path(data)
    if not p.exists() and (
        not p.is_absolute() and (Path.cwd() / p).exists()
    ):
        p = Path.cwd() / p
    if p.is_dir():
        return _load_directory(
            p, restore_from_cloud=restore_from_cloud,
            required=required, bridges=bridges,
        )
    if p.is_file():
        return pl.read_parquet(p)
    raise FileNotFoundError(f'no such data path: {data}')


def _missing_for_restore(
    runs_path: Path,
    traces_path: Path,
    trace_reads: frozenset[str],
    manifest_path: Path,
) -> list[str] | None:
    """Decide which files we need restored from cloud for this
    corpus.

    `runs.parquet` is always required (the row store). `traces.
    parquet` is required only when any required measurable or
    declared analysis-read pulls from a trace-store column AND the
    remote manifest carries it AND it's missing/stub locally.

    Returns the list of relpaths to restore, or None if nothing
    needs restoring. Stub local files (size < 1KB) are treated as
    missing — some corpora carry zero-byte placeholders."""
    targets: list[str] = []
    if not _file_present(runs_path):
        targets.append('runs.parquet')
    if trace_reads and not _file_present(traces_path):
        # Two manifest shapes count as carrying trace data:
        # (a) a top-level `traces.parquet` entry (canonical), or
        # (b) per-arm `tmp/*_traces.parquet` shards (older sweeps
        # that archived before the per-corpus merge step ran). For
        # (b), `_merge_shard_traces` stitches the shards into a
        # canonical `traces.parquet` after restore so downstream
        # code sees one shape.
        manifest = _read_remote_manifest(manifest_path)
        if manifest is not None:
            if manifest.has('traces.parquet'):
                targets.append('traces.parquet')
            else:
                shards = sorted(
                    rp for rp in manifest.relpaths()
                    if rp.startswith('tmp/')
                    and rp.endswith('_traces.parquet')
                )
                targets.extend(shards)
    return targets or None


def _read_remote_manifest(manifest_path: Path) -> RemoteManifest | None:
    """Parse a `_remote.json` into a typed `RemoteManifest`.
    Returns None on any I/O or shape error — caller treats that as
    "manifest not consultable" and proceeds without restore."""
    try:
        raw = cast(object, json.loads(manifest_path.read_text()))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    typed: Mapping[str, object] = cast(Mapping[str, object], raw)
    try:
        return RemoteManifest.from_dict(typed)
    except (TypeError, KeyError):
        return None


def _file_present(path: Path, *, min_size: int = 1024) -> bool:
    """A file 'counts' only if it exists, is at least `min_size`
    bytes, AND (if it's a parquet) has a valid PAR1 footer.
    Corrupt parquets — partial downloads, killed-mid-write files
    — would otherwise pass the size check but fail later when
    polars tries to parse them. Treating them as missing here
    triggers a clean re-restore from cloud rather than silently
    skipping the trace join."""
    if not path.exists():
        return False
    try:
        size = path.stat().st_size
    except OSError:
        return False
    if size < min_size:
        return False
    if path.suffix == '.parquet':
        # Parquet files end with the 4-byte magic 'PAR1'. Cheap
        # integrity check — far cheaper than full schema parse,
        # catches truncated downloads and write-killed files.
        try:
            with path.open('rb') as fh:
                fh.seek(-4, 2)
                magic = fh.read(4)
        except OSError:
            return False
        if magic != b'PAR1':
            return False
    return True


def _required_record_keys(required: Sequence[str]) -> frozenset[str]:
    """Union of leaf record-keys the required measurables read,
    walked transitively via `transitive_reads`. Used to determine
    which `traces.parquet` columns to join per-corpus before the
    measurable evaluator runs.

    Names that aren't registered are silently dropped — they're
    e.g. raw record fields the measurables don't go through."""
    out: set[str] = set()
    for name in required:
        m = get_registered(name)
        if m is None:
            continue
        try:
            out.update(transitive_reads(name))
        except KeyError:
            continue
    return frozenset(out)


def _analysis_reads_for_bridges(
    bridges: tuple[Bridge, ...],
) -> frozenset[str]:
    """Union of `Analysis.reads` for every analysis a bridge in
    `bridges` consumes. The bridge's holds_when fixture-parameter
    names ARE the analysis names — so we walk those.

    These reads are columns the analyses touch off the cell record
    DIRECTLY (i.e. `cell['<key>']`-style), bypassing the
    @measurable resolver. The runner unions them with the
    measurables' transitive reads to decide what to load from
    `traces.parquet`, AND keeps them through the per-corpus drop
    step so the analyses can find them at evaluate time."""
    import inspect as _inspect

    from corroborate._internals.introspection import get_param_default

    out: set[str] = set()
    for b in bridges:
        if b.holds_when is None:
            continue
        try:
            sig = _inspect.signature(b.holds_when)
        except (ValueError, TypeError):
            continue
        for name, param in sig.parameters.items():
            if get_param_default(param) is not _inspect.Parameter.empty:
                continue
            ar = _get_analysis(name)
            if ar is None:
                continue
            out.update(ar.reads)
    return frozenset(out)


def _load_directory(
    root: Path,
    *,
    restore_from_cloud: bool,
    required: Sequence[str],
    bridges: tuple[Bridge, ...],
) -> pl.DataFrame:
    """Walk subdirs of `root`; for each subdir's `runs.parquet`,
    load it, join the trace columns required by:

    - measurables' transitive `reads` (the @measurable resolver
      consumes these at compute time), AND
    - analyses' declared `reads` (consumed directly off the cell
      record by the analysis fn at bridge-evaluate time)

    Then compute the measurables (so the heavy-trace-input ones
    like `bootstrap_fraction` get filled in WHILE traces are in
    scope) and DROP the measurable-only trace columns before
    merging. Columns the analyses declared via `Analysis.reads`
    survive the drop — the analyses need them at evaluate time.

    The drop step is load-bearing: per-step trace columns
    (`done`, `online_std_q_per_step`, …) can be GBs per cell, so
    keeping them across the diagonal_relaxed concat would OOM."""
    measurable_reads = _required_record_keys(required)
    analysis_reads = _analysis_reads_for_bridges(bridges)
    trace_reads = measurable_reads | analysis_reads
    frames: list[pl.DataFrame] = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        runs_path = sub / 'runs.parquet'
        traces_path = sub / 'traces.parquet'
        manifest = sub / '_remote.json'
        # Restore from cloud if (a) runs.parquet is missing OR (b)
        # we need traces (any required measurable / analysis read)
        # and traces.parquet is missing/stub locally but listed in
        # the remote manifest. Without (b), corpora with local
        # runs.parquet but cloud-only traces.parquet silently lose
        # their trace-reading measurables to NaN.
        just_restored_traces = False
        if manifest.exists():
            need_restore = _missing_for_restore(
                runs_path, traces_path, trace_reads, manifest,
            )
            if need_restore:
                if restore_from_cloud:
                    from corroborate.corpus.cloud import restore
                    print(
                        f'runner: restoring {sub.name} from cloud '
                        f'({need_restore})...',
                        file=sys.stderr,
                    )
                    restore(sub, files=need_restore, overwrite=True)
                    just_restored_traces = 'traces.parquet' in need_restore
                    # Per-arm shard archives — merge into canonical
                    # traces.parquet so `_join_required_traces` reads
                    # one shape regardless of how the sweep archived.
                    if _merge_shard_traces(sub):
                        just_restored_traces = True
                else:
                    print(
                        f'runner: WARNING — {sub.name} needs '
                        f'{need_restore} from cloud; restore disabled',
                        file=sys.stderr,
                    )
                    if not runs_path.exists():
                        continue
        if not runs_path.exists():
            continue
        df = pl.read_parquet(runs_path)
        runs_columns = set(df.columns)
        df = _join_required_traces(
            df, sub / 'traces.parquet', trace_reads,
        )
        # Compute measurables NOW, while the joined trace columns
        # are in scope. Subsequent calls on the merged frame are
        # no-ops because the scalar columns are already filled.
        df = _compute_measurables(df, required)
        # Drop the heavy trace columns we joined ONLY for
        # measurable computation. Columns analyses declared via
        # `Analysis.reads` survive — those analyses read off the
        # cell record at evaluate time, after the cache load.
        joined_trace_cols = [
            c for c in df.columns
            if c in trace_reads
            and c not in runs_columns
            and c not in analysis_reads
        ]
        if joined_trace_cols:
            df = df.drop(joined_trace_cols)
        if 'corpus' not in df.columns:
            df = df.with_columns(pl.lit(sub.name).alias('corpus'))
        # Disk-evict traces.parquet — but ONLY if we ourselves
        # restored it on this run. Pre-existing local traces are
        # left alone so users with full local copies aren't
        # silently losing data. Total traces in cloud sum to ~60 GB;
        # this keeps peak local-disk usage at one-corpus-worth.
        if just_restored_traces and traces_path.exists():
            try:
                traces_path.unlink()
                print(
                    f'runner: evicted {sub.name}/traces.parquet '
                    f'(restored just-in-time, scalar measurables '
                    f'persisted in cache)',
                    file=sys.stderr,
                )
            except OSError as e:
                print(
                    f'runner: WARNING — could not evict '
                    f'{sub.name}/traces.parquet: {e}',
                    file=sys.stderr,
                )
        frames.append(df)
    if not frames:
        return pl.DataFrame()
    return _dedup_by_content(
        pl.concat(frames, how='diagonal_relaxed'),
        source='loaded directory',
    )


def _merge_shard_traces(corpus_dir: Path) -> bool:
    """Stitch per-arm `tmp/*_traces.parquet` shards into a single
    canonical `traces.parquet` at the corpus root.

    No-op when `traces.parquet` already exists or no shards are
    present. Two-pass row-group-streaming merge:

    1. **Scan schemas** of all shards (cheap — metadata only) and
       compute their union via `pa.unify_schemas`. Sweeps that
       record per-arm-specific columns (e.g. `dampened_alpha_envs`'
       wrapper-tagged columns) produce heterogeneous shards; the
       unified schema null-pads each shard's missing columns.
    2. **Stream-write** each shard's row groups through a
       `ParquetWriter` initialised with the unified schema. Each
       row group is cast (missing columns added as null arrays,
       columns reordered) before write. Shards are `unlink`-ed
       after their final row group is consumed.

    RAM stays bounded to one row group at a time — load-bearing
    for the multi-GB minatar shards. polars' `sink_parquet` of a
    diagonal_relaxed concat materialises the full panel before
    write and OOMs on those.

    Returns True when a merge happened (so the caller can flag the
    merged file for post-load eviction, matching the behaviour for
    a just-restored top-level `traces.parquet`)."""
    dest = corpus_dir / 'traces.parquet'
    if _file_present(dest):
        return False
    shards = sorted((corpus_dir / 'tmp').glob('*_traces.parquet'))
    if not shards:
        return False

    from corroborate._internals.pyarrow_shard_merge import merge_parquet_shards
    merge_parquet_shards(shards, dest)
    print(
        f'runner: merged {len(shards)} trace shard(s) in '
        f'{corpus_dir.name}/ into traces.parquet',
        file=sys.stderr,
    )
    return True


def _join_required_traces(
    runs: pl.DataFrame, traces_path: Path,
    required_reads: frozenset[str],
) -> pl.DataFrame:
    """Left-join the trace columns the required measurables need
    (`required_reads`, intersected with what `traces.parquet`
    actually carries). Both per-burst columns and per-step
    columns flow through the same path; the caller drops them
    after the measurable evaluator runs.

    Tolerates stub / corrupt traces.parquet — some corpora carry
    a 0-byte placeholder."""
    if not traces_path.exists() or 'id' not in runs.columns:
        return runs
    try:
        schema = pl.scan_parquet(traces_path).collect_schema()
    except pl.exceptions.ComputeError as e:
        print(
            f'runner: WARNING — {traces_path.parent.name}/traces.parquet '
            f'unreadable ({e!s}); skipping trace join',
            file=sys.stderr,
        )
        return runs
    available = set(schema.names())
    cols_to_load = ['id'] + sorted(required_reads & available)
    if len(cols_to_load) == 1:
        return runs
    traces = pl.read_parquet(traces_path, columns=cols_to_load)
    overlap = [c for c in traces.columns if c in runs.columns and c != 'id']
    if overlap:
        traces = traces.drop(overlap)
    return runs.join(traces, on='id', how='left')


__all__ = [
    'collect_bridges',
    'run',
]
