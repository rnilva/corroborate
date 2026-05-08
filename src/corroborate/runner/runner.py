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
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import polars as pl

from corroborate._internals.json import loads as _json_loads
from corroborate.bridge.analysis import get_registered as _get_analysis
from corroborate.bridge.bridge import (
    Bridge,
    BridgeEvaluation,
    evaluate,
    measurable_names_for_bridges,
)
from corroborate.core.hypothesis import Hypothesis
from corroborate.corpus.cloud import RemoteManifest
from corroborate.corpus.measurements import DriftReport, check_drift
from corroborate.corpus.schema import LINEAGE_FIELDS
from corroborate.measurables import (
    compute_missing_columns,
    get_registered,
    registered_names,
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


def _default_report_path(h: Hypothesis) -> Path:
    """Per-hypothesis post-run JSON report at
    `experiments/findings/<short>.run.json`. Mirrors the cache-path
    convention so the report sits next to the FINDINGS doc that
    references it. The file is committed alongside the bridges
    (audit baseline)."""
    short = h.__name__.split('.')[-1]
    return Path('experiments/findings') / f'{short}.run.json'


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
    parsed = _json_loads(path.read_text())
    if not isinstance(parsed, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in parsed.items():
        if isinstance(k, str) and isinstance(v, str):
            out[k] = v
    return out


# Local aliases — re-export the public atomic-write helpers from
# `corpus/persistence.py` under the underscore-prefixed names this
# module previously used. The originals live in persistence.py so
# both runner-side caches and the new per-corpus
# `measurements.parquet` builder share one implementation; aliasing
# keeps the existing call sites untouched.
from corroborate.corpus.persistence import (
    atomic_write_parquet as _atomic_write_parquet,
    atomic_write_text as _atomic_write_text,
)


def _write_manifest(path: Path, sigs: Mapping[str, str]) -> None:
    _atomic_write_text(
        path, json.dumps(dict(sigs), indent=2, sort_keys=True),
    )


def _invalidate_drifted(
    cache: pl.DataFrame,
    manifest: Mapping[str, str],
    required: Sequence[str],
) -> pl.DataFrame:
    """Drop columns whose stored signature doesn't match the
    current closure hash AND drop orphans (registered measurables
    that are no longer required by the current bridge set).

    **Two-way drift detection** (CACHE_BUILD.md C4):
    - Drifted: column IS required, but its closure hash differs
      from the manifest. Recomputed on the next pass.
    - Orphan: column is a registered measurable but the current
      bridges no longer ask for it. Persists forever pre-fix —
      growing the cache, slowing every read, eventually carrying
      values computed by code that may have been deleted.

    Preserves: provenance / lineage tags (id, arm_key, env_name,
    etc. — captured by `LINEAGE_FIELDS`), raw record columns
    (anything not in the measurable registry), and analysis-side
    `.reads` columns (those aren't measurables and aren't
    registered as such).

    Loud warning lists what was dropped on each axis so the user
    knows where any rebuild time went."""
    drifted: list[str] = []
    for name in required:
        if name not in cache.columns:
            continue
        current = _measurable_signature(name)
        stored = manifest.get(name)
        if current is None:
            # Anomalous: column is in `required` (so the bridge
            # graph asked for it) but the registry doesn't know it.
            # Either the registry was mutated mid-loop or a
            # required name was never registered. If we have a
            # stored hash, drop the column — the conservative move
            # when drift coverage is unavailable. Mirror of the
            # roast-#4 fix in `corpus/measurements.py`.
            if stored is not None:
                drifted.append(name)
            continue
        if stored is not None and stored != current:
            drifted.append(name)
    # Orphan detection: registered measurables in cache.columns
    # that aren't in `required`. Non-registered columns (raw
    # record fields, lineage tags) are NEVER orphan candidates.
    required_set = set(required)
    all_registered = set(registered_names())
    orphans = sorted(
        c for c in cache.columns
        if c in all_registered and c not in required_set
    )
    if not drifted and not orphans:
        return cache
    if drifted:
        print(
            f'runner: invalidating {len(drifted)} drifted measurable '
            f'column(s): {drifted}',
            file=sys.stderr,
        )
    if orphans:
        print(
            f'runner: dropping {len(orphans)} orphan measurable '
            f'column(s) (registered but not required by current '
            f'bridges): {orphans}',
            file=sys.stderr,
        )
    to_drop = drifted + orphans
    return cache.drop(to_drop)


# ============ Public surface ============


def run(
    h: Hypothesis | str,
    *,
    data: pl.DataFrame | Path | str | Sequence[Path] | None = None,
    use_cache: bool = True,
    write_cache: bool = True,
    rebuild: bool = False,
    restore_from_cloud: bool = True,
    cache_path: Path | None = None,
    report_path: Path | None = None,
    write_report: bool = False,
    bridge_filter: str | None = None,
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

    `report_path` / `write_report`: post-run audit report. Default
    is `write_report=False` (library use shouldn't silently mutate
    `experiments/findings/`); the CLI in `scripts/run_hypothesis.py`
    flips this to True so committed-baseline regeneration is the
    default for `run_hypothesis.py` invocations. When enabled,
    serializes a JSON report at `report_path` (or
    `experiments/findings/<short>.run.json` if None) capturing
    per-bridge verdict + every typed analysis result + admission
    gates + provenance (timestamp, git commit, measurable
    signatures). The report is the load-bearing audit artifact —
    small, diffable, deterministic, committed alongside the
    bridges that produced it. See `corroborate.runner.report`.

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

    # Optional bridge-name substring filter (pytest's `-k` shape).
    # When set, only bridges whose name contains the pattern run;
    # measurable computation downstream uses just THIS subset's
    # required-measurable union — much faster iteration during
    # debugging single bridges. When empty filter matches no
    # bridge, fail loud rather than silently running zero
    # bridges.
    if bridge_filter is not None:
        matched = tuple(b for b in bridges if bridge_filter in b.name)
        if not matched:
            raise SystemExit(
                f'{h.__name__}: bridge_filter {bridge_filter!r} '
                f'matched no bridges. Available: '
                f'{sorted(b.name for b in bridges)[:5]}{" ..." if len(bridges) > 5 else ""}',
            )
        bridges = matched

    resolved_cache: Path | None = None
    if use_cache:
        resolved_cache = cache_path if cache_path is not None else _default_cache_path(h)
        resolved_cache.parent.mkdir(parents=True, exist_ok=True)
        if rebuild:
            resolved_cache.unlink(missing_ok=True)
            _manifest_path(resolved_cache).unlink(missing_ok=True)

    # CACHE_ADDITIVITY.md CA2: when running cache-only (data=None
    # AND a cache file already exists), print a one-line state
    # so the user sees what they're about to evaluate against.
    # Skip when ingest is happening (the ingest path emits its
    # own progress lines).
    if (
        data is None
        and resolved_cache is not None
        and resolved_cache.exists()
    ):
        try:
            from datetime import UTC, datetime
            schema = pl.scan_parquet(resolved_cache).collect_schema()
            n_cells = cast(int, pl.scan_parquet(resolved_cache).select(
                pl.len(),
            ).collect().item())
            n_cols = len(schema.names())
            mtime = datetime.fromtimestamp(
                resolved_cache.stat().st_mtime, UTC,
            ).isoformat(timespec='seconds')
            print(
                f'cache: {n_cells} cells × {n_cols} cols, '
                f'last updated {mtime}',
                file=sys.stderr, flush=True,
            )
        except (OSError, pl.exceptions.ComputeError):
            # Cache unreadable — `_load_cache` will produce a
            # warning shortly; don't double-warn here.
            pass

    cells = _ingest_and_compute(
        bridges=bridges,
        data=data,
        cache_path=resolved_cache,
        write_cache=write_cache and use_cache,
        restore_from_cloud=restore_from_cloud,
    )

    if cells.height == 0:
        raise SystemExit(
            f'{h.__name__}: no cells available — pass '
            f'--ingest <corpus> or --ingest-all <root> to ingest '
            f'data, or check the cache at {resolved_cache}',
        )

    # Optional substrate-side outermost claim for endogeneity
    # gating. Threaded to `evaluate` via the kw-only `claim`
    # parameter; gates that need it (exogenous_source /
    # exogenous_scope) close over the leaf set, gates that don't
    # ignore it. Hypotheses without CLAIM fall back to None →
    # endogeneity gates short-circuit.
    claim = getattr(h, 'CLAIM', None)

    # Optional hypothesis-module-level scope filter. AND-combined
    # with each bridge's own `scope=` inside `evaluate`. Used to
    # encode universe-level exclusions (e.g., "this file's
    # cross-env analyses exclude bsuite diagnostic envs").
    # Hypotheses without MODULE_SCOPE pass None → bridge.scope
    # alone determines the filter.
    module_scope = getattr(h, 'MODULE_SCOPE', None)

    out: dict[str, BridgeEvaluation] = {}
    errors: dict[str, BaseException] = {}
    for b in bridges:
        try:
            out[b.name] = evaluate(
                b, cells, claim=claim, module_scope=module_scope,
            )
        except Exception as e:  # noqa: BLE001
            # An authoring bug in a bridge's `holds_when` body
            # (e.g. typo'd column name, malformed analysis call)
            # raises here. Print the error to stderr and skip
            # this bridge — DO NOT synthesise a fake verdict.
            # Conflating evaluation errors with the POWER_INSUFFICIENT
            # verdict would smuggle authoring bugs past the reader,
            # exactly what the framework's verdict layer refuses
            # (see verdict.Verdict docstring + CLAUDE.md §verdict).
            # The exception is captured for the post-run report so
            # the audit trail surfaces what stderr would lose.
            errors[b.name] = e
            print(
                f'  [bridge {b.name!r} raised during evaluation: '
                f'{type(e).__name__}: {e}]',
                file=sys.stderr,
            )

    if write_report:
        from corroborate.runner.report import (
            build_report as _build_report,
            write_report as _write_report,
        )
        resolved_report = (
            report_path if report_path is not None
            else _default_report_path(h)
        )
        resolved_report.parent.mkdir(parents=True, exist_ok=True)
        required_for_sigs = sorted(measurable_names_for_bridges(bridges))
        sigs = _signatures_for(required_for_sigs, cells)
        report = _build_report(
            hypothesis_module_name=h.__name__,
            bridges=bridges,
            results=out,
            errors=errors,
            n_cells_total=cells.height,
            cache_path=resolved_cache,
            measurable_signatures=sigs,
            # `repo_root=None` lets `build_report` walk up from the
            # report module's own location looking for `.git`. More
            # robust than `Path.cwd()` (notebooks running outside
            # the repo would have lost git_commit silently).
        )
        _write_report(report, resolved_report)
    return out


def check(
    h: Hypothesis | str,
    *,
    root: Path | str = Path('experiments/data'),
) -> DriftReport:
    """**CACHE_ADDITIVITY.md CA5** drift visibility without work.

    Read each corpus's `measurements.hashes.json` sidecar under
    `root` and compare against the current registry's closure
    hashes for `h.BRIDGES`'s required measurables. Reports per-
    corpus drift + missing columns. NOT an analysis run — does
    not load runs.parquet, does not compute, does not touch
    cloud, does not run bridges. Useful for:

    - "Did my substrate edit drift any column?" — yes if drift
      report is non-empty.
    - "Which corpora do I need to `--ingest`?" — affected names
      are listed.

    Designed to be cheap enough to run before every `--ingest`
    decision; the actual cost is a `json.loads` per corpus."""
    if isinstance(h, str):
        h = _validate_hypothesis(importlib.import_module(h))
    else:
        h = _validate_hypothesis(h)
    required = sorted(measurable_names_for_bridges(h.BRIDGES))
    return check_drift(
        Path(root),
        required=required,
        measurable_signature_fn=_measurable_signature,
    )


def evict(
    h: Hypothesis | str,
    corpora: Sequence[str],
    *,
    cache_path: Path | str | None = None,
) -> tuple[int, dict[str, int]]:
    """Filter the per-hypothesis cache parquet to drop all rows
    whose `corpus` column matches any name in `corpora`. The
    per-corpus `measurements.parquet` stores under
    `experiments/data/<corpus>/` are NOT touched — this is a
    cache-only eviction, useful when a corpus's analyses should
    be excluded temporarily without losing the underlying data.

    Returns `(total_dropped, per_corpus_counts)` where
    `per_corpus_counts` maps each requested corpus name to the
    number of cells dropped. A name not present in the cache
    contributes 0; the call doesn't raise.

    The eviction survives cache reads but NOT a subsequent
    `--ingest-all` walk: that walk re-projects every per-corpus
    store under `experiments/data/`, including ones whose data
    you just evicted from the cache. To prevent re-inclusion,
    delete the corpus directory or add an `.in_progress` sentinel.
    """
    if isinstance(h, str):
        h = _validate_hypothesis(importlib.import_module(h))
    else:
        h = _validate_hypothesis(h)
    cp = (
        Path(cache_path) if cache_path is not None
        else _default_cache_path(h)
    )
    if not cp.exists():
        return 0, dict.fromkeys(corpora, 0)
    df = pl.read_parquet(cp)
    if 'corpus' not in df.columns:
        # Pre-corpus-stamping cache (very legacy); nothing to filter.
        return 0, dict.fromkeys(corpora, 0)
    counts: dict[str, int] = {}
    for c in corpora:
        counts[c] = int(
            df.filter(pl.col('corpus') == c).height,
        )
    total = sum(counts.values())
    if total == 0:
        return 0, counts
    keep = df.filter(~pl.col('corpus').is_in(list(corpora)))
    if keep.height > 0:
        _atomic_write_parquet(keep, cp)
    else:
        # Avoid writing a 0-row parquet (polars rejects some schemas);
        # delete the cache file when nothing remains.
        cp.unlink()
    return total, counts


# ============ Cache + ingest ============


def _ingest_and_compute(
    *,
    bridges: tuple[Bridge, ...],
    data: pl.DataFrame | Path | str | Sequence[Path] | None,
    cache_path: Path | None,
    write_cache: bool,
    restore_from_cloud: bool,
) -> pl.DataFrame:
    """Resolve `data` into the per-hypothesis cache.

    **Phase 2.2** (CACHE_BUILD.md): when `data` is a directory,
    per-corpus `measurements.parquet` stores are the source of
    truth. The directory walk's output (each corpus's runs joined
    with its measurements) IS the projection — no cache-side
    merge, drift check, or per-cell recompute. The merged cache
    parquet is written as a backward-compat snapshot for callers
    that read it directly; `<cache>.hashes.json` is unlinked
    (per-corpus sidecars are authoritative).

    Legacy DataFrame/file path keeps the old shape — load cache,
    drift-invalidate against `<cache>.hashes.json`, dedup new vs
    cache, concat — for incremental adds where `data` is a single
    parquet or DataFrame to be merged into an existing cache.
    Tests exercise this path directly; substrate code paths a
    directory."""
    required = sorted(measurable_names_for_bridges(bridges))

    is_directory_walk: bool
    if data is None or isinstance(data, pl.DataFrame):
        is_directory_walk = False
    elif isinstance(data, (Path, str)):
        # `--ingest-all <root>` or `--ingest-file <path.parquet>`
        is_directory_walk = Path(data).is_dir()
    else:
        is_directory_walk = True   # named-corpora ingest (CA3)

    new_data = _load_data(
        data, restore_from_cloud=restore_from_cloud,
        required=required, bridges=bridges,
    )

    if is_directory_walk:
        # Phase 2.2: per-corpus stores already filled in measurables
        # via Phase 2.1's `build_measurements` call inside
        # `_load_one_corpus`. The walk output IS the projection.
        merged = (
            new_data if new_data is not None else pl.DataFrame()
        )
        if cache_path is not None and write_cache:
            # Closure-hash sidecar is no longer authoritative;
            # per-corpus `measurements.hashes.json` files are. Unlink
            # the legacy manifest UNCONDITIONALLY on every
            # directory-path write so a stale snapshot can't survive
            # an empty-corpora walk and mislead any reader still
            # consulting it (post-#5 roast fix: the unlink was
            # previously gated on `merged.height > 0`).
            legacy_manifest = _manifest_path(cache_path)
            if legacy_manifest.exists():
                try:
                    legacy_manifest.unlink()
                except OSError:
                    pass
            # Cache parquet only written when there's actual data —
            # polars rejects zero-row writes for some schemas, and a
            # stale snapshot with no rows isn't more useful than an
            # absent file.
            if merged.height > 0:
                _atomic_write_parquet(merged, cache_path)
        return merged

    # Legacy DataFrame/file path — incremental cache merge.
    manifest_path = (
        _manifest_path(cache_path) if cache_path is not None else None
    )
    stored_manifest = (
        _read_manifest(manifest_path) if manifest_path is not None else {}
    )
    cache = _load_cache(cache_path)
    cache = _invalidate_drifted(cache, stored_manifest, required)

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
        _atomic_write_parquet(merged, cache_path)
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
    columns changed (added / drifted).

    **Trace-dependent gap diagnostic**: required measurables whose
    transitive `reads` aren't all in the cache (typical when a new
    measurable reading per-step trace columns is added — traces
    are evicted post-ingest per CI7) can't be filled in cache-only
    mode. Pre-fix, `compute_missing_columns` would silently produce
    NaN + emit a cryptic `KeyError on ALL N cells` warning. Now we
    surface the actionable instruction: "rerun with --ingest-all to
    fill in" — distinguishes the no-data-needed case (HP-only
    measurables, fillable from cache) from the needs-trace-restore
    case."""
    if cache.height == 0:
        return cache
    _warn_trace_dep_unfillable_in_cache(cache, required)
    enriched = _compute_measurables(cache, required)
    if (
        cache_path is not None
        and write_cache
        and enriched.columns != cache.columns
    ):
        _atomic_write_parquet(enriched, cache_path)
        if manifest_path is not None:
            _write_manifest(
                manifest_path, _signatures_for(required, enriched),
            )
    return enriched


def _warn_trace_dep_unfillable_in_cache(
    cache: pl.DataFrame, required: Sequence[str],
) -> None:
    """For each required measurable not yet present in the cache,
    check whether its transitive reads are satisfied by the cache's
    columns. Print a single actionable message naming all the
    measurables that need a `--ingest-all` walk to be filled —
    rather than the per-measurable cryptic `KeyError on ALL N
    cells` warnings that compute_missing_columns emits later."""
    from corroborate.measurables.measurable import (
        get_registered, transitive_reads,
    )
    cache_cols = set(cache.columns)
    needs_ingest: list[str] = []
    for name in required:
        if name in cache_cols:
            continue
        m = get_registered(name)
        if m is None:
            continue
        try:
            reads = transitive_reads(name)
        except KeyError:
            continue
        if not all(r in cache_cols for r in reads):
            needs_ingest.append(name)
    if needs_ingest:
        import sys
        sys.stderr.write(
            f'runner: WARNING — {len(needs_ingest)} required '
            f"measurable(s) can't be filled in cache-only mode "
            f'(transitive reads not in cache, typically per-step '
            f'or per-burst trace columns). The cache will be '
            f'written with NaN for these columns and bridges '
            f'scope-filtering on them will silently drop cells:\n'
            + '\n'.join(f'  - {n}' for n in needs_ingest)
            + '\n  → rerun with `--ingest-all experiments/data/` '
            f'to restore traces and fill these properly.\n',
        )


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
#
# `env` is the env-object's `__repr__` string (e.g.
# `<gymnax.environments.misc.rooms.FourRooms object at 0x77d49bbb2ea0>`)
# — different memory addresses across separate ingests of the same
# scientific cell, so it acts as accidental run-time noise rather
# than a scientific-identity column. `env_name` is the real
# identifier; treat `env` as provenance so dedup-by-content works
# across ingest runs.
_PROVENANCE_TAGS: frozenset[str] = LINEAGE_FIELDS | {'corpus', 'env'}


_OBJECT_REPR_PATTERN = re.compile(r'<.+\sobject\sat\s0x[0-9a-f]+>')


def _volatile_object_repr_columns(df: pl.DataFrame) -> list[str]:
    """**CORPUS_INTEGRITY.md CI4** dynamic complement to
    `_PROVENANCE_TAGS`. Returns string columns whose first
    non-null value matches the Python-object-repr pattern
    `<...\\sobject\\sat\\s0x[0-9a-f]+>` (e.g. `<gymnax.envs.…
    object at 0x77d49bbb2ea0>`). The hex memory address is
    process-volatile — different across re-ingests of the same
    scientific cell — so it acts as accidental run-time noise
    rather than a content-identity column.

    `env` is the historical canonical case (already hardcoded
    in `_PROVENANCE_TAGS`); this dynamic detection generalizes
    so a future substrate column carrying e.g. `<Claim:…>`
    reprs gets caught the same way without a manual hardcode.
    Columns already in `_PROVENANCE_TAGS` are skipped to avoid
    redundant work."""
    out: list[str] = []
    for col in df.columns:
        if col in _PROVENANCE_TAGS:
            continue
        if df.schema[col] != pl.String:
            continue
        # Probe the first non-null value. Cheaper than scanning
        # the whole column and just as accurate — the repr
        # pattern is structural, not per-row.
        s = df[col].drop_nulls()
        if s.len() == 0:
            continue
        # `polars.Series[String][0]` returns `Any` in the stubs;
        # narrow via cast(object) so the `isinstance` check
        # actually narrows.
        v = cast(object, s[0])
        if isinstance(v, str) and _OBJECT_REPR_PATTERN.match(v):
            out.append(col)
    return out


def _dedup_by_content(df: pl.DataFrame, *, source: str) -> pl.DataFrame:
    """Drop rows whose non-provenance columns are all equal — i.e.
    cells that differ only in `id`/`corpus`/`timestamp` / lineage
    tags. The merge artifacts (same physical run surfaced under two
    `corpus` tags) collapse to one row; truly distinct runs are
    preserved (their measurement columns differ).

    Polars' `unique(subset=...)` handles primitive columns natively;
    list/object columns get coerced via `hash` first so the equality
    check is value-based even on heterogeneous shapes.

    **CI4** (CORPUS_INTEGRITY.md): in addition to static
    `_PROVENANCE_TAGS`, dynamically excludes string columns
    carrying Python object reprs (process-volatile memory
    addresses). Without this, two re-ingests of the same
    scientific cell with different `<…\\sobject\\sat\\s0x…>`
    addresses look like distinct rows and survive dedup —
    inflating sample counts."""
    if df.height == 0:
        return df
    volatile = _volatile_object_repr_columns(df)
    excluded = _PROVENANCE_TAGS | set(volatile)
    content_cols = [c for c in df.columns if c not in excluded]
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
    # Gate the parquet read on the same integrity check used for
    # restored trace files (`_file_present` validates min size +
    # PAR1 magic footer). The runner writes `merged.write_parquet`
    # non-atomically, so a killed-mid-write cache leaves a
    # truncated file on disk; rebuilding from scratch beats
    # crashing with a polars ComputeError on every subsequent run.
    if not _file_present(path):
        sys.stderr.write(
            f'WARNING: cache at {path} is present but truncated / '
            f'invalid parquet; treating as missing. Re-run will '
            f'rebuild from scratch.\n',
        )
        return pl.DataFrame()
    try:
        df = pl.read_parquet(path)
    except (pl.exceptions.ComputeError, OSError) as e:
        sys.stderr.write(
            f'WARNING: cache at {path} could not be read '
            f'({type(e).__name__}: {e}); treating as missing.\n',
        )
        return pl.DataFrame()
    return _dedup_by_content(df, source='cache')


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
    data: pl.DataFrame | Path | str | Sequence[Path] | None,
    *,
    restore_from_cloud: bool,
    required: Sequence[str],
    bridges: tuple[Bridge, ...],
) -> pl.DataFrame | None:
    """Resolve data into a DataFrame, with auto-restore on missing-
    raw corpora when given a directory.

    Five shapes accepted (CACHE_ADDITIVITY.md CA1-CA3):
    - `None`: cache-only mode, caller skips ingest entirely.
    - `pl.DataFrame`: in-memory cells (substrate-side build).
    - `Path` to a single `.parquet` file: read directly.
    - `Path` to a directory: walk all corpus subdirs (full
      `--ingest-all` behavior).
    - `Sequence[Path]`: named corpus dirs to ingest selectively
      (`--ingest <name>[,<name>...]`). The disk-budget calculation
      uses the first dir's parent as the volume reference.
    """
    if data is None:
        return None
    if isinstance(data, pl.DataFrame):
        return data
    if not isinstance(data, (Path, str)):
        # Sequence[Path] — named-corpora ingest. Each entry must
        # be a directory at this point (CLI resolves names; tests
        # pass Paths directly).
        corpus_paths = tuple(Path(p) for p in data)
        if not corpus_paths:
            return None
        for cp in corpus_paths:
            if not cp.is_dir():
                raise FileNotFoundError(
                    f'--ingest dir not found or not a directory: {cp}',
                )
        # Use the first corpus's parent as the disk-budget root.
        # All named corpora share a filesystem in practice.
        root = corpus_paths[0].parent
        return _load_directory(
            root, restore_from_cloud=restore_from_cloud,
            required=required, bridges=bridges,
            corpus_dirs=corpus_paths,
        )
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
        raw = _json_loads(manifest_path.read_text())
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


def _estimate_max_workers(
    sub_dirs: list[Path],
    root: Path,
    *,
    safety_factor: float = 4.0,
    hard_cap: int = 4,
) -> int:
    """Bound the worker count so K workers each holding one
    corpus's traces.parquet on disk fits within the local
    storage budget.

    Reads each manifest's `size_bytes` for any `traces.parquet`
    entry (or `tmp/*_traces.parquet` shards if that's how the
    sweep archived). Takes the MAX as the per-worker budget.
    Available disk = `shutil.disk_usage(root).free` minus a
    `safety_factor` headroom (default 4× — three of the four
    parts go to the workers, one part remains free for other
    in-flight writes / OS overhead).

    Returns at least 1, at most `hard_cap` (default 4 — fork
    overhead per worker is non-trivial and beyond ~4 workers
    Python's GIL within `_compute_measurables` becomes the
    bottleneck rather than I/O).

    Falls back to `hard_cap` when no manifests exist (no traces
    to budget; cache build is CPU-bound — full parallelism).
    Falls back to 1 when available disk can't fit even one
    worker's largest trace (e.g. minatar 1M shards exceed the
    overlay size). The user can override via env var
    CORROBORATE_CACHE_WORKERS to force a specific value."""
    import os
    import shutil
    forced = os.environ.get('CORROBORATE_CACHE_WORKERS')
    if forced is not None:
        try:
            return max(1, int(forced))
        except ValueError:
            pass

    largest_trace_bytes = 0
    for sub in sub_dirs:
        m_path = sub / '_remote.json'
        if not m_path.exists():
            continue
        manifest = _read_remote_manifest(m_path)
        if manifest is None:
            continue
        for f in manifest.files:
            # Match any traces parquet — top-level or per-arm
            # shard. tmp/cell{NNN}__{tag}__traces.parquet style
            # included.
            if f.relpath.endswith('.parquet') and 'traces' in f.relpath:
                if f.size_bytes > largest_trace_bytes:
                    largest_trace_bytes = f.size_bytes

    if largest_trace_bytes == 0:
        return hard_cap

    try:
        avail = shutil.disk_usage(root).free
    except OSError:
        return 1
    safe_avail = avail / safety_factor
    by_disk = int(safe_avail / largest_trace_bytes)
    return max(1, min(hard_cap, by_disk))


def _trace_is_cloud_recoverable(
    corpus_dir: Path, traces_path: Path,
) -> bool:
    """**CORPUS_INTEGRITY.md CI7** helper. Returns True iff the
    local `traces_path` can be re-restored from cloud:
    - `_remote.json` exists at `corpus_dir`,
    - manifest lists `traces.parquet`,
    - manifest's sha256 matches the local file's sha256.

    Equality of sha256 is the integrity signal. Without it, a
    local file matching size only could be drift (e.g., a partial
    recovery with the same byte count but different content);
    eviction would silently lose the local-canonical version.

    sha256 of a multi-GB file is ~5 sec on modern hardware —
    cheap relative to the disk pressure we save by evicting."""
    manifest_path = corpus_dir / '_remote.json'
    if not manifest_path.exists():
        return False
    manifest = _read_remote_manifest(manifest_path)
    if manifest is None:
        return False
    entry = next(
        (f for f in manifest.files if f.relpath == 'traces.parquet'),
        None,
    )
    if entry is None:
        return False
    try:
        from corroborate.corpus.cloud import _sha256_file
        local_sha = _sha256_file(traces_path)
    except OSError:
        return False
    return local_sha == entry.sha256


def _try_unlink(
    path: Path, log_lines: list[str], prefix: str,
) -> bool:
    """Attempt to evict `path`; record any error in `log_lines`.
    Returns True on success."""
    try:
        path.unlink()
        return True
    except OSError as e:
        log_lines.append(
            f'{prefix}: WARNING — could not evict '
            f'{path.name}: {e}',
        )
        return False


def _measurements_sidecar_current(
    sub: Path, required: Sequence[str],
) -> bool:
    """True iff `measurements.parquet` exists and the sidecar's
    closure-hash for every required measurable matches the current
    registry. Used by `_load_one_corpus` to skip cloud restore +
    trace join when there's demonstrably nothing to recompute.

    Hash-match is the contract: a value is current if its closure
    matches what produced it. NaN cells in a hash-current store
    represent "framework already computed that and got NaN with
    the inputs available at compute time" — re-running with the
    same inputs would just produce NaN again. Users who want to
    retry after restoring missing inputs (e.g. fresh traces) can
    `rm <corpus>/measurements.parquet` to invalidate the sidecar.

    Mirrors the per-column check in `corpus.measurements:
    check_drift` (any drift / any missing → False). Pure read;
    no side effects."""
    from corroborate.corpus.measurements import (
        MEASUREMENTS_FILENAME, current_signatures,
    )
    if not (sub / MEASUREMENTS_FILENAME).exists():
        return False
    stored = current_signatures(sub)
    if not stored:
        return False
    for name in required:
        live = _measurable_signature(name)
        if live is None:
            # Substrate doesn't define this measurable — it'll be
            # null-padded at projection. Don't gate restore on it.
            continue
        if stored.get(name) != live:
            return False
    return True


def _load_one_corpus(
    sub: Path,
    *,
    i: int,
    n_total: int,
    digit_width: int,
    restore_from_cloud: bool,
    required: Sequence[str],
    trace_reads: frozenset[str],
    analysis_reads: frozenset[str],
) -> tuple[pl.DataFrame | None, list[str]]:
    """Per-corpus pipeline: restore + load + join traces +
    compute measurables + drop trace cols + evict.

    Returns (DataFrame, log_lines) where DataFrame is None if
    the corpus was skipped. Log lines are returned (not printed)
    so the parent process can interleave them in input order
    when running in parallel.

    Each call holds at most one corpus's traces.parquet on disk
    — the worker evicts before returning. Memory budget is one
    corpus's runs+traces in scope plus the typed-trace projection.
    """
    import time as _time
    log_lines: list[str] = []
    t_corpus = _time.monotonic()
    runs_path = sub / 'runs.parquet'
    traces_path = sub / 'traces.parquet'
    manifest = sub / '_remote.json'
    just_restored_traces = False
    prefix = f'  [{i+1:>{digit_width}}/{n_total}] {sub.name}'
    # **CACHE_ADDITIVITY.md fast-path**: if every required measurable
    # has a current closure-hash sidecar entry AND
    # `measurements.parquet` exists, `build_measurements` will
    # idempotent-skip — so traces aren't actually needed. Avoid the
    # cloud restore (~7-30s/corpus) by force-skipping when
    # measurements are demonstrably current. This turns a
    # full-walk re-run from N×restore-time into N×idempotent-skip.
    measurements_current = _measurements_sidecar_current(sub, required)
    if manifest.exists() and not measurements_current:
        need_restore = _missing_for_restore(
            runs_path, traces_path, trace_reads, manifest,
        )
        if need_restore:
            if restore_from_cloud:
                from corroborate.corpus.cloud import restore
                log_lines.append(
                    f'{prefix}: restoring '
                    f'{[Path(p).name for p in need_restore]}...',
                )
                restore(sub, files=need_restore, overwrite=True)
                just_restored_traces = (
                    'traces.parquet' in need_restore
                )
            else:
                log_lines.append(
                    f'{prefix}: WARNING — needs '
                    f'{need_restore} from cloud; restore disabled',
                )
                if not runs_path.exists():
                    return None, log_lines
    # Stitch per-cell trace shards (`tmp/*_traces.parquet`) into a
    # single `traces.parquet` regardless of cloud-archive presence.
    # Pre-fix this only fired inside the `manifest.exists()` block,
    # so local-only sweeps whose merge step never completed (only
    # tmp/ shards present) silently fell back to a no-traces ingest
    # — the dependent measurables stamped NaN and the bridge
    # silently dropped those cells. Now we always attempt the
    # stitch when there's no consolidated traces.parquet.
    if _merge_shard_traces(sub):
        just_restored_traces = True
    if not runs_path.exists():
        log_lines.append(f'{prefix}: SKIPPED (no runs.parquet)')
        return None, log_lines
    df = pl.read_parquet(runs_path)
    runs_columns = set(df.columns)
    # **CORPUS_INTEGRITY.md CI8**: refuse a contaminated
    # `traces.parquet` (cloud-collision residue from pre-CI3 era
    # that overwrote distinct corpora's archives at the same
    # remote_root). The runner falls back to "no cloud traces"
    # for the offending corpus rather than producing all-null
    # join output — honest partial-coverage instead of silent
    # nullification.
    from corroborate.corpus.integrity import (
        TraceContaminationError, assert_traces_subset_of_runs,
    )
    try:
        assert_traces_subset_of_runs(sub)
    except TraceContaminationError as e:
        log_lines.append(
            f'{prefix}: WARNING — CI8 contamination detected; '
            f'skipping trace join '
            f'(spurious={e.stats.spurious_count}, '
            f'overlap={e.stats.overlap_count}, '
            f'runs={e.stats.runs_count}). Trace-dependent '
            f'measurables will be null for this corpus.',
        )
        # Evict the bogus local traces.parquet so the next
        # restore-from-cloud doesn't keep re-validating the same
        # contamination. The cloud copy stays untouched (cleanup
        # is a separate manual step — manifest fix).
        if just_restored_traces:
            try:
                (sub / 'traces.parquet').unlink(missing_ok=True)
                just_restored_traces = False
            except OSError:
                pass
    else:
        df = _join_required_traces(
            df, sub / 'traces.parquet', trace_reads,
        )
    # **Phase 2.1** (CACHE_BUILD.md): route per-cell measurable
    # computation through `build_measurements` so the per-corpus
    # `measurements.parquet` store is populated as a side effect.
    # The store carries `id` + measurable cols only; trace cols
    # are dropped at persistence boundary (see `build_measurements`
    # body). We then `load_measurements` back to keep the
    # downstream shape (runs + traces + measurables) unchanged.
    if 'id' in df.columns:
        from corroborate.corpus.measurements import (
            build_measurements,
            load_measurements,
        )
        # **CACHE_ADDITIVITY.md fast-path** (cont.): when the sidecar
        # is current we already short-circuited the cloud restore.
        # Now also skip `build_measurements` itself — runs.parquet
        # may carry sweep-time NaN-stamped trace-dependent
        # measurables that would trip `compute_missing_columns`'s
        # partial-nullity branch into recomputing without traces
        # (since restore was skipped), silently overwriting the
        # previously-finite values in measurements.parquet with
        # NaN. The per-corpus store is the source of truth in this
        # branch — load it directly into df.
        if not measurements_current:
            build_measurements(
                sub, required=required, runs_df=df,
                measurable_signature_fn=_measurable_signature,
            )
        loaded = load_measurements(sub, columns=list(required))
        present_required = [c for c in required if c in loaded.columns]
        if present_required:
            # Drop any pre-existing required-measurable columns on
            # df (rare — only when the input runs.parquet already
            # carried them) to avoid join-side collisions.
            collide = [c for c in present_required if c in df.columns]
            if collide:
                df = df.drop(collide)
            df = df.join(
                loaded.select(['id', *present_required]),
                on='id', how='left',
            )
    else:
        # Edge case: legacy runs.parquet without `id` column.
        # Fall back to the inline path so older corpora still work.
        df = _compute_measurables(df, required)
    # Drop trace-derived intermediates (per-step / per-burst arrays
    # joined from traces.parquet to feed measurable evaluation). Don't
    # drop columns that are themselves in `required` — those ARE the
    # bridge-consumed values, even if they happen to also appear in
    # `trace_reads` because OTHER measurables list them as a `reads`
    # dependency (e.g. `jensen_dormancy_gap.reads=('jensen_gap',)` —
    # `jensen_gap` is both a required scalar AND an intermediate).
    required_set = set(required)
    joined_trace_cols = [
        c for c in df.columns
        if c in trace_reads
        and c not in runs_columns
        and c not in analysis_reads
        and c not in required_set
    ]
    if joined_trace_cols:
        df = df.drop(joined_trace_cols)
    if 'corpus' not in df.columns:
        df = df.with_columns(pl.lit(sub.name).alias('corpus'))
    # **CORPUS_INTEGRITY.md CI7**: evict locally-cached trace
    # files when they're cloud-recoverable, not just when
    # downloaded THIS session. Pre-fix, pre-existing local
    # traces accumulated forever (`ddqn_better_hp` 3.4 GB,
    # `fourrooms_1m` 3.2 GB, three `polyak_tau_intervention_*`
    # at 2.7 GB each — 14 GB unreclaimed). Subsequent rebuilds
    # held the 14 GB while restoring more, hitting disk-full
    # at corpus 22.
    #
    # The cloud-recoverable check: manifest exists AND lists
    # traces.parquet AND its sha256 matches the local file's.
    # Skip eviction for local-only traces (no manifest = no
    # recovery path; deletion would lose data permanently).
    evicted = False
    if traces_path.exists():
        if just_restored_traces:
            evicted = _try_unlink(traces_path, log_lines, prefix)
        elif _trace_is_cloud_recoverable(sub, traces_path):
            evicted = _try_unlink(traces_path, log_lines, prefix)
    elapsed = _time.monotonic() - t_corpus
    log_lines.append(
        f'{prefix}: {df.height} cells × {len(df.columns)} cols'
        + (' (traces evicted)' if evicted else '')
        + f' in {elapsed:.1f}s',
    )
    return df, log_lines


def _load_directory(
    root: Path,
    *,
    restore_from_cloud: bool,
    required: Sequence[str],
    bridges: tuple[Bridge, ...],
    corpus_dirs: Sequence[Path] | None = None,
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
    keeping them across the diagonal_relaxed concat would OOM.

    `corpus_dirs` (CACHE_ADDITIVITY.md CA3): when provided, walk
    only those named corpus dirs instead of `root.iterdir()`. CI1
    fires per-corpus rather than across the root. The
    disk-budget calculation still uses `root` as the volume
    reference (the named dirs share a filesystem)."""
    # CORPUS_INTEGRITY.md CI1: refuse nested corpora at ingest
    # rather than silently drop the inner ones (the runner walks
    # one level deep). Caller fixes the layout, then retries.
    # Corpus dirs marked with `.in_progress` (sweep mid-flight)
    # are skipped by both the audit and the walk.
    from corroborate.corpus.integrity import (
        IN_PROGRESS_SENTINEL,
        assert_named_corpora_no_nested,
        assert_no_nested_corpora,
        is_in_progress,
    )
    if corpus_dirs is None:
        assert_no_nested_corpora(root)
        sub_dirs_all = sorted(p for p in root.iterdir() if p.is_dir())
    else:
        assert_named_corpora_no_nested(corpus_dirs)
        sub_dirs_all = list(corpus_dirs)
    import time as _time
    measurable_reads = _required_record_keys(required)
    analysis_reads = _analysis_reads_for_bridges(bridges)
    trace_reads = measurable_reads | analysis_reads
    in_progress_dirs = [p for p in sub_dirs_all if is_in_progress(p)]
    if in_progress_dirs:
        for p in in_progress_dirs:
            print(
                f'runner: SKIPPING {p.name}/ — '
                f'`{IN_PROGRESS_SENTINEL}` sentinel present '
                f'(sweep mid-flight)',
                file=sys.stderr, flush=True,
            )
    sub_dirs = [p for p in sub_dirs_all if not is_in_progress(p)]
    n_total = len(sub_dirs)
    if n_total == 0:
        print(
            f'runner: NO subdirs under {root}', file=sys.stderr,
            flush=True,
        )
        return pl.DataFrame()
    # **Per-corpus parallelism** (CACHE_BUILD.md Phase 0 #5):
    # bound K workers by available local disk. Each worker holds
    # one corpus's traces.parquet on disk during compute; K × max-
    # trace-size must fit. `_estimate_max_workers` reads each
    # manifest's `size_bytes` for traces, takes the max as the
    # per-worker budget, divides into available disk minus a
    # safety headroom.
    max_workers = _estimate_max_workers(sub_dirs, root)
    print(
        f'runner: ingesting {n_total} corpora from {root} '
        f'(parallelism: {max_workers} worker{"s" if max_workers > 1 else ""})',
        file=sys.stderr,
        flush=True,
    )
    t_walk_start = _time.monotonic()
    digit_width = len(str(n_total))

    # Worker results map subdir name → (frame_or_None, log_lines)
    # so the parent can print log lines in completion order while
    # still emitting the canonical `[i/N] <corpus>` prefix in the
    # input ordering.
    if max_workers == 1:
        # Sequential path — preserves backward-compatible ordering
        # for runs that opt out of parallelism (e.g. tests that
        # capture stderr).
        results: list[tuple[pl.DataFrame | None, list[str]]] = []
        for i, sub in enumerate(sub_dirs):
            results.append(_load_one_corpus(
                sub, i=i, n_total=n_total,
                digit_width=digit_width,
                restore_from_cloud=restore_from_cloud,
                required=required,
                trace_reads=trace_reads,
                analysis_reads=analysis_reads,
            ))
            for line in results[-1][1]:
                print(line, file=sys.stderr, flush=True)
    else:
        # Parallel path — `fork` start method on Linux inherits the
        # parent's measurable registry via copy-on-write; no
        # per-worker re-import needed. Each worker is one
        # ProcessPoolExecutor task per subdir.
        import concurrent.futures as _cf
        import multiprocessing as _mp
        ctx = _mp.get_context('fork')
        results = [
            (None, []) for _ in sub_dirs
        ]
        with _cf.ProcessPoolExecutor(
            max_workers=max_workers, mp_context=ctx,
        ) as pool:
            futures: dict[_cf.Future[
                tuple[pl.DataFrame | None, list[str]]
            ], int] = {}
            for i, sub in enumerate(sub_dirs):
                fut = pool.submit(
                    _load_one_corpus,
                    sub, i=i, n_total=n_total,
                    digit_width=digit_width,
                    restore_from_cloud=restore_from_cloud,
                    required=required,
                    trace_reads=trace_reads,
                    analysis_reads=analysis_reads,
                )
                futures[fut] = i
            for fut in _cf.as_completed(futures):
                idx = futures[fut]
                try:
                    results[idx] = fut.result()
                except Exception as exc:  # noqa: BLE001
                    sub = sub_dirs[idx]
                    msg = (
                        f'  [{idx+1:>{digit_width}}/{n_total}] '
                        f'{sub.name}: ERROR — '
                        f'{type(exc).__name__}: {exc}'
                    )
                    results[idx] = (None, [msg])
                # Print in completion order so the user sees
                # progress as workers finish, not after all are done.
                for line in results[idx][1]:
                    print(line, file=sys.stderr, flush=True)

    frames = [df for df, _ in results if df is not None]
    n_loaded = len(frames)
    n_skipped = n_total - n_loaded
    walk_elapsed = _time.monotonic() - t_walk_start
    if not frames:
        print(
            f'runner: NO corpora loaded ({n_skipped} skipped) in '
            f'{walk_elapsed:.1f}s',
            file=sys.stderr,
            flush=True,
        )
        return pl.DataFrame()
    merged = _dedup_by_content(
        pl.concat(frames, how='diagonal_relaxed'),
        source='loaded directory',
    )
    print(
        f'runner: ingested {n_loaded}/{n_total} corpora '
        f'({n_skipped} skipped) → {merged.height} cells × '
        f'{len(merged.columns)} cols in {walk_elapsed/60:.1f} min',
        file=sys.stderr,
        flush=True,
    )
    return merged


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
