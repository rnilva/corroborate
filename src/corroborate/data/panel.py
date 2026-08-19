"""`Panel` — bridge-resolution + exploration data surface.

See module docstring on `corroborate/data/__init__.py` for the
consolidation rationale (bridge runtime + author exploration in
one type)."""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import cached_property
from pathlib import Path
from typing import Literal

import polars as pl


# Default exogenous-keys / -prefixes are imported from the
# canonical location in `corpus.catalogue` so the data subpackage
# can't silently drift from the rest of the framework's leaf
# semantics. Both surfaces are RL-substrate-shaped today; the
# implementation-author can override per-call via the diagnostics
# helper (Phase 1 doesn't expose the override on the public API
# — adds it when the next implementation needs different exogenous
# keys).
from corroborate.corpus.catalogue import (
    DEFAULT_EXOGENOUS_KEYS as _DEFAULT_EXOGENOUS_KEYS,
    DEFAULT_EXOGENOUS_PREFIXES as _DEFAULT_EXOGENOUS_PREFIXES,
)
# Framework-typed identity columns — never load-bearing for HP
# uniqueness; always excluded from the per-cell config hash.
_FRAMEWORK_IDENTITY_COLS: frozenset[str] = frozenset({
    'id', 'parent_id', 'cycle_id', 'treatment_arm_id',
    'timestamp', 'verdict', 'arm_key', 'arm_is_baseline',
    'corpus',
})


@dataclass(frozen=True, slots=True)
class CorpusSource:
    """Typed provenance entry — one corpus that contributed
    cells to a Panel. Isomorphic to the runner's
    `CacheSourceEntry` so Panel ↔ cache round-trips lose no
    sidecar provenance.

    `data_root` is `Path | None` so a Panel loaded from a cache
    whose sidecar entry pre-dates the named-ingest path
    rewrite (entry's `data_root` was `null`) survives the
    load. `with_traces` skips sources whose `data_root` is
    None — those entries can't locate `traces.parquet`."""
    corpus: str         # stamp as it appears in `cells.corpus` column
    data_root: Path | None = None  # corpus's parent dir; None when unknowable
    remote_root: str | None = None   # s3:// URI or None for local-only
    ingested_at: tuple[str, ...] = ()  # audit trail; populated on cache-write


@dataclass(frozen=True, slots=True)
class DerivedSpec:
    """Generalised per-stratum aggregate spec. Names a cell-level
    column + aggregator + per-cell filter; `Panel.derive(spec)`
    returns one float per stratum.

    Supersedes `analyses.link.cross_stratum_property_slope.
    DerivedCovariateSpec`'s `arm_filter: Literal[...]` with a full
    `cell_filter: pl.Expr` so implementation-author can express any
    filter (e.g. `pl.col('arm_key') == 'baseline'` for the
    σ_Λ_a-of-vanilla case, or `pl.col('lambda_a_late').is_finite()`
    to drop NaN cells before aggregating).

    `aggregator: Literal[...]` stays a closed set — adding entries
    is a substrate-deliberate framework change, not author-side
    flexibility. Use `'std'` for sample SD (ddof=1).

    `min_n`: stratum is skipped if fewer than `min_n` cells
    survive the column-finite + cell_filter narrowing. Default
    `None` → use `2` for `'std'` (SD undefined at n<2) and `1`
    for `'mean'`/`'median'` (no per-aggregator floor)."""
    column: str
    aggregator: Literal['mean', 'std', 'median']
    cell_filter: pl.Expr | None = None
    min_n: int | None = None

    @property
    def effective_min_n(self) -> int:
        if self.min_n is not None:
            return self.min_n
        return 2 if self.aggregator == 'std' else 1


@dataclass(frozen=True, slots=True)
class MeasurableAvailability:
    """P6 — typed per-env-per-measurable availability matrix.

    `availability`: outer key is env value (implementation-specific —
    typically env_name string); inner key is measurable name;
    value is the fraction of cells in that env where the column
    is non-null + non-NaN. Range [0.0, 1.0].

    `uniform_available`: measurable names finite on >0% of cells
    in EVERY env. These are the names a cross-env analysis can
    safely require without losing any env.

    `partial`: measurable names finite on >0% of cells in SOME
    envs but not ALL. Consumers must decide per-env: drop env,
    drop measurable, impute. The L3b per-env best-mediator script
    is the canonical example: per-env candidate auto-detection
    branches on this set.

    `unavailable`: measurable names that are >99% NaN across the
    ENTIRE panel (every env). Either substrate-unsatisfiable for
    this corpus set (e.g., reads cloud-evicted traces never
    locally available), or registered post-sweep with no
    recompute path.

    `cell_counts`: per-env total cell count — useful denominator
    when reporting fractional availability."""
    availability: Mapping[object, Mapping[str, float]]
    cell_counts: Mapping[object, int]
    uniform_available: frozenset[str]
    partial: frozenset[str]
    unavailable: frozenset[str]


def _compute_availability_matrix(
    cells: pl.DataFrame,
    names: tuple[str, ...],
    *,
    env_column: str,
) -> MeasurableAvailability:
    """Internal helper for `Panel.measurable_availability_matrix`.

    Walks each (env, measurable) pair, computes fraction of cells
    finite-not-null, classifies into uniform / partial /
    unavailable buckets."""
    if cells.height == 0 or not names:
        return MeasurableAvailability(
            availability={}, cell_counts={},
            uniform_available=frozenset(),
            partial=frozenset(),
            unavailable=frozenset(),
        )
    if env_column not in cells.columns:
        # Treat the whole panel as one env when the stratification
        # column doesn't exist. Avoids forcing the implementation-author
        # to special-case panels without env_name.
        env_values: tuple[object, ...] = ('__panel__',)
        subs = [cells]
    else:
        # Sort env values for deterministic iteration order — the
        # caller may use this to derive labels.
        unique_envs = cells[env_column].unique().to_list()
        # Stable sort: numerics first by value, strings
        # alphabetically. Mixed-dtype envs fall back to repr.
        def _sort_key(v: object) -> tuple[int, str]:
            if v is None:
                return (0, '')
            return (1, repr(v))
        env_values = tuple(sorted(unique_envs, key=_sort_key))
        subs = [
            cells.filter(pl.col(env_column) == v)
            for v in env_values
        ]
    availability: dict[object, dict[str, float]] = {}
    cell_counts: dict[object, int] = {}
    for env_v, sub in zip(env_values, subs):
        per_meas: dict[str, float] = {}
        n_cells = sub.height
        cell_counts[env_v] = n_cells
        for name in names:
            if name not in sub.columns:
                per_meas[name] = 0.0
                continue
            col = sub[name]
            if col.dtype.is_float():
                n_ok = int((col.is_not_null() & col.is_finite()).sum())
            else:
                n_ok = int(col.is_not_null().sum())
            per_meas[name] = n_ok / n_cells if n_cells > 0 else 0.0
        availability[env_v] = per_meas
    # Classify: a measurable is uniform_available iff every env
    # has nonzero finite fraction; unavailable iff every env has
    # <=0.01 finite fraction; partial otherwise.
    uniform_available: set[str] = set()
    unavailable: set[str] = set()
    partial: set[str] = set()
    for name in names:
        per_env_fracs = [
            availability[env_v].get(name, 0.0)
            for env_v in env_values
        ]
        if all(f > 0.0 for f in per_env_fracs):
            uniform_available.add(name)
        elif all(f <= 0.01 for f in per_env_fracs):
            unavailable.add(name)
        else:
            partial.add(name)
    return MeasurableAvailability(
        availability=availability,
        cell_counts=cell_counts,
        uniform_available=frozenset(uniform_available),
        partial=frozenset(partial),
        unavailable=frozenset(unavailable),
    )


@dataclass(frozen=True, slots=True)
class PanelDiagnostics:
    """Typed per-stratum facts. All four diagnostics as
    read-only attribute maps; no assertion methods.

    A bridge that cares about a specific facet reads the
    corresponding attribute and decides; bridges that don't,
    ignore it. The framework does not adjudicate
    homogeneity / balance — those are scope-tightening decisions
    the bridge author makes via the `scope_chain`."""
    n_cells_per_stratum: Mapping[tuple[object, ...], int]
    corpora_per_stratum: Mapping[tuple[object, ...], frozenset[str]]
    # Distinct `RunRow.program` values pooled into each stratum.
    # `arm_key` is the pure intervention fingerprint and is
    # program-BLIND (a `dqn` `baseline` arm and a `paired_dqn`
    # `baseline` arm share `arm_key='baseline'`), so a stratum whose
    # set here has >1 element is pooling cells from structurally
    # different root programs under one arm. Cross-program contrast
    # is legitimate (make `program` a stratify/scope dimension); this
    # surfaces the case so ACCIDENTAL pooling is visible rather than
    # silent. Empty frozenset for corpora predating the `program`
    # column (null reads as absent, not a distinct program).
    programs_per_stratum: Mapping[tuple[object, ...], frozenset[str]]
    finite_fraction_per_stratum_measurable: Mapping[
        tuple[object, ...], Mapping[str, float]
    ]
    nonunique_configs_per_stratum: Mapping[tuple[object, ...], int]
    scope_provenance: tuple[pl.Expr, ...]


def _is_excluded_col(
    col: str,
    *,
    exogenous_keys: frozenset[str],
    exogenous_prefixes: tuple[str, ...],
    measurable_names: frozenset[str],
) -> bool:
    """A column is excluded from the per-cell config hash iff it
    is framework-typed identity, exogenous, prefix-exogenous, or a
    registered measurable. Mirrors `corpus.leaf_signature.
    non_leaf_names` but operates on a pl.DataFrame column-name
    list (per-row eval would import-loop)."""
    if col in _FRAMEWORK_IDENTITY_COLS:
        return True
    if col in exogenous_keys:
        return True
    for p in exogenous_prefixes:
        if col.startswith(p):
            return True
    return col in measurable_names


def _resolve_cache_path(hypothesis_module: str) -> Path | None:
    """Resolve `<hyp_module>` → its default cache parquet path
    via the runner. Returns None when the module can't import."""
    import importlib
    from typing import cast

    from corroborate.core.hypothesis import Hypothesis
    from corroborate.runner.runner import default_cache_path
    try:
        mod = importlib.import_module(hypothesis_module)
    except ImportError:
        return None
    # Cast at the Protocol boundary: a module loaded via
    # `importlib.import_module` is a `ModuleType` (no nominal
    # `Hypothesis` relationship). The Protocol requires only
    # module-level `__name__` (every module has it) for
    # `default_cache_path`'s short-name derivation. The runtime
    # invariant is that `default_cache_path` reads ONLY
    # `h.__name__`; no other Protocol member is consulted.
    return default_cache_path(cast(Hypothesis, mod))


def _resolve_cache_target(
    *,
    hypothesis_module: str | None,
    cache_path: Path | None,
) -> Path:
    """Resolve the to_cache write target. Exactly one of the
    two must be set."""
    if hypothesis_module is not None and cache_path is not None:
        raise ValueError(
            'to_cache: pass hypothesis_module OR cache_path, not both',
        )
    if cache_path is not None:
        return cache_path
    if hypothesis_module is None:
        raise ValueError(
            'to_cache: pass hypothesis_module or cache_path',
        )
    resolved = _resolve_cache_path(hypothesis_module)
    if resolved is None:
        raise ImportError(
            f'to_cache: cannot import {hypothesis_module!r} '
            f'to resolve default cache path — pass cache_path '
            f'explicitly if the module isn\'t importable',
        )
    return resolved


def _read_sources_for_panel(
    cache_path: Path,
) -> tuple[CorpusSource, ...]:
    """Read `<cache>.sources.json` into `tuple[CorpusSource, ...]`.
    Returns empty when the sidecar is absent or malformed —
    matches the runner's sidecar-tolerance contract."""
    from corroborate.runner.runner import (
        read_sources, sources_sidecar_path,
    )
    cache_sources = read_sources(sources_sidecar_path(cache_path))
    if cache_sources is None:
        return ()
    return tuple(
        CorpusSource(
            corpus=entry.corpus,
            # `is not None` (not truthy) — preserves the
            # null/empty-string asymmetry of the sidecar JSON
            # round-trip. A `Path("")` from the runner's side
            # would otherwise drop to None and never resurface.
            data_root=(
                Path(entry.data_root)
                if entry.data_root is not None else None
            ),
            remote_root=entry.remote_root,
            ingested_at=entry.ingested_at,
        )
        for entry in cache_sources.sources
    )


def _write_manifest_for_panel(
    cells: pl.DataFrame,
    manifest_target: Path,
) -> None:
    """Compute closure-hash signatures for every cell column
    that names a registered `@measurable`, and write
    `<cache>.hashes.json`. Columns not in the registry are
    silently skipped — matches the runner's `_signatures_for`
    behaviour. Substrate-author MUST import the implementation's
    measurables module before calling Panel.to_cache for the
    manifest to be populated; otherwise the registry is empty
    and the file is `{}`."""
    import json

    from corroborate.corpus.persistence import (
        atomic_write_text as _atomic_write_text,
    )
    from corroborate.measurables.measurable import get_registered
    signatures: dict[str, str] = {}
    for col in cells.columns:
        m = get_registered(col)
        if m is None:
            continue
        signatures[col] = m.signature()
    _atomic_write_text(
        manifest_target,
        json.dumps(signatures, indent=2, sort_keys=True),
    )


def _write_sources_for_panel(
    cache_path: Path,
    sources: tuple[CorpusSource, ...],
) -> None:
    """Translate `tuple[CorpusSource, ...]` → `CacheSources`,
    append a fresh ISO-8601 UTC timestamp to each entry's
    `ingested_at` (audit trail extension), and atomically write
    the `.sources.json` sidecar. Empty `sources` writes an empty
    sidecar — mirrors the runner's emit-with-no-entries form.

    Note: timestamp semantics differ from the runner's. The
    runner's `_update_sources_with_walk` appends ONE timestamp
    per re-walk of a corpus's `runs.parquet`; Panel's `to_cache`
    appends ONE timestamp per `to_cache()` call regardless of
    whether the underlying corpora changed. Consumers counting
    distinct ingest events from `ingested_at` should not assume
    one-per-ingest when Panel writes are in play."""
    from datetime import datetime, timezone

    from corroborate.runner.runner import (
        CacheSourceEntry, CacheSources,
        sources_sidecar_path, write_sources,
    )
    now = datetime.now(timezone.utc).isoformat()
    entries = tuple(
        CacheSourceEntry(
            corpus=s.corpus,
            # `is not None` — see read-side rationale; preserve
            # null vs empty-string symmetry across round-trip.
            data_root=(
                str(s.data_root) if s.data_root is not None else None
            ),
            remote_root=s.remote_root,
            ingested_at=(*s.ingested_at, now),
        )
        for s in sources
    )
    write_sources(
        sources_sidecar_path(cache_path),
        CacheSources(sources=entries),
    )


@dataclass(frozen=True)
class Panel:
    """Bridge-resolution + exploration data surface.

    `cells` is the per-cell DataFrame. `scope_chain` is the
    ordered tuple of `pl.Expr` filters that produced these cells
    (lineage from the Panel's root). `stratify_by` names the
    grouping columns for `diagnostics`. `sources` records the
    ingest provenance (used for the `corpora_per_stratum`
    diagnostic). `required_measurables`, when non-empty,
    narrows the `finite_fraction_per_stratum_measurable` map to
    those names; otherwise the map covers every numeric column
    in `cells` (exploration mode).

    Frozen but NOT `slots=True` — `functools.cached_property`
    needs `__dict__` to memo `diagnostics`. The earlier
    `_diag_cache: list[PanelDiagnostics]` mutable-list-slot
    pattern (Phase 1 review fix) had a latent race condition
    (concurrent first reads could both append). `cached_property`
    on a non-slotted dataclass is the standard pattern;
    `frozen=True` still blocks rebinding."""
    cells: pl.DataFrame
    scope_chain: tuple[pl.Expr, ...] = ()
    stratify_by: tuple[str, ...] = ('env_name', 'arm_key')
    sources: tuple[CorpusSource, ...] = ()
    required_measurables: frozenset[str] = field(default_factory=frozenset)
    # The record's configuration registry: which columns were
    # CONFIGURED (the external counterpart of a native claim
    # composition's leaf walk). Run readers populate it from the
    # record's own artifacts; `evaluate()` consumes it for the
    # knob-aware admission gates. None means "no registry known" —
    # the gates then report their checks unverified rather than
    # silently passing. A fact the frame cannot carry itself; the
    # Panel is the typed carrier that travels with the cells.
    leaves: frozenset[str] | None = None

    @classmethod
    def from_dataframe(
        cls,
        cells: pl.DataFrame,
        *,
        scope_chain: tuple[pl.Expr, ...] = (),
        stratify_by: tuple[str, ...] = ('env_name', 'arm_key'),
        sources: tuple[CorpusSource, ...] = (),
        required_measurables: frozenset[str] = frozenset(),
        leaves: frozenset[str] | None = None,
    ) -> 'Panel':
        """Construct a Panel from a pre-built DataFrame. The
        idiomatic exploration-time constructor when the author
        already has cells in hand (e.g. a test fixture, a
        polars expression-built frame, or `pl.read_parquet(...)`
        of a cache file)."""
        return cls(
            cells=cells,
            scope_chain=scope_chain,
            stratify_by=stratify_by,
            sources=sources,
            required_measurables=required_measurables,
            leaves=leaves,
        )

    @classmethod
    def from_corpus(
        cls,
        corpus_dir: Path | str,
        *,
        join_traces: bool = False,
        stratify_by: tuple[str, ...] = ('env_name', 'arm_key'),
    ) -> 'Panel':
        """Load one corpus's `runs.parquet` + `measurements.parquet`
        without going through the runner's full ingest pipeline.

        `measurements.parquet` is left-joined onto `runs.parquet`
        by `id`; absent measurements (corpus pre-dates a registered
        measurable, or sidecar is missing) leave the column null.

        `join_traces=True` additionally left-joins `traces.parquet`
        — typically only useful when an analysis primitive directly
        consumes per-step or per-burst trace columns (most don't).
        Off by default because traces are GB-scale.

        `sources` is populated with one `CorpusSource` keyed by
        the corpus directory's name (mirrors the runner's
        bare-leaf corpus-stamp for top-level corpora; nested
        sub-corpora are not normalised on this path — use
        `from_corpora` for the parent/leaf form).

        Returns an empty Panel when `runs.parquet` is missing."""
        corpus_path = Path(corpus_dir)
        runs_path = corpus_path / 'runs.parquet'
        if not runs_path.exists():
            return cls(
                cells=pl.DataFrame(),
                stratify_by=stratify_by,
            )
        cells = pl.read_parquet(runs_path)
        meas_path = corpus_path / 'measurements.parquet'
        if meas_path.exists() and 'id' in cells.columns:
            meas = pl.read_parquet(meas_path)
            # Collision resolution delegated to
            # `corpus.measurements.resolve_runs_meas_collision`
            # — single source of truth shared with
            # `build_measurements`. The two callers differ ONLY
            # in `unregistered_policy`:
            # - `build_measurements`: `'runs_wins'` (runner has
            #   the registry; non-registered overlap is the
            #   defensive CI6 fallback)
            # - here: `'meas_wins'` (Panel is an exploration
            #   entry point where implementation may not be imported;
            #   trust the stamped measurements value over a
            #   runs-side NaN)
            from corroborate.corpus.measurements import (
                resolve_runs_meas_collision,
            )
            runs_cols = set(cells.columns)
            meas_cols = set(meas.columns)
            drop_from_cells_set, drop_from_meas_set = (
                resolve_runs_meas_collision(
                    runs_cols=runs_cols,
                    meas_cols=meas_cols,
                    unregistered_policy='meas_wins',
                )
            )
            if drop_from_cells_set:
                cells = cells.drop(list(drop_from_cells_set))
            keep_meas = [c for c in meas.columns if c not in drop_from_meas_set]
            meas = meas.select(keep_meas)
            cells = cells.join(meas, on='id', how='left')
        if join_traces:
            traces_path = corpus_path / 'traces.parquet'
            if traces_path.exists() and 'id' in cells.columns:
                traces = pl.read_parquet(traces_path)
                overlap = [
                    c for c in traces.columns
                    if c != 'id' and c in cells.columns
                ]
                if overlap:
                    traces = traces.drop(overlap)
                cells = cells.join(traces, on='id', how='left')
        # Stamp the corpus column for cells that don't carry it
        # (pre-Phase-3 corpora may lack it; modern corpora include
        # it). Mirrors `runner._corpus_stamp`'s parent/leaf form:
        # if the parent dir has its own `runs.parquet`, the
        # corpus is a NESTED sub-corpus and the stamp is
        # `parent.name/sub.name`. Without this, same-leaf-name
        # sub-corpora from different parents collide in
        # `corpora_per_stratum` diagnostics.
        parent_runs = corpus_path.parent / 'runs.parquet'
        if parent_runs.exists():
            corpus_stamp = f'{corpus_path.parent.name}/{corpus_path.name}'
            data_root = corpus_path.parent.parent.resolve()
        else:
            corpus_stamp = corpus_path.name
            data_root = corpus_path.parent.resolve()
        if 'corpus' not in cells.columns:
            cells = cells.with_columns(
                pl.lit(corpus_stamp).alias('corpus'),
            )
        source = CorpusSource(
            corpus=corpus_stamp,
            data_root=data_root,
        )
        return cls(
            cells=cells,
            stratify_by=stratify_by,
            sources=(source,),
        )

    @classmethod
    def from_cache(
        cls,
        hypothesis_module: str,
        *,
        stratify_by: tuple[str, ...] = ('env_name', 'arm_key'),
    ) -> 'Panel':
        """Load the existing per-hypothesis cache parquet as a
        Panel. Day-1 entry when a hypothesis has already been
        run and aggregated — bypasses re-ingest for follow-up
        probes on the same cohort.

        Delegates path resolution to the runner's
        `_default_cache_path` (single source of truth — if the
        runner ever changes the cache-stamp convention, both
        runner-side ingest and Panel-side load shift together).
        Returns an empty Panel when the file doesn't exist.

        `sources` is populated from `<hyp>.sources.json` when
        present; each `CacheSourceEntry` becomes one
        `CorpusSource` (per-cell provenance still lives in the
        `corpus` column on cells). The `to_cache` sibling closes
        the round-trip: an exploration Panel with a `sources`
        tuple persists into a sidecar-matched on-disk pair."""
        cache_path = _resolve_cache_path(hypothesis_module)
        if cache_path is None or not cache_path.exists():
            return cls(cells=pl.DataFrame(), stratify_by=stratify_by)
        cells = pl.read_parquet(cache_path)
        sources = _read_sources_for_panel(cache_path)
        return cls(
            cells=cells,
            stratify_by=stratify_by,
            sources=sources,
        )

    def to_cache(
        self,
        hypothesis_module: str | None = None,
        *,
        cache_path: Path | None = None,
        write_sidecar: bool = True,
        write_manifest: bool = True,
    ) -> Path:
        """Promote this Panel into the per-hypothesis cache. The
        substrate-author's "my exploration found a real signal —
        ship it as the production cache" operation. Either
        `hypothesis_module` (resolves via the runner's
        `default_cache_path`) or `cache_path` (explicit) must
        be supplied; passing both raises.

        Writes:
        - `<cache>.parquet`: `self.cells` as-is.
        - `<cache>.sources.json` (when `write_sidecar=True`):
          one entry per `self.sources` element, with `ingested_at`
          extended by an ISO-8601 UTC timestamp for THIS write.
        - `<cache>.hashes.json` (when `write_manifest=True`): one
          entry per cell column whose name matches a currently-
          registered `@measurable`. Built from the live registry,
          so the implementation's measurables module MUST be imported
          before calling `to_cache` for the manifest to be
          meaningful. Columns that don't match a registered name
          are silently skipped (matches the runner's
          `_signatures_for` semantics).

        Passing `write_manifest=False` unlinks any pre-existing
        manifest without writing a fresh one — use when the
        Panel was assembled WITHOUT importing the implementation's
        measurables module (the manifest would be misleadingly
        empty); the next `corroborate hypothesis <module>` pass
        will rebuild from scratch.

        Cloud-mirror caveat: `to_cache` is local-only. Use
        `corroborate archive` afterwards if you want the
        promoted cache visible to colleagues via S3.

        Returns the cache parquet's absolute path so the caller
        can chain (`run(...)` on the same target, etc.)."""
        target = _resolve_cache_target(
            hypothesis_module=hypothesis_module,
            cache_path=cache_path,
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        # Lazy import: `runner` already imports `data` indirectly
        # through Hypothesis Protocol — keep this off the
        # module-load critical path so `from corroborate.data
        # import Panel` doesn't pay JAX-import cost.
        from corroborate.corpus.persistence import (
            atomic_write_parquet as _atomic_write_parquet,
        )
        from corroborate.runner.runner import manifest_path
        _atomic_write_parquet(self.cells, target)
        if write_sidecar:
            _write_sources_for_panel(target, self.sources)
        manifest_target = manifest_path(target)
        if write_manifest:
            _write_manifest_for_panel(self.cells, manifest_target)
        else:
            # Unlink any stale manifest. The promoted row-set may
            # not match the previous manifest's measurable
            # signatures; let the next runner pass build a fresh
            # one rather than serve a stale entry as authoritative.
            manifest_target.unlink(missing_ok=True)
        return target

    def with_traces(
        self, cols: Sequence[str] | None = None,
    ) -> 'Panel':
        """Left-join the named trace columns from this Panel's
        source corpora into `cells`. Use case: explored without
        traces (cheap), then realised the next probe needs e.g.
        `online_max_q_per_step`. `cols=None` joins ALL trace
        columns (expensive — typical traces are GB-scale); pass
        a list to narrow.

        Requires Panel's `sources` to be non-empty (constructed
        via `from_corpus` / `from_corpora`). For panels built
        from `from_dataframe` or `from_cache` (no source-corpus
        provenance), this is a no-op — the trace files are
        wherever the dataframe came from, opaque to Panel.

        Already-present columns are not re-joined (skipped).
        Missing trace files on any source are silently skipped
        (matches `from_corpus`'s no-traces-no-fail convention)."""
        if not self.sources or self.cells.height == 0:
            return self
        if 'id' not in self.cells.columns:
            return self
        # Load each source's traces (filtered to requested cols),
        # then `pl.concat` diagonally so the trace frame has one
        # row per (corpus, id) — schema-aligned across corpora,
        # NaN-padded where a corpus doesn't carry a col. Single
        # left-join against cells avoids the cross-corpus NaN-
        # then-suffix bug of per-source sequential joins (each
        # individual join saw all panel cells, not just its
        # corpus's cells, producing `<col>_right` suffixes on
        # the second iteration).
        per_source_traces: list[pl.DataFrame] = []
        for src in self.sources:
            if src.data_root is None:
                # Loaded from a cache whose sidecar didn't stamp
                # the corpus root — no way to locate trace files.
                continue
            corpus_path = src.data_root / src.corpus
            traces_path = corpus_path / 'traces.parquet'
            if not traces_path.exists():
                continue
            try:
                schema_names = set(
                    pl.scan_parquet(traces_path).collect_schema().names()
                )
            except pl.exceptions.ComputeError:
                continue
            if cols is None:
                target_cols = [
                    c for c in schema_names
                    if c != 'id' and c not in self.cells.columns
                ]
            else:
                target_cols = [
                    c for c in cols
                    if c in schema_names and c not in self.cells.columns
                ]
            if not target_cols:
                continue
            per_source_traces.append(pl.read_parquet(
                traces_path, columns=['id', *target_cols],
            ))
        if not per_source_traces:
            return self
        traces = pl.concat(per_source_traces, how='diagonal_relaxed')
        new_cells = self.cells.join(traces, on='id', how='left')
        return replace(self, cells=new_cells)

    @classmethod
    def from_corpora(
        cls,
        corpus_dirs: Iterable[Path | str],
        *,
        join_traces: bool = False,
        stratify_by: tuple[str, ...] = ('env_name', 'arm_key'),
    ) -> 'Panel':
        """Union multiple corpora's cells into one Panel.
        Diagonal-relaxed concat handles schema drift across
        corpora (newer ones with extra columns, older ones with
        fewer). Each corpus contributes one `CorpusSource`
        provenance entry."""
        panels: list[Panel] = []
        for d in corpus_dirs:
            panels.append(cls.from_corpus(
                d, join_traces=join_traces, stratify_by=stratify_by,
            ))
        if not panels:
            return cls(cells=pl.DataFrame(), stratify_by=stratify_by)
        cells = pl.concat(
            [p.cells for p in panels if p.cells.height > 0],
            how='diagonal_relaxed',
        ) if any(p.cells.height > 0 for p in panels) else pl.DataFrame()
        sources_combined: tuple[CorpusSource, ...] = tuple(
            s for p in panels for s in p.sources
        )
        return cls(
            cells=cells,
            stratify_by=stratify_by,
            sources=sources_combined,
        )

    def narrow(self, expr: pl.Expr) -> 'Panel':
        """Apply an additional scope filter; return a new Panel
        with `expr` appended to `scope_chain`. Diagnostics on
        the narrowed panel are recomputed from the narrowed cells
        (cheap; the `scope_provenance` chain preserves the parent
        expressions for audit)."""
        return replace(
            self,
            cells=self.cells.filter(expr),
            scope_chain=self.scope_chain + (expr,),
        )

    def with_columns(self, *exprs: pl.Expr) -> 'Panel':
        """`cells.with_columns(...)` that stays a Panel — the
        provenance, registry, and scope lineage travel with the
        enriched frame. The idiomatic way to stamp analyst-known
        context the record itself doesn't carry (an SB3 checkpoint
        doesn't name its environment; the analyst does:
        `panel.with_columns(pl.lit('CartPole-v1').alias('env_id'))`).
        Columns added here are analyst context, not configuration —
        they do not join `leaves`."""
        return replace(self, cells=self.cells.with_columns(*exprs))

    def split_by(
        self, *keys: str,
    ) -> Mapping[tuple[object, ...], 'Panel']:
        """Partition cells by the named keys; return a mapping
        from stratum-id tuple to sub-Panel. Each sub-Panel
        inherits `scope_chain`, `sources`, `required_measurables`;
        its `stratify_by` is unchanged (the panel can still
        compute per-stratum diagnostics with a coarser grouping
        than the split keys).

        Uses `polars.DataFrame.partition_by(... as_dict=True)`
        — a single-pass partition rather than per-stratum
        `cells.filter(mask)` (which was O(n_strata × n_cells))."""
        keys_used = keys or self.stratify_by
        if not keys_used or self.cells.height == 0:
            return {}
        keys_list = list(keys_used)
        # `partition_by(..., as_dict=True)` returns a mapping
        # from stratum-key-tuple to sub-DataFrame in a single
        # scan. Stable sort over stratum keys for predictable
        # iteration order (implementation consumers may rely on it).
        partitions = self.cells.partition_by(
            keys_list, as_dict=True, maintain_order=False,
        )
        out: dict[tuple[object, ...], 'Panel'] = {}
        # Sort by stratum-id for deterministic iteration order.
        def _sort_key(k: tuple[object, ...]) -> tuple[str, ...]:
            # `str(v)` rather than `repr(v)`: stable across
            # Python versions for floats (no trailing-zero edge
            # cases) and consistent for mixed-type stratify keys.
            return tuple(str(v) for v in k)
        for k_raw in sorted(partitions, key=_sort_key):
            stratum_id: tuple[object, ...] = tuple(k_raw)
            out[stratum_id] = Panel(
                cells=partitions[k_raw],
                scope_chain=self.scope_chain,
                stratify_by=self.stratify_by,
                sources=self.sources,
                required_measurables=self.required_measurables,
            )
        return out

    def derive(
        self, spec: DerivedSpec,
    ) -> Mapping[tuple[object, ...], float]:
        """Per-stratum aggregate of `spec.column` via
        `spec.aggregator`, optionally filtered by
        `spec.cell_filter`. Returns `{stratum_id: aggregate}`.

        Delegates to `corroborate.data.kernel.per_stratum_
        aggregate` — single source of truth shared with the
        cells-input analysis primitives (Phase 2 migration). The
        kernel takes a polars DataFrame + structured spec; Panel
        passes `self.cells` directly while cells-input adapters
        materialise via `cells_to_dataframe`."""
        from corroborate.data.kernel import per_stratum_aggregate
        return per_stratum_aggregate(
            self.cells,
            column=spec.column,
            aggregator=spec.aggregator,
            stratify_by=self.stratify_by,
            cell_filter=spec.cell_filter,
            min_n=spec.effective_min_n,
        )

    def with_measurables(
        self, names: Sequence[str],
    ) -> 'Panel':
        """Return a new Panel with the named measurables computed
        for cells where the column is absent or null/NaN. No-op
        for names already present and fully finite. Names not in
        the `@measurable` registry are silently skipped.

        Uses `compute_missing_columns` — same topo-sort cascade
        as `build_measurements`. For absent transitive trace
        reads (e.g. `online_top12_margin_per_step` not joined),
        the computation will return NaN per cell, matching the
        framework's "unsatisfiable → NaN" contract."""
        if self.cells.height == 0:
            return self
        # Lazy import to avoid the `measurables ↔ data` cycle
        # if `compute_missing_columns` ever needs to consult
        # Panel-side state.
        from corroborate.measurables import compute_missing_columns
        new_cells = compute_missing_columns(self.cells, names)
        return replace(self, cells=new_cells)

    def measurable_availability_matrix(
        self,
        names: Sequence[str] | None = None,
        *,
        env_column: str = 'env_name',
    ) -> 'MeasurableAvailability':
        """**P6 fix** — per-env-per-measurable availability matrix.

        Different corpora carry different subsets of registered
        measurables (P6 in `FRAMEWORK_INGEST_PITFALLS.md`): a
        sweep that ran before a measurable was added has it as
        NaN, while a newer sweep has it populated. The cache
        silently joins inconsistent stores into a single DataFrame
        with column-specific NaN — consumers that filter on
        `is_finite()` accidentally drop entire envs.

        This surface makes the divergence first-class:

        - `availability`: `{env_value: {measurable_name: float}}`
          — fraction of finite cells per (env, measurable).
        - `uniform_available`: names finite on >0% of cells in
          EVERY env (consumers can safely require these).
        - `partial`: names finite on >0% of cells in SOME envs
          but not all (operator must decide: drop env / drop
          measurable / impute).
        - `unavailable`: names that are >99% NaN across the
          panel (a measurable that's effectively absent — either
          unsatisfiable for this corpus set, or registered
          post-sweep with no recompute path).

        `names`: subset to consider (default: every column on
        `self.cells` that names a registered `@measurable`).

        `env_column`: stratification key. Defaults to `env_name`;
        implementation-author may override (`corpus`, `arm_key`, etc.)
        for finer-grained availability views.

        Pure read; no side effects."""
        from corroborate.measurables import registered_names
        measurable_set = frozenset(registered_names())
        if names is None:
            candidate_names = tuple(
                c for c in self.cells.columns
                if c in measurable_set
            )
        else:
            candidate_names = tuple(
                n for n in names if n in self.cells.columns
            )
        return _compute_availability_matrix(
            self.cells, candidate_names, env_column=env_column,
        )

    @cached_property
    def diagnostics(self) -> PanelDiagnostics:
        """Typed per-stratum facts. Cached on first access via
        `functools.cached_property` — subsequent reads return the
        same `PanelDiagnostics` object, no recompute.

        Frozen-but-non-slots dataclass: `cached_property` writes
        the cached value to `self.__dict__` on first read; `frozen=True`
        still blocks `__setattr__` for declared fields but
        `cached_property` uses descriptor-level dict access that
        the frozen guard doesn't intercept. Standard Python
        pattern; the earlier `_diag_cache: list[PanelDiagnostics]`
        mutable-slot workaround had a latent race condition
        (concurrent first reads could both append).

        Subsequent narrow/split_by/with_traces/with_measurables
        construct NEW Panel instances with empty caches — the
        recompute fires once per panel-shape."""
        return self._compute_diagnostics(
            exogenous_keys=_DEFAULT_EXOGENOUS_KEYS,
            exogenous_prefixes=_DEFAULT_EXOGENOUS_PREFIXES,
        )

    def _compute_diagnostics(
        self,
        *,
        exogenous_keys: frozenset[str],
        exogenous_prefixes: tuple[str, ...],
    ) -> PanelDiagnostics:
        if self.cells.height == 0 or not self.stratify_by:
            return PanelDiagnostics(
                n_cells_per_stratum={},
                corpora_per_stratum={},
                programs_per_stratum={},
                finite_fraction_per_stratum_measurable={},
                nonunique_configs_per_stratum={},
                scope_provenance=self.scope_chain,
            )
        # Cells split by stratify_by once; reused for all four
        # diagnostics.
        # Lazy import — measurables registry import must not be
        # forced when the data subpackage is loaded standalone.
        from corroborate.measurables import registered_names
        measurable_names = frozenset(registered_names())
        config_cols = [
            c for c in self.cells.columns
            if not _is_excluded_col(
                c,
                exogenous_keys=exogenous_keys,
                exogenous_prefixes=exogenous_prefixes,
                measurable_names=measurable_names,
            )
        ]
        # Which measurable cols to compute finiteness for.
        if self.required_measurables:
            meas_cols = [
                c for c in self.cells.columns
                if c in self.required_measurables
            ]
        else:
            # Exploration mode: every numeric col that IS a
            # registered measurable counts.
            meas_cols = [
                c for c in self.cells.columns
                if c in measurable_names
            ]
        n_counts: dict[tuple[object, ...], int] = {}
        corpora: dict[tuple[object, ...], frozenset[str]] = {}
        programs: dict[tuple[object, ...], frozenset[str]] = {}
        finite_frac: dict[
            tuple[object, ...], Mapping[str, float]
        ] = {}
        nonunique: dict[tuple[object, ...], int] = {}
        stratify_list = list(self.stratify_by)
        grouped = self.cells.group_by(stratify_list)
        for stratum_id_obj, sub in grouped:
            # Polars normalises group-by keys to a tuple-of-key-
            # values regardless of how many keys; `tuple(...)`
            # preserves shape + drops the `tuple[Any, ...]` stub
            # taint that propagates from polars' iter API.
            stratum_id: tuple[object, ...] = tuple(stratum_id_obj)
            n_counts[stratum_id] = sub.height
            # corpora_per_stratum: 'corpus' column required.
            if 'corpus' in sub.columns:
                corpus_vals: list[object] = list(sub['corpus'].to_list())
                corpora[stratum_id] = frozenset(
                    str(v) for v in corpus_vals if v is not None
                )
            else:
                corpora[stratum_id] = frozenset()
            # programs_per_stratum: distinct non-null 'program'
            # values. Surfaces program-blind arm_key pooling (a
            # stratum with >1 program mixes root programs under one
            # arm). Absent/all-null column → empty set.
            if 'program' in sub.columns:
                program_vals: list[object] = list(sub['program'].to_list())
                programs[stratum_id] = frozenset(
                    str(v) for v in program_vals if v is not None
                )
            else:
                programs[stratum_id] = frozenset()
            # finite_fraction_per_stratum_measurable.
            per_meas: dict[str, float] = {}
            for mc in meas_cols:
                col = sub[mc]
                if col.dtype.is_float():
                    n_ok = int((col.is_not_null() & col.is_finite()).sum())
                else:
                    n_ok = int(col.is_not_null().sum())
                per_meas[mc] = n_ok / sub.height if sub.height else 0.0
            finite_frac[stratum_id] = per_meas
            # nonunique_configs_per_stratum: hash per-cell config
            # rows + count distinct.
            if config_cols:
                config_rows = sub.select(config_cols).iter_rows()
                fingerprints: set[tuple[object, ...]] = {
                    tuple(row) for row in config_rows
                }
                nonunique[stratum_id] = len(fingerprints)
            else:
                nonunique[stratum_id] = 0
        return PanelDiagnostics(
            n_cells_per_stratum=n_counts,
            corpora_per_stratum=corpora,
            programs_per_stratum=programs,
            finite_fraction_per_stratum_measurable=finite_frac,
            nonunique_configs_per_stratum=nonunique,
            scope_provenance=self.scope_chain,
        )


def concat_panels(panels: Sequence[Panel]) -> Panel:
    """Pool batches of a growing record into one Panel.

    Cells concatenate diagonally (heterogeneous columns null-pad),
    sources concatenate, and the configuration registries union —
    unless ANY batch has no registry, in which case the pool has
    none either (an unknown part makes the whole unknown; the
    gates then report their checks unverified). Scope lineage does
    not survive pooling: the result is a fresh root, like
    `from_corpora`. `stratify_by` must agree across batches —
    silently keeping one of two disagreeing groupings would make
    `diagnostics` lie about half the pool."""
    if not panels:
        raise ValueError('concat_panels: no panels to pool')
    stratify = {p.stratify_by for p in panels}
    if len(stratify) != 1:
        raise ValueError(
            f'concat_panels: panels disagree on stratify_by: '
            f'{sorted(stratify)!r}',
        )
    leaves: frozenset[str] | None
    if any(p.leaves is None for p in panels):
        leaves = None
    else:
        leaves = frozenset().union(
            *(p.leaves for p in panels if p.leaves is not None),
        )
    return Panel(
        cells=pl.concat(
            [p.cells for p in panels], how='diagonal_relaxed',
        ),
        scope_chain=(),
        stratify_by=panels[0].stratify_by,
        sources=tuple(s for p in panels for s in p.sources),
        required_measurables=frozenset().union(
            *(p.required_measurables for p in panels),
        ),
        leaves=leaves,
    )


__all__ = [
    'CorpusSource',
    'DerivedSpec',
    'MeasurableAvailability',
    'Panel',
    'PanelDiagnostics',
    'concat_panels',
]
