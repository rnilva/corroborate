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

Module surface — every bridges module satisfies `HypothesisModule`
by declaring (at least) `BRIDGES: tuple[Bridge, ...]`. The cache
file is keyed off the module's dotted-path leaf
(`mod.__name__.split('.')[-1]`) — there is no override surface; if
the cache lives in a non-default location, write a thin script
that calls `run_module(..., cache_path=...)` directly (the kwarg
exists for that purpose).

This module is library-only — no argparse, no `if __name__ ==
'__main__'`. The CLI thin-wrapper lives at
`scripts/run_hypothesis.py`."""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast, runtime_checkable

import polars as pl

from corroborate.claim_bridge import (
    Bridge,
    BridgeEvaluation,
    evaluate,
    measurable_names_for_bridges,
)
from corroborate.measurable import (
    evaluate_with_measurables,
    get_registered,
    transitive_measurables,
)


# ============ Module Protocol ============


@runtime_checkable
class HypothesisModule(Protocol):
    """Module-level Protocol: any module exporting `BRIDGES` (a
    tuple of `Bridge` instances) satisfies it. The runner uses
    `isinstance(mod, HypothesisModule)` for the structural check;
    pyright narrows accordingly inside the if-block.

    `runtime_checkable` only validates attribute *presence* (not
    element types) — `_validate_module` adds the element-type
    check on top so a malformed BRIDGES tuple fails loudly at
    runner-dispatch time.

    `__name__` is the module's dotted path (every Python module
    has it, but the Protocol declares it explicitly so pyright
    can narrow attribute access through the protocol type)."""

    BRIDGES: tuple[Bridge, ...]
    __name__: str


def _validate_module(mod: ModuleType) -> HypothesisModule:
    """Narrow `mod` to `HypothesisModule` via the Protocol's
    `__instancecheck__`, then verify each `BRIDGES` element is a
    `Bridge` (Protocol's runtime check doesn't validate element
    types). Raises `TypeError` on shape errors."""
    if not isinstance(mod, HypothesisModule):
        raise TypeError(
            f'{mod.__name__} is not a HypothesisModule: missing '
            f'`BRIDGES: tuple[Bridge, ...]` at module level. '
            f'Bridges files must export the canonical name '
            f'`BRIDGES` (alias `BRIDGES = LEGACY_NAME` is fine).',
        )
    # Protocol typing has narrowed `mod.BRIDGES` to tuple[Bridge,
    # ...] for the static checker. `runtime_checkable.__instancecheck__`
    # only validates attribute *presence*, so a defensive element-
    # type check defends against malformed authoring (e.g. a non-
    # Bridge slipped into the tuple via a copy-paste mistake).
    for b in mod.BRIDGES:
        if not isinstance(b, Bridge):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(
                f'{mod.__name__}.BRIDGES contains non-Bridge: '
                f'{type(b).__name__}',
            )
    return mod


def _default_cache_path(mod: HypothesisModule) -> Path:
    """Per-module cache file at
    `experiments/data/cache/<short>.parquet`, where `<short>` is
    the last segment of the module's dotted path."""
    short = mod.__name__.split('.')[-1]
    return Path('experiments/data/cache') / f'{short}.parquet'


# ============ Measurable signature + manifest ============


def _measurable_signature(name: str) -> str | None:
    """Closure hash of a registered measurable: `sha256` of its own
    bytecode plus the sorted bytecode hashes of every transitive
    measurable dep. Returns None if `name` isn't registered (e.g.
    a column already in the cache that doesn't belong to a current
    measurable — those are left untouched).

    This is what makes "edited a measurable's body, cache is now
    stale" detectable: changing any function in the closure flips
    the resulting hex. Hex is short (16 chars) — collisions are
    irrelevant against the user-edit baseline."""
    if get_registered(name) is None:
        return None
    deps = sorted(transitive_measurables(name))
    parts: list[str] = []
    for d in deps:
        m = get_registered(d)
        if m is None:
            parts.append(f'{d}:unregistered')
            continue
        bc = bytes(m.fn.__code__.co_code)
        parts.append(f'{d}:{hashlib.sha256(bc).hexdigest()[:16]}')
    return hashlib.sha256('\n'.join(parts).encode()).hexdigest()[:16]


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


def run_module(
    module_name: str,
    *,
    data: pl.DataFrame | Path | str | None = None,
    use_cache: bool = True,
    write_cache: bool = True,
    rebuild: bool = False,
    restore_from_cloud: bool = True,
    cache_path: Path | None = None,
) -> dict[str, BridgeEvaluation]:
    """Run a bridges module on `data`, returning per-bridge verdicts.

    Cache lifecycle:

    - `use_cache=True` (default): read+write the per-module cache.
      Cells already in cache with all required measurables skip
      recomputation. New cells from `data` get measurables computed
      and appended.
    - `use_cache=False`: pure compute path; no cache read or write.
    - `write_cache=False` + `use_cache=True`: read cache, run, but
      don't persist updates.
    - `rebuild=True`: invalidate the per-module cache before
      running. Implies `use_cache=True`.

    `restore_from_cloud=True` (default): when ingesting a corpus
    directory whose `runs.parquet` is missing locally but has a
    `_remote.json` manifest, pull raw from s3. Set False to opt
    out + warn loudly on the missing data.

    `cache_path`: explicit override for the cache file. When None
    and `use_cache=True`, defaults to
    `experiments/data/cache/<module-leaf>.parquet`.

    `data` may be:
    - `None`: run on whatever's already in the cache.
    - a `pl.DataFrame`: use as-is.
    - a path to a `.parquet` file: read directly.
    - a path to a directory: walk its subdirs for per-corpus
      `runs.parquet` (with auto-restore), concat via
      `diagonal_relaxed`."""
    mod = _validate_module(importlib.import_module(module_name))
    bridges = mod.BRIDGES

    resolved_cache: Path | None = None
    if use_cache:
        resolved_cache = cache_path if cache_path is not None else _default_cache_path(mod)
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
            f'{module_name}: no cells available — pass --data to '
            f'ingest a corpus, or check the cache at {resolved_cache}',
        )

    out: dict[str, BridgeEvaluation] = {}
    for b in bridges:
        try:
            out[b.name] = evaluate(b, cells)
        except Exception as e:  # noqa: BLE001
            # Rather than crash the whole module on one bad bridge,
            # surface the failure as a marker. The CLI prints them
            # alongside successful verdicts.
            out[b.name] = _error_evaluation(b.name, e)
    return out


def _error_evaluation(name: str, e: Exception) -> BridgeEvaluation:
    """Synthesize an error verdict so a single bad bridge doesn't
    abort the module run. Verdict is POWER_INSUFFICIENT — analyses
    that depend on this bridge will treat it as no-data, not
    pseudo-evidence."""
    from types import MappingProxyType

    from corroborate.verdict import Verdict
    return BridgeEvaluation(
        bridge_name=name,
        verdict=Verdict.POWER_INSUFFICIENT,
        analysis_results=MappingProxyType({'error': repr(e)}),
    )


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
    new_data = _load_data(data, restore_from_cloud=restore_from_cloud)

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


def _load_cache(path: Path | None) -> pl.DataFrame:
    if path is None or not path.exists():
        return pl.DataFrame()
    return pl.read_parquet(path)


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
    per-cell and add as a column. Existing columns are preserved."""
    if df.height == 0:
        return df
    missing = [m for m in required if m not in df.columns]
    if not missing:
        return df

    # Heterogeneous return types — build per-column lists then
    # construct Series. Per-cell evaluator caches transitive
    # measurables across calls, but only within a single cell.
    cells = df.to_dicts()
    new_cols: dict[str, list[object]] = {m: [] for m in missing}
    for cell in cells:
        per_cell_cache: dict[str, object] = {}
        for m in missing:
            mobj = get_registered(m)
            if mobj is None:
                new_cols[m].append(None)
                continue
            try:
                v = evaluate_with_measurables(
                    mobj.fn, cell, cache=per_cell_cache,
                )
            except Exception:  # noqa: BLE001
                # Record-level evaluation failure (missing inputs,
                # etc.) maps to None for this cell + measurable.
                # Downstream analyses NaN-skip these cells.
                v = None
            new_cols[m].append(v)

    return df.with_columns(
        [pl.Series(name, vals) for name, vals in new_cols.items()],
    )


# ============ Data loading ============


def _load_data(
    data: pl.DataFrame | Path | str | None,
    *,
    restore_from_cloud: bool,
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
        return _load_directory(p, restore_from_cloud=restore_from_cloud)
    if p.is_file():
        return pl.read_parquet(p)
    raise FileNotFoundError(f'no such data path: {data}')


def _load_directory(
    root: Path,
    *,
    restore_from_cloud: bool,
) -> pl.DataFrame:
    """Walk subdirs of `root`; for each subdir's `runs.parquet`,
    load it (auto-restore from s3 if local missing). Concat via
    `diagonal_relaxed` so heterogeneous schemas null-pad."""
    frames: list[pl.DataFrame] = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        runs_path = sub / 'runs.parquet'
        manifest = sub / '_remote.json'
        if not runs_path.exists() and manifest.exists():
            if restore_from_cloud:
                from corroborate.cloud import restore
                print(
                    f'runner: restoring {sub.name} from cloud...',
                    file=sys.stderr,
                )
                restore(sub)
            else:
                print(
                    f'runner: WARNING — {sub.name} has _remote.json '
                    f'but no local runs.parquet; restore disabled',
                    file=sys.stderr,
                )
                continue
        if not runs_path.exists():
            continue
        df = pl.read_parquet(runs_path)
        if 'corpus' not in df.columns:
            df = df.with_columns(pl.lit(sub.name).alias('corpus'))
        frames.append(df)
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how='diagonal_relaxed')


__all__ = [
    'HypothesisModule',
    'run_module',
]
