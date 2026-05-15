"""Inventory of all per-sweep corpora under a `data_root`, optionally
cross-referenced against cloud archives under a `remote_prefix`.

The catalogue is a discovery primitive: it answers "what data
exists, and where, and in what state?" without taking any
actions. Each row carries typed `local` and `cloud` slices
(`LocalCorpusInfo | None`, `CloudCorpusInfo | None`), plus a
`status` discriminator naming the cell in the cloud × local
matrix. `in_progress` overlays orthogonally as a bool.

Provenance awareness: `runs.parquet` is the per-cell provenance
store (RunRow / TraceRow join by UUID). The catalogue surfaces
both sides at runs-parquet granularity: cloud
`runs_n_row_ids` (from the manifest's `runs.parquet` entry
specifically — NOT summed across all manifest files, which would
conflate runs/traces/measurements id sets) AND local
`runs_row_count` (read from `runs.parquet.id` via
`cloud.sniff_row_ids`). Mismatch on `CLOUD_AND_LOCAL` rows is the
drift signal.

Heavier integrity work (cross-checking the actual id sets of
runs vs traces) lives in `integrity.audit_trace_contamination`
(integrity.py). The catalogue composes the lightweight side
(row count); call `audit_trace_contamination` directly for full
CI8-grade verification of any specific corpus.
"""
from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import polars as pl

from corroborate.corpus import cloud
from corroborate.corpus.integrity import is_in_progress
from corroborate._internals import fsspec as _fs


# ============ Dataclasses ============

@dataclass(frozen=True, slots=True)
class LocalCorpusInfo:
    """What the catalogue learned about a corpus from local disk."""
    path: Path
    has_manifest: bool
    parquet_count: int
    runs_row_count: int | None
    """Height of `runs.parquet`'s `id` column. None when
    runs.parquet is absent. 0 when present but the `id` column
    is missing OR the file has zero rows (mirrors
    `cloud.sniff_row_ids` semantics — same `()` return for
    both cases)."""


@dataclass(frozen=True, slots=True)
class CloudCorpusInfo:
    """What the catalogue learned about a corpus from its cloud manifest."""
    remote_root: str
    n_files: int
    total_bytes: int
    runs_n_row_ids: int | None
    """row_ids count from the manifest's runs.parquet entry
    specifically. None when no runs.parquet in the manifest."""
    latest_pushed_at: str
    """Raw ISO-8601 from `RemoteFile.pushed_at`. Caller can
    `pl.col(...).str.to_datetime()` if needed."""


_Status = Literal[
    'CLOUD_AND_LOCAL', 'CLOUD_EVICTED', 'STALE_MANIFEST',
    'LOCAL_ONLY', 'CLOUD_ORPHAN', 'LINKAGE_LOST',
    'IN_PROGRESS_SCAFFOLD',
]
_Kind = Literal['corpus', 'misc']


@dataclass(frozen=True, slots=True)
class CorpusInventoryRow:
    """One inventory row. `local` / `cloud` slices are None when
    the corresponding side is absent or unqueried."""
    name: str
    parent: str
    status: _Status
    kind: _Kind
    in_progress: bool
    local: LocalCorpusInfo | None
    cloud: CloudCorpusInfo | None


# ============ Walk constants ============

_NON_CORPUS_DIRS = frozenset({'cache', '_old_logs'})
"""Dirs that may contain top-level parquets but aren't sweep
corpora. `cache/` carries aggregated analysis outputs;
`_old_logs/` carries `.log` files. Both pass the
"parquet-presence" qualifier and need a by-name skip.
Tagged `kind='misc'`; filtered unless `include_misc=True`."""

_PRUNE_DIRS = frozenset({'tmp'})
"""Directories never descended into. `tmp/<arm>/` are per-arm
shard dirs from in-progress sweeps; the merge step removes them
post-sweep but a crashed sweep leaves them behind. They aren't
corpora."""

_MAX_DEPTH = 2
"""data_root + corpus + sub-corpus. The live tree has no
3-level-deep nesting (`find … -mindepth 4 -name _remote.json`
is empty)."""


# ============ Polars schema (exported) ============

POLARS_SCHEMA: Mapping[str, pl.DataType | type[pl.DataType]] = {
    'parent':              pl.String,
    'name':                pl.String,
    'status':              pl.String,
    'kind':                pl.String,
    'in_progress':         pl.Boolean,
    'local_path':          pl.String,
    'local_has_manifest':  pl.Boolean,
    'parquet_count':       pl.UInt32,
    'runs_row_count':      pl.UInt64,
    'remote_root':         pl.String,
    'n_files':             pl.UInt32,
    'total_bytes':         pl.UInt64,
    'runs_n_row_ids':      pl.UInt64,
    'latest_pushed_at':    pl.String,
}


# ============ Classifier ============

def _classify(
    *,
    has_local_dir: bool,
    has_local_manifest: bool,
    has_local_parquets: bool,
    has_cloud_manifest: bool,
) -> _Status:
    """Pure classification of one corpus state from four bool inputs.

    Six of the 16 cells (those with `local_dir=F` AND
    `manifest=T` OR `parquets=T`) are UNREACHABLE by filesystem
    ontology and raise. One cell (`F, F, F, F`) is filtered by
    the walk qualifier and also raises.
    """
    if not has_local_dir:
        if has_local_manifest or has_local_parquets:
            raise AssertionError(
                f'unreachable: local_dir=F but '
                f'has_local_manifest={has_local_manifest} '
                f'has_local_parquets={has_local_parquets}',
            )
        if not has_cloud_manifest:
            raise AssertionError(
                'upstream qualifier filters this cell',
            )
        return 'CLOUD_ORPHAN'

    if has_local_manifest and has_cloud_manifest:
        return 'CLOUD_AND_LOCAL' if has_local_parquets else 'CLOUD_EVICTED'
    if has_local_manifest:
        return 'STALE_MANIFEST'
    if has_cloud_manifest:
        return 'LINKAGE_LOST' if has_local_parquets else 'CLOUD_ORPHAN'
    if has_local_parquets:
        return 'LOCAL_ONLY'
    return 'IN_PROGRESS_SCAFFOLD'


# ============ Internal discovery record ============

@dataclass(frozen=True, slots=True)
class _LocalDiscovery:
    parent: str
    name: str
    dir_path: Path
    has_manifest: bool
    parquet_count: int
    runs_row_count: int | None
    in_progress: bool
    kind: _Kind


def _walk_local(data_root: Path) -> tuple[_LocalDiscovery, ...]:
    data_root = data_root.resolve()
    found: list[_LocalDiscovery] = []

    for dirpath_str, dirnames, _filenames in os.walk(
        data_root, topdown=True,
    ):
        # Prune (in-place per topdown contract).
        dirnames[:] = [d for d in dirnames if d not in _PRUNE_DIRS]

        dirpath = Path(dirpath_str)
        try:
            rel = dirpath.relative_to(data_root)
        except ValueError:
            continue
        depth = len(rel.parts)

        # Cap descent at _MAX_DEPTH (do this BEFORE the skip-root
        # check so root-level dirs are visited but their children
        # are pruned at depth 2).
        if depth >= _MAX_DEPTH:
            dirnames[:] = []

        if depth == 0:
            continue  # data_root itself

        has_manifest = (dirpath / cloud.MANIFEST_NAME).exists()
        parquet_count = sum(1 for _ in dirpath.glob('*.parquet'))
        in_progress_flag = is_in_progress(dirpath)

        # Qualifier: emit only if there's something to inventory.
        if not (has_manifest or parquet_count > 0 or in_progress_flag):
            continue

        kind: _Kind = 'misc' if rel.parts[0] in _NON_CORPUS_DIRS else 'corpus'

        runs_path = dirpath / 'runs.parquet'
        if runs_path.exists():
            runs_row_count: int | None = len(
                cloud.sniff_row_ids(runs_path),
            )
        else:
            runs_row_count = None

        parent_str = str(rel.parent) if rel.parent != Path('.') else ''
        found.append(_LocalDiscovery(
            parent=parent_str,
            name=rel.name,
            dir_path=dirpath,
            has_manifest=has_manifest,
            parquet_count=parquet_count,
            runs_row_count=runs_row_count,
            in_progress=in_progress_flag,
            kind=kind,
        ))

    return tuple(found)


# ============ Cloud discovery ============

def _list_archives_two_level(prefix: str) -> tuple[str, ...]:
    """Return remote_root URIs with MANIFEST.json reachable, up to
    one level of nesting under `prefix`. Top-level archives come
    from `cloud.list_archives`; nested ones from a second LIST on
    each top-level child that lacks its own MANIFEST.json."""
    if not prefix.endswith('/'):
        prefix = prefix + '/'
    direct = {r.rstrip('/') for r in cloud.list_archives(prefix)}
    out: set[str] = set(direct)
    all_direct = _fs.remote_list_dir(prefix)
    for child in all_direct:
        child_clean = child.rstrip('/')
        if child_clean in direct:
            continue
        for grandchild in cloud.list_archives(child_clean):
            out.add(grandchild.rstrip('/'))
    return tuple(sorted(out))


def _build_cloud_info(manifest: cloud.RemoteManifest) -> CloudCorpusInfo:
    files = manifest.files
    n_files = len(files)
    total_bytes = sum(f.size_bytes for f in files)
    runs_n_row_ids: int | None = None
    for f in files:
        if f.relpath == 'runs.parquet':
            runs_n_row_ids = len(f.row_ids)
            break
    latest_pushed_at = max(f.pushed_at for f in files) if files else ''
    return CloudCorpusInfo(
        remote_root=manifest.remote_root,
        n_files=n_files,
        total_bytes=total_bytes,
        runs_n_row_ids=runs_n_row_ids,
        latest_pushed_at=latest_pushed_at,
    )


def _remote_root_to_parent_name(
    rr: str, prefix: str,
) -> tuple[str, str] | None:
    """Recover (parent, name) for a remote_root relative to `prefix`.
    Returns None when the root doesn't sit under the prefix."""
    prefix_clean = prefix.rstrip('/')
    if not rr.startswith(prefix_clean):
        return None
    inner = rr[len(prefix_clean):].lstrip('/')
    if not inner:
        return None
    parts = inner.split('/')
    return '/'.join(parts[:-1]), parts[-1]


# ============ Entry point ============

def catalogue(
    data_root: Path,
    remote_prefix: str | None = None,
    *,
    include_misc: bool = False,
) -> tuple[CorpusInventoryRow, ...]:
    """Inventory all corpora under `data_root` (recursively, up to
    one level of nesting).

    When `remote_prefix` is provided, cross-reference each corpus
    against cloud archives discoverable under that prefix
    (one level of nesting deep). `None` skips cloud discovery
    entirely — statuses then only span the local-derivable subset
    ({LOCAL_ONLY, STALE_MANIFEST, IN_PROGRESS_SCAFFOLD}) and
    `STALE_MANIFEST` means "local manifest present, cloud
    unverified" rather than "cloud verified absent".

    `include_misc=False` filters out `kind='misc'` rows (cache,
    log dirs). Pass True to surface them.
    """
    locals_found = _walk_local(data_root)

    cloud_by_root: dict[str, cloud.RemoteManifest] = {}
    if remote_prefix is not None:
        for r in _list_archives_two_level(remote_prefix):
            m = cloud.fetch_remote_manifest(r)
            if m is not None:
                cloud_by_root[r] = m

    matched_cloud_roots: set[str] = set()
    rows: list[CorpusInventoryRow] = []

    for d in locals_found:
        cloud_manifest: cloud.RemoteManifest | None = None

        if d.has_manifest:
            local_m = cloud.load_manifest(d.dir_path)
            if local_m is not None:
                rr = local_m.remote_root.rstrip('/')
                cloud_manifest = cloud_by_root.get(rr)
                if cloud_manifest is not None:
                    matched_cloud_roots.add(rr)
        elif remote_prefix is not None:
            # Name-match: local has parquets/sentinel but no _remote.json;
            # see if a cloud archive lives at the matching path.
            for rr, m in cloud_by_root.items():
                pn = _remote_root_to_parent_name(rr, remote_prefix)
                if pn == (d.parent, d.name):
                    cloud_manifest = m
                    matched_cloud_roots.add(rr)
                    break

        status = _classify(
            has_local_dir=True,
            has_local_manifest=d.has_manifest,
            has_local_parquets=d.parquet_count > 0,
            has_cloud_manifest=cloud_manifest is not None,
        )
        local_info = LocalCorpusInfo(
            path=d.dir_path,
            has_manifest=d.has_manifest,
            parquet_count=d.parquet_count,
            runs_row_count=d.runs_row_count,
        )
        cloud_info = (
            _build_cloud_info(cloud_manifest)
            if cloud_manifest is not None else None
        )
        rows.append(CorpusInventoryRow(
            name=d.name,
            parent=d.parent,
            status=status,
            kind=d.kind,
            in_progress=d.in_progress,
            local=local_info,
            cloud=cloud_info,
        ))

    # Cloud-only orphans (no matching local discovery).
    if remote_prefix is not None:
        for rr, m in cloud_by_root.items():
            if rr in matched_cloud_roots:
                continue
            pn = _remote_root_to_parent_name(rr, remote_prefix)
            if pn is None:
                continue
            parent, name = pn
            rows.append(CorpusInventoryRow(
                name=name,
                parent=parent,
                status='CLOUD_ORPHAN',
                kind='corpus',
                in_progress=False,
                local=None,
                cloud=_build_cloud_info(m),
            ))

    if not include_misc:
        rows = [r for r in rows if r.kind == 'corpus']

    rows.sort(key=lambda r: (r.parent, r.name))
    return tuple(rows)


# ============ Polars view ============

def _row_to_dict(r: CorpusInventoryRow) -> dict[str, object]:
    if r.local is not None:
        local_path: str | None = str(r.local.path)
        local_has_manifest: bool | None = r.local.has_manifest
        parquet_count: int | None = r.local.parquet_count
        runs_row_count: int | None = r.local.runs_row_count
    else:
        local_path = None
        local_has_manifest = None
        parquet_count = None
        runs_row_count = None

    if r.cloud is not None:
        remote_root: str | None = r.cloud.remote_root
        n_files: int | None = r.cloud.n_files
        total_bytes: int | None = r.cloud.total_bytes
        runs_n_row_ids: int | None = r.cloud.runs_n_row_ids
        latest_pushed_at: str | None = r.cloud.latest_pushed_at
    else:
        remote_root = None
        n_files = None
        total_bytes = None
        runs_n_row_ids = None
        latest_pushed_at = None

    return {
        'parent': r.parent,
        'name': r.name,
        'status': r.status,
        'kind': r.kind,
        'in_progress': r.in_progress,
        'local_path': local_path,
        'local_has_manifest': local_has_manifest,
        'parquet_count': parquet_count,
        'runs_row_count': runs_row_count,
        'remote_root': remote_root,
        'n_files': n_files,
        'total_bytes': total_bytes,
        'runs_n_row_ids': runs_n_row_ids,
        'latest_pushed_at': latest_pushed_at,
    }


def to_polars(rows: Sequence[CorpusInventoryRow]) -> pl.DataFrame:
    """Flatten slice composition into a polars DataFrame.

    The schema is pinned (`POLARS_SCHEMA`) so dtype inference can't
    drift on a heterogeneous row set with many None columns.
    """
    flat = [_row_to_dict(r) for r in rows]
    return pl.DataFrame(flat, schema=POLARS_SCHEMA)
