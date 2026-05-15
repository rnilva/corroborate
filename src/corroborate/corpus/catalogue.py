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
from typing import Literal, cast

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
    data_root: Path | Sequence[Path],
    remote_prefix: str | None = None,
    *,
    include_misc: bool = False,
) -> tuple[CorpusInventoryRow, ...]:
    """Inventory all corpora under `data_root` (recursively, up to
    one level of nesting).

    `data_root` accepts either a single `Path` or a sequence of
    `Path`s. The project convention has two corpus-bearing roots
    (`experiments/data/` for canonical sweeps + `experiments/probes/`
    for ad-hoc pilots); pass both to avoid false-orphan reports on
    corpora that live in one root but were `cloud.archive`d to a
    prefix the other root's walk doesn't surface. Local discoveries
    are deduplicated by absolute resolved path; cloud orphans are
    deduplicated by `remote_root`.

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
    roots: tuple[Path, ...] = (
        (data_root,) if isinstance(data_root, Path) else tuple(data_root)
    )
    locals_by_path: dict[Path, _LocalDiscovery] = {}
    for root in roots:
        for d in _walk_local(root):
            key = d.dir_path.resolve()
            locals_by_path.setdefault(key, d)
    locals_found: tuple[_LocalDiscovery, ...] = tuple(locals_by_path.values())

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


# ============ Arm/leaves view (per-cell content) ============

@dataclass(frozen=True, slots=True)
class ArmLeafProfile:
    """Per-(corpus, arm) configurational fingerprint, sourced from
    `runs.parquet` columns. `leaves` maps each leaf path to a sorted
    tuple of distinct values observed across that arm's cells —
    length 1 for constant leaves; longer for sweep arms (each
    distinct value, stringified)."""
    corpus: str                              # parent/name addr
    arm: str
    n_cells: int
    envs: tuple[str, ...]                    # sorted distinct env_name values
    leaves: Mapping[str, tuple[str, ...]]


# Framework-typed RunRow fields. Single source of truth lives in
# schema.py — auto-derived from `dataclasses.fields(RunRow)` so a
# new typed field on RunRow propagates here without manual sync.
from corroborate.corpus.schema import (  # noqa: E402,PLC0415
    _RUN_ROW_TYPED_FIELDS as _RUNROW_FIELDS,  # pyright: ignore[reportPrivateUsage]
)

# Default substrate-exogenous keys for the RL substrate. Mirrors
# the `Annotated[..., Exogenous]` set declared on the `dqn` claim
# (corroborate_rl.dqn.dqn:dqn). The framework doesn't hardcode RL
# keys; this default is a convenience for the only substrate that
# currently uses this view. Callers with other substrates pass
# their own `exogenous_keys` + `exogenous_prefixes`. NOTE: keys
# like `total_steps`, `eval_every`, `n_episodes` are NOT in this
# set — they're plain int defaults on the dqn claim, NOT
# Exogenous, so they ARE leaves per the framework's vocabulary.
_DEFAULT_EXOGENOUS_KEYS: frozenset[str] = frozenset({
    'env_name', 'seed', 'wrappers',
    'env', 'env_params', 'obs_shape', 'n_actions',
    'state_hash', 'eval_episode_cap',
})
_DEFAULT_EXOGENOUS_PREFIXES: tuple[str, ...] = (
    'env_params.', 'env.',
)


def _leaf_columns(
    columns: Sequence[str],
    dtypes: Mapping[str, pl.DataType],
    *,
    exogenous_keys: frozenset[str],
    exogenous_prefixes: tuple[str, ...],
    measurable_names: frozenset[str],
) -> tuple[str, ...]:
    """Filter a parquet's column list down to leaf columns —
    excluding framework-typed RunRow fields, registered measurables,
    substrate-declared exogenous keys, trajectory (List-dtype)
    columns, and bundle-placeholder columns (the `optimizer`
    column when `optimizer.inner.lr` exists)."""
    candidates: list[str] = []
    for c in columns:
        if c in _RUNROW_FIELDS or c in exogenous_keys or c in measurable_names:
            continue
        if any(c.startswith(p) for p in exogenous_prefixes):
            continue
        dt = dtypes.get(c)
        if isinstance(dt, pl.List):
            continue
        candidates.append(c)
    # Drop bundle placeholders: column X where X+'.' is a prefix of
    # some other candidate.
    cand_set = set(candidates)
    final: list[str] = []
    for c in candidates:
        if any(other.startswith(c + '.') for other in cand_set):
            continue
        final.append(c)
    return tuple(sorted(final))


def _scalar_to_str(v: object) -> str | None:
    """Stringify a parquet cell value for leaf signature use.
    Tuples / lists rendered as `(x,y,z)` for parity with the
    `q_network.hidden = (64, 64)` pattern. Returns None for null
    inputs so callers can filter empties."""
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        # Use 'None' for null elements so the rendering is
        # unambiguous (avoids '(1,,3)' colliding with '(1,"",3)').
        parts: list[str] = []
        for x in v:
            if x is None:
                parts.append('None')
            else:
                s = _scalar_to_str(x)
                parts.append(s if s is not None else 'None')
        return '(' + ','.join(parts) + ')'
    return str(v)


def _sort_leaf_values(values: set[str]) -> tuple[str, ...]:
    """Sort distinct stringified leaf values. If every value parses
    as a float, sort numerically; else fall back to lexicographic.
    Fixes the `('1','10','2',...)` lexicographic-on-numeric quirk."""
    try:
        floats = [(float(v), v) for v in values]
    except ValueError:
        return tuple(sorted(values))
    return tuple(v for _, v in sorted(floats))


def arm_leaves(
    data_root: Path | Sequence[Path],
    *,
    include_misc: bool = False,
    exogenous_keys: frozenset[str] | None = None,
    exogenous_prefixes: tuple[str, ...] | None = None,
) -> tuple[ArmLeafProfile, ...]:
    """Per-(corpus, arm) leaf profile across local corpora.

    Reads `runs.parquet` for every catalogue row whose `local` slice
    has parquets on disk (CLOUD_AND_LOCAL / LOCAL_ONLY / etc.).
    Returns one `ArmLeafProfile` per distinct (corpus, arm_key) pair.
    The `leaves` mapping carries the distinct stringified values
    each leaf takes across that arm's cells: length-1 for constant
    arms, longer for sweeps.

    Reads are column-projected for speed: only `arm_key`,
    `env_name`, and leaf candidates are decoded.

    `exogenous_keys` and `exogenous_prefixes` let the caller
    override the default RL substrate's exogenous-key set
    (cf. CLAUDE.md: substrate declares exogenous via
    `Annotated[T, Exogenous]`; framework doesn't hardcode).
    """
    from corroborate.measurables import registered_names
    ex_keys = (exogenous_keys
               if exogenous_keys is not None else _DEFAULT_EXOGENOUS_KEYS)
    ex_pre = (exogenous_prefixes
              if exogenous_prefixes is not None else _DEFAULT_EXOGENOUS_PREFIXES)
    measurables = frozenset(registered_names())

    rows = catalogue(data_root, remote_prefix=None,
                     include_misc=include_misc)
    profiles: list[ArmLeafProfile] = []
    for r in rows:
        if r.local is None or r.local.parquet_count == 0:
            continue
        path = r.local.path / 'runs.parquet'
        if not path.exists():
            continue
        # Read schema first (cheap), pick columns, then read those.
        try:
            schema = pl.scan_parquet(path).collect_schema()
        except (pl.exceptions.ComputeError, FileNotFoundError, OSError):
            continue
        col_names = schema.names()
        dtypes: dict[str, pl.DataType] = dict(zip(col_names, schema.dtypes()))
        leaves = _leaf_columns(
            col_names, dtypes,
            exogenous_keys=ex_keys,
            exogenous_prefixes=ex_pre,
            measurable_names=measurables,
        )
        # Legacy parquets may not carry `arm_key` (RunRow defaults
        # to 'baseline'; pre-arm-key corpora omit the column entirely
        # — same convention `RunRow.from_row_dict` honors at
        # schema.py:269+). Inject the default rather than crashing.
        has_arm_key = 'arm_key' in col_names
        wanted: list[str] = []
        if has_arm_key:
            wanted.append('arm_key')
        if 'env_name' in col_names:
            wanted.append('env_name')
        wanted.extend(leaves)
        try:
            df = pl.read_parquet(path, columns=wanted)
        except (pl.exceptions.ComputeError,
                pl.exceptions.ColumnNotFoundError,
                FileNotFoundError, OSError):
            continue
        if not has_arm_key:
            df = df.with_columns(
                pl.lit('baseline').alias('arm_key'),
            )
        addr = f'{r.parent}/{r.name}' if r.parent else r.name
        arm_values = cast(list[object], df['arm_key'].to_list())
        for arm in sorted({str(a) for a in arm_values if a is not None}):
            sub = df.filter(pl.col('arm_key') == arm)
            if 'env_name' in sub.columns:
                env_values = cast(list[object], sub['env_name'].to_list())
                envs = tuple(sorted(
                    {str(e) for e in env_values if e is not None}
                ))
            else:
                envs = ()
            leaf_map: dict[str, tuple[str, ...]] = {}
            for leaf in leaves:
                leaf_vals = cast(list[object], sub[leaf].to_list())
                vals = {
                    s for s in (_scalar_to_str(v) for v in leaf_vals)
                    if s is not None
                }
                if vals:
                    leaf_map[leaf] = _sort_leaf_values(vals)
            profiles.append(ArmLeafProfile(
                corpus=addr, arm=arm, n_cells=sub.height,
                envs=envs, leaves=leaf_map,
            ))
    profiles.sort(key=lambda p: (p.corpus, p.arm))
    return tuple(profiles)


_LEAVES_LONG_SCHEMA: Mapping[str, pl.DataType | type[pl.DataType]] = {
    'corpus':     pl.String,
    'arm':        pl.String,
    'n_cells':    pl.UInt32,
    'envs':       pl.String,
    'leaf_path':  pl.String,
    'leaf_value': pl.String,
    'n_values':   pl.UInt32,
}


def arm_leaves_to_polars_long(
    profiles: Sequence[ArmLeafProfile],
) -> pl.DataFrame:
    """One row per `(corpus, arm, leaf_path, leaf_value)` —
    sweep arms produce multiple rows for the same leaf. Suitable
    for filter / aggregate queries: `df.filter(pl.col('leaf_path')
    == 'gamma')` etc."""
    flat: list[dict[str, object]] = []
    for p in profiles:
        envs = ','.join(p.envs)
        for leaf_path, values in sorted(p.leaves.items()):
            for v in values:
                flat.append({
                    'corpus':     p.corpus,
                    'arm':        p.arm,
                    'n_cells':    p.n_cells,
                    'envs':       envs,
                    'leaf_path':  leaf_path,
                    'leaf_value': v,
                    'n_values':   len(values),
                })
    return pl.DataFrame(flat, schema=_LEAVES_LONG_SCHEMA)


def arm_leaves_to_polars_wide(
    profiles: Sequence[ArmLeafProfile],
) -> pl.DataFrame:
    """One row per `(corpus, arm)` with each leaf as its own
    column. Sweep leaves collapse to `'MULTI:[v1,v2,...]'` strings.
    Sparse — many nulls when leaves are corpus-specific. Useful
    for at-a-glance scan; long-format is better for queries."""
    all_leaves: list[str] = []
    seen: set[str] = set()
    for p in profiles:
        for leaf in sorted(p.leaves.keys()):
            if leaf not in seen:
                seen.add(leaf); all_leaves.append(leaf)
    flat: list[dict[str, object]] = []
    for p in profiles:
        row: dict[str, object] = {
            'corpus':  p.corpus,
            'arm':     p.arm,
            'n_cells': p.n_cells,
            'envs':    ','.join(p.envs),
        }
        for leaf in all_leaves:
            vals = p.leaves.get(leaf, ())
            if not vals:
                row[leaf] = None
            elif len(vals) == 1:
                row[leaf] = vals[0]
            else:
                row[leaf] = f'MULTI:[{",".join(vals)}]'
        flat.append(row)
    schema: dict[str, pl.DataType | type[pl.DataType]] = {
        'corpus':  pl.String,
        'arm':     pl.String,
        'n_cells': pl.UInt32,
        'envs':    pl.String,
    }
    for leaf in all_leaves:
        schema[leaf] = pl.String
    return pl.DataFrame(flat, schema=schema)
