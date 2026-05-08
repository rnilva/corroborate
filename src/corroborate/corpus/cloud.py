"""Cloud archive — push/pull sweep parquets to fsspec-backed
remote storage with a per-sweep manifest.

Lifecycle (typical, two-step):

    archive(sweep_dir, "s3://bucket/sweeps/ddqn")  # upload
    purge(sweep_dir)                                # free local disk
    # ... later ...
    restore(sweep_dir)                              # pull back

Or one-shot: `archive(..., purge_local=True)`.

The manifest at `<sweep_dir>/_remote.json` records the
`remote_root` URI and one entry per archived file with
`relpath`, `size_bytes`, `sha256`, and `pushed_at` (ISO-8601
UTC). Manifest is saved atomically (tmp + rename) after each
file so a crash mid-batch leaves consistent partial state.

Idempotency: re-archiving a file already in the manifest with
matching sha256 is a no-op (unless `force=True`). Re-restoring a
file already present locally with matching sha256 is also a
no-op.

Verification: `archive` re-stats the remote object after upload
and aborts if the size disagrees. `restore` recomputes sha256
post-download and aborts if it disagrees with the manifest.

Provider neutrality: all I/O goes through fsspec via
`_fsspec_boundary`. Backends supported out of the box: local,
memory (testing). Cloud backends (s3://, gs://, az://, r2://)
require their respective fsspec extras (`s3fs`, `gcsfs`,
`adlfs`, `s3fs` with R2 endpoint) — install separately when you
pin a target."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

from corroborate._internals import fsspec as _fs
from corroborate._internals.json import loads as _json_loads
from corroborate._internals.narrow import (
    is_list_of_object,
    is_mapping_str_object,
    require_int,
    require_str,
)


MANIFEST_NAME = '_remote.json'


class ConflictingArchive(RuntimeError):
    """Invariant I2 (SWEEP_PERSISTENCY.md): an upload would
    overwrite a previously-archived object with different content.

    Raised by `archive()` when the local file's sha256 differs
    from the manifest's prior entry for the same relpath. The
    user must explicitly opt into overwrite by passing
    `force=True` (the existing parameter); otherwise a re-run
    that produced different bytes (substrate code change, RNG
    drift, deliberate corpus refresh) silently last-writer-wins.

    `relpath`, `prior_sha256`, and `local_sha256` are surfaced so
    the user can decide whether the new bytes should replace the
    old, or whether they ran the wrong code."""

    def __init__(
        self,
        relpath: str,
        *,
        prior_sha256: str,
        local_sha256: str,
        remote_uri: str,
    ) -> None:
        super().__init__(
            f'archive conflict at {relpath!r}: '
            f'manifest sha256={prior_sha256[:12]}…, '
            f'local sha256={local_sha256[:12]}… '
            f'(remote={remote_uri}). Pass force=True to '
            f'explicitly overwrite, or investigate the '
            f'content drift.',
        )
        self.relpath = relpath
        self.prior_sha256 = prior_sha256
        self.local_sha256 = local_sha256
        self.remote_uri = remote_uri


@dataclass(frozen=True, slots=True)
class RemoteFile:
    """One archived file's manifest entry. `relpath` is relative
    to the sweep directory; `size_bytes` and `sha256` are
    computed pre-upload from the local file; `pushed_at` is
    ISO-8601 UTC.

    `row_ids` (invariant I5 in SWEEP_PERSISTENCY.md): when the
    archived file is a parquet carrying an `id` column (RunRow
    or TraceRow shards), the list of IDs it contains. Empty for
    non-parquet files OR parquets without an `id` column.
    Enables `id → shard → cell address` traceability when
    debugging anomalous rows in a merged corpus. JSON-omitted
    when empty to keep older manifests round-trippable."""

    relpath: str
    size_bytes: int
    sha256: str
    pushed_at: str
    row_ids: tuple[str, ...] = ()

    def as_dict(self) -> Mapping[str, object]:
        out: dict[str, object] = {
            'relpath': self.relpath,
            'size_bytes': self.size_bytes,
            'sha256': self.sha256,
            'pushed_at': self.pushed_at,
        }
        # Omit empty `row_ids` so manifests for non-parquet shards
        # don't carry an empty list, and so old manifests written
        # before this field existed remain byte-identical post-rewrite.
        if self.row_ids:
            out['row_ids'] = list(self.row_ids)
        return out

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        # row_ids is optional for backward compatibility with
        # manifests written before invariant I5 landed.
        raw_ids = d.get('row_ids')
        if raw_ids is None:
            row_ids: tuple[str, ...] = ()
        elif isinstance(raw_ids, list):
            row_ids = tuple(
                str(item) for item in raw_ids if isinstance(item, str)
            )
        else:
            raise TypeError(
                f"manifest 'row_ids' must be list or absent, got "
                f"{type(raw_ids).__name__}",
            )
        return cls(
            relpath=require_str(d, 'relpath'),
            size_bytes=require_int(d, 'size_bytes'),
            sha256=require_str(d, 'sha256'),
            pushed_at=require_str(d, 'pushed_at'),
            row_ids=row_ids,
        )


@dataclass(frozen=True, slots=True)
class RemoteManifest:
    """Per-sweep manifest. `remote_root` is the fsspec URI prefix
    (e.g. `s3://bucket/sweeps/ddqn`); each `files` entry's
    `relpath` joins onto it."""

    remote_root: str
    files: tuple[RemoteFile, ...]

    def as_dict(self) -> Mapping[str, object]:
        return {
            'remote_root': self.remote_root,
            'files': [dict(f.as_dict()) for f in self.files],
        }

    def relpaths(self) -> frozenset[str]:
        """All archived relpaths, as a hashable set for membership
        / set-arithmetic queries — saves consumers the
        `{f.relpath for f in m.files}` boilerplate."""
        return frozenset(f.relpath for f in self.files)

    def has(self, relpath: str) -> bool:
        """Whether this manifest archives `relpath`. Equivalent to
        `relpath in self.relpaths()` but avoids materialising the
        set when the caller only needs one membership check."""
        return any(f.relpath == relpath for f in self.files)

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        files_raw = d.get('files')
        if not is_list_of_object(files_raw):
            raise TypeError(
                f"manifest 'files' must be list, got "
                f"{type(files_raw).__name__}",
            )
        files: list[RemoteFile] = []
        for item in files_raw:
            if not is_mapping_str_object(item):
                raise TypeError(
                    f"manifest 'files' entry must be mapping, got "
                    f"{type(item).__name__}",
                )
            files.append(RemoteFile.from_dict(item))
        return cls(
            remote_root=require_str(d, 'remote_root'),
            files=tuple(files),
        )


# ============ Manifest I/O ============

def _manifest_path(sweep_dir: Path) -> Path:
    return sweep_dir / MANIFEST_NAME


def _load_manifest(sweep_dir: Path) -> RemoteManifest | None:
    """Return the existing manifest, or None if no archive yet
    exists for `sweep_dir`."""
    p = _manifest_path(sweep_dir)
    if not p.exists():
        return None
    raw = _json_loads(p.read_text())
    if not is_mapping_str_object(raw):
        raise TypeError(
            f'{p}: expected JSON object, got {type(raw).__name__}',
        )
    return RemoteManifest.from_dict(raw)


def _save_manifest(sweep_dir: Path, manifest: RemoteManifest) -> None:
    """Atomically write the manifest. tmp + rename ensures
    readers never see a half-written file even if the process
    crashes mid-archive."""
    p = _manifest_path(sweep_dir)
    payload = json.dumps(manifest.as_dict(), indent=2)
    tmp = p.with_suffix(p.suffix + '.tmp')
    _ = tmp.write_text(payload)
    tmp.replace(p)


# ============ Helpers ============

def _sniff_row_ids(source: Path | str) -> tuple[str, ...]:
    """Read the `id` column from a parquet file or URI if present;
    return empty tuple otherwise. The provenance breadcrumb for
    invariant I5: per-shard `RunRow.id` lists land in the manifest
    so a merged-corpus row can be traced back to its source shard.

    Accepts either a local `Path` or an fsspec URI string
    (e.g. `s3://bucket/sweeps/.../runs.parquet`). Polars' parquet
    reader handles both transparently via fsspec, with column
    projection pushdown — only the `id` column's pages are
    fetched on remote reads.

    Quietly returns `()` for non-parquet inputs, parquets without
    an `id` column, and parquets whose `id` column doesn't contain
    strings — the framework is robust to non-row-shaped parquets
    in the manifest (e.g. graph sidecars when those eventually
    land in the same archive)."""
    if isinstance(source, Path):
        if source.suffix != '.parquet':
            return ()
        target: Path | str = source
    else:
        if not source.endswith('.parquet'):
            return ()
        target = source
    import polars as pl
    try:
        df = pl.read_parquet(target, columns=['id'])
    except (pl.exceptions.ColumnNotFoundError, pl.exceptions.ComputeError):
        return ()
    except FileNotFoundError:
        return ()
    if df.height == 0:
        return ()
    raw = df['id'].to_list()
    return tuple(str(x) for x in raw if isinstance(x, str))


def _sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """Compute sha256 of a local file. Streamed in 1 MiB chunks
    so memory stays bounded for arbitrarily large files."""
    h = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _join_remote(remote_root: str, relpath: str) -> str:
    return f'{remote_root.rstrip("/")}/{relpath.lstrip("/")}'


def _default_files(sweep_dir: Path) -> list[str]:
    """Default selection: top-level `*.parquet` files in the
    sweep directory (non-recursive). The `tmp/` shard
    subdirectory is intentionally excluded — when present, the
    merged top-level parquet supersedes the shards. Users who
    want shards too pass `--files tmp/<arm>...` explicitly."""
    return sorted(
        p.name for p in sweep_dir.iterdir()
        if p.is_file() and p.suffix == '.parquet'
    )


def _warn_if_trace_schema_incomplete(traces_path: Path) -> None:
    """Warn if `traces.parquet` is missing trace columns currently
    declared as `reads` by registered measurables.

    The cloud archive is the durable record. If a sweep archives
    traces lacking columns that future ingests will need (e.g. a
    new reduction was added to the substrate after this sweep
    ran), those ingests will produce all-NaN values on this
    corpus's cells without ever surfacing why. The check makes the
    gap loud at archive time so the substrate author can decide:
    re-archive after fixing, or accept the partial schema (some
    measurables won't compute on this corpus).

    Walks the registered measurable graph for `transitive_reads`
    that look like trace columns (per-step / per-burst arrays —
    heuristically: not `gamma`, `seed`, `env_name`, etc.) and
    diffs against the trace's actual schema.

    Skips silently if polars schema read fails (the file is
    already past CI5 validation but compute_schema can hit
    edge-cases on certain compression formats); the warning is
    advisory."""
    try:
        import polars as pl
        present = set(pl.scan_parquet(traces_path).collect_schema().names())
    except Exception:
        return
    from corroborate.measurables.measurable import (
        registered_names, transitive_reads,
    )
    needed: set[str] = set()
    for name in registered_names():
        try:
            needed.update(transitive_reads(name))
        except KeyError:
            continue
    # Heuristic: HP-only reads (gamma, n_actions, env_name, …) are
    # in runs.parquet, not traces.parquet. Filter to columns that
    # at least one of: live in `present` already (we know they're
    # trace cols on this corpus), or look like a per-step
    # reduction (`*_per_step`, `*_per_burst`).
    trace_like = {
        c for c in needed
        if c in present
        or '_per_step' in c
        or '_per_burst' in c
        or c in ('mc_return', 'predicted_q_at_start',
                 'episode_length', 'done', 'reward', 'loss',
                 'td_error')
    }
    missing = trace_like - present
    if missing:
        import sys
        sorted_missing = sorted(missing)
        sys.stderr.write(
            f'archive: WARNING — {traces_path} is missing trace '
            f'columns declared by registered measurables: '
            f'{sorted_missing}. Future ingests will produce '
            f'all-NaN for measurables that depend on these. '
            f're-run the sweep with the current substrate if you '
            f'want full schema, or accept partial coverage.\n',
        )


def _sorted_by_relpath(items: Iterable[RemoteFile]) -> list[RemoteFile]:
    return sorted(items, key=lambda f: f.relpath)


# ============ Public API ============

def archive(
    sweep_dir: Path,
    remote_root: str,
    *,
    files: Sequence[str] | None = None,
    force: bool = False,
    purge_local: bool = False,
    validate: bool = True,
) -> RemoteManifest:
    """Upload `files` from `sweep_dir` to `remote_root/<relpath>`,
    update the per-sweep manifest, optionally delete local
    copies.

    `files`: relpaths within `sweep_dir`. Default: top-level
    `*.parquet` files (non-recursive, excludes `tmp/`).

    `force`: re-upload even if the file is already in the
    manifest with a matching sha256. Default false.

    `purge_local`: delete each successfully-archived local file
    after size verification. Default false (safer two-step
    lifecycle: archive → verify → purge).

    `validate`: run CORPUS_INTEGRITY.md CI5 archive-eligibility
    check on each local file (min size + PAR1 footer for
    parquets) before uploading. Default true. Pass `False` only
    when the local files are deliberately small / non-standard
    (test fixtures using opaque byte payloads, etc.).

    Manifest is saved atomically after EACH file. Returns the
    final updated manifest."""
    if not sweep_dir.is_dir():
        raise NotADirectoryError(f'{sweep_dir}: not a directory')

    existing = _load_manifest(sweep_dir)
    if existing is not None and existing.remote_root != remote_root:
        raise ValueError(
            f'manifest at {sweep_dir} already pinned to '
            f'{existing.remote_root!r}; refusing to retarget to '
            f'{remote_root!r}. Restore + re-archive if intentional.',
        )

    # CORPUS_INTEGRITY.md CI3: refuse if a sibling corpus already
    # claims this `remote_root`. Two local corpora pushing to the
    # same s3 prefix silently overwrite each other on every
    # archive call. Check only fires when this corpus is NEW (no
    # existing manifest) — re-archiving the same sweep_dir to its
    # own remote_root is always fine.
    if existing is None:
        from corroborate.corpus.integrity import assert_unique_remote_root
        assert_unique_remote_root(sweep_dir, remote_root)

    selected = list(files) if files is not None else _default_files(sweep_dir)
    if not selected:
        raise ValueError(
            f'no files to archive in {sweep_dir} (pass files= or '
            f'add *.parquet at the top level)',
        )

    by_relpath: dict[str, RemoteFile] = (
        {f.relpath: f for f in existing.files} if existing else {}
    )
    purge_targets: list[Path] = []

    for relpath in selected:
        local = sweep_dir / relpath
        if not local.is_file():
            raise FileNotFoundError(f'{local}: not a file')

        # CORPUS_INTEGRITY.md CI5: refuse to push a file that
        # fails the archive precondition (too small, missing
        # PAR1 footer, etc.). The check fires BEFORE we hash or
        # touch the cloud — a 0-byte placeholder from an
        # interrupted sweep merge gets caught here rather than
        # silently overwriting the cloud's authoritative copy.
        if validate:
            from corroborate.corpus.integrity import (
                assert_archive_eligible,
            )
            assert_archive_eligible(local)
            if relpath == 'traces.parquet' or relpath.endswith(
                '/traces.parquet',
            ):
                _warn_if_trace_schema_incomplete(local)

        sha256 = _sha256_file(local)
        prior = by_relpath.get(relpath)
        remote_uri = _join_remote(remote_root, relpath)
        if prior is not None and prior.sha256 == sha256 and not force:
            # Already archived, content matches; idempotent skip.
            # Local file is still eligible for purge — the remote
            # is verified by sha256 equality with the manifest.
            purge_targets.append(local)
            continue
        if prior is not None and prior.sha256 != sha256 and not force:
            # Invariant I2 (SWEEP_PERSISTENCY.md): the manifest
            # records a prior archive of this relpath with
            # DIFFERENT content. Silently overwriting would
            # last-writer-wins, the very pattern that lost data
            # in the minatar_sync_curve_resume incident. Raise
            # loudly so the user investigates — pass force=True
            # to opt into overwrite.
            raise ConflictingArchive(
                relpath,
                prior_sha256=prior.sha256,
                local_sha256=sha256,
                remote_uri=remote_uri,
            )
        _fs.put_file(local, remote_uri)
        local_size = local.stat().st_size
        try:
            r_size = _fs.remote_size(remote_uri)
        except Exception as e:
            raise RuntimeError(
                f'{remote_uri}: post-upload stat failed: {e}',
            ) from e
        if r_size != local_size:
            raise RuntimeError(
                f'{remote_uri}: size mismatch after upload '
                f'(local {local_size} vs remote {r_size})',
            )

        entry = RemoteFile(
            relpath=relpath,
            size_bytes=local_size,
            sha256=sha256,
            pushed_at=datetime.now(UTC).isoformat(timespec='seconds'),
            row_ids=_sniff_row_ids(local),
        )
        by_relpath[relpath] = entry
        # Save after every successful file — partial archives
        # remain consistent.
        _save_manifest(
            sweep_dir,
            RemoteManifest(
                remote_root=remote_root,
                files=tuple(_sorted_by_relpath(by_relpath.values())),
            ),
        )
        purge_targets.append(local)

    final = RemoteManifest(
        remote_root=remote_root,
        files=tuple(_sorted_by_relpath(by_relpath.values())),
    )

    if purge_local:
        for p in purge_targets:
            p.unlink(missing_ok=True)

    return final


def restore(
    sweep_dir: Path,
    *,
    files: Sequence[str] | None = None,
    overwrite: bool = False,
) -> list[str]:
    """Download archived files from the manifest back into
    `sweep_dir`. Recomputes sha256 of each restored file against
    the manifest entry; mismatch raises and removes the partial
    download.

    `files`: relpaths to restore (must be subset of the
    manifest). Default = all files in the manifest.

    `overwrite`: if a local file already exists with a
    mismatched sha256, replace it. Default false (raises
    instead, to surface drift before clobbering).

    Returns the list of relpaths actually restored (excludes
    files that were already present with matching sha256)."""
    manifest = _load_manifest(sweep_dir)
    if manifest is None:
        raise FileNotFoundError(
            f'{sweep_dir}: no manifest at {MANIFEST_NAME}',
        )
    by_relpath = {f.relpath: f for f in manifest.files}

    if files is not None:
        missing = [r for r in files if r not in by_relpath]
        if missing:
            raise KeyError(
                f'manifest does not contain: {sorted(missing)}',
            )
        targets = [by_relpath[r] for r in files]
    else:
        targets = list(manifest.files)

    restored: list[str] = []
    for entry in targets:
        local = sweep_dir / entry.relpath
        if local.exists():
            local_sha = _sha256_file(local)
            if local_sha == entry.sha256:
                continue  # already present, skip
            if not overwrite:
                raise RuntimeError(
                    f'{local}: exists with sha256 {local_sha} ≠ '
                    f'manifest {entry.sha256}; pass overwrite=True '
                    f'to replace',
                )

        remote_uri = _join_remote(manifest.remote_root, entry.relpath)
        _fs.get_file(remote_uri, local)
        got_sha = _sha256_file(local)
        if got_sha != entry.sha256:
            local.unlink(missing_ok=True)
            raise RuntimeError(
                f'{remote_uri}: sha256 mismatch after restore '
                f'(got {got_sha}, expected {entry.sha256})',
            )
        restored.append(entry.relpath)
    return restored


def is_archived(sweep_dir: Path, relpath: str) -> bool:
    """True iff `relpath` (relative to `sweep_dir`) is recorded in
    the per-sweep manifest. Used by sweep-loop resume: skip arms
    whose tmp parquets are already on the remote so a relaunch
    after a partial crash doesn't redo finished work."""
    manifest = _load_manifest(sweep_dir)
    if manifest is None:
        return False
    return any(f.relpath == relpath for f in manifest.files)


def archived_uri(remote_root: str, relpath: str) -> str:
    """Public wrapper around the URI-join helper. Sweep-loop
    resume uses this to record archived URIs for the merge step
    when skipping a run-the-arm step."""
    return _join_remote(remote_root, relpath)


def backfill_row_ids(sweep_dir: Path) -> int:
    """One-shot upgrade for manifests that predate I5: populate
    `row_ids` on every entry whose field is empty by reading the
    `id` column from the remote parquet.

    Provenance breadcrumb backfill (`SWEEP_PERSISTENCY.md` I5).
    Manifests written before the I5 schema bump have empty
    `row_ids` tuples; this function fetches just the `id` column
    from each remote shard and rewrites the manifest in place.

    Polars' parquet reader uses column-projection pushdown over
    fsspec, so only the `id` column's pages are fetched per
    shard — cheap even on S3.

    Skips entries whose:
      - `row_ids` is already populated (idempotent — running
        twice is a no-op);
      - relpath ends in something other than `.parquet`
        (no `id` column to read);
      - read fails (object absent, network error, schema mismatch)
        — those entries stay empty and can be retried later.

    Returns the count of entries successfully updated. Use as:

        n = backfill_row_ids(Path('experiments/data/my_sweep'))
        print(f'updated {n} manifest entries')
    """
    manifest = _load_manifest(sweep_dir)
    if manifest is None:
        return 0
    updated_count = 0
    new_files: list[RemoteFile] = []
    for entry in manifest.files:
        if entry.row_ids or not entry.relpath.endswith('.parquet'):
            new_files.append(entry)
            continue
        remote_uri = _join_remote(manifest.remote_root, entry.relpath)
        ids = _sniff_row_ids(remote_uri)
        if not ids:
            new_files.append(entry)
            continue
        new_files.append(RemoteFile(
            relpath=entry.relpath,
            size_bytes=entry.size_bytes,
            sha256=entry.sha256,
            pushed_at=entry.pushed_at,
            row_ids=ids,
        ))
        updated_count += 1
    if updated_count > 0:
        _save_manifest(
            sweep_dir,
            RemoteManifest(
                remote_root=manifest.remote_root,
                files=tuple(_sorted_by_relpath(new_files)),
            ),
        )
    return updated_count


def archived_shard_uris(
    sweep_dir: Path,
    *,
    prefix: str,
    suffix: str,
) -> list[str]:
    """Return sorted remote URIs for every manifest entry whose
    `relpath` starts with `prefix` and ends with `suffix`.

    The I3 primitive (per `SWEEP_PERSISTENCY.md`): the merge step
    in `run_intervention` MUST be a pure function of the manifest,
    not call-local state. Without this, paired-sweep dispatches
    that target the same `out_dir` across multiple `run_intervention`
    calls have each call's merge clobber the prior one's
    `<out_dir>/runs.parquet` with a SUBSET of the corpus's cells.
    Reading from the manifest accumulates cells across all calls.

    Returns an empty list when no manifest exists for `sweep_dir`
    (the all-local-no-cloud path; caller falls back to
    `runs_paths` / `traces_paths` collected during iteration).
    Sorted by relpath for deterministic concat order — important
    so re-merging from the same manifest produces byte-identical
    output."""
    manifest = _load_manifest(sweep_dir)
    if manifest is None:
        return []
    matches = [
        f for f in manifest.files
        if f.relpath.startswith(prefix) and f.relpath.endswith(suffix)
    ]
    matches.sort(key=lambda f: f.relpath)
    return [_join_remote(manifest.remote_root, f.relpath) for f in matches]


def ls(sweep_dir: Path) -> RemoteManifest:
    """Read and return the per-sweep manifest. Raises
    `FileNotFoundError` if no archive yet exists for
    `sweep_dir`."""
    m = _load_manifest(sweep_dir)
    if m is None:
        raise FileNotFoundError(
            f'{sweep_dir}: no manifest at {MANIFEST_NAME}',
        )
    return m


def purge(
    sweep_dir: Path,
    *,
    files: Sequence[str] | None = None,
) -> list[str]:
    """Delete LOCAL copies of files the manifest says are
    archived. The manifest itself is preserved so `restore`
    stays available.

    `files`: relpaths to purge (must each be in the manifest).
    Default = all manifest files.

    Returns the list of relpaths actually deleted (excludes
    files that were already absent locally)."""
    manifest = _load_manifest(sweep_dir)
    if manifest is None:
        raise FileNotFoundError(
            f'{sweep_dir}: no manifest at {MANIFEST_NAME}',
        )
    by_relpath = {f.relpath: f for f in manifest.files}

    if files is not None:
        missing = [r for r in files if r not in by_relpath]
        if missing:
            raise KeyError(
                f'manifest does not contain: {sorted(missing)}',
            )
        targets = list(files)
    else:
        targets = [f.relpath for f in manifest.files]

    deleted: list[str] = []
    for relpath in targets:
        local = sweep_dir / relpath
        if local.exists():
            local.unlink()
            deleted.append(relpath)
    return deleted
