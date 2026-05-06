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


@dataclass(frozen=True, slots=True)
class RemoteFile:
    """One archived file's manifest entry. `relpath` is relative
    to the sweep directory; `size_bytes` and `sha256` are
    computed pre-upload from the local file; `pushed_at` is
    ISO-8601 UTC."""

    relpath: str
    size_bytes: int
    sha256: str
    pushed_at: str

    def as_dict(self) -> Mapping[str, object]:
        return {
            'relpath': self.relpath,
            'size_bytes': self.size_bytes,
            'sha256': self.sha256,
            'pushed_at': self.pushed_at,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        return cls(
            relpath=require_str(d, 'relpath'),
            size_bytes=require_int(d, 'size_bytes'),
            sha256=require_str(d, 'sha256'),
            pushed_at=require_str(d, 'pushed_at'),
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

        sha256 = _sha256_file(local)
        prior = by_relpath.get(relpath)
        if prior is not None and prior.sha256 == sha256 and not force:
            # Already archived, content matches; idempotent skip.
            # Local file is still eligible for purge — the remote
            # is verified by sha256 equality with the manifest.
            purge_targets.append(local)
            continue

        remote_uri = _join_remote(remote_root, relpath)
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
