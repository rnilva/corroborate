"""Fsspec boundary — typed wrapper around fsspec's untyped surface.

fsspec is untyped at the public-API level (`url_to_fs` has no
annotations; `AbstractFileSystem` methods leak `Any`). This module
declares a structural Protocol covering the small set of methods
the cloud module needs (upload, download, stat, exists, delete,
mkdir), narrows fsspec's return value to it once via a single
`cast(FsspecFs, ...)` at the boundary, and exposes typed
primitives downstream.

Mirrors `_json_boundary.py` / `_polars_boundary.py`. Same
discipline: confine the laundering to a tiny boundary module
where it can be reviewed at a glance.

The Protocol's runtime invariant: every fsspec backend
(LocalFileSystem, MemoryFileSystem, S3FileSystem, GCSFileSystem,
…) implements the AbstractFileSystem interface; we just can't
type-check that without fsspec stubs.

Module name is underscore-prefixed to signal **internal use
only**. External users should import fsspec directly."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, cast

import fsspec.core


class FsspecFs(Protocol):
    """Structural Protocol over the AbstractFileSystem methods
    used by the cloud module. Method return types are typed
    `object` where fsspec's contract is best-effort (rm/put_file
    return varies by backend); `info` is narrowed to
    `Mapping[str, object]` and dispatched to typed helpers below."""

    def put_file(self, lpath: str, rpath: str) -> object: ...
    def get_file(self, rpath: str, lpath: str) -> object: ...
    def info(self, path: str) -> Mapping[str, object]: ...
    def exists(self, path: str) -> bool: ...
    def rm(self, path: str) -> object: ...
    def makedirs(self, path: str, exist_ok: bool = False) -> object: ...


def _resolve(uri: str) -> tuple[FsspecFs, str]:
    """Return (fs, remote_path) for `uri`. Single `cast` site:
    fsspec's `url_to_fs` lacks annotations, but every dispatched
    backend implements the AbstractFileSystem interface — the
    Protocol captures that runtime invariant."""
    pair = fsspec.core.url_to_fs(uri)
    path = pair[1]
    if not isinstance(path, str):
        raise TypeError(
            f'expected str path from fsspec.url_to_fs, got '
            f'{type(path).__name__}',
        )
    # Runtime invariant: fsspec dispatches to AbstractFileSystem
    # subclasses, all of which implement the FsspecFs surface.
    # Single cast site for the boundary; downstream is fully typed.
    fs = cast(FsspecFs, pair[0])
    return fs, path


def put_file(local_path: Path, remote_uri: str) -> None:
    """Upload `local_path` to `remote_uri`. Best-effort `makedirs`
    on the parent (no-op for object stores; mkdir for posix-like).

    `OSError` is caught alongside the structural exceptions because
    Cloudflare R2's S3 endpoint translates `makedirs(bucket/prefix,
    exist_ok=True)` into a `create_bucket` request that R2 rejects
    with `Errno 22 Credential access key has length 33, should be
    32`. The bucket already exists on first archive; we don't need
    s3fs to ensure it. Falling through to `put_file` performs the
    actual upload."""
    fs, path = _resolve(remote_uri)
    parent = path.rsplit('/', 1)[0] if '/' in path else ''
    if parent:
        try:
            fs.makedirs(parent, exist_ok=True)
        except (NotImplementedError, AttributeError, OSError):
            pass
    fs.put_file(str(local_path), path)


def get_file(remote_uri: str, local_path: Path) -> None:
    """Download `remote_uri` to `local_path`. Creates the parent
    local directory if missing."""
    fs, path = _resolve(remote_uri)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    fs.get_file(path, str(local_path))


def remote_size(remote_uri: str) -> int:
    """Return the size in bytes of the remote object. Raises
    `TypeError` if the backend's `info` payload is malformed,
    `FileNotFoundError` if the object is missing."""
    fs, path = _resolve(remote_uri)
    info = fs.info(path)
    size = info.get('size')
    if isinstance(size, bool) or not isinstance(size, int):
        raise TypeError(
            f"fsspec.info()['size'] not int: "
            f"{type(size).__name__}",
        )
    return size


def remote_exists(remote_uri: str) -> bool:
    fs, path = _resolve(remote_uri)
    return fs.exists(path)


def remote_list_dir(remote_uri: str) -> list[str]:
    """List immediate children of a remote directory URI.

    Returns full URIs (scheme + path) for each child, or an empty
    list when the directory doesn't exist. Used by `cloud.list_archives`
    for cloud-side discovery.

    fsspec's `ls()` returns paths without the scheme; we re-attach
    the scheme so callers can pass the result back to other
    `_fs.*` functions."""
    fs, path = _resolve(remote_uri)
    try:
        children = fs.ls(path, detail=False)  # pyright: ignore[reportAttributeAccessIssue]
    except FileNotFoundError:
        return []
    scheme = remote_uri.split('://', 1)[0] if '://' in remote_uri else ''
    out: list[str] = []
    for c in children:
        if not isinstance(c, str):
            continue
        out.append(f'{scheme}://{c}' if scheme else c)
    return out


def remote_delete(remote_uri: str) -> None:
    """Delete a single remote object."""
    fs, path = _resolve(remote_uri)
    fs.rm(path)
