"""Content-addressed seal + path-safe reads for external study bundles.

Private support module for `corroborate.data.adapter`. Only
`seal_bundle` is re-exported on the public surface — producers
need exactly one blessed way to generate the manifest the
adapter's digest check re-derives; everything else here is a
canonicalisation detail (stable JSON bytes, streaming SHA-256,
escape-proof path resolution) that would invite drift if callers
composed it themselves.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

MANIFEST_NAME = 'manifest.json'
MANIFEST_VERSION = 1


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    """One sealed file: content digest + size, keyed by its
    bundle-relative POSIX path in the manifest's `files` map."""
    sha256: str
    size: int


def canonical_json_bytes(value: object) -> bytes:
    """Stable byte encoding — sorted keys, no whitespace, no NaN —
    so the same logical document always hashes identically."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=False,
        allow_nan=False,
    ).encode('utf-8')


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def bundle_digest(files: Mapping[str, ManifestEntry]) -> str:
    """Digest the ordered path → (sha256, size) manifest — the
    bundle's identity is its content, never local mtimes."""
    projected = {
        path: {'sha256': entry.sha256, 'size': entry.size}
        for path, entry in sorted(files.items())
    }
    return hashlib.sha256(canonical_json_bytes(projected)).hexdigest()


def safe_bundle_path(root: Path, relative: str) -> Path:
    """Resolve a manifest path, rejecting absolute / escaping
    paths — a hostile manifest must not read outside the bundle."""
    posix = PurePosixPath(relative)
    if posix.is_absolute() or '..' in posix.parts or not posix.parts:
        raise ValueError(f'unsafe bundle path: {relative!r}')
    root_resolved = root.resolve()
    candidate = root_resolved.joinpath(*posix.parts).resolve()
    if not candidate.is_relative_to(root_resolved):
        raise ValueError(f'bundle path escapes root: {relative!r}')
    return candidate


def read_json(path: Path) -> object:
    with path.open('r', encoding='utf-8') as stream:
        value: object = json.load(stream)
    return value


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open('r', encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value: object = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f'{path.name}:{line_number}: expected a JSON object',
                )
            # Runtime invariant: json.loads object keys are always str.
            typed_row: dict[str, object] = {
                str(key): item for key, item in value.items()
            }
            rows.append(typed_row)
    return rows


def seal_bundle(
    root: Path | str,
    relative_paths: Iterable[str] | None = None,
) -> Path:
    """Write the content-addressed `manifest.json` that makes a
    study directory a sealed bundle.

    Producers call this once, after the last file is written; the
    adapter then re-derives every digest from the files it
    actually reads. `relative_paths=None` seals every regular
    file under `root` (the manifest itself excluded) — the
    common case. Returns the manifest path.
    """
    root_path = Path(root)
    if relative_paths is None:
        relative_paths = [
            path.relative_to(root_path).as_posix()
            for path in sorted(root_path.rglob('*'))
            if path.is_file() and path.name != MANIFEST_NAME
        ]
    files: dict[str, ManifestEntry] = {}
    for relative in sorted(set(relative_paths)):
        path = safe_bundle_path(root_path, relative)
        files[relative] = ManifestEntry(
            sha256=sha256_file(path),
            size=path.stat().st_size,
        )
    manifest: dict[str, object] = {
        'manifest_version': MANIFEST_VERSION,
        'files': {
            relative: {'sha256': entry.sha256, 'size': entry.size}
            for relative, entry in files.items()
        },
        'bundle_digest': bundle_digest(files),
    }
    manifest_path = root_path / MANIFEST_NAME
    text = json.dumps(
        manifest,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    manifest_path.write_text(text + '\n', encoding='utf-8')
    return manifest_path
