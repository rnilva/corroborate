"""Path-safe reads for externally-produced run directories.

Private support module for `corroborate.data.loader`: JSON /
JSONL decoding plus escape-proof path resolution. A run
directory is plain files — evidence is a live, growing record,
not a frozen artifact, so there is no seal here. Corroborate
neither infers provenance or chronology from those files nor
attests their integrity; snapshot and version management remain
external. Verdicts recompute from whichever rows the caller
supplies whenever the record grows.
"""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath


def safe_run_path(root: Path, relative: str) -> Path:
    """Resolve a root-relative path, rejecting absolute / escaping
    paths — a hostile run record must not read outside its
    directory."""
    posix = PurePosixPath(relative)
    if posix.is_absolute() or '..' in posix.parts or not posix.parts:
        raise ValueError(f'unsafe run-relative path: {relative!r}')
    root_resolved = root.resolve()
    candidate = root_resolved.joinpath(*posix.parts).resolve()
    if not candidate.is_relative_to(root_resolved):
        raise ValueError(f'path escapes the run directory: {relative!r}')
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
