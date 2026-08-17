"""Path-safe reads for external study bundles.

Private support module for `corroborate.data.adapter`: JSON /
JSONL decoding plus escape-proof path resolution. A bundle is a
plain directory of files — evidence is a live, growing record,
not a frozen artifact, so there is no seal here: integrity over
time is the producer's version control's job, and the framework's
verdicts recompute whenever the record grows (that is the point
of the system, not a hazard to guard against).
"""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath


def safe_bundle_path(root: Path, relative: str) -> Path:
    """Resolve a bundle-relative path, rejecting absolute /
    escaping paths — a hostile contract must not read outside the
    bundle."""
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
