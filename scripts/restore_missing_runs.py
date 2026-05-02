"""Restore missing `runs.parquet` files from corpora archived
to remote storage but absent locally.

Walks `experiments/data/<corpus>/_remote.json` manifests; for
each corpus where `runs.parquet` is in the manifest but not on
disk, calls `corroborate.cloud.restore` to fetch ONLY that file
(skipping the big `traces.parquet` siblings — those are
typically multi-GB and not needed for bridge evaluation when
trace-derived measurables are already cached or computed
on-demand).

Usage:
    set -a; . .env; set +a   # AWS_* env from .env
    uv run python scripts/restore_missing_runs.py
"""
from __future__ import annotations

import os

# Pure boto/fsspec; force CPU before any JAX-touching imports wander in.
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import json
from pathlib import Path

from corroborate.cloud import restore


def main() -> None:
    data_root = Path('experiments/data')
    if not data_root.is_dir():
        raise SystemExit(f'no experiments/data at {data_root.resolve()}')

    missing: list[Path] = []
    for d in sorted(data_root.iterdir()):
        if not d.is_dir():
            continue
        runs = d / 'runs.parquet'
        remote = d / '_remote.json'
        if runs.exists() or not remote.exists():
            continue
        try:
            manifest = json.loads(remote.read_text())
        except json.JSONDecodeError:
            print(f'[{d.name}] manifest unreadable; skip')
            continue
        relpaths = {f['relpath'] for f in manifest.get('files', [])}
        if 'runs.parquet' not in relpaths:
            print(f'[{d.name}] no runs.parquet in manifest; skip')
            continue
        missing.append(d)

    if not missing:
        print('no missing runs.parquet files; nothing to restore')
        return

    print(f'restoring runs.parquet for {len(missing)} corpora:')
    for d in missing:
        print(f'  {d.name}')
    print()

    restored: list[str] = []
    failed: list[tuple[str, str]] = []
    for d in missing:
        try:
            files = restore(d, files=['runs.parquet'])
            restored.extend(f'{d.name}/{r}' for r in files)
            print(f'[{d.name}] OK ({len(files)} files)')
        except Exception as exc:  # noqa: BLE001 — log + skip per corpus
            failed.append((d.name, repr(exc)))
            print(f'[{d.name}] FAILED: {exc!r}')

    print()
    print(f'restored: {len(restored)} files')
    if failed:
        print(f'failed: {len(failed)} corpora')
        for name, err in failed:
            print(f'  {name}: {err}')


if __name__ == '__main__':
    main()
