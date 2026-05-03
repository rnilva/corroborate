"""Archive unarchived in-cache sweep directories to
`s3://corroborate-archive/<name>/`, purge local top-level files +
the redundant `tmp/` shard subdirectory, and report disk delta.

Each sweep dir typically holds:

- `runs.parquet`, `traces.parquet`, `runs_with_bridge_cache.parquet`
  (top-level merged results — the source of truth post-merge).
- `tmp/<arm>__<env>__<arm-spec>__{runs,traces}.parquet` (per-arm
  shards — redundant with top-level once merged).

This script archives top-level `*.parquet` (cloud.archive
default), then `rm -rf tmp/`. The shards are not pushed; the
merged form on s3 is the canonical backup.

Idempotent: dirs already carrying `_remote.json` are skipped.

Requires `.env` with `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
`AWS_ENDPOINT_URL` to be loaded into the environment beforehand.

Usage (one or more corpus names):
    set -a; source .env; set +a
    PYTHONPATH=. uv run python scripts/archive_unarchived.py \\
        adaptive_dqn_acrobot cartpole_sync_1k cartpole_sync_10k
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

from corroborate.cloud import archive  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / 'experiments' / 'data'
REMOTE_BASE = 's3://corroborate-archive'


def _disk_free_mib() -> int:
    usage = shutil.disk_usage(REPO_ROOT)
    return usage.free // (1024 * 1024)


def archive_one(corpus: str) -> None:
    sweep_dir = DATA_ROOT / corpus
    if not sweep_dir.is_dir():
        print(f'  {corpus}: skip (no such directory)')
        return
    if (sweep_dir / '_remote.json').exists():
        print(f'  {corpus}: skip (already has manifest)')
        return

    free_before = _disk_free_mib()
    remote = f'{REMOTE_BASE}/{corpus}'
    print(f'  {corpus}: archiving -> {remote}')
    manifest = archive(sweep_dir, remote, purge_local=True)
    print(
        f'  {corpus}: pushed {len(manifest.files)} files, '
        f'sum={sum(f.size_bytes for f in manifest.files):,} bytes',
    )
    tmp = sweep_dir / 'tmp'
    if tmp.is_dir():
        shutil.rmtree(tmp)
        print(f'  {corpus}: removed redundant tmp/ shards')
    free_after = _disk_free_mib()
    print(f'  {corpus}: disk freed {free_after - free_before} MiB')


def main(argv: list[str]) -> None:
    names = argv[1:]
    if not names:
        raise SystemExit(
            'usage: archive_unarchived.py <corpus> [<corpus> ...]',
        )
    for corpus in names:
        archive_one(corpus)


if __name__ == '__main__':
    main(sys.argv)
