"""One-shot upgrade: populate `row_ids` on every sweep manifest
under `experiments/data/<corpus>/_remote.json` that predates the
I5 schema (SWEEP_PERSISTENCY.md).

Streaming-only: `cloud.backfill_row_ids` reads each remote shard's
`id` column via fsspec column-projection pushdown — no full-file
downloads, no local storage burst. One sweep at a time, sequential,
prints per-corpus progress so a long-running job is observable.

Idempotent: skipping any corpus whose manifest is already
populated. Safe to re-run if interrupted.

Usage:
  set -a; . .env; set +a   # AWS creds for S3 reads
  uv run python scripts/backfill_all_row_ids.py

Logs each corpus + count of entries updated to stderr; emits a
JSON summary at `/tmp/backfill_row_ids_<timestamp>.json` for
post-hoc audit.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

# Ensure JAX doesn't preallocate GPU memory (this script is
# parquet-IO-only, no jax kernels — but importing corroborate
# may pull in jax via the analyses module).
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

from corroborate.corpus import cloud


def _iter_sweep_dirs(data_root: Path) -> list[Path]:
    """Find every directory under `data_root` that holds a
    `_remote.json` manifest (one level deep — sweep dirs are
    `experiments/data/<corpus>/`)."""
    return sorted(
        p.parent for p in data_root.glob(f'*/{cloud.MANIFEST_NAME}')
    )


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    data_root = repo_root / 'experiments' / 'data'
    if not data_root.is_dir():
        sys.stderr.write(f'ERROR: {data_root} not found\n')
        return 1

    sweep_dirs = _iter_sweep_dirs(data_root)
    sys.stderr.write(
        f'backfill_row_ids: {len(sweep_dirs)} corpora to scan\n'
    )

    results: list[dict[str, object]] = []
    t_start = time.monotonic()
    for i, d in enumerate(sweep_dirs):
        t_corpus = time.monotonic()
        sys.stderr.write(
            f'[{i+1}/{len(sweep_dirs)}] {d.name} ... '
        )
        sys.stderr.flush()
        try:
            n_updated = cloud.backfill_row_ids(d)
        except Exception as e:  # noqa: BLE001
            elapsed = time.monotonic() - t_corpus
            sys.stderr.write(
                f'FAILED in {elapsed:.1f}s: {type(e).__name__}: {e}\n'
            )
            results.append({
                'corpus': d.name,
                'status': 'error',
                'error_type': type(e).__name__,
                'error_message': str(e),
                'duration_s': elapsed,
            })
            continue
        elapsed = time.monotonic() - t_corpus
        sys.stderr.write(
            f'{n_updated} entries updated in {elapsed:.1f}s\n'
        )
        results.append({
            'corpus': d.name,
            'status': 'ok',
            'entries_updated': n_updated,
            'duration_s': elapsed,
        })

    total_elapsed = time.monotonic() - t_start
    timestamp = datetime.now(UTC).isoformat(timespec='seconds').replace(
        ':', '-',
    )
    log_path = Path(f'/tmp/backfill_row_ids_{timestamp}.json')
    summary = {
        'timestamp_utc': datetime.now(UTC).isoformat(timespec='seconds'),
        'data_root': str(data_root),
        'n_corpora': len(sweep_dirs),
        'total_duration_s': total_elapsed,
        'total_entries_updated': sum(
            int(r['entries_updated'])
            for r in results
            if r['status'] == 'ok'
        ),
        'n_errors': sum(1 for r in results if r['status'] == 'error'),
        'per_corpus': results,
    }
    log_path.write_text(json.dumps(summary, indent=2))
    sys.stderr.write(
        f'\nfinished in {total_elapsed/60:.1f} min '
        f'({summary["total_entries_updated"]} entries updated, '
        f'{summary["n_errors"]} corpora errored). '
        f'Summary: {log_path}\n',
    )
    return 0 if summary['n_errors'] == 0 else 2


if __name__ == '__main__':
    sys.exit(main())
