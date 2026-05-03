"""One-shot: restore minatar_sync_intervention's per-shard traces
from s3, concat into a single traces.parquet, push the merged
file back to s3 (so the standard restore path picks it up next
rebuild). Cleans up local shards after upload verification.

Why: the corpus was archived as 16 per-arm shards under `tmp/`
rather than a single `traces.parquet`. The runner's restore
hook looks for `traces.parquet` at the top level; the merged
file existed only locally and was lost. The cloud shards (~14
GB total) are intact; this script reconstitutes the merged form
and re-archives it canonically."""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from corroborate.cloud import archive, restore


CORPUS = 'minatar_sync_intervention'
DATA = Path('experiments/data') / CORPUS
REMOTE_ROOT = f's3://corroborate-archive/{CORPUS}'


def _shard_relpaths() -> list[str]:
    manifest = json.loads((DATA / '_remote.json').read_text())
    return sorted(
        f['relpath'] for f in manifest['files']
        if f['relpath'].startswith('tmp/')
        and f['relpath'].endswith('__traces.parquet')
    )


def main() -> None:
    import pyarrow.parquet as pq

    shards = _shard_relpaths()
    out = DATA / 'traces.parquet'
    if out.exists():
        out.unlink()
    print(f'will restore + stream-concat {len(shards)} shards from {REMOTE_ROOT}')

    # Restore one shard at a time, append its row groups to the
    # output writer, drop the shard from disk before next restore.
    # Memory peak: ~one shard's row group (~900 MB max). Safer than
    # `pl.concat([... 16 frames ...]).write_parquet(out)` which
    # materialises all 14 GB in RAM and triggers OOM.
    writer: pq.ParquetWriter | None = None
    try:
        for i, relpath in enumerate(shards):
            print(f'  [{i + 1}/{len(shards)}] {relpath}', flush=True)
            restore(DATA, files=[relpath], overwrite=True)
            local = DATA / relpath
            pf = pq.ParquetFile(local)
            if writer is None:
                writer = pq.ParquetWriter(
                    out, pf.schema_arrow,
                    compression='zstd', compression_level=3,
                )
            for rg_idx in range(pf.num_row_groups):
                writer.write_table(pf.read_row_group(rg_idx))
            del pf
            local.unlink()
    finally:
        if writer is not None:
            writer.close()

    if not out.exists() or out.stat().st_size == 0:
        raise RuntimeError(f'write failed: {out} is empty')
    size_gb = out.stat().st_size / 1e9
    print(f'wrote {out}: {size_gb:.2f} GB')

    # Sanity-check: count rows.
    n = pl.scan_parquet(out).select(pl.len()).collect().item()
    print(f'verified: {n} rows in merged file')

    print(f'archiving traces.parquet to {REMOTE_ROOT}...', flush=True)
    archive(DATA, REMOTE_ROOT, files=['traces.parquet'])
    print('done — traces.parquet now in cloud manifest')

    # Drop the now-redundant per-arm shards locally; cloud shards
    # stay until the user explicitly purges them.
    tmp = DATA / 'tmp'
    if tmp.exists():
        for f in tmp.glob('*.parquet'):
            f.unlink()
        try:
            tmp.rmdir()
        except OSError:
            pass


if __name__ == '__main__':
    main()
