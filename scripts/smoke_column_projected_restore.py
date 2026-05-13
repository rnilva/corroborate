"""Smoke test: does polars+fsspec column-projection actually fetch
only the projected columns from S3-hosted parquet, or download the
whole file?

If the wire-traffic timing scales with column count (not file size),
column-projected restore is viable as a Layer-2 speedup for the
trace-backfill path. If timing is constant in column count, polars
or fsspec is downloading the whole file regardless and the
"projection" is local-only.

Reads a manifest, picks `traces.parquet`, times two reads:
  (1) full schema (all columns)
  (2) projection to ['id'] + the columns `q_per_burst` reads

Reports wall time and bytes-loaded ratio. Run from a shell with .env
sourced (AWS creds)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import polars as pl


# Pick a corpus with a beefy traces.parquet on S3
CORPUS_DIR = Path('experiments/data/survive_sync_freeway_si_chunk10')

# What `q_per_burst` actually reads
NEEDED_COLS = ['id', 'online_max_q_per_step', 'eval_step_index']


def _s3_uri(corpus_dir: Path, relpath: str) -> str:
    manifest_path = corpus_dir / '_remote.json'
    manifest = json.loads(manifest_path.read_text())
    return f"{manifest['remote_root'].rstrip('/')}/{relpath}"


def _size_estimate(df: pl.DataFrame) -> int:
    return df.estimated_size('b')


def main() -> None:
    if not (CORPUS_DIR / '_remote.json').exists():
        print(f'no manifest at {CORPUS_DIR}/_remote.json; aborting', file=sys.stderr)
        sys.exit(1)

    s3_traces = _s3_uri(CORPUS_DIR, 'traces.parquet')
    file_size_mb = json.loads((CORPUS_DIR / '_remote.json').read_text())
    file_size_mb = next(
        f['size_bytes'] / 1024 / 1024
        for f in file_size_mb['files']
        if f['relpath'] == 'traces.parquet'
    )
    print(f'target: {s3_traces}')
    print(f'remote size: {file_size_mb:.1f} MB')
    print()

    # Pass 1: column projection
    print('=== Test 1: column projection (3 columns) ===')
    t0 = time.perf_counter()
    proj_df = pl.scan_parquet(s3_traces).select(NEEDED_COLS).collect()
    t_proj = time.perf_counter() - t0
    proj_mb = _size_estimate(proj_df) / 1024 / 1024
    print(f'  wall: {t_proj:.1f}s, materialized: {proj_mb:.1f} MB, '
          f'rows: {proj_df.height}, cols: {len(proj_df.columns)}')
    print(f'  cols got: {proj_df.columns}')

    # Pass 2: schema-only (cheapest possible call, baseline for S3 metadata RTT)
    print()
    print('=== Test 2: schema-only ===')
    t0 = time.perf_counter()
    schema = pl.scan_parquet(s3_traces).collect_schema()
    t_schema = time.perf_counter() - t0
    print(f'  wall: {t_schema:.2f}s, cols seen: {len(schema)}')

    # Pass 3: full materialize (to verify projection is actually a speedup)
    print()
    print('=== Test 3: full read (all columns) — SLOW ===')
    t0 = time.perf_counter()
    full_df = pl.scan_parquet(s3_traces).collect()
    t_full = time.perf_counter() - t0
    full_mb = _size_estimate(full_df) / 1024 / 1024
    print(f'  wall: {t_full:.1f}s, materialized: {full_mb:.1f} MB, '
          f'rows: {full_df.height}, cols: {len(full_df.columns)}')

    print()
    print('=== Verdict ===')
    print(f'projection / full wall ratio: {t_proj/t_full:.2%}')
    print(f'projection / full bytes ratio: {proj_mb/full_mb:.2%}')
    print(f'(if wall ratio ≪ 100% → polars+fsspec is doing real column pushdown)')


if __name__ == '__main__':
    main()
