"""One-shot: merge the 36 200k-step arm parquets in tmp/ into
`{runs,traces}_200k.parquet`. Leaves the existing 50k corpus
files untouched.

Splits the corpus by total_steps so analysis can pick which to
read; avoids the cross-corpus diagonal_relaxed concat that
repeatedly OOM'd / got killed in collect_ddqn_runs.py's merge.

Uses pyarrow's incremental `ParquetWriter` — writes each arm's
table into the output and immediately deletes the source.
Disk usage stays roughly constant rather than peaking at
source-size + destination-size.

Assumes all 200k arms share an identical schema (they do —
same env catalogue, same total_steps, same per-step
reductions). For the cross-(50k, 200k) concat, schemas
differ on `eval_step_index` length and the script uses polars
diagonal_relaxed instead of pyarrow direct."""
from __future__ import annotations

import time
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq


_DATA = Path('experiments/data/ddqn')
_TMP = _DATA / 'tmp'


def main() -> None:
    runs_files = sorted(_TMP.glob('*total_steps=200000*runs.parquet'))
    traces_files = sorted(_TMP.glob('*total_steps=200000*traces.parquet'))
    print(f'200k arm files: {len(runs_files)} runs, '
          f'{len(traces_files)} traces')

    # Runs concat — small, fast. Doesn't need streaming.
    t0 = time.time()
    runs_out = _DATA / 'runs_200k.parquet'
    pl.concat(
        [pl.scan_parquet(p) for p in runs_files],
        how='diagonal_relaxed',
    ).sink_parquet(runs_out)
    print(f'  runs_200k.parquet: '
          f'{runs_out.stat().st_size / 1e6:.1f} MB '
          f'({time.time() - t0:.1f}s)')

    # Traces — streaming write with per-arm source deletion.
    # pyarrow's ParquetWriter takes an explicit schema (from arm 0)
    # and writes each subsequent arm's table directly, no full-
    # corpus materialization. Source deleted after write so disk
    # stays roughly constant at ~one-arm + destination-so-far.
    traces_tmp = _DATA / 'traces_200k.parquet.tmp'
    if traces_tmp.exists():
        traces_tmp.unlink()

    t_total = time.time()
    schema = None
    writer: pq.ParquetWriter | None = None
    bytes_written = 0
    for i, src in enumerate(traces_files):
        t0 = time.time()
        table = pq.read_table(src)
        if schema is None:
            schema = table.schema
            writer = pq.ParquetWriter(traces_tmp, schema)
        # Cast to the unified schema — should be no-op if all arms
        # share the same shape.
        if not table.schema.equals(schema, check_metadata=False):
            table = table.cast(schema)
        assert writer is not None
        writer.write_table(table)
        del table
        # Delete the source to free disk.
        src_size = src.stat().st_size / 1e6
        src.unlink()
        bytes_written += int(traces_tmp.stat().st_size) - bytes_written
        cur_size = traces_tmp.stat().st_size / 1e9
        print(f'  arm {i+1}/{len(traces_files)}: '
              f'+{src_size:.0f} MB → {cur_size:.2f} GB '
              f'({time.time() - t0:.1f}s)')
    if writer is not None:
        writer.close()

    final = _DATA / 'traces_200k.parquet'
    traces_tmp.replace(final)
    print(f'  traces_200k.parquet: '
          f'{final.stat().st_size / 1e9:.2f} GB '
          f'({time.time() - t_total:.1f}s total)')


if __name__ == '__main__':
    main()
