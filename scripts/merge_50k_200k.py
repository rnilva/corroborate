"""One-shot: merge `{runs,traces}_200k.parquet` with the 50k
canonical `{runs,traces}.parquet` into a single set of canonical
files, distinguished by the `total_steps` measurement column.

Drops `sample_indices` (~12 GB / 56% of the 200k traces) — only
the `replay_uniformity` invariant reads it, and that invariant's
scalar verdict is already in `runs.parquet`. Keeping the 2-D
indices array would only enable re-evaluating replay-uniformity
under different parameters; that's not on the roadmap.

Schema unification: 200k's column-set is a strict superset of
50k's (the §5 measurable extension landed during the 200k sweep).
Target schema = 200k schema minus `sample_indices`. 50k rows
null-pad the 4 columns 50k doesn't carry: `mc_return`,
`pearson_stats`, `predicted_q_at_start`, `episode_length`.

Traces use pyarrow incremental write (per-row-group streaming).
Runs use polars `diagonal_relaxed` concat — both files are <100 KB.

After both writes succeed, the four source files are deleted and
the new files atomic-rename into the canonical paths."""
from __future__ import annotations

import time
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq


_DATA = Path('experiments/data/ddqn')
_DROP_COLS = frozenset({'sample_indices'})


def _cast_to_target_schema(
    tbl: pa.Table, target_schema: pa.Schema,
) -> pa.Table:
    """Project / null-pad a table to match `target_schema`. Existing
    columns cast to the target type; missing columns inserted as
    full-null arrays of the target type."""
    n = len(tbl)
    cols: list[pa.Array | pa.ChunkedArray] = []
    for f in target_schema:
        if f.name in tbl.column_names:
            col = tbl.column(f.name)
            if col.type != f.type:
                col = col.cast(f.type)
            cols.append(col)
        else:
            cols.append(pa.nulls(n, type=f.type))
    return pa.Table.from_arrays(cols, schema=target_schema)


def _merge_traces() -> None:
    src_50k = _DATA / 'traces.parquet'
    src_200k = _DATA / 'traces_200k.parquet'
    if not src_200k.exists():
        raise SystemExit(f'not found: {src_200k}')
    if not src_50k.exists():
        raise SystemExit(f'not found: {src_50k}')

    print(f'merging traces:')
    print(f'  50k:  {src_50k} ({src_50k.stat().st_size / 1e9:.2f} GB)')
    print(f'  200k: {src_200k} ({src_200k.stat().st_size / 1e9:.2f} GB)')

    pf200 = pq.ParquetFile(src_200k)
    full_schema = pf200.schema_arrow
    target_fields = [f for f in full_schema if f.name not in _DROP_COLS]
    target_schema = pa.schema(target_fields, metadata=full_schema.metadata)
    print(f'  target schema: {len(target_schema)} columns '
          f'(dropped: {sorted(_DROP_COLS)})')

    tmp = _DATA / 'traces.parquet.merged.tmp'
    if tmp.exists():
        tmp.unlink()

    t_total = time.time()
    writer = pq.ParquetWriter(
        tmp, target_schema, compression='zstd', compression_level=3,
    )
    try:
        # 200k row groups: drop sample_indices, write.
        for i in range(pf200.num_row_groups):
            t0 = time.time()
            tbl = pf200.read_row_group(i)
            tbl = tbl.drop_columns(list(_DROP_COLS & set(tbl.column_names)))
            writer.write_table(tbl)
            del tbl
            cur_size = tmp.stat().st_size / 1e9
            print(f'  200k group {i+1}/{pf200.num_row_groups}: '
                  f'→ {cur_size:.2f} GB  ({time.time() - t0:.1f}s)')
        del pf200

        # 50k row group(s): cast to target schema (null-pad 4 missing
        # columns), write.
        pf50 = pq.ParquetFile(src_50k)
        for i in range(pf50.num_row_groups):
            t0 = time.time()
            tbl = pf50.read_row_group(i)
            tbl = _cast_to_target_schema(tbl, target_schema)
            writer.write_table(tbl)
            del tbl
            cur_size = tmp.stat().st_size / 1e9
            print(f'  50k group {i+1}/{pf50.num_row_groups}: '
                  f'→ {cur_size:.2f} GB  ({time.time() - t0:.1f}s)')
        del pf50
    finally:
        writer.close()

    out_size = tmp.stat().st_size
    print(f'  merged traces: {out_size / 1e9:.2f} GB '
          f'({time.time() - t_total:.1f}s total)')

    # Atomic-rename to canonical path. Source 50k file gets
    # overwritten; the 200k file is removed afterward.
    src_50k.unlink()
    tmp.replace(src_50k)
    src_200k.unlink()
    print(f'  → {src_50k}')
    print(f'  removed: {src_200k.name}')


def _merge_runs() -> None:
    src_50k = _DATA / 'runs.parquet'
    src_200k = _DATA / 'runs_200k.parquet'
    if not src_200k.exists():
        raise SystemExit(f'not found: {src_200k}')
    if not src_50k.exists():
        raise SystemExit(f'not found: {src_50k}')

    print(f'merging runs:')
    print(f'  50k:  {src_50k.stat().st_size / 1e3:.0f} KB')
    print(f'  200k: {src_200k.stat().st_size / 1e3:.0f} KB')

    tmp = _DATA / 'runs.parquet.merged.tmp'
    pl.concat(
        [pl.scan_parquet(src_50k), pl.scan_parquet(src_200k)],
        how='diagonal_relaxed',
    ).sink_parquet(tmp)

    n_rows = pl.scan_parquet(tmp).select(pl.len()).collect().item()
    print(f'  merged runs: {n_rows} rows, '
          f'{tmp.stat().st_size / 1e3:.0f} KB')

    src_50k.unlink()
    tmp.replace(src_50k)
    src_200k.unlink()
    print(f'  → {src_50k}')
    print(f'  removed: {src_200k.name}')


def main() -> None:
    _merge_runs()
    print()
    _merge_traces()


if __name__ == '__main__':
    main()
