"""Post-merge pass: tighten dtypes on a traces parquet.

Reads the parquet at the given path, recursively narrows
`list<double>` → `list<float>` and `list<int64>` → `list<int32>`
in the schema, casts each row group to the new schema, and writes
back via atomic-rename (.tmp + replace).

Why pyarrow row-group streaming (not polars sink_parquet): polars
streams the whole cast as one plan and dies silently on multi-tens-
of-GB casts (observed on 27 GB traces_200k). Per-row-group iteration
bounds peak memory at ~one row group's uncompressed size + the
ParquetWriter's pending buffer, which scales cleanly to corpus size.

Why ZSTD on the destination: snappy on the source is fast but
under-compresses smooth float trajectories. ZSTD@3 is only modestly
slower at write time and consistently 30-50% smaller for the same
data — combined with the Float64→Float32 narrowing it gives
substantial headroom on the destination volume.

The cell_runner round-trip from JAX is now narrow at source (numpy
arrays through to polars, no `arr.tolist()` upcast); this script
exists to backfill corpora that were written before that fix.
New sweeps don't need it.

Run: `uv run python scripts/tighten_traces.py path/to/traces.parquet
              [--out /alt/path]`

`--out` redirects the tightened file's WRITE-PHASE location, not its
final resting place. Useful when the source's volume can't fit the
source + tightened-tmp simultaneously (peak disk = source + ~tmp);
point `--out` at a roomier volume (e.g. `/dev/shm` for tmpfs / RAM-
backed) and the script (a) writes there, (b) deletes the source on
the source volume, (c) moves the tightened file back to the source
path. After completion the corpus lives at the original path; the
alt-volume buffer is reclaimed.
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def _narrow_type(t: pa.DataType) -> pa.DataType:
    """Recursively narrow list-element dtypes. Idempotent — already-
    narrow leaves pass through unchanged."""
    if pa.types.is_large_list(t) or pa.types.is_list(t):
        list_t = t  # `large_list<elem>` or `list<elem>`
        narrowed_elem = _narrow_type(list_t.value_type)
        if pa.types.is_large_list(t):
            return pa.large_list(narrowed_elem)
        return pa.list_(narrowed_elem)
    if pa.types.is_float64(t):
        return pa.float32()
    if pa.types.is_int64(t):
        return pa.int32()
    return t


def _narrow_schema(schema: pa.Schema) -> pa.Schema:
    fields = [
        pa.field(f.name, _narrow_type(f.type), nullable=f.nullable)
        for f in schema
    ]
    return pa.schema(fields, metadata=schema.metadata)


def main(path: Path, out_dir: Path | None = None) -> None:
    if not path.exists():
        raise SystemExit(f'not found: {path}')
    src_size = path.stat().st_size
    print(f'tightening: {path}  ({src_size / 1e9:.2f} GB)')

    pf = pq.ParquetFile(path)
    src_schema = pf.schema_arrow
    tgt_schema = _narrow_schema(src_schema)

    n_groups = pf.num_row_groups
    print(f'  {n_groups} row groups, {pf.metadata.num_rows} rows total')

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp = out_dir / (path.name + '.tightened.tmp')
        print(f'  write phase → {tmp} (alt-volume staging)')
    else:
        tmp = path.with_suffix(path.suffix + '.tightened.tmp')
    if tmp.exists():
        tmp.unlink()

    t_total = time.time()
    # ZSTD@3 — float-trajectory-friendly, modest write-time cost,
    # 30-50% smaller than snappy. Higher levels (tested @9) gave
    # bit-identical per-group sizes on the §3 corpus — smooth
    # floats already saturate ZSTD's dictionary, longer-window
    # search doesn't find more redundancy. Stay at @3 to avoid
    # the CPU tax. Per-row-group streaming bounds memory at one
    # uncompressed group (~1 GB on traces_200k).
    writer = pq.ParquetWriter(tmp, tgt_schema, compression='zstd',
                              compression_level=3)
    try:
        for i in range(n_groups):
            t0 = time.time()
            tbl = pf.read_row_group(i)
            tbl = tbl.cast(tgt_schema)
            writer.write_table(tbl)
            del tbl
            cur_size = tmp.stat().st_size / 1e9
            print(f'  group {i+1}/{n_groups}: '
                  f'→ {cur_size:.2f} GB '
                  f'({time.time() - t0:.1f}s)')
    finally:
        writer.close()

    tightened_size = tmp.stat().st_size
    ratio = tightened_size / src_size if src_size > 0 else 1.0
    print(f'  before: {src_size / 1e9:.2f} GB')
    print(f'  after:  {tightened_size / 1e9:.2f} GB  ({ratio:.0%})')
    print(f'  total:  {time.time() - t_total:.1f}s')

    # When `tmp` lives on a different volume than `path`, `Path.replace`
    # would fail with EXDEV (cross-device). Free the source first
    # (it's still intact at this point), then `shutil.move` does the
    # cross-volume copy. When same-volume, this is just a rename.
    #
    # Critical: drop the pyarrow handle BEFORE the unlink. On Linux
    # the OS only reclaims the file's blocks once every fd to the
    # inode is closed; the pyarrow `ParquetFile` here still holds
    # one. Without `del pf`, `path.unlink()` succeeds but the disk
    # space stays held, and the cross-volume copy then ENOSPCs into
    # the partially-reclaimed volume. (Hit on the §3 200k corpus.)
    if out_dir is not None:
        del pf
        path.unlink()
        shutil.move(str(tmp), str(path))
    else:
        tmp.replace(path)
    print(f'  swapped → {path}')


if __name__ == '__main__':
    args = sys.argv[1:]
    out_dir: Path | None = None
    if '--out' in args:
        i = args.index('--out')
        out_dir = Path(args[i + 1])
        args = args[:i] + args[i + 2:]
    if len(args) != 1:
        raise SystemExit(
            'usage: uv run python scripts/tighten_traces.py '
            '<traces.parquet> [--out /alt/volume]'
        )
    main(Path(args[0]), out_dir)
