"""Migrate `experiments/data/ddqn/traces.parquet` to the new
parquet+zarr split.

Reads the existing 4.9 GB traces.parquet, splits each row's
multi-dim list columns out to `arrays.zarr/{cell_id}/{name}`,
writes a slimmed parquet (scalars + 1-D series only).

Non-destructive: writes to `traces.parquet.new` + `arrays.zarr.new`.
Caller verifies §3 verdicts unchanged via `smoke_ddqn_threeway.py`,
then atomic-renames.

Schema of multi-dim columns to migrate (verified against current
on-disk corpus):
- episode_length:        List(List(Int64))
- mc_return:             List(List(Float64))
- predicted_q_at_start:  List(List(Float64))
- sample_indices:        List(List(Int64))
- pearson_stats:         List(List(Float64))

Run: `uv run python scripts/migrate_traces_to_zarr.py`."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import polars as pl
import zarr  # type: ignore[reportMissingTypeStubs]
from zarr.codecs import BloscCodec  # type: ignore[reportMissingTypeStubs]


MULTI_DIM_COLUMNS: tuple[str, ...] = (
    'episode_length',
    'mc_return',
    'predicted_q_at_start',
    'sample_indices',
    'pearson_stats',
)


def main() -> None:
    data_dir = Path('experiments/data/ddqn')
    src = data_dir / 'traces.parquet'
    dst_parquet = data_dir / 'traces.parquet.new'
    dst_zarr = data_dir / 'arrays.zarr.new'

    if not src.exists():
        raise FileNotFoundError(f'{src} not found')
    if dst_parquet.exists() or dst_zarr.exists():
        raise FileExistsError(
            f'{dst_parquet} or {dst_zarr} already exists; '
            f'remove them before running migration.',
        )

    print(f'Source: {src} ({src.stat().st_size / 1e9:.2f} GB)')
    print(
        f'Multi-dim columns to migrate: '
        f'{", ".join(MULTI_DIM_COLUMNS)}',
    )

    # === Pass 1: slim parquet via polars lazy + sink_parquet ===
    # Drops multi-dim columns; writes scalars + 1-D series only.
    t0 = time.time()
    print(f'\nPass 1: slimming parquet (drops multi-dim columns)...')
    lf = pl.scan_parquet(src)
    existing_cols = set(lf.collect_schema().names())
    drop_cols = [c for c in MULTI_DIM_COLUMNS if c in existing_cols]
    if not drop_cols:
        print('  no multi-dim columns found; corpus already migrated?')
    else:
        lf = lf.drop(drop_cols)
    lf.sink_parquet(dst_parquet)
    print(
        f'  wrote {dst_parquet.name} '
        f'({dst_parquet.stat().st_size / 1e6:.1f} MB) in '
        f'{time.time() - t0:.1f}s',
    )

    if not drop_cols:
        return

    # === Pass 2: per-row zarr writes ===
    # Read the SOURCE parquet (still has the multi-dim cols) row by
    # row. For each cell, extract the multi-dim values, convert to
    # numpy, write to zarr keyed by cell_id.
    t0 = time.time()
    print(f'\nPass 2: writing arrays to zarr...')
    root = zarr.open_group(  # type: ignore[reportUnknownMemberType]
        str(dst_zarr), mode='a',
    )
    compressor = BloscCodec(  # type: ignore[reportUnknownMemberType]
        cname='zstd', clevel=3, shuffle='shuffle',
    )
    df_iter = pl.read_parquet(
        src, columns=['id', *drop_cols],
    )
    n_cells = df_iter.height
    n_arrays_written = 0
    for i, row in enumerate(df_iter.iter_rows(named=True)):
        cell_id = row['id']
        if not isinstance(cell_id, str):
            continue
        if cell_id in root:
            del root[cell_id]  # type: ignore[reportUnknownMemberType]
        grp = root.create_group(cell_id)  # type: ignore[reportUnknownMemberType]
        for col in drop_cols:
            value = row.get(col)
            if value is None:
                continue
            np_arr = np.asarray(value)
            if np_arr.size == 0:
                continue
            zarr_arr = grp.create_array(  # type: ignore[reportUnknownMemberType]
                name=col, shape=np_arr.shape, dtype=np_arr.dtype,
                compressors=compressor,
            )
            zarr_arr[:] = np_arr  # type: ignore[reportUnknownMemberType]
            n_arrays_written += 1
        if (i + 1) % 100 == 0:
            print(
                f'  {i + 1}/{n_cells} cells migrated '
                f'({time.time() - t0:.1f}s)',
                flush=True,
            )

    elapsed = time.time() - t0
    print(
        f'  wrote {n_arrays_written} arrays across {n_cells} cells '
        f'in {elapsed:.1f}s',
    )

    # === Summary ===
    src_size = src.stat().st_size
    pq_size = dst_parquet.stat().st_size
    zarr_size = sum(
        p.stat().st_size for p in dst_zarr.rglob('*') if p.is_file()
    )
    total_new = pq_size + zarr_size
    print()
    print('=' * 60)
    print('Size comparison:')
    print(f'  Old single parquet:  {src_size / 1e9:5.2f} GB')
    print(f'  New parquet:         {pq_size / 1e9:5.2f} GB')
    print(f'  New zarr:            {zarr_size / 1e9:5.2f} GB')
    print(f'  New total:           {total_new / 1e9:5.2f} GB')
    print(
        f'  Ratio:               '
        f'{total_new / src_size:.2%} of original',
    )
    print()
    print(
        'Verify §3 verdicts unchanged via:\n'
        '  JAX_PLATFORMS=cpu uv run python experiments/'
        'smoke_ddqn_threeway.py\n'
        '\n'
        'If verdicts match, atomic-rename:\n'
        '  mv experiments/data/ddqn/traces.parquet '
        'experiments/data/ddqn/traces.parquet.old\n'
        '  mv experiments/data/ddqn/traces.parquet.new '
        'experiments/data/ddqn/traces.parquet\n'
        '  mv experiments/data/ddqn/arrays.zarr.new '
        'experiments/data/ddqn/arrays.zarr',
    )


if __name__ == '__main__':
    main()
