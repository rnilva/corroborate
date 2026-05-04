"""Shard-merge wrapper isolating pyarrow + polars's untyped surfaces.

Per-arm parquet shards in `tmp/*_traces.parquet` need to be merged
into one canonical `traces.parquet`. Two pressures collide:

1. **Heterogeneous shard schemas.** Sweeps that record per-arm
   wrapper-tagged columns (e.g. `dampened_alpha_envs`) produce
   shards whose schemas differ in column set. Across sweep
   designs that mix bandit and continuous-control envs in one
   corpus (e.g. `gamma_sweep_more`), the SAME column can carry
   incompatible types — `reward` is `bool` for some envs and
   `double` for others. pyarrow's `unify_schemas` rejects this;
   polars's `diagonal_relaxed` concat promotes correctly.

2. **GB-scale shards.** Pixel-env corpora like `minatar_1M`
   archive ~2 GB per shard. polars's eager `concat + sink_parquet`
   materialises the cross-shard panel before writing and OOMs
   on these. pyarrow's row-group `ParquetWriter` keeps RAM
   bounded.

Hybrid: polars derives the unified schema (one cheap `head(1)`
collect on a `diagonal_relaxed` concat across lazy frames),
pyarrow's `ParquetWriter` streams each shard's data into the
canonical file. Per-shard memory peaks at ~one shard. Shards
are `unlink`-ed after their data is committed.

`# pyright: basic` because pyarrow ships without type stubs in
this environment; the surface used here is stable across
versions, runtime invariants are upstream-documented."""
# pyright: basic
from __future__ import annotations

from pathlib import Path

import polars as pl
import pyarrow.parquet as pq


def merge_parquet_shards(shards: list[Path], dest: Path) -> None:
    """Stream-merge `shards` into a single parquet at `dest`.

    Schemas are unified via polars' `diagonal_relaxed` concat
    (null-pads missing columns, promotes incompatible numeric
    types). The unified arrow schema is captured from a 1-row
    materialisation; the body of the merge then per-shard reads
    one DataFrame at a time, casts to the unified polars schema,
    converts to a pyarrow Table, writes via streaming
    `ParquetWriter`. Shard files are removed after their data is
    committed."""
    lazy_concat = pl.concat(
        [pl.scan_parquet(s, glob=False) for s in shards],
        how='diagonal_relaxed',
    )
    unified_schema = lazy_concat.collect_schema()
    target_names = list(unified_schema.names())
    sample_arrow = lazy_concat.head(1).collect().to_arrow()
    writer = pq.ParquetWriter(dest, sample_arrow.schema, compression='zstd')
    try:
        for shard in shards:
            df = pl.read_parquet(shard, glob=False)
            missing = [
                name for name in target_names
                if name not in df.columns
            ]
            if missing:
                df = df.with_columns([
                    pl.lit(None).cast(unified_schema[name]).alias(name)
                    for name in missing
                ])
            df = df.cast(unified_schema).select(target_names)
            writer.write_table(df.to_arrow())
            shard.unlink()
    finally:
        writer.close()


__all__ = ['merge_parquet_shards']
