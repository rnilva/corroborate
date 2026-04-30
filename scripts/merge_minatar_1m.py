"""Merge the MinAtar 1M arm parquets into a single 4-env corpus.

Input layout (after R2 restore + local SpaceInvaders sweep):
  experiments/data/minatar_1M/tmp/arm{000..005}__{Asterix,Breakout,Freeway}-MinAtar__{vanilla_dqn,ddqn}__{runs,traces}.parquet
  experiments/data/minatar_1M_spaceinvaders/tmp/arm{000,001}__SpaceInvaders-MinAtar__{vanilla_dqn,ddqn}__{runs,traces}.parquet

  (or, if the SpaceInvaders tmp directory wasn't kept after its
  own merge ran, the already-merged top-level files at
  `experiments/data/minatar_1M_spaceinvaders/{runs,traces}.parquet`.
  This script handles both — it globs both directories and uses
  whichever exists.)

Output:
  experiments/data/minatar_1M/runs.parquet     (8 arms × 30 seeds = 240 cells)
  experiments/data/minatar_1M/traces.parquet   (~5GB, streamed)

Streaming pyarrow ParquetWriter for traces; source files are
deleted after each write to keep peak disk roughly constant.
Runs is small enough for direct concat. Pass --keep-sources to
skip the source-deletion step."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq


_MAIN = Path('experiments/data/minatar_1M')
_SPACEINVADERS = Path('experiments/data/minatar_1M_spaceinvaders')


def _collect_arm_files(suffix: str) -> list[Path]:
    """Find every per-arm `*__<suffix>` parquet across the two
    source directories. `suffix` is `runs.parquet` or
    `traces.parquet`."""
    out: list[Path] = []
    for src in (_MAIN, _SPACEINVADERS):
        tmp = src / 'tmp'
        if tmp.exists():
            out.extend(sorted(tmp.glob(f'arm*__*MinAtar__*__{suffix}')))
    # Fallback: if SpaceInvaders has only the merged top-level
    # files (its own per-arm tmp/ may have been cleaned), include
    # those directly.
    si_top = _SPACEINVADERS / suffix
    if si_top.exists() and not any(
        'SpaceInvaders' in p.name for p in out
    ):
        out.append(si_top)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--keep-sources', action='store_true',
        help='Skip source-file deletion after streaming write.',
    )
    args = parser.parse_args()
    delete_after_write = not args.keep_sources
    runs_files = _collect_arm_files('runs.parquet')
    traces_files = _collect_arm_files('traces.parquet')
    print(f'arm files: {len(runs_files)} runs, '
          f'{len(traces_files)} traces')
    for p in runs_files:
        print(f'  runs: {p}')
    for p in traces_files:
        sz_mb = p.stat().st_size / 1e6
        print(f'  traces: {p}  ({sz_mb:.1f} MB)')

    # Runs concat (small, fast).
    t0 = time.time()
    runs_out = _MAIN / 'runs.parquet'
    pl.concat(
        [pl.scan_parquet(p) for p in runs_files],
        how='diagonal_relaxed',
    ).sink_parquet(runs_out)
    print(f'runs.parquet: '
          f'{runs_out.stat().st_size / 1e6:.1f} MB '
          f'({time.time() - t0:.1f}s)')

    # Traces — streaming pyarrow ParquetWriter. Keep peak disk
    # constant: read each arm's table, write to output, source
    # remains.
    traces_tmp = _MAIN / 'traces.parquet.tmp'
    if traces_tmp.exists():
        traces_tmp.unlink()
    t_total = time.time()
    schema = None
    writer: pq.ParquetWriter | None = None
    try:
        for i, p in enumerate(traces_files):
            t_arm = time.time()
            tbl = pq.read_table(str(p))
            if writer is None:
                schema = tbl.schema
                writer = pq.ParquetWriter(
                    str(traces_tmp), schema, compression='snappy',
                )
            writer.write_table(tbl)
            del tbl
            elapsed = time.time() - t_arm
            print(f'  [{i + 1}/{len(traces_files)}] {p.name} '
                  f'({elapsed:.1f}s)')
            if delete_after_write:
                p.unlink()
    finally:
        if writer is not None:
            writer.close()
    traces_out = _MAIN / 'traces.parquet'
    if traces_out.exists():
        traces_out.unlink()
    traces_tmp.rename(traces_out)
    print(f'traces.parquet: '
          f'{traces_out.stat().st_size / 1e9:.2f} GB '
          f'({time.time() - t_total:.1f}s)')

    # Quick sanity check on the merged runs.
    df = pl.read_parquet(runs_out)
    print()
    print(f'Merged corpus: {len(df)} cells')
    print('per (env, arm):')
    summary = df.group_by('env_name', 'intervention_name').agg(
        pl.len().alias('n'),
        pl.col('outcome.eval_best_burst_mean').mean().alias('mean_best'),
        pl.col('outcome.eval_final_mean').mean().alias('mean_final'),
    ).sort('env_name', 'intervention_name')
    print(summary)


if __name__ == '__main__':
    main()
