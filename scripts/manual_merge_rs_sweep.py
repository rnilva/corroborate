"""Manual merge of per-cell shards in
`experiments/data/rs_sweep_with_traces/{vanilla_dqn,ddqn}/tmp/`
into top-level runs.parquet + traces.parquet.

The auto-merge inside `run_intervention` died with OOM during
the recursive concat phase (traces are ~360MB compressed per
cell × `chunk_size=4` = ~6GB compressed in-flight, decompressed
20-30GB → exceeds available RAM after polars overhead).

This script does the merge with `chunk_size=1` (one input loaded
at a time) — peak RAM is one decompressed trace, ~5-10GB. Slow
but stable.

Usage:
    PYTHONPATH=. uv run python scripts/manual_merge_rs_sweep.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, '.')

from corroborate.corpus.persistence import stream_concat_parquets

ROOT = Path('experiments/data/rs_sweep_with_traces')


def merge_arm(arm_dir: Path) -> None:
    """Merge tmp/cell*__runs.parquet → arm_dir/runs.parquet (small,
    no OOM risk) and tmp/cell*__traces.parquet → arm_dir/traces.parquet
    (large, chunk_size=1 to avoid OOM)."""
    if not arm_dir.exists():
        print(f'  {arm_dir}: missing, skipping')
        return
    tmp = arm_dir / 'tmp'
    if not tmp.exists():
        print(f'  {arm_dir}: no tmp/, skipping')
        return

    runs_shards = sorted(tmp.glob('cell*__runs.parquet'))
    traces_shards = sorted(tmp.glob('cell*__traces.parquet'))
    print(f'  {arm_dir.name}: {len(runs_shards)} runs shards, '
          f'{len(traces_shards)} traces shards')

    if runs_shards:
        runs_out = arm_dir / 'runs.parquet'
        print(f'    merging runs → {runs_out}')
        stream_concat_parquets(
            list(runs_shards), runs_out,
            chunk_size=4,  # runs are small, fine at default
        )
        sz = runs_out.stat().st_size / 1024
        print(f'      done: {sz:.0f} KB')

    if traces_shards:
        traces_out = arm_dir / 'traces.parquet'
        print(f'    merging traces → {traces_out} (chunk_size=2)')
        # chunk_size=2: pass-1 produces N/2 chunks of 2 inputs each
        # (loads ~2× decompressed trace = ~10-15GB, acceptable),
        # pass-2 recursion halves count each pass until ≤2.
        # chunk_size=1 is degenerate (no count reduction, hangs).
        stream_concat_parquets(
            list(traces_shards), traces_out,
            chunk_size=2,
        )
        sz = traces_out.stat().st_size / (1024 * 1024)
        print(f'      done: {sz:.0f} MB')


def main() -> None:
    print(f'manual merge from {ROOT}/')
    for arm in ('vanilla_dqn', 'ddqn'):
        arm_dir = ROOT / arm
        merge_arm(arm_dir)
    print('done.')


if __name__ == '__main__':
    main()
