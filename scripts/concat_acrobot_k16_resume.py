"""One-off concat: merge the 10 complete tmp cells from
`k_sweep_acrobot_k16_retry` (seeds 0..24 × 2 arms) with the
2-cell resume corpus `k_sweep_acrobot_k16_resume` (seeds 25..29
× 2 arms) into the original sweep's top-level runs.parquet +
traces.parquet. Then drop the `.in_progress` sentinel.

After this, the original corpus is a complete 12-cell / 60-seed
substrate for the k=16 sweep at canonical 1M.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))

from corroborate.corpus.persistence import stream_concat_parquets

ORIG_DIR = REPO / 'experiments' / 'data' / 'k_sweep_acrobot_k16_retry'
RESUME_DIR = REPO / 'experiments' / 'data' / 'k_sweep_acrobot_k16_resume'

# Cells live one level deep under the hypothesis dir.
HYPO = 'ddqn_vs_vanilla'
TMP = ORIG_DIR / HYPO / 'tmp'


def _collect_tmp(suffix: str) -> list[Path]:
    """All matching files in the tmp dir, sorted by cell index."""
    files = sorted(
        TMP.iterdir(),
        key=lambda p: p.name,
    )
    return [p for p in files if p.name.endswith(suffix)]


def main() -> None:
    tmp_runs = _collect_tmp('__runs.parquet')
    tmp_traces = _collect_tmp('__traces.parquet')
    print(f'tmp/ runs.parquet shards: {len(tmp_runs)}')
    print(f'tmp/ traces.parquet shards: {len(tmp_traces)}')
    assert len(tmp_runs) == 10, f'expected 10 tmp runs, got {len(tmp_runs)}'
    assert len(tmp_traces) == 10, f'expected 10 tmp traces, got {len(tmp_traces)}'

    resume_runs = RESUME_DIR / 'runs.parquet'
    resume_traces = RESUME_DIR / 'traces.parquet'
    assert resume_runs.exists()
    assert resume_traces.exists()

    # Concat order: tmp cells first (seeds 0..24), then resume (seeds 25..29).
    runs_inputs = [*tmp_runs, resume_runs]
    traces_inputs = [*tmp_traces, resume_traces]

    final_runs = ORIG_DIR / 'runs.parquet'
    final_traces = ORIG_DIR / 'traces.parquet'
    print(f'concating {len(runs_inputs)} runs shards → {final_runs}')
    stream_concat_parquets(runs_inputs, final_runs)
    print(f'concating {len(traces_inputs)} traces shards → {final_traces}')
    stream_concat_parquets(traces_inputs, final_traces)

    # Verify cell count.
    import polars as pl
    df = pl.read_parquet(final_runs)
    seeds = sorted(df['seed'].unique().to_list())
    print(f'merged runs: shape={df.shape}, seeds={seeds[:5]}..{seeds[-5:]}')

    # Clean up .in_progress sentinel.
    sentinel = ORIG_DIR / '.in_progress'
    if sentinel.exists():
        sentinel.unlink()
        print(f'removed sentinel: {sentinel}')


if __name__ == '__main__':
    main()
