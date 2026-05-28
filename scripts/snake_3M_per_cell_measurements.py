"""Per-cell measurable computation for snake_g099_canonical_3M_ckpt.

The streaming primitive can't help: the traces.parquet has 1 row
group holding all 60 cells (11.5 GB compressed), so row-group
iteration loads everything. Use polars lazy scan + per-id predicate
pushdown so we materialise one cell at a time.

Output: `experiments/data/snake_g099_canonical_3M_ckpt/measurements.parquet`
with id + the per-burst mediator columns 05b needs. A subsequent
`corroborate hypothesis hasselt_clean --ingest snake_g099_canonical_3M_ckpt
 --no-restore` will pick this up and skip the OOM-prone full-trace path.
"""
from __future__ import annotations
import gc
import time
from pathlib import Path

import polars as pl

from corroborate.measurables.measurable import (
    compute_missing_columns, transitive_reads,
)

# import to register substrate measurables
import experiments.findings.hasselt_clean  # noqa: F401

CORPUS = Path('experiments/data/snake_g099_canonical_3M_ckpt')
RUNS = CORPUS / 'runs.parquet'
TRACES = CORPUS / 'traces.parquet'
OUT = CORPUS / 'measurements.parquet'

TARGETS = (
    'bootstrap_gap_magnitude_per_burst',
    'bootstrap_disagree_rate_per_burst',
    'bootstrap_disagree_gap_conditional_per_burst',
    'greedy_match_per_burst',
    'q_argmax_margin_per_burst',
    'q_action_std_per_burst',
    'q_autocorr_per_burst',
    'q_lambda_a_per_burst',
    'argmax_entropy_per_burst',
    'state_hash_n_unique_per_burst',
    'state_hash_entropy_per_burst',
    'state_repeat_rate_window64_per_burst',
    'mc_return__mean_axis_-1',
)


def main() -> None:
    runs = pl.read_parquet(RUNS)
    print(f'runs: {runs.height} cells')

    reads: set[str] = set()
    for t in TARGETS:
        reads.update(transitive_reads(t))
    schema = pl.scan_parquet(TRACES).collect_schema()
    available = set(schema.names())
    trace_cols = sorted((reads & available) | {'id'})
    print(f'trace cols to scan per cell: {len(trace_cols)}')

    accumulator: list[pl.DataFrame] = []
    t0 = time.time()
    for i, row in enumerate(runs.iter_rows(named=True)):
        cid = row['id']
        # lazy scan + filter by id → only this row materialises
        trace_row = (
            pl.scan_parquet(TRACES)
            .select(trace_cols)
            .filter(pl.col('id') == cid)
            .collect()
        )
        if trace_row.is_empty():
            print(f'  cell {i:3d} {cid[:8]}: no trace row, skip')
            continue
        joined = (
            pl.DataFrame([row])
            .join(trace_row, on='id', how='inner')
        )
        computed = compute_missing_columns(joined, list(TARGETS))
        keep_cols = ['id'] + [c for c in TARGETS if c in computed.columns]
        accumulator.append(computed.select(keep_cols))
        if (i + 1) % 5 == 0 or i == 0:
            elapsed = time.time() - t0
            print(f'  cell {i+1:3d}/{runs.height}  elapsed {elapsed:.0f}s  '
                  f'eta {elapsed * (runs.height - i - 1) / (i + 1):.0f}s', flush=True)
        del joined, trace_row, computed
        gc.collect()

    out_df = pl.concat(accumulator, how='diagonal_relaxed')
    print(f'\nfinal: {out_df.height} cells × {len(out_df.columns)} cols')
    for c in out_df.columns:
        if c == 'id':
            continue
        dtype = out_df[c].dtype
        n_nonnull = out_df[c].is_not_null().sum()
        print(f'  {c}: {dtype}, nonnull={n_nonnull}')
    out_df.write_parquet(OUT)
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
