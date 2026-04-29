"""Post-merge pass: tighten dtypes on a traces parquet.

Reads the parquet at the given path, casts `List(Float64)` →
`List(Float32)` and `List(Int64)` → `List(Int32)` via
`corroborate.persistence.tighten_trace_dtypes`, writes back via
atomic-rename (.tmp + replace).

The cell_runner emits per-step series via `arr.tolist()` which
upcasts JAX float32 → Python float (= float64 in polars).
This pass undoes the round-trip waste — ~13% on-disk reduction
with no information loss.

Separated from `collect_ddqn_runs.py`'s merge step because
fusing the tighten into the merge's streaming sink can OOM on
large diagonal_relaxed concats. As an isolated streaming
read-tighten-write pass, it stays memory-bounded.

Run: `uv run python scripts/tighten_traces.py path/to/traces.parquet`
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import polars as pl

from corroborate.persistence import tighten_trace_dtypes


def main(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f'not found: {path}')
    src_size = path.stat().st_size
    print(f'tightening: {path}  ({src_size / 1e9:.2f} GB)')

    tmp = path.with_suffix(path.suffix + '.tightened.tmp')
    tighten_trace_dtypes(pl.scan_parquet(path)).sink_parquet(tmp)

    tightened_size = tmp.stat().st_size
    ratio = tightened_size / src_size if src_size > 0 else 1.0
    print(f'  before: {src_size / 1e9:.2f} GB')
    print(f'  after:  {tightened_size / 1e9:.2f} GB  ({ratio:.0%})')

    # Atomic-rename only after the new file is fully written.
    tmp.replace(path)
    print(f'  swapped → {path}')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit(
            'usage: uv run python scripts/tighten_traces.py '
            '<traces.parquet>'
        )
    main(Path(sys.argv[1]))
