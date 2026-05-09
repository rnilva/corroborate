# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
"""Inspect a trace parquet's schema + which measurables can be
derived from its columns.

Usage:
    PYTHONPATH=. uv run python scripts/trace_schema.py \\
        experiments/data/<corpus>/traces.parquet [<bridges_module>]

Prints two sections:
  1. **Trace columns** — all columns in the parquet, their dtype,
     and a row-group count. Read-only — no decompressed data
     materialised, just metadata.
  2. **Measurables computable from this trace** — when a bridges
     module is supplied, every registered measurable whose
     `transitive_reads` are satisfied by the trace columns gets
     listed alongside what scope its transitive reads cover.

Avoids the typical `pl.read_parquet(path)` workflow that
materialises ~5 GB of trace lists just to inspect schema.

Example:
    $ python scripts/trace_schema.py \\
          experiments/data/ddqn/traces.parquet \\
          experiments.findings.ddqn_universe

    trace columns (47):
      id: Utf8
      gamma: Float64
      online_max_q_per_step: List(Float64) [per-step]
      ... [44 more]

    file metadata:
      row groups: 12
      total rows: 2160
      compressed size: 3.2 GB

    measurables computable on this trace (with substrate
    experiments.findings.ddqn_universe loaded):
      target_staleness_late: needs {online_max_q_per_step,
                                    target_max_q_per_step} ✓
      jensen_dormancy_gap: needs {predicted_q_at_start,
                                  mc_return, online_std_q_per_step,
                                  env_name} — MISSING
                                  online_std_q_per_step
      ... [others]
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog='trace_schema',
        description='Inspect a trace parquet without loading it.',
    )
    parser.add_argument(
        'path', type=Path,
        help='trace parquet file (typically <corpus>/traces.parquet)',
    )
    parser.add_argument(
        'module', type=str, nargs='?', default=None,
        help='optional bridges module (e.g. '
             'experiments.findings.ddqn_universe). When provided, '
             'lists every registered measurable whose transitive '
             'reads are satisfied by the trace.',
    )
    args = parser.parse_args(argv)
    path: Path = args.path
    if not path.exists():
        print(f'trace_schema: file not found: {path}', file=sys.stderr)
        return 1
    return _print_schema(path, args.module)


def _print_schema(path: Path, module_name: str | None) -> int:
    import polars as pl
    import pyarrow.parquet as pq

    # Schema-only read (no decompression) via pyarrow + polars.
    schema = pl.scan_parquet(path).collect_schema()
    pq_meta = pq.ParquetFile(str(path)).metadata
    cols = list(schema.names())
    print(f'trace columns ({len(cols)}):')
    list_cols = []
    scalar_cols = []
    for c in cols:
        dt = schema[c]
        marker = ''
        if 'List' in str(dt):
            marker = ' [per-step / per-burst]'
            list_cols.append(c)
        else:
            scalar_cols.append(c)
        print(f'  {c}: {dt}{marker}')
    print()
    print('file metadata:')
    print(f'  row groups: {pq_meta.num_row_groups}')
    print(f'  total rows: {pq_meta.num_rows}')
    on_disk = path.stat().st_size
    print(f'  on-disk size: {_humanize_bytes(on_disk)}')
    # Approximate decompressed size from row-group metadata.
    decomp = sum(
        pq_meta.row_group(i).total_byte_size
        for i in range(pq_meta.num_row_groups)
    )
    print(f'  decompressed (sum row-group total_byte_size): '
          f'{_humanize_bytes(decomp)}')
    print()
    print(f'  list-typed cols: {len(list_cols)} ({", ".join(list_cols) or "—"})')
    print(f'  scalar-typed cols: {len(scalar_cols)}')

    if module_name is None:
        return 0

    # Load the substrate to register measurables.
    print()
    print(f'loading substrate: {module_name}')
    try:
        importlib.import_module(module_name)
    except ImportError as e:
        print(f'  ERROR: {e}', file=sys.stderr)
        return 2
    from corroborate.measurables.measurable import (
        registered_names, transitive_reads,
    )
    available = set(cols)
    print()
    print(
        f'measurables computable on this trace '
        f'({len(registered_names())} registered total):',
    )
    satisfied: list[tuple[str, frozenset[str]]] = []
    unsatisfied: list[tuple[str, frozenset[str], frozenset[str]]] = []
    for name in sorted(registered_names()):
        try:
            reads = transitive_reads(name)
        except KeyError:
            continue
        # Filter to reads that look like trace cols (not pure HPs).
        trace_like = {
            r for r in reads
            if r in available
            or '_per_step' in r
            or '_per_burst' in r
            or r in (
                'mc_return', 'predicted_q_at_start',
                'episode_length', 'done', 'reward', 'loss',
                'td_error',
            )
        }
        if not trace_like:
            # Pure HP measurable, doesn't need this trace.
            continue
        missing = trace_like - available
        if missing:
            unsatisfied.append((name, frozenset(reads), frozenset(missing)))
        else:
            satisfied.append((name, frozenset(reads)))
    print(f'  satisfied ({len(satisfied)}):')
    for name, reads in satisfied:
        print(f'    {name}: needs {sorted(reads)}')
    print(f'  unsatisfied ({len(unsatisfied)}):')
    for name, reads, missing in unsatisfied:
        print(
            f'    {name}: needs {sorted(reads)} '
            f'— MISSING {sorted(missing)}',
        )
    return 0


def _humanize_bytes(n: int) -> str:
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if n < 1024:
            return f'{n:.1f} {unit}'
        n //= 1024
    return f'{n:.1f} PB'


if __name__ == '__main__':
    sys.exit(main())
