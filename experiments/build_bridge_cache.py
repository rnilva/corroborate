"""Build a per-corpus measurable cache for an authored bridges file.

Discovers every measurable a `Sequence[Bridge]` consumes
(transitively via the @measurable graph), computes each scalar
per-cell from `runs.parquet × traces.parquet`, and writes
`runs_with_mediators.parquet` with the original run cols + new
measurable cols.

Replaces `experiments/compute_mediators.py` (hardcoded measurable
list) with a bridge-driven discovery.

Usage:
  uv run python -m experiments.build_bridge_cache \\
      --module experiments.findings.ddqn_universe \\
      --corpus experiments/data/ddqn

`--module` must export a `Sequence[Bridge]` named via `--bridges-attr`
(default: `DDQN_UNIVERSE_BRIDGES`).

The discovered measurable set is the union over `bridge.source`,
`bridge.target`, and any `bridge.params[*]` string values that
resolve in the @measurable registry, transitively closed.
"""
from __future__ import annotations

import argparse
import importlib
import os

# Pure numpy on persisted traces. Force CPU before any JAX import.
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl

from corroborate.claim_bridge import Bridge, measurable_names_for_bridges
from corroborate.measurable import (
    evaluate_with_measurables, get_registered, transitive_reads,
)


def _bridges_from(module_name: str, attr: str) -> Sequence[Bridge]:
    mod = importlib.import_module(module_name)
    bridges = getattr(mod, attr)
    return bridges  # pyright: ignore[reportAny]


def build_cache(
    bridges: Sequence[Bridge],
    runs_path: Path,
    traces_path: Path,
    out_path: Path,
) -> None:
    names = sorted(measurable_names_for_bridges(bridges))
    if not names:
        print('no measurables required by these bridges; nothing to cache')
        return
    print(f'measurables to cache ({len(names)}):')
    for n in names:
        print(f'  {n}')

    runs_df = pl.read_parquet(runs_path)
    print(f'runs:   {runs_df.height} rows × {len(runs_df.columns)} cols')

    # Only pull the trace columns the requested measurables read.
    # Loading the full traces.parquet is multi-GB on long sweeps;
    # narrowing to the read set avoids OOM.
    trace_reads: set[str] = set()
    for n in names:
        trace_reads |= transitive_reads(n)
    runs_cols = set(runs_df.columns)
    needed_trace_cols = sorted(
        (k for k in trace_reads if k not in runs_cols),
    )
    if traces_path.exists() and traces_path.stat().st_size > 0 and needed_trace_cols:
        traces_df = pl.read_parquet(
            traces_path, columns=['id', *needed_trace_cols],
        )
        df = runs_df.join(traces_df, on='id', how='inner')
        print(
            f'traces: {traces_df.height} rows × '
            f'{len(needed_trace_cols)} cols (filtered from full file); '
            f'joined → {df.height} cells',
        )
    else:
        df = runs_df
        print('traces: not needed or unavailable; using runs only')

    # Compute each measurable per cell. evaluate_with_measurables
    # memoizes within-cell so transitive deps run once.
    new_cols: dict[str, list[object]] = {n: [] for n in names}
    for cell in df.iter_rows(named=True):
        cache: dict[str, object] = {}
        for n in names:
            m = get_registered(n)
            if m is None:
                new_cols[n].append(None)
                continue
            try:
                v = evaluate_with_measurables(m.fn, cell, cache=cache)
            except (KeyError, TypeError, ValueError):
                # Missing leaf reads → measurable can't resolve;
                # store None so downstream can NaN-skip.
                v = None
            new_cols[n].append(_to_polars_value(v))

    enriched = runs_df.with_columns([
        pl.Series(n, new_cols[n]) for n in names
    ])
    enriched.write_parquet(out_path)
    print(f'wrote: {out_path}  ({enriched.height} rows × {len(enriched.columns)} cols)')


def _to_polars_value(v: object) -> object:
    """Coerce a measurable's output to something polars accepts:
    scalars stay scalar; numpy arrays become Python lists (polars
    encodes as list-of-float); None passes through."""
    if v is None:
        return None
    if isinstance(v, (int, float, bool, str)):
        return v
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--module', required=True)
    parser.add_argument('--bridges-attr', default='DDQN_UNIVERSE_BRIDGES')
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--out-name', default='runs_with_mediators.parquet')
    args = parser.parse_args()

    bridges = _bridges_from(args.module, args.bridges_attr)
    corpus = Path(args.corpus)
    runs_path = corpus / 'runs.parquet'
    traces_path = corpus / 'traces.parquet'
    out_path = corpus / args.out_name

    if not runs_path.exists():
        raise SystemExit(f'no runs.parquet at {runs_path}')
    build_cache(bridges, runs_path, traces_path, out_path)


if __name__ == '__main__':
    main()
