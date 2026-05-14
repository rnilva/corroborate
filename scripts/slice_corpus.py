"""Slice a corpus to a shorter training regime by truncating
per-step and per-burst arrays.

Use case: a converged env (e.g., FourRooms saturating at burst 1)
has its 1M-step traces sliced to its 200k pre-saturation regime,
producing a fresh corpus where `total_steps=200000`. The
canonical scope predicate then admits this env at its
fast-converging budget while keeping slow envs at 1M.

Pipeline:
    corroborate restore <src_dir> --files traces.parquet
    uv run python scripts/slice_corpus.py <src_dir> <dst_dir> \\
        --keep-bursts 2
    uv run python scripts/run_hypothesis.py <module> \\
        --evict <old_corpus_name>
    uv run python scripts/run_hypothesis.py <module> \\
        --ingest <new_corpus_name>

Slicing semantics:
  - new UUID for every cell (no provenance collision)
  - `total_steps` field set to `keep_bursts × eval_every`
  - per-step list columns (length == old total_steps) sliced to
    first `keep_bursts × eval_every` elements via `list.slice`
  - per-burst list columns (leading dim == old n_bursts) sliced
    to first `keep_bursts` elements via `list.slice`
  - `_remote.json` NOT copied (sliced corpus is local-derived)
"""
from __future__ import annotations

import argparse
import shutil
import sys
import uuid
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[1]


def _list_lengths(df: pl.DataFrame) -> dict[str, int]:
    """Return col → first-row list length for every List-typed column."""
    out: dict[str, int] = {}
    if df.is_empty():
        return out
    head = df.head(1)
    for col in df.columns:
        dtype = df.schema[col]
        if isinstance(dtype, pl.List):
            try:
                n = head.select(pl.col(col).list.len()).item()
                if n is not None:
                    out[col] = int(n)
            except Exception:
                pass
    return out


def slice_corpus(
    src: Path, dst: Path, *, keep_bursts: int,
) -> None:
    """Slice src → dst keeping the first `keep_bursts` evaluation bursts.

    Streams through polars expressions — no per-cell Python loop
    over the large per-step list columns."""
    runs_path = src / 'runs.parquet'
    traces_path = src / 'traces.parquet'
    if not runs_path.exists():
        raise SystemExit(f'no runs.parquet at {src}')
    if not traces_path.exists():
        raise SystemExit(
            f'no traces.parquet at {src} — restore via '
            f'`corroborate restore {src} --files traces.parquet`',
        )

    runs = pl.read_parquet(runs_path)
    traces = pl.read_parquet(traces_path)

    # Per-cell derived sizing — must be uniform within corpus.
    r0 = runs.head(1).to_dicts()[0]
    eval_every = int(r0['eval_every'])
    old_total_steps = int(r0['total_steps'])
    old_n_bursts = old_total_steps // eval_every
    if keep_bursts >= old_n_bursts:
        raise SystemExit(
            f'keep_bursts={keep_bursts} >= old_n_bursts='
            f'{old_n_bursts}; no slicing would occur.',
        )
    new_total_steps = keep_bursts * eval_every
    keep_steps = new_total_steps

    # Map old_id → new_id so traces inherit matched IDs.
    id_map = {cid: str(uuid.uuid4()) for cid in runs['id'].to_list()}
    id_remap_expr = pl.col('id').replace_strict(id_map)

    def slice_lists(df: pl.DataFrame) -> pl.DataFrame:
        """Apply list.slice per recognised column shape."""
        lens = _list_lengths(df)
        exprs: list[pl.Expr] = []
        for col, n in lens.items():
            if n == old_total_steps:
                exprs.append(pl.col(col).list.slice(0, keep_steps))
            elif n == old_n_bursts:
                exprs.append(pl.col(col).list.slice(0, keep_bursts))
            # else: leave as-is; not a recognised grain
        if exprs:
            df = df.with_columns(exprs)
        return df

    runs_new = slice_lists(runs).with_columns(
        id_remap_expr,
        pl.lit(new_total_steps).alias('total_steps'),
    )
    traces_new = slice_lists(traces).with_columns(
        id_remap_expr,
        pl.lit(new_total_steps).alias('total_steps'),
    )

    dst.mkdir(parents=True, exist_ok=True)
    runs_new.write_parquet(dst / 'runs.parquet')
    traces_new.write_parquet(dst / 'traces.parquet')

    graphs_src = src / 'graphs.json'
    if graphs_src.exists():
        shutil.copy(graphs_src, dst / 'graphs.json')

    print(f'sliced {src.name} → {dst.name}')
    print(f'  cells:        {runs.shape[0]} → {runs_new.shape[0]}')
    print(f'  total_steps:  {old_total_steps:,} → {new_total_steps:,}')
    print(f'  bursts:       {old_n_bursts} → {keep_bursts}')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='slice_corpus')
    parser.add_argument('src', type=Path)
    parser.add_argument('dst', type=Path)
    parser.add_argument('--keep-bursts', type=int, required=True)
    args = parser.parse_args(argv)
    slice_corpus(args.src, args.dst, keep_bursts=args.keep_bursts)
    return 0


if __name__ == '__main__':
    sys.exit(main())
