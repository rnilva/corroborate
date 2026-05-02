"""One-shot rename: strip `outcome.` / `mechanism.` / `invariant.`
prefixes from `runs.parquet` columns across all corpora.

The Phase 5 schema rename ("drop substrate-paper-narrative
prefixes — bare measurable names") changed cell_runner's emit
from prefixed (`mechanism.jensen_gap`) to bare (`jensen_gap`).
Bridges + analyses now reference bare names. Pre-Phase-5
corpora on disk still carry the prefixes, so paired_g and the
test fixture corpora can't resolve names. This script renames
in-place + writes back to the same parquet path.

Operates on `runs.parquet` only — `traces.parquet` does not
carry prefixed columns.

Skips a column when stripping the prefix would collide with an
existing bare column on the same row (none observed across the
29 corpora as of 2026-05-02, but checked defensively).

Usage:
    uv run python scripts/sanitise_runs_parquet_prefixes.py
"""
from __future__ import annotations

import os
from pathlib import Path

# Pure polars rename; force CPU before any JAX-touching deps wander in.
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import polars as pl


_PREFIXES: tuple[str, ...] = ('mechanism.', 'outcome.', 'invariant.')


def _rename_map(cols: list[str]) -> dict[str, str]:
    """Build the {prefixed → bare} rename mapping for a column
    list. Skips columns whose bare form would collide with an
    existing bare column."""
    bare_set = set(cols)
    out: dict[str, str] = {}
    for c in cols:
        for p in _PREFIXES:
            if c.startswith(p):
                bare = c[len(p):]
                if not bare:
                    break
                if bare in bare_set:
                    print(
                        f'  COLLISION: {c!r} -> {bare!r} '
                        f'(bare exists; keeping prefixed)',
                    )
                    break
                if bare in out.values():
                    print(
                        f'  COLLISION: {c!r} -> {bare!r} '
                        f'(another column already remaps to it)',
                    )
                    break
                out[c] = bare
                break
    return out


def sanitise(parquet_path: Path) -> bool:
    """Rename in-place. Returns True if anything was renamed."""
    cols = pl.scan_parquet(parquet_path).collect_schema().names()
    rename = _rename_map(cols)
    if not rename:
        return False
    df = pl.read_parquet(parquet_path)
    df = df.rename(rename)
    df.write_parquet(parquet_path)
    return True


def main() -> None:
    data_root = Path('experiments/data')
    if not data_root.is_dir():
        raise SystemExit(f'no experiments/data at {data_root.resolve()}')

    n_renamed = 0
    n_total = 0
    for d in sorted(data_root.iterdir()):
        if not d.is_dir():
            continue
        runs = d / 'runs.parquet'
        if not runs.exists() or runs.stat().st_size < 1024:
            continue
        n_total += 1
        cols_before = pl.scan_parquet(runs).collect_schema().names()
        rename = _rename_map(cols_before)
        if not rename:
            continue
        print(f'\n[{d.name}] renaming {len(rename)} cols:')
        for old, new in sorted(rename.items()):
            print(f'  {old}  ->  {new}')
        if sanitise(runs):
            n_renamed += 1
    print()
    print(f'sanitised {n_renamed} / {n_total} corpora')


if __name__ == '__main__':
    main()
