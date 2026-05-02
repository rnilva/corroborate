"""One-shot rename: strip `outcome.` / `mechanism.` / `invariant.`
prefixes from columns across all parquet files under
`experiments/data/`.

The Phase 5 schema rename ("drop substrate-paper-narrative
prefixes — bare measurable names") changed cell_runner's emit
from prefixed (`mechanism.jensen_gap`) to bare (`jensen_gap`).
Bridges + analyses now reference bare names. Pre-Phase-5
artifacts on disk still carry the prefixes, so paired_g and
the test fixture corpora can't resolve names. This script
renames in-place + writes back to the same parquet path.

Walks every `*.parquet` under `experiments/data/` recursively:
- per-corpus `runs.parquet`
- per-arm shards (`arm000__<env>__<arm>__runs.parquet` etc.)
- bridge caches (`runs_with_bridge_cache.parquet`)
- the universal evidence parquet (`universal_evidence.parquet`)
- anything else that contains prefixed columns

`traces.parquet` files don't carry prefixed columns; they
flow through unchanged when re-scanned.

Skips a column when stripping the prefix would collide with an
existing bare column (none observed across the 29 corpora as of
2026-05-02, but checked defensively).

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
    for parquet in sorted(data_root.rglob('*.parquet')):
        try:
            cols_before = pl.scan_parquet(parquet).collect_schema().names()
        except Exception as exc:  # noqa: BLE001 — log + skip unreadable
            print(f'[{parquet}] UNREADABLE: {exc!r}')
            continue
        n_total += 1
        rename = _rename_map(cols_before)
        if not rename:
            continue
        rel = parquet.relative_to(data_root)
        print(f'\n[{rel}] renaming {len(rename)} cols')
        if sanitise(parquet):
            n_renamed += 1
    print()
    print(f'sanitised {n_renamed} / {n_total} parquet files')


if __name__ == '__main__':
    main()
