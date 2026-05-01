"""Append a single sweep corpus to the universal paired-delta
datasets — additively, no full rebuild.

Pattern: each corpus owns its own paired-delta extraction; the
universal parquet is a `diagonal_relaxed` concat of all
per-corpus shards. New corpus → run this script once → its rows
land in the universal. New column added by one corpus →
null-pads in others under `diagonal_relaxed`. No rebuild
required.

Usage:
  uv run python scripts/append_corpus_to_universal.py <corpus_name>

Both `paired_delta_cells.parquet` (cell-mean) and
`paired_delta_per_burst.parquet` (per-burst, when available) are
appended. The universal parquets at
`experiments/data/ddqn_universal/` accumulate.
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

# Re-use the per-corpus extractors from the existing builders.
# Their _build_corpus_cells / _build_corpus_burst functions
# already encapsulate the per-corpus logic; we just call them
# without iterating the hard-coded _CORPORA list.
sys.path.insert(0, str(Path(__file__).parent.parent))
from experiments.build_universal_ddqn_dataset import (  # noqa: E402
    _build_corpus_cells, _OUT_FILE as _CELLS_OUT,
)
from experiments.build_universal_per_burst_dataset import (  # noqa: E402
    _build_corpus_per_burst, _OUT as _BURST_OUT,
)


def _append_to_universal(
    new_rows: list[dict[str, object]],
    universal_path: Path,
    label: str,
) -> None:
    """Append `new_rows` to `universal_path` via
    `diagonal_relaxed` concat. Null-pads any column mismatches.
    No-op when `new_rows` is empty."""
    if not new_rows:
        print(f'  {label}: no rows to append, skipping')
        return
    new_df = pl.DataFrame(new_rows)
    if universal_path.exists():
        existing = pl.read_parquet(universal_path)
        n_before = existing.shape[0]
        # diagonal_relaxed null-pads + type-widens across schema
        # diffs — exactly what an additive append should do.
        merged = pl.concat([existing, new_df], how='diagonal_relaxed')
        n_after = merged.shape[0]
    else:
        n_before = 0
        merged = new_df
        n_after = new_df.shape[0]
    universal_path.parent.mkdir(parents=True, exist_ok=True)
    merged.write_parquet(universal_path)
    print(
        f'  {label}: {n_before} → {n_after} rows '
        f'({n_after - n_before:+d}) at {universal_path}',
    )


def main() -> None:
    if len(sys.argv) != 2:
        print('usage: append_corpus_to_universal.py <corpus_name>')
        sys.exit(1)
    corpus = sys.argv[1]

    print(f'corpus: {corpus}\n')
    cells = _build_corpus_cells(corpus) or []
    burst_rows = _build_corpus_per_burst(corpus) or []
    print(
        f'  extracted: {len(cells)} cell-mean rows, '
        f'{len(burst_rows)} per-burst rows',
    )
    print()
    _append_to_universal(cells, _CELLS_OUT, 'paired_delta_cells')
    _append_to_universal(burst_rows, _BURST_OUT, 'paired_delta_per_burst')


if __name__ == '__main__':
    main()
