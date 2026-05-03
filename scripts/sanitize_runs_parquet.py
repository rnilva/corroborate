"""Sanitise per-corpus `runs.parquet` files: drop legacy invariant-
machinery DSL columns and rename the surviving verdict to its
plain framework name.

Phase 4 (`scripts/migrate_universal_cache.py`) cleaned the
universal cache once. The per-corpus sources were never touched —
their `runs.parquet` files still carry the seven legacy
`at_most[jensen_dormancy_gap<=0].{reason,stats.*,targets,verdict}`
columns. That makes "raw traces" unusable for the framework's
"bridges should verify against raw, even when no cache exists"
contract: a fresh `build_cache` from a dirty `runs.parquet`
re-emits the legacy schema.

Two transforms (idempotent — re-running is a no-op):

1. **Drop legacy invariant sibling columns**
   `at_most[jensen_dormancy_gap<=0].reason`,
   `.targets`, `.stats.gap_value`, `.stats.threshold`,
   `.stats.measurable`, `.stats.kind`, `.stats.of_claim`.

2. **Rename the verdict column**
   `at_most[jensen_dormancy_gap<=0].verdict` →
   `jensen_dormancy_premise_active` (the plain-name form Phase 5
   migrated to).

Sanitisation is in-place via tmp + atomic rename, so a crash mid-
write leaves the original intact. `runs.parquet` files are tiny
(22–45 KB), so the temp footprint is negligible.

Atomic-rename means an already-archived dir's manifest sha256
will MISMATCH after sanitisation. The caller must either re-
archive the sanitised file (force=True) or accept the local-
clean / s3-dirty divergence.

Usage:
    PYTHONPATH=. uv run python scripts/sanitize_runs_parquet.py
        # Sanitise every experiments/data/<corpus>/runs.parquet
        # that has legacy columns. Idempotent.

    PYTHONPATH=. uv run python scripts/sanitize_runs_parquet.py \\
        action_dim_sweep cartpole_sync_1k
        # Sanitise only the named corpora.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import polars as pl  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / 'experiments' / 'data'

LEGACY_INVARIANT_COLUMNS: tuple[str, ...] = (
    'at_most[jensen_dormancy_gap<=0].reason',
    'at_most[jensen_dormancy_gap<=0].targets',
    'at_most[jensen_dormancy_gap<=0].stats.gap_value',
    'at_most[jensen_dormancy_gap<=0].stats.threshold',
    'at_most[jensen_dormancy_gap<=0].stats.measurable',
    'at_most[jensen_dormancy_gap<=0].stats.kind',
    'at_most[jensen_dormancy_gap<=0].stats.of_claim',
)
COLUMN_RENAMES: dict[str, str] = {
    'at_most[jensen_dormancy_gap<=0].verdict': 'jensen_dormancy_premise_active',
}


def sanitise_one(runs_path: Path) -> tuple[int, int]:
    """Sanitise `runs_path` in-place. Returns
    `(n_dropped, n_renamed)`. Idempotent: a clean parquet returns
    `(0, 0)` and is not rewritten."""
    cols = pl.scan_parquet(runs_path).collect_schema().names()
    cols_set = set(cols)
    to_drop = [c for c in LEGACY_INVARIANT_COLUMNS if c in cols_set]
    to_rename = {
        old: new for old, new in COLUMN_RENAMES.items() if old in cols_set
    }
    if not to_drop and not to_rename:
        return (0, 0)

    df = pl.read_parquet(runs_path)
    if to_drop:
        df = df.drop(to_drop)
    if to_rename:
        df = df.rename(to_rename)

    tmp = runs_path.with_suffix(runs_path.suffix + '.sanitise.tmp')
    df.write_parquet(tmp)
    tmp.replace(runs_path)  # atomic on POSIX
    return (len(to_drop), len(to_rename))


def discover_runs_parquets(
    only: tuple[str, ...] = (),
) -> list[Path]:
    """Find every `runs.parquet` under `experiments/data/<corpus>/`,
    optionally restricted to the named corpora."""
    if only:
        paths = [DATA_ROOT / name / 'runs.parquet' for name in only]
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise SystemExit(
                f'no runs.parquet at: {missing}',
            )
        return paths
    out: list[Path] = []
    for d in sorted(DATA_ROOT.iterdir()):
        if not d.is_dir():
            continue
        runs = d / 'runs.parquet'
        if runs.exists():
            out.append(runs)
    return out


def main(argv: list[str]) -> None:
    only = tuple(argv[1:])
    paths = discover_runs_parquets(only)
    print(f'scanning {len(paths)} runs.parquet')
    n_changed = 0
    for runs_path in paths:
        corpus = runs_path.parent.name
        try:
            n_drop, n_ren = sanitise_one(runs_path)
        except Exception as e:  # noqa: BLE001
            print(f'  {corpus}: ERROR {e!r}')
            continue
        if n_drop or n_ren:
            print(
                f'  {corpus}: dropped {n_drop} legacy cols, '
                f'renamed {n_ren} verdict cols.',
            )
            n_changed += 1
        else:
            print(f'  {corpus}: clean (no-op).')
    print(f'\nsanitised {n_changed}/{len(paths)} corpora.')


if __name__ == '__main__':
    main(sys.argv)
