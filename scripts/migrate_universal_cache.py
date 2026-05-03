"""One-shot cleanup of `experiments/data/universal_evidence.parquet`.

Two transforms (both reversible — original parquet is preserved
alongside the migrated one):

1. **Drop legacy invariant-machinery sibling columns**
   `at_most[jensen_dormancy_gap<=0].reason`, `.targets`,
   `.stats.gap_value`, `.stats.threshold`, `.stats.measurable`,
   `.stats.kind`, `.stats.of_claim` — runtime invariant-evaluation
   sub-fields that leaked alongside the registered measurable's
   `.verdict`. Phase 5 (`jensen_dormancy_premise_active` rename)
   moved to a clean structured form; the sibling columns are
   purely legacy noise.

2. **Rename the verdict column**
   `at_most[jensen_dormancy_gap<=0].verdict` →
   `jensen_dormancy_premise_active`. Aligns the cache with the
   substrate's renamed `@measurable` so bridges that read the
   plain name find their data on the migrated parquet.

Arm-name suffix stripping (e.g. `ddqn_g099` → `ddqn`) is OUT of
scope for this script — it touches structural-vs-HP bridge
authoring decisions that need per-sweep judgment. Bridges that
need a specific HP-encoded arm consume the corpus's actual arm
shape via per-bridge `source = DoEffect(treatment_arm=...,
baseline_arm=...)` override.

Usage:
    PYTHONPATH=. uv run python scripts/migrate_universal_cache.py
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import polars as pl  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = REPO_ROOT / 'experiments' / 'data' / 'universal_evidence.parquet'
BACKUP = REPO_ROOT / 'experiments' / 'data' / 'universal_evidence.before-phase4.parquet'

# Columns dropped: legacy invariant-machinery leakage that doesn't
# belong on cells (the registered measurable now emits one clean
# verdict column, nothing else).
LEGACY_INVARIANT_COLUMNS: tuple[str, ...] = (
    'at_most[jensen_dormancy_gap<=0].reason',
    'at_most[jensen_dormancy_gap<=0].targets',
    'at_most[jensen_dormancy_gap<=0].stats.gap_value',
    'at_most[jensen_dormancy_gap<=0].stats.threshold',
    'at_most[jensen_dormancy_gap<=0].stats.measurable',
    'at_most[jensen_dormancy_gap<=0].stats.kind',
    'at_most[jensen_dormancy_gap<=0].stats.of_claim',
)

# Single rename: DSL-encoded → plain name.
COLUMN_RENAMES: dict[str, str] = {
    'at_most[jensen_dormancy_gap<=0].verdict': 'jensen_dormancy_premise_active',
}


def main() -> None:
    if not ORIGINAL.exists():
        raise SystemExit(f'no parquet at {ORIGINAL}')

    if not BACKUP.exists():
        print(f'backing up to {BACKUP.name}')
        BACKUP.write_bytes(ORIGINAL.read_bytes())
    else:
        print(f'backup already exists at {BACKUP.name}; skipping copy')

    df = pl.read_parquet(ORIGINAL)
    n_rows = len(df)
    print(f'loaded {n_rows} cells × {len(df.columns)} columns')

    cols_present = set(df.columns)

    # 1) Drop legacy invariant sibling columns
    to_drop = [c for c in LEGACY_INVARIANT_COLUMNS if c in cols_present]
    if to_drop:
        print(f'dropping {len(to_drop)} legacy invariant columns:')
        for c in to_drop:
            print(f'  - {c}')
        df = df.drop(to_drop)
    else:
        print('no legacy invariant columns to drop')

    # 2) Rename verdict column to plain name
    renamed: dict[str, str] = {}
    for old, new in COLUMN_RENAMES.items():
        if old in df.columns:
            renamed[old] = new
    if renamed:
        print(f'renaming {len(renamed)} columns:')
        for old, new in renamed.items():
            print(f'  {old} → {new}')
        df = df.rename(renamed)
    else:
        print('no columns to rename')

    df.write_parquet(ORIGINAL)
    print(f'wrote {len(df)} cells × {len(df.columns)} columns to {ORIGINAL.name}')

    # Verify
    reread = pl.read_parquet(ORIGINAL)
    assert len(reread) == n_rows, 'row count changed'
    for c in to_drop:
        assert c not in reread.columns, f'{c} still present'
    for new in renamed.values():
        assert new in reread.columns, f'{new} not present'
    print('verification passed')


if __name__ == '__main__':
    main()
