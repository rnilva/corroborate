"""One-shot migration: inject `arm_key` column into a legacy
`runs.parquet` produced by a sweep run that pre-dated the typed
`Intervention` API (`src/corroborate/intervention.py`).

Behaviour:

1. Read `runs.parquet`.
2. If `arm_key` column already present AND any row has a value
   other than `'baseline'`, exit silently — the file is already
   migrated.
3. Otherwise, derive arm_key from the `intervention_name`
   column:
   - `'vanilla_dqn'` → `'baseline'`
   - `'ddqn'` → the canonical DDQN arm_key, computed dynamically
     from `Intervention(slot_path='bootstrap',
     replacement=partial(bootstrap, greedification=
     double_greedify)).arm_key()`. Hardcoding the string would
     drift if the canonical-fingerprint format changes.
4. Write back atomically via `runs.parquet.migrate.tmp` →
   `runs.parquet`.

The script is intentionally narrow: the sweep harness produces
exactly two intervention names ('vanilla_dqn', 'ddqn'); any other
name is a sign of corpus drift and the script raises rather than
guess. Add a new branch when extending to PER, NoisyArgmax, etc.

Run `uv run python experiments/migrate_runs_inject_arm_key.py`
after the in-flight sweep completes."""
from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import polars as pl

from corroborate.intervention import Intervention
from corroborate.rl.dqn.claims.bootstrap import bootstrap, double_greedify


def ddqn_arm_key() -> str:
    """Canonical DDQN arm_key. Derived from the typed
    Intervention so any change to the fingerprint format here
    automatically updates the migration target."""
    return Intervention(
        slot_path='bootstrap',
        replacement=partial(bootstrap, greedification=double_greedify),
    ).arm_key()


def migrate(runs_path: Path) -> None:
    """Inject `arm_key` into the parquet at `runs_path`."""
    if not runs_path.exists():
        raise FileNotFoundError(f'runs.parquet not found at {runs_path}')

    df = pl.read_parquet(runs_path)
    print(f'read {len(df)} rows from {runs_path}')

    if 'intervention_name' not in df.columns:
        raise ValueError(
            f'{runs_path} has no `intervention_name` column; '
            f'cannot migrate without it',
        )

    ddqn_key = ddqn_arm_key()
    print(f'  DDQN canonical arm_key: {ddqn_key!r}')

    # Idempotency: only skip if arm_key is FULLY populated with
    # canonical values (no nulls, includes the canonical DDQN key).
    # A partially-populated column (e.g. a fresh sweep merge that
    # didn't set arm_key on the new rows) re-runs to fill the
    # gaps — the mapping is deterministic so re-population is safe.
    if 'arm_key' in df.columns:
        existing_keys = set(df['arm_key'].drop_nulls().unique().to_list())
        null_count = int(df['arm_key'].null_count())
        fully_populated = (null_count == 0 and ddqn_key in existing_keys)
        if fully_populated:
            print(
                f'arm_key column fully populated with non-default '
                f'values {sorted(existing_keys)!r}; nothing to do.'
            )
            return
        print(
            f'  re-populating arm_key '
            f'(null_count={null_count}, '
            f'has_canonical_ddqn={ddqn_key in existing_keys})'
        )

    intervention_names = set(df['intervention_name'].unique().to_list())
    print(f'  intervention names: {sorted(intervention_names)!r}')
    unknown = intervention_names - {'vanilla_dqn', 'ddqn'}
    if unknown:
        raise ValueError(
            f'unknown intervention names {sorted(unknown)!r}; '
            f'extend the migration script to map these to arm_keys',
        )

    df = df.with_columns(
        pl.when(pl.col('intervention_name') == 'ddqn')
        .then(pl.lit(ddqn_key))
        .otherwise(pl.lit('baseline'))
        .alias('arm_key'),
    )
    counts = (
        df.group_by('arm_key').agg(pl.len().alias('n')).sort('arm_key')
    )
    print(f'  arm_key distribution after migration:')
    for row in counts.iter_rows(named=True):
        print(f'    {row["arm_key"]!r}: {row["n"]}')

    tmp_path = runs_path.with_suffix('.parquet.migrate.tmp')
    df.write_parquet(tmp_path)
    tmp_path.replace(runs_path)
    print(f'wrote migrated runs.parquet ({len(df)} rows)')


def main() -> None:
    base = Path('/workspace/corroborate/experiments/data/ddqn')
    runs_path = base / 'runs.parquet'
    migrate(runs_path)
    runs_with_mediators = base / 'runs_with_mediators.parquet'
    if runs_with_mediators.exists():
        print()
        print(f'also migrating {runs_with_mediators}')
        migrate(runs_with_mediators)


if __name__ == '__main__':
    main()
    sys.exit(0)
