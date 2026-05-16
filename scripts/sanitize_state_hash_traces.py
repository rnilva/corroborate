"""One-shot: rename the trace column `state_hash` →
`state_hash_per_step` in existing `traces.parquet` files.

Background. Pre-2026-05-16, `phases.py` emitted the per-step state
bucket under the trace key `state_hash` — which collided with the
configurational LEAF `state_hash` (the env-side hash callable
surfaced by `walk_paths` as a kwarg of the `dqn` claim). The
trace/leaf collision broke the runs↔traces join for state-
conditional measurables (the leaf column carries the callable
repr, not int buckets; the join silently kept the leaf and
dropped the trace).

Commit (pending) renames the emission to `state_hash_per_step`
on the substrate side. This script does the corresponding
parquet-column rename on existing `traces.parquet` files written
under the old name.

The rename is structural only — no data is rewritten, just the
column metadata. Uses `pyarrow.parquet` row-group streaming so
peak RAM is bounded by the largest single row group (~256 MB on
default-chunked traces) rather than the whole file.

Usage
-----
Single file::

    PYTHONPATH=. uv run python scripts/sanitize_state_hash_traces.py \\
        experiments/data/<corpus>/traces.parquet

Recursive walk under a corpus root::

    PYTHONPATH=. uv run python scripts/sanitize_state_hash_traces.py \\
        --recurse experiments/data/

Skips files that don't contain a `state_hash` column (already
renamed, or never had one).

After sanitizing local traces, re-archive to cloud via the usual
`corroborate archive` flow so cloud-side mirrors carry the
renamed column too.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

OLD = 'state_hash'
NEW = 'state_hash_per_step'


def _rename_in_table(table: pa.Table) -> pa.Table:
    """Apply the column rename to a pyarrow Table. If `OLD` is not
    present (or `NEW` already is), return unchanged."""
    if OLD not in table.column_names:
        return table
    if NEW in table.column_names:
        # Would-be collision: somebody already partially renamed.
        # Refuse rather than clobber.
        raise RuntimeError(
            f'Both {OLD!r} and {NEW!r} exist in the same table — '
            'refuse to rename and clobber. Inspect the file manually.'
        )
    new_names = [NEW if n == OLD else n for n in table.column_names]
    return table.rename_columns(new_names)


def sanitize_one(path: Path, *, backup: bool, dry_run: bool) -> bool:
    """Rename `state_hash` → `state_hash_per_step` in one parquet
    file. Returns True if a rename actually happened, False if
    the file already lacked the old column (idempotent no-op).

    Streams row groups via pyarrow to keep peak RAM bounded."""
    pf = pq.ParquetFile(path)
    if OLD not in pf.schema_arrow.names:
        print(f'[skip ] {path} — no {OLD!r} column')
        return False
    if NEW in pf.schema_arrow.names:
        raise RuntimeError(
            f'{path}: both {OLD!r} and {NEW!r} present — manual inspection'
        )
    n_rows = pf.metadata.num_rows
    n_groups = pf.num_row_groups
    print(
        f'[hit  ] {path} — {n_rows} rows / {n_groups} row groups; '
        'renaming column'
    )
    if dry_run:
        return True
    if backup:
        bak = path.with_suffix(path.suffix + '.bak')
        if not bak.exists():
            shutil.copy2(path, bak)
            print(f'[backup] {path} -> {bak}')
    # Stream to a sibling temp file, then atomically replace.
    tmp = path.with_suffix(path.suffix + '.renaming')
    schema_renamed = pa.schema(
        [
            pa.field(NEW if f.name == OLD else f.name, f.type)
            for f in pf.schema_arrow
        ],
        metadata=pf.schema_arrow.metadata,
    )
    writer = pq.ParquetWriter(tmp, schema_renamed, compression='zstd')
    try:
        for i in range(n_groups):
            tbl = pf.read_row_group(i)
            tbl_renamed = _rename_in_table(tbl)
            # cast to the unified schema (preserves metadata)
            tbl_renamed = tbl_renamed.cast(schema_renamed)
            writer.write_table(tbl_renamed)
    finally:
        writer.close()
    tmp.replace(path)
    print(f'[write] {path}')
    return True


def _walk(root: Path) -> list[Path]:
    """Yield every `traces.parquet` under `root` (recursive)."""
    return sorted(root.rglob('traces.parquet'))


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        'paths',
        nargs='+',
        type=Path,
        help='Either traces.parquet file(s) or directories (with --recurse)',
    )
    p.add_argument(
        '--recurse',
        action='store_true',
        help='Treat each path as a directory; walk recursively for traces.parquet',
    )
    p.add_argument(
        '--no-backup',
        action='store_true',
        help='Skip the .bak backup (default: keep one for each renamed file)',
    )
    p.add_argument(
        '--dry-run',
        action='store_true',
        help='Report what would be renamed without modifying files',
    )
    args = p.parse_args(argv)

    targets: list[Path] = []
    for raw in args.paths:
        if args.recurse:
            if not raw.is_dir():
                print(f'[error] {raw} is not a directory (--recurse requires dirs)')
                return 2
            targets.extend(_walk(raw))
        else:
            if not raw.is_file():
                print(f'[error] {raw} is not a file (use --recurse for directories)')
                return 2
            targets.append(raw)

    if not targets:
        print('[error] no traces.parquet files found')
        return 1

    n_hit = 0
    for path in targets:
        try:
            if sanitize_one(path, backup=not args.no_backup, dry_run=args.dry_run):
                n_hit += 1
        except Exception as exc:
            print(f'[fail ] {path}: {exc}')
            return 3
    print(f'\n[done ] renamed in {n_hit} / {len(targets)} files')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
