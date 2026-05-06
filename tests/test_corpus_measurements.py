"""Tests for `corpus.measurements` — the per-corpus measurement
store from CACHE_BUILD.md Phase 1.

The two-level architecture: each corpus carries its own
`measurements.parquet` with every measurable ever computed for
its cells, keyed by `RunRow.id`. The per-hypothesis cache (Phase
2) becomes a cheap projection over these.

These tests pin the contract:
- `build_measurements` is idempotent (same inputs → no rewrite)
- Drift detection drops + recomputes drifted columns
- Orphan eviction drops registered measurables no longer required
- `load_measurements` projects to a column subset cheaply
- Atomic writes (no `.partial` after success)
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import polars as pl
import pytest

from corroborate.corpus.measurements import (
    MEASUREMENTS_FILENAME,
    build_measurements,
    current_signatures,
    load_measurements,
)
from corroborate.measurables import measurable


@measurable(reads=('x',))
def double_x(record: Mapping[str, object]) -> float:
    x = record.get('x')
    if not isinstance(x, (int, float)):
        raise TypeError('x missing')
    return 2.0 * float(x)


@measurable(reads=('x', 'y'))
def x_plus_y(record: Mapping[str, object]) -> float:
    x = record.get('x')
    y = record.get('y')
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        raise TypeError('x or y missing')
    return float(x) + float(y)


def _runs_df(n: int = 5) -> pl.DataFrame:
    return pl.DataFrame({
        'id': [f'cell-{i}' for i in range(n)],
        'x': [float(i) for i in range(n)],
        'y': [float(i * 10) for i in range(n)],
    })


# ============ Build basics ============


def test_build_measurements_writes_required_columns(tmp_path: Path) -> None:
    """Calling `build_measurements` populates the requested
    columns + sidecar."""
    runs_df = _runs_df(3)
    out = build_measurements(
        tmp_path,
        required=['double_x', 'x_plus_y'],
        runs_df=runs_df,
    )
    assert out == tmp_path / MEASUREMENTS_FILENAME
    assert out.exists()

    # Atomic write: no `.partial` artifact.
    assert not (
        tmp_path / (MEASUREMENTS_FILENAME + '.partial')
    ).exists()

    # Loaded contents include id + both measurable cols.
    df = load_measurements(tmp_path)
    assert df.columns == ['id', 'double_x', 'x_plus_y']
    assert df.height == 3
    assert df['double_x'].to_list() == [0.0, 2.0, 4.0]
    assert df['x_plus_y'].to_list() == [0.0, 11.0, 22.0]

    # Sidecar carries closure hashes for both measurables.
    sigs = current_signatures(tmp_path)
    assert set(sigs) == {'double_x', 'x_plus_y'}
    assert all(isinstance(v, str) and v for v in sigs.values())


def test_build_measurements_is_idempotent(tmp_path: Path) -> None:
    """Calling twice with same inputs is a no-op the second time
    (no parquet rewrite). Pin via mtime — the file's
    last-modified shouldn't change on the second call."""
    runs_df = _runs_df()
    build_measurements(
        tmp_path, required=['double_x'], runs_df=runs_df,
    )
    out_path = tmp_path / MEASUREMENTS_FILENAME
    mtime_before = out_path.stat().st_mtime_ns

    # Sleep briefly to ensure mtime granularity catches a rewrite.
    import time
    time.sleep(0.01)

    build_measurements(
        tmp_path, required=['double_x'], runs_df=runs_df,
    )
    mtime_after = out_path.stat().st_mtime_ns
    assert mtime_after == mtime_before, (
        f'idempotent rebuild rewrote parquet: '
        f'mtime {mtime_before} → {mtime_after}'
    )


def test_build_measurements_preserves_existing_columns(tmp_path: Path) -> None:
    """Adding a new required measurable preserves columns from
    the existing store. Pre-fix, a re-build with a different
    `required` list could accidentally clobber the old columns."""
    runs_df = _runs_df()
    build_measurements(
        tmp_path, required=['double_x'], runs_df=runs_df,
    )
    # Add x_plus_y. double_x should remain.
    build_measurements(
        tmp_path, required=['double_x', 'x_plus_y'], runs_df=runs_df,
    )
    df = load_measurements(tmp_path)
    assert {'double_x', 'x_plus_y'} <= set(df.columns)


# ============ Drift detection ============


def test_build_measurements_drops_drifted_column(tmp_path: Path) -> None:
    """When a measurable's closure hash changes, the corresponding
    column is dropped + recomputed on the next build. Simulate
    drift by monkeypatching the signature fn to return a
    different value the second time."""
    runs_df = _runs_df()
    # First build with the canonical signature.
    build_measurements(
        tmp_path, required=['double_x'], runs_df=runs_df,
    )
    sigs_before = dict(current_signatures(tmp_path))

    # Second build with a custom signature fn that returns a
    # NEW hash for double_x — the build sees drift and recomputes.
    def _drifted_sig(name: str) -> str | None:
        if name == 'double_x':
            return 'NEWHASH-' + sigs_before[name]
        return None

    build_measurements(
        tmp_path, required=['double_x'], runs_df=runs_df,
        measurable_signature_fn=_drifted_sig,
    )
    sigs_after = dict(current_signatures(tmp_path))
    assert sigs_after['double_x'] != sigs_before['double_x'], (
        f'drift expected: hash should have updated. '
        f'before={sigs_before["double_x"]}, after={sigs_after["double_x"]}'
    )


# ============ Orphan eviction ============


def test_build_measurements_drops_orphans(tmp_path: Path) -> None:
    """A column for a measurable NOT in the current `required`
    list is an orphan — dropped on the next build. Mirrors
    `_invalidate_drifted`'s C4 behavior at the per-corpus layer."""
    runs_df = _runs_df()
    # First build populates double_x AND x_plus_y.
    build_measurements(
        tmp_path,
        required=['double_x', 'x_plus_y'],
        runs_df=runs_df,
    )
    df = load_measurements(tmp_path)
    assert 'double_x' in df.columns
    assert 'x_plus_y' in df.columns

    # Second build only requires double_x; x_plus_y becomes orphan.
    build_measurements(
        tmp_path, required=['double_x'], runs_df=runs_df,
    )
    df_after = load_measurements(tmp_path)
    assert 'double_x' in df_after.columns
    assert 'x_plus_y' not in df_after.columns, (
        f'orphan x_plus_y not dropped: {df_after.columns}'
    )
    sigs_after = current_signatures(tmp_path)
    assert 'x_plus_y' not in sigs_after


# ============ load_measurements projection ============


def test_load_measurements_projects_columns(tmp_path: Path) -> None:
    """`columns` selects a subset; `id` always present."""
    runs_df = _runs_df()
    build_measurements(
        tmp_path,
        required=['double_x', 'x_plus_y'],
        runs_df=runs_df,
    )
    df = load_measurements(tmp_path, columns=['double_x'])
    assert df.columns == ['id', 'double_x']
    assert 'x_plus_y' not in df.columns


def test_load_measurements_returns_empty_when_absent(
    tmp_path: Path,
) -> None:
    """No measurements.parquet → empty DataFrame."""
    df = load_measurements(tmp_path)
    assert df.height == 0
    assert df.columns == []


# ============ Atomicity ============


def test_build_measurements_atomic_write_no_partial_on_success(
    tmp_path: Path,
) -> None:
    """C2/I4 invariant: no `.partial` files after successful
    build."""
    runs_df = _runs_df()
    build_measurements(
        tmp_path, required=['double_x'], runs_df=runs_df,
    )
    partials = list(tmp_path.glob('*.partial'))
    assert partials == [], (
        f'unexpected .partial files: {[p.name for p in partials]}'
    )


def test_build_measurements_validates_id_column(tmp_path: Path) -> None:
    """Missing `id` column raises ValueError. Pin the load-bearing
    primary-key contract."""
    bad_runs = pl.DataFrame({'x': [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match='`id` column'):
        build_measurements(
            tmp_path, required=['double_x'], runs_df=bad_runs,
        )


# ============ Partial-nullity awareness ============


def test_build_measurements_recomputes_null_cells_in_existing_column(
    tmp_path: Path,
) -> None:
    """**Partial-nullity recompute** (per measurable.py's check):
    when an existing column has SOME null cells, the next build
    recomputes ONLY those cells; non-null cells preserve their
    values.

    Pre-fix: build_measurements naively excluded required names
    that were already in `existing.columns`, so a column with
    NULL cells from a prior failed compute (transient missing
    input, code-fix-between-runs) stayed null forever despite
    the fix. Post-fix: pass the FULL required list to
    `compute_missing_columns`, which detects the null cells via
    `col.is_null().any()` and recomputes them.

    Construction: build measurements once with one runs_df; then
    write a measurements.parquet by hand with NULL cells; then
    rebuild. The null cells should be filled in.
    """
    runs_df = _runs_df(3)
    out_path = tmp_path / MEASUREMENTS_FILENAME

    # First build populates double_x for all 3 cells.
    build_measurements(
        tmp_path, required=['double_x'], runs_df=runs_df,
    )

    # Simulate "transient failure on cell 1" — write a manifest
    # with cell 1 nulled out. Sidecar still claims this is the
    # current closure-hash (no drift).
    df_with_null = pl.DataFrame({
        'id': ['cell-0', 'cell-1', 'cell-2'],
        'double_x': [0.0, None, 4.0],   # cell 1 is null
    })
    df_with_null.write_parquet(out_path)

    # Rebuild: the partial-nullity check should recompute cell 1.
    build_measurements(
        tmp_path, required=['double_x'], runs_df=runs_df,
    )
    df = load_measurements(tmp_path)
    vals = df.sort('id')['double_x'].to_list()
    assert vals == [0.0, 2.0, 4.0], (
        f'partial-nullity recompute failed: cell 1 should be '
        f'2.0 (recomputed), got {vals[1]}. cell 0 / 2 should '
        f'be preserved as 0.0 / 4.0.'
    )
