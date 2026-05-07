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
    check_drift,
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
    """**C4 invariant** (CACHE_BUILD.md): when a measurable's
    closure hash drifts, the column is DROPPED + RECOMPUTED on
    the next build. The previous reading-back-the-stamped-hash
    test was tautological per CLAUDE.md §"Test principle" rule 4
    — it asserted what the build just wrote, never confirming
    a recomputation actually happened.

    Substrate-grounded probe: corrupt the persisted column with
    sentinel values [999, 999, 999], trigger drift via a fake
    signature fn, build, and confirm the column is back to the
    canonical [0, 2, 4]. If drift+drop+recompute did NOT happen,
    the corrupted [999, 999, 999] sentinel would persist (the
    build would short-circuit through the no-op fast-path).
    """
    import polars as pl

    runs_df = _runs_df(3)
    build_measurements(
        tmp_path, required=['double_x'], runs_df=runs_df,
    )
    # Pin the canonical pre-corrupt values.
    canonical = load_measurements(tmp_path).sort('id')['double_x'].to_list()
    assert canonical == [0.0, 2.0, 4.0]
    sigs_before = current_signatures(tmp_path)

    # Corrupt the stored column with a sentinel — the framework
    # would never produce these for `double_x = 2 * x`.
    corrupted = pl.DataFrame({
        'id': ['cell-0', 'cell-1', 'cell-2'],
        'double_x': [999.0, 999.0, 999.0],
    })
    corrupted.write_parquet(tmp_path / MEASUREMENTS_FILENAME)
    # Sanity: the corruption took.
    assert load_measurements(tmp_path).sort('id')['double_x'].to_list() == (
        [999.0, 999.0, 999.0]
    )

    # Drift signal — sig fn reports a NEW hash for double_x.
    def _drifted_sig(name: str) -> str | None:
        if name == 'double_x':
            return 'NEWHASH-' + sigs_before[name]
        return None

    build_measurements(
        tmp_path, required=['double_x'], runs_df=runs_df,
        measurable_signature_fn=_drifted_sig,
    )

    # The drifted column must have been dropped + RECOMPUTED. If
    # the framework had instead kept the existing column (no drop)
    # OR taken the no-op fast-path, [999, 999, 999] would persist.
    recomputed = load_measurements(tmp_path).sort('id')['double_x'].to_list()
    assert recomputed == [0.0, 2.0, 4.0], (
        f'drift→drop→recompute pipeline failed: column values '
        f'should be canonical [0, 2, 4]; got {recomputed}. The '
        f'sentinel [999, 999, 999] surviving means drift was not '
        f'detected, the column was not dropped, OR recompute did '
        f'not run.'
    )

    # Sidecar updated to the new closure hash.
    sigs_after = current_signatures(tmp_path)
    assert sigs_after['double_x'] == 'NEWHASH-' + sigs_before['double_x']


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


# ============ Duplicate-id defense (post-rebuild incident) ============


def test_build_measurements_rebuilds_from_scratch_on_duplicate_ids(
    tmp_path: Path,
) -> None:
    """**Defense against id-duplicate corruption**: a stale
    `measurements.parquet` with duplicate `id` rows (legacy from
    pre-Phase-1 builds or accumulated cross-sweep merges without
    dedup) would cause the runs_df → existing left-join to
    Cartesian-multiply on the next rebuild, doubling row count
    each call. Detected in the wild on
    `minatar_sync_curve_resume` (245760 rows / 120 unique ids =
    2048× duplication). Post-fix: build_measurements detects
    `existing.height != existing['id'].n_unique()` and rebuilds
    from scratch with a stderr warning.

    Probe: pre-stage a measurements.parquet with the same id
    repeated 4×, run build_measurements. Output should have one
    row per id (matching runs_df), not 4× duplication.
    """
    runs_df = _runs_df(3)
    out_path = tmp_path / MEASUREMENTS_FILENAME

    # Hand-stage a corrupt store: 12 rows, 3 unique ids.
    corrupt = pl.DataFrame({
        'id': [
            'cell-0', 'cell-0', 'cell-0', 'cell-0',
            'cell-1', 'cell-1', 'cell-1', 'cell-1',
            'cell-2', 'cell-2', 'cell-2', 'cell-2',
        ],
        'double_x': [99.0] * 12,
    })
    corrupt.write_parquet(out_path)
    assert load_measurements(tmp_path).height == 12

    build_measurements(
        tmp_path, required=['double_x'], runs_df=runs_df,
    )
    df = load_measurements(tmp_path)
    assert df.height == 3, (
        f'expected 3 rows after rebuild from corrupt store; '
        f'got {df.height}. The duplicate-id defense did not fire.'
    )
    assert df['id'].n_unique() == 3
    assert df.sort('id')['double_x'].to_list() == [0.0, 2.0, 4.0]


# ============ CI6 — row-level orphan eviction ============


def test_build_measurements_drops_orphan_rows_not_in_runs(
    tmp_path: Path,
) -> None:
    """**CORPUS_INTEGRITY.md CI6**: rows in `measurements.parquet`
    whose `id` is no longer in `runs.parquet` are orphans —
    dropped on every rebuild. Sweep extensions or partial
    reruns that removed cells from runs.parquet would
    otherwise leave stale measurement rows accumulating
    forever.

    Probe: pre-stage measurements.parquet with 3 rows for ids
    [a, b, c]. Run build_measurements with runs_df containing
    only [a, b]. Output should have 2 rows (c dropped as
    orphan), and the canonical recompute fires.
    """
    runs_df = pl.DataFrame({
        'id': ['cell-0', 'cell-1'],
        'x': [1.0, 2.0],
        'y': [10.0, 20.0],
    })
    out_path = tmp_path / MEASUREMENTS_FILENAME

    # Pre-stage: 3 rows. cell-2 will be orphan once runs_df
    # comes in with only cell-0 and cell-1.
    pre_existing = pl.DataFrame({
        'id': ['cell-0', 'cell-1', 'cell-2'],
        'double_x': [99.0, 99.0, 99.0],
    })
    pre_existing.write_parquet(out_path)

    build_measurements(
        tmp_path, required=['double_x'], runs_df=runs_df,
    )
    df = load_measurements(tmp_path)

    ids = sorted(df['id'].to_list())
    assert ids == ['cell-0', 'cell-1'], (
        f'CI6 orphan eviction failed: expected ids [cell-0, cell-1]; '
        f'got {ids}. Stale `cell-2` row should have been dropped.'
    )
    assert df.height == 2


# ============ Idempotent-skip ID validation (post-#2 roast fix) ============


def test_build_measurements_idempotent_skip_rejects_disjoint_ids(
    tmp_path: Path,
) -> None:
    """**#2 roast fix**: the idempotent fast-path checked row
    count but not ID-set membership. A caller passing a runs_df
    with disjoint IDs but identical height + pre-populated
    measurable cols got a no-op build — the persisted store
    retained the old cells' values keyed under the OLD IDs,
    silently leaking obsolete data.

    Probe: build for IDs `[a, b, c]` first; then call again with
    `runs_df` for IDs `[d, e, f]` (same height, pre-stamped
    measurable col). The store after the second call must
    contain `[d, e, f]`'s values, NOT `[a, b, c]`'s — proving
    the skip rejected the disjoint-ID case and a real rebuild
    fired.
    """
    runs_df_first = pl.DataFrame({
        'id': ['a', 'b', 'c'],
        'x': [1.0, 2.0, 3.0],
        'y': [10.0, 20.0, 30.0],
    })
    build_measurements(
        tmp_path, required=['double_x'], runs_df=runs_df_first,
    )

    # Second call: same row count, disjoint IDs, pre-stamped
    # measurable col with values that DIFFER from the framework
    # would compute for x=[100, 200, 300] (which would be 200,
    # 400, 600).
    runs_df_disjoint = pl.DataFrame({
        'id': ['d', 'e', 'f'],
        'x': [100.0, 200.0, 300.0],
        'y': [1000.0, 2000.0, 3000.0],
        'double_x': [42.0, 42.0, 42.0],
    })
    build_measurements(
        tmp_path, required=['double_x'], runs_df=runs_df_disjoint,
    )
    df = load_measurements(tmp_path)
    ids_after = sorted(df['id'].to_list())
    assert ids_after == ['d', 'e', 'f'], (
        f'expected store to be rebuilt under disjoint IDs; '
        f'got {ids_after}. If [a, b, c] persisted, the skip '
        f'fired and the runs_df was silently ignored.'
    )
    vals_after = df.sort('id')['double_x'].to_list()
    assert vals_after == [42.0, 42.0, 42.0], (
        f'expected substrate-stamped values to win on rebuild; '
        f'got {vals_after}'
    )


# ============ Drift detection: None-signature anomaly (post-#4) ============


def test_build_measurements_drops_column_when_signature_fn_returns_none(
    tmp_path: Path,
) -> None:
    """**#4 roast fix**: when `sig_fn(col)` returns None for a
    column that IS registered, drift coverage is unavailable.
    Pre-fix the column was silently kept (mask transient
    registration races as silent staleness). Post-fix the
    column is dropped — the conservative move when we can't
    prove non-drift.

    Probe: build canonically; corrupt the persisted column;
    then build with a signature fn that returns None for
    `double_x`. The corrupted values must not survive — a real
    drop+recompute fired.
    """
    runs_df = _runs_df(3)
    build_measurements(
        tmp_path, required=['double_x'], runs_df=runs_df,
    )
    # Corrupt the persisted column.
    corrupted = pl.DataFrame({
        'id': ['cell-0', 'cell-1', 'cell-2'],
        'double_x': [-99.0, -99.0, -99.0],
    })
    corrupted.write_parquet(tmp_path / MEASUREMENTS_FILENAME)

    def _none_sig(name: str) -> str | None:
        # `double_x` IS registered (the @measurable decorator at
        # module import) but our injected fn pretends signature
        # lookup is unavailable for it.
        del name
        return None

    build_measurements(
        tmp_path, required=['double_x'], runs_df=runs_df,
        measurable_signature_fn=_none_sig,
    )
    vals = load_measurements(tmp_path).sort('id')['double_x'].to_list()
    assert vals == [0.0, 2.0, 4.0], (
        f'expected drop+recompute when signature unavailable; '
        f'got {vals}. If [-99, -99, -99] survived, the column '
        f'was silently kept despite missing drift coverage.'
    )


# ============ Phase 3 collision (post-#1 roast fix) ============


def test_build_measurements_runs_df_column_wins_on_collision(
    tmp_path: Path,
) -> None:
    """**#1 roast fix**: when `runs_df` carries a measurable column
    whose name matches one in the existing
    `measurements.parquet` (Phase 3 substrate-side stamp), the
    runs_df values are authoritative and existing values are
    dropped. Pre-fix: polars' left-join produced a `<col>_right`
    suffix on the existing column that was silently orphaned at
    the `select(measurable_cols)` projection — same result by
    accident, but the semantics were implicit and depended on
    polars' join behavior staying stable. Post-fix: explicit
    `existing.drop(overlap)` before the join makes the contract
    clear and removes the silent-data-discard seam.

    Construction: substrate stamps `double_x = [99, 88, 77]`
    (which would NOT be the framework's `2 * x` recompute).
    Existing has `double_x = [0, 2, 4]` from the prior canonical
    build. After the rebuild, the persisted store has the
    substrate-stamped values, NOT the existing-store values.
    """
    runs_df = _runs_df(3)
    # Canonical first build: double_x = [0, 2, 4].
    build_measurements(
        tmp_path, required=['double_x'], runs_df=runs_df,
    )
    canonical = (
        load_measurements(tmp_path).sort('id')['double_x'].to_list()
    )
    assert canonical == [0.0, 2.0, 4.0]

    # Substrate stamps double_x with values that disagree with the
    # framework's canonical recompute.
    runs_with_stamp = runs_df.with_columns(
        pl.Series('double_x', [99.0, 88.0, 77.0]),
    )
    build_measurements(
        tmp_path, required=['double_x'], runs_df=runs_with_stamp,
    )

    df = load_measurements(tmp_path)
    vals = df.sort('id')['double_x'].to_list()
    assert vals == [99.0, 88.0, 77.0], (
        f'substrate-stamped runs_df values must win on collision; '
        f'got {vals}. If [0, 2, 4] persisted, the existing-store '
        f'won (wrong); if [recomputed_via_framework] appeared, '
        f'partial-nullity recompute fired without preserving the '
        f'substrate stamp.'
    )

    # Negative: no `double_x_right` column leaks into the parquet.
    assert 'double_x_right' not in df.columns


def test_build_measurements_partial_stamp_recomputes_only_null_cells(
    tmp_path: Path,
) -> None:
    """**#1 roast fix, partial-nullity edge**: when runs_df partially
    stamps (some cells filled, others null), substrate-stamped
    cells preserve their values; null cells are recomputed via
    the framework's `@measurable` definition. Existing-store
    values for collision columns are dropped at the join — they
    do NOT participate in the merge.

    Probe: substrate stamps cell 0 = 99, leaves cell 1 null,
    stamps cell 2 = 77. Existing has [0, 2, 4]. Expected:
    [99, 2, 77] (cell 1 recomputed via `2 * x`).
    """
    runs_df = _runs_df(3)
    build_measurements(
        tmp_path, required=['double_x'], runs_df=runs_df,
    )

    runs_with_partial_stamp = runs_df.with_columns(
        pl.Series('double_x', [99.0, None, 77.0]),
    )
    build_measurements(
        tmp_path, required=['double_x'],
        runs_df=runs_with_partial_stamp,
    )
    vals = (
        load_measurements(tmp_path).sort('id')['double_x'].to_list()
    )
    assert vals == [99.0, 2.0, 77.0], (
        f'partial-stamp expected [99, 2, 77]; got {vals}. '
        f'Cell 0/2: substrate stamp must win. Cell 1: framework '
        f'recompute via `2 * x`. The existing-store value 2.0 at '
        f'cell 1 happens to match — but that is incidental, not '
        f'proof that existing was consulted.'
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

    # First build populates double_x for all 3 cells AND writes
    # the sidecar with the canonical closure hash.
    build_measurements(
        tmp_path, required=['double_x'], runs_df=runs_df,
    )
    sigs_canonical = current_signatures(tmp_path)
    assert 'double_x' in sigs_canonical, (
        'first build should have written sidecar entry'
    )

    # Simulate "transient failure on cell 1" — overwrite the
    # parquet with cell 1 nulled out, but DO NOT touch the
    # sidecar. The sidecar still claims the canonical closure
    # hash (no drift signal); the partial-nullity branch is the
    # only path that can fix this — disambiguation post-roast-#15.
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

    # Disambiguation (post-roast-#15): the sidecar still carries
    # the same canonical hash. If drift detection had fired
    # instead of partial-nullity, the sidecar would have been
    # rewritten with the same value but the path that produced
    # the answer would be different. Pin that the sidecar
    # entry is unchanged and that drift was NOT the trigger.
    sigs_after = current_signatures(tmp_path)
    assert sigs_after['double_x'] == sigs_canonical['double_x'], (
        f'sidecar hash should be unchanged across the partial-'
        f'nullity recompute (no drift signal was emitted); '
        f'before={sigs_canonical["double_x"]}, '
        f'after={sigs_after["double_x"]}'
    )


# ============ CACHE_ADDITIVITY.md Phase 2 — check_drift =========


def test_check_drift_clean_corpus_returns_clean_report(
    tmp_path: Path,
) -> None:
    """**Phase 2**: a corpus whose measurements.parquet was just
    built against the current registry is clean — no drift,
    no missing columns. The report's `is_clean` is True."""
    runs = _runs_df(3)
    corpus = tmp_path / 'corp_a'
    corpus.mkdir()
    (corpus / 'runs.parquet').write_bytes(b'')   # touch — sentinel
    runs.write_parquet(corpus / 'runs.parquet')
    build_measurements(corpus, required=['double_x'], runs_df=runs)

    report = check_drift(tmp_path, required=['double_x'])
    assert len(report.per_corpus) == 1
    assert report.is_clean
    assert report.affected_corpus_names() == ()


def test_check_drift_detects_drifted_column(tmp_path: Path) -> None:
    """**Phase 2**: when a measurable's signature changes (substrate
    edit) but the per-corpus sidecar still carries the old hash,
    the report flags the column as drifted."""
    runs = _runs_df(3)
    corpus = tmp_path / 'corp_a'
    corpus.mkdir()
    runs.write_parquet(corpus / 'runs.parquet')
    build_measurements(corpus, required=['double_x'], runs_df=runs)

    # Inject a drifted sig fn — pretends double_x's closure
    # changed since the per-corpus sidecar was written.
    def _drifted(name: str) -> str | None:
        return 'NEWHASH-' + name if name == 'double_x' else None

    report = check_drift(
        tmp_path, required=['double_x'],
        measurable_signature_fn=_drifted,
    )
    assert not report.is_clean
    assert report.n_corpora_drifted == 1
    assert report.per_corpus[0].drifted == ('double_x',)
    assert report.per_corpus[0].missing == ()


def test_check_drift_detects_missing_column(tmp_path: Path) -> None:
    """**Phase 2**: a required measurable that was never computed
    for a corpus shows up as missing (different from drifted)."""
    runs = _runs_df(3)
    corpus = tmp_path / 'corp_a'
    corpus.mkdir()
    runs.write_parquet(corpus / 'runs.parquet')
    # Build with double_x only; later we'll require a 2nd col.
    build_measurements(corpus, required=['double_x'], runs_df=runs)

    report = check_drift(
        tmp_path, required=['double_x', 'x_plus_y'],
    )
    assert not report.is_clean
    assert report.n_corpora_missing == 1
    assert report.per_corpus[0].drifted == ()
    assert report.per_corpus[0].missing == ('x_plus_y',)


def test_check_drift_skips_in_progress_corpora(tmp_path: Path) -> None:
    """**Phase 2**: a corpus marked with `.in_progress` (sweep mid-
    flight) is silently excluded from the audit. Same convention
    as the runner's `_load_directory`."""
    runs = _runs_df(3)
    corpus = tmp_path / 'corp_a'
    corpus.mkdir()
    runs.write_parquet(corpus / 'runs.parquet')
    (corpus / '.in_progress').touch()

    report = check_drift(tmp_path, required=['double_x'])
    assert len(report.per_corpus) == 0


def test_check_drift_skips_non_corpus_subdirs(tmp_path: Path) -> None:
    """**Phase 2**: subdirs without `runs.parquet` aren't corpora
    and don't appear in the report."""
    (tmp_path / 'cache').mkdir()  # no runs.parquet — not a corpus
    (tmp_path / 'cache' / 'arbitrary.json').write_text('{}')

    runs = _runs_df(3)
    corpus = tmp_path / 'corp_a'
    corpus.mkdir()
    runs.write_parquet(corpus / 'runs.parquet')
    build_measurements(corpus, required=['double_x'], runs_df=runs)

    report = check_drift(tmp_path, required=['double_x'])
    assert len(report.per_corpus) == 1
    assert report.per_corpus[0].corpus_dir.name == 'corp_a'
