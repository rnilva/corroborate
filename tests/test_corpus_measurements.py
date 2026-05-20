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
    SIDECAR_FILENAME,
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


# ============ recompute_corpus_measurables ============
#
# Closes the schema-gap an operator hits when a new @measurable
# is registered after a corpus was ingested. The existing
# `_load_one_corpus` path handles this via cloud restore; the
# recompute primitive is the opt-in LOCAL counterpart (no cloud
# round-trip when the trace data already lives on disk).


from corroborate.corpus.measurements import (   # noqa: E402
    RecomputeResult,
    recompute_corpus_measurables,
)


def _runs_df_with_traces(n: int = 5) -> pl.DataFrame:
    """Substrate-shaped runs frame: lineage `id` + the leaf record
    keys that `double_x` (reads x) and `x_plus_y` (reads x, y)
    consume. Models a corpus whose runs.parquet carries both
    measurable inputs inline (no separate traces.parquet needed
    for these measurables)."""
    return pl.DataFrame({
        'id': [f'cell-{i}' for i in range(n)],
        'x': [float(i) for i in range(n)],
        'y': [float(i * 10) for i in range(n)],
    })


def _write_corpus(corpus_dir: Path, runs: pl.DataFrame) -> None:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    runs.write_parquet(corpus_dir / 'runs.parquet')


def test_recompute_fills_in_missing_measurable(tmp_path: Path) -> None:
    """**The canonical case the prompt names**: a new @measurable
    was registered after the corpus's `measurements.parquet` was
    built, so the sidecar is missing the column. `recompute_corpus_
    measurables` detects the gap, satisfies it from local
    runs.parquet, and writes the column.

    Probe pre-state: build a corpus with only `double_x`, leaving
    `x_plus_y` out of the sidecar. Recompute with both measurables
    required; check the column is now populated AND its closure
    hash is recorded in the sidecar."""
    corpus = tmp_path / 'corp'
    runs = _runs_df_with_traces(4)
    _write_corpus(corpus, runs)
    build_measurements(corpus, required=['double_x'], runs_df=runs)

    sidecar_before = current_signatures(corpus)
    assert 'x_plus_y' not in sidecar_before

    result = recompute_corpus_measurables(
        corpus, required=['double_x', 'x_plus_y'],
    )
    assert isinstance(result, RecomputeResult)
    assert result.recomputed == ('x_plus_y',), (
        f'expected x_plus_y to be recomputed; got {result.recomputed}'
    )
    assert 'double_x' in result.already_current
    assert result.unsatisfiable == ()
    assert result.unregistered == ()
    assert not result.is_clean   # something was recomputed

    # The column is in the per-corpus store with the canonical
    # framework values (x + y = i + 10·i = 11·i).
    df = load_measurements(corpus)
    assert 'x_plus_y' in df.columns
    canonical = sorted(zip(
        df['id'].to_list(),
        df['x_plus_y'].to_list(),
    ))
    assert canonical == [
        ('cell-0', 0.0),
        ('cell-1', 11.0),
        ('cell-2', 22.0),
        ('cell-3', 33.0),
    ]
    # Sidecar updated with the new measurable's hash.
    sidecar_after = current_signatures(corpus)
    assert 'x_plus_y' in sidecar_after
    assert 'double_x' in sidecar_after   # preserved across recompute


def test_recompute_is_idempotent_no_op_on_current_corpus(
    tmp_path: Path,
) -> None:
    """When every required measurable's closure hash already
    matches the registry, the recompute is a no-op — no rewrite,
    `recomputed` is empty, `already_current` lists everything,
    and `is_clean` returns True.

    Pin via parquet mtime (same shape as
    `test_build_measurements_is_idempotent`): the file mustn't be
    rewritten on a no-op recompute."""
    corpus = tmp_path / 'corp'
    runs = _runs_df_with_traces(3)
    _write_corpus(corpus, runs)
    build_measurements(
        corpus, required=['double_x', 'x_plus_y'], runs_df=runs,
    )
    measurements_path = corpus / MEASUREMENTS_FILENAME
    mtime_before = measurements_path.stat().st_mtime_ns

    import time
    time.sleep(0.01)

    result = recompute_corpus_measurables(
        corpus, required=['double_x', 'x_plus_y'],
    )
    assert result.recomputed == ()
    assert set(result.already_current) == {'double_x', 'x_plus_y'}
    assert result.unsatisfiable == ()
    assert result.unregistered == ()
    assert result.is_clean

    mtime_after = measurements_path.stat().st_mtime_ns
    assert mtime_after == mtime_before, (
        f'no-op recompute rewrote the parquet (mtime '
        f'{mtime_before} → {mtime_after}); should have been '
        f'short-circuited.'
    )


def test_recompute_classifies_unsatisfiable_measurable(
    tmp_path: Path,
) -> None:
    """When a required measurable's transitive reads aren't in
    runs.parquet AND no local traces.parquet carries them, the
    measurable is classified `unsatisfiable` — NOT recomputed.
    Overwriting a finite per-corpus value with a fresh NaN
    (which would happen if the resolver ran without the input)
    would be silent data loss. The contract is strict: the
    operator gets back a list naming what was skipped.

    Probe: register a synthetic measurable `from_trace_col` that
    reads `unavailable_trace_col`. Build a corpus that has neither
    runs.parquet nor traces.parquet columns matching. Recompute
    classifies the new measurable as unsatisfiable.
    """
    @measurable(name='from_trace_col', reads=('unavailable_trace_col',))
    def _from_trace_col(record: Mapping[str, object]) -> float:
        v = record.get('unavailable_trace_col')
        if not isinstance(v, (int, float)):
            raise TypeError('unavailable_trace_col missing')
        return float(v)
    # Reference the local def so the linter doesn't flag unused.
    del _from_trace_col

    corpus = tmp_path / 'corp'
    runs = _runs_df_with_traces(3)
    _write_corpus(corpus, runs)
    build_measurements(corpus, required=['double_x'], runs_df=runs)

    result = recompute_corpus_measurables(
        corpus, required=['double_x', 'from_trace_col'],
    )
    assert result.unsatisfiable == ('from_trace_col',)
    assert result.recomputed == ()
    assert 'double_x' in result.already_current
    assert not result.is_clean  # unsatisfiable counts as non-clean

    # The non-recomputed column does NOT appear in the store —
    # avoiding any "framework computed this and got NaN"
    # ambiguity at the closure-hash layer.
    df = load_measurements(corpus)
    assert 'from_trace_col' not in df.columns


def test_recompute_classifies_unregistered_measurable(
    tmp_path: Path,
) -> None:
    """An unregistered name in `required` is a caller-side bug —
    can never be computed. `recompute_corpus_measurables` reports
    it under `unregistered` so the CLI can surface the typo
    distinctly from "missing trace data" (an operational fault)."""
    corpus = tmp_path / 'corp'
    runs = _runs_df_with_traces(3)
    _write_corpus(corpus, runs)
    build_measurements(corpus, required=['double_x'], runs_df=runs)

    result = recompute_corpus_measurables(
        corpus,
        required=['double_x', 'no_such_measurable_xyz'],
    )
    assert result.unregistered == ('no_such_measurable_xyz',)
    assert result.recomputed == ()
    assert not result.is_clean


def test_recompute_no_runs_parquet_returns_empty_unregistered(
    tmp_path: Path,
) -> None:
    """A corpus dir without `runs.parquet` is not a corpus — the
    recompute returns a `RecomputeResult` with `required` mirrored
    into `unregistered` (defensive — semantically "nothing usable
    here"). The caller's CLI logs the SKIPPED state per directory
    and continues."""
    empty = tmp_path / 'no_runs'
    empty.mkdir()
    result = recompute_corpus_measurables(
        empty, required=['double_x', 'x_plus_y'],
    )
    assert result.recomputed == ()
    assert result.already_current == ()
    assert result.unsatisfiable == ()
    assert set(result.unregistered) == {'double_x', 'x_plus_y'}


def test_recompute_reads_trace_columns_when_local_traces_present(
    tmp_path: Path,
) -> None:
    """When the new measurable reads a column that lives in
    `traces.parquet` (NOT `runs.parquet`), the recompute joins
    the trace col before evaluating. Closes the production
    failure mode: a per-step trace col like `online_max_q_per_step`
    is what the canonical RL measurables read, and the recompute
    must handle that case.

    Probe: register `trace_double` that reads `step_value` (a
    per-step trace col). Build a corpus with `runs.parquet`
    (no step_value) AND `traces.parquet` (carries step_value).
    Recompute fills in the column."""
    @measurable(name='trace_double', reads=('step_value',))
    def _trace_double(record: Mapping[str, object]) -> float:
        v = record.get('step_value')
        if not isinstance(v, (int, float)):
            raise TypeError('step_value missing')
        return 2.0 * float(v)
    del _trace_double   # silence linter; the decorator registers

    corpus = tmp_path / 'corp'
    runs = pl.DataFrame({
        'id': ['cell-0', 'cell-1', 'cell-2'],
        'x': [1.0, 2.0, 3.0],
    })
    traces = pl.DataFrame({
        'id': ['cell-0', 'cell-1', 'cell-2'],
        'step_value': [10.0, 20.0, 30.0],
    })
    _write_corpus(corpus, runs)
    traces.write_parquet(corpus / 'traces.parquet')

    result = recompute_corpus_measurables(
        corpus, required=['trace_double'],
    )
    assert result.recomputed == ('trace_double',)
    assert result.unsatisfiable == ()
    df = load_measurements(corpus)
    assert 'trace_double' in df.columns
    vals = sorted(zip(df['id'].to_list(), df['trace_double'].to_list()))
    assert vals == [
        ('cell-0', 20.0),
        ('cell-1', 40.0),
        ('cell-2', 60.0),
    ]


def test_recompute_does_not_clobber_already_current_columns(
    tmp_path: Path,
) -> None:
    """When the recompute walks the gap (missing or drifted), it
    must preserve the already-current columns in the persisted
    store. This guards against "passing only the gap to
    build_measurements drops the rest as orphans" — the function
    passes the FULL `required` list so build's drift/orphan logic
    keeps the current cols intact.

    Probe: build with `double_x` and `x_plus_y` populated. Then
    recompute with the same required list — the values for both
    must survive the call unchanged."""
    corpus = tmp_path / 'corp'
    runs = _runs_df_with_traces(3)
    _write_corpus(corpus, runs)
    build_measurements(
        corpus, required=['double_x', 'x_plus_y'], runs_df=runs,
    )
    before = load_measurements(corpus).sort('id')

    # Now corrupt double_x's stored signature so it shows up as
    # drifted, forcing a recompute. x_plus_y stays current.
    sigs = current_signatures(corpus)
    sigs['double_x'] = 'CORRUPTED-HASH'
    import json
    (corpus / SIDECAR_FILENAME).write_text(json.dumps(sigs))

    result = recompute_corpus_measurables(
        corpus, required=['double_x', 'x_plus_y'],
    )
    assert 'double_x' in result.recomputed
    assert 'x_plus_y' in result.already_current

    after = load_measurements(corpus).sort('id')
    # Both columns survive with their canonical values.
    assert after['double_x'].to_list() == before['double_x'].to_list()
    assert after['x_plus_y'].to_list() == before['x_plus_y'].to_list()


def test_recompute_idempotent_on_drifted_column(tmp_path: Path) -> None:
    """A drifted (sidecar hash != live registry hash) column
    counts as gap, gets recomputed. The closure-hash contract
    means corrupting the stored hash forces a rebuild regardless
    of whether the column values are sound."""
    corpus = tmp_path / 'corp'
    runs = _runs_df_with_traces(3)
    _write_corpus(corpus, runs)
    build_measurements(corpus, required=['double_x'], runs_df=runs)

    sigs = current_signatures(corpus)
    sigs['double_x'] = 'INTENTIONAL-DRIFT-' + sigs['double_x']
    import json
    (corpus / SIDECAR_FILENAME).write_text(json.dumps(sigs))

    result = recompute_corpus_measurables(
        corpus, required=['double_x'],
    )
    assert result.recomputed == ('double_x',), (
        f'drift should trigger recompute; got {result.recomputed}'
    )
    # Sidecar now matches the live hash again.
    new_sigs = current_signatures(corpus)
    assert 'INTENTIONAL-DRIFT-' not in new_sigs['double_x']


# ============ CLI wiring: _recompute_ingest_targets ============
#
# Exercises the dispatch helper that resolves the `--ingest`
# argument shape (named list / directory walk / single file) into
# concrete per-corpus recompute calls.


@measurable(reads=('x', 'y'))
def _xy_minus(record: Mapping[str, object]) -> float:
    """Synthetic measurable for the CLI-integration tests below —
    distinct from the `x_plus_y` defined at the top of the file
    so the same test file can register both safely (no name
    collision in the @measurable registry)."""
    x = record.get('x')
    y = record.get('y')
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        raise TypeError('x or y missing')
    return float(x) - float(y)


# Bridge stand-in for the CLI tests: declares a `_xy_minus`
# dependency via its holds_when signature. `measurable_names_for_
# bridges` walks the analysis fixture-param names AND the
# `transitive_reads` of registered measurables on the analysis
# return; the simplest way to surface `_xy_minus` as required is
# via a holds_when fixture parameter named `_xy_minus`.
from corroborate.bridge.analysis import analysis as _cli_analysis   # noqa: E402
from corroborate.bridge.bridge import claim_bridge as _claim_bridge   # noqa: E402
from corroborate.bridge.verdict import Verdict as _Verdict   # noqa: E402
from corroborate.graph.causal import Direction as _Direction, Tier as _Tier   # noqa: E402


@_cli_analysis
def _xy_minus_analysis(
    cells: list[Mapping[str, object]],
    _xy_minus: object,   # noqa: ARG001
) -> int:
    return len(cells)


@_claim_bridge(
    source='x', target='_xy_minus',
    direction=_Direction.DIRECT, tier=_Tier.ASSOCIATIONAL,
    pair_by=(),
)
def _bridge_for_cli_recompute(
    _xy_minus_analysis: int,   # noqa: ARG001
) -> _Verdict:
    return _Verdict.HELD


def test_cli_recompute_targets_named_list_calls_recompute(
    tmp_path: Path,
) -> None:
    """`_recompute_ingest_targets` walks the named-corpus list
    and triggers `recompute_corpus_measurables` per corpus. Pin
    via the per-corpus measurements.parquet state before/after
    the call.

    The bridge `_bridge_for_cli_recompute` declares `_xy_minus`
    as its analysis source — the CLI helper walks
    `measurable_names_for_bridges(bridges)` which surfaces
    `_xy_minus`, and the recompute fills it in for the named
    corpus."""
    import types
    import sys
    from corroborate.cli.hypothesis import (
        _recompute_ingest_targets,  # pyright: ignore[reportPrivateUsage]
    )

    corpus = tmp_path / 'corp_a'
    runs = _runs_df_with_traces(3)
    _write_corpus(corpus, runs)
    # Pre-state: build only `double_x`; `_xy_minus` is NOT in
    # the sidecar.
    build_measurements(corpus, required=['double_x'], runs_df=runs)
    sigs_before = current_signatures(corpus)
    assert '_xy_minus' not in sigs_before

    # Stash a throwaway module so the CLI's
    # `importlib.import_module(module_name)` resolves. The CLI
    # only reads the optional `REQUIRED_MEASURABLES` escape
    # hatch via `getattr(default=())`, so the module shape can
    # be minimal — no `BRIDGES` attribute needed (bridges flow
    # in as a function argument).
    h_mod = types.ModuleType('test_recompute_named_inline')
    sys.modules['test_recompute_named_inline'] = h_mod

    _recompute_ingest_targets(
        module_name='test_recompute_named_inline',
        bridges=(_bridge_for_cli_recompute,),
        data=[corpus],
    )
    # `_xy_minus` now appears in the per-corpus store.
    sigs_after = current_signatures(corpus)
    assert '_xy_minus' in sigs_after, (
        f'expected CLI helper to fill `_xy_minus`; sidecar after: '
        f'{sorted(sigs_after)}'
    )
    df = load_measurements(corpus)
    assert '_xy_minus' in df.columns
    # Values are the canonical `x - y` framework computation.
    vals = sorted(zip(df['id'].to_list(), df['_xy_minus'].to_list()))
    assert vals == [
        ('cell-0', 0.0),
        ('cell-1', -9.0),
        ('cell-2', -18.0),
    ]


def test_cli_recompute_targets_no_data_skips_with_log(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`--recompute-measurables` with no `--ingest` mode set is
    a no-op; the function logs the skip but doesn't raise."""
    from corroborate.cli.hypothesis import (
        _recompute_ingest_targets,  # pyright: ignore[reportPrivateUsage]
    )

    _recompute_ingest_targets(
        module_name='unused', bridges=(), data=None,
    )
    captured = capsys.readouterr()
    assert 'no --ingest mode set' in captured.err


def test_cli_recompute_targets_ingest_file_skips_with_log(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`--recompute-measurables` paired with `--ingest-file
    <path.parquet>` doesn't have a per-corpus dir to rebuild —
    the function logs the skip rather than crashing on the
    file-vs-dir mismatch."""
    from corroborate.cli.hypothesis import (
        _recompute_ingest_targets,  # pyright: ignore[reportPrivateUsage]
    )

    parquet = tmp_path / 'just_a_file.parquet'
    pl.DataFrame({'id': ['a']}).write_parquet(parquet)
    _recompute_ingest_targets(
        module_name='unused', bridges=(), data=parquet,
    )
    captured = capsys.readouterr()
    assert 'ingest-file' in captured.err
    assert 'skipping recompute' in captured.err


def test_cli_recompute_targets_directory_walks_one_level(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`--ingest-all <root>`: the helper walks `root.iterdir()`
    and recomputes each corpus subdir that has a `runs.parquet`.
    Non-corpus subdirs (no runs.parquet) are silently skipped
    via the iteration filter."""
    import types
    import sys
    from corroborate.cli.hypothesis import (
        _recompute_ingest_targets,  # pyright: ignore[reportPrivateUsage]
    )

    root = tmp_path / 'root'
    root.mkdir()
    # Two real corpora at the first level.
    for name in ('corp_a', 'corp_b'):
        runs = _runs_df_with_traces(2)
        _write_corpus(root / name, runs)
        build_measurements(
            root / name, required=['double_x'], runs_df=runs,
        )
    # One non-corpus subdir — silently skipped.
    (root / 'not_a_corpus').mkdir()
    (root / 'not_a_corpus' / 'unrelated.json').write_text('{}')

    h_mod = types.ModuleType('test_recompute_walk_inline')
    sys.modules['test_recompute_walk_inline'] = h_mod

    _recompute_ingest_targets(
        module_name='test_recompute_walk_inline',
        bridges=(_bridge_for_cli_recompute,),
        data=root,
    )
    captured = capsys.readouterr()
    # Both corpora walked; the not_a_corpus subdir didn't error.
    assert 'walking 2 corpus dir(s)' in captured.err
    # And the gap (`_xy_minus`) was filled on each.
    for name in ('corp_a', 'corp_b'):
        assert '_xy_minus' in current_signatures(root / name)
