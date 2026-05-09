"""Tests for `run_intervention` — paired-sweep-per-intervention
runner.

The sweep primitive composes treatment + baseline claims from a
DoEffect's Intervention tuples + a substrate-supplied `base`,
iterates the Cartesian product of the exogenous grid, and
dispatches both arms per grid point. Pairing is intrinsic.

These tests use a synthetic substrate: a tiny `base` callable
plus a stub Runner that produces deterministic SweepCellResults.
The framework's job (claim composition, arm_key derivation,
parquet persistence) gets exercised end-to-end without an RL
substrate."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from corroborate.bridge.bridge import Bridge
from corroborate.bridge.verdict import Verdict
from corroborate.core.claim import claim
from corroborate.core.hypothesis import Hypothesis
from corroborate.core.intervention import DoEffect, Intervention
from corroborate.corpus.persistence import read_runrows
from corroborate.corpus.schema import RunRow, TraceRow
from corroborate.measurables import Measurable
from corroborate.runner.sweep import (
    MergeIntegrityError,
    SweepCellResult,
    empty_graph,
    run_intervention,
)


@claim
def _treatment_op(record: Mapping[str, object]) -> Mapping[str, object]:
    return record


@claim
def _base_theory(
    record: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    """Stand-in for the substrate's theory. Returns a record-
    shaped mapping; the synthetic test just exercises the
    framework's claim composition + arm_key plumbing."""
    return record or {}


_BRIDGES: tuple[Bridge, ...] = ()


def _stub_measurable() -> Measurable[Mapping[str, object], object]:
    def _fn(record: Mapping[str, object]) -> object:
        del record
        return 1.0
    return Measurable(fn=_fn, name='stub_outcome', reads=())


_MEASURABLES: tuple[Measurable[Mapping[str, object], object], ...] = (
    _stub_measurable(),
)


@dataclass(frozen=True)
class _StubHypothesis:
    """Class-as-Hypothesis: satisfies the Protocol via ClassVars."""
    INTERVENTION: ClassVar[DoEffect] = DoEffect(
        treatment=(
            Intervention(slot_path='op', replacement=_treatment_op),
        ),
        baseline=(),
    )
    BRIDGES: ClassVar[tuple[Bridge, ...]] = _BRIDGES
    MEASURABLES: ClassVar[
        tuple[Measurable[Mapping[str, object], object], ...]
    ] = _MEASURABLES


def test_class_satisfies_hypothesis_protocol() -> None:
    """Class-as-Hypothesis: a frozen dataclass with the required
    ClassVar fields satisfies the runtime_checkable Protocol."""
    assert isinstance(_StubHypothesis, Hypothesis)


def _make_run(arm_key: str, **measurements: object) -> RunRow:
    import uuid
    leaf_measurements: dict[str, str | int | float | bool] = {}
    for k, v in measurements.items():
        if isinstance(v, (str, int, float, bool)):
            leaf_measurements[k] = v
    return RunRow(
        id=str(uuid.uuid4()), parent_id=None, cycle_id=None,
        timestamp='2026-05-04T00:00:00Z',
        verdict=Verdict.HELD,
        arm_key=arm_key,
        measurements=leaf_measurements,
    )


def _stub_runner(
    claim: Callable[..., Mapping[str, object]],
    arm_key: str,
    measurables: tuple[Measurable[Mapping[str, object], object], ...],
    grid_point: Mapping[str, object],
) -> SweepCellResult:
    """Trivial runner: emits one RunRow per call carrying the
    arm_key + grid_point measurements."""
    del claim, measurables
    run = _make_run(arm_key, **dict(grid_point))
    return SweepCellResult(
        runs=(run,),
        traces=(TraceRow(
            id=run.id, cycle_id=None,
            timestamp='2026-05-04T00:00:00Z', leaves={},
        ),),
        graph=empty_graph(),
    )


def test_run_intervention_pairs_arms_at_each_grid_point(
    tmp_path: Path,
) -> None:
    """A 2-grid sweep produces 4 cells: 2 grid points × 2 arms
    (treatment, baseline). Each grid point has a paired
    (treatment, baseline) cell with matching grid keys."""
    runs_path, _ = run_intervention(
        _StubHypothesis.INTERVENTION,
        base=_base_theory,
        measurables=_StubHypothesis.MEASURABLES,
        grid_points=[{'replicate': 0}, {'replicate': 1}],
        runner=_stub_runner,
        out_dir=tmp_path,
    )
    rows = read_runrows(runs_path)
    assert len(rows) == 4
    arm_keys = sorted({r.arm_key for r in rows})
    treatment_key = _StubHypothesis.INTERVENTION.treatment_arm_key()
    baseline_key = _StubHypothesis.INTERVENTION.baseline_arm_key()
    assert arm_keys == sorted([treatment_key, baseline_key])
    # Pairing intrinsic: each replicate has both arms.
    for rep in (0, 1):
        rep_rows = [r for r in rows if r.measurements.get('replicate') == rep]
        assert len(rep_rows) == 2
        assert {r.arm_key for r in rep_rows} == {treatment_key, baseline_key}


def test_run_intervention_arm_keys_from_canonical_str(
    tmp_path: Path,
) -> None:
    """`arm_key` on emitted cells comes from
    `combined_arm_key(intervention_tuple)` — the canonical
    fingerprint of the typed Intervention. Treatment carries the
    structural delta's fingerprint; baseline (empty tuple) is
    `'baseline'`."""
    runs_path, _ = run_intervention(
        _StubHypothesis.INTERVENTION,
        base=_base_theory,
        measurables=_StubHypothesis.MEASURABLES,
        grid_points=[{}],
        runner=_stub_runner,
        out_dir=tmp_path,
    )
    rows = read_runrows(runs_path)
    assert len(rows) == 2
    arm_keys = sorted({r.arm_key for r in rows})
    assert arm_keys == sorted([
        'op=Claim:_treatment_op',
        'baseline',
    ])


def test_run_intervention_single_empty_grid_point_runs_once_per_arm(
    tmp_path: Path,
) -> None:
    """`grid_points=[{}]` → one cell per arm = 2 cells (treatment +
    baseline)."""
    runs_path, _ = run_intervention(
        _StubHypothesis.INTERVENTION,
        base=_base_theory,
        measurables=_StubHypothesis.MEASURABLES,
        grid_points=[{}],
        runner=_stub_runner,
        out_dir=tmp_path,
    )
    rows = read_runrows(runs_path)
    assert len(rows) == 2


# ============ I3: manifest-driven merge across multiple calls ============


def test_two_run_intervention_calls_share_out_dir_merge_includes_all_cells(
    tmp_path: Path,
) -> None:
    """**Invariant I3** (SWEEP_PERSISTENCY.md): when two
    `run_intervention` calls target the same `out_dir` with the
    same `archive_remote`, the final `<out_dir>/runs.parquet`
    must contain the UNION of cells from both calls.

    Pre-fix bug: `run_intervention`'s merge step used
    `archived_runs_uris` populated only from THIS call's iteration.
    The second call clobbered the first call's merged output with
    a SUBSET (only its own cells). This was the
    `minatar_sync_curve_resume` data-loss incident.

    Post-fix: the merge reads URIs from the manifest, which
    accumulates entries across all calls into the same `out_dir`.

    Construction: archive_remote is a file:// URI so fsspec uploads
    locally without S3 credentials. Two grid_points per call, two
    arms per grid_point → 4 cells per call → 8 cells in the final
    union.
    """
    archive_remote = (tmp_path / 'remote').as_uri()

    # Tag includes replicate so each call's cells have distinct
    # relpaths (otherwise the resume-skip path triggers on call 2,
    # masking the I3 bug). The I1 invariant is what protects against
    # this in production via per-arm-config namespacing; for this
    # I3-specific test we make the disambiguation explicit.
    def _replicate_tag(arm_key: str, gp: Mapping[str, object]) -> str:
        rep = gp.get('replicate', '?')
        return f'{arm_key}__rep{rep}'

    # Call 1: grid_points = [replicate=0, replicate=1]
    run_intervention(
        _StubHypothesis.INTERVENTION,
        base=_base_theory,
        measurables=_StubHypothesis.MEASURABLES,
        grid_points=[{'replicate': 0}, {'replicate': 1}],
        runner=_stub_runner,
        out_dir=tmp_path,
        archive_remote=archive_remote,
        arm_tag=_replicate_tag,
    )

    # Call 2: grid_points = [replicate=2, replicate=3] (disjoint)
    runs_path, _ = run_intervention(
        _StubHypothesis.INTERVENTION,
        base=_base_theory,
        measurables=_StubHypothesis.MEASURABLES,
        grid_points=[{'replicate': 2}, {'replicate': 3}],
        runner=_stub_runner,
        out_dir=tmp_path,
        archive_remote=archive_remote,
        arm_tag=_replicate_tag,
    )

    rows = read_runrows(runs_path)
    # 4 cells per call × 2 calls = 8 cells total. Pre-fix: 4
    # (only the second call's cells; the first call's merged
    # output was clobbered).
    assert len(rows) == 8, (
        f'expected 8 cells across 2 calls (4 each), got {len(rows)}. '
        f'Pre-fix bug: per-call merge clobbered the prior call s '
        f'merged file. Post-fix: manifest-driven merge reads from '
        f'the accumulated manifest.'
    )

    # All four replicates present, both arms each.
    replicates = sorted(
        {r.measurements.get('replicate') for r in rows}
    )
    assert replicates == [0, 1, 2, 3]
    treatment_key = _StubHypothesis.INTERVENTION.treatment_arm_key()
    baseline_key = _StubHypothesis.INTERVENTION.baseline_arm_key()
    for rep in replicates:
        rep_rows = [
            r for r in rows
            if r.measurements.get('replicate') == rep
        ]
        assert len(rep_rows) == 2, (
            f'replicate {rep}: expected 2 paired rows, got '
            f'{len(rep_rows)}'
        )
        assert {r.arm_key for r in rep_rows} == {
            treatment_key, baseline_key,
        }


# ============ Resume robustness: stale 0-byte tmp parquets ============


def test_run_intervention_reruns_cell_when_tmp_parquet_is_zero_bytes(
    tmp_path: Path,
) -> None:
    """When a previous sweep crashed mid-write of a cell's parquet
    (e.g. SIGKILL during stream_concat), the leftover 0-byte file
    poisons subsequent merges with a polars
    `File out of specification: must contain a header and footer
    with at least 12 bytes` ComputeError.

    Pre-fix: the resume-skip check was `runs_path.exists() and
    traces_path.exists()` — `Path.exists()` returns True on a
    0-byte file, so the runner happily skipped the cell, then the
    merge step tried to read the corrupt parquet and crashed.

    Post-fix: the skip-cached check requires `stat().st_size > 0`
    on both parquets. A partial leftover triggers cleanup +
    re-run via the new `elif runs_path.exists() or
    traces_path.exists():` branch.

    Reproduces the metamaze_g099_1M_postfix incident
    (2026-05-09): cell002 traces.parquet was 0 bytes, finalize
    sweep crashed in merge.
    """
    # Run a 2-grid sweep to produce 4 cells normally.
    run_intervention(
        _StubHypothesis.INTERVENTION,
        base=_base_theory,
        measurables=_StubHypothesis.MEASURABLES,
        grid_points=[{'replicate': 0}, {'replicate': 1}],
        runner=_stub_runner,
        out_dir=tmp_path,
    )

    # Simulate a crashed mid-write: truncate one cell's traces to 0.
    # The framework writes per-cell parquets under
    # `<out_dir>/<arm_tag>/tmp/cell<N>__<tag>__traces.parquet`.
    tmp_files = list(tmp_path.rglob('cell*__traces.parquet'))
    assert tmp_files, 'expected per-cell tmp traces.parquet to exist'
    poisoned = tmp_files[0]
    poisoned.write_bytes(b'')  # 0-byte corruption
    assert poisoned.stat().st_size == 0

    # Re-run the same sweep. The skip-cached path should detect the
    # poisoned parquet, clean it, and re-run that cell — NOT skip
    # and choke later in the merge.
    runs_path, _ = run_intervention(
        _StubHypothesis.INTERVENTION,
        base=_base_theory,
        measurables=_StubHypothesis.MEASURABLES,
        grid_points=[{'replicate': 0}, {'replicate': 1}],
        runner=_stub_runner,
        out_dir=tmp_path,
    )

    # Merge succeeded → final runs.parquet exists and contains all
    # 4 cells.
    rows = read_runrows(runs_path)
    assert len(rows) == 4
    # And the previously-poisoned tmp file is now non-zero (re-
    # written by the cell runner).
    assert poisoned.stat().st_size > 0


# ============ Merge integrity cross-check ============


def test_merge_integrity_check_raises_when_manifest_filter_drops_shards(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """**Cross-check for I3 + force=True risk**: if
    `archived_shard_uris` returns a SUBSET of the manifest's
    cell shards (e.g., a regression in the prefix/suffix filter),
    the merged parquet has fewer rows than the manifest's
    `row_ids` would predict. The integrity check must raise
    BEFORE the corrupt merged output is uploaded.

    Simulate the bug by monkeypatching `archived_shard_uris` to
    return only HALF the URIs. The merge produces a smaller-
    than-expected parquet; `_check_merge_integrity` catches the
    discrepancy via the manifest's `row_ids` provenance.

    Pre-fix: silent corruption uploaded via force=True. Post-fix:
    MergeIntegrityError before upload.
    """
    archive_remote = (tmp_path / 'remote').as_uri()

    def _replicate_tag(arm_key: str, gp: Mapping[str, object]) -> str:
        rep = gp.get('replicate', '?')
        return f'{arm_key}__rep{rep}'

    # Run a 4-cell sweep to populate the manifest.
    run_intervention(
        _StubHypothesis.INTERVENTION,
        base=_base_theory,
        measurables=_StubHypothesis.MEASURABLES,
        grid_points=[{'replicate': 0}, {'replicate': 1}],
        runner=_stub_runner,
        out_dir=tmp_path,
        archive_remote=archive_remote,
        arm_tag=_replicate_tag,
    )

    # Now simulate a `archived_shard_uris` regression: monkeypatch
    # the cloud-module function to return only the FIRST half of
    # URIs. The next merge will produce a parquet with HALF the
    # expected rows.
    import corroborate.corpus.cloud as cloud_mod
    real_archived_shard_uris = cloud_mod.archived_shard_uris

    def buggy_archived_shard_uris(
        out_dir, *, prefix, suffix,
    ):
        full = real_archived_shard_uris(
            out_dir, prefix=prefix, suffix=suffix,
        )
        # Drop half — simulating a manifest filter regression.
        return full[: len(full) // 2]

    monkeypatch.setattr(
        cloud_mod,
        'archived_shard_uris',
        buggy_archived_shard_uris,
    )

    # Re-run with two NEW grid_points (extends the manifest).
    # The integrity check should fire on the merge step.
    import pytest
    with pytest.raises(MergeIntegrityError) as exc_info:
        run_intervention(
            _StubHypothesis.INTERVENTION,
            base=_base_theory,
            measurables=_StubHypothesis.MEASURABLES,
            grid_points=[{'replicate': 2}, {'replicate': 3}],
            runner=_stub_runner,
            out_dir=tmp_path,
            archive_remote=archive_remote,
            arm_tag=_replicate_tag,
        )
    err = exc_info.value
    assert err.actual_rows < err.expected_rows, (
        f'integrity check should detect FEWER rows than expected; '
        f'got actual={err.actual_rows}, expected={err.expected_rows}'
    )
    assert err.kind in ('runs', 'traces')
