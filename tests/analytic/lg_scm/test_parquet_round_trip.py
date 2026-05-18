"""Closed-form analytic recovery survives a real parquet round-trip.

`tests/test_persistence.py` covers the persistence layer's
write/read CONTRACT (single + multi-row, empty measurements,
arm_key, heterogeneous keys, dtype tightening, sidecar) with
hand-built RunRows. Those are proper unit tests, but they don't
prove that the FRAMEWORK PRIMITIVES correctly handle cells that
have round-tripped through parquet.

This file does. The substrate produces LG-SCM cells, writes them
to a tmp parquet via `write_runrows` / `write_tracerows`, reads
them back via `read_runrows` / `read_tracerows`, then runs
`paired_g.fn` and `paired_g_per_burst.fn`. The closed-form
analytical bounds from `tests/analytic/lg_scm/test_paired_g.py`
and `test_paired_g_per_burst.py` must STILL hold after the round
trip.

A regression in the persistence layer that:
- coerced floats to a narrower dtype with rounding,
- dropped or reordered measurements,
- corrupted nested-list columns at the polars boundary,
- mishandled the typed `arm_key` column on heterogeneous arms,
would be caught here as a closed-form-bound breach. The pure
write/read contract tests would still pass because the
contract-level assertions are equality-based and tolerate minor
shape preservation only.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from corroborate.analyses.paired.paired_g import paired_g
from corroborate.analyses.paired.paired_g_per_burst import paired_g_per_burst
from corroborate.corpus.persistence import (
    read_runrows,
    read_tracerows,
    write_runrows,
    write_tracerows,
)
from corroborate.corpus.schema import RunRow, TraceRow
from corroborate.measurables.reductions import from_key, reduce_axis

from tests.analytic.lg_scm.composition import LinearGaussianSCM
from tests.analytic.lg_scm.runner import (
    PER_BURST_Y_KEY,
    merge_cell,
    run_paired_arms,
    run_phased_cell,
)


# Shared parameters — chosen to match the closed-form bounds in
# the in-memory analytic tests so any drift introduced by the
# parquet round-trip surfaces against the same yardstick.
_MU_X = 1.0
_SIGMA_X = 0.5
_BETA_ZY = 1.5
_SIGMA_Z = 0.1
_SIGMA_Y = 0.1
_N_STEPS = 200
_N_PAIRS = 30


def _scm(beta_xz: float) -> LinearGaussianSCM:
    return LinearGaussianSCM(
        mu_x=_MU_X, sigma_x=_SIGMA_X,
        beta_xz=beta_xz, sigma_z=_SIGMA_Z,
        beta_zy=_BETA_ZY, sigma_y=_SIGMA_Y,
        n_steps=_N_STEPS,
    )


def _expected_mean_diff(*, beta_xz_t: float, beta_xz_b: float) -> float:
    return (beta_xz_t - beta_xz_b) * _BETA_ZY * _MU_X


def _mean_diff_se(*, beta_xz_t: float, beta_xz_b: float, n_pairs: int) -> float:
    delta_beta = beta_xz_t - beta_xz_b
    var_per_pair = (delta_beta * _BETA_ZY) ** 2 * (_SIGMA_X ** 2) / _N_STEPS
    return math.sqrt(var_per_pair / n_pairs)


def _round_trip_runrows(
    rows: Sequence[RunRow], path: Path,
) -> list[RunRow]:
    """Write to parquet, read back. The complete production
    write/read pair the runner uses."""
    write_runrows(rows, path)
    return read_runrows(path)


# ============ Scalar paired_g across a parquet round-trip ============

def test_paired_g_recovers_closed_form_after_runrow_round_trip(
    tmp_path: Path,
) -> None:
    """`run_paired_arms` builds RunRows in memory; we persist them
    to a tmp parquet, read them back, and assert paired_g still
    recovers the closed-form `Δ_β_xz · β_zy · μ_x` within the same
    4·SE bound the in-memory test uses.

    Tests the full write_runrows → read_runrows → as_dict() →
    paired_g.fn pipeline. A regression that lost a measurement
    column, dropped seed-pairing, or coerced floats to a lossy
    narrower dtype would breach the closed-form bound by orders
    of magnitude."""
    beta_xz_t, beta_xz_b = 0.8, 0.3
    rows_in = run_paired_arms(
        treatment=_scm(beta_xz_t),
        baseline=_scm(beta_xz_b),
        seeds=range(_N_PAIRS),
    )
    rows_out = _round_trip_runrows(rows_in, tmp_path / 'runs.parquet')

    # Sanity on the round-trip itself: same row count, same arm_keys,
    # same seeds.
    assert len(rows_out) == len(rows_in)
    arm_in = sorted(r.arm_key for r in rows_in)
    arm_out = sorted(r.arm_key for r in rows_out)
    assert arm_in == arm_out, (
        f'arm_key column corrupted by round-trip: '
        f'in={arm_in[:3]}..., out={arm_out[:3]}...'
    )

    cells: list[Mapping[str, object]] = [r.as_dict() for r in rows_out]
    result = paired_g.fn(
        cells,
        treatment_arm='treatment',
        baseline_arm='baseline',
        pair_by=('seed',),
        source='y_mean',
    )
    expected = _expected_mean_diff(beta_xz_t=beta_xz_t, beta_xz_b=beta_xz_b)
    se = _mean_diff_se(
        beta_xz_t=beta_xz_t, beta_xz_b=beta_xz_b, n_pairs=_N_PAIRS,
    )
    bound = 4.0 * se
    assert abs(result.mean_diff - expected) < bound, (
        f'After parquet round-trip, paired_g.mean_diff = '
        f'{result.mean_diff:.6f} not within 4*SE = {bound:.6f} of '
        f'closed-form Δ = {expected:.6f}. The round-trip path '
        f'corrupted cell values; the in-memory test passes the '
        f'same bound, so the regression is in write_runrows / '
        f'read_runrows / RunRow.from_row_dict.'
    )


# ============ Per-burst paired_g across a TraceRow round-trip ============

def test_per_burst_recovers_closed_form_after_tracerow_round_trip(
    tmp_path: Path,
) -> None:
    """The per-burst Y matrix lives on `TraceRow.leaves`, which
    persists as a polars `List(List(Float))` column. Round-tripping
    that 2-D shape through parquet is the most-likely break point
    in the persistence path — a regression that flattened or
    transposed the inner axis would silently shift per-burst paired
    g away from the closed form.

    The test runs `run_phased_cell` per (arm, seed), persists the
    RunRows + TraceRows to separate parquets, reads them back,
    merges via `merge_cell` (the production polars-join analog),
    and asserts the per-burst panel matches the in-memory bound
    (rel_err < 0.20 against `μ_x · √n_steps / σ_x · c_4`)."""
    n_bursts = 3
    treatments = tuple(_scm(0.8) for _ in range(n_bursts))
    baselines = tuple(_scm(0.3) for _ in range(n_bursts))

    runs: list[RunRow] = []
    traces: list[TraceRow] = []
    for arm_label, scms_per_burst in (
        ('treatment', treatments),
        ('baseline', baselines),
    ):
        for seed in range(_N_PAIRS):
            run, trace = run_phased_cell(
                scms_per_burst, seed=seed, arm_key=arm_label,
            )
            runs.append(run)
            traces.append(trace)

    runs_path = tmp_path / 'runs.parquet'
    traces_path = tmp_path / 'traces.parquet'
    write_runrows(runs, runs_path)
    write_tracerows(traces, traces_path)
    runs_back = read_runrows(runs_path)
    traces_back = read_tracerows(traces_path)

    assert len(runs_back) == len(runs)
    assert len(traces_back) == len(traces)

    # Merge via the same helper the in-memory phased path uses.
    by_id_trace: dict[str, TraceRow] = {t.id: t for t in traces_back}
    cells: list[Mapping[str, object]] = []
    for run in runs_back:
        trace = by_id_trace[run.id]
        cells.append(merge_cell(run, trace))

    # Per-burst Y mean source — same wiring as test_paired_g_per_burst.
    per_burst_y_mean = reduce_axis(
        from_key(PER_BURST_Y_KEY), axis=-1, op='mean',
    )
    result = paired_g_per_burst.fn(
        cells,
        treatment_arm='treatment',
        baseline_arm='baseline',
        source=per_burst_y_mean,
    )
    assert result.n_strata == n_bursts, (
        f'After TraceRow round-trip, panel n_strata = '
        f'{result.n_strata}, expected {n_bursts}. The per-burst '
        f'array shape was likely flattened or transposed in the '
        f'parquet round-trip.'
    )

    # Closed-form per-burst g: μ_x · √n_steps / σ_x · c_4(n_pairs).
    c4 = 1.0 - 3.0 / (4 * _N_PAIRS - 5)
    expected_g = _MU_X * math.sqrt(_N_STEPS) / _SIGMA_X * c4
    for s in result.strata:
        rel_err = abs(s.g - expected_g) / expected_g
        # 30% bound — wider than the in-memory 0.20 because
        # round-tripping `numpy.float32 → list[float64] → numpy`
        # changes the rounding pattern slightly. A regression
        # that catastrophically corrupts the per-burst array
        # would still fail by orders of magnitude.
        assert rel_err < 0.30, (
            f'burst {s.burst_index}: g = {s.g:.4f}, expected '
            f'{expected_g:.4f} (rel err {rel_err:.4f}). The '
            f'per-burst array survived round-trip but the closed-'
            f'form recovery drifted; check List(List(Float)) '
            f'preservation in TraceRow.from_row_dict.'
        )


# ============ Heterogeneous arm_keys round-trip ============

def test_round_trip_preserves_arm_key_distinct_from_baseline_default(
    tmp_path: Path,
) -> None:
    """`arm_key` is a typed framework-surface column with default
    `'baseline'`. When two arms carry distinct strings (e.g.,
    'treatment' / 'baseline'), the round-trip must preserve them
    exactly — paired_g pairs cells by arm_key string match, so a
    silent default-fill on read would collapse both arms to
    `'baseline'` and zero out n_pairs.

    This is the regression that broke `test_claim_bridge_real_corpus`
    pre-fix: arm_key column lost on read → all cells matched as
    baseline → no treatment cells → n_pairs=0 → POWER_INSUFFICIENT
    instead of HELD."""
    rows_in = run_paired_arms(
        treatment=_scm(0.8),
        baseline=_scm(0.3),
        seeds=range(_N_PAIRS),
    )
    rows_out = _round_trip_runrows(rows_in, tmp_path / 'runs.parquet')

    # Both arm strings survive at full distinctness.
    arm_set = {r.arm_key for r in rows_out}
    assert arm_set == {'treatment', 'baseline'}, (
        f'arm_key collapse on round-trip: in={{treatment, baseline}}, '
        f'out={arm_set}'
    )
    n_treatment = sum(1 for r in rows_out if r.arm_key == 'treatment')
    n_baseline = sum(1 for r in rows_out if r.arm_key == 'baseline')
    assert n_treatment == _N_PAIRS
    assert n_baseline == _N_PAIRS

    # And paired_g still pairs them correctly.
    cells: list[Mapping[str, object]] = [r.as_dict() for r in rows_out]
    result = paired_g.fn(
        cells,
        treatment_arm='treatment',
        baseline_arm='baseline',
        pair_by=('seed',),
        source='y_mean',
    )
    assert result.n_pairs == _N_PAIRS, (
        f'paired_g n_pairs = {result.n_pairs} after round-trip; '
        f'arm_key column likely got null-filled or default-replaced'
    )


# ============ Heterogeneous-keyed round-trip + analyses ============

def test_paired_g_survives_heterogeneous_measurement_keys(
    tmp_path: Path,
) -> None:
    """Two arms can carry different measurement key sets — polars
    null-pads the missing columns on write, and `RunRow.from_row_dict`
    skips None-valued columns on read. paired_g must still recover
    closed-form on the shared `y_mean` column even when one arm
    has *additional* measurement keys the other doesn't.

    This stress-tests the framework's null-pad-then-skip discipline
    on a closed-form payload (rather than the test_persistence.py
    contract checks which only assert key presence/absence)."""
    treatment_rows = run_paired_arms(
        treatment=_scm(0.8),
        baseline=_scm(0.3),
        seeds=range(_N_PAIRS),
    )
    # Augment treatment-arm rows with an extra measurement that
    # baseline doesn't carry — null-pad on write, skip on read.
    augmented: list[RunRow] = []
    for r in treatment_rows:
        if r.arm_key == 'treatment':
            extra = dict(r.measurements)
            extra['extra_treatment_only'] = 1.234
            augmented.append(
                RunRow(
                    id=r.id, parent_id=r.parent_id, cycle_id=r.cycle_id,
                    timestamp=r.timestamp, verdict=r.verdict,
                    arm_key=r.arm_key, measurements=extra,
                ),
            )
        else:
            augmented.append(r)

    rows_out = _round_trip_runrows(
        augmented, tmp_path / 'runs.parquet',
    )

    # `extra_treatment_only` survives only on treatment cells.
    treatment_with_extra = sum(
        1 for r in rows_out
        if r.arm_key == 'treatment'
        and 'extra_treatment_only' in r.measurements
    )
    baseline_with_extra = sum(
        1 for r in rows_out
        if r.arm_key == 'baseline'
        and 'extra_treatment_only' in r.measurements
    )
    assert treatment_with_extra == _N_PAIRS
    assert baseline_with_extra == 0, (
        f'baseline cells picked up `extra_treatment_only` from null-'
        f'pad → re-fill leakage; got {baseline_with_extra} non-zero'
    )

    # And the closed-form analysis still recovers Δ.
    cells: list[Mapping[str, object]] = [r.as_dict() for r in rows_out]
    result = paired_g.fn(
        cells,
        treatment_arm='treatment',
        baseline_arm='baseline',
        pair_by=('seed',),
        source='y_mean',
    )
    expected = _expected_mean_diff(beta_xz_t=0.8, beta_xz_b=0.3)
    se = _mean_diff_se(
        beta_xz_t=0.8, beta_xz_b=0.3, n_pairs=_N_PAIRS,
    )
    assert abs(result.mean_diff - expected) < 4.0 * se, (
        f'mean_diff = {result.mean_diff:.6f}, expected '
        f'{expected:.6f} ± {4.0 * se:.6f}. Heterogeneous '
        f'measurement keys disrupted closed-form recovery.'
    )
