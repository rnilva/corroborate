"""Direct tests on `_per_env_burst_predictor_mean` — the helper
that averages a precomputed per-burst array across cells of a
single env (optionally arm-filtered). Used by `mundlak_paired_g_per_burst`.

The integration tests on the Mundlak primitive cover the
end-to-end happy path; this file pins filter / boundary / NaN
branches the integration tests don't isolate."""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pytest

from corroborate.analyses.mundlak_paired_g_per_burst import (
    _per_env_burst_predictor_mean,
)


def _cell(
    cell_id: str, env: str, arm: str,
) -> Mapping[str, object]:
    return {'id': cell_id, 'env_name': env, 'arm_key': arm}


def test_per_env_burst_returns_average_at_burst_index() -> None:
    """3 cells in env_a, predictor array per cell. Average at
    burst_index=2 is the average of the 2nd-indexed value of each."""
    cells = [_cell(f'c{i}', 'env_a', 'baseline') for i in range(3)]
    arrays = {
        'c0': np.array([0.0, 1.0, 10.0]),
        'c1': np.array([0.0, 1.0, 20.0]),
        'c2': np.array([0.0, 1.0, 30.0]),
    }
    val = _per_env_burst_predictor_mean(
        cells, burst_index=2, per_cell_array=arrays,
        env_name='env_a', arm_filter='baseline',
    )
    assert val == pytest.approx(20.0)


def test_per_env_burst_filters_by_env_name() -> None:
    """Only cells with matching env_name contribute. Pin the
    `c.get('env_name') != env_name: continue` filter."""
    cells = [
        _cell('c0', 'env_a', 'arm'),
        _cell('c1', 'env_b', 'arm'),    # excluded
        _cell('c2', 'env_a', 'arm'),
    ]
    arrays = {
        'c0': np.array([0.0, 5.0]),
        'c1': np.array([0.0, 100.0]),    # would dominate if included
        'c2': np.array([0.0, 7.0]),
    }
    val = _per_env_burst_predictor_mean(
        cells, burst_index=1, per_cell_array=arrays,
        env_name='env_a', arm_filter=None,
    )
    assert val == pytest.approx(6.0)    # avg(5, 7), not affected by env_b


def test_per_env_burst_filters_by_arm_when_filter_provided() -> None:
    """When `arm_filter` is given, only cells with matching
    `arm_key` contribute. Pin:

    - `arm_filter is not None` (vs `is None` mutant which would
      invert the gate)
    - `c.get('arm_key') != arm_filter` (vs `==` mutant)"""
    cells = [
        _cell('c0', 'env_a', 'baseline'),
        _cell('c1', 'env_a', 'treatment'),    # excluded
        _cell('c2', 'env_a', 'baseline'),
    ]
    arrays = {
        'c0': np.array([0.0, 5.0]),
        'c1': np.array([0.0, 100.0]),
        'c2': np.array([0.0, 7.0]),
    }
    val = _per_env_burst_predictor_mean(
        cells, burst_index=1, per_cell_array=arrays,
        env_name='env_a', arm_filter='baseline',
    )
    assert val == pytest.approx(6.0)


def test_per_env_burst_arm_filter_none_admits_all_arms() -> None:
    """`arm_filter=None` skips the arm-filter guard entirely. Pin
    the `arm_filter is not None` short-circuit."""
    cells = [
        _cell('c0', 'env_a', 'arm_x'),
        _cell('c1', 'env_a', 'arm_y'),
    ]
    arrays = {
        'c0': np.array([0.0, 4.0]),
        'c1': np.array([0.0, 8.0]),
    }
    val = _per_env_burst_predictor_mean(
        cells, burst_index=1, per_cell_array=arrays,
        env_name='env_a', arm_filter=None,
    )
    assert val == pytest.approx(6.0)


def test_per_env_burst_skips_cells_with_burst_index_out_of_range() -> None:
    """Pin `burst_index >= arr.shape[0]`: cells whose array is
    shorter than burst_index+1 are skipped, not crash, not pad.
    `>= arr.shape[0]` mutated to `> arr.shape[0]` would let
    burst_index=arr.shape[0] through, indexing past the end."""
    cells = [
        _cell('c0', 'env_a', 'arm'),
        _cell('c1', 'env_a', 'arm'),
    ]
    arrays = {
        'c0': np.array([1.0, 2.0]),         # shape[0] = 2
        'c1': np.array([10.0, 20.0, 30.0]),  # shape[0] = 3
    }
    # burst_index=2: c0 out of range, c1 has value 30.
    val = _per_env_burst_predictor_mean(
        cells, burst_index=2, per_cell_array=arrays,
        env_name='env_a', arm_filter=None,
    )
    assert val == pytest.approx(30.0)


def test_per_env_burst_skips_cells_with_no_array_entry() -> None:
    """Cells whose id is missing from `per_cell_array` are
    skipped (the `arr is None` branch)."""
    cells = [
        _cell('c0', 'env_a', 'arm'),
        _cell('c_missing', 'env_a', 'arm'),
    ]
    arrays = {'c0': np.array([0.0, 5.0])}
    val = _per_env_burst_predictor_mean(
        cells, burst_index=1, per_cell_array=arrays,
        env_name='env_a', arm_filter=None,
    )
    assert val == pytest.approx(5.0)


def test_per_env_burst_returns_nan_when_no_cells_match() -> None:
    """Empty result list → NaN. Pin `float('nan')` against
    `float(None)` (TypeError) and `float('XXnanXX')` (ValueError)."""
    val = _per_env_burst_predictor_mean(
        cells=[], burst_index=0, per_cell_array={},
        env_name='env_a', arm_filter=None,
    )
    assert math.isnan(val)


def test_per_env_burst_skips_non_string_id_then_continues() -> None:
    """Cells whose `id` is not a string are skipped (the
    `if not isinstance(cell_id, str): continue` branch).
    Pin `continue` against `break` mutant — under break the loop
    would terminate at the first non-string-id cell, dropping
    later valid cells from the average."""
    cells: list[Mapping[str, object]] = [
        {'id': 'c0', 'env_name': 'env_a', 'arm_key': 'arm'},
        {'id': 12345, 'env_name': 'env_a', 'arm_key': 'arm'},  # int id
        {'id': 'c2', 'env_name': 'env_a', 'arm_key': 'arm'},
    ]
    arrays = {
        'c0': np.array([0.0, 4.0]),
        'c2': np.array([0.0, 8.0]),
    }
    val = _per_env_burst_predictor_mean(
        cells, burst_index=1, per_cell_array=arrays,
        env_name='env_a', arm_filter=None,
    )
    # Original: skip the int-id cell, average c0 + c2 → 6.
    # Mutant break: stop at int-id cell → only c0 → 4.
    assert val == pytest.approx(6.0)


def test_per_env_burst_skips_nan_predictor_values() -> None:
    """Cells with NaN at the burst index are excluded from the
    average (the `if not math.isnan(v): vals.append(v)` branch).
    Cf. mutant `continue` → `break` would stop the loop."""
    cells = [_cell(f'c{i}', 'env_a', 'arm') for i in range(3)]
    arrays = {
        'c0': np.array([0.0, 4.0]),
        'c1': np.array([0.0, float('nan')]),    # skipped
        'c2': np.array([0.0, 8.0]),
    }
    val = _per_env_burst_predictor_mean(
        cells, burst_index=1, per_cell_array=arrays,
        env_name='env_a', arm_filter=None,
    )
    assert val == pytest.approx(6.0)
