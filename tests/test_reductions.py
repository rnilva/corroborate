"""Tests for the framework-level reductions module — `from_key`,
`max_abs`, `mean_window`, `growth_window`, `late_window_mean`.

Verifies:
- `from_key` lifts a record key into a `Measurable` with `reads`
  populated.
- Reductions propagate `reads` from operand to output.
- Composed names are deterministic and informative.
- The reductions compute the expected scalar over a synthetic
  trajectory."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import numpy.typing as npt
import pytest

from corroborate.measurables import Measurable
from corroborate.measurables.reductions import (
    from_key,
    growth_window,
    late_window_mean,
    max_abs,
    mean_peak_window,
    mean_window,
    peak_centered_window,
    reduce_axis,
    select_at,
)


# ============ Leaf: from_key ============

def test_from_key_returns_measurable() -> None:
    m = from_key('q_max')
    assert isinstance(m, Measurable)
    assert m.name == 'q_max'
    assert m.reads == ('q_max',)


def test_from_key_reads_record_value() -> None:
    arr = np.asarray([1.0, 2.0, 3.0])
    record: Mapping[str, npt.NDArray[np.floating]] = {'q_max': arr}
    m = from_key('q_max')
    assert np.allclose(m(record), arr)


# ============ max_abs ============

def test_max_abs_propagates_reads() -> None:
    m = max_abs(from_key('q_max'))
    assert m.reads == ('q_max',)
    assert m.name == 'q_max__max_abs'


def test_max_abs_computes_max_of_absolute() -> None:
    record: Mapping[str, npt.NDArray[np.floating]] = {
        'q_max': np.asarray([1.0, -5.0, 3.0, -2.0]),
    }
    m = max_abs(from_key('q_max'))
    assert m(record) == 5.0


# ============ mean_window ============

def test_mean_window_late_10pct() -> None:
    record: Mapping[str, npt.NDArray[np.floating]] = {
        'x': np.arange(10, dtype=np.float32),  # 0..9
    }
    m = mean_window(from_key('x'), 0.9, 1.0)
    # Last 10% of 10 elements = index 9 = value 9.0
    assert m(record) == pytest.approx(9.0)


def test_mean_window_first_quarter() -> None:
    record: Mapping[str, npt.NDArray[np.floating]] = {
        'x': np.arange(8, dtype=np.float32),  # 0..7
    }
    m = mean_window(from_key('x'), 0.0, 0.25)
    # First 25% of 8 elements = indices [0, 2) → mean(0, 1) = 0.5
    assert m(record) == pytest.approx(0.5)


def test_mean_window_propagates_reads_and_names_window() -> None:
    m = mean_window(from_key('q_max'), 0.75, 1.0)
    assert m.reads == ('q_max',)
    assert m.name == 'q_max__mean_75_100'


def test_mean_window_rejects_inverted_bounds() -> None:
    with pytest.raises(ValueError):
        mean_window(from_key('x'), 0.5, 0.5)
    with pytest.raises(ValueError):
        mean_window(from_key('x'), -0.1, 0.5)
    with pytest.raises(ValueError):
        mean_window(from_key('x'), 0.5, 1.1)


def test_mean_window_handles_tiny_arrays() -> None:
    """When `n` is so small the window collapses, the reduction
    still returns a valid mean (i_hi gets bumped to i_lo + 1)."""
    record: Mapping[str, npt.NDArray[np.floating]] = {
        'x': np.asarray([7.0, 3.0]),
    }
    m = mean_window(from_key('x'), 0.0, 0.25)
    # 0.25 * 2 = 0; i_hi = 0 → bumped to 1 → mean(arr[0:1]) = 7
    assert m(record) == pytest.approx(7.0)


# ============ growth_window ============

def test_growth_window_decay_returns_lt_one() -> None:
    record: Mapping[str, npt.NDArray[np.floating]] = {
        # Geometric decay 100 → ~6
        'r': np.asarray([100.0, 50.0, 25.0, 12.0, 6.0]),
    }
    m = growth_window(from_key('r'))
    # early = mean(arr[0:1]) = 100; late = mean(arr[3:5]) = 9
    g = m(record)
    assert 0.0 < g < 0.5


def test_growth_window_growth_returns_gt_one() -> None:
    record: Mapping[str, npt.NDArray[np.floating]] = {
        'q': np.asarray([1.0, 2.0, 4.0, 8.0]),
    }
    m = growth_window(from_key('q'))
    # early ≈ 1, late ≈ 8 → growth ≈ 8
    assert m(record) > 4.0


def test_growth_window_propagates_reads() -> None:
    m = growth_window(from_key('q_max'))
    assert m.reads == ('q_max',)
    assert 'q_max__growth' in m.name


# ============ late_window_mean (outcome projection) ============

def test_late_window_mean_returns_late_fraction() -> None:
    record: Mapping[str, npt.NDArray[np.floating]] = {
        'ep_return': np.asarray([1.0] * 9 + [10.0]),
    }
    m = late_window_mean('ep_return', fraction=0.1)
    # Last 10% of 10 = index [9, 10) = 10.0
    assert m(record) == pytest.approx(10.0)
    assert m.reads == ('ep_return',)


def test_late_window_mean_rejects_invalid_fraction() -> None:
    with pytest.raises(ValueError):
        late_window_mean('x', fraction=0.0)
    with pytest.raises(ValueError):
        late_window_mean('x', fraction=1.5)


# ============ mean_peak_window ============

def test_mean_peak_window_pre_half() -> None:
    """Peak at index 8 of 10; pre_frac=0.5 → window [4, 8].
    arr = [0, 1, 2, ..., 9]; mean over [4, 5, 6, 7] = 5.5."""
    arr = np.asarray([float(i) for i in range(10)])
    record: Mapping[str, object] = {
        'q': arr, 'peak_idx': 8,
    }
    m = mean_peak_window(from_key('q'), 'peak_idx', pre_frac=0.5)
    assert m(record) == pytest.approx(5.5)


def test_mean_peak_window_propagates_reads() -> None:
    m = mean_peak_window(from_key('q'), 'peak_idx', pre_frac=0.5)
    assert m.reads == ('q', 'peak_idx')
    assert 'peak_pre50' in m.name


def test_mean_peak_window_handles_missing_peak() -> None:
    """Peak key missing from record → NaN, not crash."""
    arr = np.asarray([float(i) for i in range(10)])
    record: Mapping[str, npt.NDArray[np.floating]] = {'q': arr}
    m = mean_peak_window(from_key('q'), 'peak_idx', pre_frac=0.5)
    import math
    assert math.isnan(m(record))


def test_mean_peak_window_rejects_bad_pre_frac() -> None:
    with pytest.raises(ValueError):
        mean_peak_window(from_key('x'), 'peak', pre_frac=0.0)
    with pytest.raises(ValueError):
        mean_peak_window(from_key('x'), 'peak', pre_frac=1.5)


# ============ peak_centered_window ============

def test_peak_centered_window_default() -> None:
    """20-element array, peak at 10, half_width=0.125 (h=2).
    Window [8, 12); arr=[0..19]; mean = (8+9+10+11)/4 = 9.5."""
    arr = np.asarray([float(i) for i in range(20)])
    record: Mapping[str, object] = {
        'q': arr, 'peak_idx': 10,
    }
    m = peak_centered_window(from_key('q'), 'peak_idx')
    assert m(record) == pytest.approx(9.5)


def test_peak_centered_window_clipped_at_end() -> None:
    """Peak at end: window [n-h, n] (clipped). For n=10, peak=10,
    h=1: window [9, 10] = [9.0]; mean=9.0."""
    arr = np.asarray([float(i) for i in range(10)])
    record: Mapping[str, object] = {
        'q': arr, 'peak_idx': 10,
    }
    m = peak_centered_window(
        from_key('q'), 'peak_idx', half_width_frac=0.1,
    )
    # Window is too narrow (1 element) → NaN per the < 2 guard.
    import math
    assert math.isnan(m(record))


def test_peak_centered_window_rejects_bad_half_width() -> None:
    with pytest.raises(ValueError):
        peak_centered_window(from_key('x'), 'peak', half_width_frac=0.0)
    with pytest.raises(ValueError):
        peak_centered_window(from_key('x'), 'peak', half_width_frac=0.6)


# ============ reduce_axis old-protocol shim ============


def test_reduce_axis_modern_2d_protocol() -> None:
    """Standard (n_bursts, n_episodes) → (n_bursts,) reduction
    along the inner axis. The canonical chain shape — bridges
    expect this."""
    modern_2d = Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ](
        fn=lambda r: np.asarray([[1.0, 3.0], [2.0, 4.0]]),
        name='modern_2d',
        reads=(),
    )
    reduced = reduce_axis(modern_2d, axis=-1, op='mean')
    out = reduced({})
    assert out.shape == (2,)
    assert np.allclose(out, [2.0, 3.0])


def test_reduce_axis_zero_dim_legacy_raises_axiserror() -> None:
    """**Documented behavior**: 0-d input (legacy "scalar
    mc_return" protocol) raises `numpy.exceptions.AxisError`.
    `compute_missing_columns`'s per-cell `except (KeyError,
    TypeError, ValueError, ZeroDivisionError)` catches AxisError
    via the ValueError branch (`AxisError extends ValueError`)
    and stores None for that cell. The "Old-protocol shim"
    attempted earlier is reverted because input shapes across
    affected corpora are heterogeneous (0-d, 1-d, 2-d, 3-d) and
    a uniform promotion broke the downstream `mean_window` chain.
    Cells with non-2-d `mc_return` consistently null-store —
    honest, even if it costs n_pairs."""
    legacy_scalar = Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ](
        fn=lambda r: np.asarray(2.5),
        name='legacy_scalar',
        reads=(),
    )
    reduced = reduce_axis(legacy_scalar, axis=-1, op='mean')
    with pytest.raises(np.exceptions.AxisError):
        _ = reduced({})


# ============ select_at: select value at argmax/argmin of indicator ============


def test_select_at_picks_value_at_indicator_argmax() -> None:
    """Generic primitive: reduce array A to scalar by indexing at
    the argmax of array B."""
    values_arr = np.asarray([10.0, 20.0, 30.0, 40.0])
    indicator_arr = np.asarray([1.0, 5.0, 2.0, 3.0])
    record: Mapping[str, npt.NDArray[np.floating]] = {
        'values': values_arr, 'indicator': indicator_arr,
    }
    m = select_at(from_key('values'), from_key('indicator'), op='argmax')
    # indicator argmax at index 1, values[1] = 20.0
    assert m(record) == 20.0


def test_select_at_picks_value_at_indicator_argmin() -> None:
    values_arr = np.asarray([10.0, 20.0, 30.0, 40.0])
    indicator_arr = np.asarray([5.0, 1.0, 3.0, 2.0])
    record: Mapping[str, npt.NDArray[np.floating]] = {
        'values': values_arr, 'indicator': indicator_arr,
    }
    m = select_at(from_key('values'), from_key('indicator'), op='argmin')
    # indicator argmin at index 1, values[1] = 20.0
    assert m(record) == 20.0


def test_select_at_propagates_reads_union() -> None:
    m = select_at(from_key('q'), from_key('mc'), op='argmax')
    assert m.reads == ('mc', 'q')  # sorted union
    assert m.name == 'q__argmax_mc_axis_-1'


def test_select_at_returns_nan_on_empty() -> None:
    import math
    record: Mapping[str, npt.NDArray[np.floating]] = {
        'values': np.asarray([]), 'indicator': np.asarray([1.0]),
    }
    m = select_at(from_key('values'), from_key('indicator'))
    assert math.isnan(m(record))


def test_select_at_returns_nan_on_shape_mismatch() -> None:
    import math
    record: Mapping[str, npt.NDArray[np.floating]] = {
        'values': np.asarray([1.0, 2.0]),
        'indicator': np.asarray([1.0, 2.0, 3.0]),
    }
    m = select_at(from_key('values'), from_key('indicator'))
    assert math.isnan(m(record))


def test_select_at_returns_nan_on_all_nan_indicator() -> None:
    import math
    record: Mapping[str, npt.NDArray[np.floating]] = {
        'values': np.asarray([1.0, 2.0, 3.0]),
        'indicator': np.asarray([float('nan')] * 3),
    }
    m = select_at(from_key('values'), from_key('indicator'))
    assert math.isnan(m(record))


def test_select_at_handles_nan_in_indicator() -> None:
    """Indicator has a NaN, but other values are finite. argmax
    should pick the largest finite value."""
    values_arr = np.asarray([10.0, 20.0, 30.0])
    indicator_arr = np.asarray([1.0, float('nan'), 5.0])
    record: Mapping[str, npt.NDArray[np.floating]] = {
        'values': values_arr, 'indicator': indicator_arr,
    }
    m = select_at(from_key('values'), from_key('indicator'), op='argmax')
    # argmax (skipping NaN) at index 2 → values[2] = 30
    assert m(record) == 30.0
