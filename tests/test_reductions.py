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

import jax.numpy as jnp
import pytest

from corroborate.measurable import Measurable
from corroborate.reductions import (
    from_key,
    growth_window,
    late_window_mean,
    max_abs,
    mean_window,
)


# ============ Leaf: from_key ============

def test_from_key_returns_measurable() -> None:
    m = from_key('q_max')
    assert isinstance(m, Measurable)
    assert m.name == 'q_max'
    assert m.reads == ('q_max',)


def test_from_key_reads_record_value() -> None:
    arr = jnp.asarray([1.0, 2.0, 3.0])
    record: Mapping[str, jnp.ndarray] = {'q_max': arr}
    m = from_key('q_max')
    assert jnp.allclose(m(record), arr)


# ============ max_abs ============

def test_max_abs_propagates_reads() -> None:
    m = max_abs(from_key('q_max'))
    assert m.reads == ('q_max',)
    assert m.name == 'q_max__max_abs'


def test_max_abs_computes_max_of_absolute() -> None:
    record: Mapping[str, jnp.ndarray] = {
        'q_max': jnp.asarray([1.0, -5.0, 3.0, -2.0]),
    }
    m = max_abs(from_key('q_max'))
    assert m(record) == 5.0


# ============ mean_window ============

def test_mean_window_late_10pct() -> None:
    record: Mapping[str, jnp.ndarray] = {
        'x': jnp.arange(10, dtype=jnp.float32),  # 0..9
    }
    m = mean_window(from_key('x'), 0.9, 1.0)
    # Last 10% of 10 elements = index 9 = value 9.0
    assert m(record) == pytest.approx(9.0)


def test_mean_window_first_quarter() -> None:
    record: Mapping[str, jnp.ndarray] = {
        'x': jnp.arange(8, dtype=jnp.float32),  # 0..7
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
    record: Mapping[str, jnp.ndarray] = {
        'x': jnp.asarray([7.0, 3.0]),
    }
    m = mean_window(from_key('x'), 0.0, 0.25)
    # 0.25 * 2 = 0; i_hi = 0 → bumped to 1 → mean(arr[0:1]) = 7
    assert m(record) == pytest.approx(7.0)


# ============ growth_window ============

def test_growth_window_decay_returns_lt_one() -> None:
    record: Mapping[str, jnp.ndarray] = {
        # Geometric decay 100 → ~6
        'r': jnp.asarray([100.0, 50.0, 25.0, 12.0, 6.0]),
    }
    m = growth_window(from_key('r'))
    # early = mean(arr[0:1]) = 100; late = mean(arr[3:5]) = 9
    g = m(record)
    assert 0.0 < g < 0.5


def test_growth_window_growth_returns_gt_one() -> None:
    record: Mapping[str, jnp.ndarray] = {
        'q': jnp.asarray([1.0, 2.0, 4.0, 8.0]),
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
    record: Mapping[str, jnp.ndarray] = {
        'ep_return': jnp.asarray([1.0] * 9 + [10.0]),
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
