"""Regression tests for `compute_missing_columns` failure-mode handling.

Two contracts:

1. Per-cell measurable failures map to None per-cell; downstream
   NaN-skip cleanly. (Existing behavior; this confirms nothing
   broke after narrowing the catch-all `except Exception`.)
2. A measurable that ALWAYS fails (typo in its body, wrong arg
   shape) emits a stderr warning so authoring bugs don't disappear
   into a silent all-null column.
"""
from __future__ import annotations

import sys
from collections.abc import Mapping
from io import StringIO
from typing import TextIO

import polars as pl
import pytest

from corroborate.measurables import (
    compute_missing_columns, measurable,
)


@pytest.fixture
def stderr_capture() -> StringIO:
    """Replace stderr with a StringIO for the duration of one test."""
    return StringIO()


def _swap_stderr(buf: StringIO) -> TextIO:
    """Context-manager-free helper: returns the previous stderr so
    the caller can restore."""
    prev = sys.stderr
    sys.stderr = buf
    return prev


def test_per_cell_failure_maps_to_none() -> None:
    """A measurable that fails on SOME cells (not all) maps those
    cells to None silently — no warning. The legitimate
    "missing inputs on a subset" case."""

    @measurable(reads=('x',))
    def needs_positive_x(record: Mapping[str, object]) -> float:
        x = record.get('x')
        if not isinstance(x, (int, float)):
            raise TypeError(f'x = {x!r}')
        if x < 0:
            raise ValueError('x negative')
        return float(x) ** 0.5
    _ = needs_positive_x   # @measurable auto-registers on definition

    df = pl.DataFrame({
        'x': [1.0, 4.0, -1.0, 9.0],
        'id': ['a', 'b', 'c', 'd'],
    })
    buf = StringIO()
    prev = _swap_stderr(buf)
    try:
        out = compute_missing_columns(df, ['needs_positive_x'])
    finally:
        sys.stderr = prev

    # Mixed success/failure → no warning (some cells succeeded).
    assert 'WARNING' not in buf.getvalue()
    assert 'needs_positive_x' in out.columns
    vals = out['needs_positive_x'].to_list()
    # 3 successes, 1 None for the negative-x row.
    assert vals[0] == 1.0
    assert vals[1] == 2.0
    assert vals[2] is None    # ValueError → mapped to None
    assert vals[3] == 3.0


def test_always_failing_measurable_warns_to_stderr() -> None:
    """A measurable that fails on EVERY cell (e.g., references a
    column that never exists) is an authoring bug, not a missing-
    input situation. The all-null column should be paired with a
    stderr warning naming the measurable + the exception class."""

    @measurable(reads=('does_not_exist',))
    def always_fails(record: Mapping[str, object]) -> float:
        # KeyError on every cell — reads from a never-present key.
        return float(record['does_not_exist'])  # type: ignore[arg-type]
    _ = always_fails

    df = pl.DataFrame({'x': [1.0, 2.0, 3.0]})

    buf = StringIO()
    prev = _swap_stderr(buf)
    try:
        out = compute_missing_columns(df, ['always_fails'])
    finally:
        sys.stderr = prev

    # All-null column emerges (downstream NaN-skip).
    assert out['always_fails'].null_count() == 3

    # Warning fired naming the measurable + exception type.
    warning = buf.getvalue()
    assert 'WARNING' in warning, f'no warning emitted: {warning!r}'
    assert 'always_fails' in warning
    assert 'KeyError' in warning


def test_unrelated_exception_propagates() -> None:
    """A measurable raising something OUTSIDE the narrowed
    `(KeyError, TypeError, ValueError, ZeroDivisionError)` whitelist
    propagates up, not silently mapping to None.

    Specifically: AttributeError (typo on a record-side attribute
    access — distinct from KeyError on dict access) is an authoring
    bug that should fail loudly. Pre-fix, the bare `except Exception`
    swallowed everything; post-fix it propagates."""

    @measurable(reads=('x',))
    def authoring_typo(record: Mapping[str, object]) -> float:
        # Pretend the author meant `record.get('x')` but typo'd.
        x = record  # noqa
        return x.nonexistent_attribute()  # type: ignore[attr-defined]
    _ = authoring_typo

    df = pl.DataFrame({'x': [1.0, 2.0, 3.0]})
    with pytest.raises(AttributeError):
        compute_missing_columns(df, ['authoring_typo'])
