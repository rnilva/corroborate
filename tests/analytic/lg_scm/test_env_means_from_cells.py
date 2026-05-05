"""Direct tests on `_env_means_from_cells` — the helper that
lifts cell-level columns to env-level mean covariates.

Pin the per-cell `continue` branches that drop:
- cells lacking a string env_name
- columns with non-numeric values
- NaN-bearing values

against `break` mutants that would terminate the loop early
and skip later valid cells."""
from __future__ import annotations

from collections.abc import Mapping

import pytest

from corroborate.analyses.meta_regression_per_burst import (
    _env_means_from_cells,
)


def test_env_means_skips_non_string_env_continues_to_next_cell() -> None:
    """A cell with non-string `env_name` is skipped (continue,
    not break). Pin against `break` mutant: under break, the
    function would terminate at the first non-string env_name
    cell and drop later valid cells.

    Construct: cell0 valid (env_a), cell1 has int env (skipped),
    cell2 valid (env_a). Original keeps cell0 + cell2 → mean = 6.
    Mutant break: stops at cell1 → only cell0 → mean = 4."""
    cells: list[Mapping[str, object]] = [
        {'env_name': 'env_a', 'col1': 4.0},
        {'env_name': 12345, 'col1': 100.0},        # non-string
        {'env_name': 'env_a', 'col1': 8.0},
    ]
    result = _env_means_from_cells(cells, columns=('col1',))
    assert 'env_a' in result
    assert result['env_a']['col1'] == pytest.approx(6.0)


def test_env_means_skips_non_numeric_value_continues_to_next_column() -> None:
    """When a column value is non-numeric for one cell, it's
    skipped FOR THAT CELL'S COLUMN ONLY (continue inner loop).
    Pin against `break` mutant: break would exit the column
    loop entirely, losing later columns for that cell.

    Construct: 1 cell with col1=valid, col2=string, col3=valid.
    Original collects col1 and col3 (skips col2). Mutant break
    after col2 string fails → only col1 collected, no col3."""
    cells: list[Mapping[str, object]] = [
        {'env_name': 'env_a', 'col1': 4.0, 'col2': 'string!', 'col3': 7.0},
    ]
    result = _env_means_from_cells(
        cells, columns=('col1', 'col2', 'col3'),
    )
    assert result['env_a']['col1'] == pytest.approx(4.0)
    # col2 not collected (non-numeric).
    assert 'col2' not in result['env_a']
    # col3 must still be collected — pin against break-after-col2.
    assert result['env_a']['col3'] == pytest.approx(7.0)


def test_env_means_skips_nan_value_continues_to_next_column() -> None:
    """NaN value in a column is skipped (continue), not break.
    Pin against `break` mutant: break would exit the column
    loop, missing later valid columns for the same cell.

    Construct: 1 cell with col1=valid, col2=NaN, col3=valid.
    Original collects col1 and col3. Mutant break: only col1."""
    cells: list[Mapping[str, object]] = [
        {'env_name': 'env_a', 'col1': 5.0, 'col2': float('nan'), 'col3': 9.0},
    ]
    result = _env_means_from_cells(
        cells, columns=('col1', 'col2', 'col3'),
    )
    assert result['env_a']['col1'] == pytest.approx(5.0)
    assert 'col2' not in result['env_a']
    assert result['env_a']['col3'] == pytest.approx(9.0)
