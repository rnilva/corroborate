"""Direct tests on `_cells_to_dataframe` — the helper that
projects a corpus to a pandas DataFrame for DoWhy. Pin the
type-coercion + complete-row branches that the integration
tests don't isolate."""
from __future__ import annotations

from collections.abc import Mapping

from corroborate.analyses._dowhy_internal import _cells_to_dataframe


def test_cells_to_dataframe_includes_bool_as_float() -> None:
    """`isinstance(v, bool)` branch coerces bool → float (0.0/1.0).
    Pin against:
    - `row[k] = None` mutant (would make the column None)
    - `row[k] = float(None)` mutant (TypeError)"""
    cells: list[Mapping[str, object]] = [
        {'a': True, 'b': 1.0},
        {'a': False, 'b': 2.0},
    ]
    df = _cells_to_dataframe(cells, keys=['a', 'b'])
    assert len(df) == 2
    assert df['a'].tolist() == [1.0, 0.0]


def test_cells_to_dataframe_drops_incomplete_cell() -> None:
    """A cell missing one of the requested keys is skipped (the
    `complete = False; break` path). Pin:
    - `complete = True` mutant (would keep the incomplete row)
    - `break` → `return` mutant (would exit the function early
      and drop later valid cells too)

    Construct: cell0 valid, cell1 missing 'b', cell2 valid.
    Original keeps cell0 + cell2 (2 rows). Mutant complete=True
    keeps all 3 (with NaN/missing for b in cell1). Mutant
    break→return exits at cell1 → 1 row only."""
    cells: list[Mapping[str, object]] = [
        {'a': 1.0, 'b': 1.0},
        {'a': 2.0},                # missing 'b' → skip this cell
        {'a': 3.0, 'b': 3.0},
    ]
    df = _cells_to_dataframe(cells, keys=['a', 'b'])
    assert len(df) == 2    # cell0 + cell2 only
    assert df['a'].tolist() == [1.0, 3.0]


def test_cells_to_dataframe_drops_non_scalar_values() -> None:
    """Non-bool, non-int, non-float values are skipped via the
    same `complete = False; break` path."""
    cells: list[Mapping[str, object]] = [
        {'a': 1.0, 'b': 'string!'},    # b is non-scalar → skip
        {'a': 2.0, 'b': 2.0},
    ]
    df = _cells_to_dataframe(cells, keys=['a', 'b'])
    assert len(df) == 1
    assert df['a'].tolist() == [2.0]


def test_cells_to_dataframe_int_value_coerced_to_float() -> None:
    """int (non-bool) is coerced to float via the second isinstance
    branch."""
    cells: list[Mapping[str, object]] = [
        {'a': 5, 'b': 1.5},
    ]
    df = _cells_to_dataframe(cells, keys=['a', 'b'])
    assert df['a'].tolist() == [5.0]
    assert df['a'].dtype.kind == 'f'    # float dtype, not int
