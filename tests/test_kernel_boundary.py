"""`corroborate.data.to_dataframe` — the one entry-boundary
conversion. These pin the properties every rows-holding caller
relies on, replacing the per-analysis DataFrame-vs-rows parity
tests (which, after the plain-DataFrame migration, routed both
arms through this same helper and so pinned nothing).
"""
from __future__ import annotations

import polars as pl
import pytest

from corroborate.data import to_dataframe


def test_full_scan_keeps_late_appearing_columns() -> None:
    """Schema inference must scan every row: a column first
    appearing after row 100 (diagonal-relaxed corpora do this
    routinely) survives with its values — the bare
    `pl.from_dicts` default would silently drop it."""
    rows: list[dict[str, object]] = [
        {'id': f'c{i}', 'x': float(i)} for i in range(120)
    ]
    for i, row in enumerate(rows[110:]):
        row['late'] = float(i)
    df = to_dataframe(rows)
    assert 'late' in df.columns
    assert df['late'].drop_nulls().len() == 10


def test_mixed_int_float_promotes_losslessly() -> None:
    df = to_dataframe([{'v': 1}, {'v': 2.5}])
    assert df['v'].to_list() == [1.0, 2.5]


def test_dataframe_passes_through_by_identity() -> None:
    df = pl.DataFrame({'a': [1, 2]})
    assert to_dataframe(df) is df


def test_mixed_scalar_categories_raise_by_name() -> None:
    """One stray text value among numbers would be silently
    stringified by inference — after which numeric readers see
    NaN everywhere and a verdict can quietly degrade. Loud
    instead, naming the column."""
    with pytest.raises(ValueError, match="column 'v' mixes"):
        _ = to_dataframe([{'v': 1.0}, {'v': 'diverged'}])


def test_empty_rows_make_an_empty_frame() -> None:
    df = to_dataframe([])
    assert df.height == 0
