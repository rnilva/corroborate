"""Direct tests on `_env_means_from_df` — the frame-native helper
that lifts cell-level columns to env-level mean covariates.

The rows-era helper made its skip decisions per cell; the frame
shape moves them to the honest granularity: NaN skips per VALUE
(fill_nan → null-skipping mean), non-numeric and absent columns
are excluded as COLUMNS (a frame column has one dtype), and a
non-string env groups into its own stratum and is dropped at
read-out — later valid data must survive every one of these.
"""
from __future__ import annotations

import polars as pl
import pytest

from corroborate.analyses.panel.meta_regression_per_burst import (
    _env_means_from_df,
)


def test_non_string_env_rows_drop_without_losing_later_rows() -> None:
    df = pl.DataFrame({
        'env_name': ['env_a', None, 'env_a'],
        'col1': [4.0, 100.0, 8.0],
    })
    result = _env_means_from_df(df, columns=('col1',))
    assert set(result) == {'env_a'}
    assert result['env_a']['col1'] == pytest.approx(6.0)


def test_non_numeric_column_excluded_others_survive() -> None:
    df = pl.DataFrame({
        'env_name': ['env_a'],
        'col1': [4.0],
        'col2': ['string!'],
        'col3': [7.0],
    })
    result = _env_means_from_df(df, columns=('col1', 'col2', 'col3'))
    assert result['env_a']['col1'] == pytest.approx(4.0)
    assert 'col2' not in result['env_a']
    assert result['env_a']['col3'] == pytest.approx(7.0)


def test_nan_values_skip_within_column_not_across() -> None:
    """NaN is a per-value miss: the column's mean is taken over
    the finite values, and an all-NaN column drops from that
    env's mapping while later columns survive."""
    df = pl.DataFrame({
        'env_name': ['env_a', 'env_a'],
        'col1': [5.0, 7.0],
        'col2': [float('nan'), float('nan')],
        'col3': [9.0, float('nan')],
    })
    result = _env_means_from_df(df, columns=('col1', 'col2', 'col3'))
    assert result['env_a']['col1'] == pytest.approx(6.0)
    assert 'col2' not in result['env_a']
    assert result['env_a']['col3'] == pytest.approx(9.0)


def test_absent_column_and_boolean_lift() -> None:
    df = pl.DataFrame({
        'env_name': ['env_a', 'env_a'],
        'flag': [True, False],
    })
    result = _env_means_from_df(df, columns=('flag', 'missing'))
    assert result['env_a']['flag'] == pytest.approx(0.5)
    assert 'missing' not in result['env_a']
