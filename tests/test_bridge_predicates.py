"""Tests for `corroborate.bridge.predicates` — the NaN-safe scope
helpers that encapsulate polars's NaN-comparison gotchas."""
from __future__ import annotations

import math

import polars as pl
import pytest

from corroborate.bridge.predicates import (
    finite,
    finite_between,
    finite_ge,
    finite_gt,
    finite_le,
    finite_lt,
    partition_aggregate,
)


@pytest.fixture
def df_with_nan() -> pl.DataFrame:
    """3 rows: one finite, one NaN, one inf. The NaN row is the
    canonical pitfall — polars's `>=` admits it through the filter
    unless guarded. The inf row catches the broader is_finite
    semantics."""
    return pl.DataFrame({
        'x': [1.5, float('nan'), float('inf')],
        'group': ['a', 'a', 'b'],
    })


def test_finite_excludes_nan_and_inf(df_with_nan: pl.DataFrame) -> None:
    out = df_with_nan.filter(finite('x'))
    assert out.height == 1
    assert out['x'][0] == 1.5


def test_finite_gt_naked_polars_admits_nan() -> None:
    """Document the underlying polars pitfall the helper exists
    to fix: `pl.col(x) > 0` keeps NaN rows."""
    df = pl.DataFrame({'x': [1.0, float('nan'), -1.0]})
    naked = df.filter(pl.col('x') > 0)
    # Polars's NaN-greater-than-zero promotes to True, so the NaN
    # row is admitted. This is the bug the helper fixes.
    assert naked.height == 2
    safe = df.filter(finite_gt('x', 0))
    assert safe.height == 1
    assert safe['x'][0] == 1.0


def test_finite_lt_excludes_nan() -> None:
    df = pl.DataFrame({'x': [1.0, float('nan'), -1.0]})
    out = df.filter(finite_lt('x', 0))
    assert out.height == 1
    assert out['x'][0] == -1.0


def test_finite_ge_le_inclusive() -> None:
    df = pl.DataFrame({'x': [1.0, float('nan'), 2.0, 3.0]})
    assert df.filter(finite_ge('x', 2.0)).height == 2  # 2.0 + 3.0
    assert df.filter(finite_le('x', 2.0)).height == 2  # 1.0 + 2.0


def test_finite_between_inclusive_both_ends() -> None:
    df = pl.DataFrame({'x': [0.5, 1.0, float('nan'), 2.0, 2.5]})
    out = df.filter(finite_between('x', 1.0, 2.0))
    assert out.height == 2
    assert sorted(out['x'].to_list()) == [1.0, 2.0]


def test_partition_aggregate_skips_nan() -> None:
    """Without `fill_nan(None)`, polars's `.mean().over(key)`
    propagates NaN: a single NaN cell makes the whole partition's
    aggregate NaN. The helper uses `fill_nan(None)` so polars
    skips NaN like null."""
    df = pl.DataFrame({
        'q': [10.0, 20.0, float('nan'), 5.0, 5.0],
        'env': ['a', 'a', 'a', 'b', 'b'],
    })
    out = df.with_columns([
        partition_aggregate('q', by='env', op='mean').alias('env_mean'),
        # contrast: the naive expression we're avoiding
        pl.col('q').mean().over('env').alias('naive_mean'),
    ])
    # Group 'a' should have mean (10+20)/2 = 15 (NaN excluded);
    # naive should be NaN.
    a_rows = out.filter(pl.col('env') == 'a')
    assert a_rows['env_mean'][0] == pytest.approx(15.0)
    assert math.isnan(a_rows['naive_mean'][0])
    # Group 'b' has no NaN — both should be 5.
    b_rows = out.filter(pl.col('env') == 'b')
    assert b_rows['env_mean'][0] == pytest.approx(5.0)
    assert b_rows['naive_mean'][0] == pytest.approx(5.0)


def test_partition_aggregate_multi_key() -> None:
    df = pl.DataFrame({
        'q': [1.0, 2.0, 3.0, 4.0],
        'env': ['a', 'a', 'a', 'a'],
        'gen': [1, 1, 2, 2],
    })
    out = df.with_columns([
        partition_aggregate('q', by=['env', 'gen'], op='mean').alias('m'),
    ])
    # (a, 1) → mean(1,2)=1.5; (a, 2) → mean(3,4)=3.5
    gen1 = out.filter(pl.col('gen') == 1)
    gen2 = out.filter(pl.col('gen') == 2)
    assert all(v == pytest.approx(1.5) for v in gen1['m'].to_list())
    assert all(v == pytest.approx(3.5) for v in gen2['m'].to_list())


@pytest.mark.parametrize(
    ('op', 'expected'),
    [
        ('mean', 1.5),
        ('median', 1.5),
        ('min', 1.0),
        ('max', 2.0),
        ('sum', 3.0),
    ],
)
def test_partition_aggregate_ops(op: str, expected: float) -> None:
    df = pl.DataFrame({'q': [1.0, 2.0, float('nan')], 'env': ['a', 'a', 'a']})
    out = df.with_columns([
        partition_aggregate('q', by='env', op=op).alias('agg'),  # type: ignore[arg-type]
    ])
    assert out['agg'][0] == pytest.approx(expected)


def test_partition_aggregate_then_compare_chains_naturally() -> None:
    """The helper composes with `>` / `<` / `&` like any
    pl.Expr — bridge authors chain `partition_aggregate(...) > 1`
    inline."""
    df = pl.DataFrame({
        'q': [10.0, 20.0, 0.5, 0.5],
        'env': ['high', 'high', 'low', 'low'],
    })
    out = df.filter(partition_aggregate('q', by='env', op='mean') > 1.0)
    assert out.height == 2
    assert set(out['env'].to_list()) == {'high'}


def test_finite_accepts_pl_expr_not_just_string() -> None:
    """The `_Source` union accepts a column name OR a pl.Expr,
    so callers can wrap derived expressions."""
    df = pl.DataFrame({'a': [1.0, 2.0], 'b': [3.0, float('nan')]})
    derived = pl.col('a') + pl.col('b')
    out = df.filter(finite(derived))
    assert out.height == 1
    assert out['a'][0] == 1.0
