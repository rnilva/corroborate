"""Tests for `corroborate.data.kernel.per_stratum_aggregate`.

Phase 2.5 — shared kernel that backs both Panel.derive and
(future) Panel-input variants of canonical analyses. The kernel
takes a polars DataFrame + structured spec and returns
{stratum_id: aggregate}. Tests cover:

- The three aggregators (mean / std / median).
- `cell_filter` narrowing.
- `min_n` floor (skip strata too small).
- Empty-input early returns.
- The Panel.derive ↔ kernel delegation: identical outputs on
  identical inputs (semantic invariant).
"""
from __future__ import annotations

import math

import polars as pl

from corroborate.data import DerivedSpec, Panel
from corroborate.data.kernel import (
    cells_to_dataframe,
    per_stratum_aggregate,
)


def _make_cells() -> pl.DataFrame:
    """3 envs × 2 arms × 3 seeds = 18 cells; column `v` with
    one NaN per (env, arm) stratum (so each stratum has 2
    finite + 1 NaN value)."""
    return pl.DataFrame({
        'env_name': ['env1', 'env1', 'env1', 'env1', 'env1', 'env1',
                     'env2', 'env2', 'env2', 'env2', 'env2', 'env2',
                     'env3', 'env3', 'env3', 'env3', 'env3', 'env3'],
        'arm_key': ['baseline', 'baseline', 'baseline',
                    'ddqn', 'ddqn', 'ddqn'] * 3,
        'seed': [0, 1, 2] * 6,
        'v': [
            1.0, 2.0, float('nan'),
            5.0, 6.0, float('nan'),
            10.0, 20.0, float('nan'),
            50.0, 60.0, float('nan'),
            100.0, 200.0, float('nan'),
            500.0, 600.0, float('nan'),
        ],
    })


def test_per_stratum_aggregate_mean() -> None:
    """Per-stratum mean drops NaN cells before aggregation;
    each stratum has 2 finite values."""
    out = per_stratum_aggregate(
        _make_cells(),
        column='v',
        aggregator='mean',
        stratify_by=('env_name', 'arm_key'),
    )
    assert math.isclose(out[('env1', 'baseline')], 1.5)
    assert math.isclose(out[('env1', 'ddqn')], 5.5)
    assert math.isclose(out[('env3', 'ddqn')], 550.0)


def test_per_stratum_aggregate_std_sample_ddof() -> None:
    """Per-stratum SD uses ddof=1 (sample SD). With values
    [1.0, 2.0], ddof=1 SD = sqrt(0.5)."""
    out = per_stratum_aggregate(
        _make_cells(),
        column='v',
        aggregator='std',
        stratify_by=('env_name', 'arm_key'),
        min_n=2,
    )
    expected = math.sqrt(0.5)
    assert math.isclose(out[('env1', 'baseline')], expected, abs_tol=1e-9)


def test_per_stratum_aggregate_median() -> None:
    """Median of [1.0, 2.0] = 1.5."""
    out = per_stratum_aggregate(
        _make_cells(),
        column='v',
        aggregator='median',
        stratify_by=('env_name', 'arm_key'),
    )
    assert math.isclose(out[('env1', 'baseline')], 1.5)


def test_per_stratum_aggregate_cell_filter() -> None:
    """`cell_filter` narrows BEFORE aggregation. Filter to
    baseline arm only — DDQN strata vanish."""
    out = per_stratum_aggregate(
        _make_cells(),
        column='v',
        aggregator='mean',
        stratify_by=('env_name', 'arm_key'),
        cell_filter=pl.col('arm_key') == 'baseline',
    )
    assert set(out.keys()) == {
        ('env1', 'baseline'),
        ('env2', 'baseline'),
        ('env3', 'baseline'),
    }


def test_per_stratum_aggregate_min_n_floors() -> None:
    """`min_n` skips strata with too few surviving cells.
    With NaN drop, each stratum has 2 finite values; `min_n=3`
    skips all."""
    out = per_stratum_aggregate(
        _make_cells(),
        column='v',
        aggregator='mean',
        stratify_by=('env_name', 'arm_key'),
        min_n=3,
    )
    assert out == {}


def test_per_stratum_aggregate_empty_cells_returns_empty() -> None:
    """Empty cells DataFrame → empty output, no error."""
    out = per_stratum_aggregate(
        pl.DataFrame(),
        column='v',
        aggregator='mean',
        stratify_by=('env_name', 'arm_key'),
    )
    assert out == {}


def test_per_stratum_aggregate_missing_column_returns_empty() -> None:
    """Column not present in cells → empty output."""
    out = per_stratum_aggregate(
        _make_cells(),
        column='nonexistent_col',
        aggregator='mean',
        stratify_by=('env_name', 'arm_key'),
    )
    assert out == {}


def test_panel_derive_delegates_to_kernel() -> None:
    """Panel.derive(spec) MUST produce the same output as
    per_stratum_aggregate(panel.cells, ...) — the kernel is
    the single source of truth. If this test ever diverges,
    the delegation broke."""
    cells = _make_cells()
    panel = Panel.from_dataframe(cells, stratify_by=('env_name', 'arm_key'))
    spec = DerivedSpec(column='v', aggregator='mean')
    via_panel = panel.derive(spec)
    via_kernel = per_stratum_aggregate(
        cells,
        column='v',
        aggregator='mean',
        stratify_by=('env_name', 'arm_key'),
        cell_filter=None,
        min_n=spec.effective_min_n,
    )
    assert dict(via_panel) == dict(via_kernel)


def test_cells_to_dataframe_passthrough() -> None:
    """When passed a DataFrame, return as-is (zero-copy)."""
    df = _make_cells()
    out = cells_to_dataframe(df)
    assert out is df


def test_cells_to_dataframe_from_iterable_of_mappings() -> None:
    """Convert a list of dict-rows to DataFrame."""
    rows = [
        {'env_name': 'env1', 'arm_key': 'baseline', 'v': 1.0},
        {'env_name': 'env1', 'arm_key': 'ddqn', 'v': 2.0},
    ]
    out = cells_to_dataframe(rows)
    assert out.height == 2
    assert set(out.columns) >= {'env_name', 'arm_key', 'v'}


def test_cells_to_dataframe_empty_returns_empty() -> None:
    """Empty iterable → empty DataFrame."""
    out = cells_to_dataframe([])
    assert out.height == 0
