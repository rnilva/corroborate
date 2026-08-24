"""`partial_spearman` (unified) dispatch + contract tests.

The unified primitive subsumed 5 legacy Spearman primitives in
`analyses.spearman`. After the migration the legacy wrappers were
deleted; these tests assert that the unified primitive's per-k
dispatch and per-type granularity detection produce results
matching the underlying `graph.discovery` primitives directly:

  - k=0 marginal       → `stratified_spearman_rho`
  - k=1 single-Z       → `stratified_partial_spearman_rho`
  - k≥2 multi-Z        → `stratified_partial_spearman_rho_multi`

Per-row vs per-burst granularity is detected from input types
(`str` → per-row; `Measurable[..., NDArray]` → per-burst). The
per-burst form unfolds each row into n_bursts (x, y, z) rows
before pooling — verified here against a hand-built dataset
where the expected number of contributing observations is
explicitly computable.

Plus type-detection and empty-input contract tests.
"""
from __future__ import annotations

import math
import zlib
from collections.abc import Mapping

import numpy as np
import numpy.typing as npt
import pytest

from corroborate.analyses.spearman.partial_spearman import partial_spearman
from corroborate.graph.discovery import (
    stratified_partial_spearman_rho,
    stratified_partial_spearman_rho_multi,
    stratified_spearman_rho,
)
from corroborate.measurables import Measurable
from corroborate.data import to_dataframe


def _det_seed(*parts: object) -> int:
    return zlib.adler32(repr(parts).encode()) & 0xFFFF_FFFF


def _build_per_row_rows(
    *, n_per_env: int = 30, n_envs: int = 3, with_z: int = 0,
) -> list[Mapping[str, object]]:
    """LG-SCM-shaped per-row rows. Each env has a different
    coupling strength so per-stratum ρ varies; pooled Fisher-z
    integrates them. Optional Z columns are correlated with X
    so the partial-form test exercises non-trivial conditioning."""
    rng = np.random.default_rng(_det_seed('per_row', n_per_env, n_envs, with_z))
    rows: list[Mapping[str, object]] = []
    for env_idx in range(n_envs):
        env_name = f'env_{env_idx}'
        alpha = 0.5 + 0.2 * env_idx  # per-env coupling
        for seed in range(n_per_env):
            x_val = float(rng.normal(0.0, 1.0))
            y_val = float(alpha * x_val + rng.normal(0.0, 0.5))
            row: dict[str, object] = {
                'env_name': env_name,
                'seed': seed,
                'x_col': x_val,
                'y_col': y_val,
            }
            for k in range(with_z):
                z_val = float(0.3 * x_val + rng.normal(0.0, 1.0))
                row[f'z_col_{k}'] = z_val
            rows.append(row)
    return rows


def _per_burst_measurable(
    column_key: str,
) -> Measurable[Mapping[str, object], npt.NDArray[np.floating]]:
    from corroborate.measurables.reductions import from_key
    return from_key(column_key)


def _build_per_burst_rows(
    *, n_per_env: int = 30, n_envs: int = 3, n_bursts: int = 5,
    with_z: int = 0,
) -> list[Mapping[str, object]]:
    rng = np.random.default_rng(
        _det_seed('per_burst', n_per_env, n_envs, n_bursts, with_z),
    )
    rows: list[Mapping[str, object]] = []
    for env_idx in range(n_envs):
        env_name = f'env_{env_idx}'
        alpha = 0.5 + 0.2 * env_idx
        for seed in range(n_per_env):
            x_arr = rng.normal(0.0, 1.0, size=n_bursts)
            y_arr = alpha * x_arr + rng.normal(0.0, 0.5, size=n_bursts)
            row: dict[str, object] = {
                'env_name': env_name,
                'seed': seed,
                'x_col': x_arr.tolist(),
                'y_col': y_arr.tolist(),
            }
            for k in range(with_z):
                z_arr = 0.3 * x_arr + rng.normal(0.0, 1.0, size=n_bursts)
                row[f'z_col_{k}'] = z_arr.tolist()
            rows.append(row)
    return rows


def _rows_to_arrays_per_row(
    rows: list[Mapping[str, object]],
    *, x: str, y: str, conditioning: tuple[str, ...],
) -> tuple[
    npt.NDArray[np.float64], npt.NDArray[np.float64],
    npt.NDArray[np.float64], list[object],
]:
    """Replicate the unified primitive's per-row observation
    extraction so the graph.discovery reference call sees the
    same data. Used by the dispatch-equivalence tests."""
    xs: list[float] = []
    ys: list[float] = []
    zs: list[list[float]] = [[] for _ in conditioning]
    strata: list[object] = []
    for c in rows:
        xv = c[x]
        yv = c[y]
        assert isinstance(xv, (int, float))
        assert isinstance(yv, (int, float))
        xs.append(float(xv))
        ys.append(float(yv))
        for col, k in zip(zs, conditioning):
            v = c[k]
            assert isinstance(v, (int, float))
            col.append(float(v))
        sk = c['env_name']
        strata.append(sk)
    x_np = np.asarray(xs, dtype=np.float64)
    y_np = np.asarray(ys, dtype=np.float64)
    z_np = (
        np.column_stack([np.asarray(c, dtype=np.float64) for c in zs])
        if conditioning else np.empty((len(xs), 0), dtype=np.float64)
    )
    return x_np, y_np, z_np, strata


# ============ Dispatch: k=0 → marginal ============

def test_unified_k0_dispatches_to_marginal_path() -> None:
    """`conditioning=()` must produce the same (ρ, p) as a direct
    call to `graph.discovery.stratified_spearman_rho` on the same
    extracted observations. Bit-exact — both use the same
    Fisher-z pooling primitive."""
    rows = _build_per_row_rows(with_z=0)
    x_np, y_np, _, strata = _rows_to_arrays_per_row(
        rows, x='x_col', y='y_col', conditioning=(),
    )
    ref_rho, ref_p = stratified_spearman_rho(x_np, y_np, strata)
    unified = partial_spearman.fn(to_dataframe(rows), x='x_col', y='y_col')
    assert unified.rho_pooled == ref_rho
    assert unified.p_value == ref_p
    assert unified.granularity == 'per_row'
    assert unified.conditioning == ()


# ============ Dispatch: k=1 → closed-form single-Z ============

def test_unified_k1_dispatches_to_closed_form_single_z() -> None:
    """`conditioning=('z_col_0',)` must dispatch to the closed-
    form `stratified_partial_spearman_rho` (NOT to _multi at
    k=1) — both are correct under joint-normality but
    numerically distinct on finite samples (multi uses OLS-
    residual-on-ranked-Z, picking up tie-handling drift the
    closed form doesn't). Dispatching k=1 to the closed-form
    path was load-bearing for verdict stability across the
    migration."""
    rows = _build_per_row_rows(with_z=1)
    x_np, y_np, z_np, strata = _rows_to_arrays_per_row(
        rows, x='x_col', y='y_col', conditioning=('z_col_0',),
    )
    # z_np is (n, 1); single-Z primitive wants (n,)
    ref_rho, ref_p = stratified_partial_spearman_rho(
        x_np, y_np, z_np[:, 0], strata,
    )
    unified = partial_spearman.fn(
        to_dataframe(rows), x='x_col', y='y_col', conditioning=('z_col_0',),
    )
    assert unified.rho_pooled == ref_rho
    assert unified.p_value == ref_p


# ============ Dispatch: k≥2 → multi-Z OLS-residual ============

def test_unified_k2_dispatches_to_multi_z_path() -> None:
    """`conditioning=('z_col_0', 'z_col_1')` must use the multi-Z
    OLS-residual primitive — only one available at k≥2."""
    rows = _build_per_row_rows(with_z=2)
    x_np, y_np, z_np, strata = _rows_to_arrays_per_row(
        rows, x='x_col', y='y_col',
        conditioning=('z_col_0', 'z_col_1'),
    )
    ref_rho, ref_p = stratified_partial_spearman_rho_multi(
        x_np, y_np, z_np, strata,
    )
    unified = partial_spearman.fn(
        to_dataframe(rows), x='x_col', y='y_col',
        conditioning=('z_col_0', 'z_col_1'),
    )
    assert unified.rho_pooled == ref_rho
    assert unified.p_value == ref_p


# ============ Per-burst granularity ============

def test_per_burst_unfolds_rows_then_pools() -> None:
    """Per-burst granularity: each row unfolds to `n_bursts`
    observations. Verify n_obs_total = n_rows * n_bursts and
    the pooled ρ matches calling `stratified_spearman_rho` on
    the explicitly-unfolded observation set."""
    rows = _build_per_burst_rows(
        n_per_env=30, n_envs=3, n_bursts=5, with_z=0,
    )
    x_m = _per_burst_measurable('x_col')
    y_m = _per_burst_measurable('y_col')
    unified = partial_spearman.fn(to_dataframe(rows), x=x_m, y=y_m)
    assert unified.granularity == 'per_burst'
    # 3 envs × 30 seeds × 5 bursts = 450 observations
    assert unified.n_obs_total == 450
    # Unfold rows explicitly and verify ρ equals the marginal
    # primitive on the unfolded data.
    xs: list[float] = []
    ys: list[float] = []
    strata: list[object] = []
    for c in rows:
        x_vals = c['x_col']
        y_vals = c['y_col']
        assert isinstance(x_vals, list) and isinstance(y_vals, list)
        env = c['env_name']
        for xv, yv in zip(x_vals, y_vals):
            xs.append(float(xv))
            ys.append(float(yv))
            strata.append(env)
    ref_rho, _ = stratified_spearman_rho(
        np.asarray(xs), np.asarray(ys), strata,
    )
    assert unified.rho_pooled == ref_rho


# ============ Type detection ============

def test_mixed_str_measurable_raises_typeerror() -> None:
    """str x + Measurable y is incoherent: either both per-row
    (str) or both per-burst (Measurable). Silent coercion would
    flatten a per-burst array to a single number or broadcast a
    scalar across n_bursts — both wrong. The primitive raises
    TypeError naming the count of each kind."""
    rows = _build_per_row_rows(with_z=0)
    y_m = _per_burst_measurable('y_col')
    with pytest.raises(TypeError, match='must all be str.*OR all Measurable'):
        partial_spearman.fn(to_dataframe(rows), x='x_col', y=y_m)


def test_mixed_conditioning_raises_typeerror() -> None:
    """Same protection for the conditioning tuple: one str + one
    Measurable z is incoherent."""
    y_m = _per_burst_measurable('y_col')
    z_m = _per_burst_measurable('z_col_0')
    rows = _build_per_burst_rows(with_z=1)
    with pytest.raises(TypeError, match='must all be str.*OR all Measurable'):
        partial_spearman.fn(
            to_dataframe(rows), x=y_m, y=y_m, conditioning=('z_col_0', z_m),
        )


def test_empty_rows_returns_nan_result() -> None:
    """No rows → NaN ρ/p, zero n_obs/n_strata. Should not raise.
    The contract holds across granularities and conditioning
    shapes."""
    result = partial_spearman.fn(to_dataframe([]), x='x_col', y='y_col')
    assert math.isnan(result.rho_pooled)
    assert math.isnan(result.p_value)
    assert result.n_obs_total == 0
    assert result.n_strata == 0
