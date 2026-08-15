"""Closed-form recovery of Cliff's δ on the LG-SCM-style synthetic
implementation — validates the new `cliff_delta_paired` primitive.

Cliff's δ measures `P(Δ > 0) - P(Δ < 0)`. Under Δ ~ N(μ, σ²):

    δ_population = 2·Φ(μ/σ) - 1

where Φ is the standard normal CDF. For (μ=1, σ=2): μ/σ=0.5,
δ_pop = 2·Φ(0.5) - 1 ≈ 0.383.

Sample Cliff's δ has analytical SE ≈ √((1 - δ²) / (n - 1)) under
Cliff (1996). The framework's reported δ and SE should match these
within sampling-distribution-derived bounds.
"""
from __future__ import annotations

import math
import zlib

import numpy as np
from scipy.stats import norm

from corroborate.analyses.paired.cliff_delta_paired import cliff_delta_paired


def _det_seed(*parts: object) -> int:
    return zlib.adler32(repr(parts).encode()) & 0xFFFF_FFFF


_MU, _SIGMA = 1.0, 2.0
_DELTA_POP = 2.0 * float(norm.cdf(_MU / _SIGMA)) - 1.0   # ≈ 0.3829


def _make_paired_cells(deltas: list[float]) -> list[dict[str, object]]:
    cells: list[dict[str, object]] = []
    for s, d in enumerate(deltas):
        cells.append({'arm_key': 'T', 'seed': s, 'value': float(d)})
        cells.append({'arm_key': 'B', 'seed': s, 'value': 0.0})
    return cells


def test_cliff_delta_recovers_population_value_under_normal_delta() -> None:
    """At n=200, sample δ should be within sampling SE of
    population δ ≈ 0.383.

    Closed-form SE under H_0 (δ_pop=0): 1/√n.
    Closed-form SE under H_1: √((1-δ²)/(n-1)).
    At δ=0.383, n=200: SE ≈ √((1-0.147)/199) ≈ 0.066.
    Bound: |sample δ - δ_pop| < 4·SE ≈ 0.26.
    """
    rng = np.random.default_rng(_det_seed('cliff_recover', 200))
    deltas = rng.normal(_MU, _SIGMA, 200).tolist()
    result = cliff_delta_paired.fn(
        _make_paired_cells(deltas),
        treatment_arm='T', baseline_arm='B',
        pair_by=('seed',), source='value',
    )
    se_struct = math.sqrt((1 - _DELTA_POP ** 2) / 199)
    assert abs(result.delta - _DELTA_POP) < 4 * se_struct, (
        f'sample δ = {result.delta:.4f}, population δ = '
        f'{_DELTA_POP:.4f}, 4·SE = {4*se_struct:.4f}.'
    )


def test_cliff_delta_se_matches_closed_form_under_normal_delta() -> None:
    """Framework's reported SE should match `√((1-δ²)/(n-1))`
    within float-noise."""
    rng = np.random.default_rng(_det_seed('cliff_se', 200))
    deltas = rng.normal(_MU, _SIGMA, 200).tolist()
    result = cliff_delta_paired.fn(
        _make_paired_cells(deltas),
        treatment_arm='T', baseline_arm='B',
        pair_by=('seed',), source='value',
    )
    se_expected = math.sqrt((1 - result.delta ** 2) / (result.n_pairs - 1))
    assert abs(result.se - se_expected) < 1e-9, (
        f'reported SE = {result.se:.6f}, closed-form = '
        f'{se_expected:.6f}.'
    )


def test_cliff_delta_negative_when_treatment_worse() -> None:
    """Sign convention: when treatment is structurally WORSE than
    baseline (μ_Δ < 0), δ < 0. Pin direction at the construction
    level — a regression that flipped n_positive/n_negative would
    breach."""
    rng = np.random.default_rng(_det_seed('cliff_neg', 100))
    deltas = rng.normal(-1.0, 2.0, 100).tolist()
    result = cliff_delta_paired.fn(
        _make_paired_cells(deltas),
        treatment_arm='T', baseline_arm='B',
        pair_by=('seed',), source='value',
    )
    # Closed-form δ_pop = 2·Φ(-0.5) - 1 ≈ -0.383
    expected_pop = 2.0 * float(norm.cdf(-0.5)) - 1.0
    assert result.delta < 0, (
        f'expected negative δ when treatment worse; got {result.delta:.4f}'
    )
    se = math.sqrt((1 - expected_pop ** 2) / 99)
    assert abs(result.delta - expected_pop) < 4 * se


def test_cliff_delta_extreme_when_all_pairs_helped() -> None:
    """When EVERY pair has Δ > 0, δ = 1 (extreme positive). Pin
    the boundary case: n_positive = n_pairs, n_negative = 0,
    δ = 1.0 exactly."""
    deltas = [1.0, 0.5, 2.0, 1.5, 0.1]   # all positive
    result = cliff_delta_paired.fn(
        _make_paired_cells(deltas),
        treatment_arm='T', baseline_arm='B',
        pair_by=('seed',), source='value',
    )
    assert result.delta == 1.0, f'all-positive Δ should give δ=1.0; got {result.delta}'
    assert result.n_positive == 5
    assert result.n_negative == 0
    assert result.n_tied == 0
    # SE = √((1-1)/(n-1)) = 0 at extreme δ.
    assert result.se == 0.0


def test_cliff_delta_zero_under_symmetric_delta() -> None:
    """Under symmetric Δ centered at 0 (δ_pop = 0), sample δ ≈ 0
    within sampling SE = 1/√n. At n=200: SE = 0.071, bound 4·SE
    = 0.28."""
    rng = np.random.default_rng(_det_seed('cliff_zero', 200))
    # Mean-zero log-normal: subtract median.
    raw = rng.lognormal(0.0, 0.7, 200)
    median = math.exp(0.0)   # log-normal median for μ_log=0
    deltas = (raw - median).tolist()
    result = cliff_delta_paired.fn(
        _make_paired_cells(deltas),
        treatment_arm='T', baseline_arm='B',
        pair_by=('seed',), source='value',
    )
    se_h0 = 1.0 / math.sqrt(200)
    assert abs(result.delta) < 4 * se_h0, (
        f'symmetric (median-shifted) Δ should give δ ≈ 0; got '
        f'{result.delta:.4f}, 4·SE = {4*se_h0:.4f}.'
    )


def test_cliff_delta_n_pairs_zero_returns_nan() -> None:
    """Boundary: 0 paired cells → NaN delta + NaN SE. Pins the
    n_pairs < 2 NaN guard."""
    result = cliff_delta_paired.fn(
        [],
        treatment_arm='T', baseline_arm='B',
        pair_by=('seed',), source='value',
    )
    assert result.n_pairs == 0
    assert math.isnan(result.delta)
    assert math.isnan(result.se)


def test_cliff_delta_registered_in_analysis_registry() -> None:
    """Importing `corroborate.analyses` populates the registry
    with `cliff_delta_paired`. Bridges that declare a fixture
    parameter named `cliff_delta_paired` resolve to this analysis."""
    import corroborate.analyses  # noqa: F401  # pyright: ignore[reportUnusedImport]
    from corroborate.bridge.analysis import get_registered
    assert get_registered('cliff_delta_paired') is not None
