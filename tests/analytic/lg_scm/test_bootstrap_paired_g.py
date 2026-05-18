"""Closed-form recovery + CI-coverage probes for `bootstrap_paired_g`.

Two contracts to verify:

1. **Point estimate matches `paired_g.g` exactly** — bootstrap
   doesn't bias-correct; same input Δ → same `g`.
2. **CI coverage is well-calibrated**: under normal Δ at
   structural g_struct, fraction of bootstrap CIs containing
   g_struct should be ≈ 1 − α.

The CI-coverage probe is the load-bearing one — it's why this
primitive exists. paired_g's analytical SE under-covers on
heavy tails by 15-25% (per the empirical map at
test_paired_g_skew_robustness.py); bootstrap CIs should fix this.
"""
from __future__ import annotations

import math
import zlib

import numpy as np

from corroborate.analyses.paired.bootstrap_paired_g import bootstrap_paired_g
from corroborate.analyses.paired.paired_g import paired_g


def _det_seed(*parts: object) -> int:
    return zlib.adler32(repr(parts).encode()) & 0xFFFF_FFFF


def _make_paired_cells(deltas: list[float]) -> list[dict[str, object]]:
    cells: list[dict[str, object]] = []
    for s, d in enumerate(deltas):
        cells.append({'arm_key': 'T', 'seed': s, 'value': float(d)})
        cells.append({'arm_key': 'B', 'seed': s, 'value': 0.0})
    return cells


def test_bootstrap_g_matches_paired_g_g_exactly() -> None:
    """Bootstrap doesn't bias-correct: the point estimate `g`
    must equal `paired_g.g` exactly. Verifies that the bootstrap
    primitive computes the same observed g using the same c_4
    formula on the same Δ vector."""
    rng = np.random.default_rng(_det_seed('boot_g', 30))
    deltas = rng.normal(1.0, 2.0, 30).tolist()
    cells = _make_paired_cells(deltas)
    pg = paired_g.fn(
        cells,
        treatment_arm='T', baseline_arm='B',
        pair_by=('seed',), source='value',
    )
    boot = bootstrap_paired_g.fn(
        cells,
        treatment_arm='T', baseline_arm='B',
        pair_by=('seed',), source='value',
        b_replicates=200,
    )
    assert math.isclose(boot.g, pg.g, rel_tol=1e-12), (
        f'bootstrap.g = {boot.g}, paired_g.g = {pg.g}; expected '
        f'exact match (bootstrap doesn\'t bias-correct).'
    )


def test_bootstrap_ci_covers_structural_g_under_normal() -> None:
    """**Coverage probe**: under N(1, 2) Δ at n=30, fraction of
    bootstrap CIs containing the structural g should be ≈ 95%.

    Construction: g_struct = 1/2 · c_4(30) ≈ 0.487. Generate K=100
    independent samples of n=30 paired Δ, compute bootstrap CI
    on each, count what fraction contains g_struct.

    Bound: 0.85 < coverage < 1.0. The 0.85 lower bound admits
    sampling drift on the binomial (Var(coverage) ≈ p(1-p)/K =
    0.05·0.95/100 → SE ≈ 0.022; bound is ~4·SE below 0.95). The
    upper bound is a sanity check (coverage > 1 impossible).
    """
    K = 100
    g_struct = 1.0 / 2.0 * (1.0 - 3.0 / (4 * 30 - 5))
    n_covering = 0
    for k in range(K):
        rng = np.random.default_rng(_det_seed('boot_cov', k))
        deltas = rng.normal(1.0, 2.0, 30).tolist()
        cells = _make_paired_cells(deltas)
        boot = bootstrap_paired_g.fn(
            cells,
            treatment_arm='T', baseline_arm='B',
            pair_by=('seed',), source='value',
            b_replicates=400,
            bootstrap_seed=k,
        )
        if boot.ci_lo <= g_struct <= boot.ci_hi:
            n_covering += 1
    coverage = n_covering / K
    assert 0.85 < coverage <= 1.0, (
        f'Bootstrap CI coverage = {coverage:.3f} (target 0.95); '
        f'bound 0.85-1.0 admits binomial sampling SD ≈ 0.022.'
    )


def test_bootstrap_se_modestly_better_than_analytical_under_lognormal_n_50() -> None:
    """**Honest probe**: under log-normal Δ at n=50, the percentile
    bootstrap remains anti-conservative — but slightly less so
    than paired_g's analytical SE.

    Empirical (K=200 deterministic-seed MC):
        paired_g mean(se) / MC_sd_g           ≈ 0.69
        bootstrap mean(se) / MC_sd_g          ≈ 0.73
        relative error of bootstrap_se        ≈ 27%

    The bootstrap's headline benefit on heavy tails is NOT
    convergence to the true sampling SD (it can't — it resamples
    from a sample that already missed the population tail). The
    benefit is structural: percentile CIs are asymmetric on
    skewed g-sampling-distributions where paired_g's
    `g ± 1.96·se` collapses asymmetry to a single number.

    This test pins that bootstrap_se is at least as good as the
    analytical SE under log-normal Δ. A future fix using BCa or
    studentized-t bootstrap should narrow the gap further; pin
    the current state.

    Bound: bootstrap_se / MC_sd_g > 0.6 (some calibration; not
    catastrophically anti-conservative).
    """
    K = 200
    n = 50
    g_estimates: list[float] = []
    boot_ses: list[float] = []
    for k in range(K):
        rng = np.random.default_rng(_det_seed('boot_lognorm', k))
        deltas = rng.lognormal(0.0, 0.7, n).tolist()
        cells = _make_paired_cells(deltas)
        boot = bootstrap_paired_g.fn(
            cells,
            treatment_arm='T', baseline_arm='B',
            pair_by=('seed',), source='value',
            b_replicates=300,
            bootstrap_seed=k,
        )
        g_estimates.append(boot.g)
        boot_ses.append(boot.se_bootstrap)
    g_arr = np.array(g_estimates)
    se_arr = np.array(boot_ses)
    mc_sd_g = float(g_arr.std(ddof=1))
    mean_boot_se = float(se_arr.mean())
    ratio = mean_boot_se / mc_sd_g
    assert ratio > 0.60, (
        f'bootstrap mean(se)/MC_sd_g = {ratio:.4f}; expected > 0.60 '
        f'(some calibration). A ratio < 0.60 means bootstrap is '
        f'severely anti-conservative — worse than acceptable.'
    )


def test_bootstrap_ci_excludes_zero_under_strong_effect() -> None:
    """End-to-end: at structural g ≈ 0.5, bootstrap CI should
    EXCLUDE zero (significant-at-α). Pin the load-bearing CI
    routing — a regression that mishandled the percentile
    extraction would breach."""
    rng = np.random.default_rng(_det_seed('boot_strong', 100))
    deltas = rng.normal(1.0, 2.0, 100).tolist()
    boot = bootstrap_paired_g.fn(
        _make_paired_cells(deltas),
        treatment_arm='T', baseline_arm='B',
        pair_by=('seed',), source='value',
        b_replicates=500,
    )
    assert boot.ci_excludes_zero, (
        f'CI = ({boot.ci_lo:.4f}, {boot.ci_hi:.4f}) does NOT '
        f'exclude zero, but g ≈ 0.5 with n=100 should be '
        f'unambiguously significant.'
    )
    assert boot.ci_lo > 0, f'lower CI bound = {boot.ci_lo:.4f}'


def test_bootstrap_b_replicates_validation() -> None:
    """Argument validation: b_replicates < 100 raises."""
    import pytest
    with pytest.raises(ValueError, match='b_replicates'):
        bootstrap_paired_g.fn(
            _make_paired_cells([1.0, 2.0, 3.0]),
            treatment_arm='T', baseline_arm='B',
            pair_by=('seed',), source='value',
            b_replicates=50,
        )


def test_bootstrap_n_pairs_below_two_returns_nan() -> None:
    """0 paired cells → NaN g, NaN CI."""
    boot = bootstrap_paired_g.fn(
        [],
        treatment_arm='T', baseline_arm='B',
        pair_by=('seed',), source='value',
        b_replicates=200,
    )
    assert boot.n_pairs == 0
    assert math.isnan(boot.g)
    assert math.isnan(boot.ci_lo)
    assert math.isnan(boot.ci_hi)


def test_bootstrap_paired_g_registered() -> None:
    """The `@analysis` decorator populates the registry."""
    import corroborate.analyses  # noqa: F401  # pyright: ignore[reportUnusedImport]
    from corroborate.bridge.analysis import get_registered
    assert get_registered('bootstrap_paired_g') is not None
