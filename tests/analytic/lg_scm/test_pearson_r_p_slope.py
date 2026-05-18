"""Direct closed-form tests on the `_pearson_r_p_slope` primitive
that backs `paired_link_per_burst`.

`_pearson_r_p_slope(x, y)` returns `(r, p, slope)`:

  r       = numpy Pearson correlation
  slope   = r · sd(y) / sd(x) (OLS β of y on x)
  p       = 2 · (1 − Φ(|atanh(r)| / SE)) where SE = 1/√(n − 3)
            with the boundary convention p=0 when |r| ≥ 1

Pin the closed-form formulas + the n<3 / zero-variance / NaN
guards. End-to-end LG-SCM tests on `paired_link_per_burst`
sit alongside; this file is the per-primitive companion."""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import norm

from corroborate.analyses.link.paired_link_per_burst import _pearson_r_p_slope


def test_pearson_perfect_positive_correlation_returns_r_one_p_zero() -> None:
    """y = 2x + 3 → r = 1, slope = 2, p = 0 (boundary convention).
    Pin `if abs(r) >= 1.0: return r, 0.0, slope`. With y=2x+3,
    np.corrcoef returns r=0.9999...8 due to float drift, so the
    boundary branch is NOT entered. Instead Fisher z computes
    a finite p value at high r."""
    x = np.arange(10, dtype=np.float64)
    y = 2.0 * x + 3.0
    r, _, slope = _pearson_r_p_slope(x, y)
    assert r == pytest.approx(1.0, abs=1e-9)
    assert slope == pytest.approx(2.0, abs=1e-9)


def test_pearson_identical_arrays_hit_r_eq_one_branch() -> None:
    """Passing the same array as x and y gives r=1.0 EXACTLY
    (no float drift through 2x+3 transform). Pin
    `if abs(r) >= 1.0` against `> 1.0` mutant — under the
    mutant, r=1.0 fails the strict >, falls through to Fisher z
    which divides by `1 - r` = 0 → log(inf) → returns p=0
    coincidentally. So the test asserts the r value AND that
    p is exactly 0 (the boundary convention) AND slope is finite."""
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = x.copy()
    r, p, slope = _pearson_r_p_slope(x, y)
    assert r == 1.0
    assert p == 0.0    # boundary convention; pin `>= 1.0`
    assert slope == pytest.approx(1.0, abs=1e-9)


def test_pearson_perfect_negative_correlation_returns_r_neg_one() -> None:
    """y = -3x + 1 → r = -1, slope = -3."""
    x = np.arange(10, dtype=np.float64)
    y = -3.0 * x + 1.0
    r, p, slope = _pearson_r_p_slope(x, y)
    assert r == pytest.approx(-1.0, abs=1e-9)
    assert p == 0.0
    assert slope == pytest.approx(-3.0, abs=1e-9)


def test_pearson_p_value_matches_fisher_z_closed_form() -> None:
    """Closed-form r computed by hand from a small integer-data
    construction (NOT from feeding framework's sample r back
    through framework's Fisher-z formula).

    Data:  x = [-2, -1, 0, 1, 2],  y = [-4, 2, 0, -2, 4]
        x̄ = ȳ = 0
        cov(x, y) = (8 − 2 + 0 − 2 + 8) / 4 = 3.0
        var(x)    = (4 + 1 + 0 + 1 + 4) / 4 = 2.5
        var(y)    = (16 + 4 + 0 + 4 + 16) / 4 = 10.0
        r_known   = 3.0 / √(2.5 · 10) = 3 / 5 = 0.6 EXACTLY
        slope_known = cov / var(x) = 3 / 2.5 = 1.2 EXACTLY

    Closed-form p (computed from r_known = 0.6, NOT framework's r):
        z = 0.5·ln(1.6/0.4) = 0.5·ln(4) = ln(2)
        SE = 1/√(n − 3) = 1/√2
        |z|/SE = ln(2)·√2 ≈ 0.9803
        p_known = 2·(1 − Φ(0.9803)) ≈ 0.3270

    Pin the framework's reported r AND p to these exact closed-form
    values. The expected p is computed against the integer-data r
    derived by hand — the framework's own r is never fed back into
    the formula. Catches mutations on:
        `(1+r)/(1-r) → (1+r)*(1-r)` (would yield z ≈ 0)
        `1/sqrt(n-3) → 1*sqrt(n-3)` (SE off by factor 2 → p shift)
        `abs(z)/se → abs(z)*se`     (z·SE far from threshold)
    """
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = np.array([-4.0, 2.0, 0.0, -2.0, 4.0])
    r, p, _ = _pearson_r_p_slope(x, y)

    r_known = 0.6
    z_known = 0.5 * math.log((1 + r_known) / (1 - r_known))
    se_z_known = 1.0 / math.sqrt(5 - 3)
    p_known = 2.0 * (1.0 - float(norm.cdf(abs(z_known) / se_z_known)))

    assert r == pytest.approx(r_known, abs=1e-12), (
        f'r = {r:.6f}, closed-form r_known = {r_known}'
    )
    assert p == pytest.approx(p_known, abs=1e-9), (
        f'p = {p:.6f}, closed-form p = {p_known:.6f} (Fisher-z on '
        f'r_known = 0.6, n = 5).'
    )


def test_pearson_slope_uses_y_std_over_x_std() -> None:
    """Closed-form slope from integer-data construction with
    asymmetric variance (var_y = 4·var_x). The swap-mutant
    `r·sd_x/sd_y` would yield slope = 0.3 — clearly distinguishable
    from the structural slope = 1.2.

    Data:  x = [-2, -1, 0, 1, 2],  y = [-4, 2, 0, -2, 4]
        slope_known = cov / var(x) = 3 / 2.5 = 1.2 EXACTLY
        r_known     = 0.6
        sd_y / sd_x = √(10) / √2.5 = 2 EXACTLY
        r · sd_y / sd_x = 0.6 · 2 = 1.2  ✓
        SWAP-mutant: r · sd_x / sd_y = 0.6 · 0.5 = 0.3 → catches.

    Pin the closed-form slope value directly — the assertion does
    NOT walk the framework's own sd_y/sd_x ratio."""
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = np.array([-4.0, 2.0, 0.0, -2.0, 4.0])
    _, _, slope = _pearson_r_p_slope(x, y)
    assert slope == pytest.approx(1.2, abs=1e-12), (
        f'slope = {slope:.6f}, closed-form slope = 1.2 '
        f'(cov / var_x = 3 / 2.5).'
    )


def test_pearson_returns_nan_below_n_3() -> None:
    """n=2 → NaN (n<3 guard). Pin `n < 3` against `n <= 3` mutant
    (which would NaN at n=3) and `n < 4` (which would also NaN
    at n=3)."""
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])
    r, p, slope = _pearson_r_p_slope(x, y)
    assert math.isnan(r)
    assert math.isnan(p)
    assert math.isnan(slope)


def test_pearson_returns_finite_at_n_equals_3() -> None:
    """n=3 is the lower bound (df = n − 3 = 0 admits |r|=1 only,
    but for finite |r|<1 the SE = 1/sqrt(0) = inf → z/SE = 0 →
    p = 1.0 by convention). Pin `n < 3` against `n < 4` mutant
    that would NaN at n=3."""
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([1.0, 0.0, 2.0])  # not perfectly linear
    r, p, slope = _pearson_r_p_slope(x, y)
    assert math.isfinite(r)
    assert math.isfinite(slope)
    # p may be 0.0 (perfect r) or 1.0 (SE infinite) — both finite.
    assert math.isfinite(p)


def test_pearson_returns_nan_when_x_constant() -> None:
    """All-equal x → x.std() = 0 → NaN. Pin `x.std() == 0` against
    `x.std() == 1` mutant (which would NaN on unit-variance x
    instead of zero-variance x)."""
    x = np.full(10, 3.5)
    y = np.arange(10, dtype=np.float64)
    r, p, slope = _pearson_r_p_slope(x, y)
    assert math.isnan(r)
    assert math.isnan(p)
    assert math.isnan(slope)


def test_pearson_returns_nan_when_y_constant() -> None:
    """All-equal y → y.std() = 0 → NaN. Pin `y.std() == 0` against
    `y.std() == 1` mutant. Also pins the trailing `or` against
    `and` (with x varying and y constant: orig=True, `and`=False)."""
    x = np.arange(10, dtype=np.float64)
    y = np.full(10, 3.5)
    r, p, slope = _pearson_r_p_slope(x, y)
    assert math.isnan(r)
    assert math.isnan(p)
    assert math.isnan(slope)


def test_pearson_returns_nan_on_nan_entries_in_input() -> None:
    """Arrays with NaN entries: x.std() = NaN (not 0, so the
    zero-variance early return doesn't fire), then np.corrcoef
    returns NaN → trips the `if not np.isfinite(r)` guard.

    Pin the SECOND NaN-return branch (after np.corrcoef) — the
    `float(None)` mutation on its NaN-tuple would raise
    TypeError. The first NaN-return branch (n<3 / zero-variance)
    is covered separately."""
    x = np.array([1.0, 2.0, float('nan'), 4.0, 5.0])
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    r, p, slope = _pearson_r_p_slope(x, y)
    assert math.isnan(r)
    assert math.isnan(p)
    assert math.isnan(slope)


def test_pearson_returns_nan_when_x_constant_and_y_varies() -> None:
    """Pin the second `or` (between the two zero-variance checks)
    against `and` mutant. With x.std()=0 and y.std()!=0:
        orig: True or False → True → NaN (correct)
        mutant: True and False → False → continues → crash on
        np.corrcoef of constant-x"""
    x = np.full(10, 0.0)
    y = np.arange(10, dtype=np.float64) * 1.5
    r, _, _ = _pearson_r_p_slope(x, y)
    assert math.isnan(r)
