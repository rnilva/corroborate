"""Tests for `proportion_mediated` — linear-mediation
decomposition.

Coverage: synthetic ground-truth (perfect mediation, zero
mediation, broken assumptions / proportion outside [0,1]) +
edge cases (n_pairs < 3, zero variance in mediator)."""
from __future__ import annotations

import math

from corroborate.analyses.proportion_mediated import (
    ProportionMediatedResult, proportion_mediated,
)


def _cells(
    deltas_y: list[float], deltas_m: list[float],
    *, ts: float = 1.0, bs: float = 0.0,
) -> list[dict[str, object]]:
    """Build synthetic paired cells with declared Δ_Y and Δ_M.

    Treatment cell at seed s gets `(target=ts+Δ_Y_s, mediator=ts+Δ_M_s)`;
    baseline gets `(target=bs, mediator=bs)`. So
    Δ_Y_pair = (ts+Δ_Y_s) − bs and likewise for M; subtracting
    the constant base values, the per-pair Δ matches the inputs."""
    cells: list[dict[str, object]] = []
    for i, (dy, dm) in enumerate(zip(deltas_y, deltas_m)):
        cells.append({
            'arm_key': 'baseline', 'seed': i,
            'target': bs, 'mediator': bs,
        })
        cells.append({
            'arm_key': 'treatment', 'seed': i,
            'target': bs + dy, 'mediator': bs + dm,
        })
    return cells


def test_perfect_mediation() -> None:
    """When `Δ_Y = β · Δ_M` exactly (no direct effect), the
    proportion is 1.0."""
    deltas_m = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    beta = 2.0
    deltas_y = [beta * m for m in deltas_m]
    r = proportion_mediated.fn(
        _cells(deltas_y, deltas_m),
        target='target', mediator='mediator',
        treatment_arm='treatment', baseline_arm='baseline',
        pair_by=('seed',),
    )
    assert isinstance(r, ProportionMediatedResult)
    assert math.isclose(r.proportion, 1.0, abs_tol=1e-9)
    assert r.in_unit_interval is True


def test_zero_mediation_constant_mediator() -> None:
    """When Δ_M is constant across pairs (zero variance), the
    OLS slope is undefined; result is NaN."""
    deltas_m = [0.5] * 10
    deltas_y = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    r = proportion_mediated.fn(
        _cells(deltas_y, deltas_m),
        target='target', mediator='mediator',
        treatment_arm='treatment', baseline_arm='baseline',
        pair_by=('seed',),
    )
    assert math.isnan(r.proportion)
    assert math.isnan(r.slope_y_on_m)


def test_zero_mediation_uncorrelated() -> None:
    """When Δ_Y and Δ_M are uncorrelated (slope ≈ 0), the
    indirect effect is small and proportion ≈ 0."""
    # Δ_M alternates +1 / −1; Δ_Y is constant 0.5 — no linear
    # dependence between them.
    deltas_m = [+1.0, -1.0] * 10
    deltas_y = [0.5] * 20
    r = proportion_mediated.fn(
        _cells(deltas_y, deltas_m),
        target='target', mediator='mediator',
        treatment_arm='treatment', baseline_arm='baseline',
        pair_by=('seed',),
    )
    assert abs(r.proportion) < 0.01
    assert r.in_unit_interval is True
    assert math.isclose(r.total, 0.5, abs_tol=1e-9)


def test_mixed_partial_mediation() -> None:
    """Δ_Y = 0.5·Δ_M + 1.0 (constant direct + linear mediator).
    Mean(Δ_M) > 0 so indirect is positive. Total is direct +
    indirect; proportion = (0.5·mean_M) / (1.0 + 0.5·mean_M)."""
    deltas_m = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    deltas_y = [0.5 * m + 1.0 for m in deltas_m]
    mean_m = sum(deltas_m) / len(deltas_m)
    expected_proportion = (0.5 * mean_m) / (1.0 + 0.5 * mean_m)
    r = proportion_mediated.fn(
        _cells(deltas_y, deltas_m),
        target='target', mediator='mediator',
        treatment_arm='treatment', baseline_arm='baseline',
        pair_by=('seed',),
    )
    assert math.isclose(r.proportion, expected_proportion, abs_tol=1e-9)
    assert r.in_unit_interval is True


def test_proportion_outside_unit_interval_flags_diagnostic() -> None:
    """When the slope flips the indirect against the total
    (suppressor effect), proportion goes negative → in_unit_interval
    is False. Bridge author should see the diagnostic and not
    treat the value as a real share."""
    # Δ_M positive on average but Δ_Y is exactly the slope-times
    # mediator FLIPPED — the OLS slope of Y on M will be negative,
    # making indirect negative when mean_M is positive, so
    # proportion = negative / positive_total < 0.
    deltas_m = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # Δ_Y inversely tied to Δ_M (slope is negative), but with a
    # large positive intercept so total Δ_Y is positive.
    deltas_y = [-1.0 * m + 1.5 for m in deltas_m]
    r = proportion_mediated.fn(
        _cells(deltas_y, deltas_m),
        target='target', mediator='mediator',
        treatment_arm='treatment', baseline_arm='baseline',
        pair_by=('seed',),
    )
    assert r.proportion < 0.0
    assert r.in_unit_interval is False


def test_too_few_pairs_returns_nan() -> None:
    """n_pairs < 3 → NaN result; the mediation regression needs
    at least 3 pairs to estimate a slope honestly."""
    r = proportion_mediated.fn(
        _cells([0.1, 0.2], [0.5, 0.6]),
        target='target', mediator='mediator',
        treatment_arm='treatment', baseline_arm='baseline',
        pair_by=('seed',),
    )
    assert math.isnan(r.proportion)
    assert r.n_pairs == 2
