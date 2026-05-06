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


# ============ upstream conditioning ============

def _chain_cells(
    deltas_u: list[float],
    deltas_m: list[float],
    deltas_y: list[float],
) -> list[dict[str, object]]:
    """Construct chain cells with declared Δ_U / Δ_M / Δ_Y per pair.
    Baseline is at zero on all three; treatment carries the deltas."""
    cells: list[dict[str, object]] = []
    for i, (du, dm, dy) in enumerate(zip(deltas_u, deltas_m, deltas_y)):
        cells.append({
            'arm_key': 'baseline', 'seed': i,
            'upstream': 0.0, 'mediator': 0.0, 'target': 0.0,
        })
        cells.append({
            'arm_key': 'treatment', 'seed': i,
            'upstream': du, 'mediator': dm, 'target': dy,
        })
    return cells


def test_upstream_conditioning_max_delta_drops_dormant_pairs() -> None:
    """`upstream_max_delta=0.0` keeps only pairs where
    Δ_upstream < 0 (the upstream step fired in the predicted
    direction). Pairs where the upstream step was dormant
    (Δ_U ≥ 0) are dropped before mediation regression.

    Construction: 5 pairs with Δ_U < 0 (mech HELD; mediation
    chain active) where Δ_Y = 2·Δ_M (perfect mediation), plus 5
    pairs with Δ_U > 0 (mech REVERSED; the chain shouldn't apply
    but they have noise that would dilute pooled regression).
    Without conditioning, the pooled estimate mixes both regimes.
    With `upstream_max_delta=0.0`, only the 5 mech-HELD pairs
    contribute → proportion ≈ 1.0 cleanly recovered."""
    # 5 mech-HELD pairs (Δ_U < 0): perfect mediation chain.
    held_u = [-0.1, -0.2, -0.3, -0.4, -0.5]
    held_m = [0.1, 0.2, 0.3, 0.4, 0.5]
    held_y = [2 * m for m in held_m]
    # 5 mech-REVERSED pairs (Δ_U > 0): no real mediation; noise.
    reversed_u = [0.1, 0.2, 0.3, 0.4, 0.5]
    reversed_m = [0.05, -0.05, 0.0, 0.1, -0.1]    # decoupled from Y
    reversed_y = [-0.2, 0.5, 0.0, -0.4, 0.3]      # decoupled from M

    cells = _chain_cells(
        held_u + reversed_u,
        held_m + reversed_m,
        held_y + reversed_y,
    )
    # Without conditioning: pools both regimes → proportion drifts
    # away from the perfect-mediation 1.0.
    r_unconditioned = proportion_mediated.fn(
        cells,
        target='target', mediator='mediator',
        treatment_arm='treatment', baseline_arm='baseline',
        pair_by=('seed',),
    )
    assert r_unconditioned.n_pairs == 10

    # With upstream_max_delta=0.0: only the 5 mech-HELD pairs
    # contribute. Perfect mediation (Δ_Y = 2·Δ_M) → proportion ≈ 1.0.
    r_conditioned = proportion_mediated.fn(
        cells,
        target='target', mediator='mediator',
        treatment_arm='treatment', baseline_arm='baseline',
        pair_by=('seed',),
        upstream_source='upstream',
        upstream_max_delta=0.0,
    )
    assert r_conditioned.n_pairs == 5, (
        f'expected 5 mech-HELD pairs after conditioning; got '
        f'{r_conditioned.n_pairs}'
    )
    assert math.isclose(r_conditioned.proportion, 1.0, abs_tol=1e-9), (
        f'mech-HELD subset has perfect mediation (Δ_Y = 2·Δ_M); '
        f'proportion = {r_conditioned.proportion}, expected 1.0'
    )
    assert r_conditioned.in_unit_interval is True


def test_upstream_conditioning_min_delta_inverse_direction() -> None:
    """`upstream_min_delta=0.0` keeps only pairs where Δ_U > 0.
    Mirror of the max_delta case: tests the inverse-polarity
    regime (Q-amplification under DDQN's reversal — when the
    upstream step INCREASES). Same construction; the conditioning
    flips which subset is kept."""
    held_u = [-0.1, -0.2, -0.3, -0.4, -0.5]
    held_m = [0.1, 0.2, 0.3, 0.4, 0.5]
    held_y = [2 * m for m in held_m]
    reversed_u = [0.1, 0.2, 0.3, 0.4, 0.5]
    reversed_m = [0.1, 0.2, 0.3, 0.4, 0.5]   # same chain on this regime
    reversed_y = [3 * m for m in reversed_m] # but β=3 here, distinct slope

    cells = _chain_cells(
        held_u + reversed_u,
        held_m + reversed_m,
        held_y + reversed_y,
    )
    r = proportion_mediated.fn(
        cells,
        target='target', mediator='mediator',
        treatment_arm='treatment', baseline_arm='baseline',
        pair_by=('seed',),
        upstream_source='upstream',
        upstream_min_delta=0.0,
    )
    assert r.n_pairs == 5
    assert math.isclose(r.proportion, 1.0, abs_tol=1e-9)


def test_upstream_conditioning_no_op_when_thresholds_none() -> None:
    """**Backwards-compat negative control**: when both
    `upstream_max_delta` and `upstream_min_delta` are None,
    `upstream_source` is ignored (no filtering applied). All
    paired cells flow through. Confirms the new feature doesn't
    silently filter when only `upstream_source` is provided."""
    cells = _chain_cells(
        [-0.1, -0.2, -0.3, 0.1, 0.2, 0.3],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        [0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
    )
    r_no_thresh = proportion_mediated.fn(
        cells,
        target='target', mediator='mediator',
        treatment_arm='treatment', baseline_arm='baseline',
        pair_by=('seed',),
        upstream_source='upstream',
        # No threshold → no filter.
    )
    assert r_no_thresh.n_pairs == 6, (
        f'no-threshold case should keep all pairs; got '
        f'{r_no_thresh.n_pairs}'
    )


def test_upstream_conditioning_validates_mutual_exclusion() -> None:
    """Passing both `upstream_max_delta` and `upstream_min_delta`
    raises ValueError — the bridge author has to commit to one
    direction."""
    import pytest
    with pytest.raises(ValueError, match='at most one'):
        proportion_mediated.fn(
            _chain_cells([0.1, 0.2, 0.3], [0.4, 0.5, 0.6],
                         [0.7, 0.8, 0.9]),
            target='target', mediator='mediator',
            treatment_arm='treatment', baseline_arm='baseline',
            pair_by=('seed',),
            upstream_source='upstream',
            upstream_max_delta=0.0,
            upstream_min_delta=0.0,
        )


def test_upstream_threshold_without_source_raises() -> None:
    """`upstream_max_delta` / `upstream_min_delta` require
    `upstream_source`. Passing a threshold without naming the
    upstream variable raises — the framework refuses to silently
    no-op."""
    import pytest
    with pytest.raises(ValueError, match='require.*upstream_source'):
        proportion_mediated.fn(
            _chain_cells([0.1, 0.2, 0.3], [0.4, 0.5, 0.6],
                         [0.7, 0.8, 0.9]),
            target='target', mediator='mediator',
            treatment_arm='treatment', baseline_arm='baseline',
            pair_by=('seed',),
            upstream_max_delta=0.0,
            # no upstream_source — should raise
        )
