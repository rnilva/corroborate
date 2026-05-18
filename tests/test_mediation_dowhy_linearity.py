"""`mediation_dowhy` LinearityStatus classifier + integration tests.

The salvaged `mediation_dowhy` exposes a typed `linearity_status:
LinearityStatus` field on its result that surfaces the v10
failure-mode taxonomy (CASE_STUDY_LESSONS §2.11, reproduced on
the FR γ-WHY corpus per the module docstring).

The classifier is a pure function of (total_ate, direct_ate,
indirect_proportion, identified, eps); these closed-form tests
pin each branch's semantics against constructed inputs. Plus
one integration test that runs the full primitive on a
synthetic dataset where the underlying linear structure makes
the expected status deterministic.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pytest

from corroborate.analyses.dowhy.mediation_dowhy import (
    LinearityStatus, _classify_linearity, mediation_dowhy,
)


# ============ Classifier branches ============

def test_unidentified_when_dag_admits_no_backdoor() -> None:
    """`identified=False` → UNIDENTIFIED regardless of magnitudes
    (NaN total / direct typically accompany unidentified)."""
    s = _classify_linearity(
        total_ate=float('nan'), direct_ate=float('nan'),
        indirect_proportion=float('nan'),
        identified=False, eps=1e-8,
    )
    assert s == LinearityStatus.UNIDENTIFIED


def test_power_insufficient_when_total_near_zero() -> None:
    """`|total| < eps` → POWER_INSUFFICIENT. The decomposition
    can't surface a meaningful proportion when total is at
    noise floor."""
    s = _classify_linearity(
        total_ate=1e-12, direct_ate=0.5,
        indirect_proportion=float('nan'),
        identified=True, eps=1e-8,
    )
    assert s == LinearityStatus.POWER_INSUFFICIENT


def test_power_insufficient_when_direct_is_nan() -> None:
    """OLS rank-deficiency → direct_ate NaN; classifier maps to
    POWER_INSUFFICIENT (the Stage-2 estimation failed)."""
    s = _classify_linearity(
        total_ate=10.0, direct_ate=float('nan'),
        indirect_proportion=float('nan'),
        identified=True, eps=1e-8,
    )
    assert s == LinearityStatus.POWER_INSUFFICIENT


def test_sign_flipped_when_direct_opposite_sign_to_total() -> None:
    """Classic multicollinearity artifact: direct/total opposite
    signs. The FR γ-WHY documented case (total=+1023, direct=−57)
    lands here. SIGN_FLIPPED is the load-bearing flag — bridges
    consuming a mediation magnitude under this status are NOT
    trusted (per CLAUDE.md mediation recipe)."""
    s = _classify_linearity(
        total_ate=+1023.36, direct_ate=-57.31,
        indirect_proportion=+1.056,
        identified=True, eps=1e-8,
    )
    assert s == LinearityStatus.SIGN_FLIPPED


def test_sign_flipped_takes_precedence_over_out_of_bounds() -> None:
    """When BOTH conditions are true (FR γ-WHY: sign flipped AND
    proportion outside [0, 1]), SIGN_FLIPPED is the more
    diagnostic label — it names the load-bearing failure mode
    rather than the symptom."""
    s = _classify_linearity(
        total_ate=+10.0, direct_ate=-3.0,
        indirect_proportion=+1.3,
        identified=True, eps=1e-8,
    )
    assert s == LinearityStatus.SIGN_FLIPPED


def test_out_of_bounds_when_proportion_above_one_same_sign() -> None:
    """Suppression case: direct/total same sign, but the
    mediator's transmitted effect overshoots so |direct| > |total|
    in the same direction → indirect/total > 1. Linear
    decomposition still gives a coherent same-sign answer but
    falls outside the [0, 1] envelope."""
    s = _classify_linearity(
        total_ate=+10.0, direct_ate=+1.0,
        indirect_proportion=+0.9,  # in-range; sanity check
        identified=True, eps=1e-8,
    )
    assert s == LinearityStatus.RELIABLE
    # Now overshoot
    s = _classify_linearity(
        total_ate=+10.0, direct_ate=+1.0,
        indirect_proportion=+1.3,
        identified=True, eps=1e-8,
    )
    assert s == LinearityStatus.OUT_OF_BOUNDS


def test_out_of_bounds_when_proportion_below_zero_same_sign() -> None:
    """Suppression in the other direction: direct > total in
    the same sign → indirect = total − direct has OPPOSITE sign
    → proportion < 0. Direct overshoots the total in same-sign
    direction. Same as above flipped."""
    s = _classify_linearity(
        total_ate=+10.0, direct_ate=+15.0,
        indirect_proportion=-0.5,
        identified=True, eps=1e-8,
    )
    assert s == LinearityStatus.OUT_OF_BOUNDS


def test_reliable_when_full_partial_mediation_coherent() -> None:
    """Both mediated and direct paths contribute, same sign,
    proportion in [0, 1]. Classical Baron-Kenny scenario; linear
    decomposition gives a coherent reading."""
    s = _classify_linearity(
        total_ate=+10.0, direct_ate=+4.0,
        indirect_proportion=+0.6,
        identified=True, eps=1e-8,
    )
    assert s == LinearityStatus.RELIABLE


def test_reliable_at_zero_direct_full_mediation() -> None:
    """Full mediation: direct=0 → indirect=total → proportion=1.
    The degenerate `sign_direct == 0` path lands in RELIABLE
    because the linear decomposition is internally consistent."""
    s = _classify_linearity(
        total_ate=+10.0, direct_ate=0.0,
        indirect_proportion=+1.0,
        identified=True, eps=1e-8,
    )
    assert s == LinearityStatus.RELIABLE


def test_reliable_negative_treatment_effect() -> None:
    """Same-signed direct/total works in the negative direction
    too; the classifier is sign-symmetric."""
    s = _classify_linearity(
        total_ate=-10.0, direct_ate=-4.0,
        indirect_proportion=+0.6,
        identified=True, eps=1e-8,
    )
    assert s == LinearityStatus.RELIABLE


# ============ Integration: full pipeline on synthetic data ============

def _build_linear_mediation_cells(
    *, n: int = 200, beta_treat_med: float = 1.0,
    beta_med_out: float = 1.0, beta_treat_out_direct: float = 0.5,
    noise_med: float = 0.3, noise_out: float = 0.3,
    seed: int = 0,
) -> list[Mapping[str, object]]:
    """Linear DGP: treatment → mediator → outcome + direct
    treatment → outcome path. With same-signed coefficients +
    moderate noise + n=200 + a clean DAG, the OLS decomposition
    recovers RELIABLE status without multicollinearity-induced
    sign flips. This is the substrate condition CLAUDE.md says
    `mediation_dowhy` can be trusted at."""
    rng = np.random.default_rng(seed)
    cells: list[Mapping[str, object]] = []
    for i in range(n):
        t = float(rng.normal(0.0, 1.0))
        m = beta_treat_med * t + float(rng.normal(0.0, noise_med))
        y = (
            beta_treat_out_direct * t + beta_med_out * m
            + float(rng.normal(0.0, noise_out))
        )
        cells.append({
            'env_name': 'lg_scm',
            'seed': i,
            'arm_key': 'treatment',
            'treatment': t,
            'mediator': m,
            'outcome': y,
        })
    return cells


def test_full_pipeline_reliable_under_clean_linear_dgp() -> None:
    """End-to-end: linear DGP with same-signed mediation, low
    noise, n=200 cells, clean DAG → `linearity_status == RELIABLE`.
    Verifies the classifier wires correctly through the full
    `mediation_dowhy(...)` call.

    Closed-form expectations: total ATE ≈ β_treat_med · β_med_out
    + β_treat_out_direct = 1.0 · 1.0 + 0.5 = 1.5; direct ATE ≈
    β_treat_out_direct = 0.5; indirect proportion ≈ 1.0 / 1.5 ≈
    0.667. These should comfortably fall in the RELIABLE
    envelope (same sign, in [0, 1]).
    """
    pytest.importorskip('dowhy')
    pytest.importorskip('sklearn')
    cells = _build_linear_mediation_cells(n=200, seed=42)
    dag: list[tuple[str, str]] = [
        ('treatment', 'mediator'),
        ('mediator', 'outcome'),
        ('treatment', 'outcome'),
    ]
    result = mediation_dowhy.fn(
        cells, treatment='treatment', outcome='outcome',
        mediators=('mediator',), dag=dag,
    )
    assert result.identified
    assert result.linearity_status == LinearityStatus.RELIABLE
    # Magnitudes within sampling bounds (n=200, σ=0.3 → SE ≈ 0.04
    # on the OLS coefficient; 30% tolerance comfortably absorbs
    # both numerator + denominator sampling variance on the ratio).
    assert abs(result.total_ate - 1.5) < 0.5, (
        f'expected total_ate ≈ 1.5; got {result.total_ate}'
    )
    assert abs(result.direct_ate - 0.5) < 0.2, (
        f'expected direct_ate ≈ 0.5; got {result.direct_ate}'
    )
    assert 0.5 < result.indirect_proportion < 0.85, (
        f'expected indirect_proportion ≈ 0.667; got '
        f'{result.indirect_proportion}'
    )


def test_full_pipeline_sign_flipped_under_high_multicollinearity() -> None:
    """High-multicollinearity DGP: two mediators that are nearly
    identical (m1 ≈ m2 + tiny noise) → OLS coefficients fight
    each other → direct ATE can sign-flip relative to total.

    This reproduces the FR γ-WHY failure mode under controlled
    conditions: r(m1, m2) ≈ 0.99 makes OLS pick essentially
    arbitrary coefficient splits, and the direct (treatment)
    coefficient ends up small + opposite-signed.

    Status SIGN_FLIPPED IS the diagnostic that the linear
    assumption can't be read here — a bridge consuming this
    primitive at such a scope sees the flag and should fall back
    to `partial_spearman`."""
    pytest.importorskip('dowhy')
    pytest.importorskip('sklearn')
    rng = np.random.default_rng(7)
    n = 200
    cells: list[Mapping[str, object]] = []
    for i in range(n):
        t = float(rng.normal(0.0, 1.0))
        # Two near-identical mediators (collinearity engineered)
        m_base = 2.0 * t + float(rng.normal(0.0, 0.3))
        m1 = m_base + float(rng.normal(0.0, 0.05))
        m2 = m_base + float(rng.normal(0.0, 0.05))
        # Outcome depends on both mediators; total effect is
        # strong positive. With m1 ≈ m2 in OLS, the treatment
        # coefficient (direct) becomes an arbitrary residual
        # subject to multicollinearity drift.
        y = 1.0 * m1 + 1.0 * m2 + float(rng.normal(0.0, 0.3))
        cells.append({
            'env_name': 'lg_scm',
            'seed': i,
            'arm_key': 'treatment',
            'treatment': t,
            'mediator_a': m1,
            'mediator_b': m2,
            'outcome': y,
        })
    dag: list[tuple[str, str]] = [
        ('treatment', 'mediator_a'),
        ('treatment', 'mediator_b'),
        ('mediator_a', 'outcome'),
        ('mediator_b', 'outcome'),
    ]
    result = mediation_dowhy.fn(
        cells, treatment='treatment', outcome='outcome',
        mediators=('mediator_a', 'mediator_b'), dag=dag,
    )
    assert result.identified
    # Total ATE is well-identified (treatment → outcome through
    # both mediators, ATE ≈ 4.0). The diagnostic surfaces the
    # multicollinearity failure regardless of total magnitude.
    assert result.total_ate > 1.0, (
        f'total_ate should be substantially positive; got '
        f'{result.total_ate}'
    )
    # With m1 ≈ m2 collinear, the OLS direct coefficient on
    # `treatment` (with both mediators conditioned on) collapses
    # toward zero (mediators absorb almost all the structural
    # signal). Either the direct ATE flips negative
    # (SIGN_FLIPPED) or proportion overshoots [0, 1]
    # (OUT_OF_BOUNDS) — both signal "linear assumption breaks
    # here, don't read magnitudes." The status is one of the
    # two diagnostic-failure modes.
    assert result.linearity_status in (
        LinearityStatus.SIGN_FLIPPED, LinearityStatus.OUT_OF_BOUNDS,
    ), (
        f'expected linearity diagnostic to flag collinearity '
        f'failure; got status={result.linearity_status}, '
        f'total={result.total_ate}, direct={result.direct_ate}, '
        f'proportion={result.indirect_proportion}'
    )


def test_empty_mediators_raises() -> None:
    """Empty mediators tuple is incoherent for this primitive
    (use `dowhy.backdoor_ate` for total-only). Raise loudly
    rather than degenerate."""
    with pytest.raises(ValueError, match='non-empty tuple'):
        mediation_dowhy.fn(
            [], treatment='t', outcome='y', mediators=(),
            dag=[('t', 'y')],
        )


def test_proportion_nan_when_total_is_zero() -> None:
    """When |total| < eps the proportion is NaN by guard; the
    classifier maps to POWER_INSUFFICIENT (no decomposition
    possible). Direct construction so we don't need a DAG."""
    s = _classify_linearity(
        total_ate=0.0, direct_ate=0.0,
        indirect_proportion=float('nan'),
        identified=True, eps=1e-8,
    )
    assert s == LinearityStatus.POWER_INSUFFICIENT
    # Sanity: explicit NaN proportion with non-zero total also
    # returns POWER_INSUFFICIENT (defensive; shouldn't happen
    # under normal flow but the classifier shouldn't crash).
    s2 = _classify_linearity(
        total_ate=5.0, direct_ate=2.0,
        indirect_proportion=float('nan'),
        identified=True, eps=1e-8,
    )
    assert s2 == LinearityStatus.POWER_INSUFFICIENT


def test_status_strenum_serialization_value() -> None:
    """`LinearityStatus` is a StrEnum; its `.value` is the
    lowercased status string. This matters for runner-report
    serialization (frozen-dataclass coercion uses str(enum) =
    enum's value)."""
    assert LinearityStatus.RELIABLE.value == 'reliable'
    assert LinearityStatus.SIGN_FLIPPED.value == 'sign_flipped'
    assert LinearityStatus.OUT_OF_BOUNDS.value == 'out_of_bounds'
    assert LinearityStatus.UNIDENTIFIED.value == 'unidentified'
    assert (
        LinearityStatus.POWER_INSUFFICIENT.value == 'power_insufficient'
    )
    # PEP 698 — StrEnum participates in string equality.
    assert LinearityStatus.RELIABLE == 'reliable'


# Marker: math.isnan reachability — keeps `import math` from
# being deemed unused if classifier evolves to drop math calls.
assert callable(math.isnan)
