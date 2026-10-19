"""Reference tests for configured paired directional inference."""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping

import numpy as np
import pytest
from scipy.stats import t
from scipy.stats import skew

from corroborate.analyses.paired.paired_directional import (
    DirectionalAlternative,
    paired_directional,
    paired_directional_verdict,
)
from corroborate.analyses.paired.paired_g import (
    _paired_g_assumption_violations,
    paired_g,
)
from corroborate.bridge.verdict import RefutationClass, Verdict
from corroborate.core.hypothesis import PredictedDirection
from corroborate.data import to_dataframe
from corroborate.measurables import measurable


@measurable(name='directional_duplicate_outcome', reads=('directional_duplicate_outcome',))
def directional_duplicate_outcome(record: Mapping[str, object]) -> float:
    value = record['directional_duplicate_outcome']
    assert isinstance(value, (int, float))
    return float(value)


def _rows(deltas: np.ndarray) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed, delta in enumerate(deltas.tolist()):
        rows.extend((
            {
                'id': f'b-{seed}',
                'seed': seed,
                'arm_key': 'baseline',
                'metric': 0.0,
            },
            {
                'id': f't-{seed}',
                'seed': seed,
                'arm_key': 'treatment',
                'metric': float(delta),
            },
        ))
    return rows


def _run(
    deltas: np.ndarray,
    *,
    minimum_pairs: int = 52,
    predicted_direction: DirectionalAlternative = 'a_gt_b',
    alpha: float = 0.05,
    sesoi_dz: float = 0.35,
):
    return paired_directional.fn(
        to_dataframe(_rows(deltas)),
        source='metric',
        treatment_arm='treatment',
        baseline_arm='baseline',
        pair_by=('seed',),
        predicted_direction=predicted_direction,
        alpha=alpha,
        sesoi_dz=sesoi_dz,
        minimum_pairs=minimum_pairs,
    )


def _centred_spread(n: int, scale: float) -> np.ndarray:
    values = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    return scale * values / np.std(values, ddof=1)


def test_support_p_value_matches_scipy_paired_difference_reference() -> None:
    deltas = 0.40 + _centred_spread(64, 0.50)
    result = _run(deltas)
    reference_statistic = float(
        np.mean(deltas) / (np.std(deltas, ddof=1) / math.sqrt(len(deltas))),
    )
    reference_p = float(t.sf(reference_statistic, df=len(deltas) - 1))
    assert result.t_statistic == pytest.approx(reference_statistic)
    assert result.p_value == pytest.approx(reference_p)
    assert result.opposite_p_value == pytest.approx(1.0 - reference_p)
    assert result.degrees_of_freedom == len(deltas) - 1
    assert result.dz == pytest.approx(
        float(np.mean(deltas) / np.std(deltas, ddof=1)),
    )
    assert result.minimum_pairs_met
    assert paired_directional_verdict(result) == (Verdict.HELD, None)


def test_configuration_round_trips_into_result() -> None:
    result = _run(0.40 + _centred_spread(64, 0.50))
    assert result.predicted_direction == 'a_gt_b'
    assert result.alpha == 0.05
    assert result.sesoi_dz == 0.35
    assert result.minimum_pairs == 52


def test_equivalence_is_positive_evidence_not_failed_significance() -> None:
    deltas = _centred_spread(64, 0.10)
    result = _run(deltas)
    assert result.p_value == pytest.approx(0.5)
    assert result.practically_equivalent
    assert result.equivalence_p_lower < 0.05
    assert result.equivalence_p_upper < 0.05
    assert paired_directional_verdict(result) == (
        Verdict.NO_EFFECT,
        RefutationClass.NULL_EFFECT,
    )


def test_opposite_direction_is_sign_flip() -> None:
    deltas = -0.40 + _centred_spread(64, 0.50)
    result = _run(deltas)
    assert result.opposite_p_value < 0.05
    assert paired_directional_verdict(result) == (
        Verdict.NO_EFFECT,
        RefutationClass.SIGN_FLIP,
    )


def test_lower_prediction_uses_lower_tail() -> None:
    deltas = -0.40 + _centred_spread(64, 0.50)
    result = _run(deltas, predicted_direction='a_lt_b')
    assert result.p_value < 0.05
    assert paired_directional_verdict(result) == (Verdict.HELD, None)


def test_minimum_pairs_gate_remains_inconclusive_even_if_large() -> None:
    deltas = 2.0 + _centred_spread(12, 0.20)
    result = _run(deltas)
    assert result.p_value < 1e-8
    assert not result.minimum_pairs_met
    assert paired_directional_verdict(result) == (
        Verdict.POWER_INSUFFICIENT,
        RefutationClass.UNDERPOWERED,
    )
    assert any(
        'minimum_pairs_not_met' in flag
        for flag in result.assumption_violations
    )


def test_noncentral_t_interval_contains_point_estimate() -> None:
    deltas = 0.20 + _centred_spread(64, 0.70)
    result = _run(deltas)
    lo, hi = result.dz_ci
    assert math.isfinite(lo) and math.isfinite(hi)
    assert lo < result.dz < hi
    raw_lo, raw_hi = result.mean_diff_ci
    assert raw_lo < result.mean_diff < raw_hi


def test_paired_skew_diagnostic_matches_adjusted_fisher_pearson() -> None:
    deltas = [0.0, 0.0, 0.2, 0.3, 0.4, 4.0]
    expected = float(skew(np.asarray(deltas), bias=False))
    violations = _paired_g_assumption_violations(deltas)
    skew_flag = next(flag for flag in violations if 'skew_bias_likely' in flag)
    assert f'skew={expected:+.2f}' in skew_flag


@pytest.mark.parametrize('duplicate_value', [1.0, 101.0])
def test_directional_refuses_duplicate_registered_outcomes(
    duplicate_value: float,
) -> None:
    """Replay aliases must not silently change a directional estimand.

    The general paired-g helper intentionally averages aliases whose
    differences are confined to registered outcomes and provenance.
    Directional inference promises stricter experimental-unit uniqueness,
    including an identical replay and a conflicting cached outcome.
    """
    rows = _rows(np.asarray([1.0, 2.0, 3.0]))
    for row in rows:
        row['directional_duplicate_outcome'] = row.pop('metric')
    rows.append({
        'id': 't-0-replayed',
        'seed': 0,
        'arm_key': 'treatment',
        'directional_duplicate_outcome': duplicate_value,
    })
    df = to_dataframe(rows)
    general = paired_g.fn(
        df,
        source='directional_duplicate_outcome',
        treatment_arm='treatment',
        baseline_arm='baseline',
        dedupe_strategy='raise',
    )
    assert general.n_pairs == 3
    assert general.mean_diff == pytest.approx(
        ((1.0 + duplicate_value) / 2.0 + 2.0 + 3.0) / 3.0,
    )
    with pytest.raises(ValueError, match='paired_directional: duplicate'):
        paired_directional.fn(
            df,
            source='directional_duplicate_outcome',
            treatment_arm='treatment',
            baseline_arm='baseline',
            predicted_direction='a_gt_b',
        )


def test_directional_ignores_duplicates_outside_declared_contrast() -> None:
    rows = _rows(np.asarray([1.0, 2.0, 3.0]))
    rows.extend([
        {'id': 'o-0', 'seed': 0, 'arm_key': 'other', 'metric': 100.0},
        {'id': 'o-1', 'seed': 0, 'arm_key': 'other', 'metric': 200.0},
    ])
    result = paired_directional.fn(
        to_dataframe(rows),
        source='metric',
        treatment_arm='treatment',
        baseline_arm='baseline',
        predicted_direction='a_gt_b',
    )
    assert result.n_pairs == 3
    assert result.mean_diff == pytest.approx(2.0)


def _invalid_run(
    *,
    predicted_direction: PredictedDirection = 'a_gt_b',
    alpha: float = 0.05,
    sesoi_dz: float = 0.35,
    minimum_pairs: int = 52,
) -> object:
    return paired_directional.fn(
        to_dataframe(_rows(np.asarray([0.1, 0.2]))),
        source='metric',
        treatment_arm='treatment',
        baseline_arm='baseline',
        predicted_direction=predicted_direction,
        alpha=alpha,
        sesoi_dz=sesoi_dz,
        minimum_pairs=minimum_pairs,
    )


@pytest.mark.parametrize(
    ('construct', 'message'),
    [
        (lambda: _invalid_run(alpha=0.6), 'alpha'),
        (lambda: _invalid_run(sesoi_dz=0.0), 'sesoi'),
        (lambda: _invalid_run(minimum_pairs=1), 'minimum_pairs'),
        (
            # Valid bridge metadata, but not a directional one-sided
            # alternative: the analysis must still fail closed.
            lambda: _invalid_run(predicted_direction='two_sided'),
            'predicted_direction',
        ),
    ],
)
def test_invalid_configuration_fails_closed(
    construct: Callable[[], object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        construct()
