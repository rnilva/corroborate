"""Regression tests for the independent-arm comparison."""
from __future__ import annotations

import math

from corroborate.analyses.paired.arm_mean_diff import arm_mean_diff


def test_empty_pair_by_disables_pairing_diagnostic() -> None:
    """An explicitly unpaired comparison must not invent key ``()``."""
    cells = [
        {'arm_key': 'treatment', 'seed': 1, 'score': 2.0},
        {'arm_key': 'treatment', 'seed': 2, 'score': 3.0},
        {'arm_key': 'baseline', 'seed': 1, 'score': 1.0},
        {'arm_key': 'baseline', 'seed': 2, 'score': 1.5},
    ]
    result = arm_mean_diff.fn(
        cells,
        source='score',
        treatment_arm='treatment',
        baseline_arm='baseline',
        pair_by=(),
    )

    assert result.n_treatment == 2
    assert result.n_baseline == 2
    assert result.n_paired == 0
    assert math.isnan(result.pairing_rho)
    assert math.isnan(result.pairing_rho_se)
