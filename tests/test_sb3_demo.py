"""End-to-end regression for the external-runs claim path."""
from __future__ import annotations

from pathlib import Path

from corroborate.analyses.paired.paired_directional import (
    PairedDirectionalResult,
)
from corroborate.bridge.bridge import evaluate
from corroborate.bridge.verdict import RefutationClass, Verdict
from corroborate.data import load_runs

from examples.sb3_demo.sb3_claim import higher_gamma_improves_return


_RUNS = Path(__file__).parents[1] / 'examples' / 'sb3_demo' / 'runs'


def test_loaded_runs_evaluate_data_independent_claim_module() -> None:
    df = load_runs(_RUNS)

    evaluation = evaluate(higher_gamma_improves_return, df)

    assert evaluation.n_cells_in_scope == 6
    assert evaluation.extent_hash != 0
    assert evaluation.blocked_by is None
    assert evaluation.warnings == ()
    assert evaluation.verdict is Verdict.POWER_INSUFFICIENT
    assert evaluation.refutation_class is RefutationClass.UNDERPOWERED

    result = evaluation.analysis_results['paired_directional']
    assert isinstance(result, PairedDirectionalResult)
    assert result.measurable == 'return_mean'
    assert result.baseline_arm == 'gamma=0.8'
    assert result.treatment_arm == 'gamma=0.99'
    assert result.n_pairs == 3
    assert result.predicted_direction == 'a_gt_b'
    assert result.alpha == 0.05
    assert result.sesoi_dz == 0.5
    assert result.minimum_pairs == 3
