"""End-to-end regression for the external-study claim path."""
from __future__ import annotations

from pathlib import Path

from corroborate.analyses.paired.paired_directional import (
    PairedDirectionalResult,
)
from corroborate.bridge.bridge import evaluate
from corroborate.bridge.verdict import RefutationClass, Verdict
from corroborate.data import adapt_study

from examples.sb3_demo.sb3_claim import higher_gamma_improves_return


_BUNDLE = Path(__file__).parents[1] / 'examples' / 'sb3_demo' / 'bundle'


def test_verified_bundle_evaluates_data_independent_claim_module() -> None:
    study = adapt_study(_BUNDLE)
    panel = study.to_panel()

    evaluation = evaluate(
        higher_gamma_improves_return,
        panel.cells,
        recorded_contrast=study.contrast,
    )

    assert study.receipt.admissible
    assert evaluation.n_cells_in_scope == 6
    assert evaluation.extent_hash != 0
    assert evaluation.verdict is Verdict.POWER_INSUFFICIENT
    assert evaluation.refutation_class is RefutationClass.UNDERPOWERED

    result = evaluation.analysis_results['paired_directional']
    assert isinstance(result, PairedDirectionalResult)
    assert result.measurable == 'return_mean'
    assert result.baseline_arm == study.contrast.baseline_key
    assert result.treatment_arm == study.contrast.treatment_key
    assert result.n_pairs == 3
    assert result.predicted_direction == 'a_gt_b'
    assert result.alpha == 0.05
    assert result.sesoi_dz == 0.5
    assert result.minimum_pairs == 3
