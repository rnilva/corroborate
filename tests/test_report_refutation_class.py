"""The saved audit report preserves exact NO_EFFECT subtypes."""
from __future__ import annotations

from types import MappingProxyType

from corroborate.bridge.bridge import BridgeEvaluation, claim_bridge
from corroborate.bridge.verdict import RefutationClass, Verdict
from corroborate.runner.report import (
    _build_bridge_entry,
    _coerce_value,
)


@claim_bridge(source='x', target='y')
def _refuted_bridge() -> Verdict:
    return Verdict.NO_EFFECT


def test_report_serialises_refutation_class() -> None:
    evaluation = BridgeEvaluation(
        bridge_name=_refuted_bridge.name,
        verdict=Verdict.NO_EFFECT,
        refutation_class=RefutationClass.SIGN_FLIP,
        analysis_results=MappingProxyType({}),
        n_cells_in_scope=12,
    )
    entry = _build_bridge_entry(
        _refuted_bridge,
        evaluation,
        n_cells_total=12,
    )
    serialised = _coerce_value(entry)
    assert isinstance(serialised, dict)
    assert serialised['verdict'] == 'no_effect'
    assert serialised['refutation_class'] == 'sign_flip'
