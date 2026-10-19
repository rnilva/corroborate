"""Native finding verdicts survive report persistence, including partial runs."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import polars as pl
import pytest

from corroborate._internals.json import loads
from corroborate.bridge.bridge import Bridge, BridgeEvaluation, claim_bridge
from corroborate.bridge.verdict import Verdict
from corroborate.core.finding import Finding
from corroborate.core.intervention import DoEffect
from corroborate.graph.causal import ClusterVerdict, Direction, Tier
from corroborate.runner.report import RunReport, build_report, write_report
from corroborate.runner.runner import run


@claim_bridge(source='feature', target='outcome', direction=Direction.DIRECT,
              tier=Tier.ASSOCIATIONAL)
def _finding_report_first() -> Verdict:
    return Verdict.HELD


@claim_bridge(source='feature', target='outcome', direction=Direction.DIRECT,
              tier=Tier.ASSOCIATIONAL)
def _finding_report_second() -> Verdict:
    return Verdict.HELD


@claim_bridge(source='feature', target='outcome', direction=Direction.DIRECT,
              tier=Tier.ASSOCIATIONAL)
def _finding_report_error() -> Verdict:
    raise RuntimeError('Broken study analysis')


class _Finding:
    def __init__(
        self,
        bridges: tuple[Bridge, ...],
        expected: ClusterVerdict = ClusterVerdict.SUPPORTED,
        blocked_on: str | None = None,
    ) -> None:
        self.__name__ = 'study.finding'
        self.BRIDGES = bridges
        self.EXPECTED = expected
        self.BLOCKED_ON = blocked_on


class _Hypothesis:
    def __init__(self, bridges: tuple[Bridge, ...]) -> None:
        self.__name__ = 'study.hypothesis'
        self.INTERVENTION = DoEffect.from_values(
            source='setting', reference='baseline', treatment='treatment',
        )
        self.BRIDGES = bridges
        self.FINDINGS: tuple[Finding, ...] = (_Finding(bridges),)


def _evaluation(bridge: Bridge, verdict: Verdict, count: int = 2) -> BridgeEvaluation:
    return BridgeEvaluation(bridge_name=bridge.name, verdict=verdict,
                            analysis_results={}, n_rows_in_scope=count)


def _report(
    results: Mapping[str, BridgeEvaluation],
    *,
    errors: Mapping[str, BaseException] | None = None,
    expected: ClusterVerdict = ClusterVerdict.SUPPORTED,
    blocked_on: str | None = None,
) -> RunReport:
    bridges = (_finding_report_first, _finding_report_second)
    return build_report(
        hypothesis_module_name='study.hypothesis', bridges=bridges,
        results=results, errors={} if errors is None else errors,
        n_rows_total=2, cache_path=None, measurable_signatures={},
        findings=(_Finding(bridges, expected, blocked_on),),
    )


@pytest.mark.parametrize(
    ('first', 'second', 'expected_actual'),
    [
        (Verdict.HELD, Verdict.HELD, ClusterVerdict.SUPPORTED),
        (Verdict.HELD, Verdict.NO_EFFECT, ClusterVerdict.REFUTED),
        (Verdict.HELD, None, ClusterVerdict.UNDERPOWERED),
        (Verdict.NO_EFFECT, None, ClusterVerdict.REFUTED),
        (None, None, ClusterVerdict.UNDERPOWERED),
    ],
)
def test_findings_use_native_graph_composition(
    first: Verdict | None, second: Verdict | None,
    expected_actual: ClusterVerdict,
) -> None:
    results = {
        bridge.name: _evaluation(bridge, verdict)
        for bridge, verdict in [(_finding_report_first, first), (_finding_report_second, second)]
        if verdict is not None
    }
    finding = _report(results).findings[0]
    assert finding.actual is expected_actual
    assert finding.expected is ClusterVerdict.SUPPORTED
    assert finding.drift is (expected_actual is not ClusterVerdict.SUPPORTED)
    assert finding.unevaluated_bridge_names == tuple(
        b.name for b in (_finding_report_first, _finding_report_second) if b.name not in results
    )


def test_zero_row_findings_distinguish_empty_from_missing() -> None:
    results = {bridge.name: _evaluation(bridge, Verdict.POWER_INSUFFICIENT, 0)
               for bridge in (_finding_report_first, _finding_report_second)}
    complete = _report(results)
    partial = _report({_finding_report_first.name: results[_finding_report_first.name]})
    assert complete.findings[0].actual is ClusterVerdict.EMPTY_EXTENT
    assert partial.findings[0].actual is ClusterVerdict.UNDERPOWERED


def test_errored_bridge_does_not_become_synthetic_refutation() -> None:
    report = _report(
        {_finding_report_first.name: _evaluation(_finding_report_first, Verdict.HELD)},
        errors={_finding_report_second.name: ValueError('bad analysis')},
    )
    assert report.findings[0].actual is ClusterVerdict.UNDERPOWERED
    assert len(report.bridges) == 1
    assert report.errored_bridges[0].bridge_name == _finding_report_second.name
    assert report.findings[0].unevaluated_bridge_names == (_finding_report_second.name,)


def test_native_finding_serialization_preserves_blocker_and_drift(tmp_path: Path) -> None:
    report = _report(
        {_finding_report_first.name: _evaluation(_finding_report_first, Verdict.HELD)},
        expected=ClusterVerdict.UNDERPOWERED, blocked_on='Second contrast pending',
    )
    target = tmp_path / 'report.json'
    write_report(report, target)
    data = loads(target.read_text())
    assert isinstance(data, dict)
    assert data['findings'] == [{
        'finding_name': 'study.finding', 'expected': 'underpowered',
        'actual': 'underpowered', 'drift': False,
        'blocked_on': 'Second contrast pending',
        'bridge_names': [_finding_report_first.name, _finding_report_second.name],
        'unevaluated_bridge_names': [_finding_report_second.name],
    }]


def test_default_empty_findings_is_backward_compatible() -> None:
    report = build_report(
        hypothesis_module_name='study.hypothesis', bridges=(), results={}, errors={},
        n_rows_total=0, cache_path=None, measurable_signatures={},
    )
    assert report.findings == ()


@pytest.mark.parametrize('filtered', [False, True])
def test_runner_keeps_authored_finding_members_when_filtering(
    tmp_path: Path, filtered: bool,
) -> None:
    hypothesis = _Hypothesis((_finding_report_first, _finding_report_second))
    path = tmp_path / 'report.json'
    results = run(
        hypothesis,
        data=pl.DataFrame({'id': ['one', 'two'], 'feature': [1., 2.], 'outcome': [2., 4.]}),
        use_cache=False, write_report=True, report_path=path,
        bridge_filter='first' if filtered else None,
    )
    assert len(results) == (1 if filtered else 2)
    payload = loads(path.read_text())
    assert isinstance(payload, dict)
    findings = payload['findings']
    assert isinstance(findings, list) and len(findings) == 1
    finding = findings[0]
    assert isinstance(finding, dict)
    assert finding['actual'] == ('underpowered' if filtered else 'supported')
    assert finding['unevaluated_bridge_names'] == ([_finding_report_second.name] if filtered else [])


def test_runner_persists_finding_when_one_analysis_raises(tmp_path: Path) -> None:
    hypothesis = _Hypothesis((_finding_report_first, _finding_report_error))
    path = tmp_path / 'report.json'
    results = run(
        hypothesis,
        data=pl.DataFrame({'id': ['one', 'two'], 'feature': [1., 2.], 'outcome': [2., 4.]}),
        use_cache=False, write_report=True, report_path=path,
    )
    assert list(results) == [_finding_report_first.name]
    payload = loads(path.read_text())
    assert isinstance(payload, dict)
    findings = payload['findings']
    assert isinstance(findings, list)
    finding = findings[0]
    assert isinstance(finding, dict) and finding['actual'] == 'underpowered'
    errors = payload['errored_bridges']
    assert isinstance(errors, list) and len(errors) == 1
