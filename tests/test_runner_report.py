"""Tests for `corroborate.runner.report` — the post-run JSON
audit serializer."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType

import numpy as np
import polars as pl
import pytest

from corroborate.analyses.dowhy import BackdoorResult, RefutationResult
from corroborate.analyses.factorial_2x2 import Factorial2x2Result, FactorialPerEnv
from corroborate.analyses.paired_g import PairedGResult
from corroborate.analyses.paired_g_per_burst import PerBurstResult, PerBurstStratum
from corroborate.analyses.proportion_mediated import ProportionMediatedResult
from corroborate.analyses.verdict_distribution import (
    VerdictCounts, VerdictDistributionResult,
)
from corroborate.bridge.admission import GateLevel, GateResult
from corroborate.bridge.bridge import BridgeEvaluation
from corroborate.runner.report import (
    BridgeReportEntry,
    ErroredBridgeEntry,
    RunReport,
    SCHEMA_VERSION,
    _coerce_value,
    build_report,
    write_report,
)
from corroborate.bridge.verdict import Verdict


# ============ _coerce_value — primitive rules ============


def test_coerce_none_bool_str_int_unchanged() -> None:
    assert _coerce_value(None) is None
    assert _coerce_value(True) is True
    assert _coerce_value(False) is False
    assert _coerce_value('hello') == 'hello'
    assert _coerce_value(42) == 42


def test_coerce_finite_float_unchanged() -> None:
    assert _coerce_value(1.5) == 1.5
    assert _coerce_value(0.0) == 0.0
    assert _coerce_value(-3.14) == -3.14


def test_coerce_nan_inf_to_string_sentinels() -> None:
    """NaN and inf encoded as string sentinels — preserves
    'computed and degenerate' vs 'not measured' (null)."""
    assert _coerce_value(float('nan')) == 'NaN'
    assert _coerce_value(float('inf')) == 'Infinity'
    assert _coerce_value(float('-inf')) == '-Infinity'


def test_coerce_numpy_scalars() -> None:
    """numpy.generic → .item() then re-coerce."""
    assert _coerce_value(np.float64(1.5)) == 1.5
    assert _coerce_value(np.int64(42)) == 42
    assert _coerce_value(np.bool_(True)) is True
    # NaN through numpy
    assert _coerce_value(np.float64('nan')) == 'NaN'


def test_coerce_numpy_array_to_list() -> None:
    arr = np.array([1.0, float('nan'), 3.0])
    out = _coerce_value(arr)
    assert out == [1.0, 'NaN', 3.0]


def test_coerce_enum_to_value() -> None:
    """Enum.value used (idiomatic per RunRow.as_dict at schema.py)."""
    assert _coerce_value(Verdict.HELD) == 'held'
    assert _coerce_value(GateLevel.WARN) == 'warn'


def test_coerce_mapping_recurses() -> None:
    out = _coerce_value({'a': 1.0, 'b': float('nan'), 'c': [True, False]})
    assert out == {'a': 1.0, 'b': 'NaN', 'c': [True, False]}


def test_coerce_tuple_and_list_recurse() -> None:
    assert _coerce_value((1, 2.0, 'x')) == [1, 2.0, 'x']
    assert _coerce_value([float('nan'), 1]) == ['NaN', 1]


def test_coerce_set_sorted() -> None:
    """Sets get sorted by repr for deterministic output."""
    out = _coerce_value({3, 1, 2})
    assert out == [1, 2, 3]


def test_coerce_callable_returns_skip_sentinel() -> None:
    """Callables are skipped in upstream filters."""
    from corroborate.runner.report import _SKIP
    assert _coerce_value(lambda x: x) is _SKIP


# ============ Dataclass introspection: fields + properties ============


def test_coerce_paired_g_includes_property_p_value() -> None:
    """`p_value` is a `@property` on PairedGResult, not a field —
    serializer must walk both. Without this the headline number for
    half the bridges would be missing from the report."""
    r = PairedGResult(
        g=2.0, se=0.5, mean_diff=4.0, mean_diff_se=1.0,
        n_pairs=30, n_treatment=30, n_baseline=30,
        helped_fraction=0.85, pair_by=('seed',), measurable='outcome',
        treatment_arm='ddqn', baseline_arm='vanilla',
    )
    out = _coerce_value(r)
    assert isinstance(out, dict)
    # Fields
    assert out['g'] == 2.0
    assert out['mean_diff'] == 4.0
    assert out['n_pairs'] == 30
    # Properties
    assert 'p_value' in out and 0.0 <= out['p_value'] < 1.0
    assert 'mean_diff_p_value' in out and 0.0 <= out['mean_diff_p_value'] < 1.0


def test_coerce_paired_g_degenerate_se_zero_p_value_nan_string() -> None:
    """When SE is zero, p_value property returns NaN — must serialize
    to "NaN" string sentinel, not raise."""
    r = PairedGResult(
        g=float('nan'), se=0.0, mean_diff=float('nan'), mean_diff_se=0.0,
        n_pairs=0, n_treatment=0, n_baseline=0,
        helped_fraction=float('nan'), pair_by=('seed',),
        measurable='outcome', treatment_arm='ddqn', baseline_arm='vanilla',
    )
    out = _coerce_value(r)
    assert isinstance(out, dict)
    assert out['p_value'] == 'NaN'
    assert out['mean_diff_p_value'] == 'NaN'


def test_coerce_verdict_counts_property_fractions() -> None:
    """VerdictCounts has held_fraction / violation_fraction as
    @property accessors. Serializer captures both."""
    vc = VerdictCounts(
        held=15, invariant_violation=5, power_insufficient=10,
        other=0, total=30, dominant='held',
    )
    out = _coerce_value(vc)
    assert isinstance(out, dict)
    assert out['held'] == 15
    assert out['held_fraction'] == 0.5
    assert out['violation_fraction'] == pytest.approx(5 / 30)


def test_coerce_verdict_counts_zero_total_property_returns_nan_string() -> None:
    vc = VerdictCounts(held=0, invariant_violation=0, power_insufficient=0,
                       other=0, total=0, dominant='')
    out = _coerce_value(vc)
    assert isinstance(out, dict)
    assert out['held_fraction'] == 'NaN'
    assert out['violation_fraction'] == 'NaN'


def test_property_that_raises_yields_nan_string() -> None:
    """Properties that raise (instead of returning NaN) become "NaN"
    in the report — shape stays stable, failure visible."""
    @dataclass(frozen=True, slots=True)
    class _Quirky:
        x: float

        @property
        def doubles_or_dies(self) -> float:
            raise RuntimeError('boom')

    out = _coerce_value(_Quirky(x=1.0))
    assert isinstance(out, dict)
    assert out['x'] == 1.0
    assert out['doubles_or_dies'] == 'NaN'


# ============ Composite / nested Result classes ============


def test_coerce_nested_dataclass_proportion_mediated() -> None:
    r = ProportionMediatedResult(
        proportion=0.4, total=2.5, direct=1.5, indirect=1.0,
        slope_y_on_m=-0.3, in_unit_interval=True, n_pairs=50,
        target='outcome', mediator='effective_horizon',
        treatment_arm='ddqn', baseline_arm='vanilla',
        pair_by=('seed', 'env_name'),
    )
    out = _coerce_value(r)
    assert isinstance(out, dict)
    assert out['proportion'] == 0.4
    assert out['in_unit_interval'] is True
    assert out['pair_by'] == ['seed', 'env_name']


def test_coerce_per_burst_result_full_panel() -> None:
    """Per Q1 user choice: panels serialized FULL (no summarization)."""
    strata = tuple(
        PerBurstStratum(
            env_name='Pong', burst_index=b, g=0.5, se=0.1,
            n_pairs=20, helped_fraction=0.7,
        )
        for b in range(5)
    )
    r = PerBurstResult(
        strata=strata, measurable='outcome',
        treatment_arm='ddqn', baseline_arm='vanilla',
        pair_by=('seed',),
    )
    out = _coerce_value(r)
    assert isinstance(out, dict)
    assert isinstance(out['strata'], list)
    assert len(out['strata']) == 5
    # Each stratum is a dict with the field values
    s0 = out['strata'][0]
    assert isinstance(s0, dict)
    assert s0['env_name'] == 'Pong'
    assert s0['burst_index'] == 0


def test_coerce_dowhy_results() -> None:
    bd = BackdoorResult(
        ate=-0.5, identified=True, estimand_str='Y|X,Z',
        method_name='backdoor.linear_regression',
        treatment='X', outcome='Y', n_rows=100,
    )
    out = _coerce_value(bd)
    assert isinstance(out, dict)
    assert out['ate'] == -0.5
    assert out['identified'] is True

    rf = RefutationResult(
        real_ate=-0.5, refuted_ate=0.01, drift=0.51,
        method_name='backdoor.linear_regression',
        refuter_name='placebo_treatment_refuter',
        treatment='X', outcome='Y', n_rows=100,
    )
    out = _coerce_value(rf)
    assert isinstance(out, dict)
    assert out['drift'] == pytest.approx(0.51)


def test_coerce_factorial_2x2_with_nested_per_env() -> None:
    per_env = (
        FactorialPerEnv(
            env_name='FourRooms-misc',
            g_b_minus_a=0.4, se_b_minus_a=0.1,
            g_d_minus_c=-0.79, se_d_minus_c=0.2,
            g_c_minus_a=0.3, se_c_minus_a=0.1,
            g_d_minus_b=-0.5, se_d_minus_b=0.2,
            g_interaction=-1.19, se_interaction=0.25,
            n_pairs=30,
        ),
    )
    r = Factorial2x2Result(
        per_env=per_env,
        arm_a='vanilla_n1', arm_b='ddqn_n1',
        arm_c='vanilla_n3', arm_d='ddqn_n3',
        source='outcome',
    )
    out = _coerce_value(r)
    assert isinstance(out, dict)
    nested = out['per_env']
    assert isinstance(nested, list)
    assert len(nested) == 1
    inner = nested[0]
    assert isinstance(inner, dict)
    assert inner['env_name'] == 'FourRooms-misc'
    assert inner['g_interaction'] == -1.19


def test_coerce_verdict_distribution_with_property_per_env() -> None:
    per_env = MappingProxyType({
        'env_a': VerdictCounts(held=10, invariant_violation=0,
                                power_insufficient=5, other=0,
                                total=15, dominant='held'),
    })
    r = VerdictDistributionResult(per_env=per_env)
    out = _coerce_value(r)
    assert isinstance(out, dict)
    assert 'env_a' in out['per_env']
    inner = out['per_env']['env_a']
    assert inner['held'] == 10
    # Nested VerdictCounts properties surface
    assert inner['held_fraction'] == pytest.approx(10 / 15)


def test_coerce_gate_result_to_dict() -> None:
    g = GateResult(
        gate_name='exogenous_source', level=GateLevel.WARN,
        passed=False, message='source is endogenous',
    )
    out = _coerce_value(g)
    assert isinstance(out, dict)
    assert out['gate_name'] == 'exogenous_source'
    assert out['level'] == 'warn'
    assert out['passed'] is False
    assert out['message'] == 'source is endogenous'


# ============ build_report — orchestration ============


def _empty_cells() -> pl.DataFrame:
    return pl.DataFrame({
        'id': ['c0'], 'env_name': ['Pong'], 'arm_key': ['baseline'],
    })


def test_build_report_skeleton(tmp_path: Path) -> None:
    """Empty bridges + no errors → minimal RunReport with provenance."""
    report = build_report(
        hypothesis_module_name='experiments.findings.fake',
        bridges=(),
        results={},
        errors={},
        cells=_empty_cells(),
        cache_path=None,
        measurable_signatures={'mock': 'abc123'},
        repo_root=tmp_path,
    )
    assert report.schema_version == SCHEMA_VERSION
    assert report.hypothesis_module == 'experiments.findings.fake'
    assert report.n_cells_total == 1
    assert report.cache_path is None
    assert dict(report.measurable_signatures) == {'mock': 'abc123'}
    assert report.bridges == ()
    assert report.errored_bridges == ()
    assert report.timestamp_utc.endswith('+00:00')


def test_build_report_with_errored_bridge() -> None:
    """Bridges that raised get an ErroredBridgeEntry — captures
    bug authoring failures that previously vanished into stderr."""
    from corroborate.bridge.bridge import claim_bridge
    from corroborate.graph.causal import Direction, Tier

    @claim_bridge(
        source='outcome', target='outcome', direction=Direction.DIRECT,
        tier=Tier.ASSOCIATIONAL,
    )
    def my_bridge(paired_g: PairedGResult) -> Verdict:  # pragma: no cover
        return Verdict.HELD

    err = RuntimeError("typo'd column name")
    try:
        raise err
    except RuntimeError as caught:
        report = build_report(
            hypothesis_module_name='exp.fake',
            bridges=(my_bridge,),
            results={},
            errors={my_bridge.name: caught},
            cells=_empty_cells(),
            cache_path=None,
            measurable_signatures={},
            repo_root=Path.cwd(),
        )
    assert len(report.errored_bridges) == 1
    e = report.errored_bridges[0]
    assert e.bridge_name == my_bridge.name
    assert e.error_type == 'RuntimeError'
    assert "typo'd" in e.error_message
    assert 'RuntimeError' in e.traceback_repr


# ============ write_report — atomic + deterministic ============


def _synthetic_report() -> RunReport:
    return RunReport(
        schema_version=SCHEMA_VERSION,
        hypothesis_module='exp.fake',
        timestamp_utc='2026-05-06T00:00:00+00:00',
        git_commit='abcdef123456',
        n_cells_total=10,
        cache_path='experiments/data/cache/fake.parquet',
        measurable_signatures=MappingProxyType({'a': '1', 'b': '2'}),
        bridges=(),
        errored_bridges=(),
    )


def test_write_report_byte_deterministic(tmp_path: Path) -> None:
    """Same RunReport written twice → identical bytes (sort_keys + no
    ordering hazards)."""
    report = _synthetic_report()
    p1 = tmp_path / 'a.run.json'
    p2 = tmp_path / 'b.run.json'
    write_report(report, p1)
    write_report(report, p2)
    assert p1.read_bytes() == p2.read_bytes()


def test_write_report_atomic_no_tmp_left_behind(tmp_path: Path) -> None:
    report = _synthetic_report()
    p = tmp_path / 'r.run.json'
    write_report(report, p)
    assert p.exists()
    assert not p.with_suffix(p.suffix + '.tmp').exists()


def test_write_report_is_valid_json_with_no_nan_literal(tmp_path: Path) -> None:
    """`allow_nan=False` is safe because NaN/inf were converted to
    strings by `_coerce_value` upstream. Output must round-trip
    through the most-strict JSON parser (no Python-extension literals)."""
    @dataclass(frozen=True, slots=True)
    class _ResultWithNaN:
        x: float
        y: float

    @dataclass(frozen=True, slots=True)
    class _BridgeOnly:
        name: str
        analysis_results: dict[str, _ResultWithNaN]

    # Build a synthetic report with a NaN field
    report = RunReport(
        schema_version=SCHEMA_VERSION,
        hypothesis_module='exp.fake',
        timestamp_utc='2026-05-06T00:00:00+00:00',
        git_commit=None,
        n_cells_total=0,
        cache_path=None,
        measurable_signatures=MappingProxyType({}),
        bridges=(
            BridgeReportEntry(
                bridge_name='b',
                source_name='X', target_name='Y',
                direction='direct', tier='associational',
                pair_by=('seed',), predicted_direction=None,
                scope_repr=None, params=MappingProxyType({}),
                n_cells_pre_scope=0, n_cells_in_scope=0,
                verdict='held',
                analysis_results=MappingProxyType({
                    'paired_g': MappingProxyType({
                        'g': 'NaN', 'se': 'Infinity',
                    }),
                }),
                warnings=(), blocked_by=None,
            ),
        ),
        errored_bridges=(),
    )
    p = tmp_path / 'r.run.json'
    write_report(report, p)
    text = p.read_text()
    # Strict json.loads (no NaN literal) must succeed
    parsed = json.loads(text)
    assert parsed['bridges'][0]['analysis_results']['paired_g']['g'] == 'NaN'
    assert parsed['bridges'][0]['analysis_results']['paired_g']['se'] == 'Infinity'
    # Trailing newline (git-friendly)
    assert text.endswith('\n')


def test_write_report_human_readable_indent(tmp_path: Path) -> None:
    report = _synthetic_report()
    p = tmp_path / 'r.run.json'
    write_report(report, p)
    text = p.read_text()
    # Indent=2 → multi-line output
    assert '\n  ' in text
    assert text.count('\n') > 5


# ============ End-to-end via existing analytic substrate ============


def test_end_to_end_serializes_full_bridge_evaluation() -> None:
    """Mock-shaped: build a BridgeEvaluation directly, run through
    `_build_bridge_entry` + `_coerce_value`, assert the JSON shape
    matches what reviewers will see."""
    from corroborate.bridge.bridge import claim_bridge
    from corroborate.graph.causal import Direction, Tier
    from corroborate.runner.report import _build_bridge_entry

    @claim_bridge(
        source='outcome', target='outcome', direction=Direction.DIRECT,
        tier=Tier.ASSOCIATIONAL, pair_by=('seed',),
        predicted_direction='a_gt_b',
    )
    def my_bridge(paired_g: PairedGResult) -> Verdict:  # pragma: no cover
        return Verdict.HELD

    paired_g = PairedGResult(
        g=1.5, se=0.3, mean_diff=2.0, mean_diff_se=0.4,
        n_pairs=30, n_treatment=30, n_baseline=30,
        helped_fraction=0.8, pair_by=('seed',), measurable='outcome',
        treatment_arm='ddqn', baseline_arm='vanilla',
    )
    ev = BridgeEvaluation(
        bridge_name=my_bridge.name,
        verdict=Verdict.HELD,
        analysis_results=MappingProxyType({'paired_g': paired_g}),
        warnings=(),
        blocked_by=None,
    )
    cells = _empty_cells()
    entry = _build_bridge_entry(my_bridge, ev, cells)
    coerced = _coerce_value(entry)
    assert isinstance(coerced, dict)
    assert coerced['verdict'] == 'held'
    assert coerced['direction'] == 'direct'
    assert coerced['tier'] == 'associational'
    assert coerced['predicted_direction'] == 'a_gt_b'
    assert coerced['pair_by'] == ['seed']
    pg = coerced['analysis_results']['paired_g']
    assert pg['g'] == 1.5
    assert 'p_value' in pg  # property included
    assert pg['n_pairs'] == 30
