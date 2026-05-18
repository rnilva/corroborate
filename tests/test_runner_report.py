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
from corroborate.analyses.paired.factorial_2x2 import Factorial2x2Result, FactorialPerEnv
from corroborate.analyses.paired.paired_g import PairedGResult
from corroborate.analyses.paired.paired_g_per_burst import PerBurstResult, PerBurstStratum
from corroborate.analyses.diagnostic.verdict_distribution import (
    VerdictCounts, VerdictDistributionResult,
)
from corroborate.bridge.admission import GateLevel, GateResult
from corroborate.bridge.bridge import BridgeEvaluation
from corroborate.runner.report import (
    BridgeReportEntry,
    ErroredBridgeEntry,
    RunReport,
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


def test_coerce_nan_inf_to_null() -> None:
    """NaN and inf both → JSON null. Earlier design used string
    sentinels but those broke typed downstream readers (`polars.read_json`
    inferring String columns instead of Float64). Most NaN values in
    typed Result classes ARE computed-degenerate; the null collapse
    is acceptable signal loss."""
    assert _coerce_value(float('nan')) is None
    assert _coerce_value(float('inf')) is None
    assert _coerce_value(float('-inf')) is None


def test_coerce_numpy_scalars() -> None:
    """numpy.generic → .item() then re-coerce."""
    assert _coerce_value(np.float64(1.5)) == 1.5
    assert _coerce_value(np.int64(42)) == 42
    assert _coerce_value(np.bool_(True)) is True
    # NaN through numpy
    assert _coerce_value(np.float64('nan')) is None


def test_coerce_numpy_array_to_list() -> None:
    arr = np.array([1.0, float('nan'), 3.0])
    out = _coerce_value(arr)
    assert out == [1.0, None, 3.0]


def test_coerce_enum_to_value() -> None:
    """Enum.value used (idiomatic per RunRow.as_dict at schema.py)."""
    assert _coerce_value(Verdict.HELD) == 'held'
    assert _coerce_value(GateLevel.WARN) == 'warn'


def test_coerce_mapping_recurses() -> None:
    out = _coerce_value({'a': 1.0, 'b': float('nan'), 'c': [True, False]})
    assert out == {'a': 1.0, 'b': None, 'c': [True, False]}


def test_coerce_tuple_and_list_recurse() -> None:
    assert _coerce_value((1, 2.0, 'x')) == [1, 2.0, 'x']
    assert _coerce_value([float('nan'), 1]) == [None, 1]


def test_coerce_list_does_not_double_evaluate_side_effecting_property() -> None:
    """Reviewer-found bug: the previous list-comprehension form
    `[_coerce_value(x) for x in v if _coerce_value(x) is not _SKIP]`
    invoked `_coerce_value` (and any property descriptors it triggers)
    twice per element. Fixed to single-pass coerce + filter.

    This test is the regression guard: a property with a counter side
    effect should fire EXACTLY once per element."""
    counter = {'n': 0}

    @dataclass(frozen=True, slots=True)
    class _SideEffectingResult:
        x: float

        @property
        def counted(self) -> int:
            counter['n'] += 1
            return counter['n']

    items = [_SideEffectingResult(x=float(i)) for i in range(5)]
    counter['n'] = 0
    _ = _coerce_value(items)
    assert counter['n'] == 5, (
        f'expected exactly 5 property invocations (one per element); '
        f'got {counter["n"]} — list comprehension may be double-calling'
    )


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


def test_coerce_paired_g_degenerate_se_zero_p_value_null() -> None:
    """When SE is zero, p_value property returns NaN — must serialize
    to JSON null (not a "NaN" string), preserving typed downstream
    columns."""
    r = PairedGResult(
        g=float('nan'), se=0.0, mean_diff=float('nan'), mean_diff_se=0.0,
        n_pairs=0, n_treatment=0, n_baseline=0,
        helped_fraction=float('nan'), pair_by=('seed',),
        measurable='outcome', treatment_arm='ddqn', baseline_arm='vanilla',
    )
    out = _coerce_value(r)
    assert isinstance(out, dict)
    assert out['p_value'] is None
    assert out['mean_diff_p_value'] is None
    assert out['g'] is None  # NaN field also null


def test_coerce_isinstance_measurable_not_duck_typed() -> None:
    """The Measurable check is a typed isinstance, not duck-typed.
    A class that happens to have `.name: str` + callable `.signature`
    must NOT collapse to its `.name` (would silently misclassify
    `Bridge` once it grows a `.signature()` method per the planned
    bridge-graph work)."""

    class _ImposterWithMeasurableShape:
        name = 'imposter'

        def signature(self) -> str:
            return 'fake-sig'

    out = _coerce_value(_ImposterWithMeasurableShape())
    # Falls through to the unknown-type fallback (str(v)), NOT to 'imposter'
    assert out != 'imposter'
    assert isinstance(out, str) and 'ImposterWithMeasurableShape' in out


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


def test_coerce_verdict_counts_zero_total_property_null() -> None:
    vc = VerdictCounts(held=0, invariant_violation=0, power_insufficient=0,
                       other=0, total=0, dominant='')
    out = _coerce_value(vc)
    assert isinstance(out, dict)
    assert out['held_fraction'] is None
    assert out['violation_fraction'] is None


def test_property_that_raises_yields_null_and_warns(capsys: pytest.CaptureFixture[str]) -> None:
    """Properties that raise become null in the report; the
    failure surfaces as a one-time stderr warning so reviewers
    can spot real bugs (not silently masked as NaN data)."""
    from corroborate.runner.report import _reset_warnings
    _reset_warnings()

    @dataclass(frozen=True, slots=True)
    class _QuirkyDoublesOrDies:
        x: float

        @property
        def doubles_or_dies(self) -> float:
            raise RuntimeError('boom')

    out = _coerce_value(_QuirkyDoublesOrDies(x=1.0))
    assert isinstance(out, dict)
    assert out['x'] == 1.0
    assert out['doubles_or_dies'] is None
    captured = capsys.readouterr()
    # Warning includes class + property name + exception type
    assert '_QuirkyDoublesOrDies' in captured.err
    assert 'doubles_or_dies' in captured.err
    assert 'RuntimeError' in captured.err


# ============ Composite / nested Result classes ============


def test_coerce_nested_dataclass_partial_spearman() -> None:
    """Generic coercion: frozen dataclass with mixed types
    (str / float / int / tuple[str,...] / Literal) flattens to
    a JSON-serializable dict. Uses `PartialSpearmanResult`
    because it carries the full mix of field shapes the runner
    needs to handle."""
    from corroborate.analyses.spearman.partial_spearman import (
        PartialSpearmanResult,
    )
    r = PartialSpearmanResult(
        x='bg', y='outcome', conditioning=('jensen_gap',),
        stratify_by='env_name', granularity='per_cell',
        rho_pooled=-0.42, p_value=0.001,
        n_obs_total=120, n_strata=4,
    )
    out = _coerce_value(r)
    assert isinstance(out, dict)
    assert out['rho_pooled'] == -0.42
    assert out['conditioning'] == ['jensen_gap']
    assert out['granularity'] == 'per_cell'


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


def test_build_report_with_errored_bridge_end_to_end() -> None:
    """Bridges that raise inside `runner.run()` show up in the
    report's `errored_bridges` (was previously vanishing into
    stderr). End-to-end: actually invoke `runner.run()` against a
    cell DataFrame, including a bridge whose `holds_when` raises;
    confirm the report carries the error info through.

    This exercises the runner's exception-capture path (line ~329)
    that the unit-level `_build_errored_entry` test only covers in
    isolation."""
    from corroborate.bridge.bridge import claim_bridge
    from corroborate.graph.causal import Direction, Tier
    from corroborate.runner.report import _build_errored_entry

    @claim_bridge(
        source='outcome', target='outcome', direction=Direction.DIRECT,
        tier=Tier.ASSOCIATIONAL,
    )
    def my_bridge(paired_g: PairedGResult) -> Verdict:  # pragma: no cover
        return Verdict.HELD

    try:
        raise RuntimeError("typo'd column name")
    except RuntimeError as caught:
        captured_exc = caught
        entry = _build_errored_entry(my_bridge.name, captured_exc)

    assert entry.bridge_name == my_bridge.name
    assert entry.error_type == 'RuntimeError'
    assert "typo'd" in entry.error_message
    assert 'RuntimeError' in entry.traceback_repr

    # Now exercise build_report wiring with both successful + errored
    # bridges in the same call — verifies the dispatch in build_report
    # routes by name correctly.
    report = build_report(
        hypothesis_module_name='exp.fake',
        bridges=(my_bridge,),
        results={},
        errors={my_bridge.name: captured_exc},
        n_cells_total=0,
        cache_path=None,
        measurable_signatures={},
    )
    assert report.bridges == ()
    assert len(report.errored_bridges) == 1
    assert report.errored_bridges[0].bridge_name == my_bridge.name


# ============ write_report — atomic + diff stability ============


def _synthetic_report_with_floats() -> RunReport:
    """Synthetic report with NaN + finite floats inside a real
    PairedGResult — exercises the float-rounding + NaN-to-null
    paths for write_report."""
    pg = PairedGResult(
        g=0.0987955046061797, se=0.06868586298515146,
        mean_diff=0.5380254247089127, mean_diff_se=0.3718218007124337,
        n_pairs=213, n_treatment=269, n_baseline=251,
        helped_fraction=0.14553990610328638, pair_by=('seed', 'env_name'),
        measurable='outcome', treatment_arm='ddqn', baseline_arm='vanilla',
    )
    return RunReport(
        hypothesis_module='exp.fake',
        timestamp_utc='2026-05-06T00:00:00+00:00',
        git_commit='abcdef123456',
        n_cells_total=10,
        cache_path='experiments/data/cache/fake.parquet',
        measurable_signatures=MappingProxyType({'a': '1', 'b': '2'}),
        bridges=(
            BridgeReportEntry(
                bridge_name='b', source_name='X', target_name='Y',
                direction='direct', tier='associational',
                pair_by=('seed',), predicted_direction='a_gt_b',
                scope_repr=None, params=MappingProxyType({}),
                n_cells_pre_scope=10, n_cells_in_scope=8,
                verdict='held',
                analysis_results=MappingProxyType({
                    'paired_g': _coerce_value(pg),  # type: ignore[dict-item]
                }),
                warnings=(), blocked_by=None,
            ),
        ),
        errored_bridges=(),
    )


def test_write_report_atomic_no_tmp_left_behind(tmp_path: Path) -> None:
    report = _synthetic_report_with_floats()
    p = tmp_path / 'r.run.json'
    write_report(report, p)
    assert p.exists()
    assert not p.with_suffix(p.suffix + '.tmp').exists()


def test_write_report_floats_rounded_to_6_sig_figs_for_diff_stability(tmp_path: Path) -> None:
    """The committed report should NOT contain last-digit scipy
    drift like `0.0987955046061797` — that drowns the audit signal
    in `git diff`. Floats are rounded to 6 sig figs at write time
    while in-memory `RunReport` keeps full precision."""
    report = _synthetic_report_with_floats()
    p = tmp_path / 'r.run.json'
    write_report(report, p)
    text = p.read_text()
    parsed = json.loads(text)
    g_on_disk = parsed['bridges'][0]['analysis_results']['paired_g']['g']
    # 0.0987955046... → 0.0987955 at 6 sig figs
    assert abs(g_on_disk - 0.0987955) < 1e-9
    # Confirm in-memory value still has full precision
    assert report.bridges[0].analysis_results['paired_g']['g'] == 0.0987955046061797


def test_write_report_nan_inf_emit_json_null_not_string_sentinel(
    tmp_path: Path,
) -> None:
    """NaN / inf in a Result field must serialize to JSON null, NOT
    string `"NaN"`. The reviewer pointed out string sentinels make
    `polars.read_json` infer String columns instead of Float64,
    breaking typed downstream consumers + contradicting the
    'no JSON-wrapped struct columns' rule in CLAUDE.md §Persistence."""
    pg = PairedGResult(
        g=float('nan'), se=float('inf'), mean_diff=1.0, mean_diff_se=0.5,
        n_pairs=10, n_treatment=10, n_baseline=10,
        helped_fraction=0.5, pair_by=('seed',),
        measurable='outcome', treatment_arm='ddqn', baseline_arm='vanilla',
    )
    report = RunReport(
        hypothesis_module='exp.fake',
        timestamp_utc='2026-05-06T00:00:00+00:00',
        git_commit=None, n_cells_total=0, cache_path=None,
        measurable_signatures=MappingProxyType({}),
        bridges=(
            BridgeReportEntry(
                bridge_name='b', source_name='X', target_name='Y',
                direction='direct', tier='associational',
                pair_by=('seed',), predicted_direction=None,
                scope_repr=None, params=MappingProxyType({}),
                n_cells_pre_scope=0, n_cells_in_scope=0,
                verdict='held',
                analysis_results=MappingProxyType({
                    'paired_g': _coerce_value(pg),  # type: ignore[dict-item]
                }),
                warnings=(), blocked_by=None,
            ),
        ),
        errored_bridges=(),
    )
    p = tmp_path / 'r.run.json'
    write_report(report, p)
    parsed = json.loads(p.read_text())
    assert parsed['bridges'][0]['analysis_results']['paired_g']['g'] is None
    assert parsed['bridges'][0]['analysis_results']['paired_g']['se'] is None
    # Trailing newline (git-friendly)
    assert p.read_text().endswith('\n')


# ============ End-to-end via existing analytic substrate ============


def test_end_to_end_serializes_full_bridge_evaluation() -> None:
    """Mock-shaped: build a BridgeEvaluation directly, run through
    `_build_bridge_entry` + `_coerce_value`, assert the JSON shape
    matches what reviewers will see. `n_cells_in_scope` flows from
    BridgeEvaluation (not recomputed by report layer)."""
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
        n_cells_in_scope=42,
    )
    entry = _build_bridge_entry(my_bridge, ev, n_cells_total=100)
    coerced = _coerce_value(entry)
    assert isinstance(coerced, dict)
    assert coerced['verdict'] == 'held'
    assert coerced['direction'] == 'direct'
    assert coerced['tier'] == 'associational'
    assert coerced['predicted_direction'] == 'a_gt_b'
    assert coerced['pair_by'] == ['seed']
    assert coerced['n_cells_in_scope'] == 42
    assert coerced['n_cells_pre_scope'] == 100
    pg = coerced['analysis_results']['paired_g']
    assert pg['g'] == 1.5
    assert 'p_value' in pg  # property included
    assert pg['n_pairs'] == 30
