"""Admission gates — phase 1+2 coverage.

Each framework auto-gate gets one positive (gate fires) and one
negative (gate stays silent) test. Per-bridge `gates=(...)` tuple
gets its own pair: per-bridge gates are *appended* to the auto
list, and the overall gate-loop short-circuits the bridge body on
the first BLOCK and accumulates WARN/INFO results onto
`BridgeEvaluation.warnings`.

The gate primitives are pure callables — these tests exercise
each gate both *directly* (calling the gate fn on a synthetic
Bridge) and *integrated* (through `evaluate()` so the wrap-up
behaviour — verdict landing on `INADMISSIBLE`, warnings flowing
through, body skipped on BLOCK — is covered too)."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import polars as pl
import pytest

# Importing analyses populates the registry (paired_g lives there).
import corroborate.analyses  # noqa: F401  # pyright: ignore[reportUnusedImport]

from corroborate.analyses.paired_g import PairedGResult
from corroborate.bridge.admission import (
    AUTO_GATES,
    AdmissionGate,
    GateLevel,
    GateResult,
    distinct_arms,
    exogenous_scope,
    exogenous_source,
    no_predicted_direction,
)
from corroborate.bridge.bridge import (
    Bridge,
    Direction,
    Tier,
    claim_bridge,
    evaluate,
)
from corroborate.bridge.verdict import Verdict
from corroborate.core.claim import claim
from corroborate.core.intervention import DoEffect, Intervention


# ---------- shared fixtures ----------


@claim
def _treatment_op(x: int) -> int:
    return x


@claim
def _baseline_op(x: int) -> int:
    return x


_TREATMENT_ARMS: tuple[Intervention, ...] = (
    Intervention(slot_path='op', replacement=_treatment_op),
)
_BASELINE_ARMS: tuple[Intervention, ...] = (
    Intervention(slot_path='op', replacement=_baseline_op),
)
_INTERVENTION = DoEffect(
    treatment=_TREATMENT_ARMS, baseline=_BASELINE_ARMS,
)
_TREATMENT_KEY = _INTERVENTION.treatment_arm_key()
_BASELINE_KEY = _INTERVENTION.baseline_arm_key()


def _synthetic_cells(
    *, n_seeds: int = 30,
) -> list[dict[str, object]]:
    """Strong-signal corpus paired by seed."""
    import random
    rng = random.Random(0)
    out: list[dict[str, object]] = []
    for s in range(n_seeds):
        out.append({
            'arm_key': _TREATMENT_KEY,
            'seed': s,
            'env_name': 'TestEnv',
            'eval_best_burst_mean': 1.0 + rng.gauss(0, 0.1),
        })
        out.append({
            'arm_key': _BASELINE_KEY,
            'seed': s,
            'env_name': 'TestEnv',
            'eval_best_burst_mean': 0.0 + rng.gauss(0, 0.1),
        })
    return out


_NO_CELLS: Sequence[Mapping[str, object]] = ()


# ---------- distinct_arms (BLOCK) ----------


def test_distinct_arms_blocks_self_vs_self_do_effect() -> None:
    """Both arms empty tuples → both `treatment_arm_key()` and
    `baseline_arm_key()` resolve to `'baseline'`. The contrast is
    structurally self-vs-self; replaces the legacy paired_g
    runtime ValueError with a clean Verdict.INADMISSIBLE."""
    self_vs_self = DoEffect(treatment=(), baseline=())
    bridge = Bridge(
        name='self_vs_self',
        source=self_vs_self,
        target='eval_best_burst_mean',
        tier=Tier.INTERVENTIONAL,
        holds_when=lambda paired_g: Verdict.HELD,
    )
    result = distinct_arms(bridge, _NO_CELLS)
    assert result is not None
    assert result.level is GateLevel.BLOCK
    assert result.passed is False
    assert 'self-vs-self' in result.message


def test_distinct_arms_silent_when_arms_differ() -> None:
    bridge = Bridge(
        name='ok',
        source=_INTERVENTION,
        target='eval_best_burst_mean',
        tier=Tier.INTERVENTIONAL,
        holds_when=lambda paired_g: Verdict.HELD,
    )
    assert distinct_arms(bridge, _NO_CELLS) is None


def test_distinct_arms_silent_for_non_do_effect_source() -> None:
    """Gate is do-effect-specific. String-sourced bridges aren't
    contrast bridges; gate doesn't apply."""
    bridge = Bridge(
        name='associational',
        source='jensen_gap',
        target='eval_best_burst_mean',
        tier=Tier.ASSOCIATIONAL,
        holds_when=lambda paired_g: Verdict.HELD,
    )
    assert distinct_arms(bridge, _NO_CELLS) is None


# ---------- exogenous_source (BLOCK) ----------


def test_exogenous_source_blocks_hp_string_on_interventional() -> None:
    """Tier.INTERVENTIONAL with `source='gamma'` (an HP, not a
    registered measurable nor standard metadata). BLOCK."""
    bridge = Bridge(
        name='gamma_to_outcome',
        source='gamma',
        target='eval_best_burst_mean',
        tier=Tier.INTERVENTIONAL,
        holds_when=lambda paired_g: Verdict.HELD,
    )
    result = exogenous_source(bridge, _NO_CELLS)
    assert result is not None
    assert result.level is GateLevel.BLOCK
    assert "'gamma'" in result.message


def test_exogenous_source_passes_for_do_effect() -> None:
    """DoEffect's Interventions carry Claim-shaped replacements
    (callable). Passes the gate."""
    bridge = Bridge(
        name='ok',
        source=_INTERVENTION,
        target='eval_best_burst_mean',
        tier=Tier.INTERVENTIONAL,
        holds_when=lambda paired_g: Verdict.HELD,
    )
    assert exogenous_source(bridge, _NO_CELLS) is None


@pytest.fixture
def _registered_endogenous() -> str:
    """Register a single measurable for the duration of this test;
    the auto-pruning happens at process exit, but the registry's
    same-name idempotence means subsequent re-registrations are
    no-ops."""
    from corroborate.measurables import (
        Measurable, register, registered_names,
    )
    name = '_test_admission_endogenous'
    if name in registered_names():
        return name
    fn = Measurable(
        fn=lambda record: 0.0,
        name=name,
        reads=(),
    )
    register(fn)
    return name


def test_exogenous_source_passes_for_registered_measurable(
    _registered_endogenous: str,
) -> None:
    """A registered measurable name on a string source is the
    canonical endogenous slot. Passes."""
    bridge = Bridge(
        name='ok',
        source=_registered_endogenous,
        target='eval_best_burst_mean',
        tier=Tier.INTERVENTIONAL,
        holds_when=lambda paired_g: Verdict.HELD,
    )
    assert exogenous_source(bridge, _NO_CELLS) is None


def test_exogenous_source_passes_on_associational_tier() -> None:
    """Gate fires only on Tier.INTERVENTIONAL. ASSOCIATIONAL
    bridges may take any source — that's the rung-1 freedom."""
    bridge = Bridge(
        name='ok',
        source='gamma',
        target='eval_best_burst_mean',
        tier=Tier.ASSOCIATIONAL,
        holds_when=lambda paired_g: Verdict.HELD,
    )
    assert exogenous_source(bridge, _NO_CELLS) is None


# ---------- exogenous_scope (WARN) ----------


def test_exogenous_scope_warns_on_hp_only_filter() -> None:
    """`scope = pl.col('lr') == 1e-4` references only an HP leaf;
    no endogenous columns in the predicate → WARN."""
    bridge = Bridge(
        name='hp_envelope',
        source='jensen_gap',
        target='eval_best_burst_mean',
        tier=Tier.ASSOCIATIONAL,
        scope=pl.col('lr') == 1e-4,
        holds_when=lambda paired_g: Verdict.HELD,
    )
    result = exogenous_scope(bridge, _NO_CELLS)
    assert result is not None
    assert result.level is GateLevel.WARN
    assert "'lr'" in result.message


def test_exogenous_scope_silent_on_endogenous_filter() -> None:
    """`scope = pl.col('env_name') == 'TestEnv'` references a
    standard-metadata column → endogenous, no WARN."""
    bridge = Bridge(
        name='env_filter',
        source='jensen_gap',
        target='eval_best_burst_mean',
        tier=Tier.ASSOCIATIONAL,
        scope=pl.col('env_name') == 'TestEnv',
        holds_when=lambda paired_g: Verdict.HELD,
    )
    assert exogenous_scope(bridge, _NO_CELLS) is None


def test_exogenous_scope_silent_when_mixed() -> None:
    """A predicate that references both — `env_name == 'X' & lr ==
    1e-4` — is fine. The principled scope axis is the env_name
    branch; the HP branch refines, doesn't substitute."""
    bridge = Bridge(
        name='mixed',
        source='jensen_gap',
        target='eval_best_burst_mean',
        scope=(pl.col('env_name') == 'TestEnv') & (pl.col('lr') == 1e-4),
        holds_when=lambda paired_g: Verdict.HELD,
    )
    assert exogenous_scope(bridge, _NO_CELLS) is None


def test_exogenous_scope_silent_when_no_scope() -> None:
    bridge = Bridge(
        name='no_scope',
        source='jensen_gap',
        target='eval_best_burst_mean',
        scope=None,
        holds_when=lambda paired_g: Verdict.HELD,
    )
    assert exogenous_scope(bridge, _NO_CELLS) is None


# ---------- no_predicted_direction (INFO) ----------


def test_no_predicted_direction_fires_when_absent() -> None:
    bridge = Bridge(
        name='no_dir',
        source='jensen_gap',
        target='eval_best_burst_mean',
        predicted_direction=None,
        holds_when=lambda paired_g: Verdict.HELD,
    )
    result = no_predicted_direction(bridge, _NO_CELLS)
    assert result is not None
    assert result.level is GateLevel.INFO


def test_no_predicted_direction_silent_when_set() -> None:
    bridge = Bridge(
        name='with_dir',
        source='jensen_gap',
        target='eval_best_burst_mean',
        predicted_direction='a_lt_b',
        holds_when=lambda paired_g: Verdict.HELD,
    )
    assert no_predicted_direction(bridge, _NO_CELLS) is None


# ---------- AUTO_GATES wiring ----------


def test_auto_gates_tuple_contains_all_four() -> None:
    """Sanity: the framework's auto-gate list is exactly the
    four functions Phase 2 ships."""
    assert AUTO_GATES == (
        distinct_arms, exogenous_source, exogenous_scope,
        no_predicted_direction,
    )


# ---------- evaluate() integration: BLOCK short-circuits ----------


def test_evaluate_returns_inadmissible_on_block() -> None:
    """Self-vs-self DoEffect → INADMISSIBLE; body never runs;
    `blocked_by` carries the gate's GateResult."""
    self_vs_self = DoEffect(treatment=(), baseline=())

    body_calls: list[int] = []

    @claim_bridge(
        source=self_vs_self,
        target='eval_best_burst_mean',
        direction=Direction.DIRECT,
        tier=Tier.INTERVENTIONAL,
        predicted_direction='a_gt_b',
    )
    def blocked_bridge(paired_g: PairedGResult) -> Verdict:
        body_calls.append(1)
        del paired_g
        return Verdict.HELD

    out = evaluate(blocked_bridge, _synthetic_cells())
    assert out.verdict is Verdict.INADMISSIBLE
    assert out.blocked_by is not None
    assert out.blocked_by.gate_name == 'distinct_arms'
    assert out.blocked_by.level is GateLevel.BLOCK
    assert out.analysis_results == {}
    assert body_calls == [], 'body must not run when gate blocks'


# ---------- evaluate() integration: WARN flows through ----------


def test_evaluate_propagates_warnings() -> None:
    """HP-only scope + missing predicted_direction: bridge body
    runs (no BLOCK), verdict comes from `holds_when`, but BOTH
    warnings sit on `BridgeEvaluation.warnings`."""

    @claim_bridge(
        source=_INTERVENTION,
        target='eval_best_burst_mean',
        direction=Direction.DIRECT,
        tier=Tier.INTERVENTIONAL,
        scope=pl.col('lr') == 1e-4,
        predicted_direction=None,
    )
    def warn_bridge(paired_g: PairedGResult) -> Verdict:
        if paired_g.n_pairs == 0:
            return Verdict.POWER_INSUFFICIENT
        return Verdict.HELD

    cells = _synthetic_cells()
    for c in cells:
        c['lr'] = 1e-4
    out = evaluate(warn_bridge, cells)
    # body executed (lr filter let cells through, signal strong)
    assert out.verdict is Verdict.HELD
    # both gates fired
    gate_names = {w.gate_name for w in out.warnings}
    assert 'exogenous_scope' in gate_names
    assert 'no_predicted_direction' in gate_names
    levels = {w.level for w in out.warnings}
    assert GateLevel.WARN in levels
    assert GateLevel.INFO in levels


# ---------- per-bridge gates are appended ----------


def test_per_bridge_gate_appended_and_blocks() -> None:
    """A bridge author can declare extra gates via
    `gates=(...)`. They run after auto-gates; same BLOCK
    semantics."""

    def always_block(
        bridge: Bridge,
        cells: Sequence[Mapping[str, object]],
    ) -> GateResult | None:
        del bridge, cells
        return GateResult(
            gate_name='custom',
            level=GateLevel.BLOCK,
            passed=False,
            message='custom block',
        )

    custom: tuple[AdmissionGate, ...] = (always_block,)

    @claim_bridge(
        source=_INTERVENTION,
        target='eval_best_burst_mean',
        direction=Direction.DIRECT,
        tier=Tier.INTERVENTIONAL,
        predicted_direction='a_gt_b',
        gates=custom,
    )
    def with_custom_gate(paired_g: PairedGResult) -> Verdict:
        del paired_g
        return Verdict.HELD

    out = evaluate(with_custom_gate, _synthetic_cells())
    assert out.verdict is Verdict.INADMISSIBLE
    assert out.blocked_by is not None
    assert out.blocked_by.gate_name == 'custom'
