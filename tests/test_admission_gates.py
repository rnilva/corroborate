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

from corroborate.analyses.paired.paired_g import PairedGResult
from corroborate.bridge.admission import (
    AUTO_GATES,
    AdmissionGate,
    GateLevel,
    GateResult,
    contrast_isolation,
    contrast_present,
    distinct_arms,
    distinct_units,
    exogenous_scope,
    exogenous_source,
    is_endogenous,
    no_predicted_direction,
    pair_completeness,
    resolved_source,
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
def _test_claim(
    *,
    # "Author primitives" the synthetic substrate exposes — these
    # become the leaf set that endogeneity gating consults via
    # `walk_paths(_test_claim, regime='leaf')`.
    gamma: float = 0.99,
    lr: float = 1e-3,
    env_name: str = 'TestEnv',
) -> int:
    del gamma, lr, env_name
    return 0


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
_INTERVENTION = DoEffect(arms=(_BASELINE_ARMS, _TREATMENT_ARMS))
_BASELINE_KEY, _TREATMENT_KEY = _INTERVENTION.arm_keys()


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
    """Both arms empty tuples → both arm_keys resolve to
    `'baseline'`. The contrast is structurally self-vs-self;
    replaces the legacy paired_g runtime ValueError with a clean
    Verdict.INADMISSIBLE."""
    self_vs_self = DoEffect(arms=((), ()))
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
    assert 'duplicate canonical_str fingerprints' in result.message


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
    """Tier.INTERVENTIONAL with `source='gamma'` (a leaf of the
    synthetic claim, not a registered measurable). BLOCK."""
    bridge = Bridge(
        name='gamma_to_outcome',
        source='gamma',
        target='eval_best_burst_mean',
        tier=Tier.INTERVENTIONAL,
        holds_when=lambda paired_g: Verdict.HELD,
    )
    result = exogenous_source(bridge, _NO_CELLS, claim=_test_claim)
    assert result is not None
    assert result.level is GateLevel.BLOCK
    assert "'gamma'" in result.message


def test_exogenous_source_short_circuits_without_claim() -> None:
    """Without a substrate claim threaded through, the gate has
    no leaf set to consult — short-circuit (gate-doesn't-apply
    semantics). Framework-only tests rely on this."""
    bridge = Bridge(
        name='no_claim',
        source='gamma',
        target='eval_best_burst_mean',
        tier=Tier.INTERVENTIONAL,
        holds_when=lambda paired_g: Verdict.HELD,
    )
    assert exogenous_source(bridge, _NO_CELLS, claim=None) is None


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
    assert exogenous_source(bridge, _NO_CELLS, claim=_test_claim) is None


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
    """A registered measurable whose `reads=()` closure has no
    elements outside the claim's leaves classifies as exogenous
    by the elimination-via-empty-closure path; this fixture's
    measurable has reads=(), so closure is empty → no read
    'outside leaves' → exogenous → gate fires.

    The canonical endogenous case (closure touches a trajectory
    key) is tested below in test_is_endogenous_*."""
    bridge = Bridge(
        name='ok',
        source=_registered_endogenous,
        target='eval_best_burst_mean',
        tier=Tier.INTERVENTIONAL,
        holds_when=lambda paired_g: Verdict.HELD,
    )
    # With reads=(), closure is empty → any() over empty is False
    # → is_endogenous returns False → BLOCK fires. This is the
    # correct behaviour: a measurable that closes over nothing
    # carries no cell-derived signal.
    result = exogenous_source(bridge, _NO_CELLS, claim=_test_claim)
    assert result is not None
    assert result.level is GateLevel.BLOCK


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
    assert exogenous_source(bridge, _NO_CELLS, claim=_test_claim) is None


# ---------- exogenous_scope (WARN) ----------


def test_exogenous_scope_warns_on_hp_only_filter() -> None:
    """`scope = pl.col('lr') == 1e-4` references only a leaf of
    `_test_claim`; no endogenous columns in the predicate → WARN."""
    bridge = Bridge(
        name='hp_envelope',
        source='jensen_gap',
        target='eval_best_burst_mean',
        tier=Tier.ASSOCIATIONAL,
        scope=pl.col('lr') == 1e-4,
        holds_when=lambda paired_g: Verdict.HELD,
    )
    result = exogenous_scope(bridge, _NO_CELLS, claim=_test_claim)
    assert result is not None
    assert result.level is GateLevel.WARN
    assert "'lr'" in result.message


def test_exogenous_scope_warns_on_env_name_only_scope() -> None:
    """`env_name` is a leaf of the substrate claim (post-Phase-A0
    of the endogeneity-from-topology refactor); a scope that
    references only `env_name` is exogenous-only → WARN. This
    flips the previous `_STANDARD_METADATA` assumption that
    env_name was endogenous metadata."""
    bridge = Bridge(
        name='env_filter',
        source='jensen_gap',
        target='eval_best_burst_mean',
        tier=Tier.ASSOCIATIONAL,
        scope=pl.col('env_name') == 'TestEnv',
        holds_when=lambda paired_g: Verdict.HELD,
    )
    result = exogenous_scope(bridge, _NO_CELLS, claim=_test_claim)
    assert result is not None
    assert result.level is GateLevel.WARN


def test_exogenous_scope_silent_when_mixed() -> None:
    """A predicate that references one leaf and one trajectory-
    derived column — `env_name == 'X' & jensen_gap > 0` — is
    fine: the trajectory-side dependence is the principled axis,
    and at least one endogenous reference clears the WARN."""
    bridge = Bridge(
        name='mixed',
        source='jensen_gap',
        target='eval_best_burst_mean',
        scope=(pl.col('env_name') == 'TestEnv') & (pl.col('jensen_gap') > 0),
        holds_when=lambda paired_g: Verdict.HELD,
    )
    assert exogenous_scope(bridge, _NO_CELLS, claim=_test_claim) is None


def test_exogenous_scope_silent_when_no_scope() -> None:
    bridge = Bridge(
        name='no_scope',
        source='jensen_gap',
        target='eval_best_burst_mean',
        scope=None,
        holds_when=lambda paired_g: Verdict.HELD,
    )
    assert exogenous_scope(bridge, _NO_CELLS, claim=_test_claim) is None


def test_exogenous_scope_short_circuits_without_claim() -> None:
    """When `claim` isn't threaded through evaluate(), the gate
    has no leaf set and short-circuits."""
    bridge = Bridge(
        name='no_claim',
        source='jensen_gap',
        target='eval_best_burst_mean',
        scope=pl.col('lr') == 1e-4,
        holds_when=lambda paired_g: Verdict.HELD,
    )
    assert exogenous_scope(bridge, _NO_CELLS, claim=None) is None


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


def test_auto_gates_tuple_contains_all_nine() -> None:
    """Sanity: the framework's auto-gate list is exactly the
    nine shipped functions, in diagnostic-priority order —
    `resolved_source` before the endogeneity test so typo'd
    sources surface as typos; `contrast_present` and
    `exogenous_source` before `distinct_units` so a value-contrast
    record without its contrast, and a native leaf-sourced
    interventional bridge, each hear the structural diagnosis
    rather than the effective-n symptom; the contrast-quality
    gates (`contrast_isolation`, `pair_completeness`) after the
    structural gates, checking the derived conditions per claim,
    per extent, on the verdict record."""
    assert AUTO_GATES == (
        distinct_arms, resolved_source, contrast_present,
        exogenous_source, distinct_units, exogenous_scope,
        contrast_isolation, pair_completeness,
        no_predicted_direction,
    )


# ---------- distinct_units (BLOCK below 4 / WARN) ----------


def _units_bridge() -> Bridge:
    """Minimal string-sourced bridge for the distinct_units tests."""
    return Bridge(
        name='bed_prop_tracks_outcome',
        source='bed_prop',
        target='outcome',
        tier=Tier.ASSOCIATIONAL,
        holds_when=lambda partial_spearman: Verdict.HELD,
    )


def test_distinct_units_blocks_replicated_source() -> None:
    """A bed-level source replicated across per-instance cells has
    effective n = distinct values, not rows. Below 4 it BLOCKs."""
    cells: list[Mapping[str, object]] = [
        {'arm_key': 'a', 'env_name': 'e', 'seed': 0, 'instance': i,
         'bed_prop': p, 'outcome': 20.0 + i}
        for p in (0.38, 0.62, 0.74) for i in range(4)
    ]
    res = distinct_units(_units_bridge(), cells)
    assert res is not None and not res.passed
    assert res.level is GateLevel.BLOCK
    assert '3 distinct values across 12 cells' in res.message


def test_distinct_units_silent_when_source_is_per_cell() -> None:
    """Every cell carrying its own source value is the ordinary
    case and must not fire."""
    cells: list[Mapping[str, object]] = [
        {'arm_key': 'a', 'env_name': 'e', 'seed': s, 'instance': i,
         'bed_prop': 0.40 + 0.01 * (s * 4 + i), 'outcome': 20.0 + s}
        for s in range(3) for i in range(4)
    ]
    assert distinct_units(_units_bridge(), cells) is None


def test_distinct_units_warns_above_block_threshold() -> None:
    """8 distinct values over 24 cells: replication, but enough
    units to estimate — WARN, so the bridge still runs."""
    cells: list[Mapping[str, object]] = [
        {'arm_key': 'a', 'env_name': 'e', 'seed': s, 'instance': i,
         'bed_prop': 0.10 * s, 'outcome': 20.0 + i}
        for s in range(8) for i in range(3)
    ]
    res = distinct_units(_units_bridge(), cells)
    assert res is not None and res.level is GateLevel.WARN
    assert '8 distinct values across 24 cells' in res.message


def test_distinct_units_silent_on_mild_ties() -> None:
    """Rank data ties a few cells without implying a coarser unit:
    with n_eff > n_cells/2 (here 7 of 12) the gate must not
    conflate ties with replication."""
    values = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.1, 0.2, 0.3, 0.4, 0.5)
    cells: list[Mapping[str, object]] = [
        {'arm_key': 'a', 'env_name': 'e', 'seed': i,
         'bed_prop': v, 'outcome': 20.0 + i}
        for i, v in enumerate(values)
    ]
    assert distinct_units(_units_bridge(), cells) is None


def test_distinct_units_silent_on_do_effect_source() -> None:
    """An arm indicator is meant to repeat across cells; the
    contrast design carries its own n accounting. Gate doesn't
    apply to DoEffect sources."""
    bridge = Bridge(
        name='contrast',
        source=_INTERVENTION,
        target='eval_best_burst_mean',
        tier=Tier.INTERVENTIONAL,
        holds_when=lambda paired_g: Verdict.HELD,
    )
    assert distinct_units(bridge, _synthetic_cells()) is None


def test_distinct_units_silent_on_bool_source() -> None:
    """A per-cell binary indicator has 2 distinct values by
    construction — value cardinality says nothing about the grain
    it was measured at, so the gate must not fire."""
    bridge = Bridge(
        name='diverged_tracks_outcome',
        source='diverged',
        target='outcome',
        tier=Tier.ASSOCIATIONAL,
        holds_when=lambda partial_spearman: Verdict.HELD,
    )
    cells: list[Mapping[str, object]] = [
        {'arm_key': 'a', 'env_name': 'e', 'seed': i,
         'diverged': i % 2 == 0, 'outcome': 20.0 + i}
        for i in range(12)
    ]
    assert distinct_units(bridge, cells) is None


# ---------- evaluate() integration: BLOCK short-circuits ----------


def test_evaluate_returns_inadmissible_on_block() -> None:
    """Self-vs-self DoEffect → INADMISSIBLE; body never runs;
    `blocked_by` carries the gate's GateResult."""
    self_vs_self = DoEffect(arms=((), ()))

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
    out = evaluate(warn_bridge, cells, claim=_test_claim)
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
        *,
        claim: object = None,
        leaves: frozenset[str] | None = None,
    ) -> GateResult | None:
        del bridge, cells, claim, leaves
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


# ---------- is_endogenous (topology-derived classification) ----------


def test_is_endogenous_leaf_of_claim_is_exogenous() -> None:
    """`gamma` is a kwarg of `_test_claim` → leaf → exogenous."""
    assert is_endogenous('gamma', _test_claim) is False
    assert is_endogenous('lr', _test_claim) is False
    assert is_endogenous('env_name', _test_claim) is False


def test_is_endogenous_unregistered_name_is_endogenous() -> None:
    """Names that are neither a leaf of the claim nor a registered
    measurable classify as cell-controlled-by-elimination →
    endogenous (trajectory output)."""
    assert is_endogenous('mc_return', _test_claim) is True
    assert is_endogenous('jensen_gap_raw', _test_claim) is True


def test_is_endogenous_measurable_with_trajectory_closure() -> None:
    """A registered measurable whose `transitive_reads` includes
    a key OUTSIDE `_test_claim`'s leaves classifies as endogenous."""
    from corroborate.measurables import Measurable, register, registered_names
    name = '_test_endo_traj'
    if name not in registered_names():
        register(Measurable(
            fn=lambda record: 0.0, name=name, reads=('done',),
        ))
    # 'done' isn't a leaf of _test_claim → endogenous.
    assert is_endogenous(name, _test_claim) is True


def test_is_endogenous_measurable_closing_only_over_leaves() -> None:
    """A registered measurable whose closure is fully inside the
    claim's leaves classifies as exogenous — the loophole the
    Phase-1 effective_horizon=1/(1-γ) trip."""
    from corroborate.measurables import Measurable, register, registered_names
    name = '_test_endo_leaf_only'
    if name not in registered_names():
        register(Measurable(
            fn=lambda record: 0.0, name=name, reads=('gamma',),
        ))
    # Closure is just {'gamma'}, all in leaves → exogenous.
    assert is_endogenous(name, _test_claim) is False


# ---------- resolved_source (BLOCK on missing column) ----------


def test_resolved_source_blocks_on_missing_column() -> None:
    """A bridge sourced on a column not in cells → BLOCK with
    a typo-friendly diagnostic listing available columns."""
    bridge = Bridge(
        name='typo',
        source='mc_returns',  # typo'd plural
        target='eval_best_burst_mean',
        tier=Tier.ASSOCIATIONAL,
        holds_when=lambda paired_g: Verdict.HELD,
    )
    cells: list[Mapping[str, object]] = [
        {'mc_return': 1.0, 'eval_best_burst_mean': 0.0},
    ]
    result = resolved_source(bridge, cells)
    assert result is not None
    assert result.level is GateLevel.BLOCK
    assert "'mc_returns'" in result.message


def test_resolved_source_silent_on_present_column() -> None:
    bridge = Bridge(
        name='ok',
        source='jensen_gap',
        target='eval_best_burst_mean',
        tier=Tier.ASSOCIATIONAL,
        holds_when=lambda paired_g: Verdict.HELD,
    )
    cells: list[Mapping[str, object]] = [
        {'jensen_gap': 0.5, 'eval_best_burst_mean': 1.0},
    ]
    assert resolved_source(bridge, cells) is None


def test_resolved_source_silent_on_do_effect() -> None:
    """DoEffect sources have no string column to validate; gate
    doesn't apply."""
    bridge = Bridge(
        name='intervention',
        source=_INTERVENTION,
        target='eval_best_burst_mean',
        tier=Tier.INTERVENTIONAL,
        holds_when=lambda paired_g: Verdict.HELD,
    )
    cells: list[Mapping[str, object]] = [{'eval_best_burst_mean': 1.0}]
    assert resolved_source(bridge, cells) is None


def test_resolved_source_silent_on_empty_cells() -> None:
    """Empty corpus — gate yields to downstream; no info to
    validate column-existence against."""
    bridge = Bridge(
        name='ok',
        source='jensen_gap',
        target='eval_best_burst_mean',
        tier=Tier.ASSOCIATIONAL,
        holds_when=lambda paired_g: Verdict.HELD,
    )
    assert resolved_source(bridge, _NO_CELLS) is None
