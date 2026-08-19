"""End-to-end smoke for the claim-bridge + analysis pattern.

Synthetic corpus where the truth is known: treatment cells have
outcome ≈ 1.0, baseline cells have outcome ≈ 0.0, paired by
seed across 30 seeds. A bridge that asserts "treatment moves
outcome by g > 0.3 with p < 0.05" should HELD; one with a
much higher threshold should NO_EFFECT.

The smoke proves:
- The `@analysis` decorator + registry round-trip.
- The `@claim_bridge` decorator factory accepts bridge metadata as
  kwargs and produces a typed Bridge.
- `evaluate(bridge, cells)` resolves each fixture (parameter
  without a default) by name against the analysis registry,
  parameterises from the bridge's structural fields + params
  bag, runs, injects, and routes through the bridge body.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import cast

import polars as pl
import pytest

# Importing analyses populates the registry.
import corroborate.analyses  # noqa: F401  # pyright: ignore[reportUnusedImport]

from corroborate.analyses.paired.paired_g import PairedGResult
from corroborate.bridge.bridge import (
    Bridge, Direction, Tier, claim_bridge, evaluate,
    measurable_names_for_bridges,
)
from corroborate.core.claim import claim
from corroborate.core.intervention import DoEffect, Intervention
from corroborate.bridge.verdict import Verdict


# Synthetic intervention arms for the top-level bridges in this file.
@claim
def _treatment_op(x: int) -> int:
    return x


@claim
def _baseline_op(x: int) -> int:
    return x


@claim
def _external_program(gamma: float) -> float:
    return gamma


_TREATMENT_ARMS: tuple[Intervention, ...] = (
    Intervention(slot_path='op', replacement=_treatment_op),
)
_BASELINE_ARMS: tuple[Intervention, ...] = (
    Intervention(slot_path='op', replacement=_baseline_op),
)
INTERVENTION = DoEffect(arms=(_BASELINE_ARMS, _TREATMENT_ARMS))
_BASELINE_KEY, _TREATMENT_KEY = INTERVENTION.arm_keys()


def _synthetic_cells(
    *,
    n_seeds: int = 30,
    treatment_mean: float = 1.0,
    baseline_mean: float = 0.0,
    noise: float = 0.1,
) -> list[dict[str, object]]:
    import random
    rng = random.Random(0)
    out: list[dict[str, object]] = []
    for s in range(n_seeds):
        out.append({
            'arm_key': _TREATMENT_KEY,
            'seed': s,
            'env_name': 'TestEnv',
            'eval_best_burst_mean': (
                treatment_mean + rng.gauss(0, noise)
            ),
        })
        out.append({
            'arm_key': _BASELINE_KEY,
            'seed': s,
            'env_name': 'TestEnv',
            'eval_best_burst_mean': (
                baseline_mean + rng.gauss(0, noise)
            ),
        })
    return out


def test_paired_g_analysis_runs_directly() -> None:
    """The analysis is callable on its own — no bridge needed.
    Cell-level filtering lives upstream on `Bridge.scope`;
    when calling paired_g.fn directly the test pre-filters cells."""
    from corroborate.analyses.paired.paired_g import paired_g
    cells = [
        c for c in _synthetic_cells()
        if c.get('env_name') == 'TestEnv'
    ]
    result = paired_g.fn(
        cells,
        treatment_arm=_TREATMENT_KEY,
        baseline_arm=_BASELINE_KEY,
        pair_by=('seed',),
        source='eval_best_burst_mean',
    )
    assert isinstance(result, PairedGResult)
    assert result.n_pairs == 30
    assert result.g > 3.0, f'expected g > 3, got {result.g}'
    assert result.p_value < 1e-6


def test_analysis_registered_globally() -> None:
    """Importing `corroborate.analyses` registers `paired_g` under
    its function name — the lookup the bridge resolver uses."""
    from corroborate.bridge.analysis import (
        get_registered, registered_names,
    )
    assert 'paired_g' in registered_names()
    assert get_registered('paired_g') is not None


@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=pl.col('env_name') == 'TestEnv',
)
def treatment_helps_outcome(
    paired_g: PairedGResult,
) -> Verdict:
    if paired_g.n_pairs < 10:
        return Verdict.POWER_INSUFFICIENT
    if paired_g.g > 0.3 and paired_g.p_value < 0.05:
        return Verdict.HELD
    return Verdict.NO_EFFECT


def test_bridge_held_under_explicit_threshold() -> None:
    """Authoring path: the bridge is declared via decorator args.
    Synthetic corpus with strong effect → HELD."""
    cells = _synthetic_cells()
    out = evaluate(treatment_helps_outcome, cells)
    assert out.verdict == Verdict.HELD
    assert out.bridge_name == 'treatment_helps_outcome'
    pg = cast(PairedGResult, out.analysis_results['paired_g'])
    assert pg.n_pairs == 30
    assert pg.g > 0.3


def test_bridge_no_effect_when_signal_absent() -> None:
    """Treatment ≈ baseline → NO_EFFECT."""
    cells = _synthetic_cells(treatment_mean=0.0, baseline_mean=0.0)
    out = evaluate(treatment_helps_outcome, cells)
    assert out.verdict == Verdict.NO_EFFECT


def _contrast_cells(
    seeds: range = range(12),
) -> list[dict[str, object]]:
    """External-style cells: parameter value + outcome, no arm
    vocabulary of any kind."""
    rows: list[dict[str, object]] = []
    for seed in seeds:
        rows.extend((
            {
                'id': f'control-{seed}',
                'seed': seed,
                'gamma': 0.8,
                'return_mean': float(seed) / 100.0,
            },
            {
                'id': f'high-{seed}',
                'seed': seed,
                'gamma': 0.99,
                'return_mean': 1.0 + float(seed) / 100.0,
            },
        ))
    return rows


_LEAVES = frozenset({'gamma'})
GAMMA_EFFECT = DoEffect.from_values(
    source='gamma', reference=0.8, treatment=0.99,
)


@claim_bridge(
    source=GAMMA_EFFECT,
    target='return_mean',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    pair_by=('seed',),
    predicted_direction='a_gt_b',
)
def higher_gamma_value_helps(
    paired_g: PairedGResult,
) -> Verdict:
    return Verdict.HELD if paired_g.mean_diff > 0.0 else Verdict.NO_EFFECT


def test_value_effect_uses_declared_values_and_symbolic_arms() -> None:
    """The DoEffect fixes membership and orientation independently
    of observed support. Without a leaf registry the claim still
    evaluates, but the record shows what went unchecked: knob-ness
    of the declared source (WARN) and registered-configuration
    balance (INFO). With the registry, both clear."""
    cells = _contrast_cells()
    out = evaluate(higher_gamma_value_helps, cells)
    assert out.verdict is Verdict.HELD
    result = cast(PairedGResult, out.analysis_results['paired_g'])
    assert result.measurable == 'return_mean'
    assert result.baseline_arm == 'baseline'
    assert result.treatment_arm == 'treatment'
    assert result.n_pairs == 12
    # The caller's cells are not mutated by the label derivation.
    assert all('contrast_arm' not in c for c in cells)
    knobness = next(
        warning for warning in out.warnings
        if warning.gate_name == 'exogenous_source'
    )
    assert knobness.level.value == 'warn'
    assert 'no configuration registry' in knobness.message
    balance = next(
        warning for warning in out.warnings
        if warning.gate_name == 'contrast_isolation'
    )
    assert balance.level.value == 'info'
    assert 'unassessed' in balance.message

    registered = evaluate(higher_gamma_value_helps, cells, leaves=_LEAVES)
    assert registered.verdict is Verdict.HELD
    assert registered.warnings == ()


def test_value_effect_preserves_declared_categorical_orientation() -> None:
    """Orientation is declared, not lexicographically inferred."""
    cells: list[Mapping[str, object]] = []
    for seed in range(6):
        cells.extend((
            {'id': f'adam-{seed}', 'seed': seed, 'optimizer': 'adam',
             'return_mean': float(seed)},
            {'id': f'sgd-{seed}', 'seed': seed, 'optimizer': 'sgd',
             'return_mean': 10.0 + float(seed)},
        ))

    @claim_bridge(
        source=DoEffect.from_values(
            source='optimizer', reference='sgd', treatment='adam',
        ),
        target='return_mean',
        direction=Direction.DIRECT,
        tier=Tier.INTERVENTIONAL,
        pair_by=('seed',),
        predicted_direction='a_gt_b',
    )
    def _adam_is_lower(paired_g: PairedGResult) -> Verdict:
        return Verdict.HELD if paired_g.mean_diff < 0 else Verdict.NO_EFFECT

    out = evaluate(_adam_is_lower, cells)
    assert out.verdict is Verdict.HELD
    result = cast(PairedGResult, out.analysis_results['paired_g'])
    assert result.baseline_arm == 'baseline'
    assert result.treatment_arm == 'treatment'
    assert result.mean_diff < 0


def test_joint_value_effect_owns_its_lockstep_partner() -> None:
    """Two knobs assigned together: declared jointly, the lockstep
    partner is part of the estimand — no isolation flag. Declared
    singly, the same registered partner is a co-varied knob and
    the contrast blocks; the fix the message names is widening the
    declaration."""
    cells: list[dict[str, object]] = []
    for seed in range(6):
        cells.extend((
            {'id': f'lo-{seed}', 'seed': seed, 'gamma': 0.8,
             'n_step': 1, 'return_mean': float(seed)},
            {'id': f'hi-{seed}', 'seed': seed, 'gamma': 0.99,
             'n_step': 3, 'return_mean': 10.0 + float(seed)},
        ))
    registry = frozenset({'gamma', 'n_step'})

    @claim_bridge(
        source=DoEffect.from_values(
            reference={'gamma': 0.8, 'n_step': 1},
            treatment={'gamma': 0.99, 'n_step': 3},
        ),
        target='return_mean',
        direction=Direction.DIRECT,
        tier=Tier.INTERVENTIONAL,
        pair_by=('seed',),
        predicted_direction='a_gt_b',
    )
    def _joint_helps(paired_g: PairedGResult) -> Verdict:
        return Verdict.HELD if paired_g.mean_diff > 0 else Verdict.NO_EFFECT

    joint = evaluate(_joint_helps, cells, leaves=registry)
    assert joint.verdict is Verdict.HELD
    assert joint.blocked_by is None
    assert joint.warnings == ()
    result = cast(PairedGResult, joint.analysis_results['paired_g'])
    assert result.n_pairs == 6

    single = evaluate(higher_gamma_value_helps, cells, leaves=registry)
    assert single.verdict is Verdict.INADMISSIBLE
    assert single.blocked_by is not None
    assert single.blocked_by.gate_name == 'contrast_isolation'
    assert 'n_step' in single.blocked_by.message
    assert 'widen the declaration' in single.blocked_by.message


def test_value_effect_does_not_format_float_arm_identity() -> None:
    """Nearby floats remain distinct even though `:g` would render
    both as `0.8`; arm identity is the symbolic declaration role."""
    effect = DoEffect.from_values(
        source='gamma', reference=0.8000001, treatment=0.8000002,
    )

    @claim_bridge(
        source=effect,
        target='return_mean',
        pair_by=('seed',),
    )
    def _nearby_values(paired_g: PairedGResult) -> Verdict:
        return Verdict.HELD if paired_g.mean_diff > 0 else Verdict.NO_EFFECT

    cells: list[Mapping[str, object]] = []
    for seed in range(4):
        cells.extend((
            {'seed': seed, 'gamma': 0.8000001, 'return_mean': 0.0},
            {'seed': seed, 'gamma': 0.8000002, 'return_mean': 1.0},
        ))
    out = evaluate(_nearby_values, cells, leaves=frozenset({'gamma'}))
    result = cast(PairedGResult, out.analysis_results['paired_g'])
    assert out.verdict is Verdict.HELD
    assert result.n_pairs == 4
    assert result.baseline_arm == 'baseline'
    assert result.treatment_arm == 'treatment'


def test_value_effect_source_measurable_is_materialised() -> None:
    """The graph endpoint is a do-node, but the data-side source is
    still part of the bridge's transitive measurable requirements."""
    from corroborate.measurables import (
        Measurable, register, registered_names,
    )
    name = '_test_value_effect_source_measurable'
    if name not in registered_names():
        register(Measurable(
            fn=lambda record: record.get('raw_gamma'),
            name=name,
            reads=('raw_gamma',),
        ))

    @claim_bridge(
        source=DoEffect.from_values(
            source=name, reference=0.8, treatment=0.99,
        ),
        target='return_mean',
        tier=Tier.INTERVENTIONAL,
    )
    def _uses_derived_source(paired_g: PairedGResult) -> Verdict:
        return Verdict.HELD

    assert name in measurable_names_for_bridges((_uses_derived_source,))


def test_native_claim_reasserts_the_endogenous_source_doctrine() -> None:
    """Native string/measurable sources still use claim topology."""

    @claim_bridge(
        source='gamma',
        target='return_mean',
        tier=Tier.INTERVENTIONAL,
    )
    def _native_leaf(paired_g: PairedGResult) -> Verdict:
        return Verdict.HELD

    out = evaluate(
        _native_leaf,
        _contrast_cells(),
        claim=_external_program,
    )
    assert out.verdict is Verdict.INADMISSIBLE
    assert out.blocked_by is not None
    assert out.blocked_by.gate_name == 'exogenous_source'


def test_bare_interventional_without_any_registry_blocks() -> None:
    """An external string cannot infer an intervention from data."""

    @claim_bridge(
        source='gamma',
        target='return_mean',
        tier=Tier.INTERVENTIONAL,
    )
    def _undeclared_effect(paired_g: PairedGResult) -> Verdict:
        return Verdict.HELD

    out = evaluate(
        _undeclared_effect, _contrast_cells(), leaves=_LEAVES,
    )
    assert out.verdict is Verdict.INADMISSIBLE
    assert out.blocked_by is not None
    assert out.blocked_by.gate_name == 'exogenous_source'
    assert 'DoEffect.from_values' in out.blocked_by.message


def test_measured_source_cannot_claim_interventional() -> None:
    """With a leaf registry present, an interventional bridge
    sourcing on a column OUTSIDE it — a measurement — blocks: a
    measured column cannot be the assigned parameter."""

    @claim_bridge(
        source='return_mean',
        target='return_mean',
        direction=Direction.DIRECT,
        tier=Tier.INTERVENTIONAL,
        pair_by=('seed',),
        predicted_direction='a_gt_b',
    )
    def _measured_source(paired_g: PairedGResult) -> Verdict:
        return Verdict.HELD

    out = evaluate(_measured_source, _contrast_cells(), leaves=_LEAVES)
    assert out.verdict is Verdict.INADMISSIBLE
    assert out.blocked_by is not None
    assert out.blocked_by.gate_name == 'exogenous_source'
    assert 'DoEffect.from_values' in out.blocked_by.message


def test_associational_tier_keeps_the_effective_n_guard() -> None:
    """Condition derivation is interventional-tier semantics; an
    associational bridge uses the leaf as a REGRESSOR, where k
    distinct values really do bound effective n — the
    `distinct_units` guard stays."""

    @claim_bridge(
        source='gamma',
        target='return_mean',
        direction=Direction.DIRECT,
        tier=Tier.ASSOCIATIONAL,
        pair_by=('seed',),
        predicted_direction='a_gt_b',
    )
    def _correlational_gamma(paired_g: PairedGResult) -> Verdict:
        return Verdict.HELD

    out = evaluate(_correlational_gamma, _contrast_cells(), leaves=_LEAVES)
    assert out.verdict is Verdict.INADMISSIBLE
    assert out.blocked_by is not None
    assert out.blocked_by.gate_name == 'distinct_units'


def test_contrast_present_blocks_a_record_without_the_contrast() -> None:
    """One value of the source in scope means no contrast exists
    in this record — INADMISSIBLE, not a raise: the record can
    grow the missing condition later."""
    only_baseline = [
        c for c in _contrast_cells() if c['gamma'] == 0.8
    ]
    out = evaluate(higher_gamma_value_helps, only_baseline)
    assert out.verdict is Verdict.INADMISSIBLE
    assert out.blocked_by is not None
    assert out.blocked_by.gate_name == 'contrast_present'
    assert "missing declared arm(s) ['treatment']" in out.blocked_by.message


def test_value_effect_reports_an_absent_source_column() -> None:
    cells = [
        {key: value for key, value in cell.items() if key != 'gamma'}
        for cell in _contrast_cells()
    ]
    out = evaluate(higher_gamma_value_helps, cells)
    assert out.verdict is Verdict.INADMISSIBLE
    assert out.blocked_by is not None
    assert out.blocked_by.gate_name == 'resolved_source'
    assert "column 'gamma'" in out.blocked_by.message


def test_value_contrast_rejects_reserved_column_collision() -> None:
    cells = _contrast_cells()
    cells[0]['contrast_arm'] = 'stale-label'
    with pytest.raises(ValueError, match='reserved column'):
        _ = evaluate(higher_gamma_value_helps, cells)


def test_value_contrast_pools_batches_of_the_same_study() -> None:
    """Evidence is a live record: a later batch of seeds joins the
    earlier one and the fixed declaration selects both batches.
    The verdict recomputes over all matching evidence; unrelated
    values may accumulate in the same record."""
    pooled = _contrast_cells(range(12)) + _contrast_cells(range(12, 20))
    # A third accumulated arm belongs to a different estimand and is
    # ignored without invalidating this declaration.
    pooled.extend({
        'id': f'mid-{seed}',
        'seed': seed,
        'gamma': 0.9,
        'return_mean': 1000.0,
    } for seed in range(20))
    out = evaluate(higher_gamma_value_helps, pooled)
    assert out.verdict is Verdict.HELD
    assert out.n_cells_in_scope == 40
    result = cast(PairedGResult, out.analysis_results['paired_g'])
    assert result.n_pairs == 20


def test_value_effect_ignores_heterogeneous_rows_without_source() -> None:
    """A live heterogeneous stream may contain unrelated row shapes."""
    cells: list[Mapping[str, object]] = [
        {'id': 'unrelated', 'other_measurement': 1.0},
        *_contrast_cells(range(3)),
    ]

    out = evaluate(higher_gamma_value_helps, cells)

    assert out.verdict is Verdict.HELD
    assert out.blocked_by is None
    assert out.n_cells_in_scope == 6
    result = cast(PairedGResult, out.analysis_results['paired_g'])
    assert result.n_pairs == 3


def test_contrast_isolation_blocks_a_co_varied_leaf() -> None:
    """A leaf may vary globally yet still differ inside every pair;
    isolation is checked per pairing unit rather than per arm."""
    cells = _contrast_cells()
    for cell in cells:
        seed = cast(int, cell['seed'])
        cell['exploration_schedule'] = seed + (
            100 if cell['gamma'] == 0.99 else 0
        )
    out = evaluate(
        higher_gamma_value_helps,
        cells,
        leaves=frozenset({'gamma', 'exploration_schedule'}),
    )
    assert out.verdict is Verdict.INADMISSIBLE
    assert out.blocked_by is not None
    assert out.blocked_by.gate_name == 'contrast_isolation'
    assert 'exploration_schedule' in out.blocked_by.message


def test_missing_leaf_does_not_hide_observed_pair_imbalance() -> None:
    cells = _contrast_cells()
    for cell in cells:
        cell['schedule'] = 'high' if cell['gamma'] == 0.99 else 'low'
    out = evaluate(
        higher_gamma_value_helps,
        cells,
        leaves=frozenset({'gamma', 'schedule', 'absent_setting'}),
    )
    assert out.verdict is Verdict.INADMISSIBLE
    assert out.blocked_by is not None
    assert out.blocked_by.gate_name == 'contrast_isolation'
    assert 'schedule' in out.blocked_by.message


def test_contrast_isolation_warns_on_an_unregistered_rider() -> None:
    """A column outside the registry moving in lockstep with the
    contrast may be a producer label (harmless) or an unlogged
    knob (a confound) — only the author can say, so it lands on
    the record as a WARN, never a block and never silence."""
    cells = _contrast_cells()
    for cell in cells:
        cell['physical_arm'] = 'exp' if cell['gamma'] == 0.99 else 'ctl'
    out = evaluate(higher_gamma_value_helps, cells, leaves=_LEAVES)
    assert out.verdict is Verdict.HELD
    rider = next(
        w for w in out.warnings if w.gate_name == 'contrast_isolation'
    )
    assert rider.level.value == 'warn'
    assert 'physical_arm' in rider.message
    # Outcome columns vary within each arm and never read as riders.
    assert 'return_mean' not in rider.message


def test_contrast_isolation_balances_registered_leaves_per_pair() -> None:
    """A setting may vary between seeds provided each matched pair
    shares it."""
    cells = _contrast_cells()
    for cell in cells:
        cell['learning_rate'] = 1e-3 / (1 + cast(int, cell['seed']))
    out = evaluate(
        higher_gamma_value_helps,
        cells,
        leaves=frozenset({'gamma', 'learning_rate'}),
    )
    assert out.verdict is Verdict.HELD
    assert out.blocked_by is None
    assert all(
        warning.gate_name != 'contrast_isolation'
        for warning in out.warnings
    )


def test_contrast_isolation_detects_balance_at_one_complete_pair() -> None:
    """A registered mismatch is observable even with one complete pair."""
    cells = _contrast_cells(range(1))
    for cell in cells:
        cell['schedule'] = 'high' if cell['gamma'] == 0.99 else 'low'
    out = evaluate(
        higher_gamma_value_helps,
        cells,
        leaves=frozenset({'gamma', 'schedule'}),
    )
    assert out.verdict is Verdict.INADMISSIBLE
    assert out.blocked_by is not None
    assert out.blocked_by.gate_name == 'contrast_isolation'


def test_pair_completeness_warns_on_missing_partner() -> None:
    """Paired analyses drop incomplete pairs silently; the gate
    makes the drop visible on the verdict record."""
    cells = [
        c for c in _contrast_cells(range(3)) if c['id'] != 'high-2'
    ]
    out = evaluate(higher_gamma_value_helps, cells)
    assert out.verdict is Verdict.HELD
    warning = next(
        w for w in out.warnings if w.gate_name == 'pair_completeness'
    )
    assert '1 of 3' in warning.message
    result = cast(PairedGResult, out.analysis_results['paired_g'])
    assert result.n_pairs == 2


@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=pl.col('env_name') == 'TestEnv',
)
def want_30_pairs(
    paired_g: PairedGResult,
) -> Verdict:
    if paired_g.n_pairs < 30:
        return Verdict.POWER_INSUFFICIENT
    if paired_g.g > 0.3 and paired_g.p_value < 0.05:
        return Verdict.HELD
    return Verdict.NO_EFFECT


def test_bridge_power_insufficient_with_few_seeds() -> None:
    """Few pairs → POWER_INSUFFICIENT. The threshold is encoded in
    the bridge body, not the analysis."""
    cells = _synthetic_cells(n_seeds=5)
    out = evaluate(want_30_pairs, cells)
    assert out.verdict == Verdict.POWER_INSUFFICIENT


def test_scope_with_repeated_column_reference_in_predicate() -> None:
    """Regression: `filter_cells` must dedupe
    `expr.meta.root_names()` before resolving missing columns.

    A bridge predicate like `(pl.col('n_step') == 1) | (pl.col('n_step') == 3)`
    references the same column TWICE — `root_names()` returns
    `['n_step', 'n_step']`. If the column is truly missing (not
    a registered measurable, not in the cell DataFrame), the
    pre-fix code emitted `[pl.lit(None).alias('n_step'),
    pl.lit(None).alias('n_step')]` which polars rejects with:
        ComputeError: the name 'n_step' passed to
        `LazyFrame.with_columns` is duplicate.

    After the fix (`list(dict.fromkeys(...))`), the duplicate is
    collapsed before the missing-column resolution runs.

    Construction: cells WITHOUT an `n_step` column; bridge's scope
    references `n_step` twice. The bridge should evaluate without
    crashing. The filtered cell-set is empty (n_step is missing
    everywhere → null → predicate False), so the verdict routes
    through whatever the bridge body returns on n_pairs=0.
    """
    @claim_bridge(
        source=INTERVENTION,
        target='eval_best_burst_mean',
        scope=(pl.col('n_step') == 1) | (pl.col('n_step') == 3),
    )
    def repeats_n_step(
        paired_g: PairedGResult,
    ) -> Verdict:
        if paired_g.n_pairs == 0:
            return Verdict.POWER_INSUFFICIENT
        return Verdict.HELD

    cells = _synthetic_cells()    # no n_step column
    # Pre-fix: this `evaluate` call raised polars ComputeError.
    out = evaluate(repeats_n_step, cells)
    assert out.verdict == Verdict.POWER_INSUFFICIENT
    # n_step missing → all rows null on that column → both branches
    # of the disjunction are null → predicate is null → polars
    # `filter` excludes them → 0 cells in scope.
    assert out.n_cells_in_scope == 0


def test_unknown_fixture_raises() -> None:
    """A fixture parameter (no default) that doesn't match a
    registered analysis fails fast at evaluation."""
    @claim_bridge(source='x', target='y')
    def broken(
        not_a_real_analysis: object,
    ) -> Verdict:
        del not_a_real_analysis
        return Verdict.HELD

    # `x` and `y` columns must exist in cells so RESOLVED_SOURCE
    # passes — the test exercises the fixture-resolution path,
    # not gate behaviour.
    cells: list[Mapping[str, object]] = [{'x': 0.0, 'y': 0.0, 'env_name': 'X'}]
    with pytest.raises(KeyError, match='not_a_real_analysis'):
        _ = evaluate(broken, cells)


def test_bridge_carries_typed_intervention() -> None:
    """A do-effect bridge declares source=DoEffect(...) in the
    decorator; the framework routes it to the structural
    `Bridge.source` field. The framework can then emit a
    `do(treatment|vs=baseline) → target` graph edge.

    `DoEffect` carries Intervention tuples per arm; arm keys
    derive from `combined_arm_key` (canonical_str) — the
    structural link between the do-contrast and the claim graph
    the implementation's intervention_arms produce."""
    @claim_bridge(
        source=INTERVENTION,
        target='eval_best_burst_mean',
        direction=Direction.DIRECT,
        tier=Tier.INTERVENTIONAL,
    )
    def carries_intervention(
        paired_g: PairedGResult,
    ) -> Verdict:
        del paired_g
        return Verdict.HELD

    assert isinstance(carries_intervention, Bridge)
    assert isinstance(carries_intervention.source, DoEffect)
    assert carries_intervention.source.arms == (
        _BASELINE_ARMS, _TREATMENT_ARMS,
    )
    arm_keys = carries_intervention.source.arm_keys()
    assert arm_keys == (_BASELINE_KEY, _TREATMENT_KEY)
    assert (
        carries_intervention.source.node_key()
        == f'do({_BASELINE_KEY}|{_TREATMENT_KEY})'
    )


def test_bridge_rejects_non_doeffect_intervention() -> None:
    """If `source` is not a str, Measurable, or DoEffect, the
    decorator raises TypeError loudly — typed metadata."""
    with pytest.raises(TypeError, match='source.*str or Measurable'):
        @claim_bridge(
            source=42,  # type: ignore[arg-type]
            target='B',
        )
        def bad_source(
            paired_g: PairedGResult,
        ) -> Verdict:
            del paired_g
            return Verdict.HELD


def test_bridge_requires_source_and_target() -> None:
    """A bridge declaration without `source`/`target` in the
    decorator raises at decoration time — the structural contract
    is enforced at authoring."""
    with pytest.raises(TypeError):
        # Missing required `source` and `target` args.
        @claim_bridge()  # type: ignore[call-overload]
        def _no_source_target(
            paired_g: PairedGResult,
            *,
            direction: Direction = Direction.DIRECT,
        ) -> Verdict:
            del paired_g, direction
            return Verdict.HELD


def test_bridge_carries_typed_predicted_direction() -> None:
    """A bridge with `predicted_direction='a_gt_b'` in the
    decorator lands on `Bridge.predicted_direction` as the typed
    structural field — not buried in `params`. Promoted because
    paired/RE analyses consume it as shared metadata across most
    bridges."""
    @claim_bridge(
        source='A',
        target='B',
        predicted_direction='a_gt_b',
    )
    def carries_predicted_direction(
        paired_g: PairedGResult,
    ) -> Verdict:
        del paired_g
        return Verdict.HELD

    assert isinstance(carries_predicted_direction, Bridge)
    assert carries_predicted_direction.predicted_direction == 'a_gt_b'
    # Not leaked into params.
    assert 'predicted_direction' not in (
        carries_predicted_direction.params
    )


def test_bridge_predicted_direction_null_is_admitted() -> None:
    """The xfail-style `predicted_direction='null'` declares "I
    expect no effect"; the bridge body returns HELD when the null
    is observed (small |g|), NO_EFFECT when an effect was observed
    (the unexpected-pass / xpass analog). The framework admits
    the literal at decoration time."""
    @claim_bridge(
        source='A',
        target='B',
        predicted_direction='null',
    )
    def predicts_null(
        paired_g: PairedGResult,
    ) -> Verdict:
        del paired_g
        return Verdict.HELD

    assert predicts_null.predicted_direction == 'null'


def test_bridge_predicted_direction_unknown_literal_rejected() -> None:
    """An unrecognised `predicted_direction` literal raises
    TypeError at decoration time (early failure, not silent
    pass-through)."""
    import pytest as _pytest

    with _pytest.raises(TypeError, match='predicted_direction'):
        @claim_bridge(
            source='A',
            target='B',
            predicted_direction='wrong_literal',  # pyright: ignore[reportArgumentType]
        )
        def _bad(paired_g: PairedGResult) -> Verdict:
            del paired_g
            return Verdict.HELD


def test_evaluate_forwards_predicted_direction_to_analyses() -> None:
    """`evaluate` injects `predicted_direction` into bridge_params
    so analyses that take it as a kwarg resolve transparently —
    same channel as `source`/`target`/`tier`."""
    captured: dict[str, object] = {}

    from corroborate.bridge.analysis import analysis

    @analysis
    def _captures_pd(
        cells: pl.DataFrame | Iterable[Mapping[str, object]],
        *,
        predicted_direction: object,
        source: str = 'A',
    ) -> int:
        del cells, source
        captured['predicted_direction'] = predicted_direction
        return 1

    @claim_bridge(
        source='A',
        target='B',
        predicted_direction='a_lt_b',
    )
    def consumer(
        _captures_pd: int,
    ) -> Verdict:
        del _captures_pd
        return Verdict.HELD

    cells: list[Mapping[str, object]] = [
        {'A': 0.0, 'B': 0.0, 'env_name': 'X'},
    ]
    _ = evaluate(consumer, cells)
    assert captured['predicted_direction'] == 'a_lt_b'


def test_bridge_accepts_measurable_as_source() -> None:
    """`source` / `target` may be a `Measurable` instance passed by
    value (typically a value-composed reduction). The decorator
    auto-registers it so the cache walker finds it; analyses see
    `bridge.source_name` (the auto-generated column name)."""
    from corroborate.measurables import (
        Measurable, get_registered, registered_names,
    )
    from corroborate.measurables.reductions import from_key, mean_window

    q_max_late = mean_window(
        from_key('online_max_q_per_step'), 0.5, 1.0,
    )

    @claim_bridge(
        source=cast(Measurable[Mapping[str, object], object], q_max_late),
        target='outcome.eval_best_burst_mean',
        direction=Direction.DIRECT,
        tier=Tier.ASSOCIATIONAL,
    )
    def reduces_q_max_late(
        paired_g: PairedGResult,
        *,
        treatment_arm: str = 'ddqn',
        baseline_arm: str = 'vanilla_dqn',
    ) -> Verdict:
        del paired_g
        return Verdict.HELD

    assert isinstance(reduces_q_max_late, Bridge)
    assert reduces_q_max_late.source is q_max_late
    assert (
        reduces_q_max_late.source_name
        == 'online_max_q_per_step__mean_50_100'
    )
    assert reduces_q_max_late.target_name == 'outcome.eval_best_burst_mean'
    # Auto-registered in the global registry — cache walker finds it.
    assert get_registered(q_max_late.name) is q_max_late
    assert q_max_late.name in registered_names()


# ============ MODULE_SCOPE — file-level scope filter ============


def test_evaluate_module_scope_intersects_with_bridge_scope() -> None:
    """`evaluate(..., module_scope=expr)` AND-combines with the
    bridge's own `scope=` — a hypothesis-module-level filter.
    Cells matching bridge.scope but NOT module_scope are excluded.
    """
    cells = _synthetic_cells(n_seeds=30)
    # Half the cells get an 'excluded'-tagged env so module_scope
    # can reject them at the file level.
    for c in cells[:30]:  # first 30 → 15 pairs
        c['env_class'] = 'excluded'
    for c in cells[30:]:
        c['env_class'] = 'included'

    # No module_scope: all 30 pairs visible → HELD.
    out_full = evaluate(treatment_helps_outcome, cells)
    assert out_full.verdict == Verdict.HELD
    assert out_full.n_cells_in_scope == 60

    # With module_scope: only 'included' cells survive → 15 pairs.
    # treatment_helps_outcome's body keeps HELD if g > 0.3 sig
    # AND n_pairs ≥ 10, which 15 pairs still satisfies.
    out_filtered = evaluate(
        treatment_helps_outcome, cells,
        module_scope=pl.col('env_class') == 'included',
    )
    assert out_filtered.n_cells_in_scope == 30
    # Cells halved → still enough power.
    assert out_filtered.verdict == Verdict.HELD


def test_evaluate_module_scope_can_zero_a_bridge() -> None:
    """When bridge.scope is incompatible with module_scope (e.g.
    bridge filters to env=A, module_scope excludes A), the
    intersection is empty and the bridge sees zero cells.
    """
    cells = _synthetic_cells(n_seeds=30)
    out = evaluate(
        # treatment_helps_outcome.scope is `env_name == 'TestEnv'`.
        # module_scope says `env_name != 'TestEnv'` → no overlap.
        treatment_helps_outcome, cells,
        module_scope=pl.col('env_name') != 'TestEnv',
    )
    assert out.n_cells_in_scope == 0
    assert out.verdict == Verdict.POWER_INSUFFICIENT


def test_evaluate_module_scope_alone_when_bridge_scope_none() -> None:
    """A bridge with `scope=None` picks up the module_scope alone.
    Without module_scope, all cells flow through; with it, only
    the matching subset reaches the analysis.
    """
    @claim_bridge(
        source=INTERVENTION,
        target='eval_best_burst_mean',
        direction=Direction.DIRECT,
        tier=Tier.ASSOCIATIONAL,
        # No scope= — bridge.scope is None.
    )
    def unscoped_bridge(paired_g: PairedGResult) -> Verdict:
        if paired_g.n_pairs < 10:
            return Verdict.POWER_INSUFFICIENT
        if paired_g.g > 0.3 and paired_g.p_value < 0.05:
            return Verdict.HELD
        return Verdict.NO_EFFECT

    cells = _synthetic_cells(n_seeds=30)
    for c in cells[:30]:
        c['kind'] = 'A'
    for c in cells[30:]:
        c['kind'] = 'B'

    # With module_scope filtering to half the cells — fewer pairs,
    # but enough to still yield a verdict.
    out = evaluate(
        unscoped_bridge, cells,
        module_scope=pl.col('kind') == 'B',
    )
    assert out.n_cells_in_scope == 30


def test_evaluate_no_module_scope_preserves_legacy_behavior() -> None:
    """Default (kwarg unset) → bridge.scope alone. Existing
    callers don't see behavior change."""
    cells = _synthetic_cells(n_seeds=30)
    out = evaluate(treatment_helps_outcome, cells)
    assert out.n_cells_in_scope == 60
    assert out.verdict == Verdict.HELD


def test_module_scope_via_hypothesis_attribute() -> None:
    """The runner reads `MODULE_SCOPE` via `getattr(h, ..., None)`
    and forwards to `evaluate`. Simulating that path: a hypothesis-
    shaped object exposes `MODULE_SCOPE`; pulling it via
    `getattr(..., None)` and threading to `evaluate` gives the
    expected end-to-end behavior.
    """
    import types

    h = types.SimpleNamespace(
        __name__='test_hypothesis',
        INTERVENTION=INTERVENTION,
        BRIDGES=(treatment_helps_outcome,),
        MODULE_SCOPE=pl.col('env_name') != 'TestEnv',  # excludes ALL
    )
    cells = _synthetic_cells(n_seeds=30)
    df = pl.from_dicts(cells)
    module_scope = getattr(h, 'MODULE_SCOPE', None)
    out = evaluate(treatment_helps_outcome, df, module_scope=module_scope)
    assert out.n_cells_in_scope == 0
    assert out.verdict == Verdict.POWER_INSUFFICIENT


def test_module_scope_attribute_absent_means_none() -> None:
    """Hypotheses without `MODULE_SCOPE` → `getattr` falls back to
    None → bridge.scope alone determines the filter (legacy
    behavior preserved)."""
    import types

    h = types.SimpleNamespace(
        __name__='test_hypothesis_legacy',
        INTERVENTION=INTERVENTION,
        BRIDGES=(treatment_helps_outcome,),
        # No MODULE_SCOPE attribute.
    )
    cells = _synthetic_cells(n_seeds=30)
    df = pl.from_dicts(cells)
    module_scope = getattr(h, 'MODULE_SCOPE', None)
    assert module_scope is None
    out = evaluate(treatment_helps_outcome, df, module_scope=module_scope)
    assert out.n_cells_in_scope == 60
    assert out.verdict == Verdict.HELD


def test_analysis_wrapper_is_directly_callable() -> None:
    """`Analysis.__call__` delegates to `fn` with kwargs passed
    through UNFILTERED — the exploration / test-fixture spelling.
    Contrast `run_for`, which filters `bridge_params` down to the
    analysis signature; a direct call surfaces a mismatched kwarg
    as a TypeError instead of silently dropping it.

    Closed form: mean of {1, 2, 3, 4} scaled by 10 = 25.0 exactly
    (exact rational arithmetic — no sampling bound applies)."""
    from corroborate._internals.polars import as_rows
    from corroborate.bridge.analysis import analysis

    @analysis
    def _scaled_mean(
        cells: pl.DataFrame | Iterable[Mapping[str, object]],
        *,
        scale: float = 1.0,
    ) -> float:
        vals = [float(cast(float, c['v'])) for c in as_rows(cells)]
        return scale * sum(vals) / len(vals)

    cells: list[Mapping[str, object]] = [
        {'v': 1.0}, {'v': 2.0}, {'v': 3.0}, {'v': 4.0},
    ]
    assert _scaled_mean(cells, scale=10.0) == 25.0
    # With `Analysis[C, O, **P]` the mismatched kwarg is a STATIC
    # error at the call site; the ignore keeps the runtime
    # demonstration for unchecked callers (scripts outside pyright's
    # scope still get the TypeError, not a silent drop).
    with pytest.raises(TypeError):
        _scaled_mean(cells, not_a_param=1)  # pyright: ignore[reportCallIssue]


def test_analysis_call_passes_cells_through_unchanged() -> None:
    """`Analysis.__call__` is pure delegation — no hidden
    conversion, no registration-time signature reflection. Every
    analysis accepts the canonical cells union and normalises at
    its own entry, so the probe receives exactly the object shape
    the caller passed, in both directions."""
    from corroborate._internals.polars import as_rows
    from corroborate.bridge.analysis import Analysis, analysis

    @analysis
    def _union_probe(
        cells: pl.DataFrame | Iterable[Mapping[str, object]],
    ) -> tuple[str, int]:
        if isinstance(cells, pl.DataFrame):
            return ('dataframe', cells.height)
        return ('rows', len(list(cells)))

    @analysis
    def _normalising_probe(
        cells: pl.DataFrame | Iterable[Mapping[str, object]],
    ) -> int:
        return len(list(as_rows(cells)))

    df = pl.DataFrame({'a': [1, 2, 3]})
    assert _union_probe(df) == ('dataframe', 3)
    assert _union_probe([{'a': 1}]) == ('rows', 1)
    # `.fn` preserves the wrapped function's positional-or-keyword
    # first parameter instead of falsely exposing it as positional-only.
    assert _union_probe.fn(cells=df) == ('dataframe', 3)
    # The entry normalisation makes both shapes equivalent for a
    # row-consuming body — the convention the registry-wide guard
    # below pins for every production analysis.
    assert _normalising_probe(df) == 3
    assert _normalising_probe(df.to_dicts()) == 3

    # Backward-compatible public annotation: the original two
    # arguments remain cells/result; ParamSpec is optional third.
    legacy: Analysis[
        pl.DataFrame | Iterable[Mapping[str, object]], tuple[str, int]
    ] = _union_probe
    assert legacy(df) == ('dataframe', 3)


def test_analysis_call_preserves_wrapped_signature_statically() -> None:
    """`Analysis[C, O, **P]` preserves the wrapped fn's surface
    through `__call__` (CLAUDE.md: ParamSpec preserves caller
    signature through generic wrappers): the result type is the
    fn's declared return type, checked here with `assert_type`
    under the pyright-strict gate that runs on tests. The negative
    direction — a mistyped kwarg is a static reportCallIssue — is
    pinned by the ignore in
    test_analysis_wrapper_is_directly_callable."""
    from typing import assert_type

    from corroborate.analyses.paired.paired_g import paired_g

    cells = [
        c for c in _synthetic_cells()
        if c.get('env_name') == 'TestEnv'
    ]
    result = paired_g(
        cells,
        treatment_arm=_TREATMENT_KEY,
        baseline_arm=_BASELINE_KEY,
        pair_by=('seed',),
        source='eval_best_burst_mean',
    )
    assert_type(result, PairedGResult)
    assert result.n_pairs == 30


def test_analysis_registration_rejects_noncanonical_cells() -> None:
    """`@analysis` is the enforcement point for the canonical cells
    contract: a first parameter that does not spell the union fails
    registration with an instructive TypeError — at import time,
    never as a silent mis-shape at call time. Missing annotations
    fail the same way."""
    from corroborate.bridge.analysis import analysis

    with pytest.raises(TypeError, match='canonical union'):

        @analysis
        def _rows_only_probe(
            cells: Iterable[Mapping[str, object]],
        ) -> int:
            return len(list(cells))

    with pytest.raises(TypeError, match='canonical union'):

        @analysis
        def _frame_only_probe(cells: pl.DataFrame) -> int:
            return cells.height

    with pytest.raises(TypeError, match='canonical union'):

        @analysis
        def _unannotated_probe(cells) -> int:  # pyright: ignore[reportMissingParameterType]
            del cells
            return 0


def test_every_registered_analysis_accepts_the_cells_union() -> None:
    """Registry-wide proof that the shipped analysis surface passed
    the registration gate: every `corroborate.analyses` submodule is
    imported EXPLICITLY here (pkgutil walk), so the check does not
    depend on the package `__init__`'s wiring or on which other
    test modules happened to import first — a new analysis module
    left out of `__init__` is still discovered and still checked.
    Conformance itself is enforced by `@analysis` at registration;
    this guard proves coverage. Test-local probes (registered from
    test modules) are outside the `corroborate.*` filter."""
    import importlib
    import inspect
    import pkgutil

    import corroborate.analyses as analyses_pkg
    from corroborate.bridge.analysis import get_registered, registered_names

    for module_info in pkgutil.walk_packages(
        analyses_pkg.__path__, prefix='corroborate.analyses.',
    ):
        importlib.import_module(module_info.name)

    production_names = [
        name for name in registered_names()
        if (a := get_registered(name)) is not None
        and a.fn.__module__.startswith('corroborate.')
    ]
    assert len(production_names) >= 40
    offenders: list[str] = []
    for name in production_names:
        analysis_obj = get_registered(name)
        assert analysis_obj is not None
        first_param = next(
            iter(inspect.signature(analysis_obj.fn).parameters.values()),
        )
        # Name-agnostic: the cells argument is positional-first by
        # contract but may be semantically named (`panel`, …).
        if 'pl.DataFrame | ' not in str(first_param):
            offenders.append(f'{name}: {first_param}')
    assert not offenders, (
        'analyses whose cells parameter does not declare the '
        f'canonical union: {offenders}'
    )


def test_deterministic_outcome_is_never_reported_as_a_rider() -> None:
    """A zero-variance treatment effect makes the target constant
    within each arm — the constancy scan's signature. The outcome
    under test is the hypothesis, not a confound; it is excluded
    from isolation diagnostics."""
    cells: list[dict[str, object]] = []
    for seed in range(4):
        cells.append({'id': f'a{seed}', 'seed': seed, 'gamma': 0.8,
                      'return_mean': 5.0})
        cells.append({'id': f'b{seed}', 'seed': seed, 'gamma': 0.99,
                      'return_mean': 9.0})
    out = evaluate(higher_gamma_value_helps, cells, leaves=_LEAVES)
    assert out.verdict is Verdict.HELD
    assert all(
        'return_mean' not in w.message
        for w in out.warnings
        if w.gate_name == 'contrast_isolation'
    )
    assert out.warnings == ()


def test_unpaired_value_effect_evaluates_independent_arms() -> None:
    """`pair_by=()` declares an independent-samples design, not a
    malformed paired one: pair completeness doesn't apply, and the
    balance check compares whole arms as the single unit."""
    from corroborate.analyses.paired.arm_mean_diff import (
        ArmMeanDiffResult,
    )

    @claim_bridge(
        source=GAMMA_EFFECT,
        target='return_mean',
        direction=Direction.DIRECT,
        tier=Tier.INTERVENTIONAL,
        pair_by=(),
        predicted_direction='a_gt_b',
    )
    def _unpaired(arm_mean_diff: ArmMeanDiffResult) -> Verdict:
        return Verdict.HELD if arm_mean_diff.mean_diff > 0 else (
            Verdict.NO_EFFECT
        )

    cells: list[dict[str, object]] = []
    for i in range(6):
        cells.append({'id': f'r{i}', 'gamma': 0.8,
                      'return_mean': float(i)})
        cells.append({'id': f't{i}', 'gamma': 0.99,
                      'return_mean': 10.0 + i})
    out = evaluate(_unpaired, cells, leaves=_LEAVES)
    assert out.verdict is Verdict.HELD
    assert out.blocked_by is None
    assert all(
        w.gate_name != 'pair_completeness' for w in out.warnings
    )
    result = cast(ArmMeanDiffResult, out.analysis_results['arm_mean_diff'])
    assert result.n_treatment == 6
    assert result.n_baseline == 6
