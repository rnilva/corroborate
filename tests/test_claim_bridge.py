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

from collections.abc import Mapping
from typing import cast

import polars as pl
import pytest

# Importing analyses populates the registry.
import corroborate.analyses  # noqa: F401  # pyright: ignore[reportUnusedImport]

from corroborate.analyses.paired_g import PairedGResult
from corroborate.claim_bridge import (
    Bridge, Direction, Tier, claim_bridge, evaluate,
)
from corroborate.intervention import DoEffect
from corroborate.verdict import Verdict


# Module-level contrast for the top-level bridges in this file.
INTERVENTION = DoEffect(treatment_arm='treatment', baseline_arm='baseline')


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
            'intervention_name': 'treatment',
            'seed': s,
            'env_name': 'TestEnv',
            'eval_best_burst_mean': (
                treatment_mean + rng.gauss(0, noise)
            ),
        })
        out.append({
            'intervention_name': 'baseline',
            'seed': s,
            'env_name': 'TestEnv',
            'eval_best_burst_mean': (
                baseline_mean + rng.gauss(0, noise)
            ),
        })
    return out


def test_paired_g_analysis_runs_directly() -> None:
    """The analysis is callable on its own — no bridge needed.
    Cell-level scope (env filtering) lives upstream on Bridge.scope;
    when calling paired_g.fn directly the test pre-filters cells."""
    from corroborate.analyses.paired_g import paired_g
    cells = [
        c for c in _synthetic_cells()
        if c.get('env_name') == 'TestEnv'
    ]
    result = paired_g.fn(
        cells,
        treatment_arm='treatment',
        baseline_arm='baseline',
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
    from corroborate.analysis import (
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


def test_unknown_fixture_raises() -> None:
    """A fixture parameter (no default) that doesn't match a
    registered analysis fails fast at evaluation."""
    @claim_bridge(source='x', target='y')
    def broken(
        not_a_real_analysis: object,
    ) -> Verdict:
        del not_a_real_analysis
        return Verdict.HELD

    cells: list[Mapping[str, object]] = [{'env_name': 'X'}]
    with pytest.raises(KeyError, match='not_a_real_analysis'):
        _ = evaluate(broken, cells)


def test_bridge_carries_structural_metadata() -> None:
    """The decorator preserves the structural declaration as
    typed Bridge fields for downstream introspection."""
    @claim_bridge(
        source='A',
        target='B',
        direction=Direction.INVERSE,
        tier=Tier.INTERVENTIONAL,
    )
    def carries_metadata(
        paired_g: PairedGResult,
        *,
        treatment_arm: str = 'ddqn',
        baseline_arm: str = 'vanilla_dqn',
    ) -> Verdict:
        del paired_g
        return Verdict.HELD

    assert isinstance(carries_metadata, Bridge)
    assert carries_metadata.name == 'carries_metadata'
    assert carries_metadata.source == 'A'
    assert carries_metadata.target == 'B'
    assert carries_metadata.direction == Direction.INVERSE
    assert carries_metadata.tier == Tier.INTERVENTIONAL
    assert carries_metadata.params['treatment_arm'] == 'ddqn'
    assert carries_metadata.params['baseline_arm'] == 'vanilla_dqn'


def test_bridge_carries_typed_intervention() -> None:
    """A do-effect bridge declares source=DoEffect(...) in the
    decorator; the framework routes it to the structural
    `Bridge.source` field. The framework can then emit a
    `do(treatment|vs=baseline) → target` graph edge."""
    @claim_bridge(
        source=DoEffect(treatment_arm='ddqn', baseline_arm='vanilla_dqn'),
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
    assert carries_intervention.source.treatment_arm == 'ddqn'
    assert carries_intervention.source.baseline_arm == 'vanilla_dqn'
    assert (
        carries_intervention.source.node_key()
        == 'do(ddqn|vs=vanilla_dqn)'
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


def test_bridge_predicted_direction_defaults_to_none() -> None:
    """A bridge that does NOT declare `predicted_direction` carries
    None on the typed field. Backwards-compatible with all existing
    @claim_bridge declarations."""
    @claim_bridge(source='A', target='B')
    def no_predicted_direction(
        paired_g: PairedGResult,
    ) -> Verdict:
        del paired_g
        return Verdict.HELD

    assert no_predicted_direction.predicted_direction is None


def test_bridge_rejects_invalid_predicted_direction() -> None:
    """Only `'a_gt_b' | 'a_lt_b' | 'two_sided' | None` accepted —
    typed validation at decoration."""
    with pytest.raises(TypeError, match='predicted_direction'):
        @claim_bridge(
            source='A',
            target='B',
            predicted_direction='positive',  # type: ignore[arg-type]
        )
        def bad_predicted_direction(
            paired_g: PairedGResult,
        ) -> Verdict:
            del paired_g
            return Verdict.HELD


def test_evaluate_forwards_predicted_direction_to_analyses() -> None:
    """`evaluate` injects `predicted_direction` into bridge_params
    so analyses that take it as a kwarg resolve transparently —
    same channel as `source`/`target`/`tier`."""
    captured: dict[str, object] = {}

    from corroborate.analysis import analysis

    @analysis
    def _captures_pd(
        cells: list[Mapping[str, object]],
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

    cells: list[Mapping[str, object]] = [{'env_name': 'X'}]
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


def test_bridge_rejects_non_str_non_measurable_source() -> None:
    """Anything other than str | Measurable | DoEffect for source is
    an authoring mistake — fail loudly at decoration time."""
    with pytest.raises(TypeError, match='source.*str or Measurable'):
        @claim_bridge(
            source=42,  # type: ignore[arg-type]
            target='B',
        )
        def _bad_source(
            paired_g: PairedGResult,
        ) -> Verdict:
            del paired_g
            return Verdict.HELD
