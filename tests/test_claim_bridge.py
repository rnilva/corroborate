"""End-to-end smoke for the claim-bridge + analysis pattern.

Synthetic corpus where the truth is known: treatment cells have
outcome ≈ 1.0, baseline cells have outcome ≈ 0.0, paired by
seed across 30 seeds. A bridge that asserts "treatment moves
outcome by g > 0.3 with p < 0.05" should HELD; one with a
much higher threshold should NO_EFFECT.

The smoke proves:
- The `@analysis` decorator + registry round-trip.
- The `@claim_bridge` decorator reads bridge metadata from
  the function's signature defaults and produces a typed Bridge.
- `evaluate(bridge, cells)` resolves each fixture (parameter
  without a default) by name against the analysis registry,
  parameterises from the bridge's structural fields + params
  bag, runs, injects, and routes through the bridge body.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import pytest

# Importing analyses populates the registry.
import corroborate.analyses  # noqa: F401  # pyright: ignore[reportUnusedImport]

from corroborate.analyses.paired_g import PairedGResult
from corroborate.claim_bridge import (
    Bridge, Direction, Tier, claim_bridge, evaluate,
)
from corroborate.verdict import Verdict


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
            'outcome.eval_best_burst_mean': (
                treatment_mean + rng.gauss(0, noise)
            ),
        })
        out.append({
            'intervention_name': 'baseline',
            'seed': s,
            'env_name': 'TestEnv',
            'outcome.eval_best_burst_mean': (
                baseline_mean + rng.gauss(0, noise)
            ),
        })
    return out


def test_paired_g_analysis_runs_directly() -> None:
    """The analysis is callable on its own — no bridge needed."""
    from corroborate.analyses.paired_g import paired_g
    cells = _synthetic_cells()
    result = paired_g.fn(
        cells,
        treatment_arm='treatment',
        baseline_arm='baseline',
        pair_by=('seed',),
        source='outcome.eval_best_burst_mean',
        env_name='TestEnv',
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


@claim_bridge
def treatment_helps_outcome(
    paired_g: PairedGResult,
    *,
    source: str = 'outcome.eval_best_burst_mean',
    target: str = 'outcome.eval_best_burst_mean',
    direction: Direction = Direction.DIRECT,
    tier: Tier = Tier.ASSOCIATIONAL,
    treatment_arm: str = 'treatment',
    baseline_arm: str = 'baseline',
    pair_by: tuple[str, ...] = ('seed',),
    env_name: str = 'TestEnv',
) -> Verdict:
    del source, target, direction, tier
    del treatment_arm, baseline_arm, pair_by, env_name
    if paired_g.n_pairs < 10:
        return Verdict.POWER_INSUFFICIENT
    if paired_g.g > 0.3 and paired_g.p_value < 0.05:
        return Verdict.HELD
    return Verdict.NO_EFFECT


def test_bridge_held_under_explicit_threshold() -> None:
    """Authoring path: the bridge is just a function whose
    signature carries the metadata. Synthetic corpus with strong
    effect → HELD."""
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


@claim_bridge
def want_30_pairs(
    paired_g: PairedGResult,
    *,
    source: str = 'outcome.eval_best_burst_mean',
    target: str = 'outcome.eval_best_burst_mean',
    direction: Direction = Direction.DIRECT,
    tier: Tier = Tier.ASSOCIATIONAL,
    treatment_arm: str = 'treatment',
    baseline_arm: str = 'baseline',
    pair_by: tuple[str, ...] = ('seed',),
    env_name: str = 'TestEnv',
) -> Verdict:
    del source, target, direction, tier
    del treatment_arm, baseline_arm, pair_by, env_name
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
    @claim_bridge
    def broken(
        not_a_real_analysis: object,
        *,
        source: str = 'x',
        target: str = 'y',
    ) -> Verdict:
        del not_a_real_analysis
        return Verdict.HELD

    cells: list[Mapping[str, object]] = [{'env_name': 'X'}]
    with pytest.raises(KeyError, match='not_a_real_analysis'):
        _ = evaluate(broken, cells)


def test_bridge_carries_structural_metadata() -> None:
    """The decorator preserves the structural declaration as
    typed Bridge fields for downstream introspection."""
    @claim_bridge
    def carries_metadata(
        paired_g: PairedGResult,
        *,
        source: str = 'A',
        target: str = 'B',
        direction: Direction = Direction.INVERSE,
        tier: Tier = Tier.INTERVENTIONAL,
        treatment_arm: str = 'ddqn',
        baseline_arm: str = 'vanilla_dqn',
    ) -> Verdict:
        del paired_g, source, target, direction, tier
        del treatment_arm, baseline_arm
        return Verdict.HELD

    assert isinstance(carries_metadata, Bridge)
    assert carries_metadata.name == 'carries_metadata'
    assert carries_metadata.source == 'A'
    assert carries_metadata.target == 'B'
    assert carries_metadata.direction == Direction.INVERSE
    assert carries_metadata.tier == Tier.INTERVENTIONAL
    assert carries_metadata.params['treatment_arm'] == 'ddqn'
    assert carries_metadata.params['baseline_arm'] == 'vanilla_dqn'
    assert carries_metadata.intervention is None


def test_bridge_carries_typed_intervention() -> None:
    """A do-effect bridge declares `intervention=DoEffect(...)` as
    a defaulted kwarg; the decorator routes it to the structural
    `Bridge.intervention` field. The framework can then emit a
    `do(treatment|vs=baseline) → target` graph edge instead of
    burying the arm names in `params`."""
    from corroborate.intervention import DoEffect

    @claim_bridge
    def carries_intervention(
        paired_g: PairedGResult,
        *,
        source: str = 'outcome_native',
        target: str = 'outcome.eval_best_burst_mean',
        direction: Direction = Direction.DIRECT,
        tier: Tier = Tier.INTERVENTIONAL,
        intervention: DoEffect = DoEffect(
            treatment_arm='ddqn', baseline_arm='vanilla_dqn',
        ),
    ) -> Verdict:
        del paired_g, source, target, direction, tier, intervention
        return Verdict.HELD

    assert isinstance(carries_intervention, Bridge)
    assert carries_intervention.intervention is not None
    assert carries_intervention.intervention.treatment_arm == 'ddqn'
    assert carries_intervention.intervention.baseline_arm == 'vanilla_dqn'
    assert (
        carries_intervention.intervention.node_key()
        == 'do(ddqn|vs=vanilla_dqn)'
    )


def test_bridge_rejects_non_doeffect_intervention() -> None:
    """If `intervention=` is set but isn't a DoEffect, the
    decorator raises TypeError loudly — typed metadata."""
    with pytest.raises(TypeError, match='intervention'):
        @claim_bridge
        def bad_intervention(
            paired_g: PairedGResult,
            *,
            source: str = 'A',
            target: str = 'B',
            intervention: str = 'not-a-doeffect',
        ) -> Verdict:
            del paired_g, source, target, intervention
            return Verdict.HELD


def test_bridge_requires_source_and_target() -> None:
    """A bridge declaration without `source`/`target` defaults
    raises at decoration time — the structural contract is
    enforced at authoring."""
    def _no_source_target(
        paired_g: PairedGResult,
        *,
        direction: Direction = Direction.DIRECT,
    ) -> Verdict:
        del paired_g, direction
        return Verdict.HELD

    with pytest.raises(TypeError, match='source.*target'):
        _ = claim_bridge(_no_source_target)


def test_bridge_carries_typed_predicted_direction() -> None:
    """A bridge with `predicted_direction='a_gt_b'` defaulted lands
    on `Bridge.predicted_direction` as the typed structural field —
    not buried in `params`. Promoted because paired/RE analyses
    consume it as shared metadata across most bridges."""
    @claim_bridge
    def carries_predicted_direction(
        paired_g: PairedGResult,
        *,
        source: str = 'A',
        target: str = 'B',
        predicted_direction: str = 'a_gt_b',
    ) -> Verdict:
        del paired_g, source, target, predicted_direction
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
    @claim_bridge
    def no_predicted_direction(
        paired_g: PairedGResult,
        *,
        source: str = 'A',
        target: str = 'B',
    ) -> Verdict:
        del paired_g, source, target
        return Verdict.HELD

    assert no_predicted_direction.predicted_direction is None


def test_bridge_rejects_invalid_predicted_direction() -> None:
    """Only `'a_gt_b' | 'a_lt_b' | 'two_sided' | None` accepted —
    typed validation at decoration."""
    with pytest.raises(TypeError, match='predicted_direction'):
        @claim_bridge
        def bad_predicted_direction(
            paired_g: PairedGResult,
            *,
            source: str = 'A',
            target: str = 'B',
            predicted_direction: str = 'positive',
        ) -> Verdict:
            del paired_g, source, target, predicted_direction
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

    @claim_bridge
    def consumer(
        _captures_pd: int,
        *,
        source: str = 'A',
        target: str = 'B',
        predicted_direction: str = 'a_lt_b',
    ) -> Verdict:
        del _captures_pd, source, target, predicted_direction
        return Verdict.HELD

    cells: list[Mapping[str, object]] = [{'env_name': 'X'}]
    _ = evaluate(consumer, cells)
    assert captured['predicted_direction'] == 'a_lt_b'
