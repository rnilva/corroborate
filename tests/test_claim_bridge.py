"""End-to-end smoke for the claim-bridge + analysis pattern.

Synthetic corpus where the truth is known: treatment cells have
outcome ≈ 1.0, baseline cells have outcome ≈ 0.0, paired by
seed across 30 seeds. A bridge that asserts "treatment moves
outcome by g > 0.3 with p < 0.05" should HELD; one with a
much higher threshold should NO_EFFECT.

The smoke proves:
- The `@analysis` decorator + registry round-trip.
- The `@claim_bridge` decorator builds a typed Bridge with
  structural fields + `holds_when`.
- `evaluate(bridge, cells)` resolves the analysis from the
  `holds_when` parameter name, parameterises it from the
  bridge's structural fields, runs it, and routes the result
  through the bridge's threshold body.
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
    """Build a paired corpus: n_seeds × {treatment, baseline}
    cells, each with `intervention_name`, `seed`, `env_name`, and
    a single outcome path."""
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
    # treatment_mean=1.0, baseline_mean=0.0, noise=0.1 → g should
    # be very large (Cohen-d ~ 7+, Hedges-corrected ~ 6.8).
    assert result.g > 3.0, f'expected g > 3, got {result.g}'
    assert result.p_value < 1e-6


def test_analysis_registered_globally() -> None:
    """Importing `corroborate.analyses` registers `paired_g` under
    that name — the lookup the bridge resolver uses."""
    from corroborate.analysis import (
        get_registered, registered_names,
    )
    assert 'paired_g' in registered_names()
    assert get_registered('paired_g') is not None


def test_bridge_held_under_explicit_threshold() -> None:
    """Authoring path: a bridge that consumes paired_g and
    asserts a sign+magnitude+power threshold. Synthetic corpus
    with strong effect → HELD."""

    @claim_bridge(
        name='treatment_helps_outcome',
        source='outcome.eval_best_burst_mean',
        target='outcome.eval_best_burst_mean',
        direction=Direction.DIRECT,
        tier=Tier.ASSOCIATIONAL,
        treatment_arm='treatment',
        baseline_arm='baseline',
        pair_by=('seed',),
        env_name='TestEnv',
    )
    def claim(paired_g: PairedGResult) -> Verdict:
        if paired_g.n_pairs < 10:
            return Verdict.POWER_INSUFFICIENT
        if paired_g.g > 0.3 and paired_g.p_value < 0.05:
            return Verdict.HELD
        return Verdict.NO_EFFECT

    cells = _synthetic_cells()
    out = evaluate(claim, cells)
    assert out.verdict == Verdict.HELD
    assert out.bridge_name == 'treatment_helps_outcome'
    pg = cast(PairedGResult, out.analysis_results['paired_g'])
    assert pg.n_pairs == 30
    assert pg.g > 0.3


def test_bridge_no_effect_when_signal_absent() -> None:
    """Same bridge shape, treatment ≈ baseline → NO_EFFECT."""

    @claim_bridge(
        name='no_real_effect',
        source='outcome.eval_best_burst_mean',
        target='outcome.eval_best_burst_mean',
        direction=Direction.DIRECT,
        tier=Tier.ASSOCIATIONAL,
        treatment_arm='treatment',
        baseline_arm='baseline',
        pair_by=('seed',),
        env_name='TestEnv',
    )
    def claim(paired_g: PairedGResult) -> Verdict:
        if paired_g.g > 0.3 and paired_g.p_value < 0.05:
            return Verdict.HELD
        return Verdict.NO_EFFECT

    cells = _synthetic_cells(treatment_mean=0.0, baseline_mean=0.0)
    out = evaluate(claim, cells)
    assert out.verdict == Verdict.NO_EFFECT


def test_bridge_power_insufficient_with_few_seeds() -> None:
    """Few pairs → bridge's `holds_when` returns
    POWER_INSUFFICIENT. The threshold is encoded in the bridge,
    not the analysis."""

    @claim_bridge(
        name='want_30_pairs',
        source='outcome.eval_best_burst_mean',
        target='outcome.eval_best_burst_mean',
        direction=Direction.DIRECT,
        tier=Tier.ASSOCIATIONAL,
        treatment_arm='treatment',
        baseline_arm='baseline',
        pair_by=('seed',),
        env_name='TestEnv',
    )
    def claim(paired_g: PairedGResult) -> Verdict:
        if paired_g.n_pairs < 30:
            return Verdict.POWER_INSUFFICIENT
        if paired_g.g > 0.3 and paired_g.p_value < 0.05:
            return Verdict.HELD
        return Verdict.NO_EFFECT

    cells = _synthetic_cells(n_seeds=5)
    out = evaluate(claim, cells)
    assert out.verdict == Verdict.POWER_INSUFFICIENT


def test_unknown_analysis_in_holds_when_raises() -> None:
    """A bridge that names an unregistered analysis fails fast at
    evaluation, not silently."""
    @claim_bridge(
        name='broken',
        source='x', target='y',
    )
    def claim(not_a_real_analysis: object) -> Verdict:
        del not_a_real_analysis
        return Verdict.HELD

    cells: list[Mapping[str, object]] = [{'env_name': 'X'}]
    with pytest.raises(KeyError, match='not_a_real_analysis'):
        _ = evaluate(claim, cells)


def test_bridge_carries_structural_metadata() -> None:
    """The bridge preserves the structural declaration as typed
    fields (for downstream introspection / persistence)."""

    @claim_bridge(
        name='X', source='A', target='B',
        direction=Direction.INVERSE,
        tier=Tier.INTERVENTIONAL,
        treatment_arm='ddqn',
        baseline_arm='vanilla_dqn',
    )
    def claim(paired_g: PairedGResult) -> Verdict:
        del paired_g
        return Verdict.HELD

    assert isinstance(claim, Bridge)
    assert claim.name == 'X'
    assert claim.source == 'A'
    assert claim.target == 'B'
    assert claim.direction == Direction.INVERSE
    assert claim.tier == Tier.INTERVENTIONAL
    assert claim.params['treatment_arm'] == 'ddqn'
    assert claim.params['baseline_arm'] == 'vanilla_dqn'
