"""Multi-fixture pattern smoke: a bridge consuming three DoWhy
analyses (backdoor_ate, placebo_refutation,
random_common_cause_refutation) simultaneously.

Synthetic linear corpus where the truth is known: outcome =
treatment * +2 + noise, with HPs as common causes. The expected
verdicts:
- backdoor_ate: ATE ≈ +2, identified=True
- placebo_refutation: real_ate ≈ +2, refuted_ate ≈ 0,
  drift ≈ |real - 0| ≈ 2 (large — the placebo destroys signal,
  which is what we want)
- random_common_cause_refutation: drift small (synthetic
  confounder shouldn't move the estimate much)

A bridge that asserts "ATE positive AND robust to RCC" consumes
two fixtures and routes through the threshold body — that's the
test of the multi-fixture composition pattern."""
from __future__ import annotations

import math
import random
from typing import cast

import pytest

import corroborate.analyses  # noqa: F401  # pyright: ignore[reportUnusedImport]

from corroborate.analyses.dowhy import (
    BackdoorResult, RefutationResult,
)
from corroborate.bridge.claim_bridge import (
    Direction, Tier, claim_bridge, evaluate,
)
from corroborate.bridge.verdict import Verdict


def _linear_corpus(
    *,
    n: int = 100,
    ate: float = 2.0,
    noise: float = 0.5,
) -> list[dict[str, object]]:
    """Build a corpus where outcome = ate * treatment + noise.
    Treatment and outcome are floats per cell; HPs are
    constant (no real confounding) but exposed so DoWhy has
    nodes to put in the DAG."""
    rng = random.Random(0)
    cells: list[dict[str, object]] = []
    for i in range(n):
        t = float(rng.uniform(0.0, 1.0))
        y = ate * t + rng.gauss(0, noise)
        cells.append({
            'treatment_var': t,
            'outcome_var': y,
            'capacity': float(rng.choice([10000, 50000])),
            'lr': float(rng.choice([1e-3, 1e-4])),
            'cell_id': i,
        })
    return cells


SYNTHETIC_DAG: list[tuple[str, str]] = [
    ('treatment_var', 'outcome_var'),
    ('capacity', 'treatment_var'),
    ('capacity', 'outcome_var'),
    ('lr', 'treatment_var'),
    ('lr', 'outcome_var'),
]


def test_backdoor_ate_runs_directly() -> None:
    """Single-fixture: bridge consuming only backdoor_ate."""
    @claim_bridge(
        source='treatment_var',
        target='outcome_var',
        direction=Direction.DIRECT,
        tier=Tier.INTERVENTIONAL,
    )
    def claim(
        backdoor_ate: BackdoorResult,
        *,
        treatment: str = 'treatment_var',
        outcome: str = 'outcome_var',
        dag: list[tuple[str, str]] = SYNTHETIC_DAG,
    ) -> Verdict:
        del treatment, outcome, dag
        if not backdoor_ate.identified:
            return Verdict.POWER_INSUFFICIENT
        if backdoor_ate.ate > 1.0:
            return Verdict.HELD
        return Verdict.NO_EFFECT

    out = evaluate(claim, _linear_corpus())
    bd = cast(BackdoorResult, out.analysis_results['backdoor_ate'])
    assert out.verdict == Verdict.HELD
    assert bd.identified is True
    assert bd.ate > 1.5, f'expected ATE near +2, got {bd.ate}'
    assert bd.n_rows == 100


def test_multi_fixture_bridge_consumes_three_analyses() -> None:
    """The headline multi-fixture pattern: one bridge consumes
    three independent analyses (backdoor_ate, placebo_refutation,
    random_common_cause_refutation). The framework resolves each
    by parameter name, runs them, injects all three results
    into the bridge body."""
    @claim_bridge(
        source='treatment_var',
        target='outcome_var',
        direction=Direction.DIRECT,
        tier=Tier.INTERVENTIONAL,
    )
    def claim(
        backdoor_ate: BackdoorResult,
        placebo_refutation: RefutationResult,
        random_common_cause_refutation: RefutationResult,
        *,
        treatment: str = 'treatment_var',
        outcome: str = 'outcome_var',
        dag: list[tuple[str, str]] = SYNTHETIC_DAG,
    ) -> Verdict:
        del treatment, outcome, dag
        if not backdoor_ate.identified:
            return Verdict.POWER_INSUFFICIENT
        ate_ok = backdoor_ate.ate > 1.0
        placebo_kills_signal = (
            abs(placebo_refutation.refuted_ate)
            < 0.5 * abs(placebo_refutation.real_ate)
        )
        rcc_robust = random_common_cause_refutation.drift < 0.5
        if ate_ok and placebo_kills_signal and rcc_robust:
            return Verdict.HELD
        return Verdict.NO_EFFECT

    out = evaluate(claim, _linear_corpus())

    bd = cast(BackdoorResult, out.analysis_results['backdoor_ate'])
    pl = cast(
        RefutationResult,
        out.analysis_results['placebo_refutation'],
    )
    rcc = cast(
        RefutationResult,
        out.analysis_results['random_common_cause_refutation'],
    )

    assert out.verdict == Verdict.HELD, (
        f'expected HELD on synthetic strong signal '
        f'(ate={bd.ate:.3f}, placebo refuted={pl.refuted_ate:.3f}, '
        f'rcc drift={rcc.drift:.3f})'
    )
    # Sanity: all three analyses returned finite results.
    assert math.isfinite(bd.ate)
    assert math.isfinite(pl.real_ate)
    assert math.isfinite(rcc.drift)
    # All saw the same n.
    assert bd.n_rows == pl.n_rows == rcc.n_rows == 100


@pytest.mark.parametrize(
    'analysis_name',
    ['backdoor_ate', 'placebo_refutation', 'random_common_cause_refutation'],
)
def test_dowhy_analyses_registered(analysis_name: str) -> None:
    """All three DoWhy analyses populate the registry on
    `corroborate.analyses` import."""
    from corroborate.analysis import get_registered
    assert get_registered(analysis_name) is not None
