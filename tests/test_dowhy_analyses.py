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
from corroborate.bridge.bridge import (
    Direction, Tier, claim_bridge, evaluate,
)
from corroborate.bridge.verdict import Verdict


def _linear_corpus(
    *,
    n: int = 400,
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


# NOTE: `test_backdoor_ate_runs_directly` was deleted in the
# analytic-suite redundancy pass. Its single-fixture bridge
# composition path is exercised by `test_multi_fixture_bridge_
# consumes_three_analyses` below (which consumes backdoor_ate
# as one of three fixtures via the same `@claim_bridge` +
# `evaluate` surface). The closed-form ATE recovery is now
# tested at `tests/analytic/lg_scm/test_dowhy.py` against
# `beta_xz · beta_zy = 0.75` with rel_err < 0.05.


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
        # ASSOCIATIONAL: the test exercises the analysis-injection
        # machinery on synthetic columns, not a Pearl-rung-2 claim
        # on a registered measurable. INTERVENTIONAL would
        # (correctly) trip the EXOGENOUS_SOURCE BLOCK gate since
        # `treatment_var` is a synthetic column not in
        # `registered_names() | _STANDARD_METADATA`.
        tier=Tier.ASSOCIATIONAL,
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

    # Closed-form recovery assertions — the bridge body's
    # thresholds (`> 1.0`, `< 0.5*real`, `< 0.5`) leave 5-20× slack
    # over the structural truth (ate=2.0, refuted=0, rcc-drift≈0).
    # A DoWhy stub returning `BackdoorResult(ate=1.5)` regardless
    # of input would pass the bridge but breach these closed-form
    # bounds. Bypass the lax thresholds and check structural
    # recovery directly.
    #
    # Construction: outcome = 2.0·t + ε, ε ~ N(0, 0.5²), n=400.
    # Population OLS slope SE on t ~ U(0,1) is
    # σ_ε / (σ_t · √n) = 0.5 / (0.289 · 20) ≈ 0.087. 4·SE ≈ 0.35.
    # The `abs=0.35` bound is the structural 4σ band around the
    # true ATE = 2.0. A DoWhy stub returning constant `1.5` gives
    # |1.5 − 2.0| = 0.5 > 0.35 — caught.
    assert bd.ate == pytest.approx(2.0, abs=0.35), (
        f'backdoor_ate.ate = {bd.ate:.4f}; closed-form structural '
        f'ATE = 2.0 (outcome = 2.0·t + ε). A DoWhy regression '
        f'returning a constant (e.g., 1.5) would breach this.'
    )
    # Placebo refutation: random treatment column → estimate ≈ 0.
    # Sampling SE on placebo ≈ 0.5/√400 = 0.025. Bound abs=0.10
    # absorbs sampling drift (4·SE) and rejects a placebo refuter
    # that aliased the real treatment column → would return ≈ 2.0.
    assert pl.refuted_ate == pytest.approx(0.0, abs=0.10), (
        f'placebo_refutation.refuted_ate = {pl.refuted_ate:.4f}; '
        f'closed-form structural placebo ate = 0 (random treatment '
        f'has no effect). A refuter that aliased the real column '
        f'would yield ≈ 2.0; a refuter that ignored the placebo '
        f'and returned `real_ate` would also breach.'
    )
    # Random common cause refutation: drift = |original - refuted|
    # under a synthetic random confounder. With n=100 and the
    # confounder being noise-only (no true effect on either node),
    # drift is sampling-driven; structural drift = 0. Empirical
    # drift on this seed is < 0.02. Bound `< 0.05` is 2.5×
    # empirical floor — rejects RCC stubs that uniformly inflate
    # drift (`drift = 0.4` would pass the bridge's `< 0.5` but
    # breach this).
    assert rcc.drift < 0.05, (
        f'random_common_cause_refutation.drift = {rcc.drift:.4f}; '
        f'expected < 0.05 (synthetic confounder is noise-only and '
        f'should NOT meaningfully shift the ATE). The bridge\'s '
        f'< 0.5 threshold is 10× too loose to catch a stub '
        f'returning constant drift = 0.4.'
    )

    # All three analyses returned finite results and saw the same n.
    assert math.isfinite(pl.real_ate)
    assert bd.n_rows == pl.n_rows == rcc.n_rows == 400


@pytest.mark.parametrize(
    'analysis_name',
    ['backdoor_ate', 'placebo_refutation', 'random_common_cause_refutation'],
)
def test_dowhy_analyses_registered(analysis_name: str) -> None:
    """All three DoWhy analyses populate the registry on
    `corroborate.analyses` import."""
    from corroborate.bridge.analysis import get_registered
    assert get_registered(analysis_name) is not None
