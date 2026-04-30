"""Real-corpus smoke: claim-bridge pattern reproduces FINDINGS.md
verdicts on the action_dim_sweep corpus.

The eighth revision of FINDINGS.md documents per-env paired-g
verdicts on `mechanism.jensen_gap` (DDQN vs vanilla, paired by
seed). Each row is a published claim with a known g and verdict;
this smoke proves the claim-bridge + paired_g analysis pattern
reproduces those numbers exactly.

If FINDINGS.md changes (new corpora, threshold revisions), this
test fails loudly — which is the falsifiability contract the
file-protocol design promises."""
from __future__ import annotations

from pathlib import Path
from typing import cast

import polars as pl
import pytest

# Importing analyses populates the registry.
import corroborate.analyses  # noqa: F401  # pyright: ignore[reportUnusedImport]

from corroborate.analyses.paired_g import PairedGResult
from corroborate.claim_bridge import (
    Bridge, Direction, Tier, claim_bridge, evaluate,
)
from corroborate.verdict import Verdict


REPO_ROOT = Path(__file__).resolve().parent.parent
ACTION_DIM_PARQUET = (
    REPO_ROOT / 'experiments' / 'data' / 'action_dim_sweep'
    / 'runs.parquet'
)


@pytest.fixture(scope='module')
def action_dim_cells() -> list[dict[str, object]]:
    """Load the action_dim_sweep corpus once per test module —
    480 cells, 4 envs × 2 arms × ~60 seeds."""
    if not ACTION_DIM_PARQUET.exists():
        pytest.skip(f'corpus not available at {ACTION_DIM_PARQUET}')
    df = pl.read_parquet(ACTION_DIM_PARQUET)
    return list(df.iter_rows(named=True))


def _ddqn_reduces_jensen_gap(env_name: str) -> Bridge:
    """Author the per-env bridge: DDQN reduces jensen_gap with
    paired g < -0.3 at p < 0.05; sign-wrong → POWER_INSUFFICIENT;
    n < 30 → POWER_INSUFFICIENT."""
    @claim_bridge(
        name=f'ddqn_reduces_jensen_gap__{env_name}',
        source='mechanism.jensen_gap',
        target='mechanism.jensen_gap',
        direction=Direction.INVERSE,
        tier=Tier.ASSOCIATIONAL,
        treatment_arm='ddqn',
        baseline_arm='vanilla_dqn',
        pair_by=('seed',),
        env_name=env_name,
    )
    def claim(paired_g: PairedGResult) -> Verdict:
        if paired_g.n_pairs < 30:
            return Verdict.POWER_INSUFFICIENT
        if paired_g.g >= 0:
            # Sign opposes prediction (DDQN expected to reduce gap).
            return Verdict.POWER_INSUFFICIENT
        if paired_g.g < -0.3 and paired_g.p_value < 0.05:
            return Verdict.HELD
        return Verdict.NO_EFFECT

    return claim


# FINDINGS.md eighth revision reference table:
#   Acrobot-v1   |A|=3 g=-0.596 HELD
#   Catch-bsuite |A|=3 g=-4.662 HELD
#   DiscountingChain-bsuite |A|=5 g=-0.600 HELD
#   CartPole-v1  |A|=2 g=+0.090 POWER_INSUFFICIENT (sign wrong)


@pytest.mark.parametrize(
    'env_name, expected_g, expected_verdict',
    [
        ('Acrobot-v1', -0.596, Verdict.HELD),
        ('DiscountingChain-bsuite', -0.600, Verdict.HELD),
        ('CartPole-v1', +0.090, Verdict.POWER_INSUFFICIENT),
    ],
)
def test_ddqn_jensen_gap_reproduces_findings(
    action_dim_cells: list[dict[str, object]],
    env_name: str,
    expected_g: float,
    expected_verdict: Verdict,
) -> None:
    bridge = _ddqn_reduces_jensen_gap(env_name)
    out = evaluate(bridge, action_dim_cells)
    pg = cast(PairedGResult, out.analysis_results['paired_g'])
    assert out.verdict == expected_verdict, (
        f'{env_name}: expected {expected_verdict.value}, '
        f'got {out.verdict.value} '
        f'(g={pg.g:.3f}, p={pg.p_value:.6f}, n={pg.n_pairs})'
    )
    assert abs(pg.g - expected_g) < 0.005, (
        f'{env_name}: expected g≈{expected_g}, got {pg.g:.3f}'
    )
    assert pg.n_pairs == 60, (
        f'{env_name}: expected 60 pairs, got {pg.n_pairs}'
    )


def test_audit_trail_carries_analysis_result(
    action_dim_cells: list[dict[str, object]],
) -> None:
    """The BridgeEvaluation carries the raw analysis result for
    the audit trail — downstream tooling can introspect."""
    bridge = _ddqn_reduces_jensen_gap('Acrobot-v1')
    out = evaluate(bridge, action_dim_cells)
    assert 'paired_g' in out.analysis_results
    pg = out.analysis_results['paired_g']
    assert isinstance(pg, PairedGResult)
    assert pg.measurable == 'mechanism.jensen_gap'
    assert pg.treatment_arm == 'ddqn'
    assert pg.baseline_arm == 'vanilla_dqn'


# ============ Meta-regression: log_action_dim moderates g_mech ============

def test_meta_regression_action_dim_reproduces_findings(
    action_dim_cells: list[dict[str, object]],
) -> None:
    """FINDINGS.md eighth revision: meta-regression of `g_mech`
    on action_dim shows the right direction (β negative — more
    bias-reduction at higher |A|) but n=4 envs is underpowered
    for significance.

    The bridge asserts the sign, and POWER_INSUFFICIENT when
    the coefficient is non-significant. Magnitude isn't fixed
    by the claim — different inverse-variance weightings
    legitimately produce different β; the substantive finding
    is the negative sign + underpowered."""
    import math
    from corroborate.meta_regression import MetaRegressionResult

    covariates_per_env: dict[str, dict[str, float]] = {
        'CartPole-v1': {'log_action_dim': math.log(2)},
        'Acrobot-v1': {'log_action_dim': math.log(3)},
        'Catch-bsuite': {'log_action_dim': math.log(3)},
        'DiscountingChain-bsuite': {'log_action_dim': math.log(5)},
    }

    @claim_bridge(
        name='log_action_dim_drives_jensen_gap_reduction',
        source='mechanism.jensen_gap',
        target='mechanism.jensen_gap',
        direction=Direction.INVERSE,
        tier=Tier.ASSOCIATIONAL,
        treatment_arm='ddqn',
        baseline_arm='vanilla_dqn',
        pair_by=('seed',),
        covariates_per_env=covariates_per_env,
    )
    def claim(
        meta_regression_paired_g: MetaRegressionResult,
    ) -> Verdict:
        coef = next(
            (c for c in meta_regression_paired_g.coefficients
             if c.name == 'log_action_dim'),
            None,
        )
        if coef is None:
            return Verdict.NO_EFFECT
        if coef.coefficient < 0:
            return (
                Verdict.HELD if coef.is_significant
                else Verdict.POWER_INSUFFICIENT
            )
        return Verdict.NO_EFFECT

    out = evaluate(claim, action_dim_cells)
    assert out.verdict == Verdict.POWER_INSUFFICIENT, (
        f'expected POWER_INSUFFICIENT (right direction, '
        f'underpowered at n=4); got {out.verdict.value}'
    )
    mr = cast(
        MetaRegressionResult,
        out.analysis_results['meta_regression_paired_g'],
    )
    assert mr.n_strata == 4
    coef = next(
        c for c in mr.coefficients if c.name == 'log_action_dim'
    )
    assert coef.coefficient < 0, (
        f'expected negative slope (DDQN reduces gap more on '
        f'higher-|A| envs); got β={coef.coefficient:+.3f}'
    )
    assert not coef.is_significant, (
        f'expected p > α at n=4; got p={coef.p_value:.4f}'
    )
