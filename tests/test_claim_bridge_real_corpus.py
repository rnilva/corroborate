"""Real-corpus smoke: claim-bridge pattern reproduces FINDINGS.md
verdicts on the action_dim_sweep corpus.

The eighth revision of FINDINGS.md documents per-env paired-g
verdicts on `mechanism.jensen_gap` (DDQN vs vanilla, paired by
seed). Each row is a published claim with a known g and verdict;
this smoke proves the claim-bridge + paired_g + meta-regression
pattern reproduces them exactly.

If FINDINGS.md changes (new corpora, threshold revisions), this
test fails loudly — which is the falsifiability contract the
file-protocol design promises."""
from __future__ import annotations

import math
from pathlib import Path
from typing import cast

import polars as pl
import pytest

# Importing analyses populates the registry.
import corroborate.analyses  # noqa: F401  # pyright: ignore[reportUnusedImport]

from corroborate.analyses.paired_g import PairedGResult
from corroborate.claim_bridge import (
    Direction, Tier, claim_bridge, evaluate,
)
from corroborate.meta_regression import MetaRegressionResult
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


# ============ Per-env paired-g jensen_gap claims ============
#
# FINDINGS.md eighth revision reference table:
#   Acrobot-v1   |A|=3 g=-0.596 HELD
#   Catch-bsuite |A|=3 g=-4.662 HELD
#   DiscountingChain-bsuite |A|=5 g=-0.600 HELD
#   CartPole-v1  |A|=2 g=+0.090 POWER_INSUFFICIENT (sign wrong)


@claim_bridge
def ddqn_reduces_jensen_gap__acrobot(
    paired_g: PairedGResult,
    *,
    source: str = 'mechanism.jensen_gap',
    target: str = 'mechanism.jensen_gap',
    direction: Direction = Direction.INVERSE,
    tier: Tier = Tier.ASSOCIATIONAL,
    treatment_arm: str = 'ddqn',
    baseline_arm: str = 'vanilla_dqn',
    pair_by: tuple[str, ...] = ('seed',),
    env_name: str = 'Acrobot-v1',
) -> Verdict:
    del source, target, direction, tier
    del treatment_arm, baseline_arm, pair_by, env_name
    if paired_g.n_pairs < 30:
        return Verdict.POWER_INSUFFICIENT
    if paired_g.g >= 0:
        return Verdict.POWER_INSUFFICIENT  # sign opposes prediction
    if paired_g.g < -0.3 and paired_g.p_value < 0.05:
        return Verdict.HELD
    return Verdict.NO_EFFECT


@claim_bridge
def ddqn_reduces_jensen_gap__discounting_chain(
    paired_g: PairedGResult,
    *,
    source: str = 'mechanism.jensen_gap',
    target: str = 'mechanism.jensen_gap',
    direction: Direction = Direction.INVERSE,
    tier: Tier = Tier.ASSOCIATIONAL,
    treatment_arm: str = 'ddqn',
    baseline_arm: str = 'vanilla_dqn',
    pair_by: tuple[str, ...] = ('seed',),
    env_name: str = 'DiscountingChain-bsuite',
) -> Verdict:
    del source, target, direction, tier
    del treatment_arm, baseline_arm, pair_by, env_name
    if paired_g.n_pairs < 30:
        return Verdict.POWER_INSUFFICIENT
    if paired_g.g >= 0:
        return Verdict.POWER_INSUFFICIENT
    if paired_g.g < -0.3 and paired_g.p_value < 0.05:
        return Verdict.HELD
    return Verdict.NO_EFFECT


@claim_bridge
def ddqn_reduces_jensen_gap__cartpole(
    paired_g: PairedGResult,
    *,
    source: str = 'mechanism.jensen_gap',
    target: str = 'mechanism.jensen_gap',
    direction: Direction = Direction.INVERSE,
    tier: Tier = Tier.ASSOCIATIONAL,
    treatment_arm: str = 'ddqn',
    baseline_arm: str = 'vanilla_dqn',
    pair_by: tuple[str, ...] = ('seed',),
    env_name: str = 'CartPole-v1',
) -> Verdict:
    del source, target, direction, tier
    del treatment_arm, baseline_arm, pair_by, env_name
    if paired_g.n_pairs < 30:
        return Verdict.POWER_INSUFFICIENT
    if paired_g.g >= 0:
        return Verdict.POWER_INSUFFICIENT
    if paired_g.g < -0.3 and paired_g.p_value < 0.05:
        return Verdict.HELD
    return Verdict.NO_EFFECT


@pytest.mark.parametrize(
    'bridge_obj, expected_g, expected_verdict',
    [
        (
            ddqn_reduces_jensen_gap__acrobot,
            -0.596, Verdict.HELD,
        ),
        (
            ddqn_reduces_jensen_gap__discounting_chain,
            -0.600, Verdict.HELD,
        ),
        (
            ddqn_reduces_jensen_gap__cartpole,
            +0.090, Verdict.POWER_INSUFFICIENT,
        ),
    ],
)
def test_ddqn_jensen_gap_reproduces_findings(
    action_dim_cells: list[dict[str, object]],
    bridge_obj: object,
    expected_g: float,
    expected_verdict: Verdict,
) -> None:
    from corroborate.claim_bridge import Bridge
    bridge = cast(Bridge, bridge_obj)
    out = evaluate(bridge, action_dim_cells)
    pg = cast(PairedGResult, out.analysis_results['paired_g'])
    assert out.verdict == expected_verdict, (
        f'{bridge.name}: expected {expected_verdict.value}, '
        f'got {out.verdict.value} '
        f'(g={pg.g:.3f}, p={pg.p_value:.6f}, n={pg.n_pairs})'
    )
    assert abs(pg.g - expected_g) < 0.005, (
        f'{bridge.name}: expected g≈{expected_g}, got {pg.g:.3f}'
    )
    assert pg.n_pairs == 60


def test_audit_trail_carries_analysis_result(
    action_dim_cells: list[dict[str, object]],
) -> None:
    """The BridgeEvaluation carries the raw analysis result for
    the audit trail — downstream tooling can introspect."""
    out = evaluate(
        ddqn_reduces_jensen_gap__acrobot, action_dim_cells,
    )
    assert 'paired_g' in out.analysis_results
    pg = out.analysis_results['paired_g']
    assert isinstance(pg, PairedGResult)
    assert pg.measurable == 'mechanism.jensen_gap'
    assert pg.treatment_arm == 'ddqn'
    assert pg.baseline_arm == 'vanilla_dqn'


# ============ Meta-regression: log_action_dim moderates g_mech ============

_COVARIATES_PER_ENV: dict[str, dict[str, float]] = {
    'CartPole-v1': {'log_action_dim': math.log(2)},
    'Acrobot-v1': {'log_action_dim': math.log(3)},
    'Catch-bsuite': {'log_action_dim': math.log(3)},
    'DiscountingChain-bsuite': {'log_action_dim': math.log(5)},
}


@claim_bridge
def log_action_dim_drives_jensen_gap_reduction(
    meta_regression_paired_g: MetaRegressionResult,
    *,
    source: str = 'mechanism.jensen_gap',
    target: str = 'mechanism.jensen_gap',
    direction: Direction = Direction.INVERSE,
    tier: Tier = Tier.ASSOCIATIONAL,
    treatment_arm: str = 'ddqn',
    baseline_arm: str = 'vanilla_dqn',
    pair_by: tuple[str, ...] = ('seed',),
    covariates_per_env: dict[str, dict[str, float]] = (
        _COVARIATES_PER_ENV
    ),
) -> Verdict:
    del source, target, direction, tier
    del treatment_arm, baseline_arm, pair_by, covariates_per_env
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


def test_meta_regression_action_dim_reproduces_findings(
    action_dim_cells: list[dict[str, object]],
) -> None:
    """FINDINGS.md eighth revision: meta-regression of g_mech on
    action_dim shows the right direction (β negative) but n=4
    envs is underpowered."""
    out = evaluate(
        log_action_dim_drives_jensen_gap_reduction, action_dim_cells,
    )
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
    assert coef.coefficient < 0
    assert not coef.is_significant
