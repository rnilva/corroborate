"""Real-corpus smoke: bridges from `experiments/findings/dqn_bridges.py`
reproduce FINDINGS.md verdicts on the action_dim_sweep corpus.

This is the file-protocol regression: the bridge file is the
authored artifact; running it against the parquet should
reproduce the documented verdicts. If FINDINGS.md changes (new
corpus, revised threshold), this test fails loudly.

The bridges live in `experiments/findings/dqn_bridges.py`; this
test imports them and evaluates each. Running them outside the
test (`uv run python -m experiments.findings.run_dqn_bridges`)
prints a verdict table for the same corpus."""
from __future__ import annotations

from pathlib import Path
from typing import cast

import polars as pl
import pytest

import corroborate.analyses  # noqa: F401  # pyright: ignore[reportUnusedImport]

from corroborate.analyses.paired.paired_g import PairedGResult
from corroborate.bridge.bridge import Bridge, evaluate
from corroborate.stats import MetaRegressionResult
from corroborate.bridge.verdict import Verdict
from experiments.findings.dqn_bridges import (
    INTERVENTION,
    ddqn_reduces_jensen_gap__acrobot,
    ddqn_reduces_jensen_gap__cartpole,
    ddqn_reduces_jensen_gap__catch,
    ddqn_reduces_jensen_gap__discounting_chain,
    log_action_dim_drives_jensen_gap_reduction,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
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


@pytest.mark.parametrize(
    'bridge_obj, expected_g, expected_verdict',
    [
        (
            ddqn_reduces_jensen_gap__acrobot,
            -0.596, Verdict.HELD,
        ),
        (
            ddqn_reduces_jensen_gap__catch,
            -4.662, Verdict.HELD,
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
def test_authored_bridge_reproduces_findings(
    action_dim_cells: list[dict[str, object]],
    bridge_obj: object,
    expected_g: float,
    expected_verdict: Verdict,
) -> None:
    """Each authored bridge in `dqn_bridges.py` reproduces its
    documented FINDINGS.md verdict on the action_dim_sweep
    corpus."""
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
    assert pg.measurable == 'jensen_gap'
    # Post-Phase-6: treatment/baseline arm strings are
    # canonical_str fingerprints derived from the typed
    # DoEffect's Intervention tuples.
    _baseline_key, _treatment_key = INTERVENTION.arm_keys()
    assert pg.treatment_arm == _treatment_key
    assert pg.baseline_arm == _baseline_key


# ============ Meta-regression: log_action_dim moderates g_mech ============


def test_meta_regression_reproduces_findings(
    action_dim_cells: list[dict[str, object]],
) -> None:
    """FINDINGS.md eighth revision: meta-regression of g_mech on
    log_action_dim shows the right direction (β negative) but
    n=4 envs is underpowered."""
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


# ============ Per-burst panel claims (revisions 9, 12) ============


EXPECTILE_RUNS = (
    REPO_ROOT / 'experiments' / 'data' / 'expectile_3way'
    / 'runs.parquet'
)
EXPECTILE_TRACES = (
    REPO_ROOT / 'experiments' / 'data' / 'expectile_3way'
    / 'traces.parquet'
)


@pytest.fixture(scope='module')
def expectile_per_burst_cells() -> list[dict[str, object]]:
    """Joined runs × traces, projected to columns the per-burst
    analysis needs. Skipped if the corpus isn't available."""
    if not (EXPECTILE_RUNS.exists() and EXPECTILE_TRACES.exists()):
        pytest.skip('expectile_3way corpus not available')
    runs = pl.read_parquet(
        EXPECTILE_RUNS,
        # The bridge dispatches `paired_g_per_burst` via
        # INTERVENTION.treatment_arm_key() → the canonical
        # `combined_arm_key`-derived string. The corpus stores
        # this in the `arm_key` column. Loading only the legacy
        # `intervention_name` column (an aliased semantic name)
        # leaves cells without the arm key the analysis filters
        # on → empty panel → spurious POWER_INSUFFICIENT.
        columns=['id', 'arm_key', 'env_name', 'seed'],
    )
    traces = pl.read_parquet(
        EXPECTILE_TRACES,
        columns=['id', 'mc_return', 'predicted_q_at_start'],
    )
    return list(
        runs.join(traces, on='id', how='inner').iter_rows(named=True),
    )


def test_per_burst_fourrooms_reproduces_revision_9(
    expectile_per_burst_cells: list[dict[str, object]],
) -> None:
    """FINDINGS revision 9: 'DDQN outcome benefit is stable
    across all bursts on FourRooms.' The bridge requires ≥9/10
    bursts positive AND mean g > 0.3 → HELD."""
    from experiments.findings.dqn_bridges import (
        ddqn_outcome_stable_across_bursts__fourrooms,
    )
    out = evaluate(
        ddqn_outcome_stable_across_bursts__fourrooms,
        expectile_per_burst_cells,
    )
    assert out.verdict == Verdict.HELD


def test_per_burst_catch_reproduces_revision_12(
    expectile_per_burst_cells: list[dict[str, object]],
) -> None:
    """FINDINGS revision 12: 'DDQN at n=1 has *exactly* zero
    effect (g = +0.00) on Catch — both arms saturate.' The
    bridge requires |g| < 0.1 across every burst → NO_EFFECT."""
    from experiments.findings.dqn_bridges import (
        ddqn_outcome_zero_across_bursts__catch,
    )
    out = evaluate(
        ddqn_outcome_zero_across_bursts__catch,
        expectile_per_burst_cells,
    )
    assert out.verdict == Verdict.NO_EFFECT


# ============ DoWhy SCV → outcome (revision 4) ============


CARTPOLE_HP_MEDIATORS = (
    REPO_ROOT / 'experiments' / 'data' / 'cartpole_hp_v2'
    / 'runs_with_mediators.parquet'
)


@pytest.fixture(scope='module')
def cartpole_hp_mediator_cells() -> list[dict[str, object]]:
    """Mediator-augmented CartPole HP corpus. Restores from R2
    if absent locally."""
    if not CARTPOLE_HP_MEDIATORS.exists():
        try:
            from corroborate.corpus.cloud import restore
            _ = restore(CARTPOLE_HP_MEDIATORS.parent)
        except Exception as exc:
            pytest.skip(
                f'cartpole_hp_v2 unavailable and restore failed: {exc}',
            )
    if not CARTPOLE_HP_MEDIATORS.exists():
        pytest.skip('cartpole_hp_v2 corpus unavailable')
    return list(
        pl.read_parquet(CARTPOLE_HP_MEDIATORS).iter_rows(named=True),
    )


def test_state_coverage_kl_causes_outcome_dowhy_bridge(
    cartpole_hp_mediator_cells: list[dict[str, object]],
) -> None:
    """FINDINGS revision 4: state_coverage_kl is the first
    mediator on the CartPole HP corpus that survives every check
    (backdoor + placebo + RCC). The multi-fixture bridge
    consumes all three analyses and asserts the conjunction.

    Verdict: HELD (ATE > 0, placebo destroys signal, RCC drift
    near zero). Exact ATE magnitude varies across CartPole HP
    corpora — the bridge claim is qualitative (sign + refuter
    survival), not magnitude-pinned."""
    from corroborate.analyses.dowhy import (
        BackdoorResult, RefutationResult,
    )
    from experiments.findings.dqn_bridges import (
        state_coverage_kl_causes_outcome,
    )
    out = evaluate(
        state_coverage_kl_causes_outcome,
        cartpole_hp_mediator_cells,
    )
    assert out.verdict == Verdict.HELD
    bd = cast(BackdoorResult, out.analysis_results['backdoor_ate'])
    pl_ = cast(
        RefutationResult,
        out.analysis_results['placebo_refutation'],
    )
    rcc = cast(
        RefutationResult,
        out.analysis_results['random_common_cause_refutation'],
    )
    assert bd.identified
    assert bd.ate > 0
    # Placebo refutation: refuted ATE should be much smaller than real.
    assert abs(pl_.refuted_ate) < 0.1 * abs(pl_.real_ate)
    # RCC drift small relative to real ATE.
    assert rcc.drift < 0.5
