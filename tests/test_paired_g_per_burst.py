"""Smoke for the per-burst paired-g panel analysis.

Synthetic corpus + a real-corpus sanity check on
expectile_3way. The synthetic test verifies the analysis returns
the expected per-burst structure; the real-corpus test confirms
qualitative reproduction of FINDINGS revision 12's "Catch
g≈0.00 across bursts" observation."""
from __future__ import annotations

import random
from pathlib import Path

import polars as pl
import pytest

import corroborate.analyses  # noqa: F401  # pyright: ignore[reportUnusedImport]

from corroborate.analyses.paired_g_per_burst import (
    DEFAULT_PER_BURST_SOURCE, panel_for_env, paired_g_per_burst,
)
from corroborate.reductions import from_key, reduce_axis
from corroborate.rl.dqn.measurables import jensen_bias_per_eps


def _synthetic_burst_cells(
    *,
    n_seeds: int = 30,
    n_bursts: int = 5,
    n_episodes: int = 3,
    treatment_burst_means: list[float] | None = None,
    baseline_burst_means: list[float] | None = None,
    noise: float = 0.05,
) -> list[dict[str, object]]:
    """Build a corpus where each cell carries a (n_bursts, n_episodes)
    `mc_return` array. `treatment_burst_means` lists the per-burst
    expected mean for the treatment arm; baseline mirrors. Different
    means → non-zero g per burst."""
    if treatment_burst_means is None:
        treatment_burst_means = [1.0] * n_bursts
    if baseline_burst_means is None:
        baseline_burst_means = [0.0] * n_bursts
    rng = random.Random(0)
    out: list[dict[str, object]] = []
    for s in range(n_seeds):
        for arm, means in (
            ('treatment', treatment_burst_means),
            ('baseline', baseline_burst_means),
        ):
            mc = [
                [m + rng.gauss(0, noise) for _ in range(n_episodes)]
                for m in means
            ]
            out.append({
                'intervention_name': arm,
                'env_name': 'TestEnv',
                'seed': s,
                'mc_return': mc,
            })
    return out


def test_per_burst_synthetic_strong_signal() -> None:
    """Per-burst means [+0.0, +0.5, +1.0, +1.5, +2.0] should
    produce monotonically growing |g| across bursts."""
    cells = _synthetic_burst_cells(
        treatment_burst_means=[0.0, 0.5, 1.0, 1.5, 2.0],
        baseline_burst_means=[0.0, 0.0, 0.0, 0.0, 0.0],
    )
    result = paired_g_per_burst.fn(
        cells,
        treatment_arm='treatment',
        baseline_arm='baseline',
        pair_by=('seed',),
        source=DEFAULT_PER_BURST_SOURCE,
    )
    assert result.n_strata == 5
    panel = panel_for_env(result, 'TestEnv')
    # First burst: g ≈ 0 (no signal). Last: g large.
    assert abs(panel[0].g) < 1.0, panel[0].g
    assert panel[-1].g > 5.0, panel[-1].g
    # Monotone growth in g.
    gs = [s.g for s in panel]
    assert gs == sorted(gs), f'expected monotone g across bursts: {gs}'


def test_per_burst_synthetic_no_signal() -> None:
    """Equal means → g ≈ 0 across all bursts."""
    cells = _synthetic_burst_cells(
        treatment_burst_means=[1.0] * 5,
        baseline_burst_means=[1.0] * 5,
    )
    result = paired_g_per_burst.fn(
        cells,
        treatment_arm='treatment',
        baseline_arm='baseline',
        pair_by=('seed',),
        source=DEFAULT_PER_BURST_SOURCE,
    )
    panel = panel_for_env(result, 'TestEnv')
    for s in panel:
        assert abs(s.g) < 1.0, f'expected g ≈ 0, got {s.g}'


def test_jensen_bias_per_eps_reduction() -> None:
    """Composing the named `jensen_bias_per_eps` measurable with
    `reduce_axis(_, axis=-1, op='mean')` produces the Jensen-bias
    per-burst gap (Q − MC), the same quantity the old
    `reduction='mc_minus_q'` string-dispatch did."""
    rng = random.Random(0)
    cells: list[dict[str, object]] = []
    for s in range(20):
        for arm, q_mean, mc_mean in (
            ('treatment', 1.0, 0.5),  # ddqn: small bias
            ('baseline', 1.5, 0.5),    # vanilla: bigger bias
        ):
            cells.append({
                'intervention_name': arm,
                'env_name': 'TestEnv',
                'seed': s,
                'predicted_q_at_start': [
                    [q_mean + rng.gauss(0, 0.01) for _ in range(3)]
                    for _ in range(4)
                ],
                'mc_return': [
                    [mc_mean + rng.gauss(0, 0.01) for _ in range(3)]
                    for _ in range(4)
                ],
            })
    bias_per_burst_mean = reduce_axis(
        jensen_bias_per_eps, axis=-1, op='mean',
    )
    result = paired_g_per_burst.fn(
        cells,
        treatment_arm='treatment',
        baseline_arm='baseline',
        pair_by=('seed',),
        source=bias_per_burst_mean,
    )
    panel = panel_for_env(result, 'TestEnv')
    assert result.measurable == bias_per_burst_mean.name
    # treatment bias = 0.5 (q=1.0, mc=0.5), baseline bias = 1.0
    # Δ = -0.5 per burst → g should be strongly negative.
    for s in panel:
        assert s.g < -3.0, f'expected strongly negative g, got {s.g}'


def test_per_burst_via_explicit_from_key() -> None:
    """The default `DEFAULT_PER_BURST_SOURCE` and an explicit
    `reduce_axis(from_key('mc_return'), axis=-1, op='mean')` give
    the same per-burst panel — sanity that the default is just
    the canonical composition."""
    cells = _synthetic_burst_cells(
        treatment_burst_means=[0.0, 1.0, 2.0],
        baseline_burst_means=[0.0, 0.0, 0.0],
    )
    explicit = reduce_axis(from_key('mc_return'), axis=-1, op='mean')
    a = paired_g_per_burst.fn(
        cells, treatment_arm='treatment', baseline_arm='baseline',
        source=DEFAULT_PER_BURST_SOURCE,
    )
    b = paired_g_per_burst.fn(
        cells, treatment_arm='treatment', baseline_arm='baseline',
        source=explicit,
    )
    assert a.n_strata == b.n_strata
    for sa, sb in zip(
        sorted(a.strata, key=lambda s: s.burst_index),
        sorted(b.strata, key=lambda s: s.burst_index),
        strict=True,
    ):
        assert abs(sa.g - sb.g) < 1e-9


def test_analysis_registered() -> None:
    """`paired_g_per_burst` populates the registry on import."""
    from corroborate.analysis import get_registered
    assert get_registered('paired_g_per_burst') is not None


# ============ Real-corpus reproduction smoke ============

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPECTILE_RUNS = (
    REPO_ROOT / 'experiments' / 'data' / 'expectile_3way'
    / 'runs.parquet'
)
EXPECTILE_TRACES = (
    REPO_ROOT / 'experiments' / 'data' / 'expectile_3way'
    / 'traces.parquet'
)


@pytest.fixture(scope='module')
def expectile_3way_cells() -> list[dict[str, object]]:
    """Project to needed columns to keep memory bounded; join
    runs + traces on `id`."""
    if not (EXPECTILE_RUNS.exists() and EXPECTILE_TRACES.exists()):
        pytest.skip('expectile_3way corpus not available')
    runs = pl.read_parquet(
        EXPECTILE_RUNS,
        columns=['id', 'intervention_name', 'env_name', 'seed'],
    )
    traces = pl.read_parquet(
        EXPECTILE_TRACES,
        columns=['id', 'mc_return', 'predicted_q_at_start'],
    )
    return list(
        runs.join(traces, on='id', how='inner').iter_rows(named=True),
    )


def test_real_corpus_catch_bsuite_zero_across_bursts(
    expectile_3way_cells: list[dict[str, object]],
) -> None:
    """FINDINGS revision 12: 'DDQN at n=1 has *exactly* zero effect
    (g = +0.00) on Catch — both arms saturate at mc_return ≈ +0.92
    and converge to the same near-optimal policy.'

    Per-burst panel should show g ≈ 0 across every burst on Catch.
    """
    result = paired_g_per_burst.fn(
        expectile_3way_cells,
        treatment_arm='ddqn',
        baseline_arm='vanilla_dqn',
        pair_by=('seed',),
        source=DEFAULT_PER_BURST_SOURCE,
    )
    catch_panel = panel_for_env(result, 'Catch-bsuite')
    assert len(catch_panel) > 0
    for s in catch_panel:
        assert abs(s.g) < 0.1, (
            f'Catch burst {s.burst_index}: expected g ≈ 0, '
            f'got {s.g:+.3f}'
        )


def test_real_corpus_fourrooms_positive_across_bursts(
    expectile_3way_cells: list[dict[str, object]],
) -> None:
    """FINDINGS revision 9 (and revision 12 for the (C-A) cell):
    'g_link(C-A) ≈ +0.79 across bursts on FourRooms — DDQN
    benefit is stable throughout.'

    Per-burst panel should show positive g across every burst on
    FourRooms (DDQN improves outcome relative to vanilla)."""
    result = paired_g_per_burst.fn(
        expectile_3way_cells,
        treatment_arm='ddqn',
        baseline_arm='vanilla_dqn',
        pair_by=('seed',),
        source=DEFAULT_PER_BURST_SOURCE,
    )
    fr_panel = panel_for_env(result, 'FourRooms-misc')
    assert len(fr_panel) > 0
    positive_bursts = sum(1 for s in fr_panel if s.g > 0)
    assert positive_bursts >= len(fr_panel) - 1, (
        f'expected positive g on most FourRooms bursts; '
        f'got {positive_bursts}/{len(fr_panel)} positive'
    )
    mean_g = sum(s.g for s in fr_panel) / len(fr_panel)
    assert mean_g > 0.3, (
        f'expected mean burst g > 0.3 (FINDINGS reports ~+0.79); '
        f'got {mean_g:.3f}'
    )
