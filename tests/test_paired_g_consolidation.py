"""Tests for the Phase 8 paired_g consolidation surface.

Two entry points to exercise:

- `per_env_paired_g_panel(...)` — shared per-env loop helper.
- `meta_regress_panel(panel, covariates_per_stratum=...)` —
  StratumG[K] panel → MetaRegressionResult bridge.

Cell-level scope (env_name / extra_filters / extra_min_pairs /
cell_predicate) lives upstream on `Bridge.scope` as a polars
Expr; analyses themselves no longer accept those kwargs. Tests
that need scoped cell-sets pre-filter the input.

The legacy paired_g_pooled / paired_g_among_solvers /
meta_regression_paired_g entry points are exercised indirectly
via their existing tests + the bridge files that consume them;
this test module focuses on the freshly-introduced helpers."""
from __future__ import annotations

from collections.abc import Mapping

from corroborate.analyses.paired_g import paired_g, per_env_paired_g_panel
from corroborate.stats import meta_regress_panel
from corroborate.corpus.schema import StratumG


def _cell(env: str, arm: str, seed: int, value: float) -> Mapping[str, object]:
    return {
        'env_name': env,
        'arm_key': arm,
        'seed': seed,
        'eval_best_burst_mean': value,
    }


def _two_env_corpus() -> list[Mapping[str, object]]:
    """Two envs × two arms × four seeds. ddqn beats vanilla on
    Acrobot (g positive); ties on CartPole."""
    rows: list[Mapping[str, object]] = []
    for s in range(4):
        rows.append(_cell('Acrobot-v1', 'ddqn', s, 1.0 + 0.1 * s))
        rows.append(_cell('Acrobot-v1', 'vanilla_dqn', s, 0.5 + 0.1 * s))
        rows.append(_cell('CartPole-v1', 'ddqn', s, 1.0 + 0.05 * s))
        rows.append(_cell('CartPole-v1', 'vanilla_dqn', s, 1.0 + 0.05 * s))
    return rows


def _filter_env(
    cells: list[Mapping[str, object]], env: str,
) -> list[Mapping[str, object]]:
    """Pre-filter cells by env. Bridges express this via
    `scope=pl.col('env_name') == env`; tests calling
    paired_g.fn directly handle it inline."""
    return [c for c in cells if c.get('env_name') == env]


# ============ paired_g pre-scoped input ============

def test_paired_g_on_pre_scoped_subset() -> None:
    """Pre-filtering cells to one env yields paired-g over that env."""
    cells = _filter_env(_two_env_corpus(), 'Acrobot-v1')
    result = paired_g.fn(
        cells,
        treatment_arm='ddqn',
        baseline_arm='vanilla_dqn',
        pair_by=('seed',),
        source='eval_best_burst_mean',
    )
    assert result.n_pairs == 4
    assert result.g > 0.0


def test_paired_g_on_empty_subset() -> None:
    """An empty cell-set yields n_pairs == 0 (NaN g/se)."""
    result = paired_g.fn(
        [],
        treatment_arm='ddqn',
        baseline_arm='vanilla_dqn',
        pair_by=('seed',),
        source='eval_best_burst_mean',
    )
    assert result.n_pairs == 0
    # NaN — written as `g != g` to avoid a math import.
    assert result.g != result.g


# ============ per_env_paired_g_panel ============

def test_per_env_panel_returns_one_stratum_per_env() -> None:
    """Auto-detects envs from corpus when env_filter empty;
    returns a StratumG per env."""
    panel = per_env_paired_g_panel(
        _two_env_corpus(),
        treatment_arm='ddqn',
        baseline_arm='vanilla_dqn',
        source='eval_best_burst_mean',
    )
    assert len(panel) == 2
    by_env = {s.stratum_id: s for s in panel}
    assert {'Acrobot-v1', 'CartPole-v1'} == set(by_env)
    assert by_env['Acrobot-v1'].n_pairs == 4
    # Acrobot has a clear positive g (ddqn > vanilla); CartPole's
    # g is exactly zero (per-pair Δ is constant 0). The exact
    # values depend on hedges_g_paired's small-n correction; we
    # only assert sign-and-power.
    assert by_env['Acrobot-v1'].g > 0.0
    # CartPole tie: deltas are all zero → SE undefined. The panel
    # records the result and lets the caller filter.
    assert by_env['CartPole-v1'].n_pairs == 4


def test_per_env_panel_respects_env_filter() -> None:
    """Explicit env_filter restricts the panel."""
    panel = per_env_paired_g_panel(
        _two_env_corpus(),
        treatment_arm='ddqn',
        baseline_arm='vanilla_dqn',
        source='eval_best_burst_mean',
        env_filter=('Acrobot-v1',),
    )
    assert len(panel) == 1
    assert panel[0].stratum_id == 'Acrobot-v1'


# ============ meta_regress_panel ============

def test_meta_regress_panel_drops_underpowered_strata() -> None:
    """Strata with n_pairs < 2 / NaN / se<=0 are dropped at the
    panel→regression boundary."""
    panel: tuple[StratumG[str], ...] = (
        StratumG[str](stratum_id='env_a', g=0.5, se=0.1, n_pairs=10),
        StratumG[str](stratum_id='env_b', g=0.6, se=0.12, n_pairs=10),
        StratumG[str](stratum_id='env_c', g=0.55, se=0.11, n_pairs=10),
        StratumG[str](stratum_id='env_d', g=0.7, se=0.13, n_pairs=10),
        # Underpowered: n_pairs < 2.
        StratumG[str](stratum_id='env_e', g=float('nan'), se=float('nan'), n_pairs=1),
    )
    covariates = {
        'env_a': {'x': 1.0},
        'env_b': {'x': 2.0},
        'env_c': {'x': 3.0},
        'env_d': {'x': 4.0},
        'env_e': {'x': 5.0},
    }
    result = meta_regress_panel(
        panel, covariates_per_stratum=covariates, alpha=0.05,
    )
    # 4 valid strata contributed; underpowered env_e dropped.
    assert result.n_strata == 4


def test_meta_regress_panel_no_covariates_for_stratum_uses_empty() -> None:
    """Strata absent from covariates_per_stratum get an empty
    covariate vector — they contribute via intercept only."""
    panel: tuple[StratumG[str], ...] = (
        StratumG[str](stratum_id='a', g=0.5, se=0.1, n_pairs=8),
        StratumG[str](stratum_id='b', g=0.6, se=0.12, n_pairs=8),
        StratumG[str](stratum_id='c', g=0.55, se=0.11, n_pairs=8),
    )
    # Only 'a' has a covariate.
    covariates = {'a': {'x': 1.0}}
    result = meta_regress_panel(
        panel, covariates_per_stratum=covariates, alpha=0.05,
    )
    assert result.n_strata == 3
