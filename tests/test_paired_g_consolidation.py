"""Edge-case tests for the paired_g consolidation surface.

The happy-path coverage of `paired_g.fn` and
`per_env_paired_g_panel` is now centralized at the closed-form
analytic substrate:

- `tests/analytic/lg_scm/test_paired_g.py` — closed-form
  `mean_diff` and Hedges' `g` recovery on real LG-SCM cells.
- `tests/analytic/lg_scm/test_random_effects_verdict.py` —
  `per_env_paired_g_panel` driven through the corroboration
  pipeline with closed-form pooled g check.

What stays here: edge cases the analytic substrate can't
naturally reach — empty cell-sets (NaN g/se contract) and
`meta_regress_panel`'s underpowered/missing-covariate dropping
policy on hand-built panels."""
from __future__ import annotations

from corroborate.analyses.paired_g import paired_g
from corroborate.stats import meta_regress_panel
from corroborate.corpus.schema import StratumG


# ============ paired_g empty-input contract ============

def test_paired_g_on_empty_subset() -> None:
    """An empty cell-set yields n_pairs == 0 (NaN g/se). Bridges
    that scope into an empty subset rely on this contract."""
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
