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

    # Closed-form WLS recovery on the 4 surviving strata.
    # w_i = 1/se_i² = [100, 69.4, 82.6, 59.2]; Σw = 311.2.
    # x   = [1, 2, 3, 4]; g = [0.5, 0.6, 0.55, 0.7].
    # Σwx = 723.4; Σwx² = 2068.2; Σwg = 178.51; Σwxg = 435.33.
    # β   = (Σw·Σwxg − Σwx·Σwg) / (Σw·Σwx² − (Σwx)²) ≈ 0.0525
    # α   = (Σwg − β·Σwx) / Σw                       ≈ 0.4516
    #
    # Without these closed-form bounds, a stub returning
    # `n_strata=4, coefficients=[]` would pass `n_strata == 4` AND
    # is the dominant theatre risk (the regression's actual fit
    # was never verified). Pin both the structural intercept AND
    # the x slope on the dropped-stratum panel so a stub that
    # silently kept env_e's NaN row would breach via slope drift
    # toward the include-NaN OLS.
    assert abs(result.intercept - 0.4516) < 0.001, (
        f'intercept = {result.intercept:.4f}, closed-form ≈ 0.4516 '
        f'(WLS on the 4 surviving strata). A regression that '
        f'silently included the underpowered env_e (NaN) row would '
        f'shift the intercept.'
    )
    by_name = {c.name: c for c in result.coefficients}
    assert 'x' in by_name, (
        f'x covariate missing from coefficients = {list(by_name)}'
    )
    assert abs(by_name['x'].coefficient - 0.0525) < 0.001, (
        f'x coef = {by_name["x"].coefficient:.4f}, closed-form '
        f'WLS slope ≈ 0.0525. A stub returning empty/zero '
        f'coefficients would fail this.'
    )


def test_meta_regress_panel_no_covariates_for_stratum_uses_empty() -> None:
    """Strata absent from covariates_per_stratum get an empty
    covariate vector — they contribute via intercept only.

    With only 'a' carrying a covariate {x: 1.0}, all 3 strata are
    kept (n_strata=3). Pin the intercept to the WLS-pooled g across
    the three:
        w   = [100, 69.4, 82.6]; Σw  = 252.0
        Σwg = [50, 41.64, 45.43] →   137.07
        intercept ≈ Σwg / Σw         = 0.5439

    A stub returning the unweighted mean (0.55) would breach by 0.006.
    """
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
