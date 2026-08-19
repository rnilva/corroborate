"""`meta_regression_paired_g` — per-env paired g, then random-
effects meta-regression on env-level covariates.

The shape FINDINGS.md eighth and tenth revisions consume:

  per-env paired Hedges' g  →  (env, g, se, covariates) panel
                            →  inverse-variance-weighted regression
                            →  per-coefficient verdict

A bridge consuming this result asserts a per-coefficient claim
("the `log_action_dim` slope on `g_mech` is negative") with its
own threshold (sign, magnitude, p-value).

Implementation: builds the per-env panel via
`per_env_paired_g_panel`, then projects to a meta-regression
result via `meta_regress_panel`. Both helpers are shared with
`meta_regression_per_burst` and other panel-shape primitives."""
from __future__ import annotations

from collections.abc import Iterable, Mapping

import polars as pl

from corroborate._internals.polars import as_rows
from corroborate.analyses.paired.paired_g import per_env_paired_g_panel
from corroborate.bridge.analysis import analysis
from corroborate.stats import (
    MetaRegressionResult, meta_regress_panel,
)
from corroborate.stats.meta_regression import Pool


@analysis
def meta_regression_paired_g(
    cells: pl.DataFrame | Iterable[Mapping[str, object]],
    *,
    treatment_arm: str,
    baseline_arm: str,
    arm_field: str = 'arm_key',
    pair_by: tuple[str, ...],
    source: str,
    covariates_per_env: Mapping[str, Mapping[str, float]],
    alpha: float = 0.05,
    pool: Pool = 'random',
) -> MetaRegressionResult:
    """For each unique `env_name` in `cells`, compute paired g
    on `source` (treatment vs baseline by `pair_by`), then
    meta-regress those g/se on env-level covariates.

    `covariates_per_env` is the env-keyed covariate vector
    (e.g. `{'Acrobot-v1': {'log_action_dim': log(3)}}`). Envs
    not in the mapping contribute no covariates (still
    included via intercept). Envs with `n_pairs < 2` or NaN g
    are dropped from the panel.

    Returns a NaN-coefficient result (empty `coefficients` tuple,
    intercept=NaN, r_squared=NaN) when the panel is too small for
    OLS (n_strata ≤ 1 + n_covariates) or when any stratum has
    invalid SE / singular design. Bridges checking `coef is None`
    via tuple-search naturally fall through to POW_INSUF. The
    underlying `meta_regress_panel` still fail-loud raises for
    direct callers that haven't opted into this graceful shape."""
    cells = as_rows(cells)
    panel = per_env_paired_g_panel(
        list(cells),
        treatment_arm=treatment_arm,
        baseline_arm=baseline_arm,
        arm_field=arm_field,
        source=source,
        pair_by=pair_by,
    )
    try:
        return meta_regress_panel(
            panel,
            covariates_per_stratum=covariates_per_env,
            alpha=alpha,
            pool=pool,
        )
    except ValueError:
        # NaN sentinel-prone fields so downstream readers see
        # "unfit" not "homogeneous fit." `pool` and `i_squared`
        # default to misleading values (`'fixed'`, `0.0`) per
        # MetaRegressionResult's frozen-dataclass defaults.
        return MetaRegressionResult(
            n_strata=len(panel),
            intercept=float('nan'),
            coefficients=(),
            r_squared=float('nan'),
            intercept_se=float('nan'),
            intercept_ci_lo=float('nan'),
            intercept_ci_hi=float('nan'),
            intercept_p_value=float('nan'),
            tau_sq=float('nan'),
            q_statistic=float('nan'),
            i_squared=float('nan'),
            pool=pool,
        )


__all__ = ['meta_regression_paired_g']
