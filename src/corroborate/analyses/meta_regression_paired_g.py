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
`paired_g_pooled` and `meta_regression_per_burst`."""
from __future__ import annotations

from collections.abc import Iterable, Mapping

from corroborate.analyses.paired_g import per_env_paired_g_panel
from corroborate.analysis import analysis
from corroborate.meta_regression import (
    MetaRegressionResult, meta_regress_panel,
)


@analysis
def meta_regression_paired_g(
    cells: Iterable[Mapping[str, object]],
    *,
    treatment_arm: str,
    baseline_arm: str,
    pair_by: tuple[str, ...],
    source: str,
    covariates_per_env: Mapping[str, Mapping[str, float]],
    alpha: float = 0.05,
) -> MetaRegressionResult:
    """For each unique `env_name` in `cells`, compute paired g
    on `source` (treatment vs baseline by `pair_by`), then
    meta-regress those g/se on env-level covariates.

    `covariates_per_env` is the env-keyed covariate vector
    (e.g. `{'Acrobot-v1': {'log_action_dim': log(3)}}`). Envs
    not in the mapping contribute no covariates (still
    included via intercept). Envs with `n_pairs < 2` or NaN g
    are dropped from the panel."""
    panel = per_env_paired_g_panel(
        list(cells),
        treatment_arm=treatment_arm,
        baseline_arm=baseline_arm,
        source=source,
        pair_by=pair_by,
    )
    return meta_regress_panel(
        panel,
        covariates_per_stratum=covariates_per_env,
        alpha=alpha,
    )


__all__ = ['meta_regression_paired_g']
