"""`meta_regression_paired_g` — per-env paired g, then random-
effects meta-regression on env-level covariates.

The shape FINDINGS.md eighth and tenth revisions consume:

  per-env paired Hedges' g  →  (env, g, se, covariates) panel
                            →  inverse-variance-weighted regression
                            →  per-coefficient verdict

A bridge consuming this result asserts a per-coefficient claim
("the `log_action_dim` slope on `g_mech` is negative") with its
own threshold (sign, magnitude, p-value).

This is one fused analysis (panel-build + regression) for
simplicity; a future split could expose the panel as a separate
analysis a bridge consumes alongside the regression."""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

from corroborate.analyses.paired_g import paired_g
from corroborate.analysis import analysis
from corroborate.meta_regression import (
    MetaRegressionResult, StratumObservation, meta_regression,
)


@analysis(name='meta_regression_paired_g')
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
    cells_list = [dict(c) for c in cells]
    env_names: set[str] = set()
    for c in cells_list:
        env_v = c.get('env_name')
        if isinstance(env_v, str):
            env_names.add(env_v)

    observations: list[StratumObservation] = []
    for env in sorted(env_names):
        per_env = paired_g.fn(
            cells_list,
            treatment_arm=treatment_arm,
            baseline_arm=baseline_arm,
            pair_by=pair_by,
            source=source,
            env_name=env,
        )
        if per_env.n_pairs < 2:
            continue
        if math.isnan(per_env.g) or math.isnan(per_env.se):
            continue
        if per_env.se <= 0.0:
            continue
        observations.append(StratumObservation(
            stratum_id=env,
            g=per_env.g,
            se=per_env.se,
            covariates=covariates_per_env.get(env, {}),
        ))

    return meta_regression(observations, alpha=alpha)


__all__ = ['meta_regression_paired_g']
