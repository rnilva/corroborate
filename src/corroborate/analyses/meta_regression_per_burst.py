"""`meta_regression_per_burst` — per-(env, burst) panel meta-regression.

The shape FINDINGS revision 10's chain decomposition consumes:
for each (env, burst), compute paired Hedges' g on a target
(g_link from `mc_return`, g_mech from `mc_minus_q`); meta-regress
the resulting (env, burst) → (g, se) panel on env-level
covariates.

Generalizes `meta_regression_paired_g` (which strata-grains on
env only): the panel is now (env, burst), one row per
eval-burst-within-env. Covariates remain env-level — they're
attributes of the env, not the burst.

Reproduces revision 10's chain-decomposition shape:
  β(log_action_dim) on g_mech: −0.39, p=0.005 (HELD)
  β(log_action_dim) on g_link: +0.01, p=0.94 (NO_EFFECT)
  → action-dim moderates the mechanism but not the link.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

from corroborate.analyses.paired_g_per_burst import paired_g_per_burst
from corroborate.analysis import analysis
from corroborate.meta_regression import (
    MetaRegressionResult, StratumObservation, meta_regression,
)


@analysis
def meta_regression_per_burst(
    cells: Iterable[Mapping[str, object]],
    *,
    treatment_arm: str,
    baseline_arm: str,
    pair_by: tuple[str, ...] = ('seed',),
    source: str = 'mc_return',
    reduction: str = 'mean',
    covariates_per_env: Mapping[str, Mapping[str, float]],
    alpha: float = 0.05,
) -> MetaRegressionResult:
    """Per-(env, burst) panel: paired g on `source`/`reduction`
    for each (env, burst), then meta-regression on env-level
    covariates.

    `covariates_per_env` is the env-keyed covariate vector
    (e.g. `{'Acrobot-v1': {'log_action_dim': log(3),
    'log_obs_dim': log(6)}}`). Each (env, burst) stratum inherits
    the env's covariates — covariates at the burst granularity
    (e.g. eval_step_index) require an extension this analysis
    doesn't ship today.

    Strata with NaN g/SE or zero variance are dropped from the
    panel."""
    cells_list = [dict(c) for c in cells]
    per_burst = paired_g_per_burst.fn(
        cells_list,
        treatment_arm=treatment_arm,
        baseline_arm=baseline_arm,
        pair_by=pair_by,
        source=source,
        reduction=reduction,
    )

    observations: list[StratumObservation] = []
    for s in per_burst.strata:
        if s.n_pairs < 2 or math.isnan(s.g) or math.isnan(s.se):
            continue
        if s.se <= 0.0:
            continue
        env_covs = covariates_per_env.get(s.env_name, {})
        observations.append(StratumObservation(
            stratum_id=(s.env_name, s.burst_index),
            g=s.g,
            se=s.se,
            covariates=env_covs,
        ))

    return meta_regression(observations, alpha=alpha)


__all__ = ['meta_regression_per_burst']
