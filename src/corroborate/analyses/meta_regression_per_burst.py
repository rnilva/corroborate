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

Implementation: builds the per-(env, burst) panel via
`paired_g_per_burst.fn`, projects each `PerBurstStratum` to a
`StratumG[tuple[str, int]]`, then runs `meta_regress_panel` —
the panel→regression bridge shared with
`meta_regression_paired_g`. Stratum-level covariates fall back
to the env-level row (covariates at the burst granularity would
require per-burst keys in `covariates_per_env`; the analysis
broadcasts the env-level vector across all bursts in that env).
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping

from corroborate.analyses.paired_g_per_burst import paired_g_per_burst
from corroborate.analysis import analysis
from corroborate.meta_regression import (
    MetaRegressionResult, meta_regress_panel,
)
from corroborate.stratum import StratumG


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
    panel: tuple[StratumG[tuple[str, int]], ...] = tuple(
        StratumG[tuple[str, int]](
            stratum_id=(s.env_name, s.burst_index),
            g=s.g, se=s.se, n_pairs=s.n_pairs,
        )
        for s in per_burst.strata
    )
    # Broadcast env-level covariates across every (env, burst).
    covariates_per_stratum: dict[
        tuple[str, int], Mapping[str, float],
    ] = {}
    for s in panel:
        env, _ = s.stratum_id
        if env in covariates_per_env:
            covariates_per_stratum[s.stratum_id] = (
                covariates_per_env[env]
            )
    return meta_regress_panel(
        panel,
        covariates_per_stratum=covariates_per_stratum,
        alpha=alpha,
    )


__all__ = ['meta_regression_per_burst']
