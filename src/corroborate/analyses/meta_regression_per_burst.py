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
from collections.abc import Iterable, Mapping, Sequence

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
    covariates: tuple[str, ...] = (),
    covariates_per_env: Mapping[str, Mapping[str, float]] | None = None,
    alpha: float = 0.05,
) -> MetaRegressionResult:
    """Per-(env, burst) panel: paired g on `source`/`reduction`
    for each (env, burst), then meta-regression on env-level
    covariates.

    Two paths for supplying covariates:

    - `covariates: tuple[str, ...]` (preferred) — column names on
      the cells. The analysis groups by `env_name` and takes the
      per-env mean of each named column to form the env-keyed
      covariate vector. Covariate values come from the corpus
      itself; bridges declare which columns matter, not the
      frozen values. Combine with materialised
      `@measurable`-derived columns (e.g. `log_action_dim`,
      `bootstrap_fraction`) for the env-level features.
    - `covariates_per_env: Mapping[env, Mapping[name, value]]`
      (legacy) — env-keyed value-bag. Used when the bridge needs
      a frozen reference (e.g. the original-corpus moments). Wins
      when both are set.

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

    if covariates_per_env is not None:
        env_covariates: Mapping[str, Mapping[str, float]] = (
            covariates_per_env
        )
    elif covariates:
        env_covariates = _env_means_from_cells(cells_list, covariates)
    else:
        env_covariates = {}

    observations: list[StratumObservation] = []
    for s in per_burst.strata:
        if s.n_pairs < 2 or math.isnan(s.g) or math.isnan(s.se):
            continue
        if s.se <= 0.0:
            continue
        env_covs = env_covariates.get(s.env_name, {})
        observations.append(StratumObservation(
            stratum_id=(s.env_name, s.burst_index),
            g=s.g,
            se=s.se,
            covariates=env_covs,
        ))

    return meta_regression(observations, alpha=alpha)


def _env_means_from_cells(
    cells: Sequence[Mapping[str, object]],
    columns: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    """Build `{env_name: {col: mean(col over env's cells)}}` from
    a per-cell list. NaN-skip per column. Cells lacking `env_name`
    or with non-numeric column values are excluded from that
    column's mean.

    Used to lift cell-level columns (typically materialised by
    the @measurable cache) to env-level covariates for the
    meta-regression's stratum panel."""
    by_env: dict[str, dict[str, list[float]]] = {}
    for cell in cells:
        env = cell.get('env_name')
        if not isinstance(env, str):
            continue
        slot = by_env.setdefault(env, {})
        for col in columns:
            v = cell.get(col)
            if not isinstance(v, (int, float)):
                continue
            f = float(v)
            if math.isnan(f):
                continue
            slot.setdefault(col, []).append(f)
    out: dict[str, dict[str, float]] = {}
    for env, col_map in by_env.items():
        env_means: dict[str, float] = {}
        for col, vs in col_map.items():
            if vs:
                env_means[col] = sum(vs) / len(vs)
        out[env] = env_means
    return out


__all__ = ['meta_regression_per_burst']
