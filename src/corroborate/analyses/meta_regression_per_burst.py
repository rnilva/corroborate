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

import math
from collections.abc import Iterable, Mapping, Sequence

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
    # Resolve per-env covariates from either the explicit
    # `covariates_per_env` mapping or by averaging the named
    # `covariates` columns across cells.
    if covariates_per_env is not None:
        env_covariates: Mapping[str, Mapping[str, float]] = (
            covariates_per_env
        )
    elif covariates:
        env_covariates = _env_means_from_cells(cells_list, covariates)
    else:
        env_covariates = {}

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
        if env in env_covariates:
            covariates_per_stratum[s.stratum_id] = env_covariates[env]
    return meta_regress_panel(
        panel,
        covariates_per_stratum=covariates_per_stratum,
        alpha=alpha,
    )


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
