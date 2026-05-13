"""Panel-consumer fixtures: pooled aggregations built on top of
`stratum_panel`.

Each fixture takes `stratum_panel: StratumPanel` as a parameter-
injected fixture and produces a pooled result for bridges:

- `dl_pool(panel, *, source)`: DerSimonian-Laird random-effects
  pool of per-stratum Cohen's d on a single measurable.
- `fisher_z_pool(panel, *, x, y)`: Fisher-z pool of per-stratum
  within-stratum Spearman r between two measurables.
- `panel_partial(panel, *, x, y, z)`: closed-form partial Spearman
  ρ(x, y | z) per stratum, Fisher-z pooled.

Per-stratum data is accessible via the `panel` parameter passed
into the bridge body — these fixtures only build the pooled
scalar. The `scope_from_panel` primitive (sibling) reads per-
stratum data from the panel directly to build dynamic scope
filters.

Result types mirror the existing `StratifiedSpearmanResult` /
`StratifiedPartialSpearmanResult` / `PooledStats` shapes so the
existing verdict helpers (`partial_spearman_signed_verdict`,
`partial_spearman_null_verdict`, `random_effects_verdict`) work
without modification."""
from __future__ import annotations

import math
from dataclasses import dataclass

from corroborate.analyses.stratum_panel import (
    StratumPanel, pair_key,
)
from corroborate.bridge.analysis import analysis
from corroborate.stats.effect_size import (
    PooledStats, random_effects_summary,
)


@dataclass(frozen=True, slots=True)
class PanelSpearmanResult:
    """Fisher-z-pooled Spearman ρ on a per-stratum panel.

    Field shape mirrors `StratifiedSpearmanResult` so the existing
    `partial_spearman_signed_verdict` / `partial_spearman_null_verdict`
    helpers consume it without modification."""
    x: str
    y: str
    stratify_by: str
    rho_pooled: float
    p_value: float
    n_obs_total: int
    n_strata: int


@dataclass(frozen=True, slots=True)
class PanelPartialSpearmanResult:
    """Fisher-z-pooled partial Spearman ρ(x, y | z) on a per-
    stratum panel.

    Field shape mirrors `StratifiedPartialSpearmanResult`."""
    x: str
    y: str
    conditioning: str
    stratify_by: str
    rho_pooled: float
    p_value: float
    n_obs_total: int
    n_strata: int


def _fisher_z_pool(
    per_stratum_r: tuple[float, ...],
    per_stratum_n: tuple[int, ...],
    df_offset: int = 3,
) -> tuple[float, float]:
    """Fisher-z pool of per-stratum r-values, weighted by
    `(n_k − df_offset)`. `df_offset=3` for marginal Spearman;
    `df_offset=4` for partial Spearman (one extra DOF lost to the
    conditioning variable).

    Returns `(rho_pooled, p_two_sided)`."""
    z_vals: list[float] = []
    weights: list[float] = []
    for r, n in zip(per_stratum_r, per_stratum_n):
        if math.isnan(r):
            continue
        if n - df_offset < 1:
            continue
        r_c = max(-0.999999, min(0.999999, r))
        z = 0.5 * math.log((1 + r_c) / (1 - r_c))
        z_vals.append(z)
        weights.append(float(n - df_offset))
    if not z_vals:
        return float('nan'), float('nan')
    total_w = sum(weights)
    if total_w <= 0:
        return float('nan'), float('nan')
    z_pooled = sum(w * z for w, z in zip(weights, z_vals)) / total_w
    rho = math.tanh(z_pooled)
    z_stat = z_pooled * math.sqrt(total_w)
    from scipy.stats import norm
    p = float(2 * (1.0 - norm.cdf(abs(z_stat))))
    return rho, p


@analysis
def dl_pool(
    stratum_panel: StratumPanel,
    *,
    source: str,
) -> PooledStats:
    """DerSimonian-Laird random-effects pool of per-stratum
    Cohen's d on `source` measurable.

    Per-stratum (d, SE) pairs come from `panel.cohen_d(source)`
    and `panel.cohen_se(source)`; NaN-bearing or zero-SE strata
    are dropped by `random_effects_summary`.

    Returns the full `PooledStats` for bridges' verdict logic."""
    pairs = list(zip(
        stratum_panel.cohen_d(source),
        stratum_panel.cohen_se(source),
    ))
    return random_effects_summary(pairs)


@analysis
def fisher_z_pool(
    stratum_panel: StratumPanel,
    *,
    x: str,
    y: str,
) -> PanelSpearmanResult:
    """Fisher-z pool of per-stratum within-stratum Spearman r
    between `x` and `y` measurables.

    `x` and `y` must be in `stratum_panel.measurables`. The pre-
    computed `spearman_within[(x, y)]` is Fisher-z-pooled
    weighted by `(n_total_k − 3)` per stratum."""
    if x not in stratum_panel.measurables or y not in stratum_panel.measurables:
        raise KeyError(
            f'fisher_z_pool({x!r}, {y!r}): both must be in '
            f'panel.measurables {stratum_panel.measurables!r}',
        )
    key = pair_key(x, y)
    per_stratum_r = stratum_panel.spearman_within[key]
    per_stratum_n = tuple(
        stratum_panel.n_treatment[i] + stratum_panel.n_baseline[i]
        for i in range(stratum_panel.n_strata)
    )
    rho, p = _fisher_z_pool(per_stratum_r, per_stratum_n, df_offset=3)
    n_strata_contrib = sum(
        1 for r in per_stratum_r if not math.isnan(r)
    )
    return PanelSpearmanResult(
        x=x, y=y,
        stratify_by=stratum_panel.stratify_by[0]
        if stratum_panel.stratify_by else 'env_name',
        rho_pooled=rho,
        p_value=p,
        n_obs_total=sum(per_stratum_n),
        n_strata=n_strata_contrib,
    )


@analysis
def panel_partial(
    stratum_panel: StratumPanel,
    *,
    x: str,
    y: str,
    conditioning: str,
) -> PanelPartialSpearmanResult:
    """Closed-form partial Spearman ρ(x, y | z) per stratum,
    Fisher-z pooled.

    Per-stratum: pull pre-computed pairwise Spearman r values
    from `panel.spearman_within`, apply the three-correlation
    closed-form partial identity:

        r_xy.z = (r_xy − r_xz * r_yz) /
                  sqrt((1 − r_xz²) * (1 − r_yz²))

    Pool per-stratum partials via Fisher-z weighted by
    `(n_total_k − 4)` (one extra DOF lost to z)."""
    needed = {x, y, conditioning}
    missing = needed - set(stratum_panel.measurables)
    if missing:
        raise KeyError(
            f'panel_partial: {sorted(missing)!r} not in '
            f'panel.measurables {stratum_panel.measurables!r}',
        )
    r_xy_per = stratum_panel.spearman_within[pair_key(x, y)]
    r_xz_per = stratum_panel.spearman_within[pair_key(x, conditioning)]
    r_yz_per = stratum_panel.spearman_within[pair_key(y, conditioning)]
    partials: list[float] = []
    for i in range(stratum_panel.n_strata):
        rxy, rxz, ryz = r_xy_per[i], r_xz_per[i], r_yz_per[i]
        if any(math.isnan(r) for r in (rxy, rxz, ryz)):
            partials.append(float('nan'))
            continue
        denom = math.sqrt(
            max(1.0 - rxz ** 2, 0.0) * max(1.0 - ryz ** 2, 0.0)
        )
        if denom <= 1e-12:
            partials.append(float('nan'))
            continue
        rho_partial = (rxy - rxz * ryz) / denom
        rho_partial = max(-0.999999, min(0.999999, rho_partial))
        partials.append(rho_partial)
    per_stratum_n = tuple(
        stratum_panel.n_treatment[i] + stratum_panel.n_baseline[i]
        for i in range(stratum_panel.n_strata)
    )
    rho, p = _fisher_z_pool(
        tuple(partials), per_stratum_n, df_offset=4,
    )
    n_strata_contrib = sum(
        1 for r in partials if not math.isnan(r)
    )
    return PanelPartialSpearmanResult(
        x=x, y=y, conditioning=conditioning,
        stratify_by=stratum_panel.stratify_by[0]
        if stratum_panel.stratify_by else 'env_name',
        rho_pooled=rho,
        p_value=p,
        n_obs_total=sum(per_stratum_n),
        n_strata=n_strata_contrib,
    )


__all__ = [
    'PanelSpearmanResult',
    'PanelPartialSpearmanResult',
    'dl_pool',
    'fisher_z_pool',
    'panel_partial',
]
