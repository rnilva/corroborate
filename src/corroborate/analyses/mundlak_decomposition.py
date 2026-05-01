"""`mundlak_decomposition` — proper within/between regression for
panel-data moderator hypotheses.

Codifies the methodology lesson from the DDQN `log_mc_variance`
analysis (`findings_chain_bottlenecks.md` 2026-05-02 second
correction): claiming a within/between effect requires
decomposing **the same variable** into its env-mean component
and its env-mean-removed deviation, then regressing the target
on both. Joining two *different* aggregations of the same
quantity (e.g., pooled-over-everything vs averaged-within-burst)
and labelling them "between" and "within" is a misspecification
that produces spurious Simpson-paradox-shaped artifacts.

The Mundlak (1978) device:

    x_e = E[x_b | env]     # env-mean of the burst-level predictor
    x_w = x_b − x_e        # deviation from env-mean (within-env)

    y ~ β_b · x_e + β_w · x_w + ε

`β_b` is the between-env coefficient (interpretable as
moderation by env-level structure); `β_w` is the within-env
coefficient (moderation by burst-level deviations from each
env's typical level). They are statistically independent by
construction (`r(x_e, x_w) = 0`) — the joint regression cleanly
separates the two channels.

Hausman-style test: under the null that x has no within/between
distinction (a single causal effect), `β_b == β_w`. A
significant difference (e.g., opposite signs) indicates
heterogeneous moderation across the two scales.

This primitive is the framework's way of forcing future moderator
audits to declare which level (between/within) they're claiming
about, rather than mixing aggregations and getting Simpson
artifacts."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import scipy.stats as ss

from corroborate.analysis import analysis


@dataclass(frozen=True, slots=True)
class MundlakObservation:
    """One panel observation. `stratum_id` defines the
    between-level grouping (typically env_name); `x` is the
    burst-level predictor whose within/between effect we want to
    decompose; `y` and `se` are the target effect-size and its
    inverse-variance weight (SE)."""
    stratum_id: object
    x: float
    y: float
    se: float


@dataclass(frozen=True, slots=True)
class MundlakCoefficient:
    """One Mundlak coefficient with its WLS-OLS stats."""
    component: str  # 'between' or 'within'
    coefficient: float
    se: float
    p_value: float
    ci_lo: float
    ci_hi: float


@dataclass(frozen=True, slots=True)
class MundlakResult:
    """Output of a Mundlak decomposition run.

    `between` is `β_b` (env-mean coefficient); `within` is `β_w`
    (within-env-deviation coefficient). `hausman_p` is the
    p-value for `β_b == β_w` (testing whether the two channels
    have the same effect — i.e., whether decomposition was
    necessary).

    `r_squared` is the weighted R² of the joint model. `n_strata`
    is the number of distinct `stratum_id` values; `n_obs` is the
    panel row count."""
    n_strata: int
    n_obs: int
    intercept: float
    between: MundlakCoefficient
    within: MundlakCoefficient
    hausman_p: float
    r_squared: float


@analysis
def mundlak_decomposition(
    panel: Iterable[Mapping[str, object]],
    *,
    stratum_key: str = 'stratum_id',
    x_key: str = 'x',
    y_key: str = 'y',
    se_key: str = 'se',
    alpha: float = 0.05,
    cluster_robust: bool = False,
) -> MundlakResult:
    """Decompose `x` into `x_e` (between-env, env-mean) + `x_w`
    (within-env-deviation), then run weighted OLS:
    `y ~ β_b · x_e + β_w · x_w`. Weights = `1 / se²`.

    Returns both coefficients with their SEs/p-values and a
    Hausman-style test for `β_b == β_w`. The two predictors are
    orthogonal by construction — joint regression doesn't suffer
    multicollinearity between them.

    `cluster_robust` (default False): when True, compute
    Liang-Zeger CR1 sandwich SEs clustering at the stratum level
    — required for time-series-within-stratum panels (e.g., per
    -step training data) where residuals within a stratum are
    autocorrelated. The default OLS-style SE assumes IID residuals
    within stratum and overstates significance under
    autocorrelation. Use cluster_robust=True for any panel with
    n_obs / n_strata >> 1 and within-stratum dependence.

    Use when you want to claim a moderator effect on a
    panel-data target where the predictor has both env-level and
    within-env variation. Forces an explicit choice of which
    level is being claimed about."""
    panel_list = [dict(p) for p in panel]
    if not panel_list:
        raise ValueError('panel must contain at least one observation')

    # Extract typed columns
    strata: list[object] = []
    xs: list[float] = []
    ys: list[float] = []
    ses: list[float] = []
    for row in panel_list:
        s = row.get(stratum_key)
        x_v = row.get(x_key)
        y_v = row.get(y_key)
        se_v = row.get(se_key)
        if s is None or not isinstance(x_v, (int, float)) \
                or not isinstance(y_v, (int, float)) \
                or not isinstance(se_v, (int, float)):
            continue
        if se_v <= 0.0:
            continue
        strata.append(s)
        xs.append(float(x_v))
        ys.append(float(y_v))
        ses.append(float(se_v))

    if len(xs) < 4:
        raise ValueError(
            f'mundlak_decomposition needs at least 4 valid '
            f'observations; got {len(xs)}',
        )

    x_arr: npt.NDArray[np.float64] = np.asarray(xs, dtype=np.float64)
    y_arr: npt.NDArray[np.float64] = np.asarray(ys, dtype=np.float64)
    w_arr: npt.NDArray[np.float64] = 1.0 / np.asarray(ses, dtype=np.float64) ** 2

    # env-mean of x for each row, broadcast back. Use a typed
    # dict-keyed accumulator since `Sequence[object]` admits
    # heterogeneous keys (str, int, tuple, …).
    sum_x: dict[object, float] = {}
    n_x: dict[object, int] = {}
    for s, v in zip(strata, xs):
        sum_x[s] = sum_x.get(s, 0.0) + v
        n_x[s] = n_x.get(s, 0) + 1
    env_mean_x: dict[object, float] = {
        s: sum_x[s] / n_x[s] for s in sum_x
    }
    x_e: npt.NDArray[np.float64] = np.asarray(
        [env_mean_x[s] for s in strata], dtype=np.float64,
    )
    x_w: npt.NDArray[np.float64] = x_arr - x_e

    # Weighted OLS: design matrix [1, x_e, x_w].
    design: npt.NDArray[np.float64] = np.column_stack([
        np.ones_like(x_arr), x_e, x_w,
    ])
    # WLS via Cholesky on weighted normal equations
    sqrt_w = np.sqrt(w_arr)
    weighted_design = design * sqrt_w[:, None]
    weighted_y = y_arr * sqrt_w
    coefs, _resid_ss, rank, _ = np.linalg.lstsq(
        weighted_design, weighted_y, rcond=None,
    )
    if rank < design.shape[1]:
        raise ValueError(
            f'Mundlak design matrix is rank-deficient '
            f'(rank={rank}, expected {design.shape[1]}); the '
            f'panel may have zero within-env variance in `x` '
            f'(every stratum has exactly one observation, or x '
            f'is constant within stratum). Use a non-stratified '
            f'analysis when x is purely env-level.',
        )

    # Compute residuals from unweighted fit then weight for σ²
    fitted: npt.NDArray[np.float64] = design @ coefs
    resid: npt.NDArray[np.float64] = y_arr - fitted
    n = len(y_arr)
    p = design.shape[1]
    df_resid = n - p
    if df_resid < 1:
        raise ValueError(
            f'Insufficient degrees of freedom (n={n}, p={p}); '
            f'need at least p+1 observations.',
        )
    weighted_ss = float(np.sum(w_arr * resid ** 2))
    sigma2 = weighted_ss / df_resid

    # `bread` = (XᵀWX)⁻¹: shared by both SE estimators
    xt_w_x = design.T @ (w_arr[:, None] * design)
    bread = np.linalg.inv(xt_w_x)

    if cluster_robust:
        # Liang-Zeger CR1 sandwich. Group rows by stratum_id (the
        # natural cluster for panel data); within each cluster
        # accumulate (Xᵀ W u)(Xᵀ W u)ᵀ. Final variance is
        # `bread @ meat @ bread` with the standard small-sample
        # correction `G/(G−1) · (n−1)/(n−p)`.
        cluster_to_idx: dict[object, list[int]] = {}
        for i, s in enumerate(strata):
            cluster_to_idx.setdefault(s, []).append(i)
        n_clusters = len(cluster_to_idx)
        if n_clusters < 2:
            raise ValueError(
                f'cluster_robust=True needs ≥2 clusters; got '
                f'{n_clusters}',
            )
        meat = np.zeros((p, p), dtype=np.float64)
        for idx_list in cluster_to_idx.values():
            idx = np.asarray(idx_list, dtype=np.int64)
            x_g = design[idx]
            w_g = w_arr[idx]
            u_g = resid[idx]
            # Score contribution for the cluster (length p):
            #   sum_i (X_i · w_i · u_i)
            score_g = (x_g.T * w_g) @ u_g
            meat += np.outer(score_g, score_g)
        # CR1 small-sample correction
        correction = (n_clusters / (n_clusters - 1)) \
            * ((n - 1) / df_resid) if n_clusters > 1 else 1.0
        cov_beta = correction * bread @ meat @ bread
    else:
        # Conventional WLS: V = σ² · (XᵀWX)⁻¹
        cov_beta = sigma2 * bread
    se_beta: npt.NDArray[np.float64] = np.sqrt(np.diag(cov_beta))

    # t-stats and p-values
    t_stats = coefs / se_beta
    p_values: npt.NDArray[np.float64] = 2.0 * (
        1.0 - ss.t.cdf(np.abs(t_stats), df=df_resid)
    )
    z_crit = float(ss.t.ppf(1.0 - alpha / 2.0, df=df_resid))

    # Weighted R²: 1 − SS_resid_w / SS_total_w
    weighted_y_mean = float(np.sum(w_arr * y_arr) / np.sum(w_arr))
    ss_total = float(np.sum(w_arr * (y_arr - weighted_y_mean) ** 2))
    r2 = 1.0 - weighted_ss / ss_total if ss_total > 0 else float('nan')

    # Hausman: test β_b == β_w via Wald on the contrast.
    # Contrast vector c = [0, 1, -1]. Var(cβ) = c Cov(β) cᵀ
    contrast = np.array([0.0, 1.0, -1.0])
    delta = float(contrast @ coefs)
    var_delta = float(contrast @ cov_beta @ contrast)
    hausman_t = delta / np.sqrt(var_delta) if var_delta > 0 else 0.0
    hausman_p = float(2.0 * (1.0 - ss.t.cdf(abs(hausman_t), df=df_resid)))

    between = MundlakCoefficient(
        component='between',
        coefficient=float(coefs[1]),
        se=float(se_beta[1]),
        p_value=float(p_values[1]),
        ci_lo=float(coefs[1] - z_crit * se_beta[1]),
        ci_hi=float(coefs[1] + z_crit * se_beta[1]),
    )
    within = MundlakCoefficient(
        component='within',
        coefficient=float(coefs[2]),
        se=float(se_beta[2]),
        p_value=float(p_values[2]),
        ci_lo=float(coefs[2] - z_crit * se_beta[2]),
        ci_hi=float(coefs[2] + z_crit * se_beta[2]),
    )
    return MundlakResult(
        n_strata=len(env_mean_x),
        n_obs=n,
        intercept=float(coefs[0]),
        between=between,
        within=within,
        hausman_p=hausman_p,
        r_squared=r2,
    )


__all__ = [
    'MundlakObservation',
    'MundlakCoefficient',
    'MundlakResult',
    'mundlak_decomposition',
]
