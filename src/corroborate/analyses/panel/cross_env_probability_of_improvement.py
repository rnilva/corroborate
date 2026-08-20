"""`cross_env_probability_of_improvement` — per-stratum
Mann-Whitney `P(treatment > baseline)`, averaged across strata,
with two complementary inference modes (exact sign-permutation
+ stratified bootstrap CI).

Use case: cross-env consistency claims at small-to-moderate
`n_strata` (~10–50) where per-stratum outcome metrics have
incompatible scales (Acrobot returns ~−75, LunarLander ~−78–+77,
MinAtar ~0–30, success-rate envs ~0–1). Mann-Whitney
`P(X > Y)` is ordinal — scale-invariant by construction — and
saturated strata (both arms identical) contribute neutrally
(P ≈ 0.5) without requiring an explicit saturation guard.

The aggregation mirrors Agarwal et al. 2021 ("Deep RL at the
Edge of the Statistical Precipice", NeurIPS Outstanding Paper)
which made `P(X > Y)` with stratified-bootstrap CI the de facto
standard for cross-env claims in 2024–2026 deep RL submissions
(see `rliable`, https://github.com/google-research/rliable).
The framework departure: at `n_strata < 20` Agarwal's
percentile-bootstrap CI has nominal coverage error ~10–15% per
side (the empirical validation in the paper is at n=26–57). For
the small-stratum regime this primitive ALSO returns an exact
sign-permutation p-value on the (P_stratum − 0.5) deviations,
which uses per-stratum magnitudes (unlike a pure binomial sign-
test on direction) and is exact at any n.

**Use this primitive when**:
1. The claim is "treatment improves outcome consistently
   across heterogeneous envs" and per-stratum scales differ.
2. n_strata ≥ 5 (below that, no cross-env claim is rigorous).
3. Per-stratum n_seeds ≥ 5 per arm (Mann-Whitney precision).

**Use the existing `cross_env_consistency_binomial`** when the
claim is strict sign-only ("DDQN reduces metric at most/all
envs"), magnitudes are not load-bearing, or per-stratum Cohen's
d is the natural unit (e.g., effect-size meta-regression
companion). The two primitives test sibling questions: this
one uses magnitude information; the binomial sign-test
discards it.

**Distinguish from `meta_regression_unpaired_d`**: that asks
"does effect-size SCALE with env feature?" — a population
magnitude claim about heterogeneity. This primitive asks "is
the directional effect present across the panel?" — a
consistency claim about uniformity.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.stats import mannwhitneyu  # type: ignore[attr-defined]

import polars as pl

from corroborate._internals.polars import to_dicts
from corroborate.analyses._cell_value import resolve_value
from corroborate.bridge.analysis import analysis


@dataclass(frozen=True, slots=True)
class StratumProbabilityOfImprovement:
    """Per-stratum `P(treatment > baseline)` via Mann-Whitney U.

    `p_xy = U / (n_treatment * n_baseline)` where `U` is the
    Mann-Whitney statistic counting `(t_i > b_j)` pairs plus
    `0.5 ×` ties — equivalent to `P(X > Y) + 0.5 × P(X == Y)`.
    Bounded in [0, 1] by construction. `p_xy = 0.5` is the
    "no preference" point (random; or saturated; or identical
    distributions)."""
    stratum_id: tuple[object, ...]
    n_treatment: int
    n_baseline: int
    p_xy: float


@dataclass(frozen=True, slots=True)
class CrossEnvProbabilityOfImprovementResult:
    """Cross-stratum aggregate of per-stratum `P(X > Y)`.

    `p_xy_mean` is the unweighted mean of `per_stratum[i].p_xy`.
    `p_permutation` is the exact sign-permutation p-value for
    H_0: median(p_xy − 0.5) = 0 (the deviations are symmetric
    around 0). Computed by enumerating all `2 ** n_strata`
    sign-flip permutations when `2^n ≤ permutation_cap`;
    Monte-Carlo sampled with `n_permutation` draws otherwise.
    Exact at any n_strata; primary inference at n < 20 where
    bootstrap CI has known coverage issues.

    `ci_bootstrap_lo/hi` are 95% percentile-bootstrap interval
    endpoints from `n_bootstrap` stratified-resamples
    (resample seeds WITHIN each stratum's value list, recompute
    `P_stratum`, recompute `p_xy_mean`). Asymptotically valid;
    treat as descriptive at `n_strata < 20`.

    `n_strata` counts strata that survived `min_seeds_per_arm`;
    cells in skipped strata are NOT in `per_stratum`."""
    n_strata: int
    p_xy_mean: float
    p_permutation: float
    n_permutation_effective: int
    permutation_exact: bool
    ci_bootstrap_lo: float
    ci_bootstrap_hi: float
    n_bootstrap_effective: int
    per_stratum: tuple[StratumProbabilityOfImprovement, ...]
    measurable: str
    treatment_arm: str
    baseline_arm: str
    stratify_by: tuple[str, ...]


def _empty_result(
    *,
    source: str,
    treatment_arm: str,
    baseline_arm: str,
    stratify_by: tuple[str, ...],
) -> CrossEnvProbabilityOfImprovementResult:
    return CrossEnvProbabilityOfImprovementResult(
        n_strata=0,
        p_xy_mean=float('nan'),
        p_permutation=float('nan'),
        n_permutation_effective=0,
        permutation_exact=True,
        ci_bootstrap_lo=float('nan'),
        ci_bootstrap_hi=float('nan'),
        n_bootstrap_effective=0,
        per_stratum=(),
        measurable=source,
        treatment_arm=treatment_arm,
        baseline_arm=baseline_arm,
        stratify_by=stratify_by,
    )


def _bootstrap_p_xy(
    t_vals: npt.NDArray[np.floating],
    b_vals: npt.NDArray[np.floating],
) -> float:
    """Direct count form: `(sum_ij I(t_i > b_j) + 0.5 ×
    I(t_i == b_j)) / (n_t × n_b)`. Equivalent to Mann-Whitney
    U / (n_t × n_b). Inlined for the bootstrap inner loop where
    repeated `scipy.stats.mannwhitneyu` calls dominate cost."""
    n_t = len(t_vals)
    n_b = len(b_vals)
    if n_t == 0 or n_b == 0:
        return float('nan')
    diff = t_vals[:, None] - b_vals[None, :]
    gt = (diff > 0).sum()
    eq = (diff == 0).sum()
    return float(gt + 0.5 * eq) / float(n_t * n_b)


@analysis
def cross_env_probability_of_improvement(
    cells: pl.DataFrame,
    *,
    source: str,
    treatment_arm: str,
    baseline_arm: str,
    stratify_by: tuple[str, ...] = ('env_name',),
    arm_field: str = 'arm_key',
    min_seeds_per_arm: int = 5,
    n_bootstrap: int = 2000,
    n_permutation: int = 10000,
    permutation_cap: int = 16384,
    rng_seed: int = 42,
) -> CrossEnvProbabilityOfImprovementResult:
    """Compute per-stratum `P(treatment > baseline)` via
    Mann-Whitney, then aggregate.

    `stratify_by`: tuple of cell-record keys defining strata.
    Default `('env_name',)`; common implementation choice is
    `('env_name', 'gamma')` for γ-conditional panels.

    `min_seeds_per_arm`: strata with fewer than this many cells
    in EITHER arm are skipped (no per-stratum P emitted).
    Default 5 matches typical RL-paper minimum.

    `n_permutation`: Monte-Carlo permutation count when exact
    enumeration is infeasible (`2 ** n_strata > permutation_cap`).
    Default 10000 → MC SE on the p-value is ≈ √(p(1-p)/10000),
    e.g. SE ≈ 0.005 at p=0.05.

    `permutation_cap` (default 16384 = 2^14): switch from exact
    enumeration to MC sampling above this many permutations.
    At `permutation_cap=16384`, exact enumeration covers
    `n_strata ≤ 14`; beyond that, MC sampling is used.

    Returns a frozen dataclass carrying point estimate +
    permutation p (exact at any n) + bootstrap CI (asymptotic,
    descriptive at small n)."""
    rows = to_dicts(cells)
    cells_list = list(rows)

    strata: dict[tuple[object, ...], dict[str, list[float]]] = {}
    for cell in cells_list:
        sid = tuple(cell.get(k) for k in stratify_by)
        arm = cell.get(arm_field)
        if not isinstance(arm, str):
            continue
        if arm != treatment_arm and arm != baseline_arm:
            continue
        try:
            val = resolve_value(cell, source)
        except KeyError:
            continue
        if not math.isfinite(val):
            continue
        bucket = strata.setdefault(sid, {treatment_arm: [], baseline_arm: []})
        bucket.setdefault(arm, []).append(val)

    sorted_sids = sorted(
        strata.keys(),
        key=lambda sid: tuple(repr(x) for x in sid),
    )

    per_stratum: list[StratumProbabilityOfImprovement] = []
    p_xy_values: list[float] = []
    t_arrays: list[npt.NDArray[np.floating]] = []
    b_arrays: list[npt.NDArray[np.floating]] = []
    for sid in sorted_sids:
        buckets = strata[sid]
        t_vals = buckets.get(treatment_arm, [])
        b_vals = buckets.get(baseline_arm, [])
        if len(t_vals) < min_seeds_per_arm or len(b_vals) < min_seeds_per_arm:
            continue
        t_arr = np.asarray(t_vals, dtype=np.float64)
        b_arr = np.asarray(b_vals, dtype=np.float64)
        u_stat, _ = mannwhitneyu(
            t_arr, b_arr, alternative='two-sided', use_continuity=False,
        )
        p_xy = float(u_stat) / (len(t_arr) * len(b_arr))
        per_stratum.append(StratumProbabilityOfImprovement(
            stratum_id=sid,
            n_treatment=len(t_arr),
            n_baseline=len(b_arr),
            p_xy=p_xy,
        ))
        p_xy_values.append(p_xy)
        t_arrays.append(t_arr)
        b_arrays.append(b_arr)

    n_strata = len(p_xy_values)
    if n_strata == 0:
        return _empty_result(
            source=source,
            treatment_arm=treatment_arm,
            baseline_arm=baseline_arm,
            stratify_by=stratify_by,
        )

    arr = np.asarray(p_xy_values, dtype=np.float64)
    p_xy_mean = float(arr.mean())

    deviations = arr - 0.5
    observed_dev = float(deviations.sum())

    if 2 ** n_strata <= permutation_cap:
        permutation_exact = True
        n_perm_eff = 2 ** n_strata
        n_ge = 0
        for signs_tuple in itertools.product((-1.0, 1.0), repeat=n_strata):
            signs_arr = np.asarray(signs_tuple, dtype=np.float64)
            perm_dev = float(np.dot(deviations, signs_arr))
            if perm_dev >= observed_dev:
                n_ge += 1
        p_permutation = n_ge / n_perm_eff
    else:
        permutation_exact = False
        n_perm_eff = n_permutation
        rng = np.random.default_rng(rng_seed)
        signs = rng.choice(
            np.asarray([-1.0, 1.0], dtype=np.float64),
            size=(n_perm_eff, n_strata),
        )
        perm_devs = signs @ deviations
        n_ge = int((perm_devs >= observed_dev).sum())
        p_permutation = float(n_ge) / float(n_perm_eff)

    rng_boot = np.random.default_rng(rng_seed + 1)
    boot_means = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        sum_p = 0.0
        for t_arr, b_arr in zip(t_arrays, b_arrays):
            t_re = rng_boot.choice(t_arr, size=len(t_arr), replace=True)
            b_re = rng_boot.choice(b_arr, size=len(b_arr), replace=True)
            sum_p += _bootstrap_p_xy(t_re, b_re)
        boot_means[b] = sum_p / n_strata
    ci_lo = float(np.percentile(boot_means, 2.5))
    ci_hi = float(np.percentile(boot_means, 97.5))

    return CrossEnvProbabilityOfImprovementResult(
        n_strata=n_strata,
        p_xy_mean=p_xy_mean,
        p_permutation=p_permutation,
        n_permutation_effective=n_perm_eff,
        permutation_exact=permutation_exact,
        ci_bootstrap_lo=ci_lo,
        ci_bootstrap_hi=ci_hi,
        n_bootstrap_effective=n_bootstrap,
        per_stratum=tuple(per_stratum),
        measurable=source,
        treatment_arm=treatment_arm,
        baseline_arm=baseline_arm,
        stratify_by=stratify_by,
    )


__all__ = [
    'CrossEnvProbabilityOfImprovementResult',
    'StratumProbabilityOfImprovement',
    'cross_env_probability_of_improvement',
]
