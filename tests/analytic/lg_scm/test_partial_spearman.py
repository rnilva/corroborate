"""Closed-form assertions on `partial_spearman` (unified) over
the LG-SCM substrate.

The canonical mediation primitive — every mediation bridge in
`experiments/findings/` routes through this. Pre-consolidation
the legacy `stratified_partial_spearman` / `_multi` /
`stratified_spearman` carried analytic coverage in their now-
deleted test files; the unified primitive only had dispatch-
equivalence tests against `graph.discovery` until this file.

Closed-form Pearson r under the LG-SCM chain X → Z → Y:

    r(X, Y) = β_xz · β_zy · σ_x²
              / sqrt(σ_x² · (β_zy² · (β_xz²·σ_x² + σ_z²) + σ_y²))

For the implementation parameters here (β_xz ∈ {0.5, 0.7, 0.9},
β_zy=1.0, σ_x=0.5, σ_z=σ_y=0.4) the closed-form pearson r
per-env is in {0.40, 0.53, 0.62} and pools (Fisher-z) to ≈ 0.52.
Spearman ≈ Pearson under Gaussian-linear (the (6/π)·arcsin(r/2)
adjustment lands within 5% of Pearson r at these magnitudes).

ρ(X, Y | Z) = 0 in population (Z d-separates X from Y — fully
mediating chain). The closed-form partial-Spearman estimator
`(r_xy − r_xz·r_zy) / sqrt((1−r_xz²)(1−r_zy²))` is unbiased
asymptotically; finite-sample sampling variance dominates the
bound. Fisher-z pooled SE on partial Spearman across k=3 strata
at n=120 each:

    SE_z_marginal  ≈ 1/sqrt(k·(n−3)) ≈ 0.053
    SE_z_partial   ≈ SE_z_marginal · sqrt(1/((1−r_xz²)(1−r_zy²)))
                   ≈ 0.053 · sqrt(1/((1−0.6²)(1−0.7²))) ≈ 0.093

(at r_xz ≈ 0.6, r_zy ≈ 0.7 from the implementation params). Empirical
SD across 5 deterministic-seed replicates: 0.09 — matches the
closed form. The 0.30 bound covers 3σ of partial-Spearman
sampling variation around the d-separation null.

Cells are LG-SCM realisations across N=3 envs with different
per-env β_xz so per-stratum ρ varies; Fisher-z pooling
integrates them. The test exercises:

1. Marginal ρ(X_mean, Y_mean) recovers the closed-form pooled
   value within a Fisher-z sampling-distribution bound.
2. Single-Z partial ρ(X_mean, Y_mean | Z_mean) is statistically
   indistinguishable from zero (the d-separation prediction).
3. Multi-Z dispatch: adding a second conditioning variable that
   is a noise-augmented copy of Z keeps partial ρ ≈ 0; exercises
   the k≥2 dispatch into `partial_spearman_rho_multi`.
4. NaN-empty contract: empty cells → NaN ρ, n_strata=0.
"""
from __future__ import annotations

import math
import zlib
from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt
import pytest

from corroborate.analyses.spearman.partial_spearman import partial_spearman
from corroborate.measurables import Measurable
from corroborate.measurables.reductions import from_key, reduce_axis
from corroborate.data import cells_to_dataframe

from tests.analytic.lg_scm.composition import LinearGaussianSCM
from tests.analytic.lg_scm.runner import (
    PER_BURST_X_KEY,
    PER_BURST_Y_KEY,
    PER_BURST_Z_KEY,
    run_multi_env_paired_arms,
    run_paired_phased_arms,
)


# Implementation parameters chosen so the closed-form Pearson r per
# env is moderate (0.4-0.6). High r forces the partial-Spearman
# closed-form denominator near zero and amplifies finite-sample
# bias; moderate r keeps the estimator well-conditioned. σ_z and
# σ_y are intentionally not tiny — Z is only ~60% determined by
# X, leaving room for the d-separation prediction to be testable
# without collinearity-driven instability.
_MU_X = 1.0
_SIGMA_X = 0.5
_SIGMA_Z = 0.4
_BETA_ZY = 1.0
_SIGMA_Y = 0.4
_N_STEPS = 200
_N_SEEDS_PER_ENV = 120


# Three envs with distinct β_xz so each stratum carries its own
# pearson r and Fisher-z pooling integrates them.
_ENV_BETAS: Mapping[str, float] = {
    'env_a': 0.50,
    'env_b': 0.70,
    'env_c': 0.90,
}


def _det_seed(*parts: object) -> int:
    return zlib.adler32(repr(parts).encode()) & 0xFFFF_FFFF


def _scm(beta_xz: float) -> LinearGaussianSCM:
    return LinearGaussianSCM(
        mu_x=_MU_X,
        sigma_x=_SIGMA_X,
        beta_xz=beta_xz,
        sigma_z=_SIGMA_Z,
        beta_zy=_BETA_ZY,
        sigma_y=_SIGMA_Y,
        n_steps=_N_STEPS,
    )


def _expected_pearson_r(beta_xz: float) -> float:
    """Population r(X_mean, Y_mean) under the LG-SCM. Identical
    to r(X, Y) at the per-step level — the mean-of-trajectory
    transformation scales numerator and denominator by the same
    factor so r is invariant."""
    cov_xy = beta_xz * _BETA_ZY * _SIGMA_X ** 2
    var_x = _SIGMA_X ** 2
    var_y = (
        _BETA_ZY ** 2 * (beta_xz ** 2 * _SIGMA_X ** 2 + _SIGMA_Z ** 2)
        + _SIGMA_Y ** 2
    )
    return cov_xy / math.sqrt(var_x * var_y)


def _expected_pooled_rho() -> float:
    """Fisher-z-pooled ρ across the three envs at equal n.

    Each env contributes z_i = atanh(r_i); the pool averages the
    z's and tanh's back. Spearman ρ ≈ Pearson r on Gaussian-
    linear data, well within our 0.15 bound."""
    zs = [math.atanh(_expected_pearson_r(b)) for b in _ENV_BETAS.values()]
    z_avg = sum(zs) / len(zs)
    return math.tanh(z_avg)


def _build_cells() -> list[Mapping[str, object]]:
    """Multi-env paired sweep; keep one arm only — the
    intervention axis is degenerate here (both arms identical),
    we just need cells with x_mean/z_mean/y_mean columns. Both
    arms run on shared seeds → exact duplicates per (env, seed),
    so dropping treatment leaves exactly _N_SEEDS_PER_ENV cells
    per env without information loss."""
    envs = {
        env_name: (_scm(beta_xz), _scm(beta_xz))
        for env_name, beta_xz in _ENV_BETAS.items()
    }
    rows = run_multi_env_paired_arms(
        envs=envs,
        seeds=tuple(range(_N_SEEDS_PER_ENV)),
    )
    return [
        r.as_dict() for r in rows
        if r.arm_key == 'baseline'
    ]


def _add_independent_noise_column(
    cells: Sequence[Mapping[str, object]], *, sigma: float = 1.0,
) -> list[Mapping[str, object]]:
    """Augment cells with a fresh independent N(0, σ²) column.
    Statistically independent of every other cell variable, so
    partial ρ(X, Y | Z, noise_col) is identical to partial
    ρ(X, Y | Z) in population. Exercises the k≥2 dispatch
    without introducing collinearity that would destabilise the
    OLS-residual primitive at the n=30-per-stratum we use."""
    rng = np.random.default_rng(_det_seed('indep_noise', sigma))
    out: list[Mapping[str, object]] = []
    for c in cells:
        d = dict(c)
        d['noise_indep'] = float(rng.normal(0.0, sigma))
        out.append(d)
    return out


def test_marginal_rho_recovers_closed_form() -> None:
    cells = _build_cells()
    result = partial_spearman.fn(
        cells_to_dataframe(cells), x='x_mean', y='y_mean', conditioning=(),
        stratify_by='env_name',
    )
    expected = _expected_pooled_rho()
    # Fisher-z SE per stratum ≈ 1/sqrt(n-3) ≈ 0.092 at n=120.
    # Pooled across 3 strata ≈ 0.053. At ρ ≈ 0.52 the back-
    # transformed bound is (1-ρ²) ≈ 0.73, so SE on ρ ≈ 0.039.
    # 0.10 is a 2.5× slack absorbing Spearman-vs-Pearson
    # divergence at moderate r.
    assert abs(result.rho_pooled - expected) < 0.10, (
        f'rho_pooled={result.rho_pooled:.4f} '
        f'expected={expected:.4f}'
    )
    assert result.rho_pooled > 0.3, (
        f'rho_pooled={result.rho_pooled:.4f} should be substantively '
        'positive under the LG-SCM positive coupling'
    )
    assert result.p_value < 0.01, (
        f'p_value={result.p_value:.4g} should be tiny at ρ ≈ 0.52, '
        f'n={result.n_obs_total}'
    )
    assert result.n_strata == 3
    assert result.n_obs_total == _N_SEEDS_PER_ENV * 3


def test_partial_rho_conditional_on_mediator_is_null() -> None:
    """Z fully mediates X → Y; ρ(X, Y | Z) = 0 in population."""
    cells = _build_cells()
    result = partial_spearman.fn(
        cells_to_dataframe(cells), x='x_mean', y='y_mean', conditioning=('z_mean',),
        stratify_by='env_name',
    )
    # Fisher-z pooled SE on partial Spearman at k=3 strata, n=120
    # each, with r_xz ≈ 0.6 and r_zy ≈ 0.7: ≈ 0.093 (see module
    # docstring derivation). 0.30 is a 3σ window around the
    # d-separation null. The d-separation prediction is that
    # partial drops to ≈ 0 — a factor of ~5 attenuation relative
    # to the marginal ρ ≈ 0.52 — which this bound still detects
    # unambiguously while not flaking on the sampling
    # distribution.
    assert abs(result.rho_pooled) < 0.30, (
        f'partial rho_pooled={result.rho_pooled:.4f} should be ≈ 0 '
        'when Z fully mediates X→Y'
    )
    assert result.n_strata == 3


def test_multi_z_partial_rho_dispatch() -> None:
    """k≥2 conditioning dispatches into the multi-Z OLS-residual
    primitive. With Z (the LG-SCM mediator) and an independent
    N(0, 1) noise column as joint conditioners, partial ρ stays
    ≈ 0 — the noise column carries no information about Y, and Z
    still d-separates X from Y."""
    cells = _add_independent_noise_column(_build_cells())
    result = partial_spearman.fn(
        cells_to_dataframe(cells), x='x_mean', y='y_mean',
        conditioning=('z_mean', 'noise_indep'),
        stratify_by='env_name',
    )
    # Multi-Z form uses OLS residuals; closed-form null still
    # holds. SE on multi-Z partial Spearman inflates over the
    # single-Z form by sqrt((n-k_single)/(n-k_multi)) — at n=120
    # with k_single=1 vs k_multi=2 the inflation is
    # sqrt(119/118) ≈ 1.004, negligible. 0.30 bound carries the
    # same 3σ-of-sampling rationale as the single-Z test.
    assert abs(result.rho_pooled) < 0.30, (
        f'multi-Z partial rho_pooled={result.rho_pooled:.4f} '
        'should be ≈ 0 under conditional independence'
    )
    assert result.n_strata == 3


def test_empty_cells_returns_nan_zero_strata() -> None:
    result = partial_spearman.fn(
        cells_to_dataframe([]), x='x_mean', y='y_mean', conditioning=(),
        stratify_by='env_name',
    )
    assert math.isnan(result.rho_pooled)
    assert math.isnan(result.p_value)
    assert result.n_strata == 0
    assert result.n_obs_total == 0


# ============ Per-burst dispatch (Measurable inputs) ============
#
# `partial_spearman` dispatches into `_collect_per_burst` when ALL
# of `x`, `y`, conditioning are `Measurable[..., NDArray]` rather
# than `str`. Each (cell, burst) pair contributes ONE observation,
# so phased cells × n_bursts × n_seeds give the panel its
# observation count. The per-cell tests above only cover the
# `str`-input dispatch into `_collect_per_cell`; this block
# exercises the per-burst branch with the same closed-form
# structural targets.
#
# Observation count: 3 envs × 30 seeds × 4 bursts = 360, matching
# the per-cell suite's 3 × 120 = 360 → comparable power.
#
# Population correlations are scale-invariant under the LG-SCM
# (Pearson r doesn't depend on whether we average over n_steps),
# so `_expected_pearson_r` / `_expected_pooled_rho` from the
# per-cell block apply identically to the per-burst panel.

_N_SEEDS_PER_ENV_PHASED = 30
_N_BURSTS = 4


_PER_BURST_X_MEAN = reduce_axis(
    from_key(PER_BURST_X_KEY), axis=-1, op='mean',
)
_PER_BURST_Z_MEAN = reduce_axis(
    from_key(PER_BURST_Z_KEY), axis=-1, op='mean',
)
_PER_BURST_Y_MEAN = reduce_axis(
    from_key(PER_BURST_Y_KEY), axis=-1, op='mean',
)


def _build_phased_cells() -> list[Mapping[str, object]]:
    """Multi-env, multi-burst sweep. Both arms use the same SCM per
    env so the contrast is degenerate (we don't need it for the
    mediation question); the cells carry the per-burst trace
    arrays the Measurable inputs read. Drop one arm so each (env,
    seed, burst) contributes a single observation, matching the
    per-cell suite's single-arm panel pattern."""
    rows: list[Mapping[str, object]] = []
    for env_name, beta_xz in _ENV_BETAS.items():
        scms = tuple(_scm(beta_xz) for _ in range(_N_BURSTS))
        rows.extend(run_paired_phased_arms(
            treatments_per_burst=scms,
            baselines_per_burst=scms,
            seeds=tuple(range(_N_SEEDS_PER_ENV_PHASED)),
            env_name=env_name,
        ))
    return [r for r in rows if r.get('arm_key') == 'baseline']


def _add_per_burst_noise_column(
    cells: Sequence[Mapping[str, object]], *,
    key: str = 'noise_indep_per_burst',
    sigma: float = 1.0,
) -> tuple[
    list[Mapping[str, object]],
    Measurable[Mapping[str, object], npt.NDArray[np.floating]],
]:
    """Stamp each cell with an independent per-burst noise array
    `(n_bursts,)` at the given top-level key, and return the
    Measurable that reads it. Statistically independent of every
    other LG-SCM variable, so partial ρ(X, Y | Z, noise) ≈ partial
    ρ(X, Y | Z) in population — exercises the k≥2 dispatch into
    `partial_spearman_rho_multi` from the per-burst collection
    path without introducing structural collinearity."""
    rng = np.random.default_rng(_det_seed('per_burst_noise', sigma, key))
    out: list[Mapping[str, object]] = []
    for c in cells:
        d = dict(c)
        d[key] = [
            float(rng.normal(0.0, sigma)) for _ in range(_N_BURSTS)
        ]
        out.append(d)
    # The cell-level value at `key` is already a 1-D array of
    # shape (n_bursts,); `from_key` coerces it via np.asarray.
    # No further reduction needed.
    measurable = from_key(key)
    return out, measurable


def test_marginal_rho_recovers_closed_form_per_burst() -> None:
    """Per-burst marginal ρ(X_b, Y_b) matches the closed-form
    Pearson r — exercises `_collect_per_burst` end-to-end.

    Pearson r is scale-invariant under linear-averaging
    transformations, so `_expected_pooled_rho` from the per-cell
    block applies. Fisher-z SE at 360 observations across 3
    strata of 120 each → SE on pooled ρ ≈ 0.039 at ρ ≈ 0.52.
    Same 0.10 bound as the per-cell test (2.5× sampling slack)."""
    cells = _build_phased_cells()
    result = partial_spearman.fn(
        cells_to_dataframe(cells),
        x=_PER_BURST_X_MEAN, y=_PER_BURST_Y_MEAN, conditioning=(),
        stratify_by='env_name',
    )
    expected = _expected_pooled_rho()
    assert result.granularity == 'per_burst', (
        f'granularity={result.granularity!r} — Measurable inputs '
        'should dispatch through _collect_per_burst'
    )
    assert abs(result.rho_pooled - expected) < 0.10, (
        f'rho_pooled={result.rho_pooled:.4f} '
        f'expected={expected:.4f}'
    )
    assert result.rho_pooled > 0.3
    assert result.p_value < 0.01
    assert result.n_strata == 3
    assert result.n_obs_total == _N_SEEDS_PER_ENV_PHASED * _N_BURSTS * 3, (
        f'n_obs_total={result.n_obs_total} expected '
        f'{_N_SEEDS_PER_ENV_PHASED * _N_BURSTS * 3} — '
        '_collect_per_burst should emit one observation per (cell, burst)'
    )


def test_partial_rho_conditional_on_mediator_is_null_per_burst() -> None:
    """Z d-separates X from Y at each (cell, burst); per-burst
    partial ρ(X, Y | Z) ≈ 0 in population. Same 0.30 bound as the
    per-cell sibling — Fisher-z pooled SE on partial Spearman at
    k=3 strata of 120 obs each lands at ≈ 0.093."""
    cells = _build_phased_cells()
    result = partial_spearman.fn(
        cells_to_dataframe(cells),
        x=_PER_BURST_X_MEAN, y=_PER_BURST_Y_MEAN,
        conditioning=(_PER_BURST_Z_MEAN,),
        stratify_by='env_name',
    )
    assert result.granularity == 'per_burst'
    assert abs(result.rho_pooled) < 0.30, (
        f'per-burst partial rho_pooled={result.rho_pooled:.4f} '
        'should be ≈ 0 when Z mediates X → Y at every burst'
    )
    assert result.n_strata == 3


def test_multi_z_partial_rho_dispatch_per_burst() -> None:
    """Per-burst k≥2 dispatch via `_collect_per_burst` →
    `stratified_partial_spearman_rho_multi`. Add an independent
    per-burst noise array as a second conditioner; partial ρ
    stays ≈ 0 since the noise carries no information about Y and
    Z still d-separates X from Y."""
    cells, noise_measurable = _add_per_burst_noise_column(
        _build_phased_cells(),
    )
    result = partial_spearman.fn(
        cells_to_dataframe(cells),
        x=_PER_BURST_X_MEAN, y=_PER_BURST_Y_MEAN,
        conditioning=(_PER_BURST_Z_MEAN, noise_measurable),
        stratify_by='env_name',
    )
    assert result.granularity == 'per_burst'
    assert abs(result.rho_pooled) < 0.30, (
        f'multi-Z per-burst partial rho_pooled='
        f'{result.rho_pooled:.4f} should be ≈ 0 under conditional '
        'independence'
    )
    assert result.n_strata == 3


def test_mixed_str_and_measurable_inputs_raises() -> None:
    """Mixing `str` and `Measurable` across x/y/conditioning is a
    bridge-author bug — silent coercion would flatten or broadcast
    incorrectly. The granularity detector raises TypeError."""
    cells = _build_phased_cells()
    with pytest.raises(TypeError, match='must all be str.*OR all Measurable'):
        partial_spearman.fn(
            cells_to_dataframe(cells),
            x='x_mean', y=_PER_BURST_Y_MEAN, conditioning=(),
            stratify_by='env_name',
        )


def test_partial_spearman_dataframe_input_identical_to_cells() -> None:
    """Canonical-input invariant: Iterable[Mapping] and
    pl.DataFrame inputs produce the same `PartialSpearmanResult`."""
    import polars as pl

    cells = _build_cells()

    # A closure rather than a shared kwargs dict: the single
    # spelling of the arguments stays statically checked against
    # the analysis signature (a dict would erase them to
    # `dict[str, object]`).
    def run(
        cells_in: pl.DataFrame | list[Mapping[str, object]],
    ):
        return partial_spearman.fn(
            cells_to_dataframe(cells_in),
            x='x_mean',
            y='y_mean',
            conditioning=('z_mean',),
            stratify_by='env_name',
        )

    result_cells = run(cells)
    result_panel = run(pl.DataFrame(cells))
    assert result_panel.rho_pooled == result_cells.rho_pooled
    assert result_panel.p_value == result_cells.p_value
    assert result_panel.n_obs_total == result_cells.n_obs_total
    assert result_panel.n_strata == result_cells.n_strata
