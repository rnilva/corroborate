"""Closed-form analytic assertions on `dynamic_partial_spearman`
over the LG-SCM substrate.

Builds a multi-burst, two-arm panel via `simulate_phased` (one
cell per (arm, seed); each cell carries an array of length
`n_bursts` for `y_mean_per_burst` and `z_mean_per_burst`). The
arm intervention is on `beta_xz` (treatment 0.6, baseline 0.4) so
the per-burst marginal ρ(arm, Ȳ_b) is non-zero with closed-form
prediction, and the per-burst partial ρ(arm, Ȳ_b | Z̄_b) is zero
in population (Z d-separates arm from Y in the LG-SCM chain).

Implementation parameters at the test point:
  mu_x = 1.0, sigma_x = 0.5, sigma_z = 0.4, sigma_y = 0.4,
  beta_zy = 1.0, n_steps = 20, n_seeds_per_arm = 80
  beta_xz_T = 0.6, beta_xz_B = 0.4
  Δ_μ = (0.6 − 0.4) · 1.0 · 1.0 = 0.2 (per-burst-mean Ȳ shift)

Bounds (CLAUDE.md §"Test principle"):

The choice of n_steps=20 (rather than 200) keeps Pearson r at a
non-saturating value (≈ 0.58) so Spearman ρ doesn't hit its
binary-vs-continuous ceiling. At saturating r the closed-form
first-order partial Spearman becomes unstable (its
(1−r_xz²)(1−r_zy²) denominator vanishes), and we'd have to
absorb large finite-sample drift into the partial bound. Keeping
r moderate sharpens both bounds.

Per-burst Var(Ȳ_b | arm) (sample-mean over n_steps=20 episodes):
  var = (beta_xz · beta_zy)² · σ_x² / n + β_zy² · σ_z² / n
        + σ_y² / n
  T: 0.6² · 0.0125 + 0.008 + 0.008 = 0.0045 + 0.016 = 0.0205
  B: 0.4² · 0.0125 + 0.008 + 0.008 = 0.002 + 0.016 = 0.018
  Mean within-arm var ≈ 0.0193.
  Between-arm var = 0.25 · 0.2² = 0.010.
  Overall Var(Ȳ) ≈ 0.0293 → SD(Ȳ) ≈ 0.171.
Point-biserial Pearson r = 0.2 · 0.5 / 0.171 ≈ 0.585.

Spearman ρ on this Gaussian-linear binary-vs-continuous shape
runs ~5% higher than Pearson r at moderate r (the
binary-rank-vs-continuous-rank correlation lies slightly above
Pearson when the continuous side has Gaussian dispersion).
Empirical Spearman per burst lands at ≈ 0.60-0.67 (n=160).

Fisher-z SE per burst at n=160: 1/sqrt(157) ≈ 0.080. Inverse-tanh
at ρ ≈ 0.62 gives back-transformed SE on ρ ≈ (1 − 0.62²) · 0.080
≈ 0.049. Pooled over 3 bursts: SE ≈ 0.028. 2.5σ bound on the
per-burst ρ: 0.12; per-burst pooled bound: 0.07. We use 0.15
per-burst and 0.10 pooled to absorb the Spearman-vs-Pearson
~5% divergence and any first-order partial drift from
non-bivariate-Gaussianity of the rank distributions.

Partial ρ population value is 0 (Z d-separates arm from Y in the
LG-SCM chain). At ρ_xz ≈ 0.72, ρ_yz ≈ 0.82, the closed-form
denominator (1 − ρ_xz²) · (1 − ρ_yz²) ≈ 0.158, so the
amplification factor on SE is √(1 / 0.158) ≈ 2.5. SE_partial at
n=160 ≈ 0.080 · 2.5 = 0.20. Pooled SE ≈ 0.12. 2.5σ bound on
pooled partial: 0.30. Per-burst 2.5σ: 0.50; we use 0.35 (looser
on per-burst since one burst can drift).

This is layer B per the test design: the layer-A sibling at
`tests/test_dynamic_mediation.py` covers (a) per-burst ρ matches
scipy exactly and (b) all four `TimeAggregationStatus` branches.
Layer B anchors the primitive against a closed-form population
value derived from the implementation parameters.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import polars as pl

from corroborate.analyses.dynamic_mediation import (
    TimeAggregationStatus,
    dynamic_partial_spearman,
)

from tests.analytic.lg_scm.composition import (
    LinearGaussianSCM,
    simulate_phased,
)


_MU_X = 1.0
_SIGMA_X = 0.5
_SIGMA_Z = 0.4
_BETA_ZY = 1.0
_SIGMA_Y = 0.4
_N_STEPS = 20
_N_BURSTS = 3
_N_SEEDS_PER_ARM = 80
_BETA_XZ_T = 0.6
_BETA_XZ_B = 0.4


def _scm(beta_xz: float) -> LinearGaussianSCM:
    return LinearGaussianSCM(
        mu_x=_MU_X, sigma_x=_SIGMA_X,
        beta_xz=beta_xz, sigma_z=_SIGMA_Z,
        beta_zy=_BETA_ZY, sigma_y=_SIGMA_Y,
        n_steps=_N_STEPS,
    )


def _build_panel() -> pl.DataFrame:
    """Two-arm phased panel. Each cell carries
    `y_mean_per_burst` and `z_mean_per_burst` as `List(Float64)`
    columns (length n_bursts) — the input shape
    `dynamic_partial_spearman` consumes. Seeds are shared across
    arms so the layer-A "noise cancellation" pattern would apply
    if needed; the dynamic primitive doesn't pair seeds, but
    deterministic seeds make the test reproducible."""
    scm_t = tuple(_scm(_BETA_XZ_T) for _ in range(_N_BURSTS))
    scm_b = tuple(_scm(_BETA_XZ_B) for _ in range(_N_BURSTS))
    cells: list[Mapping[str, object]] = []
    for s in range(_N_SEEDS_PER_ARM):
        obs_t = simulate_phased(scm_t, seed=s)
        cells.append({
            'env_name': 'lg_scm',
            'gamma': 0.99,
            'arm_key': 'treatment',
            'seed': s,
            'y_mean_per_burst': list(obs_t.y_mean_per_burst),
            'z_mean_per_burst': list(obs_t.z_mean_per_burst),
        })
        obs_b = simulate_phased(scm_b, seed=s)
        cells.append({
            'env_name': 'lg_scm',
            'gamma': 0.99,
            'arm_key': 'baseline',
            'seed': s,
            'y_mean_per_burst': list(obs_b.y_mean_per_burst),
            'z_mean_per_burst': list(obs_b.z_mean_per_burst),
        })
    return pl.DataFrame(cells)


def _population_point_biserial_r() -> float:
    """Closed-form point-biserial Pearson r for the substrate.

    Derivation in the module docstring. r ≈ Δ_μ · 0.5 / SD(Ȳ_overall).
    Spearman ≈ Pearson at the Gaussian-linear rank-biserial setup
    within ~1% at saturating r."""
    delta_mu = (_BETA_XZ_T - _BETA_XZ_B) * _BETA_ZY * _MU_X
    var_t = (
        (_BETA_XZ_T * _BETA_ZY) ** 2 * _SIGMA_X ** 2 / _N_STEPS
        + _BETA_ZY ** 2 * _SIGMA_Z ** 2 / _N_STEPS
        + _SIGMA_Y ** 2 / _N_STEPS
    )
    var_b = (
        (_BETA_XZ_B * _BETA_ZY) ** 2 * _SIGMA_X ** 2 / _N_STEPS
        + _BETA_ZY ** 2 * _SIGMA_Z ** 2 / _N_STEPS
        + _SIGMA_Y ** 2 / _N_STEPS
    )
    mean_within_var = 0.5 * (var_t + var_b)
    between_var = 0.25 * delta_mu ** 2
    overall_var = mean_within_var + between_var
    return float(delta_mu * 0.5 / math.sqrt(overall_var))


def _stratum_key(
    keys: Sequence[object],
) -> tuple[object, ...]:
    return tuple(keys)


def test_lg_scm_dynamic_marginal_recovers_population_r() -> None:
    """Per-burst marginal ρ(arm, Ȳ_b) recovers the closed-form
    point-biserial r ≈ 0.99 within a Fisher-z bound at every
    burst, and the pooled aggregate is sign-consistent →
    CONSISTENT_DIRECTION."""
    df = _build_panel()
    results = dynamic_partial_spearman.fn(
        df, arm_field='arm_key',
        mediator_per_burst='z_mean_per_burst',
        outcome_per_burst='y_mean_per_burst',
        stratify_by=('env_name', 'gamma'),
    )
    expected_r = _population_point_biserial_r()
    key = _stratum_key(('lg_scm', 0.99))
    assert key in results, f'expected stratum {key} in {list(results)}'
    result = results[key]
    assert result.aggregation_status is (
        TimeAggregationStatus.CONSISTENT_DIRECTION
    ), (
        f'expected CONSISTENT_DIRECTION; got '
        f'{result.aggregation_status!r}; rho_marginal='
        f'{result.rho_marginal}'
    )
    # Per-burst bound 0.15: 2.5σ on per-burst Spearman ρ at n=160
    # plus ~5% Spearman-vs-Pearson divergence at moderate r.
    for b, rho in enumerate(result.rho_marginal):
        assert abs(rho - expected_r) < 0.15, (
            f'burst {b}: rho={rho:.4f} expected={expected_r:.4f}'
        )
    # Pooled bound 0.10: 2.5σ on pooled Spearman ρ across 3 bursts
    # (SE ≈ 0.028) + the same Spearman-vs-Pearson divergence
    # accommodation. The pool is consistently slightly above
    # Pearson at this moderate r.
    assert abs(result.rho_marginal_pooled - expected_r) < 0.10, (
        f'rho_marginal_pooled={result.rho_marginal_pooled:.4f} '
        f'expected={expected_r:.4f}'
    )


def test_lg_scm_dynamic_partial_is_null_when_mediator_d_separates() -> None:
    """Z̄_b d-separates arm from Ȳ_b at every burst → partial ρ ≈
    0 in population. Bound: 2.5σ ≈ 0.16 (per derivation in module
    docstring); we use 0.30 to absorb closed-form-partial-Spearman
    boundary noise at the saturating-marginal end."""
    df = _build_panel()
    results = dynamic_partial_spearman.fn(
        df, arm_field='arm_key',
        mediator_per_burst='z_mean_per_burst',
        outcome_per_burst='y_mean_per_burst',
        stratify_by=('env_name', 'gamma'),
    )
    key = _stratum_key(('lg_scm', 0.99))
    result = results[key]
    # Per-burst bound 0.35: 2.5σ on partial Spearman at n=160 with
    # the closed-form (1−ρ_xz²)(1−ρ_yz²) denominator inflation
    # factor of ≈ 2.5 at the substrate's r_xz ≈ 0.72 / r_yz ≈ 0.82.
    for b, rho in enumerate(result.rho_partial):
        assert abs(rho) < 0.35, (
            f'burst {b}: partial rho={rho:.4f} should be ≈ 0 when Z '
            f'd-separates arm from Y in the LG-SCM chain'
        )
    # Pooled bound 0.30: 2.5σ on pooled partial Spearman across 3
    # bursts (SE ≈ 0.12 from the inflation-corrected per-burst SE
    # of 0.20).
    assert abs(result.rho_partial_pooled) < 0.30, (
        f'rho_partial_pooled={result.rho_partial_pooled:.4f} '
        f'should be ≈ 0 under d-separation'
    )


def test_lg_scm_per_burst_n_correct() -> None:
    """Sanity: every burst sees `2 * _N_SEEDS_PER_ARM` cells in
    the constructed panel."""
    df = _build_panel()
    results = dynamic_partial_spearman.fn(
        df, arm_field='arm_key',
        mediator_per_burst='z_mean_per_burst',
        outcome_per_burst='y_mean_per_burst',
        stratify_by=('env_name', 'gamma'),
    )
    key = _stratum_key(('lg_scm', 0.99))
    result = results[key]
    assert result.n_bursts == _N_BURSTS
    assert all(n == 2 * _N_SEEDS_PER_ARM for n in result.n_per_burst), (
        f'n_per_burst={result.n_per_burst}'
    )


# Sanity-check the closed-form derivation against numpy on an
# independent realisation, so a substrate-parameter change that
# breaks the docstring bound surfaces immediately.
def test_closed_form_r_matches_empirical_within_bound() -> None:
    """The closed-form `_population_point_biserial_r` value MUST
    match an empirical estimate from the panel via numpy/scipy
    within a sampling-distribution bound. Detects substrate-
    parameter drift before the framework assertions fire."""
    from scipy.stats import pearsonr
    df = _build_panel()
    arms = df.get_column('arm_key').to_list()
    y_lists = df.get_column('y_mean_per_burst').to_list()
    # Per-burst empirical Pearson r averaged across bursts.
    code = {'baseline': 0.0, 'treatment': 1.0}
    arm_codes = np.asarray(
        [code[a] for a in arms if isinstance(a, str)], dtype=np.float64,
    )
    rs: list[float] = []
    for b in range(_N_BURSTS):
        ys = np.asarray(
            [y[b] for y in y_lists], dtype=np.float64,
        )
        r, _ = pearsonr(arm_codes, ys)
        rs.append(float(r))
    empirical = float(np.mean(rs))
    expected = _population_point_biserial_r()
    # 2.5σ bound: per-burst Pearson r at n=160, ρ ≈ 0.58 has
    # SE ≈ (1-ρ²)/sqrt(n-1) ≈ 0.053. Across 3 bursts mean: SE ≈ 0.031.
    # 0.10 covers 2.5σ + Pearson-vs-population-r sampling drift.
    # This test exists to fire when implementation parameters drift,
    # not to test the framework itself.
    assert abs(empirical - expected) < 0.10, (
        f'empirical r={empirical:.4f} closed-form={expected:.4f} — '
        f'substrate parameters changed?'
    )
