"""Closed-form assertions on the Fisher-z + SE + p-value path in
`paired_link_per_burst._pearson_r_p_slope`.

The existing analytic test (`test_paired_link_per_burst.py`)
constructs cells where Δ_Y is exactly proportional to Δ_Z under
shared-seed cancellation, so r ≈ ±1 and the function short-circuits
at `if abs(r) >= 1.0: return r, 0.0, slope`. The Fisher-z, SE, and
p-value formulas are NEVER exercised — mutation testing surfaced
this as a coverage gap (mutations on `(1+r)/(1-r)`, `1/sqrt(n-3)`,
and `abs(z)/se` all survive).

This file fills the gap: synthetic cells where Δ_Y has a known
moderate correlation with Δ_Z (target |r| ≈ 0.577 from
`r = β_zy · sd(Δ_Z) / sqrt(β_zy²·sd(Δ_Z)² + 2·sd(ε)²)`), and the
framework's reported r, slope, and p-value match the closed
form within sampling-distribution-derived bounds.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from corroborate.analyses.link.paired_link_per_burst import (
    paired_link_per_burst,
)
from corroborate.measurables.reductions import from_key, reduce_axis
from corroborate.data import cells_to_dataframe


# Construction parameters chosen so closed-form |r| sits at ~0.577
# (1/√3) — well below the 1.0 short-circuit AND well above zero,
# putting the framework on the Fisher-z code path with a tight
# expected p-value (~6e-4 at n=30).
_BETA_ZY = 1.0
_SD_DELTA_Z = 1.0
_SD_EPSILON = 1.0
_N_PAIRS = 30
_N_BURSTS = 4
_TREATMENT = 'treatment'
_BASELINE = 'baseline'


def _expected_r() -> float:
    """Closed-form r(-Δ_Z, Δ_Y) under the construction below.

    Construction: z_t = 1 + z_perturb, z_b = 0 - z_perturb (anti-
    correlated → sd(Δ_Z) = 2·sd(z_perturb)). Y = β_zy·Z + ε with
    independent ε per arm (Var(Δε) = 2·sd(ε)²).

        Cov(Δ_Z, Δ_Y) = β_zy · Var(Δ_Z) = β_zy · (2·_SD_DELTA_Z)²
        Var(Δ_Z)      = (2·_SD_DELTA_Z)²
        Var(Δ_Y)      = β_zy² · Var(Δ_Z) + 2·sd(ε)²
        r(Δ_Z, Δ_Y)   = β_zy · (2·_SD_DELTA_Z) / sqrt(
                            β_zy²·(2·_SD_DELTA_Z)² + 2·sd(ε)²)

    Negation by paired_link's positive-when-active convention
    (`-Δ_predictor`) flips the sign.
    """
    delta_z_sd = 2.0 * _SD_DELTA_Z  # anti-correlated → 2× perturb sd
    num = _BETA_ZY * delta_z_sd
    den = math.sqrt(
        _BETA_ZY ** 2 * delta_z_sd ** 2 + 2.0 * _SD_EPSILON ** 2
    )
    return -num / den  # negative under paired_link's convention


def _expected_se(n_pairs: int) -> float:
    """Fisher-z SE = 1/sqrt(n - 3)."""
    return 1.0 / math.sqrt(n_pairs - 3)


def _expected_p_value(r: float, n_pairs: int) -> float:
    """Two-sided p-value via Fisher-z transform.
        z = 0.5 · ln((1 + r) / (1 - r))
        |z| / SE → standard normal cdf
    """
    z = 0.5 * math.log((1.0 + r) / (1.0 - r))
    se = _expected_se(n_pairs)
    from scipy.stats import norm
    return float(2.0 * (1.0 - norm.cdf(abs(z) / se)))


def _build_moderate_r_corpus() -> list[Mapping[str, object]]:
    """Construct per-cell `y_per_episode` and `z_per_episode`
    with structural Δ_Y = β_zy · Δ_Z + Δε, where Δε is an
    arm-independent noise component. The per-burst arrays are
    length-1 lists so `reduce_axis(_, axis=-1, op='mean')` yields
    the per-burst values directly.

    Two arms:
      treatment: per-burst Δ baseline shifted by a fixed
                 structural offset + per-cell Δ_Z noise + per-cell ε noise.
      baseline:  per-burst Δ baseline shifted by zero + same noise.
    """
    rng = np.random.default_rng(42)
    cells: list[Mapping[str, object]] = []
    for s in range(_N_PAIRS):
        # Per-(seed, burst) shared `z` realization for Δ.
        # Treatment z is shifted up by 1 vs baseline so mean Δ_Z > 0;
        # the shift is constant across (seed, burst), so per-pair
        # Δ_Z varies via the `z_perturb` random component below.
        z_perturb = rng.normal(0.0, _SD_DELTA_Z, size=_N_BURSTS)
        eps_t = rng.normal(0.0, _SD_EPSILON, size=_N_BURSTS)
        eps_b = rng.normal(0.0, _SD_EPSILON, size=_N_BURSTS)
        z_t = 1.0 + z_perturb  # treatment Z
        z_b = 0.0 - z_perturb  # baseline Z (anti-correlated to make Δ_Z vary)
        y_t = _BETA_ZY * z_t + eps_t
        y_b = _BETA_ZY * z_b + eps_b
        # 1-element-per-burst inner lists: reduce_axis(axis=-1, mean)
        # collapses to the scalar per burst.
        cells.append({
            'env_name': 'syn',
            'arm_key': _TREATMENT,
            'seed': s,
            'z_per_episode': [[float(v)] for v in z_t],
            'y_per_episode': [[float(v)] for v in y_t],
        })
        cells.append({
            'env_name': 'syn',
            'arm_key': _BASELINE,
            'seed': s,
            'z_per_episode': [[float(v)] for v in z_b],
            'y_per_episode': [[float(v)] for v in y_b],
        })
    return cells


def test_fisher_z_path_recovers_closed_form_r_and_p() -> None:
    """Moderate-r construction (closed-form |r| ≈ 0.577) puts the
    framework on the Fisher-z branch (`abs(r) < 1.0`). Each per-burst
    stratum should report r, slope, and p matching the closed form
    within sampling-distribution-derived bounds.

    The construction makes Δ_Z varies and Δ_Y = β_zy · Δ_Z + Δε
    with Δε an independent component, so:

        r(-Δ_Z, Δ_Y) = -β_zy · sd(Δ_Z) / sqrt(
                            β_zy²·sd(Δ_Z)² + 2·sd(ε)²)
        SE_Fisher    = 1/sqrt(n - 3)
        p            = 2·(1 - Φ(|z|/SE)),  z = 0.5·ln((1+r)/(1-r))

    A regression in any of the Fisher-z, SE, or p-value formulas
    (mutating `(1+r)/(1-r)` → `(1+r)*(1-r)`, `1/sqrt(n-3)` →
    `1*sqrt(n-3)` or `sqrt(n+3)`, `abs(z)/se` → `abs(z)*se`)
    would breach the bound here — exactly the mutmut-surfaced
    survivors this test is designed to kill.
    """
    cells = _build_moderate_r_corpus()
    # Per-burst-mean source: 1-element inner lists make this a no-op
    # reduction, so the per-burst array IS the cell's y_per_episode
    # / z_per_episode at face value.
    target = reduce_axis(from_key('y_per_episode'), axis=-1, op='mean')
    predictor = reduce_axis(from_key('z_per_episode'), axis=-1, op='mean')

    result = paired_link_per_burst.fn(
        cells_to_dataframe(cells),
        treatment_arm=_TREATMENT,
        baseline_arm=_BASELINE,
        target=target,
        predictor=predictor,
    )

    expected_r_pop = _expected_r()
    assert result.n_strata == _N_BURSTS

    for s in result.strata:
        # 1. Sample r is in the moderate range — NOT clamped to ±1.
        #    Closed-form is ~-0.82; allow ample sampling spread
        #    around it. Any slip-up that pushes |r| back to 1
        #    would short-circuit before Fisher-z and trip this.
        assert -0.99 < s.r < -0.20, (
            f'burst {s.burst_index}: r = {s.r:.4f} outside the '
            f'moderate-r band — short-circuit at |r|=1 would skip '
            f'Fisher-z entirely'
        )
        # 2. Sample r should be within sampling SE of the closed-
        #    form population r. SE(r) ≈ (1 - r²)/sqrt(n) per
        #    Fisher's normal-approximation; with closed-form
        #    r ~ -0.82 and n=30, SE ~ 0.06; 4·SE ~ 0.24 absorbs
        #    typical fluctuation.
        se_r = (1.0 - expected_r_pop ** 2) / math.sqrt(_N_PAIRS)
        assert abs(s.r - expected_r_pop) < 4.0 * se_r, (
            f'burst {s.burst_index}: r = {s.r:.4f}, '
            f'closed-form r = {expected_r_pop:.4f}, '
            f'4·SE = {4.0 * se_r:.4f}'
        )

        # 3. Sample p must lie within a sampling-distribution-derived
        #    band around the POPULATION p computed from the
        #    closed-form r_pop. The expected p is anchored against
        #    `_expected_r()` (population r derived from β_zy, σ_z,
        #    σ_ε), NOT against the framework's own returned `s.r`.
        #    A regression that fed sample r back through the formula
        #    on both sides of the assertion would be tautological.
        p_pop = _expected_p_value(expected_r_pop, _N_PAIRS)
        # log10(p) sensitivity to r: dz/dr = 1/(1-r²), so a 4·SE_r
        # drift of ±(4 · (1-r²)/√n) ≈ ±0.24 at r=-0.82, n=30 maps
        # to z drift ≈ ±0.74, |z|/SE drift ≈ ±3.97. Φ tail of
        # 3.97 maps to p drift of ~3 dex. Bound 4 dex absorbs this
        # AND fails any of the named formula mutations:
        #   (1+r)/(1-r) → (1+r)*(1-r) → z ≈ 0 → p ≈ 1 (huge dex shift)
        #   1/sqrt(n-3) → sqrt(n-3)   → SE off by factor n-3 = 27,
        #                                |z|/SE shrinks by 27× → p ≈ 1
        #   abs(z)/se → abs(z)*se     → product replaces ratio →
        #                                |z|·SE = 0.85·0.19 ≈ 0.16 →
        #                                p ≈ 0.87 (huge dex shift).
        if p_pop > 0 and s.p > 0:
            log_p_obs = math.log10(s.p)
            log_p_pop = math.log10(p_pop)
            assert abs(log_p_obs - log_p_pop) < 4.0, (
                f'burst {s.burst_index}: p = {s.p:.2e}, '
                f'closed-form p_pop = {p_pop:.2e} '
                f'(from r_pop = {expected_r_pop:.4f}, n = {_N_PAIRS}). '
                f'log10 diff = {abs(log_p_obs - log_p_pop):.4f}. '
                f'A formula mutation produces dex shifts ≫ 4; '
                f'sampling drift on r → p maps to ~3 dex; bound '
                f'absorbs sampling, rejects mutation.'
            )

        # 4. p must be significant (closed-form ~ 1e-7 at r_pop).
        assert s.p < 0.01, (
            f'burst {s.burst_index}: p = {s.p:.4f} not '
            f'significant; closed-form expects p well below 0.01 '
            f'at r ~ -0.82'
        )


def test_n_below_three_short_circuits_to_nan() -> None:
    """`_pearson_r_p_slope` checks `n < 3` and returns NaN. With
    only 2 paired seeds at a burst, the framework must report NaN
    rather than computing a degenerate Fisher-z (which would
    divide by sqrt(n-3) = sqrt(-1)). Catches the boundary mutation
    `n < 3` → `n <= 3`."""
    rng = np.random.default_rng(0)
    cells: list[Mapping[str, object]] = []
    for s in range(2):  # only 2 seeds → n=2 < 3
        z_perturb = rng.normal(0.0, 1.0, size=2)
        for arm, sign in ((_TREATMENT, +1.0), (_BASELINE, -1.0)):
            z = 0.5 * sign + z_perturb
            y = _BETA_ZY * z + rng.normal(0.0, _SD_EPSILON, size=2)
            cells.append({
                'env_name': 'syn', 'arm_key': arm, 'seed': s,
                'z_per_episode': [[float(v)] for v in z],
                'y_per_episode': [[float(v)] for v in y],
            })
    target = reduce_axis(from_key('y_per_episode'), axis=-1, op='mean')
    predictor = reduce_axis(from_key('z_per_episode'), axis=-1, op='mean')
    result = paired_link_per_burst.fn(
        cells_to_dataframe(cells),
        treatment_arm=_TREATMENT, baseline_arm=_BASELINE,
        target=target, predictor=predictor,
    )
    for s in result.strata:
        assert math.isnan(s.r), (
            f'burst {s.burst_index}: r = {s.r}; expected NaN at '
            f'n=2 (below the n<3 minimum for Fisher-z SE)'
        )
        assert math.isnan(s.p), (
            f'burst {s.burst_index}: p = {s.p}; expected NaN at n=2'
        )
