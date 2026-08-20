"""Closed-form assertions on `paired_link_per_burst` over the LG-SCM.

Two tests:

1. **Active link recovers r ≈ -1 across all bursts.**
   Under shared-seed cancellation in `simulate_phased`, the per-
   (burst, seed) paired Δ on Z (predictor) and Y (target) are
   exactly proportional with slope `beta_zy`:

       Delta_Z(seed, b) = Delta_beta_xz(b) * X_avg(seed, b)
       Delta_Y(seed, b) = Delta_beta_xz(b) * beta_zy(b) * X_avg(seed, b)
                       = beta_zy(b) * Delta_Z(seed, b)

   `paired_link_per_burst` reports Pearson r between
   `(-Delta_Z, Delta_Y)` (its positive-when-active convention),
   which under `beta_zy > 0` equals exactly **-1** for every
   burst. The test asserts `r < -0.999` on each stratum and
   `slope ≈ -beta_zy` (the OLS slope of Δ_Y on -Δ_Z).

2. **Phase-flipping `beta_zy` switches link active → dormant.**
   Half the bursts have `beta_zy = 1.5` (active link), half have
   `beta_zy = 0.0` (Y decoupled from Z; Δ_Y = 0 every paired
   seed, link is structurally dormant).

   On dormant bursts:
       sd_d_target ≈ 0 → Pearson r returns NaN by design
       n_pairs == n_seeds (cells still match; the analysis just
       can't compute r when target has no variance)

   `phase_link_consistency(expected_sign=-1, significance=0.05)`
   should report ~0.5 on this panel — half active, half dormant.
   This is the canonical test that the framework reports dormancy
   honestly, rather than papering over it with a spurious r.
"""
from __future__ import annotations

import math

from corroborate.analyses.link.paired_link_per_burst import (
    PerBurstLinkStratum,
    paired_link_per_burst,
    phase_link_consistency,
)
from corroborate.measurables.reductions import from_key, reduce_axis
from corroborate.data import cells_to_dataframe

from tests.analytic.lg_scm.composition import LinearGaussianSCM
from tests.analytic.lg_scm.runner import (
    PER_BURST_Y_KEY,
    PER_BURST_Z_KEY,
    run_paired_phased_arms,
)


_MU_X = 1.0
_SIGMA_X = 0.5
_BETA_ZY_ACTIVE = 1.5
_SIGMA_Z = 0.1
_SIGMA_Y = 0.1
_N_STEPS_PER_BURST = 200
_N_PAIRS = 30


_PER_BURST_Y_MEAN = reduce_axis(
    from_key(PER_BURST_Y_KEY), axis=-1, op='mean',
)
_PER_BURST_Z_MEAN = reduce_axis(
    from_key(PER_BURST_Z_KEY), axis=-1, op='mean',
)


def _scm(beta_xz: float, *, beta_zy: float = _BETA_ZY_ACTIVE) -> LinearGaussianSCM:
    return LinearGaussianSCM(
        mu_x=_MU_X,
        sigma_x=_SIGMA_X,
        beta_xz=beta_xz,
        sigma_z=_SIGMA_Z,
        beta_zy=beta_zy,
        sigma_y=_SIGMA_Y,
        n_steps=_N_STEPS_PER_BURST,
    )


def _by_burst(
    strata: tuple[PerBurstLinkStratum, ...],
) -> dict[int, PerBurstLinkStratum]:
    return {s.burst_index: s for s in strata}


# ============ Active link across all bursts ============

def test_link_recovers_perfect_anticorrelation_under_shared_noise() -> None:
    """All bursts share `beta_zy = 1.5`. Under shared-seed
    cancellation, every paired (Δ_Z, Δ_Y) at burst b satisfies
    `Δ_Y = beta_zy * Δ_Z` exactly, so

        r(-Delta_Z, Delta_Y) = -1
        slope_OLS(Delta_Y on -Delta_Z) = -beta_zy = -1.5

    The framework's link primitive should recover both within
    floating-point noise (we allow `r < -0.999` and
    `|slope - (-beta_zy)| < 0.01`).

    A regression that mispairs cells across bursts, or that
    drops the negation in the predictor, would fail this test
    immediately.
    """
    n_bursts = 5
    treatments = tuple(_scm(0.8) for _ in range(n_bursts))
    baselines = tuple(_scm(0.3) for _ in range(n_bursts))
    cells = run_paired_phased_arms(
        treatments_per_burst=treatments,
        baselines_per_burst=baselines,
        seeds=range(_N_PAIRS),
    )

    result = paired_link_per_burst.fn(
        cells_to_dataframe(cells),
        treatment_arm='treatment',
        baseline_arm='baseline',
        target=_PER_BURST_Y_MEAN,
        predictor=_PER_BURST_Z_MEAN,
    )

    assert result.n_strata == n_bursts, (
        f'expected {n_bursts} strata; got {result.n_strata}'
    )
    by_b = _by_burst(result.strata)
    expected_slope = -_BETA_ZY_ACTIVE
    for b in range(n_bursts):
        s = by_b[b]
        assert s.n_pairs == _N_PAIRS, (
            f'burst {b}: n_pairs = {s.n_pairs}, expected {_N_PAIRS}'
        )
        assert s.r < -0.999, (
            f'burst {b}: r = {s.r:.6f}, expected closed-form '
            f'r = -1 (Delta_Y = beta_zy * Delta_Z exactly under '
            f'shared-seed cancellation; r(-Delta_Z, Delta_Y) = -1)'
        )
        assert abs(s.slope - expected_slope) < 0.01, (
            f'burst {b}: slope = {s.slope:.6f}, expected '
            f'{expected_slope} (the OLS slope of Δ_Y on -Δ_Z is '
            f'-beta_zy by construction)'
        )
        assert s.p < 0.001, (
            f'burst {b}: p = {s.p:.6f}, expected near-zero on a '
            f'closed-form r=-1 link'
        )


# ============ Phase-flipping link active <-> dormant ============

def test_phase_link_consistency_reports_dormant_bursts_honestly() -> None:
    """Bursts 0,1: `beta_zy = 1.5` (active). Bursts 2,3:
    `beta_zy = 0` (Y decoupled from Z, link dormant).

    Active bursts: Δ_Y / Δ_Z proportional, r = -1, p ≈ 0.
    Dormant bursts: Δ_Y = 0 every paired seed (Y = epsilon_y;
    shared seed → Δ_epsilon_y = 0). The framework's
    `_pearson_r_p_slope` returns NaN when y.std() == 0 — so
    dormant strata report `r = NaN`, `p = NaN`, `sd_d_target = 0`.

    `phase_link_consistency(expected_sign=-1, significance=0.05)`
    counts strata with `p < 0.05 AND r * expected_sign > 0`. NaN
    fails both predicates → consistency = 2/4 = 0.5.

    The test asserts the framework treats the dormant strata
    HONESTLY — no spurious r, no false-significant p. Treating
    a dormant link as 'null effect' would corrupt phase_link_
    consistency upward.
    """
    treatments = (
        _scm(0.8, beta_zy=_BETA_ZY_ACTIVE),
        _scm(0.8, beta_zy=_BETA_ZY_ACTIVE),
        _scm(0.8, beta_zy=0.0),
        _scm(0.8, beta_zy=0.0),
    )
    baselines = (
        _scm(0.3, beta_zy=_BETA_ZY_ACTIVE),
        _scm(0.3, beta_zy=_BETA_ZY_ACTIVE),
        _scm(0.3, beta_zy=0.0),
        _scm(0.3, beta_zy=0.0),
    )
    n_bursts = len(treatments)
    cells = run_paired_phased_arms(
        treatments_per_burst=treatments,
        baselines_per_burst=baselines,
        seeds=range(_N_PAIRS),
    )

    result = paired_link_per_burst.fn(
        cells_to_dataframe(cells),
        treatment_arm='treatment',
        baseline_arm='baseline',
        target=_PER_BURST_Y_MEAN,
        predictor=_PER_BURST_Z_MEAN,
    )
    assert result.n_strata == n_bursts
    by_b = _by_burst(result.strata)

    # Active bursts: r = -1, p ≈ 0, sd_d_target > 0
    for b in (0, 1):
        s = by_b[b]
        assert s.r < -0.999, (
            f'burst {b} (active): r = {s.r:.6f}, expected -1'
        )
        assert s.p < 0.001, (
            f'burst {b} (active): p = {s.p:.6f}, expected near-zero'
        )
        assert s.sd_d_target > 0.0, (
            f'burst {b} (active): sd_d_target = {s.sd_d_target}, '
            f'expected nonzero on a structurally-active link'
        )

    # Dormant bursts: Δ_Y exactly zero (epsilon_y cancels under
    # shared seeds, beta_zy=0 makes Y = epsilon_y), so r/p are NaN
    # and sd_d_target is exactly zero.
    for b in (2, 3):
        s = by_b[b]
        assert math.isnan(s.r), (
            f'burst {b} (dormant): r = {s.r}, expected NaN — '
            f'Y has no variance on a structurally-dormant link, '
            f'and the framework must report NaN rather than 0 or 1'
        )
        assert math.isnan(s.p), (
            f'burst {b} (dormant): p = {s.p}, expected NaN'
        )
        assert s.sd_d_target == 0.0, (
            f'burst {b} (dormant): sd_d_target = {s.sd_d_target}, '
            f'expected exactly zero — beta_zy=0 + shared seeds → '
            f'Δ_Y = 0 every paired seed'
        )

    # phase_link_consistency: only 2 of 4 bursts qualify (active +
    # significant). The closed-form value is 0.5.
    consistency = phase_link_consistency(
        result, significance=0.05, expected_sign=-1,
    )
    assert consistency == 0.5, (
        f'phase_link_consistency = {consistency}, expected 0.5 '
        f'(2 active bursts of 4); the framework should NOT treat '
        f'dormant strata as confirming the expected sign'
    )
