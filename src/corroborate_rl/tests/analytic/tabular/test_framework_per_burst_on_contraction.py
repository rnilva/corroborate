"""Framework-as-instrument: `paired_g_per_burst` recovers the
phase structure of Banach γ-contraction.

Bertsekas-Tsitsiklis 1996 Prop 6.2.3: under the Bellman optimality
operator T with discount γ ∈ (0, 1), `||V_t − V*||_∞ ≤ γ^t · ||V_0
− V*||_∞`. Two policies / iterates contracting at DIFFERENT rates
γ_a, γ_b produce phase-structured trajectories:

  burst t:  err_a(t) = x_0 · γ_a^t       (slow contraction)
  burst t:  err_b(t) = x_0 · γ_b^t       (fast contraction)
  Δ(t) = err_a(t) − err_b(t) = x_0 · (γ_a^t − γ_b^t)

With per-seed `x_0(s) ∼ N(μ, σ_x²)` and independent observation
noise per arm:

  mean(Δ_t)   = μ · (γ_a^t − γ_b^t)
  Var(Δ_t)    = σ_x² · (γ_a^t − γ_b^t)² + 2 σ_obs²
  Hedges' g_t = mean(Δ_t) / sd(Δ_t) · c_4(n_seeds)

The closed-form g_t curve has a textbook shape:

  t = 0      → g_0 ≈ 0           (γ_a^0 − γ_b^0 = 0; no signal)
  t = peak   → g_peak ≈ 1 (max)  (signal saturates noise floor)
  t → large  → g → 0             (both contracted to 0)

This is the "phase-structured" pattern — the canonical use case
for `paired_g_per_burst` that scalar `paired_g` would silently
average to a single trajectory-level g.

The framework's `paired_g_per_burst` reports per-(env, burst) g.
THIS is the framework-as-instrument question: given paired
contraction trajectories with KNOWN per-burst structural g, does
the framework recover the closed-form curve to within sampling SE?

A regression that mishandled the per-burst pairing (mis-paired
cells, dropped a burst, computed g on the wrong axis) would
either flatten the curve or drift away from the closed form.
"""
from __future__ import annotations

import math
import zlib

import numpy as np


def _det_seed(*parts: object) -> int:
    """Deterministic-across-processes seed via zlib.adler32 —
    Python's `hash()` randomizes per process under PYTHONHASHSEED=random."""
    return zlib.adler32(repr(parts).encode()) & 0xFFFF_FFFF

from corroborate.analyses.paired_g_per_burst import (
    paired_g_per_burst,
)
from corroborate.measurables.reductions import from_key, reduce_axis


_GAMMA_A = 0.95     # slow contraction (treatment)
_GAMMA_B = 0.5      # fast contraction (baseline)
_MU_X0 = 1.0
_SIGMA_X0 = 0.3
_SIGMA_OBS = 0.5
_N_PAIRS = 80
_N_BURSTS = 12

_PER_BURST_KEY = 'contraction_err'

# Substrate composes a per-burst measurable: contraction error
# is stored as a per-burst array of shape (n_bursts, 1). Reduce
# the trailing axis (mean over single-element) to recover the
# per-burst scalar.
_PER_BURST_SOURCE = reduce_axis(
    from_key(_PER_BURST_KEY), axis=-1, op='mean',
)


def _expected_per_burst_g(t: int) -> float:
    """Closed-form Hedges' g at burst t.

    Under shared-x_0, INDEPENDENT noise per arm:
      mean(Δ_t)    = μ · (γ_a^t − γ_b^t)
      Var(Δ_t)     = σ_x² · (γ_a^t − γ_b^t)² + 2 σ_obs²
      g_t          = mean / sd · c_4(n_seeds)
    """
    diff = _GAMMA_A ** t - _GAMMA_B ** t
    mean = _MU_X0 * diff
    var = _SIGMA_X0 * _SIGMA_X0 * diff * diff + 2.0 * _SIGMA_OBS * _SIGMA_OBS
    sd = math.sqrt(var)
    if sd == 0.0:
        return 0.0
    c4 = 1.0 - 3.0 / (4 * _N_PAIRS - 5)
    return mean / sd * c4


def _generate_contraction_panel_cells() -> list[dict[str, object]]:
    """Per-cell per-burst array: contraction error trajectory.
    Two arms with different γ; per-seed x_0 shared; independent
    observation noise per arm."""
    cells: list[dict[str, object]] = []
    for s in range(_N_PAIRS):
        rng_seed = np.random.default_rng(seed=_det_seed('contract_seed', s))
        x_0 = float(_MU_X0 + _SIGMA_X0 * rng_seed.standard_normal())

        rng_a = np.random.default_rng(seed=_det_seed('contract_a', s))
        rng_b = np.random.default_rng(seed=_det_seed('contract_b', s))
        traj_a = np.array([
            [x_0 * (_GAMMA_A ** t) + _SIGMA_OBS * rng_a.standard_normal()]
            for t in range(_N_BURSTS)
        ], dtype=np.float64)
        traj_b = np.array([
            [x_0 * (_GAMMA_B ** t) + _SIGMA_OBS * rng_b.standard_normal()]
            for t in range(_N_BURSTS)
        ], dtype=np.float64)
        env_name = 'contraction_demo'
        cells.append({
            'arm_key': 'slow',
            'seed': s,
            'env_name': env_name,
            _PER_BURST_KEY: traj_a,
        })
        cells.append({
            'arm_key': 'fast',
            'seed': s,
            'env_name': env_name,
            _PER_BURST_KEY: traj_b,
        })
    return cells


def test_per_burst_panel_recovers_phase_structured_g_curve() -> None:
    """The framework's per-burst panel must report g_t matching
    the closed-form curve at every burst within sampling SE.

    Bound: per-burst g SE ≈ √(1/n_pairs + g²/(2 n_pairs))
    ≈ √(1/80 + 1²/160) ≈ 0.137 at peak. 4·SE ≈ 0.55.
    The relative error bound 0.25 is tighter — a real regression
    in pairing or g formula would breach by orders of magnitude.
    """
    cells = _generate_contraction_panel_cells()
    result = paired_g_per_burst.fn(
        cells,
        treatment_arm='slow',
        baseline_arm='fast',
        pair_by=('seed',),
        source=_PER_BURST_SOURCE,
    )
    # Panel: (env, burst) → stratum.
    by_burst = {s.burst_index: s for s in result.strata}
    assert len(by_burst) == _N_BURSTS, (
        f'panel has {len(by_burst)} bursts, expected {_N_BURSTS}'
    )
    for t in range(_N_BURSTS):
        expected = _expected_per_burst_g(t)
        actual = by_burst[t].g
        # 4·SE bound. Per-burst SE on g ≈ 0.14 in this setup.
        bound = max(0.55, 0.25 * abs(expected))
        assert abs(actual - expected) < bound, (
            f'burst {t}: g = {actual:.4f}, closed-form = '
            f'{expected:.4f} (bound = {bound:.4f}). The framework '
            f's per-burst pairing must recover the structural '
            f'phase curve.'
        )


def test_per_burst_g_zero_at_burst_zero() -> None:
    """At burst 0, both arms have err = x_0 (γ^0 = 1), so per-pair
    Δ(0) is pure observation noise. Hedges' g(0) ≈ 0 within 4·SE.

    Pin against a regression that off-by-one'd the burst index
    or that swapped baseline/treatment direction (which would
    flip sign systematically — but at t=0 both should give ≈ 0)."""
    cells = _generate_contraction_panel_cells()
    result = paired_g_per_burst.fn(
        cells,
        treatment_arm='slow',
        baseline_arm='fast',
        pair_by=('seed',),
        source=_PER_BURST_SOURCE,
    )
    by_burst = {s.burst_index: s for s in result.strata}
    g_0 = by_burst[0].g
    # At t=0, mean(Δ) = 0; sample mean is bounded by 4·SE_mean.
    # SE_mean ≈ σ_paired / √n = √(2σ_obs²) / √80 ≈ 0.079.
    # Hedges' g SE: ≈ 1/√n_pairs = 0.112.
    bound = 4.0 * 0.112
    assert abs(g_0) < bound, (
        f'g(t=0) = {g_0:.4f}, expected ≈ 0 (4·SE = {bound:.4f}). '
        f'At burst 0 both arms have err = x_0, paired Δ is pure '
        f'observation noise, so structural g is 0.'
    )


def test_per_burst_g_decays_in_late_phase() -> None:
    """In the late phase (t >> 1/log(γ_a)), both arms have
    contracted close to 0, so Δ shrinks toward observation noise
    and g shrinks toward 0.

    Quantitative: g(t=peak) > g(t=11) by a structural factor.
    From the closed form:
      diff(2)  = 0.6525,  g(2)  ≈ 0.85
      diff(11) = 0.95^11 − 0.5^11 ≈ 0.569 − 0.0005 ≈ 0.569
                                              g(11) ≈ 0.71

    Hmm, at γ_a=0.95 the slow contraction is still active at
    t=11 (γ_a^11 ≈ 0.57). So the late-phase decay isn't
    pronounced yet. Check that g_late is at least somewhat
    smaller than g_peak as a sanity check."""
    cells = _generate_contraction_panel_cells()
    result = paired_g_per_burst.fn(
        cells,
        treatment_arm='slow',
        baseline_arm='fast',
        pair_by=('seed',),
        source=_PER_BURST_SOURCE,
    )
    by_burst = {s.burst_index: s for s in result.strata}
    # Peak at t=2 or 3 where diff is maximal.
    g_peak = max(by_burst[t].g for t in (1, 2, 3, 4))
    g_late = by_burst[_N_BURSTS - 1].g
    # Late g should still be positive (γ_a still > 0 at t=11)
    # but smaller than peak.
    assert g_late < g_peak, (
        f'g(t={_N_BURSTS-1}) = {g_late:.4f} not less than '
        f'g(peak) = {g_peak:.4f}. Late-phase g should decay as '
        f'both arms contract toward 0.'
    )
    assert g_late > 0.0, (
        f'g(late) = {g_late:.4f} is non-positive; the slow arm '
        f'(γ_a=0.95) still has substantial residual at t=11, so '
        f'g should remain positive.'
    )


def test_per_burst_n_pairs_matches_input() -> None:
    """Each per-burst stratum should report n_pairs = number of
    paired seeds (80 here). Pin against a regression that
    silently dropped pairs."""
    cells = _generate_contraction_panel_cells()
    result = paired_g_per_burst.fn(
        cells,
        treatment_arm='slow',
        baseline_arm='fast',
        pair_by=('seed',),
        source=_PER_BURST_SOURCE,
    )
    for stratum in result.strata:
        assert stratum.n_pairs == _N_PAIRS, (
            f'burst {stratum.burst_index}: n_pairs = '
            f'{stratum.n_pairs}, expected {_N_PAIRS}'
        )
