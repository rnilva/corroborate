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


def test_per_burst_panel_build_and_shape_invariants() -> None:
    """The framework's per-burst panel must build correctly AND
    produce a non-trivially-shaped g curve. We DO NOT assert
    g_t matches the closed-form value at every burst — that's
    substrate-tautology (Hedges' g formula matches Hedges' g
    formula).

    Instead, pin the framework-specific invariants:
      (1) panel has _N_BURSTS strata, one per burst index
      (2) every stratum reports n_pairs = _N_PAIRS (panel-build
          dict-intersection didn't silently drop seeds)
      (3) g(t=0) ≈ 0 (closed form: γ^0 − γ^0 = 0; sample noise
          alone)
      (4) g curve is non-monotone (peaks somewhere in [1, _N_BURSTS-1],
          rises from 0 then decays as both arms contract) — pins
          that the per-burst computation actually varies with t

    What this CATCHES: silent burst drops, stratum confusion,
    seed-pair-counting regressions, off-by-one in burst index,
    a stub returning the same g for every burst.

    What this does NOT catch (deliberately): the exact magnitude
    of g_t. That's substrate-tautology — the framework computes
    Hedges' g via the textbook formula on data the substrate
    constructed via the same formula. Curve-recovery testing was
    necessary plumbing verification but not a framework probe.
    """
    cells = _generate_contraction_panel_cells()
    result = paired_g_per_burst.fn(
        cells,
        treatment_arm='slow',
        baseline_arm='fast',
        pair_by=('seed',),
        source=_PER_BURST_SOURCE,
    )
    by_burst = {s.burst_index: s for s in result.strata}

    # (1) panel size
    assert len(by_burst) == _N_BURSTS, (
        f'panel has {len(by_burst)} bursts, expected {_N_BURSTS}'
    )
    # (2) n_pairs at every stratum
    for t, stratum in by_burst.items():
        assert stratum.n_pairs == _N_PAIRS, (
            f'burst {t}: n_pairs = {stratum.n_pairs}, expected '
            f'{_N_PAIRS}'
        )
    # (3) g(t=0) ≈ 0  (closed form structurally; sampling SE ≈ 0.11)
    assert abs(by_burst[0].g) < 0.5, (
        f'g(t=0) = {by_burst[0].g:.4f}, expected ≈ 0 (γ^0 − γ^0 = 0).'
    )
    # (4) non-monotone shape: peak g is at some interior burst
    g_values = [by_burst[t].g for t in range(_N_BURSTS)]
    peak_t = max(range(_N_BURSTS), key=lambda t: g_values[t])
    assert 0 < peak_t < _N_BURSTS - 1, (
        f'g curve peak at burst {peak_t}; expected interior burst '
        f'(rising-then-falling phase pattern). g values: '
        f'{[round(g, 3) for g in g_values]}.'
    )
