"""Framework-as-instrument: `paired_link_per_burst` +
`phase_link_consistency` recover the closed-form mech→outcome
link under Banach γ-contraction.

When two arms contract at different rates (γ_fast, γ_slow), both
the mech proxy (`jensen_gap_t`) and the outcome proxy (good-score
`-v_error_t`) decay at rate γ^t per arm. Per paired seed s with
shared initial scale x_0(s), per burst t:

    Δ_pred(s, t) = c_p · x_0(s) · (γ_f^t − γ_s^t) + ε_p_diff(s, t)
    Δ_targ(s, t) = −c_t · x_0(s) · (γ_f^t − γ_s^t) + ε_t_diff(s, t)

The two Δ-streams are NEGATIVELY correlated across seeds (high
x_0 → more negative Δ_pred AND more positive Δ_targ). The
framework's `paired_link_per_burst` negates predictor (so positive
r = "active link"), giving:

    r_reported(t) = D(t) · c_p · c_t · Var(x_0)
                  / √((c_p² · D(t) · Var(x_0) + 2σ_p²)
                     · (c_t² · D(t) · Var(x_0) + 2σ_t²))

where `D(t) = (γ_f^t − γ_s^t)²`.

For c_p = c_t = 1, σ_p = σ_t = σ, Var(x_0) = v:
    r_reported(t) = D(t) · v / (D(t) · v + 2σ²)

This is the textbook saturation curve: at t=0 (D=0), r=0; at
peak D, r → 1; at large t (D→0 as both γ^t → 0), r → 0. The
phase pattern that `phase_link_consistency` is designed to detect.

The framework's panel must:
1. Report per-burst r matching the closed-form curve within
   sampling SE (Pearson r SE ≈ (1 − r²)/√(n_pairs − 2)).
2. `phase_link_consistency(expected_sign=+1)` should match the
   structural active-burst fraction within the discrete-bin
   sampling distribution.
3. `phase_link_consistency(expected_sign=-1)` should match 0
   (no bursts have negative r structurally; sampling can't move
   any burst's r across zero by 4·SE worth).

A regression that mishandled the per-burst pairing, the per-burst
Pearson computation, or the predictor-negation convention would
breach the per-burst band.
"""
from __future__ import annotations

import math
import zlib

import numpy as np

from corroborate.analyses.paired_link_per_burst import (
    paired_link_per_burst,
    phase_link_consistency,
)
from corroborate.measurables.reductions import from_key, reduce_axis


def _det_seed(*parts: object) -> int:
    return zlib.adler32(repr(parts).encode()) & 0xFFFF_FFFF


# Sharp γ split → clear active-then-dormant phase pattern across bursts.
_GAMMA_FAST = 0.1     # treatment (good outcome, low mech)
_GAMMA_SLOW = 0.5     # baseline  (bad outcome, high mech)
_VAR_X0 = 0.25        # per-seed initial-scale variance
_SIGMA_NOISE = 0.1    # per-burst observation noise (predictor + target)
_C_P = 1.0            # predictor scale
_C_T = 1.0            # target scale
_N_PAIRS = 80
_N_BURSTS = 12

_PRED_KEY = 'jensen_gap_per_burst'
_TARG_KEY = 'v_score_per_burst'

_PRED_SOURCE = reduce_axis(from_key(_PRED_KEY), axis=-1, op='mean')
_TARG_SOURCE = reduce_axis(from_key(_TARG_KEY), axis=-1, op='mean')


def _expected_r(t: int) -> float:
    """Closed-form Pearson r at burst t (after framework's
    predictor-negation convention)."""
    diff = _GAMMA_FAST ** t - _GAMMA_SLOW ** t
    d = diff * diff
    if d == 0.0:
        return 0.0
    num = d * _C_P * _C_T * _VAR_X0
    den_p = _C_P ** 2 * d * _VAR_X0 + 2.0 * _SIGMA_NOISE ** 2
    den_t = _C_T ** 2 * d * _VAR_X0 + 2.0 * _SIGMA_NOISE ** 2
    return num / math.sqrt(den_p * den_t)


def _expected_r_se(r: float, n_pairs: int) -> float:
    """Pearson r SE under H_1 (Fisher z standard error scaled
    back to r-space): SE(r) ≈ (1 − r²) / √(n_pairs − 2)."""
    return (1.0 - r * r) / math.sqrt(n_pairs - 2)


def _generate_link_panel_cells() -> list[dict[str, object]]:
    """Per-cell per-burst predictor + target arrays under Banach
    contraction with shared per-seed x_0."""
    cells: list[dict[str, object]] = []
    env_name = 'contraction_link_demo'
    sigma_x = math.sqrt(_VAR_X0)
    for s in range(_N_PAIRS):
        rng_x0 = np.random.default_rng(seed=_det_seed('link_x0', s))
        x_0 = float(1.0 + sigma_x * rng_x0.standard_normal())

        rng_pf = np.random.default_rng(seed=_det_seed('link_p_fast', s))
        rng_ps = np.random.default_rng(seed=_det_seed('link_p_slow', s))
        rng_tf = np.random.default_rng(seed=_det_seed('link_t_fast', s))
        rng_ts = np.random.default_rng(seed=_det_seed('link_t_slow', s))

        pred_fast = np.array([
            [_C_P * x_0 * (_GAMMA_FAST ** t)
             + _SIGMA_NOISE * rng_pf.standard_normal()]
            for t in range(_N_BURSTS)
        ], dtype=np.float64)
        pred_slow = np.array([
            [_C_P * x_0 * (_GAMMA_SLOW ** t)
             + _SIGMA_NOISE * rng_ps.standard_normal()]
            for t in range(_N_BURSTS)
        ], dtype=np.float64)
        # Target: −c_t · x_0 · γ^t  (so higher target = better; treatment
        # has higher target since γ_fast^t decays to 0 faster).
        targ_fast = np.array([
            [-_C_T * x_0 * (_GAMMA_FAST ** t)
             + _SIGMA_NOISE * rng_tf.standard_normal()]
            for t in range(_N_BURSTS)
        ], dtype=np.float64)
        targ_slow = np.array([
            [-_C_T * x_0 * (_GAMMA_SLOW ** t)
             + _SIGMA_NOISE * rng_ts.standard_normal()]
            for t in range(_N_BURSTS)
        ], dtype=np.float64)
        cells.append({
            'arm_key': 'fast',
            'seed': s,
            'env_name': env_name,
            _PRED_KEY: pred_fast,
            _TARG_KEY: targ_fast,
        })
        cells.append({
            'arm_key': 'slow',
            'seed': s,
            'env_name': env_name,
            _PRED_KEY: pred_slow,
            _TARG_KEY: targ_slow,
        })
    return cells


def test_per_burst_link_recovers_closed_form_r_curve() -> None:
    """The framework's per-burst link panel must report `r(t)`
    matching the closed-form Pearson curve within 4·SE.

    SE(r) = (1 − r²) / √(n_pairs − 2). For n_pairs=80 and the
    closed-form r curve (0, 0.67, 0.42, 0.16, 0.045, …), SE
    ranges from ~0.06 (peak) to ~0.11 (off-peak). 4·SE ranges
    0.24 to 0.45.

    A regression that mishandled the per-burst pairing or the
    Pearson computation would breach at multiple bursts at once.
    """
    cells = _generate_link_panel_cells()
    result = paired_link_per_burst.fn(
        cells,
        treatment_arm='fast',
        baseline_arm='slow',
        target=_TARG_SOURCE,
        predictor=_PRED_SOURCE,
        pair_by=('seed',),
    )
    assert len(result.strata) == _N_BURSTS

    by_burst = {s.burst_index: s for s in result.strata}
    for t in range(_N_BURSTS):
        expected = _expected_r(t)
        # 4·SE on the closed-form Pearson r SE = (1−r²)/√(n−2).
        # At r=0 (t=0): SE = 1/√78 ≈ 0.113; at r=0.67 (peak):
        # SE ≈ 0.063. SE is bounded above by 1/√78 across all
        # bursts, so no floor needed — the formula's monotone-in-r
        # behaviour gives an honest 4·SE band at every burst.
        bound = 4.0 * _expected_r_se(expected, _N_PAIRS)
        actual = by_burst[t].r
        assert abs(actual - expected) < bound, (
            f'burst {t}: r = {actual:.4f}, closed-form = '
            f'{expected:.4f} (4·SE = {bound:.4f}). The framework '
            f's per-burst Pearson r must recover the contraction '
            f'link curve.'
        )


def test_phase_link_consistency_recovers_active_phase_fraction() -> None:
    """`phase_link_consistency(expected_sign=+1)` should match the
    structural active-burst count: bursts where the closed-form r
    is large enough that sample r will pass `p < 0.05` AND
    `r > 0`.

    Closed-form r per burst (γ_f=0.1, γ_s=0.5):
        t=0: r=0       (sign 0 — never counted)
        t=1: r=0.667   (p ≈ 1e-11; always counted)
        t=2: r=0.419   (p ≈ 1e-4;  always counted)
        t=3: r=0.161   (p ≈ 0.15;  borderline ~25% counted)
        t≥4: r < 0.05  (p > 0.5;   essentially never counted)

    Expected sample count ≈ 2.3 of 12 bursts → plc ≈ 0.19.
    Discrete-bin sampling fluctuation: 1/12 to 4/12 covers
    ~95% of the distribution.
    """
    cells = _generate_link_panel_cells()
    result = paired_link_per_burst.fn(
        cells,
        treatment_arm='fast',
        baseline_arm='slow',
        target=_TARG_SOURCE,
        predictor=_PRED_SOURCE,
        pair_by=('seed',),
    )
    plc = phase_link_consistency(result, expected_sign=+1, significance=0.05)
    # Structural: ~2/12 bursts active at significance=0.05.
    # Bound 1/12 ≤ plc ≤ 4/12 covers sampling fluctuation.
    assert 1.0 / _N_BURSTS <= plc <= 4.0 / _N_BURSTS, (
        f'phase_link_consistency = {plc:.4f}, expected in '
        f'[{1.0/_N_BURSTS:.4f}, {4.0/_N_BURSTS:.4f}] '
        f'(structural = ~2/12 = 0.167). The contraction link '
        f'is active in ~2 early bursts and dormant thereafter — '
        f'a flat plc reading would mean either the framework '
        f'didn t collapse phase structure or it conflated '
        f'active/dormant detection.'
    )


def test_phase_link_consistency_with_wrong_sign_returns_zero() -> None:
    """`expected_sign=-1` is the wrong-sign hypothesis. The
    closed-form r curve is non-negative everywhere (positive in
    early bursts, zero later); no burst structurally has r < 0.

    Sampling fluctuation could push individual-burst r slightly
    below zero in the late, near-zero phase, but only when the
    population r is already < 1·SE from zero. P(sample r < 0
    AND p < 0.05) at the late bursts where pop r ≈ 0 is roughly
    P(|sample r| > 2/√(n−2) | pop r ≈ 0)/2 ≈ 0.025 per burst.
    Over 12 bursts: expected count ≈ 0.3; sample count rarely
    exceeds 1.

    Bound: plc ≤ 1/12 (allow up to one burst sample fluctuation).
    """
    cells = _generate_link_panel_cells()
    result = paired_link_per_burst.fn(
        cells,
        treatment_arm='fast',
        baseline_arm='slow',
        target=_TARG_SOURCE,
        predictor=_PRED_SOURCE,
        pair_by=('seed',),
    )
    plc_wrong = phase_link_consistency(result, expected_sign=-1, significance=0.05)
    assert plc_wrong <= 1.0 / _N_BURSTS, (
        f'phase_link_consistency(expected_sign=-1) = '
        f'{plc_wrong:.4f}, expected ≤ 1/12 ≈ 0.083. The '
        f'contraction link has r ≥ 0 structurally; the '
        f'wrong-sign hypothesis should yield essentially '
        f'zero matching bursts.'
    )


def test_per_burst_link_n_pairs_propagates() -> None:
    """Each per-burst stratum should report `n_pairs == _N_PAIRS`.
    Pin the panel-build dict-intersection logic against a
    regression that silently dropped pairs per burst (which
    would inflate sampling SE and pass the closed-form curve
    test by being LOOSER, not by being correct).
    """
    cells = _generate_link_panel_cells()
    result = paired_link_per_burst.fn(
        cells,
        treatment_arm='fast',
        baseline_arm='slow',
        target=_TARG_SOURCE,
        predictor=_PRED_SOURCE,
        pair_by=('seed',),
    )
    for stratum in result.strata:
        assert stratum.n_pairs == _N_PAIRS, (
            f'burst {stratum.burst_index}: n_pairs = '
            f'{stratum.n_pairs}, expected {_N_PAIRS}'
        )
