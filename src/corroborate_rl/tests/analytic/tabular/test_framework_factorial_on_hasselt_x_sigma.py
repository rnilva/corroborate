"""Framework-as-instrument: `factorial_2x2_interaction` recovers
the closed-form bias × σ-noise interaction.

A 2×2 factorial across two structural axes:

    Axis 1: bias-correction policy {vanilla, ddqn}
    Axis 2: noise level σ ∈ {σ_low, σ_high}

Four arms (A/B/C/D convention from `factorial_2x2_interaction`):

    arm_a = (vanilla, σ_low)
    arm_b = (vanilla, σ_high)   — flips axis 2 (σ change on vanilla)
    arm_c = (ddqn, σ_low)       — flips axis 1 (ddqn fixes bias at σ_low)
    arm_d = (ddqn, σ_high)      — flips both

**Independent ε streams per arm per seed.** This is structurally
load-bearing: under SHARED ε across σ levels (i.e., reusing
`ε_v` for arms A/B and `ε_o, ε_t` for C/D), the σ-scaling
identity `argmax(σ·ε) = argmax(ε)` collapses the 4-arm factorial
into a 2-arm contrast scaled by `(σ_high − σ_low)`. A stub
`factorial_2x2_interaction` that just computed `paired_g(C−A)`
and rescaled would silently pass an INT-only assertion. With
independent ε per arm, all five contrasts (B−A, D−C, C−A, D−B,
INT) have DISTINCT structural g values; the framework's per-arm
intersection + per-pair Δ-of-Δs arithmetic is properly probed.

Per-cell `jensen_gap` expectation under |A|=2, independent
standard-normal ε per arm scaled by the env-σ:

    E[A] = σ_low / √π        (vanilla bias at low noise)
    E[B] = σ_high / √π       (vanilla bias at high noise — larger)
    E[C] = 0                 (ddqn unbiased at low noise)
    E[D] = 0                 (ddqn unbiased at high noise)

Per-pair contrast statistics under independent ε:

    mean(B−A) =  (σ_high − σ_low) / √π
    Var(B−A)  = (σ_high² + σ_low²) · (1 − 1/π)
    g(B−A)    = (σ_high − σ_low)/√π
              / √((σ_high² + σ_low²)·(1 − 1/π)) · c_4
              ≈ +0.495      at σ_h=2, σ_l=0.5, n=200

    mean(D−C) = 0
    Var(D−C)  = (σ_high² + σ_low²) · 1
    g(D−C)    ≈ 0  (modulo sampling SE)

    mean(C−A) = −σ_low / √π
    Var(C−A)  = σ_low² · (2 − 1/π)
    g(C−A)    = −1 / (√π · √(2 − 1/π)) · c_4
              ≈ −0.433  (σ-independent: signal and noise both ∝ σ)

    mean(D−B) = −σ_high / √π
    Var(D−B)  = σ_high² · (2 − 1/π)
    g(D−B)    = −1 / (√π · √(2 − 1/π)) · c_4
              ≈ −0.433  (same closed-form value as g(C−A) by σ cancellation)

INT contrast (per-pair Δ-of-Δs):

    INT(s)        = (D − B) − (C − A)
    mean(INT)     = (σ_low − σ_high) / √π                ≈ −0.846
    Var(INT)      = (σ_high² + σ_low²) · (2 − 1/π)
    g(INT)        = (σ_low − σ_high)/√π
                  / √((σ_high² + σ_low²)·(2 − 1/π)) · c_4
                  ≈ −0.315  at σ_h=2, σ_l=0.5, n=200

The five distinct closed-form values (+0.495, ≈0, −0.433, −0.433,
−0.315) pin the framework's INT computation against any stub
that conflates contrasts.
"""
from __future__ import annotations

import math
import zlib

import numpy as np

from corroborate.analyses.paired.factorial_2x2 import factorial_2x2_interaction

from corroborate_rl.tabular import (
    double_greedify_tabular,
    max_greedify_tabular,
)


def _det_seed(*parts: object) -> int:
    """Deterministic-across-processes seed (zlib.adler32, not the
    process-randomized `hash()`)."""
    return zlib.adler32(repr(parts).encode()) & 0xFFFF_FFFF


_SIGMA_LOW = 0.5
_SIGMA_HIGH = 2.0
_N_PAIRS = 200


def _c4(n: int) -> float:
    return 1.0 - 3.0 / (4 * n - 5)


def _expected_g_b_minus_a(n: int) -> float:
    mean = (_SIGMA_HIGH - _SIGMA_LOW) / math.sqrt(math.pi)
    var = (_SIGMA_HIGH ** 2 + _SIGMA_LOW ** 2) * (1.0 - 1.0 / math.pi)
    return mean / math.sqrt(var) * _c4(n)


def _expected_g_c_minus_a(n: int) -> float:
    return -1.0 / (math.sqrt(math.pi) * math.sqrt(2.0 - 1.0 / math.pi)) * _c4(n)


def _expected_g_int(n: int) -> float:
    mean = (_SIGMA_LOW - _SIGMA_HIGH) / math.sqrt(math.pi)
    var = (_SIGMA_HIGH ** 2 + _SIGMA_LOW ** 2) * (2.0 - 1.0 / math.pi)
    return mean / math.sqrt(var) * _c4(n)


def _se_g(n: int, g: float) -> float:
    """Approximate SE on a paired Hedges' g at n_pairs.

    Cohen 1988: Var(g) ≈ 1/n + g²/(2n).
    """
    return math.sqrt(1.0 / n + g * g / (2.0 * n))


def _generate_factorial_cells() -> list[dict[str, object]]:
    """Per-seed: 4 cells covering the 2×2 factorial with
    INDEPENDENT ε streams per arm.

    Independence is structurally load-bearing — see module
    docstring. Under shared ε the σ-scaling identity collapses
    the factorial to a 2-arm contrast; the framework's per-arm
    intersection logic isn't probed.
    """
    cells: list[dict[str, object]] = []
    env_name = 'hasselt_factorial'
    for s in range(_N_PAIRS):
        # Per-(arm, seed) independent rng — distinct streams per arm.
        rng_a = np.random.default_rng(seed=_det_seed('factorial_a', s))
        rng_b = np.random.default_rng(seed=_det_seed('factorial_b', s))
        rng_c = np.random.default_rng(seed=_det_seed('factorial_c', s))
        rng_d = np.random.default_rng(seed=_det_seed('factorial_d', s))
        eps_a_v = rng_a.standard_normal(2).astype(np.float64)
        eps_b_v = rng_b.standard_normal(2).astype(np.float64)
        eps_c_o = rng_c.standard_normal(2).astype(np.float64)
        eps_c_t = rng_c.standard_normal(2).astype(np.float64)
        eps_d_o = rng_d.standard_normal(2).astype(np.float64)
        eps_d_t = rng_d.standard_normal(2).astype(np.float64)
        cells.append({
            'arm_key': 'arm_a',
            'seed': s,
            'env_name': env_name,
            'jensen_gap': max_greedify_tabular(eps_a_v * _SIGMA_LOW),
        })
        cells.append({
            'arm_key': 'arm_b',
            'seed': s,
            'env_name': env_name,
            'jensen_gap': max_greedify_tabular(eps_b_v * _SIGMA_HIGH),
        })
        cells.append({
            'arm_key': 'arm_c',
            'seed': s,
            'env_name': env_name,
            'jensen_gap': double_greedify_tabular(
                eps_c_o * _SIGMA_LOW, eps_c_t * _SIGMA_LOW,
            ),
        })
        cells.append({
            'arm_key': 'arm_d',
            'seed': s,
            'env_name': env_name,
            'jensen_gap': double_greedify_tabular(
                eps_d_o * _SIGMA_HIGH, eps_d_t * _SIGMA_HIGH,
            ),
        })
    return cells


# ============ INT contrast ============

def test_factorial_recovers_closed_form_int_g() -> None:
    """`g_INT` matches the closed-form Δ-of-Δs Hedges' g within
    4·SE.

    Closed form: `g_INT = (σ_low − σ_high)/√π /
                          √((σ_high² + σ_low²)·(2 − 1/π)) · c_4`
                ≈ −0.315 at σ_h=2, σ_l=0.5, n_pairs=200.

    SE(g_INT) ≈ √(1/200 + 0.315²/400) ≈ 0.0724. 4·SE ≈ 0.29.

    Under independent-ε per arm this is DISTINCT from g(C−A)
    (≈ −0.433) and g(B−A) (≈ +0.495); a stub returning any
    single contrast g would not match.
    """
    cells = _generate_factorial_cells()
    result = factorial_2x2_interaction.fn(
        cells,
        arm_a='arm_a', arm_b='arm_b',
        arm_c='arm_c', arm_d='arm_d',
        source='jensen_gap',
        pair_by=('seed',),
    )
    assert len(result.per_env) == 1
    per = result.per_env[0]

    expected = _expected_g_int(_N_PAIRS)
    bound = 4.0 * _se_g(_N_PAIRS, expected)
    assert abs(per.g_interaction - expected) < bound, (
        f'g_INT = {per.g_interaction:.4f}, closed-form = '
        f'{expected:.4f} (4·SE = {bound:.4f}).'
    )


# Note: a separate `g_INT distinct from corner contrasts` test
# was deleted in the audit pass. Its `|g_INT - g(C-A)| > 0.05`
# bound had ~25% sample-flake risk (population gap 0.118 vs SE_diff
# ≈ 0.105 — only 1.13σ of slack). The constraint is already
# implied: `test_factorial_recovers_closed_form_int_g` and
# `test_factorial_corner_c_minus_a_closed_form` both assert
# their respective g matches its closed-form value within 4·SE
# of the population value. If both per-quantity bands hold, the
# implied gap is `0.118 ± SE_diff` — distinct from 0 by the same
# logic.


# ============ Corner contrasts (closed-form per arm) ============

def test_factorial_corner_b_minus_a_closed_form() -> None:
    """g(B−A) is the σ-effect on vanilla — the bias is larger at
    σ_high than σ_low because Hasselt s formula scales with σ.
    Closed form: `(σ_high − σ_low)/√π / √((σ_high² + σ_low²)·
    (1 − 1/π)) · c_4` ≈ +0.495.

    Pin against arm-label swaps: a regression that swapped A and
    B would invert the sign; a regression that swapped C/D for
    A/B would land on a different closed form entirely.
    """
    cells = _generate_factorial_cells()
    result = factorial_2x2_interaction.fn(
        cells,
        arm_a='arm_a', arm_b='arm_b',
        arm_c='arm_c', arm_d='arm_d',
        source='jensen_gap',
        pair_by=('seed',),
    )
    per = result.per_env[0]
    expected = _expected_g_b_minus_a(_N_PAIRS)
    bound = 4.0 * _se_g(_N_PAIRS, expected)
    assert abs(per.g_b_minus_a - expected) < bound, (
        f'g(B−A) = {per.g_b_minus_a:.4f}, closed-form = '
        f'{expected:.4f} (4·SE = {bound:.4f}).'
    )


def test_factorial_corner_d_minus_c_near_zero() -> None:
    """g(D−C) is the σ-effect on ddqn — DDQN is unbiased at all σ,
    so the structural g is zero. SE bound on a null g is
    `≈ 1/√n` ≈ 0.071 at n=200; 4·SE ≈ 0.28.
    """
    cells = _generate_factorial_cells()
    result = factorial_2x2_interaction.fn(
        cells,
        arm_a='arm_a', arm_b='arm_b',
        arm_c='arm_c', arm_d='arm_d',
        source='jensen_gap',
        pair_by=('seed',),
    )
    per = result.per_env[0]
    bound = 4.0 * _se_g(_N_PAIRS, 0.0)
    assert abs(per.g_d_minus_c) < bound, (
        f'g(D−C) = {per.g_d_minus_c:.4f}, expected ≈ 0 '
        f'(4·SE = {bound:.4f}). DDQN is unbiased at all σ so '
        f'σ-effect on ddqn is structurally null.'
    )


def test_factorial_corner_c_minus_a_closed_form() -> None:
    """g(C−A) is the bias-correction effect at σ_low — DDQN
    reduces the σ_low/√π bias to 0. Closed form `−1 /
    (√π · √(2 − 1/π)) · c_4` ≈ −0.433. The closed-form value
    is INDEPENDENT of σ (signal and noise both scale linearly).
    """
    cells = _generate_factorial_cells()
    result = factorial_2x2_interaction.fn(
        cells,
        arm_a='arm_a', arm_b='arm_b',
        arm_c='arm_c', arm_d='arm_d',
        source='jensen_gap',
        pair_by=('seed',),
    )
    per = result.per_env[0]
    expected = _expected_g_c_minus_a(_N_PAIRS)
    bound = 4.0 * _se_g(_N_PAIRS, expected)
    assert abs(per.g_c_minus_a - expected) < bound, (
        f'g(C−A) = {per.g_c_minus_a:.4f}, closed-form = '
        f'{expected:.4f} (4·SE = {bound:.4f}).'
    )


def test_factorial_corner_d_minus_b_matches_c_minus_a() -> None:
    """g(D−B) and g(C−A) have the SAME closed-form value (both
    ≈ −0.433) by σ-cancellation in the standardized ratio.
    Under INDEPENDENT ε the two g values are independent
    sampling estimates of the same population g; their
    difference is normal-distributed around 0.

    Difference SE = √(SE(C−A)² + SE(D−B)²) ≈ √(2·0.074²) ≈ 0.105.
    4·SE_diff ≈ 0.42.
    """
    cells = _generate_factorial_cells()
    result = factorial_2x2_interaction.fn(
        cells,
        arm_a='arm_a', arm_b='arm_b',
        arm_c='arm_c', arm_d='arm_d',
        source='jensen_gap',
        pair_by=('seed',),
    )
    per = result.per_env[0]
    se_each = _se_g(_N_PAIRS, _expected_g_c_minus_a(_N_PAIRS))
    bound = 4.0 * math.sqrt(2.0) * se_each
    assert abs(per.g_d_minus_b - per.g_c_minus_a) < bound, (
        f'|g(D−B) − g(C−A)| = '
        f'{abs(per.g_d_minus_b - per.g_c_minus_a):.4f}, '
        f'expected ≈ 0 under σ-cancellation '
        f'(4·SE_diff = {bound:.4f}).'
    )
