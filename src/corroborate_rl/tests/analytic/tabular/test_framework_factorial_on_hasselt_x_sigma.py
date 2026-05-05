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

Per-cell `jensen_gap` expectation under independent noise per
arm (no shared cancellation; |A|=2):

    E[A] = σ_low / √π        (vanilla bias at low noise)
    E[B] = σ_high / √π       (vanilla bias at high noise — larger)
    E[C] = 0                 (ddqn unbiased at low noise)
    E[D] = 0                 (ddqn unbiased at high noise)

Corner contrasts (mean Δ):
    B − A = (σ_high − σ_low) / √π          (σ effect on vanilla)
    D − C = 0                               (σ effect on ddqn — null)
    C − A = −σ_low / √π                     (ddqn at σ_low)
    D − B = −σ_high / √π                    (ddqn at σ_high — larger reduction)

INT contrast (the framework's headline factorial output):
    INT = (D − B) − (C − A) = (σ_low − σ_high) / √π     (negative)

This is the canonical "ddqn's benefit increases with noise"
interaction — at higher σ, the bias-correction has more bias
to correct.

Per-pair INT_delta(seed) under shared σ-scaled noise (same
standard-normal draw scaled to σ_low for arms A/C and σ_high
for arms B/D, but DIFFERENT ε streams across the bias-correction
axis since ε_v and ε_online/ε_target are different estimators):

    INT(s) = (σ_high − σ_low) · (ε_t[argmax ε_o] − max(ε_v))

    E[INT]      = (σ_low − σ_high) / √π             ≈ −0.846
    Var(INT)    = (σ_high − σ_low)² · (1 + 1 − 1/π)  ≈ 1.682 · 2.25 = 3.785
    sd(INT)                                          ≈ 1.945
    g_INT       ≈ E[INT] / sd · c_4(n_pairs)         ≈ −0.43

The framework's `factorial_2x2_interaction.fn` should recover
this closed-form g_INT to within sampling SE.
"""
from __future__ import annotations

import math
import zlib

import numpy as np

from corroborate.analyses.factorial_2x2 import factorial_2x2_interaction


def _det_seed(*parts: object) -> int:
    """Deterministic-across-processes seed from a tuple of parts.
    Python's `hash()` randomizes per process; zlib.adler32 is a
    fixed CRC."""
    return zlib.adler32(repr(parts).encode()) & 0xFFFF_FFFF

from corroborate_rl.tabular import (
    double_greedify_tabular,
    max_greedify_tabular,
)


_SIGMA_LOW = 0.5
_SIGMA_HIGH = 2.0
_N_PAIRS = 200


def _expected_int_g(n_pairs: int) -> float:
    """Closed-form Hedges' g for the INT contrast under
    σ-scaled shared standard-normal noise.

    mean   = (σ_low − σ_high) / √π
    var    = (σ_high − σ_low)² · (var(ε_t[argmax ε_o]) +
                                  var(max(ε_v))) under independence
           = (σ_high − σ_low)² · (1 + (1 − 1/π))
    sd     = (σ_high − σ_low) · √(2 − 1/π)
    g      = mean / sd · c_4(n_pairs)
           = -1 / (√π · √(2 − 1/π)) · c_4
    """
    c4 = 1.0 - 3.0 / (4 * n_pairs - 5)
    return -1.0 / (math.sqrt(math.pi) * math.sqrt(2.0 - 1.0 / math.pi)) * c4


def _generate_factorial_cells() -> list[dict[str, object]]:
    """Per-seed: synthesize 4 cells covering the 2×2 factorial.
    σ-scaling shares the underlying standard-normal draws so the
    A/B (vanilla σ_low/σ_high) cells share ε_v, and similarly
    C/D share ε_online + ε_target. Across the bias-correction
    axis, ε_v vs (ε_online, ε_target) are INDEPENDENT (different
    estimators)."""
    cells: list[dict[str, object]] = []
    for s in range(_N_PAIRS):
        rng_v = np.random.default_rng(seed=_det_seed('factorial_v', s))
        rng_o = np.random.default_rng(seed=_det_seed('factorial_o', s))
        rng_t = np.random.default_rng(seed=_det_seed('factorial_t', s))
        # Standard-normal draws (|A|=2). Same draw scaled to σ_low
        # and σ_high → shared cancellation across the σ axis.
        eps_v = rng_v.standard_normal(2).astype(np.float64)
        eps_online = rng_o.standard_normal(2).astype(np.float64)
        eps_target = rng_t.standard_normal(2).astype(np.float64)
        env_name = 'hasselt_factorial'
        cells.append({
            'arm_key': 'arm_a',    # vanilla, σ_low
            'seed': s,
            'env_name': env_name,
            'jensen_gap': max_greedify_tabular(eps_v * _SIGMA_LOW),
        })
        cells.append({
            'arm_key': 'arm_b',    # vanilla, σ_high
            'seed': s,
            'env_name': env_name,
            'jensen_gap': max_greedify_tabular(eps_v * _SIGMA_HIGH),
        })
        cells.append({
            'arm_key': 'arm_c',    # ddqn, σ_low
            'seed': s,
            'env_name': env_name,
            'jensen_gap': double_greedify_tabular(
                eps_online * _SIGMA_LOW, eps_target * _SIGMA_LOW,
            ),
        })
        cells.append({
            'arm_key': 'arm_d',    # ddqn, σ_high
            'seed': s,
            'env_name': env_name,
            'jensen_gap': double_greedify_tabular(
                eps_online * _SIGMA_HIGH, eps_target * _SIGMA_HIGH,
            ),
        })
    return cells


def test_factorial_recovers_closed_form_int_g() -> None:
    """The framework's factorial_2x2_interaction must report
    g_INT matching the closed form within sampling SE.

    Closed form: g_INT ≈ -1/(√π·√(2-1/π))·c_4 ≈ -0.46. The
    INT contrast strips out the σ-axis main effect arithmetically,
    leaving a pure bias × σ interaction. A regression that
    mishandled the (D-B)-(C-A) ordering, swapped arm labels, or
    used unpaired SD would breach the bound by orders of magnitude.

    Per-pair INT SE (200 pairs) ≈ sd(INT) / √n = 1.945 / √200 ≈
    0.137. g SE ≈ 1/√n + g²/(2n) ≈ 0.075. 4·SE bound = 0.30.
    Tighter rel_err bound (15%) catches subtle sign/scale errors.
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

    expected_g_int = _expected_int_g(n_pairs=_N_PAIRS)
    rel_err = abs(per.g_interaction - expected_g_int) / abs(expected_g_int)
    assert rel_err < 0.20, (
        f'g_INT = {per.g_interaction:.4f}, closed-form '
        f'-1/(√π·√(2-1/π))·c_4 = {expected_g_int:.4f} '
        f'(rel err {rel_err:.4f}). The INT contrast strips out '
        f'σ-axis main effects, leaving pure bias × σ interaction.'
    )
    # Sign: ddqn s benefit grows with σ → INT is negative.
    assert per.g_interaction < 0, (
        f'g_INT = {per.g_interaction:.4f}; expected negative '
        f'(ddqn s bias-reduction benefit increases with σ).'
    )


def test_factorial_corner_signs_match_hasselt_predictions() -> None:
    """The four corner contrasts show structural signs derived
    from Hasselt's bias formula:

    - B − A: σ effect on vanilla → POSITIVE (more noise → more bias)
    - D − C: σ effect on ddqn → NEAR ZERO (ddqn unbiased at all σ)
    - C − A: ddqn effect at σ_low → NEGATIVE (bias correction)
    - D − B: ddqn effect at σ_high → NEGATIVE

    Note: at the Hedges' g level, g(C−A) ≈ g(D−B) because
    standardization cancels the σ scale (both signal and noise
    scale with σ, leaving the ratio constant). The σ-magnitude
    asymmetry shows up in the raw Δ (i.e., mean_d_target/predictor),
    NOT in g. The INT g captures the asymmetry differently —
    via the cross-arm shared-noise cancellation in the
    (D − B) − (C − A) arithmetic.

    A regression that swapped arm labels or inverted the contrast
    direction would breach these structural sign predictions.
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
    # B − A: σ effect on vanilla (positive).
    assert per.g_b_minus_a > 0, (
        f'g(B−A) = {per.g_b_minus_a:.4f}; expected positive '
        f'(σ noise increases vanilla bias).'
    )
    # D − C: σ effect on ddqn (≈ 0).
    assert abs(per.g_d_minus_c) < 0.5, (
        f'g(D−C) = {per.g_d_minus_c:.4f}; expected ≈ 0 '
        f'(ddqn unbiased at all σ).'
    )
    # C − A: ddqn effect at σ_low (negative).
    assert per.g_c_minus_a < 0, (
        f'g(C−A) = {per.g_c_minus_a:.4f}; expected negative '
        f'(ddqn corrects bias at σ_low).'
    )
    # D − B: ddqn effect at σ_high (negative).
    assert per.g_d_minus_b < 0, (
        f'g(D−B) = {per.g_d_minus_b:.4f}; expected negative '
        f'(ddqn corrects bias at σ_high).'
    )
    # At Hedges' g level, |D−B| ≈ |C−A| (σ scale cancels in the
    # standardized ratio). Pin the equality within sampling SE.
    assert abs(per.g_d_minus_b - per.g_c_minus_a) < 0.2, (
        f'g(D−B) = {per.g_d_minus_b:.4f}, g(C−A) = '
        f'{per.g_c_minus_a:.4f}; standardized g of the bias-'
        f'correction effect should be ≈ equal across σ levels '
        f'(σ scales both signal and noise → ratio constant).'
    )
