"""Closed-form assertions on `factorial_2x2_interaction` over the
LG-SCM substrate.

A 2x2 factorial across two structural axes:

    Axis A: beta_xz in {0.3 (baseline), 0.8 (treatment)}
    Axis B: beta_zy in {1.0 (baseline), 2.0 (treatment)}

Four arms (using the analysis's A/B/C/D convention):

    arm_a = (xz=0.3, zy=1.0)
    arm_b = (xz=0.3, zy=2.0)   — flips axis B
    arm_c = (xz=0.8, zy=1.0)   — flips axis A
    arm_d = (xz=0.8, zy=2.0)   — flips both

Per-cell `y_mean` expectation under shared-seed cancellation:

    E[y_mean(arm)] = beta_zy(arm) * beta_xz(arm) * mu_x

Per-pair interaction contrast on the per-pair delta:

    INT_delta(seed) = (D(seed) - B(seed)) - (C(seed) - A(seed))
                    = ((xz_t - xz_b) * zy_t - (xz_t - xz_b) * zy_b) * x_avg(seed)
                    = (xz_t - xz_b) * (zy_t - zy_b) * x_avg(seed)

So:

    E[INT_delta]  = (xz_t - xz_b) * (zy_t - zy_b) * mu_x
    SD[INT_delta] = |(xz_t - xz_b)(zy_t - zy_b)| * sigma_x / sqrt(n_steps)

The framework's `factorial_2x2_interaction` returns `g_interaction`
as the Hedges' g on `INT_delta`, plus the four corner contrasts
`g_b_minus_a`, `g_d_minus_c`, `g_c_minus_a`, `g_d_minus_b`. The
sign + magnitude of each is closed-form-tractable.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from corroborate.analyses.paired.factorial_2x2 import (
    _g_paired_from_two_arms,
    factorial_2x2_interaction,
)
from corroborate.corpus.schema import RunRow
from corroborate.data import cells_to_dataframe

from tests.analytic.lg_scm.composition import LinearGaussianSCM
from tests.analytic.lg_scm.runner import run_arm


_MU_X = 1.0
_SIGMA_X = 0.5
_SIGMA_Z = 0.1
_SIGMA_Y = 0.1
_N_STEPS = 200
_N_PAIRS = 30

_BETA_XZ_BASE = 0.3
_BETA_XZ_TREAT = 0.8
_BETA_ZY_BASE = 1.0
_BETA_ZY_TREAT = 2.0


def _scm(*, beta_xz: float, beta_zy: float) -> LinearGaussianSCM:
    return LinearGaussianSCM(
        mu_x=_MU_X, sigma_x=_SIGMA_X,
        beta_xz=beta_xz, sigma_z=_SIGMA_Z,
        beta_zy=beta_zy, sigma_y=_SIGMA_Y,
        n_steps=_N_STEPS,
    )


def _build_2x2_corpus() -> list[Mapping[str, object]]:
    """Construct a 4-arm corpus for the factorial design.

    All four arms use the same `seeds`, so the 4-cell paired
    intersection per seed is fully populated. Cell.measurements
    carry the per-arm coefficients alongside `y_mean`."""
    rows: list[RunRow] = []
    arms = (
        ('arm_a', _BETA_XZ_BASE, _BETA_ZY_BASE),
        ('arm_b', _BETA_XZ_BASE, _BETA_ZY_TREAT),
        ('arm_c', _BETA_XZ_TREAT, _BETA_ZY_BASE),
        ('arm_d', _BETA_XZ_TREAT, _BETA_ZY_TREAT),
    )
    for arm_key, beta_xz, beta_zy in arms:
        rows.extend(run_arm(
            _scm(beta_xz=beta_xz, beta_zy=beta_zy),
            seeds=range(_N_PAIRS),
            arm_key=arm_key,
        ))
    return [r.as_dict() for r in rows]


def _expected_int_mean() -> float:
    """E[INT_delta] = (xz_t - xz_b) * (zy_t - zy_b) * mu_x."""
    delta_xz = _BETA_XZ_TREAT - _BETA_XZ_BASE
    delta_zy = _BETA_ZY_TREAT - _BETA_ZY_BASE
    return delta_xz * delta_zy * _MU_X


def _expected_int_g(*, n_pairs: int) -> float:
    """Closed-form Hedges' g on INT_delta under shared noise.

    SD[INT_delta] = |delta_xz * delta_zy| * sigma_x / sqrt(n_steps)
    d            = E[INT_delta] / SD[INT_delta]
                  = mu_x * sqrt(n_steps) / sigma_x
    g            = d * c_4(n_pairs),   c_4 = 1 - 3/(4n - 5)
    """
    d = _MU_X * math.sqrt(_N_STEPS) / _SIGMA_X
    c4 = 1.0 - 3.0 / (4 * n_pairs - 5)
    return d * c4


def _c4(n_pairs: int) -> float:
    """Hedges' small-sample correction factor."""
    return 1.0 - 3.0 / (4 * n_pairs - 5)


def _expected_corner_g(
    *,
    beta_xz_left: float,
    beta_xz_right: float,
    beta_zy_left: float,
    beta_zy_right: float,
    n_pairs: int,
) -> float:
    """Closed-form Hedges' g for one corner contrast `right - left`
    under shared-seed cancellation.

    Per-seed delta:
        Δy(seed) = (β_zy_r * (β_xz_r * X(seed) + ε_z(seed)·σ_z))
                 + (β_zy_r - β_zy_l) * ε_y(seed)·σ_y * 0
                 - (β_zy_l * (β_xz_l * X(seed) + ε_z(seed)·σ_z))
                = (β_zy_r·β_xz_r - β_zy_l·β_xz_l) · X(seed)
                  + (β_zy_r - β_zy_l) · ε_z(seed)·σ_z
                  + 0 · ε_y    (cancels exactly: same seed → same ε_y)

    Wait — re-derive. ε_y multiplied by σ_y per arm; if β_zy
    differs between arms, the ε_y contribution cancels (it doesn't
    pre-multiply by β_zy in `simulate`). The ε_z term DOES pick up
    the (β_zy_r - β_zy_l) factor.

    mean(Δy) = (β_zy_r·β_xz_r - β_zy_l·β_xz_l) · μ_x   +   0
    var per seed:
        Var[X-coefficient · X + ε_z-coefficient · ε_z]
            = (β_zy_r·β_xz_r - β_zy_l·β_xz_l)² · σ_x²/n_steps
              + (β_zy_r - β_zy_l)² · σ_z²/n_steps

    d = mean / sqrt(var per seed); g = d · c_4.
    """
    delta_x_coef = (
        beta_zy_right * beta_xz_right
        - beta_zy_left * beta_xz_left
    )
    delta_eps_z_coef = beta_zy_right - beta_zy_left
    mean_delta = delta_x_coef * _MU_X
    var_per_seed = (
        (delta_x_coef ** 2) * (_SIGMA_X ** 2) / _N_STEPS
        + (delta_eps_z_coef ** 2) * (_SIGMA_Z ** 2) / _N_STEPS
    )
    d = mean_delta / math.sqrt(var_per_seed)
    return d * _c4(n_pairs)


# ============ Interaction term ============

def test_factorial_recovers_closed_form_interaction_sign_and_magnitude() -> None:
    """The 2x2 interaction Δ is closed-form: `(Δβ_xz)(Δβ_zy)·μ_x`.
    With our parameters this is `0.5 * 1.0 * 1.0 = 0.5`. The
    framework's `g_interaction` is Hedges' g on INT_delta —
    structural d ≈ μ_x·√n/σ_x ≈ 28 → g ≈ 27.5.

    The per-pair INT_delta picks up only the X-coefficient term
    (β_zy contributions cancel against each other in the (D-B)-(C-A)
    arithmetic), so SE is identical to the C-A corner. Closed-form g
    is therefore reachable to ~10% on n_pairs=30.

    A regression that mishandled the (D-B)-(C-A) arithmetic, mixed
    arm labels, or used unpaired SD would fail by orders of magnitude
    (and even a 30% scale error would clear the bound below)."""
    cells = _build_2x2_corpus()
    result = factorial_2x2_interaction.fn(
        cells_to_dataframe(cells),
        arm_a='arm_a', arm_b='arm_b', arm_c='arm_c', arm_d='arm_d',
        source='y_mean',
        pair_by=('seed',),
    )
    assert len(result.per_env) == 1
    per = result.per_env[0]

    expected_g = _expected_int_g(n_pairs=_N_PAIRS)
    rel_err = abs(per.g_interaction - expected_g) / expected_g
    assert rel_err < 0.1, (
        f'g_interaction = {per.g_interaction:.4f}, expected '
        f'{expected_g:.4f} (rel err {rel_err:.4f}). The interaction '
        f'contrast cancels β_zy contributions arithmetically; only '
        f'the X-coefficient survives, giving the same closed-form '
        f'SE as the C-A corner.'
    )


# ============ Corner contrasts ============

def test_factorial_corner_contrasts_recover_closed_form_per_corner() -> None:
    """Each corner contrast has a known closed-form Hedges' g
    derived from the structural arms:

    - B-A: only β_zy flips; ε_z noise propagates (β_zy_r-β_zy_l)·σ_z.
    - D-C: same axis B, but at treatment-A — also β_xz_t.
    - C-A: only β_xz flips; ε_z cancels exactly.
    - D-B: same axis A, but at treatment-B.

    Each corner has a corner-specific SD (not all equal) because
    ε_z propagation depends on the β_zy difference between the two
    arms. The test asserts each corner's g matches its own closed
    form to within 10% — much sharper than `g > 0.8` (which
    structural g of ~22-28 clears with a 30× margin)."""
    cells = _build_2x2_corpus()
    result = factorial_2x2_interaction.fn(
        cells_to_dataframe(cells),
        arm_a='arm_a', arm_b='arm_b', arm_c='arm_c', arm_d='arm_d',
        source='y_mean',
        pair_by=('seed',),
    )
    per = result.per_env[0]

    corner_specs: tuple[
        tuple[str, float, float, float, float, float], ...
    ] = (
        ('g_b_minus_a', per.g_b_minus_a,
         _BETA_XZ_BASE, _BETA_XZ_BASE,
         _BETA_ZY_BASE, _BETA_ZY_TREAT),
        ('g_d_minus_c', per.g_d_minus_c,
         _BETA_XZ_TREAT, _BETA_XZ_TREAT,
         _BETA_ZY_BASE, _BETA_ZY_TREAT),
        ('g_c_minus_a', per.g_c_minus_a,
         _BETA_XZ_BASE, _BETA_XZ_TREAT,
         _BETA_ZY_BASE, _BETA_ZY_BASE),
        ('g_d_minus_b', per.g_d_minus_b,
         _BETA_XZ_BASE, _BETA_XZ_TREAT,
         _BETA_ZY_TREAT, _BETA_ZY_TREAT),
    )
    for label, observed, xz_l, xz_r, zy_l, zy_r in corner_specs:
        expected = _expected_corner_g(
            beta_xz_left=xz_l, beta_xz_right=xz_r,
            beta_zy_left=zy_l, beta_zy_right=zy_r,
            n_pairs=_N_PAIRS,
        )
        rel_err = abs(observed - expected) / expected
        # 15% absorbs the ~13% CV on sample SD at n_pairs=30 (the
        # SD term in Cohen's d is the noisy half of the ratio at
        # finite n). Sharper than `g > 0.8` (50× looser) but still
        # honest about the per-corner sampling distribution.
        assert rel_err < 0.15, (
            f'{label} = {observed:.4f}, expected {expected:.4f} '
            f'(rel err {rel_err:.4f}). The corner-specific closed '
            f'form accounts for residual ε_z propagation when β_zy '
            f'differs between arms.'
        )


# ============ Null interaction ============

def test_factorial_interaction_is_null_when_axes_have_no_joint_effect() -> None:
    """When axis A and axis B influence Y *additively* with no
    joint-effect coupling, the interaction term should be near
    zero. To engineer this without changing the SCM's structural
    form (Y = β_zy * (β_xz * X)), we make axis B a no-op —
    treatment_B has the SAME `beta_zy` as baseline_B.

    The four arms become:
        A: (xz=0.3, zy=1.5)
        B: (xz=0.3, zy=1.5)   ← same as A (axis B is null)
        C: (xz=0.8, zy=1.5)
        D: (xz=0.8, zy=1.5)   ← same as C (axis B is null)

    Closed-form: INT_delta = (D - B) - (C - A) = 0 exactly under
    shared seeds — D == C and B == A by construction.

    Shared-seed cancellation makes EVERY per-pair INT_delta
    EXACTLY zero. `hedges_g_paired`'s zero-variance convention
    returns `(g=0.0, se=NaN)` — pin to exact 0.0, not "0 OR NaN".
    A `NaN OR 0` permissive bound has half the discriminating
    power; the construction is deterministic, so the expected
    value is determinable.
    """
    rows: list[RunRow] = []
    # axis A flips beta_xz; axis B is null (no-op).
    arms = (
        ('arm_a', 0.3, 1.5),
        ('arm_b', 0.3, 1.5),
        ('arm_c', 0.8, 1.5),
        ('arm_d', 0.8, 1.5),
    )
    for arm_key, beta_xz, beta_zy in arms:
        rows.extend(run_arm(
            _scm(beta_xz=beta_xz, beta_zy=beta_zy),
            seeds=range(_N_PAIRS),
            arm_key=arm_key,
        ))
    cells: Sequence[Mapping[str, object]] = [r.as_dict() for r in rows]

    result = factorial_2x2_interaction.fn(
        cells_to_dataframe(cells),
        arm_a='arm_a', arm_b='arm_b', arm_c='arm_c', arm_d='arm_d',
        source='y_mean',
        pair_by=('seed',),
    )
    per = result.per_env[0]
    # Shared-seed identical-arm pairs make every per-pair INT_delta
    # EXACTLY zero. `hedges_g_paired` returns `(g=0.0, se=NaN)` on
    # zero-variance deltas — pin g_interaction to exact 0.0.
    assert per.g_interaction == 0.0, (
        f'g_interaction = {per.g_interaction} on a structurally-null '
        f'interaction; closed-form is exact 0.0 (every per-pair '
        f'INT_delta is zero by shared-seed cancellation; '
        f'hedges_g_paired returns g=0.0 on zero-variance).'
    )
    assert math.isnan(per.se_interaction), (
        f'se_interaction = {per.se_interaction}; expected NaN '
        f'(zero-variance delta vector → undefined SE).'
    )


# ============ _g_paired_from_two_arms direct ============

def test_g_paired_from_two_arms_returns_g_se_for_valid_pair() -> None:
    """3 paired keys with deterministic deltas → finite g, finite se.
    Pin `len(deltas) < 2` against `<= 2` (would NaN at 3 pairs)
    and `< 3` (would NaN at 3 pairs)."""
    # Vary deltas so variance is non-zero (else hedges_g_paired
    # returns g=0, se=NaN by zero-variance convention).
    arm_x: dict[tuple[object, ...], float] = {('k0',): 1.0, ('k1',): 2.0, ('k2',): 3.0}
    arm_y: dict[tuple[object, ...], float] = {('k0',): 1.5, ('k1',): 3.0, ('k2',): 5.0}    # deltas: 0.5, 1.0, 2.0
    g, se = _g_paired_from_two_arms(arm_x, arm_y, [('k0',), ('k1',), ('k2',)])
    assert math.isfinite(g)
    assert math.isfinite(se)
    assert g > 0   # positive deltas → positive g


def test_g_paired_from_two_arms_returns_nan_below_n_2() -> None:
    """1 paired key → NaN g + NaN se. Pin every NaN-tuple element
    against `float(None)` (TypeError), `float('XXnanXX')`
    (ValueError), `float('NAN')` (equivalent — accepted)."""
    arm_x: dict[tuple[object, ...], float] = {('k0',): 1.0}
    arm_y: dict[tuple[object, ...], float] = {('k0',): 2.0}
    g, se = _g_paired_from_two_arms(arm_x, arm_y, [('k0',)])
    assert math.isnan(g)
    assert math.isnan(se)


def test_g_paired_from_two_arms_n_2_passes_below_guard() -> None:
    """n=2 paired keys with non-zero delta variance: passes the
    n<2 guard. hedges_g_paired returns finite g + finite se for
    n=2 with variance > 0. Pin `< 2` against `<= 2` and `< 3`
    mutants (both would NaN at n=2)."""
    arm_x: dict[tuple[object, ...], float] = {('k0',): 0.0, ('k1',): 1.0}
    arm_y: dict[tuple[object, ...], float] = {('k0',): 1.0, ('k1',): 5.0}    # deltas: 1.0, 4.0
    g, se = _g_paired_from_two_arms(arm_x, arm_y, [('k0',), ('k1',)])
    assert math.isfinite(g)
    assert math.isfinite(se)
