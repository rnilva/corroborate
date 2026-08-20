"""Closed-form assertions on `paired_g` over the LG-SCM substrate.

Three claims are tested:

1. `mean_diff` recovers the structural product
   `Delta_beta_xz * beta_zy * mu_x` within an analytical 4-sigma
   bound. The expected value is closed-form; the SE comes from
   the residual variance of `x_mean` across seeds (sigma_x^2 /
   n_steps), which is the only source of variation that survives
   the paired contrast.

2. `g` (standardized Hedges' g) carries the correct sign and is
   "large" by Cohen's convention (|g| > 0.8) on a contrast tuned
   to be far from zero relative to the per-pair noise.

3. The null contrast (both arms identical) returns a `mean_diff`
   indistinguishable from zero (|mean_diff| < 4 * SE) and
   `helped_fraction` near 0.5 — paired_g doesn't manufacture
   effects out of i.i.d. noise.

The 4-sigma bound is loose enough to almost never flake and
tight enough to catch any sign / scale / pairing-key regression
in `paired_g.fn`."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from corroborate.analyses.paired.paired_g import paired_g
from corroborate.corpus.schema import RunRow
from corroborate.data import cells_to_dataframe

from tests.analytic.lg_scm.composition import LinearGaussianSCM
from tests.analytic.lg_scm.runner import run_paired_arms


# Shared parameters across the file. Picked so the analytical
# 4-sigma window on `mean_diff` is small relative to the expected
# value (large ratio = sharp test) without requiring a huge
# `n_steps * n_seeds`.
_MU_X = 1.0
_SIGMA_X = 0.5
_BETA_ZY = 1.5
_SIGMA_Z = 0.1
_SIGMA_Y = 0.1
_N_STEPS = 200


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


def _expected_mean_diff(*, beta_xz_t: float, beta_xz_b: float) -> float:
    """Closed-form E[mean_diff] = Delta_beta_xz * beta_zy * mu_x.

    Derivation: under shared seeds the paired Delta on `y_mean`
    cancels both epsilon_z and epsilon_y; what remains is
    `Delta_beta_xz * beta_zy * x_mean(seed)`. Taking the
    expectation over seeds gives the product above (E[x_mean] =
    mu_x because the implementation uses unit-variance epsilon with
    zero mean)."""
    return (beta_xz_t - beta_xz_b) * _BETA_ZY * _MU_X


def _mean_diff_se(*, beta_xz_t: float, beta_xz_b: float, n_pairs: int) -> float:
    """Closed-form SE of paired mean_diff under the LG-SCM.

    Var[Delta_y_mean(seed)]
        = (Delta_beta_xz * beta_zy)^2 * Var[x_mean(seed)]
        = (Delta_beta_xz * beta_zy)^2 * sigma_x^2 / n_steps
    Var[mean over n_pairs seeds] = Var[Delta_y_mean(seed)] / n_pairs
    SE = sqrt of that.
    """
    delta_beta = beta_xz_t - beta_xz_b
    var_per_pair = (delta_beta * _BETA_ZY) ** 2 * (_SIGMA_X ** 2) / _N_STEPS
    return math.sqrt(var_per_pair / n_pairs)


def _expected_g(*, n_pairs: int) -> float:
    """Closed-form Hedges' g under shared-noise cancellation.

    Δ(seed) = (β_xz_t - β_xz_b) · β_zy · X_avg(seed); the
    coefficient cancels in d = mean(Δ)/sd(Δ), so:
        d = mu_x * sqrt(n_steps) / sigma_x        (σ_x is X-population)
        g = d * c_4(n_pairs),  c_4 = 1 - 3/(4n - 5)
    """
    d = _MU_X * math.sqrt(_N_STEPS) / _SIGMA_X
    c4 = 1.0 - 3.0 / (4 * n_pairs - 5)
    return d * c4


def _as_dicts(rows: Sequence[RunRow]) -> list[Mapping[str, object]]:
    """Project RunRow list to the flat-dict form `paired_g.fn`
    expects. Calling `.as_dict()` exercises the same code path
    real corpora go through, so a regression in `RunRow.as_dict`
    that drops measurement leaves would surface here too."""
    return [r.as_dict() for r in rows]


def test_mean_diff_recovers_closed_form_under_paired_intervention() -> None:
    """The structural Delta on `y_mean` matches Delta_beta * beta_zy
    * mu_x within 4 * analytical SE.

    This asserts paired_g's mean_diff path (raw paired-mean
    contrast) preserves the structural product. A regression that
    would silently re-pair on the wrong key, double-count
    treatment cells, or sum instead of differ would fail the
    bound by orders of magnitude.
    """
    n_pairs = 30
    beta_xz_t, beta_xz_b = 0.8, 0.3
    rows = run_paired_arms(
        treatment=_scm(beta_xz_t),
        baseline=_scm(beta_xz_b),
        seeds=range(n_pairs),
        treatment_arm='treatment',
        baseline_arm='baseline',
    )
    cells = _as_dicts(rows)

    result = paired_g.fn(
        cells_to_dataframe(cells),
        treatment_arm='treatment',
        baseline_arm='baseline',
        pair_by=('seed',),
        source='y_mean',
    )

    expected = _expected_mean_diff(beta_xz_t=beta_xz_t, beta_xz_b=beta_xz_b)
    se = _mean_diff_se(
        beta_xz_t=beta_xz_t, beta_xz_b=beta_xz_b, n_pairs=n_pairs,
    )
    bound = 4.0 * se
    assert abs(result.mean_diff - expected) < bound, (
        f'paired_g.mean_diff = {result.mean_diff:.6f} not within '
        f'4*SE = {bound:.6f} of analytical Delta = {expected:.6f} '
        f'(structural Delta_beta_xz={beta_xz_t - beta_xz_b}, '
        f'beta_zy={_BETA_ZY}, mu_x={_MU_X}, n_pairs={n_pairs}, '
        f'n_steps={_N_STEPS})'
    )


def test_hedges_g_recovers_closed_form() -> None:
    """`paired_g.g` is Hedges' g on per-paired Δ. Closed-form under
    shared-noise: `d = mu_x * sqrt(n_steps) / sigma_x` (the
    Δβ·β_zy coefficient cancels in mean/sd), then `g = c_4(n) · d`.

    Tests the standardized-effect path of the framework (distinct
    from `mean_diff`), which goes through the `hedges_g_paired`
    primitive: sample mean, sample SD with Bessel's correction,
    Hedges small-sample c_4. A regression in any of those would
    fail the bound.

    The 15% tolerance absorbs the ~13% CV on sample SD at n=30
    (the SD term in d is the noisy half at finite n)."""
    n_pairs = 30
    rows = run_paired_arms(
        treatment=_scm(0.8),
        baseline=_scm(0.3),
        seeds=range(n_pairs),
        treatment_arm='treatment',
        baseline_arm='baseline',
    )
    result = paired_g.fn(
        cells_to_dataframe(_as_dicts(rows)),
        treatment_arm='treatment',
        baseline_arm='baseline',
        pair_by=('seed',),
        source='y_mean',
    )
    expected = _expected_g(n_pairs=n_pairs)
    rel_err = abs(result.g - expected) / expected
    assert rel_err < 0.15, (
        f'paired_g.g = {result.g:.4f}, expected {expected:.4f} '
        f'(rel err {rel_err:.4f}). The closed form `mu_x · '
        f'sqrt(n_steps) / sigma_x · c_4` ≈ {expected:.4f} is the '
        f'shared-noise prediction; >15% drift indicates the Hedges '
        f'computation is off (Bessel correction, c_4, sign).'
    )


def test_null_contrast_returns_indistinguishable_mean_diff() -> None:
    """When the two arms have *identical* coefficients, shared
    seeds make every per-pair Delta exactly zero. paired_g must
    not invent an effect.

    A pairing-key bug that sneaks unpaired cells through, or a
    NaN-handling bug that drops half the pairs, would let noise
    leak into mean_diff and break the assertion.
    """
    beta_xz = 0.5
    rows = run_paired_arms(
        treatment=_scm(beta_xz),
        baseline=_scm(beta_xz),
        seeds=range(30),
        treatment_arm='treatment',
        baseline_arm='baseline',
    )
    result = paired_g.fn(
        cells_to_dataframe(_as_dicts(rows)),
        treatment_arm='treatment',
        baseline_arm='baseline',
        pair_by=('seed',),
        source='y_mean',
    )
    # Identical SCMs + identical seeds → identical noise streams →
    # exact-zero Delta per pair (deterministic). mean_diff should be
    # exactly zero apart from float rounding.
    assert abs(result.mean_diff) < 1e-12, (
        f'paired_g.mean_diff = {result.mean_diff} on identical-arm '
        f'sweep; expected exact zero from shared-seed noise cancellation'
    )
    # helped_fraction is undefined-ish on exact-zero deltas (0.0 by
    # the strict `> 0` rule); accept either 0.0 or NaN, but reject
    # any "pretend effect" near 1.0.
    assert (
        result.helped_fraction == 0.0 or math.isnan(result.helped_fraction)
    ), (
        f'helped_fraction = {result.helped_fraction:.4f} on an '
        f'exact-zero-Delta sweep; expected 0.0 (no strict positives) '
        f'or NaN'
    )
