"""Robustness probe: DerSimonian-Laird's `random_effects_summary`
under small G (number of strata).

The framework's `random_effects_summary` uses the DerSimonian-
Laird estimator for between-cell heterogeneity τ²:

    τ²_DL = max(0, (Q − df) / c_term)

Known limitations from the meta-analysis literature
(Veroniki et al. 2016, Langan et al. 2019):

  - Point-estimate bias is small but the **sampling variance of
    τ²_DL is enormous at small G** (≤ 5). Two studies with
    identical structural τ² can yield wildly different point
    estimates.
  - The `max(0, ·)` clip introduces an **upward bias when true
    τ² = 0**: negative-valued estimates get replaced with 0, so
    the mean of clipped estimates is positive even when truth is
    zero.
  - **I² detection power for modest heterogeneity is weak below
    G = 10**: even when structural τ² produces population I²
    ≈ 0.5, sample I² often lands below the framework's 0.5
    SCOPE_FLAG threshold.

This probe quantifies these effects via Monte Carlo across
G ∈ {3, 5, 10, 20, 50}. Findings inform implementation-author
guidance:

  - DL τ² POINT ESTIMATE is approximately unbiased at any G ≥ 3.
  - DL τ² is UNRELIABLE for inference at G ≤ 5: MC sampling SD
    is on the order of τ² itself.
  - When truth is τ² = 0, DL reports a small positive value that
    decays as 1/G — `random_effects_verdict`'s NULL_EFFECT
    routing is conservative against this artifact.
  - For small-G meta-analyses, **prefer REML or HKSJ** (not DL).
    The framework currently exposes only DL; this probe documents
    the gap and the empirical pinpoints where it matters.

Empirical numbers are anchored to deterministic seeds (zlib.adler32
of `('dl', G, k)`) — bit-for-bit reproducible.
"""
from __future__ import annotations

import math
import zlib

import numpy as np

from corroborate.stats.effect_size import random_effects_summary


def _det_seed(*parts: object) -> int:
    return zlib.adler32(repr(parts).encode()) & 0xFFFF_FFFF


_K_REPLICATES = 300
"""Monte Carlo replicate count. MC SE on mean(τ²) ≈ sd(τ²)/√300.
At G=20, sd(τ²) ≈ 0.17 → MC SE ≈ 0.010. Bound widths reflect this."""


def _measure_dl_summary(
    *,
    G: int,
    tau_true_sq: float,
    v_within: float,
    mu_overall: float = 0.5,
    n_replicates: int = _K_REPLICATES,
) -> tuple[float, float, float]:
    """Run K MC replicates of:
      μ_e ~ N(μ_overall, τ²_true)     (true env-level effects)
      g_e ~ N(μ_e, v_within)          (sampled with within-env SE)
      ses_e = √v_within               (uniform within-env SE)

    Pool with `random_effects_summary`. Return:
      (mean(τ²_DL), sd(τ²_DL), mean(I²)).
    """
    tau2_est: list[float] = []
    i2_est: list[float] = []
    for k in range(n_replicates):
        rng = np.random.default_rng(_det_seed('dl', G, k))
        # True env-level effects (zero variance when tau_true_sq=0).
        if tau_true_sq > 0:
            mus = rng.normal(mu_overall, math.sqrt(tau_true_sq), G)
        else:
            mus = np.full(G, mu_overall)
        gs = mus + rng.normal(0.0, math.sqrt(v_within), G)
        ses = np.full(G, math.sqrt(v_within))
        pool = random_effects_summary(list(zip(gs.tolist(), ses.tolist())))
        tau2_est.append(pool.tau2)
        i2_est.append(pool.I2)
    arr = np.array(tau2_est, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1)), float(np.mean(i2_est))


# ============ Sampling-variance probe at small G ============

def test_dl_tau2_sampling_variance_explodes_at_small_g() -> None:
    """**Pin the sampling-variance scale.** At G=3 with structural
    τ²=0.5, MC sampling SD of τ²_DL is ≈ 0.52 — comparable to τ²
    itself. The point estimate is meaningless for inference at
    this G.

    Empirical (K=300, structural τ²=0.5, v_within=0.025):
        G=3:  MC_sd(τ²) ≈ 0.524   (CV ≈ 105%)
        G=5:  MC_sd(τ²) ≈ 0.336   (CV ≈ 67%)
        G=10: MC_sd(τ²) ≈ 0.252   (CV ≈ 50%)
        G=20: MC_sd(τ²) ≈ 0.173   (CV ≈ 35%)
        G=50: MC_sd(τ²) ≈ 0.105   (CV ≈ 21%)

    Bound: at G=3, MC_sd(τ²) > 0.4 (extreme variance); at G=20,
    MC_sd(τ²) < 0.25 (10× shrinkage achieved by quadrupling G).
    """
    _, mc_sd_g3, _ = _measure_dl_summary(
        G=3, tau_true_sq=0.5, v_within=0.025,
    )
    assert mc_sd_g3 > 0.4, (
        f'G=3, τ²=0.5: MC_sd(τ²) = {mc_sd_g3:.4f}; expected > 0.4 '
        f'(point estimate has CV ≈ 100% at this G — DL τ² is not '
        f'usable for inference). A fix that narrows this would '
        f'breach the bound.'
    )
    _, mc_sd_g20, _ = _measure_dl_summary(
        G=20, tau_true_sq=0.5, v_within=0.025,
    )
    assert mc_sd_g20 < 0.25, (
        f'G=20, τ²=0.5: MC_sd(τ²) = {mc_sd_g20:.4f}; expected '
        f'< 0.25 (CV reduces to ~35%; usable for inference).'
    )


def test_dl_tau2_point_estimate_approximately_unbiased() -> None:
    """**Pin point-estimate bias**: across G ∈ {5, 10, 20}, mean
    DL τ² is within ±0.05 of true τ²=0.5. DL is approximately
    unbiased at any G ≥ 5 — the issue is variance, not bias.

    Empirical (K=300, deterministic seeds):
        G=5:  bias = -0.044  (8.8% rel)
        G=10: bias = -0.023  (4.5% rel)
        G=20: bias = +0.011  (2.3% rel)
        G=50: bias = +0.011  (2.3% rel)

    Bound: |bias| < 0.06 at every G ≥ 5. A regression that
    introduces systematic bias (e.g., a wrong c_term denominator
    formula) would breach.
    """
    for G, expected_bias, slack in (
        (5, -0.044, 0.02),
        (10, -0.023, 0.02),
        (20, +0.011, 0.02),
    ):
        mc_mean, _, _ = _measure_dl_summary(
            G=G, tau_true_sq=0.5, v_within=0.025,
        )
        bias = mc_mean - 0.5
        assert abs(bias) < 0.06, (
            f'G={G}, τ²=0.5: bias = {bias:+.4f}; expected '
            f'|bias| < 0.06 (DL is approximately unbiased at '
            f'G ≥ 5; variance is the issue, not bias).'
        )
        assert abs(bias - expected_bias) < slack, (
            f'G={G}, τ²=0.5: bias = {bias:+.4f}; expected '
            f'{expected_bias:+.4f} ± {slack:.3f} from fixed-seed MC.'
        )


# ============ The max(0, ·) clip artifact under true τ²=0 ============

def test_dl_tau2_inflated_under_true_zero_heterogeneity() -> None:
    """**Pin the max(0, ·) clip artifact**: when structural τ² = 0,
    DL reports a small POSITIVE mean even though truth is zero.
    Negative-valued (Q − df) get clipped to 0 — the half of the
    sampling distribution where the unclipped estimate would be
    negative still contributes 0 to the mean, biasing it upward.

    Empirical (K=300, structural τ²=0, v_within=0.025):
        G=3:  MC[τ²] ≈ 0.011  (no truth to compare against; this
                              IS the floor noise)
        G=5:  MC[τ²] ≈ 0.006
        G=10: MC[τ²] ≈ 0.004
        G=20: MC[τ²] ≈ 0.002

    The bias decays roughly as 1/G — half the sampling
    distribution gets clipped, and as G grows the half-width of
    the unclipped distribution shrinks.

    Bound: at every G, MC[τ²] > 0 (the clip ALWAYS produces a
    positive mean), AND MC[τ²] ≤ 0.02 (the bias is bounded above).
    Substrate-author guidance: a small DL τ² estimate (< 0.02 at
    G ≥ 10) is consistent with TRUE τ² = 0; don't over-interpret.
    """
    for G, expected_mean, slack in (
        (3, 0.011, 0.005),
        (5, 0.006, 0.003),
        (10, 0.004, 0.002),
        (20, 0.002, 0.002),
    ):
        mc_mean, _, _ = _measure_dl_summary(
            G=G, tau_true_sq=0.0, v_within=0.025,
        )
        assert mc_mean > 0, (
            f'G={G}, true τ²=0: MC[τ²] = {mc_mean:.4f}; expected '
            f'positive (max(0, ·) clip always produces a small '
            f'positive mean even at zero truth).'
        )
        assert mc_mean <= 0.02, (
            f'G={G}, true τ²=0: MC[τ²] = {mc_mean:.4f}; expected '
            f'≤ 0.02 (bias from clip is bounded). Larger value '
            f'would mean DL is over-estimating heterogeneity '
            f'where none exists.'
        )
        assert abs(mc_mean - expected_mean) < slack, (
            f'G={G}, true τ²=0: MC[τ²] = {mc_mean:.4f}; expected '
            f'{expected_mean:.4f} ± {slack:.3f} from fixed-seed MC.'
        )


# ============ I² detection power at small G ============

def test_dl_i2_below_scope_threshold_when_g_small_and_tau_modest() -> None:
    """**Pin the I² detection-power gap**: with structural τ²=0.05
    and v_within=0.025, the population I² is τ²/(τ²+v) =
    0.05/0.075 = 0.667 — above the framework's 0.5
    SCOPE_FLAG threshold. But sample I² lands BELOW 0.5 at G ≤ 5.

    Empirical (K=300, structural τ²=0.05):
        G=3:  MC[I²] ≈ 0.439   (below 0.5 → no SCOPE_FLAG)
        G=5:  MC[I²] ≈ 0.482   (below 0.5)
        G=10: MC[I²] ≈ 0.553   (just above 0.5)
        G=20: MC[I²] ≈ 0.629   (clearly above 0.5)
        G=50: MC[I²] ≈ 0.661   (close to population value 0.667)

    Substrate-author guidance: at G ≤ 5, a sample I² < 0.5 does
    NOT mean "no heterogeneity" — it means "DL can't detect it
    at this G." Combine with the τ² point-estimate variance probe
    above: at small G, the framework can't reliably distinguish
    "no scope" from "scope is real but underpowered."

    Bound: G=3 sample I² < 0.5; G=20 sample I² ≥ 0.55. A fix
    that improves I² detection at small G (e.g., switching to a
    different τ² estimator that doesn't suffer from clipping)
    would breach the G=3 upper bound.
    """
    _, _, mc_i2_g3 = _measure_dl_summary(
        G=3, tau_true_sq=0.05, v_within=0.025,
    )
    assert mc_i2_g3 < 0.5, (
        f'G=3, τ²=0.05 (population I²=0.667): MC[I²] = '
        f'{mc_i2_g3:.4f}; expected < 0.5 (DL can\'t detect '
        f'modest heterogeneity at G=3, leaves SCOPE_FLAG unset).'
    )
    _, _, mc_i2_g20 = _measure_dl_summary(
        G=20, tau_true_sq=0.05, v_within=0.025,
    )
    assert mc_i2_g20 >= 0.55, (
        f'G=20, τ²=0.05: MC[I²] = {mc_i2_g20:.4f}; expected '
        f'≥ 0.55 (G=20 has the power to detect modest '
        f'heterogeneity; SCOPE_FLAG triggers correctly).'
    )
