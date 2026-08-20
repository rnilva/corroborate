"""Bootstrap-CI paired Hedges' g — distribution-free SE for the
paired-g primitive.

The framework's `paired_g` reports an analytical SE derived
under asymptotic normality of `g`. Under skewed/heavy-tailed Δ,
this SE is **anti-conservative** at moderate-to-large n: confidence
intervals under-cover by 15-25% (see ROBUSTNESS.md and
`tests/analytic/robustness/test_paired_g_skew_robustness.py`).

This primitive computes percentile bootstrap CIs by resampling
the per-pair Δ vector with replacement. The bootstrap distribution
of `g` is **non-parametric** — it doesn't assume the sampling
distribution of `g` is normal, and the CI bounds capture
asymmetries the analytical SE collapses to a single SE number.

**Honest caveat**: the percentile bootstrap is NOT a panacea on
heavy-tailed Δ at small n. The bootstrap resamples FROM THE
SAMPLE; if a 50-draw sample didn't capture the population's
tail, the bootstrap can't either. Empirically at n=50 log-normal,
mean(bootstrap_se) ≈ 0.73·MC_sd_g — comparable to paired_g.se's
0.77·MC_sd_g; both are anti-conservative on heavy tails at small
n. The bootstrap's true benefits over paired_g:
  - **Asymmetric CI bounds** that capture skew in g's sampling
    distribution (paired_g's symmetric `g ± 1.96·se` flattens this).
  - **Distribution-free** by construction; no normality assumption
    on g's sampling distribution to violate.
  - **Quantile-based bounds** that don't degrade as catastrophically
    under extreme Δ shapes as the analytical formula does.

For the strongest skew/tail-bias correction, layer two
complementary primitives: `bootstrap_paired_g` for asymmetric
CIs + `cliff_delta_paired` for skew-robust point magnitude.

Algorithm:
  1. Compute observed g from the original Δ vector.
  2. For B bootstrap replicates: resample n Δ values with
     replacement; compute g_b.
  3. CI = (percentile_α/2(g_b), percentile_1-α/2(g_b)).
  4. SE_bootstrap = MC sampling SD of g_b.

Trade-off vs `paired_g`:
  - Slower (B× more compute; B=1000 default).
  - CIs are well-calibrated under non-normal Δ (the primary
    benefit; `paired_g.se` under-covers by 15-25% on heavy
    tails).
  - Bootstrap CIs DO NOT correct the SKEW BIAS in the point
    estimate of g. They give honest CIs around a still-biased
    point estimate. To address skew bias on the point itself,
    pair with `cliff_delta_paired`.

When to use:
  - Heavy-tailed Δ at moderate n (the SE-anti-conservativeness regime).
  - Bridges that report CI bounds rather than just point estimates.

When NOT to use:
  - Normal-ish Δ (paired_g.se is well-calibrated; bootstrap is
    just slower for the same answer).
  - n < 20 (bootstrap CIs themselves have wide MC noise at small n).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import polars as pl

from corroborate._internals.polars import to_dicts
from corroborate.analyses._cell_value import key_tuple, resolve_value
from corroborate.bridge.analysis import analysis


@dataclass(frozen=True, slots=True)
class BootstrapPairedGResult:
    """Output of bootstrap paired Hedges' g.

    `g` is the OBSERVED Hedges' g on the original Δ vector
    (matches `paired_g.g`). Note: bootstrap CIs do NOT bias-
    correct the point estimate; pair with `cliff_delta_paired`
    for skew-robust point-estimate magnitude.

    `se_bootstrap` is the MC sampling SD across bootstrap replicates
    — distribution-free, well-calibrated under non-normal Δ.

    `ci_lo`, `ci_hi` are the α-percentile and (1-α)-percentile
    bootstrap CI bounds at `alpha`.

    `b_replicates` is the number of bootstrap samples (default 1000).

    NaN-filled when n_pairs < 2."""
    g: float
    se_bootstrap: float
    ci_lo: float
    ci_hi: float
    n_pairs: int
    b_replicates: int
    alpha: float
    pair_by: tuple[str, ...]
    measurable: str
    treatment_arm: str
    baseline_arm: str

    @property
    def ci_excludes_zero(self) -> bool:
        """True iff the CI doesn't include zero — the "significant
        at α" verdict for bootstrap-derived bounds."""
        if math.isnan(self.ci_lo) or math.isnan(self.ci_hi):
            return False
        return self.ci_lo > 0 or self.ci_hi < 0


def _hedges_g_paired_inline(deltas: np.ndarray) -> float:
    """Inline c_4-corrected Hedges' g for paired Δ. Matches
    `corroborate.stats.hedges_g_paired` but avoids the import-cycle
    risk on the bootstrap fast path (called B times per cell)."""
    n = len(deltas)
    if n < 2:
        return float('nan')
    mean = float(deltas.mean())
    sd = float(deltas.std(ddof=1))
    if sd == 0.0:
        return float('nan')
    c4 = 1.0 - 3.0 / (4 * n - 5)
    return mean / sd * c4


@analysis
def bootstrap_paired_g(
    cells: pl.DataFrame,
    *,
    source: str,
    treatment_arm: str,
    baseline_arm: str,
    pair_by: tuple[str, ...] = ('seed',),
    arm_field: str = 'arm_key',
    dedupe_strategy: str = 'mean',
    b_replicates: int = 1000,
    alpha: float = 0.05,
    bootstrap_seed: int = 0,
) -> BootstrapPairedGResult:
    """Compute paired Hedges' g + percentile bootstrap CIs.

    `b_replicates` controls bootstrap precision. B=1000 gives MC
    SE on each percentile of ~ 1.5% × CI width (Efron-Tibshirani
    1993); raise to 5000 for very tight bounds, lower to 200 for
    smoke tests.

    `alpha` is the two-sided CI level (default 0.05 → 95% CI).

    `bootstrap_seed` controls the resampling RNG; deterministic
    output for fixed seed. Distinct from substrate seeds (the
    bootstrap is a post-hoc resample of the existing Δ vector).

    `dedupe_strategy` mirrors `paired_g`: defaults to `'mean'`
    (per-cell aggregation within each `(arm, pair_by)` bucket
    before bootstrap); pass `'raise'` to error on duplicates.

    See module docstring for when to use vs paired_g."""
    rows = to_dicts(cells)
    if dedupe_strategy not in ('raise', 'mean'):
        raise ValueError(
            f'bootstrap_paired_g: unknown dedupe_strategy '
            f'{dedupe_strategy!r}; expected "raise" or "mean"',
        )
    if b_replicates < 100:
        raise ValueError(
            f'bootstrap_paired_g: b_replicates={b_replicates} too '
            f'low for stable percentile estimation; use ≥ 100',
        )
    if not (0.0 < alpha < 1.0):
        raise ValueError(
            f'bootstrap_paired_g: alpha={alpha} out of (0, 1)',
        )

    treatment_buckets: dict[tuple[object, ...], list[float]] = {}
    baseline_buckets: dict[tuple[object, ...], list[float]] = {}
    for cell in rows:
        arm = cell.get(arm_field)
        if arm == treatment_arm:
            key = key_tuple(cell, pair_by)
            bucket = treatment_buckets.setdefault(key, [])
            if bucket and dedupe_strategy == 'raise':
                raise ValueError(
                    f'bootstrap_paired_g: duplicate cell for '
                    f'{treatment_arm!r} at pair_by={pair_by} key={key}',
                )
            bucket.append(resolve_value(cell, source))
        elif arm == baseline_arm:
            key = key_tuple(cell, pair_by)
            bucket = baseline_buckets.setdefault(key, [])
            if bucket and dedupe_strategy == 'raise':
                raise ValueError(
                    f'bootstrap_paired_g: duplicate cell for '
                    f'{baseline_arm!r} at pair_by={pair_by} key={key}',
                )
            bucket.append(resolve_value(cell, source))

    treatment: dict[tuple[object, ...], float] = {
        k: (
            sum(v for v in vs if not math.isnan(v))
            / max(1, sum(1 for v in vs if not math.isnan(v)))
        ) if any(not math.isnan(v) for v in vs) else float('nan')
        for k, vs in treatment_buckets.items()
    }
    baseline: dict[tuple[object, ...], float] = {
        k: (
            sum(v for v in vs if not math.isnan(v))
            / max(1, sum(1 for v in vs if not math.isnan(v)))
        ) if any(not math.isnan(v) for v in vs) else float('nan')
        for k, vs in baseline_buckets.items()
    }

    paired_keys = sorted(set(treatment) & set(baseline))
    deltas_list = [
        treatment[k] - baseline[k]
        for k in paired_keys
        if not (math.isnan(treatment[k]) or math.isnan(baseline[k]))
    ]
    n_pairs = len(deltas_list)
    if n_pairs < 2:
        return BootstrapPairedGResult(
            g=float('nan'), se_bootstrap=float('nan'),
            ci_lo=float('nan'), ci_hi=float('nan'),
            n_pairs=n_pairs, b_replicates=b_replicates, alpha=alpha,
            pair_by=pair_by, measurable=source,
            treatment_arm=treatment_arm, baseline_arm=baseline_arm,
        )

    deltas = np.asarray(deltas_list, dtype=np.float64)
    g_observed = _hedges_g_paired_inline(deltas)

    rng = np.random.default_rng(bootstrap_seed)
    g_replicates = np.empty(b_replicates, dtype=np.float64)
    for b in range(b_replicates):
        indices = rng.integers(0, n_pairs, size=n_pairs)
        g_replicates[b] = _hedges_g_paired_inline(deltas[indices])
    valid = g_replicates[~np.isnan(g_replicates)]
    if len(valid) < 10:
        # Bootstrap distribution is degenerate (most resamples
        # had zero variance — e.g., almost-constant Δ). Return
        # NaN CIs rather than a misleading point.
        return BootstrapPairedGResult(
            g=g_observed, se_bootstrap=float('nan'),
            ci_lo=float('nan'), ci_hi=float('nan'),
            n_pairs=n_pairs, b_replicates=b_replicates, alpha=alpha,
            pair_by=pair_by, measurable=source,
            treatment_arm=treatment_arm, baseline_arm=baseline_arm,
        )

    se_bootstrap = float(valid.std(ddof=1))
    ci_lo = float(np.quantile(valid, alpha / 2))
    ci_hi = float(np.quantile(valid, 1 - alpha / 2))

    return BootstrapPairedGResult(
        g=g_observed,
        se_bootstrap=se_bootstrap,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        n_pairs=n_pairs,
        b_replicates=b_replicates,
        alpha=alpha,
        pair_by=pair_by,
        measurable=source,
        treatment_arm=treatment_arm,
        baseline_arm=baseline_arm,
    )


__all__ = ['BootstrapPairedGResult', 'bootstrap_paired_g']
