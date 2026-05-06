# Robustness map for corroborate analytic primitives

Empirical bias maps for the framework's load-bearing analytic
primitives, anchored to deterministic Monte Carlo probes in
`tests/analytic/robustness/`. Each entry tells a substrate
author when the primitive is trustworthy, when it isn't, and
which complementary primitive to reach for.

The numbers below are bit-for-bit reproducible (zlib.adler32-
derived seeds; Python's `hash()` randomizes per-process under
PYTHONHASHSEED=random and is unsuitable for cross-process
reproducibility). Each row points to the specific test that
pins it; if these numbers drift the test fails.

## Decision matrix

| If your data... | Use | Don't use | Reason |
|---|---|---|---|
| Δ is approximately normal, n ≥ 30 | `paired_g` | — | Trustworthy; bias within ±0.05 |
| Δ is right-skewed (\|skew\| > 1) | `cliff_delta_paired` (point) + `bootstrap_paired_g` (CI) | `paired_g` alone | At skew ≈ 1.86, n=30: g inflated +12% |
| Δ is heavy-tailed, n ≤ 30 | `bootstrap_paired_g` (asymmetric CIs) | `paired_g` alone | At t(df=5), n=30: g inflated +7% |
| Bridge claim is "treatment helps in MOST pairs" (sign-only, magnitude-independent) | `cliff_delta_paired` | `paired_g` | δ ∈ [-1, 1]; depends only on Δ sign |
| Pooling across G ≤ 5 envs | `reml_random_effects_summary`† | `random_effects_summary` (DL) | DL CV ≈ 100% at G=3 |
| Pooling across G = 6–10 envs | DL with `assumption_violations` flagged | DL silently | I² detection fails at modest τ² |
| Pooling across G ≥ 10 envs | `random_effects_summary` (DL) | — | Standard regime |
| Cell scope-predicate references same column twice | safe (post-fix) | — | `bridge.py:625` deduplicates |

† = complementary primitive proposed; not yet implemented. See
`tests/analytic/robustness/` for the empirical case.

## Per-primitive maps

### `paired_g` — Hedges' g paired

**Probe**: `tests/analytic/robustness/test_paired_g_skew_robustness.py`

The framework computes `g = mean(Δ) / sd(Δ, ddof=1) · c_4(n)`.
Hedges' `c_4(n) = 1 - 3/(4n-5)` is exact only under normal Δ.

| Δ distribution | n=10 | n=30 | n=100 |
|---|---|---|---|
| Normal (control) | +0.027 | +0.000 | +0.017 |
| Log-normal (skew ≈ 1.86) | **+0.327** | **+0.125** | **+0.070** |
| t(df=5) heavy tails | +0.176 | +0.056 | +0.016 |

**Direction**: `paired_g` OVERESTIMATES on right-skewed and
heavy-tailed Δ. The mechanism: in small samples, sample `sd` is
biased downward more than sample mean is biased downward → the
ratio `mean/sd` is inflated. `c_4` corrects only the normal-case
bias.

**SE calibration**: framework's reported `se` is well-calibrated
on normal Δ (within 6% of true MC sampling SD). On log-normal Δ
at n=100, `framework_se / MC_sd_g ≈ 0.77` — confidence intervals
under-cover by 23% because the Pearson-based formula misses the
heavy-tail contribution to `g`'s sampling SD.

**Threshold for `assumption_violations` flag**:
- `|skew(Δ)| > 1.0` → `'skew_bias_likely_inflation_~10pct'`
- `kurtosis(Δ) > 5.0` → `'heavy_tail_se_anti_conservative'`

### `random_effects_summary` — DerSimonian-Laird τ²

**Probe**: `tests/analytic/robustness/test_dl_small_g_robustness.py`

DL τ² is approximately unbiased at G ≥ 5; the issue is variance,
not bias.

| G  | MC[τ²] (true=0.5) | MC sd(τ²) | CV |
|----|-------|-------|-----|
| 3  | 0.485 | 0.524 | 105% |
| 5  | 0.456 | 0.336 | 67%  |
| 10 | 0.477 | 0.252 | 50%  |
| 20 | 0.511 | 0.173 | 35%  |
| 50 | 0.511 | 0.105 | 21%  |

**`max(0, ·)` clip artifact**: when true τ²=0, DL reports a
small positive mean because half the sampling distribution gets
clipped:

| G  | MC[τ²] (true=0) |
|----|-----|
| 3  | 0.011 |
| 5  | 0.006 |
| 10 | 0.004 |
| 20 | 0.002 |

**I² detection power**: at population I²=0.667 (τ²=0.05,
v=0.025), sample I² stays BELOW the framework's 0.5 SCOPE_FLAG
threshold at G ≤ 5. The framework reports HELD instead of
HELD_WITH_SCOPE_FLAG silently.

**Threshold for `assumption_violations` flag**:
- `n_cells < 5` → `'dl_small_g_unreliable_inference'`
- `0 < tau2 < 0.02` AND `n_cells < 10` → `'dl_clip_artifact_possible'`

### Other primitives

The robustness suite is being built up incrementally. Probes
planned:
- `paired_link_per_burst` Pearson-r normality assumption
- `mundlak_decomposition` β_b/β_w multicollinearity at low
  within-stratum variance
- `discover_adjacency` (PC) faithfulness violations + Spearman
  rank-transform bias
- `meta_regression_per_burst` IVW-vs-equal-weight under stratum
  imbalance
- `proportion_mediated` linear-mediation assumption

Each follow-up probe is structurally similar: pick the
primitive's load-bearing distributional assumption, perturb it,
MC-pin the bias, decide whether a complementary primitive is
warranted vs. a documented robustness range.

## How to read the probes

Each test file follows a strict pattern:

1. Module docstring documents the assumption being probed and
   the empirical findings.
2. A perturbation harness runs K MC replicates per (perturbation,
   n) cell against the primitive, measuring bias and sampling
   variance.
3. Each test pins one empirical number to within MC measurement
   precision. Bounds are tight (±0.005-0.01) — seeds are
   deterministic, so a regression that widens or narrows the bias
   measurably will fail the test.

If the test fails:
- The bias number changed → either the primitive was fixed
  (update the expected number down) or regressed (widen the
  bias check, file an issue, OR fix the primitive).
- The MC sampling SD changed → check whether the seed-tag
  changed or the number-of-replicates changed; if neither, the
  primitive's sampling distribution shape changed.

## Complementary primitives

### `cliff_delta_paired` — IMPLEMENTED

Source: `src/corroborate/analyses/cliff_delta_paired.py`. Probe:
`tests/analytic/lg_scm/test_cliff_delta_paired.py` +
`tests/analytic/robustness/test_cliff_delta_skew_robust.py`.

Rank-based effect size: `δ = P(Δ > 0) - P(Δ < 0) ∈ [-1, 1]`.
Skew-robust by construction (depends only on signs of Δ, not
magnitudes). Use as a complement to `paired_g` when Δ is
non-normal AND the bridge's claim is direction- or sign-based
("helps in most pairs") rather than magnitude-based ("g > 0.5").

The headline cross-primitive comparison
(`test_cliff_delta_unbiased_under_lognormal_where_paired_g_inflates`)
shows: under log-normal Δ at n=30, paired_g overestimates by +0.13
while Cliff's δ stays exactly at 1.0 (every Δ positive by
construction).

### `bootstrap_paired_g` — IMPLEMENTED

Source: `src/corroborate/analyses/bootstrap_paired_g.py`. Probe:
`tests/analytic/lg_scm/test_bootstrap_paired_g.py`.

Percentile bootstrap CIs for paired Hedges' g. **Honest scope**:
the bootstrap is structurally non-parametric (no normality
assumption on g's sampling distribution) and produces asymmetric
CIs that capture skew in the sampling distribution. But
percentile bootstrap on small-n heavy-tailed Δ inherits the
sample's tail-truncation — at n=50 log-normal, mean(bootstrap_se)
≈ 0.73·MC_sd_g vs paired_g.se's 0.69·MC_sd_g. Both anti-
conservative; bootstrap modestly better.

Best used as `bootstrap_paired_g` (asymmetric CIs) +
`cliff_delta_paired` (skew-robust point magnitude) layered
together on heavy-tailed Δ corpora.

Future extensions: BCa (bias-corrected accelerated) bootstrap
or studentized-t bootstrap would close the remaining
anti-conservativeness gap; deferred to follow-up work when the
need is concrete.

### `reml_random_effects_summary` — NOT YET IMPLEMENTED

REML τ² estimator (the literature consensus alternative to DL
for small-G meta-analyses; better bias and coverage at G ≤ 10).
~250 LoC including the iterative MLE machinery; deferred until
a substrate use case demands it. The DL probe and ROBUSTNESS.md
section above document the gap.
