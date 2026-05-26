# Dynamic mediation analysis — design doc

**Scope**: extend the corroborate framework with first-class
*trajectory-resolved* mediation analysis, so the framework can
detect and refuse mediation claims that look strong in
burst-pooled aggregate but are structurally meaningless at the
per-burst level. Sibling to [[SUBSTRATE_PIPELINE_GOTCHAS.md]] —
not a fix-now task, hand off to a dedicated worktree.

## Empirical motivation

The DDQN case study uses `partial_spearman` and the Fisher-z
burst-pooled per-burst variant to estimate cross-env
bg-mediation %. Per-burst time-resolved analysis on 4 envs
(2026-05-26, [[findings-per-burst-mediation-trajectory]])
revealed three distinct pathologies that burst-pooled aggregate
hides:

1. **Sign-flipping marginal** (PacMan γ=0.99): the marginal
   correlation ρ(arm, outcome) flips sign from −0.3 (D harms
   early) to +0.3 (D helps late) at ~0.5M training steps. The
   Fisher-z burst-pool of this trajectory produces 73%
   "mediation" through bg, but per-burst the partials TRACK the
   marginal — there's no per-burst mediation. The aggregate is a
   sign-flip aggregation artifact, not a causal signal.

2. **Learning-shoulder peak then fade** (MetaMaze γ=0.99): real
   per-burst mediation through bg + rate is concentrated in the
   middle of training (0.4–0.6M steps), then both marginal and
   partials decay as both arms converge. The aggregate 88% is
   a convolution of "strong mediation when learning is
   happening" with "no effect to mediate at convergence."

3. **Factor-substitution within an aggregate** (Asterix γ=0.99,
   Q-magnitude scaling regime): the aggregate "31% bg-mediation"
   is carried entirely by the cond_gap component; rate
   contributes ~0% and bg-aggregated-mediator confounds the two.
   The aggregator doesn't surface which axis is doing the work.

These aren't bugs; they're structural features of any analysis
where (a) the mediator-outcome relationship is non-monotonic
over time, (b) treatment dose or its effect varies through
training, or (c) the aggregate mediator is itself a product of
independently varying sub-axes.

## What aggregate mediation hides

The framework currently treats mediation as a scalar fact about
a (treatment, mediator, outcome) triple on a fixed dataset. For
RL substrates where the cells are training trajectories with
intrinsic time evolution, this is the wrong granularity:

| aggregate reports | what it hides |
|---|---|
| 73% bg-mediation (PacMan) | marginal sign-flips at 0.5M; no per-burst mediation |
| 88% bg-mediation (MetaMaze) | mediation peaks mid-training, fades to ≈0 by end |
| 31% bg-mediation (Asterix) | carried by cond_gap, NOT rate |
| any pooled mediation % | direction, monotonicity, where-in-training |

The pathology has analogs in adjacent fields (see audit below),
where it's known under names like "qualitative interaction,"
"effect modification by aggregating variable," and "Simpson's
paradox under temporal aggregation."

## Proposed framework primitive

### A. `DynamicMediationResult` typed dataclass

Parallel to existing `PartialSpearmanResult`, but trajectory-
shaped:

```python
@dataclass(frozen=True, slots=True)
class DynamicMediationResult:
    """Trajectory-resolved partial Spearman mediation.

    Per-burst ρ(arm, outcome | mediator) computed at each of
    `n_bursts` training-time slices, with diagnostic enum
    flagging aggregation pathologies.
    """
    # Trajectory data
    burst_steps: tuple[int, ...]                    # length n_bursts
    rho_marginal: tuple[float, ...]                 # ρ(arm, outcome) per burst
    rho_partial: tuple[float, ...]                  # ρ(arm, outcome | M) per burst
    n_per_burst: tuple[int, ...]                    # cells used per burst

    # Aggregate (for compatibility, but flagged)
    rho_marginal_pooled: float                      # Fisher-z aggregate
    rho_partial_pooled: float                       # Fisher-z aggregate
    aggregation_status: TimeAggregationStatus       # see below

    # Provenance
    mediator_name: str
    outcome_name: str
    arm_field: str
```

### B. `TimeAggregationStatus` enum

```python
class TimeAggregationStatus(Enum):
    CONSISTENT_DIRECTION = auto()
    # All bursts' rho_marginal have the same sign; aggregate
    # is a coherent estimator of average effect.

    SIGN_FLIP_DETECTED = auto()
    # At least one burst has rho_marginal opposite sign to the
    # majority; pooled estimate is a Simpson's-paradox artifact.
    # PacMan γ=0.99 falls here.

    WEAK_TIME_VARYING = auto()
    # rho_marginal sign-consistent but |rho_marginal| varies
    # by >2× across bursts; aggregate hides where the effect
    # is concentrated.
    # MetaMaze γ=0.99 falls here.

    UNDERPOWERED_BURSTS = auto()
    # Per-burst n is too small to estimate ρ reliably; aggregate
    # is the only meaningful summary but can't be diagnosed.
```

The enum is the trajectory analogue of the framework's existing
`LinearityStatus` on `mediation_dowhy` results
(`RELIABLE / SIGN_FLIPPED / OUT_OF_BOUNDS / UNIDENTIFIED /
POWER_INSUFFICIENT`). The discipline is the same — surface
identification pathologies as first-class enum values rather
than runtime gotchas.

### C. Analysis primitive `dynamic_partial_spearman`

```python
@analysis
def dynamic_partial_spearman(
    panel: pl.DataFrame,
    x: str,                 # treatment / arm field
    y: NDArray,             # outcome per-burst (n_cells × n_bursts)
    z: NDArray,             # mediator per-burst (same shape)
    stratify_by: tuple[str, ...] = ('env_name', 'gamma'),
    aggregator: Aggregator = Aggregator.FISHER_Z_WITH_DIAGNOSTIC,
) -> dict[Stratum, DynamicMediationResult]:
    """Per-stratum dynamic mediation. Each result carries
    trajectory + aggregation status."""
```

`Aggregator` controls how (and whether) pooled summaries are
produced:
- `FISHER_Z_WITH_DIAGNOSTIC` — pool but flag if SIGN_FLIP or
  WEAK_TIME_VARYING.
- `FISHER_Z_REFUSE_ON_SIGN_FLIP` — return NaN aggregate when
  sign-flip detected; force consumers to use the trajectory.
- `TRAJECTORY_ONLY` — don't pool at all.

### C.1 Aggregator: DerSimonian-Laird random-effects

The fixed-effects Fisher-z pool (n-weighted average of
per-burst atanh(ρ_b)) is the natural aggregate when bursts
behave as i.i.d. draws from a single population ρ. The
trajectory-resolved primitives explicitly contradict that
assumption — bursts are NOT exchangeable; they're indexed by
training time, and the structural pathologies (sign-flip,
weak-time-varying, learning-shoulder peaks) make the burst-
level dispersion the LOAD-BEARING signal, not noise to be
averaged out.

The DerSimonian-Laird (DL) random-effects pool is the
methodologically correct aggregator for this data shape:

1. **Burst-level (ρ_b, n_b) → (z_b, SE_z_b)**: for each burst,
   `z_b = atanh(ρ_b)` and `SE_z_b = 1 / sqrt(n_b − df_offset)`
   (df=3 for marginal Spearman, df=4 for closed-form first-
   order partial — same convention as `fisher_z_pool`).

2. **DL formula** (reused from `stats.effect_size.
   random_effects_summary`):
   ```
   w_fixed_b = 1 / SE_z_b² = n_b − df_offset
   z_fixed   = Σ w_fixed · z / Σ w_fixed       # FE pool
   Q         = Σ w_fixed · (z − z_fixed)²       # Cochran's Q
   c         = Σ w_fixed − Σ w_fixed² / Σ w_fixed
   τ̂²        = max(0, (Q − (G − 1)) / c)        # DL estimator
   w_rand_b  = 1 / (SE_z_b² + τ̂²)
   z_pooled  = Σ w_rand · z / Σ w_rand          # RE pool
   ```

3. **Heterogeneity statistics**:
   - **τ̂²** — between-burst variance in z-units. Zero (modulo
     small-G clip bias) under planted-constant ρ; large under
     sign-flip / phase-transition trajectories.
   - **I² = max(0, (Q − (G − 1)) / Q)** — Higgins' fraction of
     total variance attributable to between-burst heterogeneity.
   - **HTS PI**: 95% Higgins-Thompson-Spiegelhalter prediction
     interval `z_pooled ± t_{G−2, 0.975} · sqrt(τ̂² + Var(z_pooled))`.
     Inverse-Fisher-z'd back to ρ-units. NaN at G < 3.

4. **Inverse Fisher-z back to ρ-units**: pooled estimate
   `ρ_pooled = tanh(z_pooled)`; PI bounds
   `ρ_pi_lo = tanh(pi_lo)`, `ρ_pi_hi = tanh(pi_hi)`. The
   `se_pooled` stays in z-units (its natural scale; ρ-unit SE
   requires the delta-method `(1 − ρ²) · SE_z` at a specific
   ρ).

**Why DL alongside FE rather than instead of**: the FE pool
remains useful as a SHARP point estimate under CONSISTENT_DIRECTION
(its n-weighting is variance-optimal when τ² = 0). Under the
diagnostic pathologies (SIGN_FLIP, WEAK_TIME_VARYING) the FE
pool is structurally suspect (NaN'd by the diagnostic gate
under SIGN_FLIP). The DL pool is **never** NaN'd by the
diagnostic gate — its τ̂²/I² ARE the quantitative signal of
the same heterogeneity the enum flags qualitatively. Consumers
read both: FE for the magnitude under a coherent trajectory,
DL τ̂²/I² for the heterogeneity diagnostic. Under SIGN_FLIP,
expect I² ≈ 1.0 (closed-form ≈ 0.97 at planted ρ trajectory
(+0.5, +0.5, −0.5) n=80) and large τ̂² (≈ 0.39).

**Typed surface**: `FisherZDLPool` frozen dataclass in
`analyses.dynamic_mediation._common`. Re-exported through the
package `__init__.py` for downstream consumption. Fields:
`rho_pooled`, `se_pooled` (z-units), `tau2`, `i2`, `q`,
`rho_pi_lo`, `rho_pi_hi` (ρ-units), `n_bursts_used`,
`assumption_violations` (passed through from the underlying
DL primitive's `PooledStats.assumption_violations` —
small-G regime warnings).

Both `DynamicMediationResult` and `DynamicPCResult` carry
`dl_marginal: FisherZDLPool` and `dl_partial: FisherZDLPool`.

**Small-G caveat**: `random_effects_summary`'s docstring +
the empirical probe at `tests/analytic/robustness/
test_dl_small_g_robustness.py` document DL's reliability gap
at G ≤ 5 (point estimate is approximately unbiased but
sampling SD is enormous; I² detection power fails below G=10).
For trajectory-resolved analyses on canonical 50-burst RL
training trajectories, G is large enough that DL is in its
well-calibrated regime; the `assumption_violations` field
surfaces the warning explicitly when G is small.

### C.2 Aggregator: cluster bootstrap

DL's prediction-interval bounds (`rho_pi_lo` / `rho_pi_hi` on
`FisherZDLPool`) are *parametric* — they assume per-burst
observations are independent. In trajectory data they are NOT:
bursts within one cell share network state, replay-buffer
contents, optimizer momentum, and dynamics. The within-cell
autocorrelation is the load-bearing structural feature of
training data — exactly the part DL's PI formula doesn't see.

The cluster bootstrap is the standard methodological fix.
Algorithm:

1. **Cell as resampling unit.** Each cell = one training
   trajectory = one independent unit. Bursts within a cell
   stay together (they're not independent, so we don't break
   them).
2. **Resample with replacement.** For each of `n_resamples`
   iterations, sample `n_cells` cell indices with replacement
   from the original panel (using
   `np.random.default_rng(seed)` for deterministic output).
3. **Recompute the pool.** Per resample, materialise the
   resampled panel's per-burst ρ trajectory (using the same
   `_gather_burst_b` + `_spearman_marginal` /
   `partial_spearman_rho` primitives the non-bootstrap path
   uses) and DL-pool to one bootstrap-replica ρ.
4. **Empirical percentile CI.** The [α/2, 1 − α/2] percentile
   range across replicas is the CI; the median is the point
   estimate (more robust than mean under asymmetric bootstrap
   distributions).

The cluster bootstrap is **assumption-free** under any
within-cell autocorrelation structure — resampling cells
preserves whatever dependence exists between bursts of the
same cell. This is the methodologically-correct CI shape for
publication-grade reports.

**Why bootstrap alongside DL rather than instead of**: DL's
τ²/I²/Q remain the canonical quantitative measures of
heterogeneity (the diagnostic enum
`TimeAggregationStatus` flags it qualitatively). The
bootstrap adds an honest empirical CI to the point estimate.
DL's PI is a *predictive* interval for a hypothetical new
burst's underlying parameter — useful, but a different
quantity from the *sampling* interval the bootstrap
estimates. The two pools answer different questions:

| pool | quantity | assumption |
|---|---|---|
| FE Fisher-z | n-weighted point estimate | bursts iid, identical population ρ |
| DL `rho_pooled` | RE point estimate | bursts iid, ρ varies between bursts |
| DL `rho_pi_lo` / `rho_pi_hi` | predictive interval for a new burst's ρ | bursts iid (independent draws from heterogeneous pop) |
| Bootstrap CI | sampling CI on the DL pool point estimate | cells iid (within-cell dependence preserved) |

**Typed surface**: `ClusterBootstrapInterval` frozen
dataclass in `analyses.dynamic_mediation._common`. Fields:
`rho_lower`, `rho_upper`, `rho_median`, `n_resamples`,
`alpha`, `seed`. Both `DynamicMediationResult` and
`DynamicPCResult` carry `bootstrap_marginal:
ClusterBootstrapInterval | None` and `bootstrap_partial:
ClusterBootstrapInterval | None` + `n_bootstrap: int`. The
fields are `None` when `n_bootstrap == 0` (the default fast
path); populated when `n_bootstrap > 0`. `n_bootstrap=1000`
is the recommended publication-grade value (CI bound
percentile-estimator sampling SD ~3% at G_bootstrap=1000); the
default 0 keeps the non-bootstrap code path bit-identical to
the pre-bootstrap behaviour for downstream consumers that
don't need the CI.

**Reference**: Pustejovsky & Tipton (2022) on CHE/RVE; Deen &
de Rooij (2020) on ClusterBootstrap. The cluster bootstrap
generalises the i.i.d. bootstrap by respecting cluster
structure (Davison & Hinkley 1997 §3.8, "blocked bootstrap"
in the time-series literature).

### C.3 Bootstrap on count outputs

The PC primitive's per-burst edge classification produces an
INTEGER count triple per stratum:
`(n_bursts_marginal_edge, n_bursts_mediator_dseparates,
n_bursts_direct_edge)`. As descriptive statistics these are
fine, but they admit a natural inferential question: **is the
classification robust to which cells we sampled?** A 6/32 dsep
fraction looks different if (a) 6 bursts robustly d-separate
across any subset of cells vs (b) one outlier cell drives the
per-burst CI decision at 5 of the 6 bursts. The fragile
verdict is a publication-risk shape that the count alone hides.

`ClusterBootstrapEdgeCounts` in `_common.py` is the typed
surface for the bootstrap-on-counts. Same cell-resampling
pattern as `_cluster_bootstrap_pool`; the inner computation
differs (recompute per-burst CI decisions + sum to a triple,
vs DL-pool the per-burst ρ trajectory). For each replica we
call the SAME `_spearman_marginal` + `partial_spearman_rho`
primitives the non-bootstrap path uses — guarantees the
bootstrap distribution centres on the original count by
construction. Empirical [α/2, 1 − α/2] percentiles of each
count separately give the CI triple; the median is the robust
integer point estimate.

Field semantics on `ClusterBootstrapEdgeCounts`:
`(marg_lower, marg_median, marg_upper)` /
`(dsep_lower, dsep_median, dsep_upper)` /
`(direct_lower, direct_median, direct_upper)` — integer-typed
throughout, with the percentile rounded to the nearest
integer (the percentile interpolation can land between
integer counts). Provenance: `n_resamples`, `alpha`, `seed`
mirror `ClusterBootstrapInterval`.

**Why a separate dataclass from `ClusterBootstrapInterval`**:
the two answer structurally distinct questions:

| field | question | type |
|---|---|---|
| `ClusterBootstrapInterval.rho_lower/upper` | what's the average effect magnitude under bootstrap resampling? | continuous ρ |
| `ClusterBootstrapEdgeCounts.dsep_lower/upper` | is the edge classification robust to which cells we sampled? | integer count |

Keeping them typed-separately surfaces the question shape at
the consumer site (`isinstance(x, ClusterBootstrapInterval)`
narrows on intent). Both populate together when
`n_bootstrap > 0`; consumers that want only the count CI pay
the ρ-pool cost too (acceptable — the bootstrap iterations
dominate runtime, not the inner reductions).

`dynamic_partial_spearman` does NOT get a parallel
`bootstrap_edge_counts` because it has no integer count
outputs — its trajectory is continuous ρ, fully covered by
the `ClusterBootstrapInterval` on the ρ-pool.

### D. Multi-mediator depth-≥2 conditioning

The depth-1 design above (one mediator) generalises directly to
depth-k joint conditioning. Both primitives accept
`mediator_per_burst: str | Measurable | tuple[..., ...]`,
parallel to the static `partial_spearman`'s `conditioning`
parameter:

- Single `str | Measurable` → k=1. Internal dispatch uses the
  closed-form `partial_spearman_rho` (Fisher-z df = n − 4) for
  bit-exact compatibility with the depth-1 primitive.
- Tuple of length k (`(z1, z2, ...)`) → k-mediator joint
  conditioning. Internal dispatch uses `partial_spearman_rho_multi`
  (Fisher-z df = n − 3 − k) — same OLS-residual primitive the
  static `partial_spearman` uses at k≥2.
- Empty tuple `()` raises `ValueError`. The marginal test (no
  conditioning) is already reported via `rho_marginal[b]` /
  `p_marginal[b]`; a silently-no-op invocation would mask a
  bridge-author bug.

Result-type changes (both primitives):

- `mediator_name: str` → `mediator_names: tuple[str, ...]`.
  Length-1 tuple at k=1 (back-compat for the singular shape);
  length-k tuple at k≥2.
- `k_conditioning` property reads `len(mediator_names)`. Used
  by consumers that need to gate on conditioning depth.

Edge-count semantics at depth ≥2 (`dynamic_pc_adjacency`): the
`n_bursts_mediator_dseparates` count's interpretation
generalises from "this one mediator d-separates arm from
outcome at burst b" (k=1) to "the JOINT mediator set
d-separates arm from outcome at burst b" (k≥2). Same
machinery, broader conditioning set.

`df_offset` accounting throughout (DL pool, FE pool, cluster
bootstrap) becomes `3 + k`:
- k=1 → df_offset=4 (matches the existing depth-1 path).
- k=2 → df_offset=5.
- k≥2 generally → df_offset = 3 + k.

Bootstrap CIs at depth-k: same cell-resampling pattern; each
replica recomputes per-burst ρ via the depth-k CI primitive.
`bootstrap_edge_counts` at depth-k inherits the depth-1
semantics (the count CIs answer "is the edge classification
robust to which cells we sampled?" — same question, broader
conditioning set).

Use cases:

- **Joint mediator analysis** — when no single mediator
  d-separates but a small set jointly does (e.g., `bg_magnitude`
  + `argmax_entropy` together capture the chain-amplification
  channel that neither alone does).
- **M1 rate × cond_gap decomposition** — conditioning on both
  the per-step DDQN clip rate AND the magnitude of the
  conditional gap to disentangle "clip fires often but mildly"
  from "clip fires rarely but sharply."
- **Higher-order PC search** — once a depth-1 mediator survives
  the framework's CI tests at α, depth-2 searches the natural
  joint sets including it. Currently authored manually by
  bridges; future framework-level PC over per-burst trajectories
  would use the same primitive.

### E. Bridge consumers

Existing bridges consume `PartialSpearmanResult.rho` and
`.p_value`. New bridges that use `DynamicMediationResult` would
have access to:
- `.rho_partial_pooled` (with risk of artifact, gated by status)
- `.aggregation_status` (used for verdict gating: a HELD
  mediation claim with SIGN_FLIP status should be downgraded to
  UNDERPOWERED or NO_EFFECT)
- `.rho_partial` (full trajectory) for fine-grained tests

Concrete first bridge:

```python
@claim_bridge
def mediation_is_stable_across_training(
    dynamic_med: DynamicMediationResult, *, min_burst_fraction: float = 0.7,
) -> Verdict:
    """Verdict: HELD iff (a) status is CONSISTENT_DIRECTION,
    (b) at least min_burst_fraction of bursts have
    |rho_partial| < |rho_marginal| × (1 - threshold), AND (c)
    aggregate rho_partial_pooled p-value is significant.
    Else NO_EFFECT or UNDERPOWERED."""
```

The point isn't this specific bridge; it's that the typed
trajectory result enables bridges to ask trajectory-shaped
questions.

### F. Backwards-compatibility with current bridges

Existing bridges that consume `PartialSpearmanResult` shouldn't
break. Two options:
1. Add a `.to_partial_spearman_result()` adapter on
   `DynamicMediationResult` that returns the pooled aggregate +
   raises `AggregationArtifactWarning` if status is
   SIGN_FLIP_DETECTED.
2. Mark the existing `partial_spearman` analysis as DEPRECATED
   on per-burst data and route through `dynamic_partial_spearman`
   transparently.

Probably (1) — preserves existing analyses, surfaces the
diagnostic only where it matters.

## RL literature audit

### Direct hits (within RL)

1. **Ge, Tsiakas, Murphy (2023) — "A Reinforcement Learning
   Framework for Dynamic Mediation Analysis"** ([PMLR 2023](https://proceedings.mlr.press/v202/ge23a.html), [arXiv 2301.13348](https://arxiv.org/abs/2301.13348)).
   Uses RL formalism to estimate mediation effects in infinite-
   horizon decision processes. Defines:
   - Immediate direct effect (IDE)
   - Immediate mediation effect (IME)
   - Delayed direct effect (DDE)
   - Delayed mediation effect (DME)

   They go the OPPOSITE direction from us — applying RL
   formalism to *estimate* dynamic mediation, where we want to
   *analyze* RL training trajectories with dynamic mediation.
   The four-way decomposition is directly relevant: our PacMan
   "marginal sign-flip" is structurally a DDE/DME conflict
   (delayed effects in opposite direction from immediate).

2. **Lan & Luo (2025, Annals of Statistics) — "Multivariate
   Dynamic Mediation Analysis under a Reinforcement Learning
   Framework"** ([arXiv 2310.16203](https://arxiv.org/abs/2310.16203)).
   Markov Mediation Process (MMP) + time-varying SEM. Decomposes
   ATE into:
   - immediate direct + immediate mediation
   - delayed direct + delayed mediation

   Methodology applies when mediators are multivariate +
   conditionally dependent. Our (rate, cond_gap, bg) trio is
   exactly this case — they're algebraically dependent
   (bg = rate × cond_gap structurally), but we treated them as
   independent mediators. The MMP framework would surface this
   dependence.

3. **Lyle et al. (ICML 2022) — "Learning Dynamics and
   Generalization in Deep RL"** ([PMLR](https://proceedings.mlr.press/v162/lyle22a/lyle22a.pdf)).
   Per-iteration analysis of update rank, gradient magnitude,
   and parameter divergence in DQN. They identify regime
   transitions during training (e.g., update rank rises after
   ~50 iterations). The methodology (track quantities
   per-iteration, not just at convergence) is exactly what our
   per-burst analysis does. They don't frame as mediation
   though.

4. **Iyer, Cobbe, Rosenberg (NeurIPS 2024) — "Investigating the
   Edge of Stability Phenomenon in RL"** ([arXiv 2307.04210](https://arxiv.org/pdf/2307.04210)).
   Reports phase-transition behavior in DQN training (e.g.,
   sharp loss-landscape changes). Identifies that DQN with
   Huber loss shows edge-of-stability while C51 doesn't —
   training-regime-conditional effects are real and replicable.
   No mediation framework but the per-iteration regime detection
   is congruent with our finding that mediation is regime-
   conditional.

5. **Various "policy churn" papers (Schaul et al. 2022,
   NeurIPS, arXiv 2206.00730)**. Track policy-instability
   per-step throughout training. Our per-burst argmax-disagree
   rate is the "between online and target nets, same step"
   variant of policy churn (Schaul's churn is "between snapshots
   in time"). Closely related methodology.

### Adjacent (RL but not mediation)

6. **Anschel et al. (2017) — Averaged-DQN**. Tracks Q-estimate
   variance over training. Per-iteration analysis, no causal
   mediation framing.

7. **Bjorck et al. (2021) — "Towards Deeper Deep RL" + others
   on plasticity loss**. Plasticity loss is itself a time-
   varying quantity that mediates outcome. Recent literature
   explicitly tracks per-iteration plasticity metrics; few
   papers formalize the mediation question.

8. **Ostrovski et al. (2021) — "The Difficulty of Passive
   Learning in Deep RL"**. Distinguishes representation
   degradation across training phases; structurally similar to
   our regime-conditional mechanism.

### Conclusion of audit

There's a **specific gap**: the statistical-mediation
literature (Ge 2023, Lan 2025) provides formal estimators for
dynamic mediation but is designed for short-horizon clinical /
behavioral RL applications. The deep-RL training-dynamics
literature (Lyle, Iyer, Schaul) provides per-iteration
mechanism-tracking but doesn't use mediation formalism. **Our
per-burst-mediation work sits in between** — applying the
formal mediation discipline to the deep-RL training-trajectory
setting where the statistical literature doesn't directly
apply (different infinite-horizon setup, different identification
assumptions).

The cleanest published analog to what we did is **Schaul 2022's
"policy churn" trajectory analysis** — same per-step
methodology, related quantity, but no formal mediation
framework.

## Cross-disciplinary analogs (brief)

Pharmacokinetics / PK-PD modeling, longitudinal epidemiology
(Robins g-methods + VanderWeele 2015), cross-lagged SEM in
psychology, dose-response mediation in econometrics
(Heckman & Pinto 2015), and time-varying climate sensitivity
all face structurally similar problems. The terminology
("qualitative interaction," "effect modification," "time-varying
confounding," "lagged mediation") differs. The methodological
discipline that's most directly transferable is **Robins' g-
methods** for time-varying mediation under time-varying
confounding — but applying it to RL training requires careful
identification-assumption work.

## Implementation plan

Worktree-scoped, ~4-8 hours focused work:

1. **Define typed primitives** (`DynamicMediationResult`,
   `TimeAggregationStatus`, `Aggregator` enum) in
   `corroborate/analyses/dynamic_mediation/`. ~50 lines.

2. **Author `dynamic_partial_spearman` @analysis primitive**
   parallel to existing `partial_spearman`. Reuses the closed-
   form first-order partial Spearman from
   `corroborate.graph.discovery` per burst. ~80 lines.

3. **Diagnostic logic for `TimeAggregationStatus`**: sign-flip
   detection, weak-time-varying detection, power thresholds.
   ~40 lines + closed-form test cases.

4. **Closed-form analytic test** in
   `tests/analytic/lg_scm/` — construct an LG-SCM substrate
   where the per-burst mediation strength is KNOWN, verify the
   primitive recovers the trajectory and flags pathologies
   correctly. ~60 lines + the LG-SCM extension to support
   per-burst data shape (task #17 was already pending for
   per-burst LG-SCM tests).

5. **Adapter on `DynamicMediationResult`** for backwards-
   compatible consumption by existing `PartialSpearmanResult`
   bridges; raises `AggregationArtifactWarning` when
   sign-flip detected. ~20 lines.

6. **Migrate hasselt_clean B4-equivalent bridge** to consume
   `dynamic_partial_spearman` instead of pooled. Verify the
   regen produces correct verdicts on the canonical cache.

7. **Documentation** in CLAUDE.md "Canonical analyses" section.
   Promote `dynamic_partial_spearman` to the recommended
   primitive for any analysis on `_per_burst` measurables.

## Open questions

- **Joint vs marginal trajectory**: should the primitive
  estimate `ρ(arm, outcome(t) | M(t))` (contemporaneous) or
  `ρ(arm, outcome(t+k) | M(t))` (lagged)? Lagged mediation
  (Lan 2025 DME) requires choosing k; contemporaneous is what
  we already do. Probably support both with a `lag: int =
  0` parameter.

- **Multivariate joint conditioning**: when rate + cond_gap
  + bg are all on the table, the proper analysis conditions
  on a joint mediator vector. The existing
  `partial_spearman_rho_multi` (multi-Z OLS-residual) handles
  this in the static case; the dynamic analogue is needed.

- **What's the right aggregate**? Fisher-z weighted by `n_per_burst`
  is one option. Lan 2025's MMP-based estimator is more
  principled under the infinite-horizon assumption but
  requires identification assumptions we may not satisfy.
  The framework's discipline is to surface the choice as a
  typed `Aggregator` enum so consumers know what they're
  getting.

- **Edge cases for `WEAK_TIME_VARYING`**: how to set the "by
  >2× variation" threshold? Empirically on our cache.

- **Connection to `PartialSpearmanResult.linearity_status`**:
  the existing static linearity status flags non-monotone
  mediator-outcome relationships in a static sample. The
  time-varying version would extend this to "monotone within
  burst but flips direction across bursts." Pair the enums.

## Status

Empirical motivation: [[findings-per-burst-mediation-trajectory]]
+ [[findings-m1-per-burst-pacman-artifact]] +
[[findings-m1-rate-vs-magnitude-decomposition]].

This doc: design only. Implementation deferred to a focused
worktree session.

Related design debt: [[SUBSTRATE_PIPELINE_GOTCHAS.md]] — the
substrate-pipeline-correctness counterpart. Both express the
"runtime-convention → typed-primitive" upgrade pattern that
the framework's discipline demands but the substrate layer has
been slower to adopt.
