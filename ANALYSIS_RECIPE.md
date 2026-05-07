# Post-sweep analysis recipe

The framework specifies *what a verdict is* (bridge holds_when
threshold over typed Hedges' g + per-group + pooled stats from
`paired_comparison`), and *what a sweep produces* (paired cells
emitted by `run_intervention`). This doc fills the gap between
those two: **given a corpus, what analyses do you run, in what
order, with what robustness checks?**

This is **guidance**, not API contract. The framework's analyses
in `corroborate.analyses.*` are the canonical primitives; this
recipe stitches them.

---

## 0. Pre-flight: classify cells by outcome status

Before any analysis, augment the corpus DataFrame with a per-cell
solve-status column. Substrate-coupled because the *threshold
table* and `is_solved` predicate are substrate-specific (the RL
substrate provides them via `corroborate_rl.env_solve_thresholds`
+ `corroborate_rl.cell_classification`).

The 4-class taxonomy:

- **saturated** — both arms reach the same per-env corpus-max;
  paired g is structurally null. *Exclude these from link
  analyses.* Outcome ceiling masks signal.
- **solved** — outcome crossed the env's canonical threshold
  WITH headroom (not at corpus-max).
- **unsolved** — outcome below threshold.
- **no_threshold** — env has no canonical solve threshold.

```python
from corroborate_rl.cell_classification import with_cell_class

df = pl.read_parquet(corpus_path)
df = with_cell_class(df, outcome_path='eval_best_burst_mean')
# `df['_cell_class']` carries the per-cell label.
```

Bridges that authored a `scope` predicate can subset by
class directly: `scope=pl.col('_cell_class') == 'solved'`.

---

## 1. Mechanism / outcome / link verdicts via bridges

The framework's verdict primitive is `bridge.evaluate(b, cells)`.
Author one bridge per causal chain edge and run them via
`runner.run(<module_name>)`:

```python
@claim_bridge(
    source=INTERVENTION,                # DoEffect
    target='jensen_gap',                # mechanism path
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    predicted_direction='a_lt_b',
    pair_by=('seed',),
)
def ddqn_reduces_jensen_gap(
    paired_g: PairedGResult,
) -> Verdict:
    if paired_g.g < -0.3 and paired_g.p_value < 0.05:
        return Verdict.HELD
    if paired_g.n_pairs < 30:
        return Verdict.POWER_INSUFFICIENT
    return Verdict.NO_EFFECT
```

**Mech / outcome / link separation discipline (per CLAUDE.md):**
the three verdicts MUST be authored independently. Never collapse
"link null" with "mechanism dormant" — different verdicts.

**File-level exclusions go on `MODULE_SCOPE`** at the hypothesis
module (e.g., `MODULE_SCOPE = ~pl.col('env_name').str.ends_with(
'-bsuite')`); per-bridge `scope=` is for claim-specific filters.
The runner reads `MODULE_SCOPE` via `getattr(h, ..., None)` and
AND-combines into every bridge's scope at evaluation time. Use
when the entire hypothesis universe excludes a class of cells
(diagnostic envs, abnormal config corpora, etc.).

Run:
```bash
uv run scripts/run_hypothesis.py <module_path> --data <corpus_dir>
```

Reports per-bridge `BridgeEvaluation` with the `holds_when`-
authored Verdict.

---

## 2. Stratified meta-regression for scope discovery

When the pooled verdict is `HELD_WITH_SCOPE_FLAG` (heterogeneous
corroboration, high I²), the question becomes *which covariate
predicts the per-stratum effect?* — the empirical scope.

```python
from corroborate.analyses.meta_regression_paired_g import (
    meta_regression_paired_g,
)

result = meta_regression_paired_g.fn(
    cells, treatment_arm=..., baseline_arm=...,
    source='jensen_gap', covariates_per_stratum={
        env: env_features(env) for env in envs
    },
    pair_by=('seed',),
)
# `result.cleavage_axes` lists covariates whose CI excludes 0.
```

Each significant coefficient is a numeric scope claim:
"mechanism activates where covariate X exceeds threshold T."
Author the threshold as a `Bridge.scope` for the next
sweep.

**Per-env vs per-cell weighting matters for cross-env claims.**
`meta_regression_paired_g` (one paired-g per env, equal
weighting) is the right default. Per-cell forms
(`meta_regression_per_burst` cell-weighted) silently amplify
the env with the most cells (FourRooms = 27% of cells in the
DDQN universe corpus) — apparent cross-env signals can be
single-env-domination artifacts. Cf. `findings_per_env_vs_per_
cell_weighting`.

---

## 3. PC discovery for moderation candidates

Stratified conservative-PC over a candidate variable set
identifies which mediators / env features separate from the
outcome conditional on others. Use as a **moderator candidate
proposer**, not a verdict — PC suggests; bridges verify.

```python
from corroborate.graph.discovery import discover_adjacency

adj = discover_adjacency(
    df, variables=PC_VARIABLES,
    alpha=0.05, max_conditioning=1,
    stratify_by='env_name',  # JCI: don't pool across envs
)
# `adj.edges` lists pairs that survive the conditional-independence
# pruning. Outcome's neighbours are the moderator candidates.
```

**JCI note:** stratify by env when env-level features (action_dim,
horizon, reward_scale) are part of the variable set. Pooling
across envs without stratification produces spurious edges
(per memory `findings_dowhy_three_probes`).

### 3a. Mediation analysis: linear (default) vs counterfactual

The framework's mediation primitives operate on the per-pair Δ
form. Two distinct decompositions, each appropriate to a
different bridge claim shape.

#### Linear mediation — proportion of effect mediated

For a (treatment, baseline) contrast and a candidate mediator,
**`proportion_mediated`** decomposes per-pair Δ_target into
direct + indirect components:

```python
from corroborate.analyses.proportion_mediated import proportion_mediated

result = proportion_mediated.fn(
    cells, target='eval_best_burst_mean', mediator='jensen_gap',
    treatment_arm=..., baseline_arm=..., pair_by=('seed',),
)
# result.proportion ∈ [0, 1] under linear-mediation assumptions:
#   1.0 = fully mediated (Δ_Y carried entirely by β·Δ_M)
#   0.0 = no mediation (direct effect only)
# result.in_unit_interval=False signals assumption failure.
```

The Spearman-rank companion is `partial_spearman_rho` (and its
JCI form `stratified_partial_spearman_rho`):

```python
from corroborate.graph.discovery import partial_spearman_rho

# Δ_outcome ⊥ Δ_candidate | Δ_jensen_gap?
rho, p = partial_spearman_rho(
    delta_outcome, delta_candidate, delta_jensen_gap,
)
# rho near zero → candidate is collinear with jensen_gap
#   (restatement, not a separate mediator).
# rho significantly non-zero → candidate carries independent
#   signal — the partial-correlation analog of "direct effect"
#   under the same linear-mediation assumptions.
```

#### When linear mediation is enough

The linear decomposition assumes:

1. No treatment × mediator interaction.
2. Linear M → Y functional form.
3. Mediator's distribution doesn't depend on treatment in
   nonlinear ways.

Three diagnostic signals that linear has broken:

- **`proportion ∉ [0, 1]`** — `in_unit_interval=False`. Suppressor
  (proportion < 0) or overshoot (> 1) indicates the additive
  decomposition is unstable.
- **Per-stratum partial-ρ heterogeneity.** Bin the data by
  mediator level and run `stratified_partial_spearman_rho`. If
  per-stratum ρ varies materially across bins, the partial-r
  isn't a single direct effect — there's a treatment×mediator
  interaction.
- **Nonlinear M → Y functional form.** Scatter Δ_M vs Δ_Y; if
  LOESS RMSE materially beats linear-fit RMSE on the residuals,
  the linear decomposition is misestimating.

When NONE of these fire, linear is the right primitive and
counterfactual decomposition won't tell you anything new.

#### When to escalate to counterfactual mediation (Pearl NDE/NIE)

Pearl's natural-direct / natural-indirect effects re-simulate
the mediator's distribution under the counterfactual treatment
— different math from partial-correlation, identifies treatment×
mediator interactions and nonlinearity. Use when:

- A diagnostic above fires positive AND the bridge's claim
  cares about the decomposition (proportion-mediated is
  load-bearing for the verdict).
- The scientific question is specifically about manipulability:
  "what would happen if we forced the mediator to value M\*?"
- Multiple mediators interact (linear decomposition assumes
  additive paths).

The framework does NOT yet ship a counterfactual mediation
primitive. The DoWhy substrate (`corroborate.analyses.dowhy`)
exposes backdoor + refutation primitives that are the ingredients
for one. Authoring is deferred per FUTURE_WORKS.md; the lift gate
is "a bridge's diagnostic fires positive on real data."

#### JCI (joint causal inference) — the canonical env adjustment

Both `partial_spearman_rho` and `proportion_mediated` admit a
**stratified** form when env-level heterogeneity is present:

```python
from corroborate.graph.discovery import stratified_partial_spearman_rho

# Per-env partial Spearman, Fisher-z-pooled — the JCI form.
rho_pooled, p = stratified_partial_spearman_rho(
    deltas_y, deltas_candidate, deltas_canonical, env_strata,
)
```

JCI is the canonical adjustment for the framework's "pool across
envs" question (memory `findings_dowhy_three_probes`). When env-
level features (action_dim, horizon, reward_scale) are part of
the analysis, stratify; otherwise pooled correlations introduce
spurious edges.

---

## 4. Robustness checks

### 4a. PC depth-2 robustness

Re-run `discover_adjacency` with `max_conditioning=2`; any edge
that vanishes is depth-1-fragile (a confounded by some 2-variable
condition).

```python
adj_d1 = discover_adjacency(df, variables=..., max_conditioning=1)
adj_d2 = discover_adjacency(df, variables=..., max_conditioning=2)
fragile = adj_d1.edges - adj_d2.edges
```

### 4b. K-fold cross-validation of meta-regression

```python
from corroborate.stats.meta_regression import (
    cross_validate_meta_regression,
)

cv = cross_validate_meta_regression(observations, k_folds=5)
# `cv.sign_consistency['<covariate>']` ∈ [0, 1]: fraction of folds
# where the coefficient's sign agrees with modal. < 0.7 → fragile.
```

A coefficient with `sign_consistency = 1.0` voted the same sign on
every fold — that's the robustness signal. Sign consistency below
~0.7 means the cleavage axis depends on which strata are in the
training set; the scope claim is fragile.

### 4c. Drop-one-env / drop-one-stratum sensitivity

Cross-env regressions can be single-env-leveraged: one outlier
env (typically a bandit / tail env structurally different from
the rest) anchors the slope. Drop one env at a time and re-fit;
if the coefficient flips sign or loses significance after
removing a single env, the cross-env claim is leverage-driven,
not population-level. Apply after k-fold CV — k-fold randomly
splits seeds, drop-one-env tests structural sensitivity to the
strata themselves. Cf. `findings_staleness_causal_inference`,
deletion of CLAIM 16's bf bridge.

---

## 5. Per-burst probes when scalar mediator returns null

Scalar paired_g over a trajectory-averaged mediator can wash
non-monotone phases (early bias correction vs late Q-explosion).
When scalar mech returns null, run **per-burst** probes:

```python
from corroborate.analyses.paired_g_per_burst import paired_g_per_burst
from corroborate.analyses.paired_link_per_burst import paired_link_per_burst

# Per-(env, burst) g panel:
per_burst = paired_g_per_burst.fn(cells, source=jensen_gap, ...)

# Per-burst Pearson r between mediator and outcome — the empirical
# link, panel-typed:
link_panel = paired_link_per_burst.fn(cells, ...)
```

The `phase_link_consistency` analysis derives a scalar from the
per-burst link panel: fraction of bursts where r is significantly
in the predicted direction.

Per memory `findings_fourrooms_time_series`, scalar mech-link
slopes silently combine causally opposite phases — per-burst
unmasks them.

**Per-burst plc as Q-stability scope predicate.** When phase
transitions exist (Q-explosion-prone envs: MinAtar sync=100,
high-γ Acrobot, etc.), pooled `q_divergence_score` *cannot*
capture per-burst Q-stability — it averages bounded-early
bursts with exploded-late ones, and the cell passes any
threshold. The "bounded Q regime where Hasselt's theorem
bites cleanly" is `phase_link_consistency >= floor` per env,
not `q_div < threshold` per cell. Cf. `findings_q_div_threshold_
too_loose`. (And note: q_div<100 is "extreme outliers excluded";
the *bounded* semantic per the docstring is q_div<1.)

### 5a. Cross-burst lag correlation (causal-precedence diagnostic)

When per-burst panels exist, the **temporal direction** of the
mech↔outcome relation is testable by cross-burst lag correlation:

```python
import scipy.stats as ss

# For each (env, pair), compute r(Δ_mediator[k], Δ_outcome[k+τ])
# pooled over (seed, k) for τ ∈ {-3, -2, -1, 0, +1, +2, +3}.
for tau in (-3, -2, -1, 0, 1, 2, 3):
    xs, ys = [], []
    for k in range(n_bursts):
        if 0 <= k + tau < n_bursts:
            xs.extend(delta_mediator[:, k])
            ys.extend(delta_outcome[:, k + tau])
    r = ss.pearsonr(xs, ys)
    print(f'τ={tau:>+2}: r={r.statistic:+.3f}')
```

**Forward asymmetry** (`r(τ=+1) > r(τ=-1)`) → mediator precedes
outcome → causal direction consistent. **Symmetric** lag profile
→ no temporal precedence; the relation may be confounded by a
common cause. Use as a **diagnostic** sanity check; PC + DoWhy
refutations remain the verdict primitives.

---

## 5b. Endogenous-discriminator search when narrow scope is empirically forced

When per-env analysis surfaces a single env where a claim holds
(or where the sign is opposite the others), the temptation is to
scope the bridge on `env_name`. Per `feedback_endogenous_scope_
predicates`, env-name scoping is lazy science. The right move is
finding the **endogenous predicate** that distinguishes the
holding-env regime from the others.

The discriminator-search loop:

1. **Per-env diagnostic panel**: compute candidate measurables
   per env (mean Q at late window, mean Δ_jens, σ Δ_outcome,
   r_max/r_min, mean baseline outcome, mech-firing rate, etc.).
2. **Find a feature that splits the held vs null envs cleanly.**
   Both env-structural (`r_max`, `r_min`, `n_actions`) and
   trajectory-derived (`q_late_mean`, `q_divergence_score`,
   `target_staleness_late`) are admissible — env-structural
   features are exogenous, trajectory features are endogenous.
3. **Prefer the endogenous downstream**: if `r_min ≥ 0`
   determines `q_late_mean > 0` per cell, scope on
   `q_late_mean > 0` (the per-cell measurable) rather than the
   exogenous structural property. This makes the bridge
   generalize to any env whose trajectory ends up in the
   target regime.

### Interaction-term test for regime-dependent effects

When the discriminator is found, verify it's not just
correlation-by-env via the **interaction-term regression**:

```python
import numpy as np

# Δ_outcome = β₀ + β_T·treatment + β_M·moderator
#               + β_int·(treatment × moderator) + ε
X = np.column_stack([np.ones(n), treatment, moderator,
                     treatment * moderator])
beta, _, rank, _ = np.linalg.lstsq(X, delta_outcome, rcond=None)
# β_int significantly nonzero ⇒ moderator's regime modulates
# treatment's effect direction. Sign tells you the polarity.
```

The interaction coefficient `β_int` directly tests "does the
treatment's effect depend on the moderator's regime?" If
significant, the discriminator IS modulating the effect causally,
not just labeling envs.

Pair with **per-stratum DoWhy backdoor + refutations**:

```python
for regime_label, mask in [('moderator > 0', cells_pos),
                            ('moderator < 0', cells_neg)]:
    r = backdoor_ate.fn(cells.filter(mask), ...)
    pl = placebo_refutation.fn(cells.filter(mask), ...)
    rcc = random_common_cause_refutation.fn(cells.filter(mask), ...)
```

Per-stratum ATEs that flip sign across regimes, both
refutation-validated, are direct rung-2 evidence for the
discriminator.

### When the discriminator is exogenous

`r_min`, `r_max`, `n_actions`, `env_name` are exogenous structural
features (set by the env's specification). They're not bad as
predicates but they don't generalize — a future env with novel
structure won't fit the existing buckets.

The fix is to find the **endogenous downstream**: the per-cell
trajectory feature that the exogenous structural property
*causes*. For r_min: `q_late_mean = mean(online_max_q over late
50%)` is the endogenous trajectory-side counterpart. The bridge
scopes on `q_late_mean > 0` (works on any future cell) instead
of `r_min ≥ 0` (works only on envs whose r_min is in the
catalogue).

A worked example lives in `experiments/findings/sync_curve_
breakout/polyak_q_regime_findings.md`: r_min causally determines
sign(q_late_mean), which determines the direction of Hasselt's
overestimation bias, which inverts ATE(stale → Δ_outcome). Bridge
`staleness_amplifies_ddqn_outcome__sparse_goal_polyak` scopes on
the endogenous `q_late_mean > 0`.

---

## 6. Tautology audit for cleavage candidates

Before publishing a cleavage axis, run the three-check audit
(`corroborate.analyses.tautology_audit`) to rule out:

- **HP shadow** — covariate is a near-monotone function of an HP
  set in the sweep.
- **Partial-correlation collapse** — covariate's marginal r
  vanishes after conditioning on a confound.
- **Convergence proxy** — covariate's effect loads on
  convergence-status only.

Surviving cleavage axes are scope claims; failed ones are
methodological artifacts.

---

## 7. Data-driven intervention selection

When the mech/outcome/link verdicts surface a scope ("DDQN works
on solved-converged envs but not unsolved"), the next question is
*"what should the next sweep target?"* The data-driven companion
to literature pattern-matching:

1. **Classify envs by convergence on the BASELINE arm** — we want
   the natural failure-mode signature, not one induced by an
   intervention.
   ```python
   from corroborate_rl.convergence import (
       classify_envs, envs_in_class, mediator_differential,
   )
   classes = classify_envs(baseline_runs)
   solved_envs = envs_in_class(classes, 'solved')
   unsolved_envs = envs_in_class(classes, 'unsolved')
   ```

2. **Mediator differential** — Hedges' g of each candidate
   mediator's value across solved-vs-unsolved baselines. Top-|g|
   mediators are the empirical failure-mode signatures.
   ```python
   diff = mediator_differential(
       baseline_runs, mediator_paths=MEDIATOR_PATHS,
       solved_envs=solved_envs, unsolved_envs=unsolved_envs,
   )
   ```

3. **PC adjacency on the panel** — for each top-|g| mediator,
   list its neighbours on the conservative-PC graph (depth-1,
   stratified by env).
   ```python
   adj = discover_adjacency(
       df, variables=PC_VARIABLES,
       alpha=0.05, max_conditioning=1, stratify_by='env_name',
   )
   for path in top_mediators:
       neighbours = [edge.other(path) for edge in adj.edges if path in edge]
   ```

4. **Adjacency = candidate intervention targets.** Each PC
   neighbour of a high-differential mediator is a variable the
   substrate author can construct an intervention against. The
   author's job stays — translating the named neighbour into a
   slot Claim swap from the literature — but the candidate set
   is no longer literature pattern-matching; it's empirically
   ranked from the corpus.

This pipeline is the framework's answer to "where do we go next?"
The scope claim says where the mechanism doesn't work; the
mediator differential says what's different there; PC adjacency
says which upstream variables are causally adjacent and worth
intervening on.

---

## Recipe summary

1. Classify cells (`with_cell_class`) → exclude saturated.
2. Run bridges (`runner.run`) → mech/outcome/link verdicts.
   File-level exclusions on `MODULE_SCOPE`.
3. If HELD_WITH_SCOPE_FLAG → meta-regression for scope axis.
   Per-env weighting (`meta_regression_paired_g`) for cross-env claims.
3a. Partial-ρ to disambiguate candidate mediators from the canonical one.
4. PC for moderator candidates (depth-1, then depth-2 robustness).
5. K-fold CV + drop-one-env sensitivity on the meta-regression.
6. Per-burst probes if scalar verdicts are null. Use `phase_link_
   consistency` as the Q-stability scope predicate, not pooled
   `q_div`.
6a. Cross-burst lag correlation as a causal-precedence diagnostic.
7. Tautology audit on cleavage candidates.
5b. Endogenous-discriminator search via interaction-term + per-stratum DoWhy.
8. Mediator differential + PC adjacency → next-sweep targets.
9. Stopping rule: delete the bridge if iterations confirm the
   cross-env variation doesn't exist (predictor too constant,
   single-env leverage, or different variable surfacing in
   per-burst panel).

Steps 1-2 are required; 3-9 are conditional on the verdicts /
the question being asked.

---

## Stopping rules: when to delete the bridge

`POWER_INSUFFICIENT` is a verdict, not a fix-the-scope prompt.
When a bridge returns POWER_INSUFFICIENT across multiple
iterations (different scope, different analysis primitive,
different threshold), the cross-env / cross-corpus variation
the claim presupposes may genuinely not exist on this corpus.
The framework refuses to smuggle "no signal" past the reader
as HELD or NO_EFFECT; the iteration should end with bridge
deletion (the claim is structurally untestable here) or a
substantively different claim (different predictor, different
unit of analysis, different scope). Iterating the bridge
indefinitely against the verdict signal is the failure mode
the framework is designed to prevent.

Concrete tells that the claim is dead, not the scope:

- The predictor doesn't vary cross-env in the corpus (e.g.,
  `bootstrap_fraction` cluster at [0.98, 1.00] across true
  chain MDPs after bandit-tail exclusion).
- After per-env weighting + drop-one-env sensitivity, the
  signal vanishes.
- The phase-aware view (per-burst plc) reveals a different
  variable as the actual cross-env predictor (e.g., Q-stability
  rather than the original predictor).

When two of those tells fire, the bridge should be deleted and
the post-mortem documented as a memory note. The historical
companion bridge (corpus-pinned baseline, if any) can stay as a
record of the artifact-shaped finding.

Cf. CLAIM 16 deletion (`findings_residual_unexplained` final
update + `findings_q_div_threshold_too_loose`).

---

## Why this isn't a Python pipeline

This recipe is markdown, not code, deliberately. Every step
involves judgment calls (which bridges to author, which
covariates to test, which threshold counts as "significant" for
your study). Codifying it as a single `run_full_analysis(corpus)`
function would either be too rigid (missing cases) or too
parameterised (configuration dwarfs analysis). The framework's
analyses are typed primitives; the recipe is a *suggested
sequence*, not a Protocol.
