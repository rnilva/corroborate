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

### 3a. Partial-ρ to disambiguate from an established mediator

When a canonical mediator exists (e.g. `jensen_gap` for the DDQN
study), candidate mediators may either carry **independent**
outcome-predictive signal or be **collinear** with the canonical
one. `partial_spearman_rho` separates them:

```python
from corroborate.graph.discovery import partial_spearman_rho

# Δ_outcome ⊥ Δ_candidate | Δ_jensen_gap?
rho, p = partial_spearman_rho(
    delta_outcome, delta_candidate, delta_jensen_gap,
)
# rho significantly non-zero → candidate carries independent signal
# rho ≈ 0 → candidate is collinear with jensen_gap (restatement)
```

A candidate surviving the partial-ρ test is a bona-fide separate
intervention target; one that collapses is a redundant restatement
of the canonical mediator's mechanism.

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
3. If HELD_WITH_SCOPE_FLAG → meta-regression for scope axis.
3a. Partial-ρ to disambiguate candidate mediators from the canonical one.
4. PC for moderator candidates (depth-1, then depth-2 robustness).
5. K-fold CV the meta-regression coefficients for sign stability.
6. Per-burst probes if scalar verdicts are null.
6a. Cross-burst lag correlation as a causal-precedence diagnostic.
7. Tautology audit on cleavage candidates.
8. Mediator differential + PC adjacency → next-sweep targets.

Steps 1-2 are required; 3-8 are conditional on the verdicts /
the question being asked.

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
