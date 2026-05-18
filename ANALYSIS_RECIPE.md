# Post-sweep analysis recipe

A workflow for going from a corpus to verdicts. Anchored to
**Pearl's ladder**: the framework's `Tier` enum on every
`@claim_bridge` IS a commitment about which rung of inference
the bridge's evidence supports.

**This doc is workflow, not API.** Per-primitive contracts live
in docstrings; the canonical-analyses table lives in CLAUDE.md;
this recipe stitches them in a *suggested sequence*. Every step
involves authorial judgment.

---

## The principle: Pearl's ladder

Three rungs of inference, plus a substrate-axiom scaffold.
Stricter assumptions, stronger conclusions.

| Tier | Question shape | Assumptions | Bridge tag |
|---|---|---|---|
| `INVARIANT` (off-ladder) | "Does the substrate's precondition hold here?" | Substrate-axiom (e.g., bias premise is active when σ_Q√(2 log A) is exceeded) | `tier=Tier.INVARIANT` + threshold predicate |
| `ASSOCIATIONAL` (rung 1) | "What co-varies with what?" | Sampling distribution; no causal claim | `tier=Tier.ASSOCIATIONAL` |
| `INTERVENTIONAL` (rung 2) | "What if we do X?" | Causal graph + identification (backdoor / IV / frontdoor) | `tier=Tier.INTERVENTIONAL` + `DoEffect` |
| Counterfactual (rung 3) | "What would have happened if Y had been Y'?" | Counterfactual identification (cross-world) | Not shipped — see `FUTURE_WORKS.md` |

**Pick the lowest rung whose answer is strong enough for your
claim.** Climbing unnecessarily is the same shape as overstating
what you have; the audit failure mode is the opposite — citing
an `ASSOCIATIONAL` verdict as causal.

`Tier.INVARIANT` is orthogonal to the ladder. INVARIANT bridges
gate the rung-1/2 verdicts: when the precondition fires
`INVARIANT_VIOLATION` on a cell, downstream associational /
interventional claims are moot for that cell. Author the
INVARIANT first; downstream bridges scope on it.

---

## 0. Pre-flight (every rung)

### 0.1 Classify cells by outcome status

```python
from corroborate_rl.cell_classification import with_cell_class
df = with_cell_class(pl.read_parquet(corpus_path),
                     outcome_path='eval_best_burst_mean')
# df['_cell_class'] ∈ {saturated, solved, unsolved, no_threshold}
```

`saturated` cells (both arms at corpus-max) carry zero paired
signal — exclude from link analyses. `_cell_class` is itself
an endogenous predicate and reusable in `scope=`.

### 0.2 Sanitize cache canonical strings (only when needed)

If the cache predates a `canonical_str` change (substrate-side
factory refactor or framework default-elision change), run

```bash
PYTHONPATH=. uv run python scripts/sanitize_cache_canonical.py \
    experiments/data/cache/<corpus>.parquet
```

Without sanitization, cells from old/new substrate versions
canonicalise differently and analyses see false regime mismatch.
The script backs up the original to `.bak` and is idempotent.

### 0.3 MODULE_SCOPE for hypothesis-wide exclusions

Set `MODULE_SCOPE = ~pl.col('env_name').str.ends_with('-bsuite')`
(or similar) at the top of the hypothesis module. The runner
AND-combines it into every bridge's scope. Use for
diagnostic-env exclusions that apply to the whole hypothesis.

---

## 1. Rung 1: Association

### 1.1 Author the bridge with `tier=Tier.ASSOCIATIONAL`

```python
@claim_bridge(
    source=INTERVENTION,                    # DoEffect tuple
    target='jensen_gap',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,                # ← rung commitment
    predicted_direction='a_lt_b',
    pair_by=('seed',),
    scope=...,                              # see §1.2
)
def ddqn_reduces_jensen_gap__regime(
    paired_g: PairedGResult,
) -> Verdict:
    if paired_g.n_pairs < 30:
        return Verdict.POWER_INSUFFICIENT
    if paired_g.g < -0.3 and paired_g.p_value < 0.05:
        return Verdict.HELD
    return Verdict.NO_EFFECT
```

**Mech / outcome / link separation discipline (per CLAUDE.md):**
the three verdicts MUST be authored independently. Never collapse
"link null" with "mechanism dormant" — different verdicts.

### 1.2 Scope discipline — three tiers, only one earns its keep

A bridge's `scope=` predicate decides which cells the bridge
sees. There are three distinct uses, with sharply different
scientific status:

**Tier A — Endogenous predicate (the right answer).** Scope
expressed in terms of measurable trajectory properties:
`pl.col('q_late_mean') > 0`, `pl.col('jensen_dormancy_gap') < 1e-9`,
`pl.col('phase_link_consistency') > 0.5`. Generalizes — any
future cell whose trajectory enters the regime is in scope.
This IS the bridge's scientific commitment about WHEN the claim
applies.

**Tier B — Exogenous HPO scaffolding (tactical debt).** Pins
on controllable knobs the experimenter set: `total_steps == 200000`,
`replay.capacity == 50000`, `sync_period == 100`. Necessary when
the cache contains contaminating sub-sweeps that share `(arm_key,
seed)` buckets — without these, the analysis silently averages
causally distinct experiments. But each HPO pin is a placeholder
for the endogenous predicate it stands in for. The bridge author's
follow-up job is to find that endogenous downstream and migrate.

**Tier C — Exogenous env-name OR corpus-name (don't).**
`pl.col('env_name') == 'X'` is lazy science — doesn't communicate
WHY that env is special; a future env with the same structural
property won't match. Cf. `feedback_endogenous_scope_predicates`.
`pl.col('corpus') == 'X'` is **worse**: corpus name is a
data-collection artifact (which sweep produced these cells), not
even a scientific property — the bridge's claim is then tied to
the operator's filesystem layout. When a corpus is archived,
renamed, or rebuilt, every corpus-scoped bridge silently breaks
(historic CLAIM 5 / Polyak τ / Acrobot γ=0.999 bridges hit
this). The right move is finding the endogenous predicate the
corpus stood in for — usually `(env_name, gamma, sync_period,
total_steps, reward_scale, target_sync.tau)` covers it.

**Reusable envelopes** (`_FOURROOMS_REGIME`, `_ACTION_DIM_SWEEP_
REGIME`, `_REV6_SAMPLE_EFFICIENCY_REGIME` in `dqn_bridges.py`)
are tier-B/C debt currently named for the corpora they pin.
They earn their keep as DRY composition operators but each is
a migration opportunity.

**`dedupe_strategy='raise'` is the strict default.** It surfaces
tier-B issues automatically. When a bridge errors with
"duplicate cells differ on ... [HP column]", the right reflex is
(1) add the HP pin to make the analysis run, (2) note the
migration opportunity to a tier-A predicate. The smarter raise
silently passes true replicates from sub-sweep `intervention_name`
aliases.

**Admission Gates (WIP) will eventually enforce tier-admissibility
at evaluate-time.** `framework_variable_scope.md` codifies the
within-vs-across-stratum admissibility rule today
(`assert_stratification_admissible`). Until the gate vocabulary
covers tier-B/C exogenous predicates, this discipline is
authorial.

### 1.3 Aggregation unit: stratum, not seed-pair

The corollary of "strata are the unit of inference, not cells":
**how you build the per-stratum effect size determines whether
your test is honest at the cross-stratum question shape**.

**Three legitimate aggregation patterns**, picked by question:

1. **Seed-paired Δ within a stratum**, then **aggregate across
   strata**. Used by `paired_g`, `paired_link_per_burst`,
   `paired_delta_link_dowhy`. Per-seed Δs ARE the within-stratum
   observations of the within-stratum effect; cross-stratum
   tests then operate on per-stratum summaries.

2. **Independent-samples Δ per stratum**, then **regress /
   pool across strata**. Used by `stratified_arm_diff_pooled`,
   `meta_regression_unpaired_d`, `stratum_delta_link_dowhy`,
   `stratum_outcome_attenuation_dowhy`. No per-seed pairing —
   stratum-mean treatment minus stratum-mean baseline. Strata
   are the unit of inference at both stages.

3. **Within-cell per-burst pairing** for phase-structured
   claims. `paired_link_per_burst`+`phase_link_consistency`
   computes a Pearson r at each (env, burst) across seed-pairs
   — within-stratum sampling, not cross-stratum
   pseudo-replication.

**Anti-pattern**: pattern 1 used for a **cross-stratum** scope
claim. `paired_g_pooled` pools N seeds × M envs as iid Δ-samples;
each env is pseudo-replicated by N. Cross-env scope claims
("DDQN HELDs across REACH cohort") need pattern 2 to weight
strata equally.

The CLAUDE.md `feedback_paired_g_in_rl` memory has the
substrate-level rule of thumb:

| Question shape | Pattern |
|---|---|
| Cross-env / cross-config pooling | 2 (independent-samples) |
| Cross-env scaling with covariate | 2 + `meta_regression_unpaired_d` |
| Within-cohort link strength | 3 (per-burst within-cell) |
| Cross-env link claim | 2 (`stratum_delta_link_dowhy`) |
| Synthetic SCM analytical test | 1 (`paired_g` — seeds ARE iid) |
| Single-stratum smoke check | 1 (`paired_g`) — OK at one stratum |

In the DDQN substrate as of 2026-05-12: pattern 1 (`paired_g`)
is reserved for analytic tests. All cross-stratum bridges use
pattern 2.

### 1.4 Pick the rung-1 primitive by question shape

The canonical-analyses table is in CLAUDE.md; this is the
question-first index pointing at it.

| Question | Primitive | When to use |
|---|---|---|
| Treatment shifts a single outcome (single stratum)? | `arm_mean_diff` (independent-samples Welch) OR `paired_g` (when seeds ARE iid, e.g. synthetic) | Default for one cohort; `arm_mean_diff` replaces `paired_g` when ρ(treatment, baseline by seed) ≈ 0. |
| Heavy-tailed Δ? | `bootstrap_paired_g` (asymmetric CIs) + `cliff_delta_paired` (skew-robust point) | When `paired_g.assumption_violations` flags skew/heavy-tail. |
| Cross-env / cross-config pooled effect? | `stratified_arm_diff_pooled` (per-stratum Cohen's d → DL random-effects pool) | **The cross-stratum primitive.** Independent-samples; strata are unit. Verdict heterogeneity-flagged (HELD / HELD_WITH_SCOPE_FLAG / NO_EFFECT / POW_INSUF). **NOT `paired_g_pooled`** — see `feedback_paired_g_in_rl`. |
| Cross-env scaling on env-level covariate? | `meta_regression_unpaired_d` (sibling of `meta_regression_paired_g`, seed-pairing-free) | Each env contributes per-config strata; covariate slope estimated with proper SE that accounts for within-env config heterogeneity. Replaces n=3 envs Pearson r (brittle — see `findings_n3_pearson_brittle`). |
| Per-(env, burst) g panel? | `paired_g_per_burst` | When Q dynamics are non-monotone (Q-explosion-prone envs). |
| Per-burst link r(Δ_target, Δ_predictor)? | `paired_link_per_burst` + `phase_link_consistency` | Phase structure unmasking. Within-stratum seed-pairing is structural here (one Pearson r per (env, burst) stratum). |
| Cross-env link claim (mech-conditioned)? | `stratum_delta_link_dowhy` (per-(env, burst) stratum-Δ panel + DoWhy backdoor + refutations) | Replaces `paired_delta_link_dowhy` which pseudo-replicated by (env, burst, seed). Built-in mech conditioning via `min_vanilla_predictor`. |
| Outcome attenuation by binary moderator (e.g. q_div > threshold)? | `stratum_outcome_attenuation_dowhy` (per-stratum Δ_outcome + binary attenuator + DoWhy + env one-hot) | Replaces `link_attenuation_dowhy` which used within-stratum seed-paired Pearson r as outcome. Built-in mech conditioning. |
| Cross-env link r predicted by moderator? | `paired_link_per_env` | CLAIM 14-shaped soft-tautology / polarity tests. |
| Cross-config moderator (sweep as lever)? | `cross_config_paired_slope` | CLAIM 21-shaped sync / Polyak slope. |
| Per-env panel meta-regression on paired g? | `meta_regression_paired_g` | Seed-paired form. Use `meta_regression_unpaired_d` instead for the seed-pairing-free RL form. Kept for synthetic substrate tests and analytic-test usage. |
| Within/between decomposition? | `mundlak_decomposition` / `mundlak_paired_g_per_burst` | Cluster-robust CR1 SE; Hausman test for `β_b == β_w`. |
| Mediation: does X→Y survive conditioning on Z? | `partial_spearman_rho` (1 Z) / `partial_spearman_rho_multi` (k Z) / `stratified_partial_spearman_rho` (JCI) | The non-parametric canonical mediation primitives. Pair with `mediation_dowhy` at the same scope for a typed `linearity_status` diagnostic (HYPOTHESIS_AS_GRAPH §3b scope-cluster). `proportion_mediated` was the v9 ratio-of-noisy-means primitive — deleted 2026-05-18; see CLAUDE.md mediation recipe for the statistical case. |
| Sample efficiency among solvers? | `paired_g_among_solvers` | Per-env gate by solve threshold. |
| Verdict landscape per env? | `verdict_distribution_per_env` | Tally INVARIANT bridges' per-cell verdicts. |
| Mediator-set tautology audit? | `tautology_audit` | HP shadow / partial-correlation collapse / convergence proxy. |

### 1.5 Robustness on rung 1

- **K-fold CV on meta-regression** —
  `cross_validate_meta_regression`. `sign_consistency['<covariate>']`
  < 0.7 → fragile; the cleavage axis depends on which strata are
  in the training set.
- **Drop-one-env sensitivity** — coefficient that flips sign on
  removing a single env is leverage-driven, not population-level.
  Apply after k-fold (k-fold randomly splits seeds; drop-one-env
  tests structural sensitivity to strata themselves).
- **Small-n cross-env Pearson r is structurally brittle.** At
  n_envs ≈ 3, Pearson r has 1 dof; a 1-SE perturbation to any
  single env's d (SE ≈ 0.15-0.18 for per-env Cohen's d at
  n_seeds=30) can swing r between +1 and -1. The pre-ingest
  CLAIM 19 r=+0.999 HELD verdict was a Type-I artifact; 30
  statistically-equivalent new FR cells flipped r to -0.85 on
  the same data quality. Use `meta_regression_unpaired_d`
  instead — each env contributes per-config strata, the
  covariate slope is estimated with proper SE. Cf.
  `findings_n3_pearson_brittle`.
- **PC depth-2 robustness** — re-run `discover_adjacency` with
  `max_conditioning=2`; vanishing edges are depth-1-fragile.
- **Per-burst probes** when scalar mediator returns null —
  scalar paired_g over a trajectory-averaged mediator washes
  non-monotone phases. Cf. `findings_fourrooms_time_series`,
  `findings_l2_acrobot_goldilocks`.
- **Tautology audit on cleavage candidates** before publishing.

### 1.6 When rung 1 isn't enough

- A correlation surfaces but a confound is plausible → §2 (rung 2).
- Pooled verdict is heterogeneous (`HELD_WITH_SCOPE_FLAG`) →
  meta-regression for the cleavage axis (still rung 1; see §1.4).
- Phase structure suspected → per-burst probes (still rung 1).
- Endogenous-discriminator search across envs → §1.5 robustness +
  per-stratum DoWhy in §2.3.

---

## 2. Rung 2: Intervention

### 2.1 Author bridges with `tier=Tier.INTERVENTIONAL` + `DoEffect`

A rung-2 bridge requires an explicit treatment / baseline
contrast (`DoEffect`) AND a causal graph that admits an
identification strategy (backdoor / frontdoor / IV). The
substrate's `INTERVENTION = DoEffect(treatment=(...),
baseline=())` declares the contrast; the framework runs DoWhy
identification on the supplied DAG.

```python
@claim_bridge(
    source=INTERVENTION,
    target='outcome_native',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,               # ← rung commitment
    scope=...,
)
def do_treatment_increases_outcome__regime(
    backdoor_ate: BackdoorResult,
    placebo_refutation: RefutationResult,
    random_common_cause_refutation: RefutationResult,
) -> Verdict:
    if not backdoor_ate.identified:
        return Verdict.POWER_INSUFFICIENT
    if abs(placebo_refutation.refuted_ate) > 0.05:
        return Verdict.NO_EFFECT
    if random_common_cause_refutation.drift > 0.05:
        return Verdict.NO_EFFECT
    return Verdict.HELD if backdoor_ate.ate > 0.1 else Verdict.NO_EFFECT
```

### 2.2 Pick the rung-2 primitive

| Question | Primitive |
|---|---|
| ATE under backdoor adjustment? | `dowhy.backdoor_ate` |
| Robust to placebo treatment? | `dowhy.placebo_refutation` (gate on `abs(refuted_ate) < tolerance` — NOT `drift`; placebo's drift ≈ \|real_ate\|) |
| Robust to synthetic confounder? | `dowhy.random_common_cause_refutation` (gate on `drift < tolerance`) |
| 2×2 factorial interaction? | `factorial_2x2_interaction` |
| Continuous baseline-arm predictor → Δ_outcome at stratum level? | `stratum_baseline_predictor_link_dowhy` — per-(env, HP) stratum row carrying `baseline_predictor` (e.g. baseline target staleness) and Δ_target; backdoor+placebo+RCC under env one-hot. |
| Continuous treatment, dose-response across HP values? | `paired_continuous_do_dowhy` — pair on (env, HP, seed) where HP is the swept treatment_var; each pair contributes one (HP_value, Δ_outcome) point. Use when the polyak/dampened sweep has only a handful of HP buckets but multiple seeds per bucket — the seed-paired form preserves data density that stratum-pooling collapses. |
| Stratum-Δ link DoWhy (cross-env, mech-conditioned)? | `stratum_delta_link_dowhy` — per-(env, burst) stratum-Δ panel, backdoor+placebo+RCC. Uses pattern-2 aggregation (independent-samples per stratum). |

**Composite trio verdicts.** When a single causal claim is
corroborated by `backdoor + placebo + RCC` working together
(the three checks always travel; placebo-passes-RCC-fails is
not a real verdict shape), author ONE bridge with a composite
verdict rather than three separate bridges citing the same
extent. The framework's `dowhy_trio_verdict` helper (in
`experiments/findings/ddqn/_verdicts.py`) takes a result
satisfying the `_DowhyTrioResult` Protocol (any `backdoor +
placebo + random_common_cause` carrying dataclass) and returns
HELD iff all three sub-checks HELD; any NO_EFFECT dominates; any
POW_INSUF (no NO_EFFECT) returns POW_INSUF.

### 2.3 Per-stratum DoWhy when env is the regime axis

Per-env regimes can flip ATE sign (e.g., `findings_polyak_q_regime`:
reward-polarity inverts staleness's effect on outcome). Run
`backdoor_ate + placebo + RCC` per regime stratum; ATEs that flip
across regimes with both refutations passing in each are direct
rung-2 evidence for the discriminator.

The interaction-term test (`Δ_outcome = β₀ + β_T·T + β_M·M + β_int·T×M`)
is the lighter-weight rung-1 companion when DoWhy adjustment isn't
needed. Pair with per-stratum DoWhy when the bridge claims
moderation.

### 2.4 When rung 2 isn't enough

- Multiple mediators interact (additive backdoor adjustment can't
  decompose) → rung 3.
- "What would have happened if M had been forced to M*?" — a
  manipulability question, not an ATE → rung 3.

---

## 3. Rung 3: Counterfactual (deferred)

NDE / NIE counterfactual identification not shipped as a typed
primitive. The framework ships `mediation_dowhy` (two-stage
backdoor + OLS-with-mediators) but **as a diagnostic, not a
magnitude estimator** — see §3.1 below. Authoring a typed
counterfactual primitive (e.g., DoWhy's mediation.two_stage with
proper SE + identification) is gated on a substrate bridge
whose claim is load-bearingly counterfactual; the lift gate is
"a real bridge fires positive on the linear-mediation
diagnostics AND the question is genuinely manipulability."

### 3.1 `mediation_dowhy` as diagnostic, not magnitude

The framework ships `corroborate.analyses.mediation_dowhy` —
v10 PoC ported forward (CASE_STUDY_LESSONS §2.11). It returns
`MediationResult(total_ate, direct_ate, indirect_ate,
indirect_proportion, ...)` via:

1. **Total ATE**: DoWhy backdoor on (treatment, outcome) under
   the DAG.
2. **Direct ATE**: OLS of outcome on (treatment, *mediators)
   — the coefficient on treatment is the controlled direct
   effect.
3. **Indirect = total − direct**; **indirect proportion =
   indirect / total** (NaN-guarded when |total| < eps).

**Use it as a diagnostic for the linearity / multicollinearity
failure mode**, not a decomposition estimator. Symptoms that
indicate the model is broken (the v10 lesson empirically
reproduced on FR × MLP × unshaped × baseline, n=120, mediators
{self_ref, σ_action}, pairwise ρ ≈ 0.78–0.93):

- `direct_ate` sign-flipped vs `total_ate` (regression
  artifact, not mechanism)
- `indirect_proportion` outside [0, 1]
- Multicollinearity warning in the residuals

When ANY of these fire on a corpus, the linear-mediation
assumption is broken — **read the result as a flag, not a
magnitude**. The canonical mediation primitive in this regime
remains `partial_spearman_rho` (or `_multi` for joint
mediators), which is rank-based and multicollinearity-robust.

### 3.2 The mediation gating recipe (v10 §2.11)

Mediation magnitudes are slippery — never read them without
prior power + topology gating:

1. **Power-gate the total ATE.** Run `dowhy.backdoor_ate` +
   `placebo_refutation` + `random_common_cause_refutation` on
   `(treatment, outcome)`. If placebo doesn't drop to ≈ 0 OR
   RCC drift > tolerance, the total ATE is not reliable enough
   to decompose. STOP.

2. **Topology-gate via PC.** Run `discover_adjacency` with
   depth ≥ 2 on the full variable set. If PC does NOT remove
   the treatment-outcome edge under the proposed mediator
   separating set, the posited DAG is suspect. Either re-DAG
   to match PC, or STOP.

3. **Mediation via partial-Spearman (canonical).** Use
   `stratified_partial_spearman` (single mediator) or
   `stratified_partial_spearman_multi` (joint mediators) to
   compute ρ(X, Y | Z) on the SAME panel. Rank-based +
   multicollinearity-robust + bounded-output → reliable
   mediation evidence.

4. **`mediation_dowhy` as diagnostic.** Sign-flips (direct/total
   opposite signs) or proportions outside [0, 1] indicate
   linear-mediation assumption is broken on this corpus.

5. **Refutations on the total.** Placebo + RCC corroborate the
   foundation; mediation magnitude doesn't inherit reliability
   beyond what stages 1 and 3 establish.

Bridges that just emit `mediation_dowhy.indirect_proportion`
without the gating pipeline are NOT trusted. The empirical
failure mode documented in `mediation_dowhy.py`'s module
docstring (FR × MLP × unshaped × baseline) is the reproducible
example of skipping this discipline. Cf.
`findings_v10_mediation_failure_reproduced` (memory) and CLAUDE.md
`### Mediation recipe` section.

---

## 4. Cross-rung discovery

`discover_adjacency` (PC) is a moderator candidate proposer.
Suggests adjacencies; rung-1 bridges then verify, rung-2 DoWhy
then tests.

```python
from corroborate.graph.discovery import discover_adjacency

adj = discover_adjacency(
    df, variables=PC_VARIABLES,
    alpha=0.05, max_conditioning=1,
    stratify_by='env_name',          # JCI when env-features in scope
)
```

JCI stratification is required when env-level features
(action_dim, horizon, reward_scale, polarity) are part of the
variable set. Pooling across envs without stratification produces
spurious edges — cf. `findings_dowhy_three_probes`,
`findings_pc_validates_claim_architecture`.

---

## 5. Stopping rules

`POWER_INSUFFICIENT` is a verdict, not a fix-the-scope prompt.
When a bridge returns POWER_INSUFFICIENT across multiple
iterations (different scope, different analysis primitive,
different threshold), the cross-env / cross-corpus variation the
claim presupposes may genuinely not exist on this corpus. The
framework refuses to smuggle "no signal" past the reader as
HELD or NO_EFFECT; the iteration should end with bridge
deletion (the claim is structurally untestable here), a
substantively different claim (different predictor, different
unit of analysis, different scope), or an honest
AWAITING-DATA placeholder.

### 5.1 Structurally dead — delete

Tells that the claim is dead, not the scope:

- The predictor doesn't vary cross-env in the corpus (e.g.,
  `bootstrap_fraction` clusters at [0.98, 1.00] across true
  chain MDPs after bandit-tail exclusion).
- After per-env weighting + drop-one-env sensitivity, the signal
  vanishes.
- The phase-aware view (per-burst plc) reveals a different
  variable as the actual cross-env predictor (Q-stability rather
  than the original predictor).

When two of those tells fire, delete the bridge and document
as a memory post-mortem. The historical companion bridge
(corpus-pinned baseline, if any) can stay as a record of the
artifact-shaped finding. Cf. CLAIM 16 deletion
(`findings_residual_unexplained` + `findings_q_div_threshold_too_loose`).

### 5.2 AWAITING DATA — keep as placeholder

Some POWER_INSUFFICIENT bridges are NOT dead; they're waiting
for data the corpus doesn't yet hold. Tells:

- Scope predicate gates on a config combination (γ=0.999,
  sync=10k, Polyak τ > 0, n_step=10, ...) absent from the
  cache, but the sub-sweep IS planned / archived / re-collectible.
- Memory documents a finding on the dedicated corpus that
  reproduces when ingested (cf. `findings_l2_acrobot_goldilocks`
  documents r=-0.93 to -0.998 per-burst link on Acrobot γ=0.999
  — the dedicated corpus had it; the universal cache doesn't).

For these, **keep the bridge** with an explicit
`AWAITING DATA: <description>` marker in the docstring (parallel
to a Finding's `BLOCKED_ON`). The bridge fires POW_INSUF on the
current cache; when the missing corpus reintegrates, the bridge
activates without scope edits. The author's discipline is to
periodically reconcile AWAITING-DATA bridges against actual data
availability — if the planned sweep is permanently cancelled or
the result no longer matters, downgrade to §5.1.

Examples in the current substrate: `extreme_q_divergence_attenuates_link__dowhy_corroborated`
(needs q_div > 1.0 cells from sync=10k MinAtar);
`ddqn_benefit_scales_with_effective_horizon__fourrooms` (needs
γ=0.999 FR cells); `acrobot_per_burst_link_active__gamma_0999`
(needs `l2_x_gamma_acrobot` corpus); Polyak τ pair (needs
target_sync.tau > 0 cells).

### 5.3 Corpus name in scope = silent breakage

A scope predicate gating on `pl.col('corpus') == 'X'` couples
the bridge's claim to filesystem layout. When the corpus is
archived to S3 (per CORPUS_INTEGRITY.md trace-eviction) or never
reintegrated post-cache-rebuild, every corpus-scoped bridge
silently flips to empty extent / POW_INSUF — without an
AWAITING-DATA marker, this looks like §5.1 (dead) when it's
actually §5.2 (placeholder). Replace corpus-name scopes with
endogenous predicates (env + γ + sync + total_steps +
reward_scale + target_sync.tau) as early as authoring; the
endogenous form survives corpus reorganization. Cf. §1.2 tier-C.

---

## 6. Data-driven next-sweep targeting

The recipe's answer to "where do we go next?":

1. **Classify envs by convergence on the BASELINE arm** —
   `corroborate_rl.convergence.classify_envs(baseline_runs)`.
   The natural failure-mode signature, not one induced by an
   intervention.
2. **Mediator differential** — Hedges' g of each candidate
   mediator across solved-vs-unsolved baselines. Top-|g|
   mediators are the empirical failure-mode signatures.
3. **PC adjacency on the panel** — depth-1 stratified by env.
   Each PC neighbour of a high-differential mediator is a
   variable the substrate author can construct an intervention
   against.

The author's job stays — translating the named neighbour into a
slot Claim swap from the literature — but the candidate set is
no longer literature pattern-matching; it's empirically ranked
from the corpus.

---

## Recipe summary

1. Pre-flight: `with_cell_class`, sanitize cache if needed,
   set MODULE_SCOPE.
2. INVARIANT bridges first — substrate preconditions for the
   downstream rungs.
3. Scope discipline (§1.2): tier-A endogenous predicate; tier-B
   HPO is debt; tier-C env-name OR corpus-name is wrong.
4. Aggregation discipline (§1.3): pick the seed-pairing pattern
   to match question shape. Cross-stratum claims → pattern 2
   (independent-samples Cohen's d per stratum, regress / pool
   across). Within-cohort link → pattern 3 (per-burst within-cell).
   Pattern 1 (per-pair paired_g) reserved for synthetic /
   single-stratum.
5. Rung 1 ASSOCIATIONAL bridges per the §1.4 question table.
6. Robustness (§1.5): k-fold CV, drop-one-env, PC depth-2,
   per-burst probes if scalar null, tautology audit. **Watch
   for small-n Pearson r brittleness** — at n_envs=3, swap to
   `meta_regression_unpaired_d`.
7. Rung 2 INTERVENTIONAL bridges when rung 1 surfaces a
   confound-plausible effect. Per-stratum DoWhy when env is the
   regime axis. Composite trio verdicts via `dowhy_trio_verdict`
   when backdoor + placebo + RCC corroborate one causal claim.
8. Rung 3 only when linear-mediation diagnostics fire AND the
   question is genuinely manipulability.
9. Cross-rung: PC for moderator candidates, JCI-stratified.
10. Stopping rules: §5.1 structurally dead (delete), §5.2
    AWAITING DATA (keep with marker), §5.3 corpus-name scope
    silently breaks (migrate to endogenous).
11. Next-sweep targeting: mediator differential + PC adjacency
    on baseline-arm convergence classes.

---

## Why this isn't a Python pipeline

Every step involves judgment calls (which bridges to author,
which covariates to test, which threshold counts as
"significant" for your study). Codifying as a single
`run_full_analysis(corpus)` function would be too rigid (missing
cases) or too parameterised (configuration dwarfs analysis).
The framework's analyses are typed primitives; the recipe is a
*suggested sequence*, not a Protocol.
