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

**Tier C — Exogenous env-name (don't).** `pl.col('env_name') ==
'X'` is lazy science. It doesn't communicate WHY that env is
special; a future env with the same structural property won't
match. Cf. `feedback_endogenous_scope_predicates`. The right
move is finding the endogenous downstream of the env's
structural property.

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

### 1.3 Pick the rung-1 primitive by question shape

The canonical-analyses table is in CLAUDE.md; this is the
question-first index pointing at it.

| Question | Primitive | When to use |
|---|---|---|
| Treatment shifts a single outcome? | `paired_g` | Default. |
| Heavy-tailed Δ? | `bootstrap_paired_g` (asymmetric CIs) + `cliff_delta_paired` (skew-robust point) | When `paired_g.assumption_violations` flags skew/heavy-tail. |
| Per-(env, burst) g panel? | `paired_g_per_burst` | When Q dynamics are non-monotone (Q-explosion-prone envs). |
| Per-burst link r(Δ_target, Δ_predictor)? | `paired_link_per_burst` + `phase_link_consistency` | Phase structure unmasking. |
| Cross-env link r predicted by moderator? | `paired_link_per_env` | CLAIM 14-shaped soft-tautology / polarity tests. |
| Cross-config moderator (sweep as lever)? | `cross_config_paired_slope` | CLAIM 21-shaped sync / Polyak slope. |
| Per-env panel meta-regression? | `meta_regression_paired_g` | One paired-g per env, equal weighting (cf. `findings_per_env_vs_per_cell_weighting`). |
| Within/between decomposition? | `mundlak_decomposition` / `mundlak_paired_g_per_burst` | Cluster-robust CR1 SE; Hausman test for `β_b == β_w`. |
| Mediation: does X→Y survive conditioning on Z? | `partial_spearman_rho` (1 Z) / `partial_spearman_rho_multi` (k Z) / `stratified_partial_spearman_rho` (JCI) | The non-parametric mediation primitives. **NOT `proportion_mediated`** (deprecated; ratio-of-noisy-means with no SE; see its module docstring). |
| Sample efficiency among solvers? | `paired_g_among_solvers` | Per-env gate by solve threshold. |
| Verdict landscape per env? | `verdict_distribution_per_env` | Tally INVARIANT bridges' per-cell verdicts. |
| Mediator-set tautology audit? | `tautology_audit` | HP shadow / partial-correlation collapse / convergence proxy. |
| Pooled across envs (random-effects)? | `paired_g_pooled` (DerSimonian-Laird) | Has known small-G limitations (cf. its docstring). |

### 1.4 Robustness on rung 1

- **K-fold CV on meta-regression** —
  `cross_validate_meta_regression`. `sign_consistency['<covariate>']`
  < 0.7 → fragile; the cleavage axis depends on which strata are
  in the training set.
- **Drop-one-env sensitivity** — coefficient that flips sign on
  removing a single env is leverage-driven, not population-level.
  Apply after k-fold (k-fold randomly splits seeds; drop-one-env
  tests structural sensitivity to strata themselves).
- **PC depth-2 robustness** — re-run `discover_adjacency` with
  `max_conditioning=2`; vanishing edges are depth-1-fragile.
- **Per-burst probes** when scalar mediator returns null —
  scalar paired_g over a trajectory-averaged mediator washes
  non-monotone phases. Cf. `findings_fourrooms_time_series`,
  `findings_l2_acrobot_goldilocks`.
- **Tautology audit on cleavage candidates** before publishing.

### 1.5 When rung 1 isn't enough

- A correlation surfaces but a confound is plausible → §2 (rung 2).
- Pooled verdict is heterogeneous (`HELD_WITH_SCOPE_FLAG`) →
  meta-regression for the cleavage axis (still rung 1; see §1.3).
- Phase structure suspected → per-burst probes (still rung 1).
- Endogenous-discriminator search across envs → §1.4 robustness +
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
| Continuous treatment? | `paired_continuous_do_dowhy` |
| Mediation as a do-effect? | `paired_delta_link_dowhy`, `link_attenuation_dowhy` |

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

NDE / NIE not yet shipped as framework primitive. DoWhy
backdoor + refutation primitives are the ingredients. Authoring
a typed counterfactual primitive is gated on a substrate bridge
whose claim is load-bearingly counterfactual; the lift gate is
"a real bridge fires positive on the linear-mediation
diagnostics AND the question is genuinely manipulability."

When the diagnostic does fire (proportion-mediated outside
[0,1], per-stratum partial-ρ heterogeneity, nonlinear M→Y), use
hand-rolled DoWhy mediation (the substrate exposes the
ingredients) and document the lift in `FUTURE_WORKS.md`.

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
deletion (the claim is structurally untestable here) or a
substantively different claim (different predictor, different
unit of analysis, different scope).

Concrete tells that the claim is dead, not the scope:

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
3. Rung 1 ASSOCIATIONAL bridges per the §1.3 question table.
   Scope discipline (§1.2): tier-A endogenous predicate; tier-B
   HPO is debt; tier-C env-name is wrong.
4. Robustness on rung 1: k-fold CV, drop-one-env, PC depth-2,
   per-burst probes if scalar null, tautology audit.
5. Rung 2 INTERVENTIONAL bridges when rung 1 surfaces a
   confound-plausible effect. Per-stratum DoWhy when env is the
   regime axis.
6. Rung 3 only when linear-mediation diagnostics fire AND the
   question is genuinely manipulability.
7. Cross-rung: PC for moderator candidates, JCI-stratified.
8. Stopping rules: delete the bridge after multiple
   POWER_INSUFFICIENT iterations + matching tells.
9. Next-sweep targeting: mediator differential + PC adjacency
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
