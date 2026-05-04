# Scope-search — canonical procedure

The framework's contribution to a research cycle is finding the
**scope** of an authored mechanism claim: the load-bearing
assumption(s) on the chain `env feature → invariance gap →
mechanism activation → outcome` whose violation makes the link
break. This document fixes the canonical procedure for that
search, naming the framework primitives used at each step and
how the causal-discovery tools interleave with meta-regression.

**Pre-reading**: `LIFECYCLE.md` (12-stage flow); `CLAUDE.md`
(typing discipline + framework gist); `claim_bridge.py` module
docstring (file-protocol bridge authoring).

## The procedure

The procedure's input is a `Hypothesis[R]` carrying a mechanism
claim. The output is one of:

- **HELD on a subscope**: a measurable scope-condition that, when
  imposed, makes the mechanism→outcome link HELD. The
  scope-condition is committed as a *verdict measurable* the
  substrate registers via `@measurable` (e.g.
  `at_most[gap<=threshold].verdict`) and pre-registers on
  `Hypothesis.measurables`.
- **NULL on every candidate scope**: the mechanism activates but
  no measurable subscope reproduces the link → the chain has
  edges the framework hasn't yet captured.
- **MECHANISM EDGE EXPOSED**: the per-env paired g splits cleanly
  by an env-feature (e.g. `|A|≥3` for DDQN's Jensen-bias regime)
  even though the link to outcome stays null — a refinement of
  the mechanism claim, not a scope for the link.

### Step 1. Articulate the causal chain

Write the chain explicitly: `env_feature → invariance_gap →
mechanism_activation → outcome`. Identify which edge the
intervention operates on. For DDQN:

```
env ─► (|A|, σ_Q) ─► Jensen-max-bias ─► argmax-bias ─► policy ─► outcome
                                       └────── DDQN ──────┘
```

DDQN intervenes at the argmax-bias edge; for the link to hold,
every upstream edge must carry signal.

### Step 2. Identify candidate scope variables

For each upstream edge, name the variable that determines whether
the edge carries signal. Two flavours:

- **Per-cell measurable** — a continuous per-run quantity computed
  from the trace (e.g. `jensen_dormancy_gap`,
  `state_coverage_kl_uniform_late`, `td_residual_late`). Express
  as `Measurable[R, float]` in the substrate via `@measurable`
  and pre-register on `Hypothesis.measurables` so cell_runner
  persists it as a scalar column on every RunRow.
- **Per-env / structural** — a categorical or integer property of
  the env / configuration (e.g. `action_dim`,
  `bootstrap_depth = γ × episode_length`). Read from the env
  catalogue or a `@measurable(reads=('env_name',))` resolver
  (`log_action_dim`, `log_obs_dim`, `r_max`, etc.).

### Step 3. Design a sweep that varies the scope variable

Use the framework's `sweep` + `DQNRunner` (or substrate-equivalent
Runner). HP knobs that *modulate* the scope variable are fine;
the load-bearing axis is the scope variable itself.

Don't data-mine an existing corpus when the variable wasn't
varied at collection time — the result is biased by whatever the
collection HPs implicitly fixed.

### Step 4. Stage-6 stratified aggregation

Per-env paired g + DerSimonian-Laird random-effects pool — one
function call:

```python
from corroborate.aggregate import hypothesis_comparison_from_cells

mech = hypothesis_comparison_from_cells(
    treatment_h, treatment_runs, baseline_runs,
    outcome_path='jensen_gap',
    pair_by=('seed',),
    group_by='env_name',
    baseline_h=baseline_h,
)
# mech.per_group: tuple[GroupStats, ...] — one per env
# mech.pooled: PooledStats — DL pool with I², PI
# mech.verdict: Verdict — Popperian aggregate
```

Repeat for the link edge (`eval_best_burst_mean`,
`eval_final_mean`, etc.). Two HypothesisComparisonRows: one per
edge.

### Step 5. Audit candidate per-cell mediators (interleaved)

Before treating a per-cell measurable as a real scope variable,
run the three-check tautology audit:

```python
from corroborate.redundancy_check import audit_mediator_panel

reports = audit_mediator_panel(
    candidate_measurables, runs,
    outcome_reads=frozenset({'mc_return'}),
    hp_axes=('replay.capacity', ...),
    outcome_path='eval_best_burst_mean',
    hp_stratum_axis=...,
)
# Each TautologyReport carries:
#   flagged_outcome (reads-jaccard ≥ 0.5)
#   flagged_hp (R² ≥ 0.95 on any HP axis)
#   flagged_no_residual_signal (within-stratum ρ ≈ 0)
```

A candidate that fails any check shouldn't be treated as a scope
variable for the link without further qualification — it's a
restatement of the outcome, a relabeling of an HP, or a
non-residual signal.

### Step 5b. Inspect time-series before collapsing to scalars

**Important caveat from the FourRooms study**: scalar reductions
(per-cell mean of a per-burst trace) can hide phase-dependent
effects. On FourRooms the env-level scalar `jensen_gap g=+0.13`
(sign reversed) coexisted with per-burst trajectories showing
DDQN *reduces* bias in bursts 0-3 (Δbias = −1.02 → −0.09) and
only diverges late (Δbias = +1.15 → +479, success-induced
Q-growth, not mechanism failure). The within-pair correlation
`r(Δbias, Δret)` is negative at every burst, confirming the
mechanism→outcome chain operates throughout — but the scalar
mean averaged the early reduction with the late explosion to
near-zero.

The substrate persists raw 2-D per-burst arrays
(`predicted_q_at_start`, `mc_return` shape `(n_bursts, K)`) so
this probe is post-hoc-derivable from existing data. The
per-burst probe to run when scalar mediator analysis returns
null:

```python
# Pair runs by seed; stack per-burst arrays.
delta_bias = ddqn_bias_per_burst - vanilla_bias_per_burst  # (n_pairs, n_bursts)
delta_ret = ddqn_ret_per_burst - vanilla_ret_per_burst    # (n_pairs, n_bursts)

for b in range(n_bursts):
    r, p = scipy.stats.pearsonr(delta_bias[:, b], delta_ret[:, b])
    print(f'burst {b}: r={r:+.3f} p={p:.3f}')
```

If the per-burst sign is *invariant* but the scalar mean is
near-zero, the mechanism operates throughout training but at
different magnitudes per phase — the chain holds; the scalar
reduction was the wrong abstraction. If the per-burst sign
*flips* between phases, the mechanism's relationship to outcome
is genuinely phase-dependent and the scalar would be a false
average.

The framework provides `paired_link_per_burst` and
`paired_g_per_burst` analyses for the panel-typed form of this
probe; consume them via `@claim_bridge` for typed verdicts.

**Don't author a `mediator_late` or `mediator_growth` Measurable
to fix this.** The substrate's raw-trace contract supports any
post-hoc reduction; ad-hoc reductions belong inline in analysis,
not in `measurables.py` (see `feedback_measurables_not_logging`).

### Step 6. Stage-9 meta-regression on covariates

Map per-env GroupStats → StratumG[str] panel, regress per-env g
on candidate covariates:

```python
from corroborate.analyses.paired_g import per_env_paired_g_panel
from corroborate.meta_regression import meta_regress_panel

panel = per_env_paired_g_panel(
    cells, treatment_arm='ddqn', baseline_arm='vanilla_dqn',
    source='jensen_gap',
)
res = meta_regress_panel(
    panel,
    covariates_per_stratum={
        env: {'log_action_dim': math.log(get(env).n_actions)}
        for env in envs
    },
)
# res.cleavage_axes: tuple[str, ...] — significant covariates
# res.coefficients: tuple[CovariateCoefficient, ...] — β + CI + p
```

A significant β with CI excluding zero is the empirical scope
claim's content — a numeric threshold on the covariate.

### Step 7. Validate via causal-discovery (interleaved)

A significant meta-regression coefficient says "the per-env g
varies systematically with this covariate". It does NOT say
"this covariate causally drives the link". For the rung-2
interventional verdict, route the covariate through the dowhy
analysis triple authored as `@claim_bridge`s:

```python
# Author intervention/refuter bridges in a substrate bridges
# module via @claim_bridge; consume `analyses.dowhy.backdoor_ate`,
# `analyses.dowhy.placebo_refutation`,
# `analyses.dowhy.random_common_cause_refutation` as fixtures.
# `runner.run_module(<bridges_module>, data=corpus)` evaluates
# every bridge via `bridge.evaluate(b, cells)` → holds_when →
# Verdict. The substrate can build a CausalGraph over the
# resulting BridgeEvaluations and pass through
# `promote_bridged_evidence` to upgrade pairs with ≥2
# INTERVENTIONAL HELDs to causal_bridged.
```

When the dowhy triple HELDs *and* the meta-regression β is
significant, the scope claim is supported at rung-2-conditional-
on-DAG. Direction-of-causation caveats from `analyses/dowhy.py`
apply.

### Step 8. Commit the threshold

If the scope variable + threshold survives validation, commit
the scope claim as a **verdict measurable** the substrate
registers via `@measurable`. Convention: the column name encodes
the threshold (e.g. `at_most[jensen_dormancy_gap<=0].verdict`),
and the body returns `'held'` / `'invariant_violation'` /
`'power_insufficient'` per cell.

```python
@measurable(
    name='at_most[gap<=threshold].verdict',
    reads=(),
)
def at_most_gap_verdict(
    record: Mapping[str, object],
    gap: float,  # auto-injected from the gap @measurable
) -> str:
    if gap != gap:  # NaN
        return 'power_insufficient'
    return 'held' if gap <= 0.0 else 'invariant_violation'
```

After registration, the substrate adds the verdict measurable to
`Hypothesis.measurables` and cell_runner persists the per-cell
verdict on every RunRow. Bridges that test "≥X% of cells fire
the predicted verdict" consume the column name directly via the
`source` field of an `@claim_bridge`.

### Step 9. Re-evaluate

Re-run the link analysis on the premise-active subscope. The
`paired_g` analysis's `cell_predicate` kwarg accepts a closure
that reads the verdict measurable column and returns True for
premise-active cells:

```python
def premise_active(cell: Mapping[str, object]) -> bool:
    return cell.get('at_most[gap<=threshold].verdict') == 'held'

link_active = paired_g.fn(
    cells, treatment_arm='ddqn', baseline_arm='vanilla_dqn',
    pair_by=('seed',), source='eval_best_burst_mean',
    env_name=env, cell_predicate=premise_active,
)
```

If the link verdict on the subscope is HELD where the marginal
verdict was NULL, the scope claim is corroborated.

## When discovery interleaves vs precedes regression

The audit (step 5) precedes regression because feeding a tautology
into meta-regression confuses the result.

DoWhy validation (step 7) follows regression because the
covariates being tested are the meta-regression-significant ones.

PC-depth structural discovery (`compare_pc_depths`) and partial-
correlation analysis (`stratified_spearman_rho`) sit alongside
meta-regression at step 6 — they answer different questions on
the same per-env table:
- Meta-regression: "is g systematically explained by feature F?"
- PC depth: "what's the conditional independency structure
  between g and {F₁, F₂, …}?"
- Stratified-ρ: "after binning on F, is there residual within-bin
  correlation?"

Use them when the question warrants. Don't over-fire — three
different verdicts on the same data is more confusion than
corroboration unless they actually disagree.

## When the procedure ends

Three terminating outcomes:

1. **Scope corroborated**: a covariate threshold reproduces the
   link. Commit as a verdict measurable (step 8); pre-register
   it on `Hypothesis.measurables`. The mechanism's scope is now
   part of the substrate's persisted columns.
2. **No covariate corroborates**: every candidate's
   meta-regression β has CI bracketing zero AND/OR fails dowhy
   validation. Either the upstream chain has edges we haven't
   measured (extend the substrate's measurable panel) or the
   sweep design is power-limited (run a wider one).
3. **Mechanism-edge refinement**: the link stays null but the
   per-env paired g on the *mechanism* (not the outcome) splits
   by a covariate. The covariate refines the *mechanism* claim
   ("DDQN's bias-reduction operates only when …") even if the
   link to outcome is independently null. Document and commit
   as a verdict measurable on the mechanism edge.

## Worked example

`experiments/analyze.py` is the canonical scope-search:

```
uv run python experiments/analyze.py \
    --corpus action_dim_wide \
    --treatment-arm ddqn --baseline-arm vanilla_dqn \
    --stages paired_g,meta_reg,pc
```

Stages 4 + 6 of the procedure correspond to `meta_reg` and `pc`
in `analyze.py`'s pipeline.
