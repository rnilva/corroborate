# corroborate lifecycle

The framework's primitives compose into a 12-stage flow (stage 1
split into substrate authoring vs hypothesis formation — see
diagram). Drawn here to make the wiring legible: each stage names
its inputs, its outputs, the modules that implement it, and its
current state (**live** = production consumer; **orphan** =
primitive exists but no consumer; **missing** = stage isn't
implemented).

This is a **diagnostic map**, not a typed contract. Stages are
narrative not Protocols. When a primitive doesn't fit any stage,
it's misplaced. When a gap appears between two stages, that's a
wire to add.

**Two research lifecycles, one diagram.** The framework supports
both *hypothesis-driven* research (mechanism authored from theory
at stage 1b, then validated through stages 2–8) and
*discovery-first* research (intervene with a simple sweep,
discover candidate mediators at stages 7+9, then *propose* a
mechanism back to 1b for validation). The short feedback edge `7,
9 → 1b` is what makes discovery-first work: it's the path by
which the mechanism edge gets *proposed* rather than *assumed*.
Real research is usually discovery-first; the §3 verdict is the
*closing* move of an investigation, not the opening.

**Terminology note (Composition).** corroborate has NO
`Composition` type. v9's `ConditionalComposition` (env-conditional
dispatch tree) was retired in favor of uniform interventions +
post-hoc aggregation (random-effects pooling + meta-regression) —
scope is empirical, not authored. The artifact at stage 2 is just
a `functools.partial[T]`; the cell_runner names it `configured`.
"Composition" appears in framework docstrings only as the English
word for "way of composing primitives," never as a typed artifact.

**Terminology note (MechanismKey).** corroborate has NO
`MechanismKey` type. v10 had an explicit `Hypothesis.mechanism_key`
artifact; corroborate derives the canonical fingerprint from
`intervention_arms` via `Hypothesis.arm_key()`, which composes
each `Intervention`'s `(slot_path, replacement)` through
`canonical_str`. Identity is derived, not declared. Two hypotheses
with the same `intervention_arms` (and possibly different HP grid
points) share an `arm_key`; HP variation is a covariate, not an
arm distinguisher.

## The flow

```
   ┌──────────────────────────────────────────────────────────────────────────┐
   │                                                                          │
   │              ┌─ SUBSTRATE (1a) ─────────────────────────┐                │
   │              │  @claim, @measurable, @bridge,           │                │
   │              │  EnvSpec, bridge factories               │                │
   │              │  (pre-research; persistent code)         │                │
   │              └────────────────┬─────────────────────────┘                │
   │                               │                                          │
   │              ┌─ HYPOTHESIS-FORMATION (1b) ─────────────┐                 │
   │              │  Hypothesis(intervention_arms, bridges) │                 │
   │              │  mechanism edge = load-bearing claim    │                 │
   │              │  ◄── short loop from 7, 9 (discovery)   │                 │
   │              │  ◄── long loop from 12 (dialectic)      │                 │
   │              └────────────────┬─────────────────────────┘                │
   │                               │                                          │
   │              ┌─ BIND (2) ─────▼─────────────────────────┐                │
   │              │  apply_interventions(claim, arms)        │                │
   │              │  → configured claim (`partial[T]`)       │                │
   │              │    in Pearl: post-do() SCM               │                │
   │              └────────────────┬─────────────────────────┘                │
   │                               │                                          │
   │              ┌─ RUN (3) ──────▼─────────────────────────┐                │
   │              │  configured claim × exogenous_grid       │                │
   │              │  → trace_record + CallRecords + bridges  │                │
   │              └────────────────┬─────────────────────────┘                │
   │                               │                                          │
   │              ┌─ PROJECT (4) ──▼─────────────────────────┐                │
   │              │  cell_runner builds:                     │                │
   │              │  • RunRow (verdicts + measurements)      │                │
   │              │  • TraceRow (trace + arrays)             │                │
   │              │  • ComputationGraph (call structure)     │                │
   │              │  • BridgeResult tuple (per-cell)         │                │
   │              └────────────────┬─────────────────────────┘                │
   │                               │                                          │
   │              ┌─ PERSIST (5) ──▼─────────────────────────┐                │
   │              │  parquet (RunRow, TraceRow scalars+1D)   │                │
   │              │  zarr    (TraceRow multi-dim arrays)     │                │
   │              │  ☒ ComputationGraph not persisted        │                │
   │              │  ☒ BridgeResult tuple not preserved      │                │
   │              └────────────────┬─────────────────────────┘                │
   │                               │                                          │
   │   ┌───────────────────────────┼─────────────────────────────┐            │
   │   │                           │                             │            │
   │   ▼ AGGREGATE (6, rung 2)     ▼ DISCOVER (7, rung 1)        │            │
   │   from_cells(...)             discover_adjacency(...)       │            │
   │   per-arm Hedges' g           PC + JCI on outcome cols      │            │
   │   per-group GroupStats        DiscoveredAdjacency           │            │
   │   pooled PooledStats          OrientedAdjacency             │            │
   │   facts: tuple[FactRow]                                     │            │
   │   reads_set                                                 │            │
   │   →  HypothesisComparisonRow                                │            │
   │                                                             │            │
   │       └────────┐                       ┌────────────────────┘            │
   │                │                       │                                 │
   │                ▼ PROMOTE (8, rung 1+2) ▼                                 │
   │           build_causal_graph(BridgeResults, links)                       │
   │           promote_bridged_evidence(graph)                                │
   │           → CausalGraph with Tier-typed BridgeEdges                      │
   │           → 'causal_bridged' upgrade when ≥2 paired admits               │
   │                                                                          │
   │                ┌────────────────────────────────┐                        │
   │                │                                │                        │
   │                ▼ SCOPE-PREDICT (9)              ▼ REWARD (10)            │
   │                meta_regression on               compute_R_info(h, reg)   │
   │                (env_features, per_env_g)        ΔI(h) per axiom 19       │
   │                → MetaRegressionReport           uses facts + reads_set   │
   │                                                                          │
   │                                                 ▼ REGISTER (11)          │
   │                                                 RegisterState            │
   │                                                 latest-wins facts        │
   │                                                                          │
   │                                                 ▼ IMPROVE/FALSIFY (12)   │
   │                                                 propose new Hypothesis ──┼──┐
   │                                                                          │  │
   └──────────────────────────────────────────────────────────────────────────┘  │
                                                                                 │
                                                                                 │
              ┌──────────────────────────────────────────────────────────────────┘
              │
              └──→ back to HYPOTHESIS-FORMATION (1b)
```

**Feedback edges.** Two loops feed back into stage 1b:

- **Short loop (within a research cycle):** stages 7 (DISCOVER) and 9
  (SCOPE-PREDICT) propose mechanism candidates back to 1b. v10 §4–§6 →
  §3 IS this loop — discovery names which algorithmic quantity to test;
  1b authors the bridge that tests it. Today this is *manual* (an author
  reads the discovery output and writes a new mechanism bridge); the
  framework provides the discovery primitives but doesn't auto-propose
  candidates.
- **Long loop (across research cycles):** stage 12 (IMPROVE/FALSIFY)
  reads the register, identifies under-tested mechanism candidates from
  stages 7+9, and authors a new Hypothesis at 1b. This is the dialectic
  loop — currently missing infrastructure (stages 10–12 not yet
  implemented).

## Per-stage status

### Stage 1a — SUBSTRATE

**Inputs**: substrate-author code.

**Outputs**: `@claim`-decorated Module/Free Claims, `@measurable`
instances, `@bridge` instances, `EnvSpec`, bridge factories.
Persistent code; written once and reused across many hypotheses.

**Modules**: `claim.py`, `measurable.py`, `bridge.py`, `bridges.py`
(factories), `intervention.py`, `verdict.py`, `reductions.py`,
`signature.py`, `invariant.py`, `rl/dqn/measurables.py`,
`rl/dqn/claims/*`, `rl/dqn/dqn.py`, `rl/env_catalogue.py`.

**Status**: **live**.

**Orphans at this stage**:
- `bridges.py` factories (`monotonic`, `correlation`,
  `mean_exceeds`, `variance_shrinks`) — authored, no Hypothesis
  attaches them today (`bridges=()` in §3 sweep).

---

### Stage 1b — HYPOTHESIS-FORMATION

**Inputs**: a research question, plus either (a) theoretical
content naming the mechanism, OR (b) discovery output from
stages 7/9 proposing candidate mechanisms, OR (c) register
output from stage 12 selecting an under-tested candidate.

**Outputs**: a typed `Hypothesis` carrying:
- `intervention_arms: tuple[Intervention, ...]` — the typed
  identity of mechanism swaps (sourced from the substrate).
  Defines `arm_key` via canonical fingerprint.
- `bridges: tuple[Bridge[R], ...]` — the per-edge tests applied
  to the resulting record. Today this is a flat tuple; the v10
  vision (queued, see step 1 below) is for a typed
  `CausalSubgraph` with role-tagged edges (mechanism / outcome /
  link / refuter) read by the §3 verdict pipeline.

**Modules**: `hypothesis.py`, `intervention.py`. The Hypothesis
construction sites today are the experiment scripts
(`experiments/collect_ddqn_runs.py`, the smokes); 1b doesn't have
its own module because the construction is one frozen-dataclass
call.

**Status**: **live for the flat-bridges shape**;
**partially-typed for the subgraph shape** — `causal_graph.py`
provides the edge / tier / direction primitives but the
Hypothesis itself doesn't yet carry a `CausalSubgraph` field.
The connector from `Hypothesis.bridges` to a typed subgraph is
the highest-leverage v0 → v1 move.

**Load-bearing observation**: the *mechanism edge* of a Hypothesis
is the central theoretical claim; outcome and link edges test
its implications. `arm_key` derives from the mechanism edge's
source (the intervention). Two hypotheses with the same
intervention but different mechanism-edge *targets* are different
hypotheses (same do, different theoretical commitment about what
the do affects).

---

### Stage 2 — BIND

**Inputs**: `Hypothesis` + outermost `@claim`.

**Outputs**: a configured claim — a `functools.partial[T]`. In
Pearl terms: the post-do() SCM. **There is no `Composition`
type** — corroborate does not have one. v9's
`ConditionalComposition` (the env-conditional dispatch tree) was
retired in favour of uniform interventions + post-hoc
aggregation; the bound theory is just `partial(claim,
**intervention_kwargs)` returning a `Callable[..., T]`. The
cell_runner uses the variable name `configured` for this
artifact.

**Modules**: `intervention.Intervention.apply`, `apply_interventions`.

**Status**: **live via parallel path** (`partial(dqn,
**intervention)` in cell_runner) but the framework primitive
`apply_interventions` is **orphan**.

**Wire to add (small)**: replace cell_runner's direct `partial(dqn,
**intervention)` with `apply_interventions(dqn,
hypothesis.intervention_arms)`. Pearl-correct semantics; lets
the framework primitive be the production path.

---

### Stage 3 — RUN

**Inputs**: configured claim × exogenous_grid (per-cell exogenous
values like rng_key, env, env_params, obs_dim, n_actions).

**Outputs**: per-cell trace_record (Mapping[str, jax.Array]) +
list[CallRecord] (under trace_context) + bridge invocations.

**Modules**: `rl/cell_runner.run_dqn_arm`, `rl/dqn/eval.train_with_eval`,
`rl/loop.scan_loop`, `loop.python_loop`, `sweep.sweep` (substrate-
agnostic; has tests but no production consumer).

**Status**: **live** (cell_runner is the consumer; sweep.sweep is
**orphan** — production rewrites orchestration in
`collect_ddqn_runs.py`).

---

### Stage 4 — PROJECT

**Inputs**: per-cell trace_record + bridges + CallRecords.

**Outputs**:
- `RunRow` (id, parent_id, cycle_id, timestamp, verdict,
  arm_key, measurements). Bridge results flatten into measurements
  as `bridge.<name>.verdict` + `bridge.<name>.stats.<key>`.
- `TraceRow` (id, leaves, arrays). Per-step series + multi-dim
  arrays.
- `ComputationGraph` from `build_computation_graph(records)`.
  Returned in `ArmResult.graph`.
- (implicitly) the per-cell BridgeResult tuple inside cell_runner.

**Modules**: `rl/cell_runner`, `computation_graph.build_computation_graph`,
`aggregate.fact_from_bridge_result` (runtime path).

**Status**: **live for RunRow / TraceRow.** **Orphan**:
`ComputationGraph` is built per arm but no consumer reads it.
**Missing**: BridgeResult tuple is consumed inline (flattened into
RunRow.measurements) but not preserved as a structured artifact —
which is why stage 8 has no input.

**Wire to add (load-bearing, ~30 LoC)**:
`aggregate.reconstruct_bridge_results(run: RunRow) ->
tuple[BridgeResult, ...]` lifts the flat-keyed `bridge.<name>.*`
back into BridgeResult objects. Restores the structure that
stage 8 consumes.

---

### Stage 5 — PERSIST

**Inputs**: rows from stage 4.

**Outputs**: parquet (runs.parquet, traces.parquet,
comparisons.parquet), zarr (arrays.zarr).

**Modules**: `persistence.write_runrows`, `write_tracerows`,
`write_comparisonrows`, `iter_trace_records`, `tighten_trace_dtypes`.

**Status**: **live**. ComputationGraph deliberately not persisted
(per design — runtime artifact only). HypothesisComparisonRow has
no read/write yet.

---

### Stage 6 — AGGREGATE (rung 2 — interventional ATE)

**Inputs**: corpus of RunRows + (treatment_runs, baseline_runs)
slice + pair_by + optional group_by.

**Outputs**: `HypothesisComparisonRow` carrying:
- per-arm stats (mean, sd, n)
- effect_size_g, se, derived_q, delta_i_population
- per_group: tuple[GroupStats] (stratified mode)
- pooled: PooledStats (random-effects DerSimonian-Laird)
- facts: tuple[FactRow] (per-bridge projection across cells)
- reads_set: frozenset[str] (union of fact reads)
- adequately_powered + verdict + refutation_class

**Modules**: `aggregate.paired_comparison_from_runs`,
`hypothesis_comparison_from_cells` (HypothesisComparisonRow.from_cells),
`statistics.{hedges_g_paired, mde_paired, derived_q_from_g_se,
delta_i_from_q, random_effects_summary, random_effects_verdict}`.

**Status**: **live**. `facts` and `reads_set` are populated but
**unread** by any current consumer — they're forward-investment
for stages 10/11.

---

### Stage 7 — DISCOVER (rung 1 — observational)

**Prerequisites**: a corpus of RunRows (NOT a Hypothesis — stage 7
is hypothesis-free; it operates on raw data).

**Inputs**: corpus dataframe + variable list + `stratify_by`.

**Outputs**: `DiscoveredAdjacency` (PC + JCI),
`OrientedAdjacency` (Meek-rule oriented), candidate-mediator
projections (typed `@measurable` reductions of trace data),
within-env Pearson correlations.

**Modules**: `causal_discovery.{discover_adjacency,
orient_adjacency, compare_pc_depths, partial_spearman_rho,
stratified_spearman_rho, ...}`, `rl/dqn/measurables.py` (typed
mediator catalog: q_gap_late, td_residual_late, greedy_match_late,
plus the value-curve family — learning_curve_auc,
time_to_threshold, return_at_25pct_steps, plateau_slope_late),
the per-env Pearson smokes.

**Status**: **live**. §4 / §5 / §6 smokes consume this.

**Sub-stages.** Discovery has internal structure worth naming:
- **7a** — PC adjacency + orientation on a chosen variable set
  (the §4 path).
- **7b** — typed mediator projection + within-env Pearson on the
  candidate-mediator set (the §5 path).
- **7c** — per-env PC, run within each stratum separately when
  pooled discovery is null (the §6 path).

The output of any sub-stage feeds back to **stage 1b** — a
mediator with a significant within-env Pearson, or an edge
adjacent to `final_return` in per-env PC, IS a mechanism
candidate. The discovery → 1b feedback edge is what makes
discovery-first research a closed loop.

---

### Stage 8 — PROMOTE (rung 1 + rung 2 typed)

**Inputs**: BridgeResults (from stage 4), per-cell HELDs (from
stage 6's per_group), cross-env link verdicts (from
`link_pearson_across_groups`).

**Outputs**: `CausalGraph` with `BridgeEdge`s typed by `Direction
× Tier × evidentiary_level`. After `promote_bridged_evidence`:
pairs with ≥2 `causal_one_sided` edges upgrade to
`causal_bridged`. PAPER §3.5's contract.

**Modules**: `causal_graph.{Direction, Tier, BridgeEdge,
build_causal_graph, compose_direction, chain_tier,
promote_bridged_evidence}`.

**Status**: **primitives exist; stage entirely missing the
input connector**. `build_causal_graph` consumes
`Iterable[BridgeResult]` but no path lifts BridgeResults out of
the corpus. Once stage 4's `reconstruct_bridge_results` lands,
stage 8 becomes live.

---

### Stage 9 — SCOPE-PREDICT

**Inputs**: per-group g + se from stage 6's `per_group` +
env_feature columns.

**Outputs**: `MetaRegressionReport` with per-feature coefficients,
SE, p-values, R².

**Modules**: `meta_regression.py`.

**Status**: **live**. `smoke_phase_c_meta_regression` consumes it.
The §7 paper section's operational scope predictor lives here.

---

### Stage 10 — REWARD (axiom 19's ΔI)

**Inputs**: HypothesisComparisonRow (from 6) + register state
(from 11).

**Outputs**: `ΔI(h)` — scalar information gain. Built from:
- per-fact ΔI from FactRow.delta_i (verdict-oriented information).
- redundancy term from `compute_redundancy(h, register)` — the
  4-factor jaccard·concord·intervention·identity overlap.
- aggregation across facts.

**Modules**: **MISSING**. v10 has `redundancy.py` (~240 LoC) and
`compute_R_info` (~100 LoC). corroborate hasn't ported either.

---

### Stage 11 — REGISTER

**Inputs**: HypothesisComparisonRow + new facts.

**Outputs**: `RegisterState` — append-only ledger of all prior
hypothesis comparisons, with a latest-wins fact projection
(`register.facts`) that downstream consumers read for redundancy
calculation + dialectic-loop frontier selection.

**Modules**: **MISSING**. v10 has `register.py` (~120 LoC).

---

### Stage 12 — IMPROVE / FALSIFY (the dialectic loop)

**Inputs**: register state + a frontier (queue of hypotheses to
improve / falsify next) + discovery output from stages 7+9
identifying under-tested mechanism candidates.

**Outputs**: a candidate mechanism / intervention spec routed
*back to stage 1b* (NOT directly to stage 2), so 1b can author
the typed Hypothesis (typed bridges, predicted directions, etc.)
before it gets bound and run.

**Modules**: **MISSING**. Orchestrator lives outside any single
substrate; would compose stages 1b–11 into a closed loop.

---

## Where to look for orphans

| Primitive | Stage | Status | Wire to consumer |
|---|---|---|---|
| `bridges.py` factories | 1a | orphan | requires Hypothesis-author work, not a framework wire |
| typed `CausalSubgraph` on Hypothesis | 1b | partial | `causal_graph.py` exists but `Hypothesis.bridges` is still a flat tuple — needs role-tagged edges |
| short feedback loop (discovery → 1b) | 7→1b | manual | author reads discovery output and writes mechanism bridge; could grow into a typed candidate-proposer |
| `Intervention.apply` | 2 | orphan (parallel path) | replace `partial(...)` in cell_runner |
| `sweep.sweep` | 3 | orphan | collect_ddqn_runs migration |
| `ComputationGraph` capture | 4 | orphan | requires stage 8 connector |
| `causal_graph` (build_causal_graph, promote) | 8 | missing-connector | needs `reconstruct_bridge_results` from stage 4 |
| `FactRow` / `reads_set` | 6 | populated, unread | requires stage 10 (compute_R_info) |
| `transitive_reads` | 6 | orphan | requires stage 10 |
| `measurable_graph` | 9 alt | orphan | §3.5 didn't migrate |
| `meta_regression` | 9 | live | — |
| value-curve mediators (D3) | 7b | live | feed candidate-mediator covariates into stage 9 |
| `cross_validate_meta_regression` (D2) | 9 | live | — |
| `compare_pc_depths` (D1) | 7a | live | — |
| `redundancy` | 10 | absent | port from v10 |
| `register` | 11 | absent | port from v10 |
| `compute_R_info` | 10 | absent | port from v10 |

## Highest-leverage wires (in order)

1. **Hypothesis-as-subgraph (stage 1b → stages 6+8)**:
   typed `CausalSubgraph` on `Hypothesis`, replacing the flat
   `bridges: tuple[Bridge[R], ...]` tuple. Role-tagged edges
   (mechanism / outcome / link / refuter) so the §3 verdict
   pipeline reads role explicitly. Now that `causal_graph.py`
   has landed the BridgeEdge / Tier / Direction primitives, the
   Hypothesis-side typing is the missing connector. Unblocks
   the v10 §3 verdict pattern as a *typed* artifact rather than
   an implicit consumer pattern.

2. **Stage 4 → Stage 8 connector**:
   `aggregate.reconstruct_bridge_results(run: RunRow) ->
   tuple[BridgeResult, ...]`. ~30 LoC. Single primitive that
   unblocks the entire `causal_graph` pipeline. After this,
   Pearl-tier rung 1+2 promotion becomes runnable on the §3
   corpus.

3. **Stage 10 + 11 port**:
   `redundancy.py` + `register.py` from v10. ~400 LoC bundle.
   Closes the dialectic-loop reward signal. Reads stage 6's
   `facts` + `reads_set` (which are already populated). After
   this, axiom 19's ΔI is computable.

4. **Stage 12 orchestrator**:
   Improve / Falsify driver composing stages 1b–11. ~200 LoC.
   This is where v0 → v1 transition happens.

## Honest caveats

- **The diagram lags.** Code is the source of truth. When a stage's
  status changes, update this file. When a stage isn't reflected,
  this file is wrong. Read commit history if uncertain.

- **Multiple lifecycles, one diagram.** The diagram collapses
  several real flows; readers should expect to traverse only one
  at a time:
  - **Per-cell lifecycle:** 3 → 4 → 5.
  - **Per-hypothesis lifecycle:** 1b → 2 → ... → 6 → 11.
  - **Per-paper-section lifecycle:** 5 → 7 (for §4); 5 → 6 → 9
    (for §3 + §7).
  - **Hypothesis-driven research lifecycle:** 1a → 1b → 2 →
    ... → 6 → 8. Mechanism authored from theory; framework
    verdicts each edge of the subgraph claim.
  - **Discovery-first research lifecycle:** 1a → 2 → 3 → 4 →
    5 → 7 → 9 → *back to 1b* (propose mechanism candidate from
    discovery output) → 2 → ... → 6 → 8. The mechanism edge gets
    *proposed* by stages 7+9, not assumed at 1b. v10 §4–§6 → §3
    is exactly this lifecycle.

- **Forward-investment is allowed but typed.** A primitive in stage
  X with no consumer at stage Y is "orphan" not "wrong." Some
  forward-investment is correct (typing the contract before the
  consumer arrives). The orphan label says "do not assume this is
  load-bearing yet."
