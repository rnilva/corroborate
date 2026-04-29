# corroborate lifecycle

The framework's primitives compose into a 12-stage flow. Drawn here
to make the wiring legible: each stage names its inputs, its
outputs, the modules that implement it, and its current state
(**live** = production consumer; **orphan** = primitive exists but
no consumer; **missing** = stage isn't implemented).

This is a **diagnostic map**, not a typed contract. Stages are
narrative not Protocols. When a primitive doesn't fit any stage,
it's misplaced. When a gap appears between two stages, that's a
wire to add.

## The flow

```
   ┌──────────────────────────────────────────────────────────────────────────┐
   │                                                                          │
   │              ┌─ AUTHOR (1) ─────────────────────────────┐                │
   │              │  @claim, @measurable, @bridge,           │                │
   │              │  Hypothesis, EnvSpec                     │                │
   │              └────────────────┬─────────────────────────┘                │
   │                               │                                          │
   │              ┌─ BIND (2) ─────▼─────────────────────────┐                │
   │              │  apply_interventions(claim, arms)        │                │
   │              │  → Composition (post-do() SCM)           │                │
   │              └────────────────┬─────────────────────────┘                │
   │                               │                                          │
   │              ┌─ RUN (3) ──────▼─────────────────────────┐                │
   │              │  Composition × exogenous_grid            │                │
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
              └──→ back to BIND (2)
```

## Per-stage status

### Stage 1 — AUTHOR

**Inputs**: code (substrate-author or paper-author).

**Outputs**: `@claim`-decorated Module/Free Claims, `@measurable`
instances, `@bridge` instances, `Hypothesis` dataclass,
`EnvSpec`.

**Modules**: `claim.py`, `measurable.py`, `bridge.py`, `bridges.py`
(factories), `hypothesis.py`, `intervention.py`, `verdict.py`,
`reductions.py`, `signature.py`, `invariant.py`,
`rl/dqn/measurables.py`, `rl/dqn/claims/*`, `rl/dqn/dqn.py`.

**Status**: **live**. Substrate authors use this surface.

**Orphans at this stage**:
- `bridges.py` factories (`monotonic`, `correlation`,
  `mean_exceeds`, `variance_shrinks`) — authored, no Hypothesis
  attaches them today (`bridges=()` in §3 sweep).

---

### Stage 2 — BIND

**Inputs**: `Hypothesis` + outermost `@claim`.

**Outputs**: a callable Composition — `partial(claim, **kwargs)`
form. In Pearl terms: the post-do() SCM.

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

**Inputs**: Composition × exogenous_grid (per-cell exogenous values
like rng_key, env, env_params, obs_dim, n_actions).

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

**Inputs**: corpus dataframe + variable list + `stratify_by`.

**Outputs**: `DiscoveredAdjacency` (PC + JCI),
`OrientedAdjacency` (Meek-rule oriented).

**Modules**: `causal_discovery.{discover_adjacency, orient_adjacency,
partial_spearman_rho, stratified_spearman_rho, ...}`.

**Status**: **live**. §4 / §5-thin / §6 smokes consume this.

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
improve / falsify next).

**Outputs**: new Hypothesis → back to stage 2.

**Modules**: **MISSING**. Orchestrator lives outside any single
substrate; would compose stages 2–11 into a closed loop.

---

## Where to look for orphans

| Primitive | Stage | Status | Wire to consumer |
|---|---|---|---|
| `bridges.py` factories | 1 | orphan | requires Hypothesis-author work, not a framework wire |
| `Intervention.apply` | 2 | orphan (parallel path) | replace `partial(...)` in cell_runner |
| `sweep.sweep` | 3 | orphan | collect_ddqn_runs migration |
| `ComputationGraph` capture | 4 | orphan | requires stage 8 connector |
| `causal_graph` (build_causal_graph, promote) | 8 | missing-connector | needs `reconstruct_bridge_results` from stage 4 |
| `FactRow` / `reads_set` | 6 | populated, unread | requires stage 10 (compute_R_info) |
| `transitive_reads` | 6 | orphan | requires stage 10 |
| `measurable_graph` | 9 alt | orphan | §3.5 didn't migrate |
| `meta_regression` | 9 | live | — |
| `redundancy` | 10 | absent | port from v10 |
| `register` | 11 | absent | port from v10 |
| `compute_R_info` | 10 | absent | port from v10 |

## Highest-leverage wires (in order)

1. **Stage 4 → Stage 8 connector**:
   `aggregate.reconstruct_bridge_results(run: RunRow) ->
   tuple[BridgeResult, ...]`. ~30 LoC. Single primitive that
   unblocks the entire `causal_graph` pipeline. After this,
   Pearl-tier rung 1+2 promotion becomes runnable on the §3
   corpus.

2. **Stage 10 + 11 port**:
   `redundancy.py` + `register.py` from v10. ~400 LoC bundle.
   Closes the dialectic-loop reward signal. Reads stage 6's
   `facts` + `reads_set` (which are already populated). After
   this, axiom 19's ΔI is computable.

3. **Stage 12 orchestrator**:
   Improve / Falsify driver composing stages 2–11. ~200 LoC.
   This is where v0 → v1 transition happens.

## Honest caveats

- **The diagram lags.** Code is the source of truth. When a stage's
  status changes, update this file. When a stage isn't reflected,
  this file is wrong. Read commit history if uncertain.

- **Multiple lifecycles, one diagram.** Per-cell lifecycle (3 → 4 →
  5), per-hypothesis lifecycle (1 → 6 → 11), per-paper-section
  lifecycle (5 → 7 / 5 → 6 → 9). The diagram collapses them. Real
  flows are sub-paths.

- **Forward-investment is allowed but typed.** A primitive in stage
  X with no consumer at stage Y is "orphan" not "wrong." Some
  forward-investment is correct (typing the contract before the
  consumer arrives). The orphan label says "do not assume this is
  load-bearing yet."
