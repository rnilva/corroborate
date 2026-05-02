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
   │              │  @claim, @measurable, @claim_bridge,     │                │
   │              │  EnvSpec                                 │                │
   │              │  (pre-research; persistent code)         │                │
   │              └────────────────┬─────────────────────────┘                │
   │                               │                                          │
   │              ┌─ HYPOTHESIS-FORMATION (1b) ─────────────┐                 │
   │              │  Hypothesis(intervention_arms,          │                 │
   │              │             edges, measurables)         │                 │
   │              │  intervention edge = load-bearing claim │                 │
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
   │              │  → trace_record + CallRecords            │                │
   │              └────────────────┬─────────────────────────┘                │
   │                               │                                          │
   │              ┌─ PROJECT (4) ──▼─────────────────────────┐                │
   │              │  cell_runner builds:                     │                │
   │              │  • RunRow (verdict + measurements)       │                │
   │              │  • TraceRow (trace + arrays)             │                │
   │              │  • ComputationGraph (call structure)     │                │
   │              └────────────────┬─────────────────────────┘                │
   │                               │                                          │
   │              ┌─ PERSIST (5) ──▼─────────────────────────┐                │
   │              │  parquet (RunRow, TraceRow scalars+1D)   │                │
   │              │  graphs.json sidecar (per-arm topology)  │                │
   │              └────────────────┬─────────────────────────┘                │
   │                               │                                          │
   │   ┌───────────────────────────┼─────────────────────────────┐            │
   │   │                           │                             │            │
   │   ▼ AGGREGATE (6, rung 2)     ▼ DISCOVER (7, rung 1)        │            │
   │   from_cells(...)             discover_adjacency(...)       │            │
   │   per-arm Hedges' g           PC + JCI on outcome cols      │            │
   │   per-group GroupStats        DiscoveredAdjacency           │            │
   │   pooled PooledStats          OrientedAdjacency             │            │
   │   →  HypothesisComparisonRow                                │            │
   │                                                             │            │
   │       └────────┐                       ┌────────────────────┘            │
   │                │                       │                                 │
   │                ▼ PROMOTE (8, rung 1+2) ▼                                 │
   │           hypothesis_subgraph_verdict(h, runs)                           │
   │           → CausalGraph[BridgeEdge] + edge_verdicts                      │
   │           promote_bridged_evidence(graph)                                │
   │           → 'causal_bridged' upgrade when ≥2 paired admits               │
   │                                                                          │
   │                ┌────────────────────────────────┐                        │
   │                │                                │                        │
   │                ▼ SCOPE-PREDICT (9)              ▼ REWARD (10)            │
   │                meta_regression on               compute_R_info(h, reg)   │
   │                (env_features, per_env_g)        ΔI(h) per axiom 19       │
   │                → MetaRegressionReport                                    │
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
instances, `@claim_bridge`-decorated bridge declarations,
`EnvSpec`. Persistent code; written once and reused across many
hypotheses.

**Modules**: `claim.py`, `measurable.py`, `claim_bridge.py`,
`intervention.py`, `verdict.py`, `reductions.py`, `signature.py`,
`stratum.py`, `_registry.py`, `rl/dqn/measurables.py`,
`rl/dqn/claims/*`, `rl/dqn/dqn.py`, `rl/env_catalogue.py`.

**Status**: **live**.

---

### Stage 1b — HYPOTHESIS-FORMATION

**Inputs**: a research question, plus either (a) theoretical
content naming the mechanism, OR (b) discovery output from
stages 7/9 proposing candidate mechanisms, OR (c) register
output from stage 12 selecting an under-tested candidate.

**Outputs**: a typed `Hypothesis[R]` carrying:
- `intervention_arms: tuple[Intervention, ...]` — the typed
  identity of mechanism swaps. Defines `arm_key` via canonical
  fingerprint.
- `edges: tuple[claim_bridge.Bridge, ...]` — typed-edge subgraph
  claim. Each `Bridge` carries `source` / `target`,
  `intervention: DoEffect | None`, `tier`, `direction`, and
  per-edge `predicted_direction`. Body-less for the verdict-walk
  path; `holds_when` populated for the file-protocol path.
- `measurables: tuple[Measurable[R, object], ...]` — pre-
  registered measurables cell_runner persists as scalar columns
  on every RunRow.

**Modules**: `hypothesis.py`, `intervention.py`,
`claim_bridge.py`.

**Status**: **live**.

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
  arm_key, measurements). Pre-registered measurables and HPs
  flatten into `measurements` under bare names; verdict is
  HELD for any successfully-completed cell.
- `TraceRow` (id, leaves). Per-step series + multi-dim arrays.
- `ComputationGraph` from `build_computation_graph(records)` —
  one per arm, captured under `trace_context()` and persisted
  to the `graphs.json` sidecar at stage 5.

**Modules**: `rl/cell_runner.run_dqn_arm`,
`computation_graph.build_computation_graph`.

**Status**: **live**.

---

### Stage 5 — PERSIST

**Inputs**: rows from stage 4.

**Outputs**: `runs.parquet`, `traces.parquet`, and
`graphs.json` (a sidecar mapping `arm_key → GraphSpec`).

**Modules**: `persistence.{write_runrows, write_tracerows,
iter_trace_records, tighten_trace_dtypes,
write_graphs_sidecar, read_graphs_sidecar}`.

**Status**: **live**. `HypothesisComparisonRow` is materialised
on demand at stage 6 and not persisted to disk.

---

### Stage 6 — AGGREGATE (rung 2 — interventional ATE)

**Inputs**: corpus of RunRows + (treatment_runs, baseline_runs)
slice + pair_by + optional group_by.

**Outputs**: `HypothesisComparisonRow` carrying:
- per-arm stats (mean, sd, n)
- `effect_size_g`, `se`, `derived_q`, `delta_i_population`
- `per_group: tuple[GroupStats, ...]` (stratified mode)
- `pooled: PooledStats` (random-effects DerSimonian-Laird)
- `adequately_powered` + `verdict` + `refutation_class`

**Modules**:
`aggregate.hypothesis_comparison_from_cells`
(`HypothesisComparisonRow.from_cells`),
`statistics.{hedges_g_paired, mde_paired, derived_q_from_g_se,
delta_i_from_q, random_effects_summary,
random_effects_verdict}`. Shared per-env loop primitive at
`analyses.paired_g.per_env_paired_g_panel`; panel→regression
bridge at `meta_regression.meta_regress_panel`.

**Status**: **live**. Materialised view; not persisted to disk.

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

**Inputs**: `Hypothesis` + treatment / baseline RunRows.

**Outputs**: `HypothesisVerdict[R]` carrying:
- `graph: CausalGraph` — `BridgeEdge` per claimed edge keyed by
  `(source, target)`, with Pearl tier × direction ×
  evidentiary_level + per-edge stats (`ate`, `rho`, `pvalue`,
  `n_observations`).
- `edge_verdicts: Mapping[(s, t), Verdict]` — raw 4-bucket
  verdicts.
- `comparison_rows: Mapping[str, HypothesisComparisonRow]` —
  rich per-edge detail for intervention edges.

After `promote_bridged_evidence`: pairs with ≥2
`causal_one_sided` edges upgrade to `causal_bridged`.

**Modules**: `hypothesis_verdict.hypothesis_subgraph_verdict`,
`causal_graph.{Direction, Tier, BridgeEdge, authored_graph,
compose_direction, chain_tier, promote_bridged_evidence}`.
Topology helpers in `computation_graph.{producing_paths,
measurables_by_attachment, measurable_scope, ScopeInfo}` let
substrates derive paper-narrative scope (mechanism / outcome /
link) from claim-graph attachment.

**Status**: **live**.

---

### Stage 9 — SCOPE-PREDICT

**Inputs**: per-stratum (g, se) panel from stage 6's `per_group`
+ env-feature columns (`covariates_per_env` mapping or
`covariates: tuple[str, ...]` column-name list averaged per-env).

**Outputs**: `MetaRegressionResult` with per-feature
coefficients, SE, p-values, R².

**Modules**: `meta_regression.py`,
`analyses.meta_regression_paired_g`,
`analyses.meta_regression_per_burst` (both delegate to
`meta_regress_panel`).

**Status**: **live**.

---

### Stage 10 — REWARD (axiom 19's ΔI)

**Inputs**: HypothesisComparisonRow (from 6) + register state
(from 11).

**Outputs**: `ΔI(h)` — scalar information gain.

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
| short feedback loop (discovery → 1b) | 7→1b | manual | author reads discovery output and writes mechanism bridge |
| `Intervention.apply` | 2 | orphan (parallel path) | replace `partial(...)` in cell_runner |
| `redundancy.py` (axiom-19 R_info) | 10 | absent | port from v10 |
| `register` | 11 | absent | port from v10 |
| `compute_R_info` | 10 | absent | port from v10 |
| `experiments/compute_mediators.py` | — | superseded | scheduled subtraction (Phase 6C); per-cell measurables now flow through cell_runner inline |

## Highest-leverage wires (in order)

1. **Stage 10 + 11 port**:
   `redundancy.py` + `register.py` from v10. ~400 LoC bundle.
   Closes the dialectic-loop reward signal.

2. **Stage 12 orchestrator**:
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
