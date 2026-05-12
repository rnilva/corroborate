# Unconsumed primitives audit + wiring plan

## Status

Open plan. Replaces the archived `BRIDGE_PRIMITIVES_MANIFEST.md`
(see `docs/archive/`). The archived manifest tried to add new
primitives; the second-round review surfaced that the framework
already has typed surfaces — they just lack downstream consumers.
Wiring work, not new types.

## Method

For each typed primitive in the framework, count:

- **producers** — sites that create the typed value
- **consumers** — sites that read the typed value and branch
  behavior on it (not just round-trip serialization)

A primitive whose consumer count is 0 (or whose only consumers are
the round-trip / re-export sites) is **dead infrastructure**.
Adding more producers / fields to it before there's a real
consumer is the anti-pattern PRIMITIVES_AUDIT.md warns against.

Per the framework's "primitive earns its keep" discipline (CLAUDE.md
§ "When to introduce a framework primitive"), unconsumed
infrastructure either gets wired or gets pruned. Both options are
preferable to "add more dead fields."

## Audit results (2026-05-12)

Numbers from `grep` on `src/`, `scripts/`, `experiments/`,
`tests/`. Excludes `__pycache__` and `.run.json` snapshot files.

### 1. `RefutationClass` enum (`src/corroborate/bridge/verdict.py:75`)

| role | sites |
|---|---|
| producer | 8 in `stats/effect_size.py`; per-bridge in `ddqn_universe.py` |
| round-trip | `_internals/narrow.py` (parquet de/encode) |
| **consumer** | **`scripts/run_hypothesis.py` (4 lines)** — groups verdict counts by `(verdict, refutation_class.value)`, renders inline string |
| test | 1 site asserts UNDERPOWERED |

**Verdict**: minimally consumed. The four enum values get rendered
as strings in the CLI verdict landscape but no behavior branches on
them — no different formatting, no different exit code, no CI
flagging. Reviewer's "dead enum" framing is too harsh; "decorative
print, no decision" is more accurate.

### 2. `BridgeEdge.condition_desc: str | None` (`src/corroborate/graph/causal.py:186`)

| role | sites |
|---|---|
| producer | 0 (field exists, never set anywhere) |
| consumer | 0 (no reader anywhere) |

**Verdict**: fully dead field. Never set, never read. Documentation
in the docstring promises "optional condition annotation (e.g.
'when reward_scale > 0')" but no callsite supplies it. Either wire
or remove.

### 3. `assumption_violations: tuple[str, ...]` (across multiple sites)

| role | sites |
|---|---|
| producer | `analyses/paired_g.py` (heavy-tail / sample-size flags); `stats/effect_size.py` (DL pooling) |
| propagation | `bridge/bridge.py` (collects from fixture results into `BridgeEvaluation`); `runner/report.py` (writes to `BridgeReportEntry`, lands in `*.run.json`) |
| **consumer** | **0** — no human-readable view differentiates bridges with vs without violations; no CI gate; no in-bridge logic that escalates verdict to POW_INSUF when violations fire |

**Verdict**: typed and persisted to disk but unread. The snapshot
JSON carries the flags but the CLI / verdict landscape doesn't
surface them.

### 4. `CausalGraph` + `BridgeEdge` + `authored_graph()` + `promote_bridged_evidence()`

| role | sites |
|---|---|
| definition | all in `src/corroborate/graph/causal.py` |
| imports outside causal.py | `__init__.py`, `graph/__init__.py`, `graph/graph.py`, `analyses/_dowhy_internal.py`, `core/intervention.py` (all re-exports / type usage, none actually CALL `authored_graph` or `promote_bridged_evidence`) |
| **caller of `authored_graph`** | **0 outside the file** |
| **caller of `promote_bridged_evidence`** | **0 outside the file** |

**Verdict**: entire graph-walker layer is dead infrastructure. The
typed `BridgeEdge` metadata is well-designed but no human-readable
view, audit script, or CI gate consumes it.

## Wiring plan

Three rounds. Each round picks one dead/under-consumed primitive
and writes a concrete consumer. Order picked by "what surfaces the
most useful operational signal per LOC":

### Round 1 — Wire `authored_graph` into a claim-graph audit script

**Where**: new `scripts/audit_claim_graph.py`.

**What**: calls `authored_graph(DDQN_UNIVERSE_BRIDGES)` (and per
other hypothesis modules), walks the result, and emits:

- Per-node fan-out / fan-in.
- Bridges sharing `(source, target)` endpoints (the "structural
  pair" pattern the archived manifest tried to detect).
- Bridges sharing source but different targets, with verdict
  divergence (e.g., HELD on outcome but NO_EFFECT on link →
  mechanism-link decoupling).
- `promote_bridged_evidence(g)` output: which edges actually
  upgrade evidentiary level under the existing walker.

**Why first**: surfaces structural facts about the claim graph that
no current view shows. Forces real use of the graph-walker layer,
which will surface gaps that justify *targeted* additions (e.g.,
if the walker needs to distinguish disjoint-scope edges, that
becomes the second use case the manifest review was demanding).

**Cost**: one script, ~150 LOC. Zero framework changes.

**Acceptance**: script runs against `ddqn_universe`, prints the
audit. Either no surprises (we learn the graph is structurally
sound), or we get specific structural findings worth acting on.

### Round 2 — Differentiate `assumption_violations` in the snapshot renderer

**Where**: `scripts/run_hypothesis.py` (the CLI verdict landscape
printer) and the `*.run.json` schema.

**What**: when a bridge has non-empty `assumption_violations`,
render alongside the verdict (similar to how `refutation_class` is
already inline-rendered). Optionally a CI gate: "X new bridges
acquired assumption violations since last commit" flags as
operational signal.

**Why second**: this is the existing channel that already covers
the "DEGENERATE_PRIMITIVE" concern from the archived manifest.
The data is in the JSON; the gap is rendering. Replaces ~30 lines
of bridge-side prose ("data is degenerate; n_pairs=92 but
proportion=20.73 outside [0, 1]") with a structured rendering.

**Cost**: ~20 LOC change in `run_hypothesis.py`, no framework
schema change.

**Acceptance**: rerun the snapshot; verdict landscape shows
assumption-violation flags next to relevant bridges. Snapshot drift
test still passes (additive output).

### Round 3 — Wire or prune `BridgeEdge.condition_desc`

**Decision point**: after rounds 1 and 2, revisit whether
`condition_desc` has a real use. If round 1's audit needs to tag
edges with their partition-axis (n_step=1 vs n_step=10, polarity
GOAL vs SURVIVAL), `condition_desc` is the channel — set it from
the bridge author or derive from scope. If not, prune the field.

Per the no-proliferation rule: don't add data to a dead field
*hoping* for a use case. Either round 1 produces the use case or
the field gets removed.

**Cost**: either 1 field-setter + audit-script consumer, or
1 field deletion.

## Out of scope

- **Adding new `RefutationClass` values** — deferred until rounds
  1-3 produce a concrete consumer that needs `DATA_ORPHAN` or
  `DEGENERATE_PRIMITIVE` to *do* something distinct from
  `UNDERPOWERED`. The current "decorative print, no decision"
  consumer wouldn't change behavior on new values.
- **`BridgeEdge.partition_axis` / falsification-pair walker** —
  deferred until round 1's audit shows the existing
  `(source, target)` indexing is concretely insufficient.
- **Pearl rung-1/2/3 tier refinement** — deferred indefinitely;
  recoverable from `scope.meta.root_names()` per the second
  review.

## Method note for future audits

The audit's discipline: **count consumers, not producers**. A
typed primitive with 10 producers and 0 behavioral consumers is
ceremony. The framework's correct extensions are downstream from
real consumer needs, not upstream from prose patterns.

Repeat this audit annually or after any bridge-layer refactor. Add
new typed primitives only after consumer-count evidence shows the
gap.
