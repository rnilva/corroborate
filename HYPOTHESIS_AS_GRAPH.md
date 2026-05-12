# HYPOTHESIS_AS_GRAPH.md — the framework's organizing principle

## The principle

**A hypothesis is a causal graph, not a claim.** Bridges contribute
edges to that graph; the hypothesis is the graph itself.

```
Hypothesis = (V, E, evidence(E))
  V — typed measurable nodes (measurables, do-effects, outcomes)
  E — typed empirical edges (one per @claim_bridge)
  evidence(E) — Pearl-rung label per edge: unevaluated /
                correlational / causal_one_sided / causal_bridged
                / refuted
```

The hypothesis file (`experiments/findings/<short>.py`) is the
container; its `BRIDGES` tuple is the canonical artifact. Every
other framework surface — findings prose, run reports, audit
tables — is interpretation of the post-evaluation graph.

## How existing primitives realize it

| Concern | Primitive | Lives in |
|---|---|---|
| Typed node | `@measurable` / `DoEffect` / raw column | `corroborate.measurables` / `core.intervention` |
| Typed edge | `@claim_bridge` → `Bridge` | `corroborate.bridge.bridge` |
| Edge identity | `(source_key, target_key, condition_desc)` | `corroborate.graph.causal.BridgeEdge` |
| Pearl rung per edge | `Tier` (associational / interventional / invariant) | `corroborate.graph.causal.Tier` |
| Post-eval evidence label | `BridgeEdge.evidentiary_level` | `corroborate.graph.causal.EvidentiaryLevel` |
| Graph topology (pre-eval) | `authored_graph(bridges)` | `corroborate.graph.causal` |
| Graph topology (post-eval) | `evaluated_graph(bridges, evals)` | (under wiring) |
| Bridge corroboration | `promote_bridged_evidence(g)` | `corroborate.graph.causal` |
| Direction composition | `compose_direction(edges)` | multiplicative algebra |
| Tier composition | `chain_tier(edges)` | min-tier along path |

No new primitives required — the framework's existing machinery
IS the implementation of the principle.

## Authoring discipline

### 1. A bridge tests ONE edge, not a claim

`@claim_bridge` carries `source`, `target`, optional `scope`. Each
bridge produces one `BridgeEdge` at evaluation time with one
verdict-derived `evidentiary_level`. **Causal-mechanism claims
do not live at the bridge level** — they live at the graph
level, derived from edge-promotion.

### 2. Bridge names should describe the edge, not the claim

Recommended pattern: `<predictor>_<relation>_<target>[__<scope>]`

`<relation>` ∈
- `corr` (ASSOCIATIONAL coupling)
- `do` (INTERVENTIONAL — do-effect)
- `partial_corr` (conditional independence)
- `backdoor` / `placebo` / `rcc` (identification / refutation
  sub-edges; pair with sibling bridges for `causal_bridged`)

Mechanism words (`mediates`, `causes`, `drives`, `attenuates`,
`amplifies`, `shadowed`, `rescues`, `refuted`) **do not belong in
bridge names**. They describe graph-level claims; a single bridge
cannot test them.

Migration target: bridges named with mechanism words should be
either (a) renamed to the edge they actually test, or (b)
sibling-bridge-paired so the cluster carries `causal_bridged`
evidence and the mechanism name reads on the cluster, not the
individual bridge.

### 3. Causal claims need REFUTATION CLUSTERS, not single bridges

A claim like "X mediates Y under scope S" requires the framework's
evidence-promotion mechanism: ≥ 2 INTERVENTIONAL bridges with HELD
verdicts on the **same edge** (same `(source, target, condition_desc)`
triple) get promoted to `causal_bridged` by
`promote_bridged_evidence`.

The canonical 3-bridge pattern (already used by
`reach_link_backdoor_ate_negative` + `_placebo_refuted` +
`_rcc_robust`):

```python
@claim_bridge(source=DOEFFECT, target='outcome', scope=S,
              tier=Tier.INTERVENTIONAL)
def edge_backdoor_ate(...): ...

@claim_bridge(source=DOEFFECT, target='outcome', scope=S,
              tier=Tier.INTERVENTIONAL)
def edge_placebo_refuted(...): ...

@claim_bridge(source=DOEFFECT, target='outcome', scope=S,
              tier=Tier.INTERVENTIONAL)
def edge_rcc_robust(...): ...
```

Authors should aim for this triple-pattern wherever a causal claim
is intended. A solo bridge is `causal_one_sided` at best — not
enough for the graph to upgrade.

### 4. Scope is part of edge identity

Two bridges with the same `(source, target)` but different scope
test different sub-claims, not the same edge. The framework's
edge identity is `(source_key, target_key, condition_desc)`;
`condition_desc` is populated from `str(bridge.scope)`.
Scope-distinct edges do NOT corroborate for `promote_bridged_evidence`.

This matters because most do(DDQN) → outcome bridges share the
same `(source, target)` but condition on different scopes (G1,
dormant, polarity-partitioned, etc.). Without scope-fingerprint,
the graph would over-promote.

## Causal claims as graph queries

The principle makes "does the bias channel hold?" a graph walk,
not a bridge lookup:

```python
# Mechanism claim: do(DDQN) → jens_gap → outcome
def bias_channel_supported(g: CausalGraph) -> bool:
    paths = paths_between(g, source='do(DDQN)', target='outcome',
                          via='jens_gap')
    for p in paths:
        if all(e.evidentiary_level in
               ('causal_one_sided', 'causal_bridged') for e in p):
            return True
    return False
```

Findings narratives quote which paths in the graph they're
walking; the graph state is the canonical artifact. The
verdict-promotion algebra in `promote_bridged_evidence` +
`chain_tier` + `compose_direction` gives the framework-honest
verdict on any graph-level claim.

## Anti-patterns

- **Single bridge claiming mediation**. `X_mediates_Y` in the
  bridge name. Mediation is a graph-level claim. Rename to
  `X_partial_corr_Y__cond_Z` (the edge actually tested) and add
  refutation siblings (placebo, RCC) for the mechanism story.
- **Findings prose without graph-walk justification**. A claim
  "DDQN helps via clip channel" in `findings_*.md` should
  reference the path through the graph that carries it (which
  edges? which `evidentiary_level`?). Otherwise it's interpretation
  unanchored to the framework's evidence.
- **`proportion_mediated` for mediation claims**. Deprecated;
  ratio-of-noisy-means. The framework's mediation answer is
  graph-walk + promotion, not a single ratio-of-means primitive.
- **HP-conditioned bridges named as if HP-invariant**. A bridge
  on sync=500 scope tests an edge at sync=500. The `condition_desc`
  carries that constraint. Don't paper over it in the bridge name.

## Connection to existing framework docs

- `CLAUDE.md` typing discipline applies at the node + edge level:
  no `Any`, frozen dataclasses, PEP 695 generics. Graph traversals
  use `corroborate.graph.Graph[N, M]`'s typed walks.
- `CORPUS_INTEGRITY.md` invariants protect node-level data
  (cells, measurables); this principle covers edge-level evidence.
- `SWEEP_PERSISTENCY.md` covers how edges' supporting data
  (runs/traces/measurements) flow through the corpus boundary.
- `PRIMITIVES_AUDIT.md` four-question test for primitives: this
  principle says "don't add chain-claim primitives — the existing
  graph + promotion algebra IS the chain primitive."

## Honest scope

This document codifies the principle; it does not exhaustively
re-fact the existing causal-graph machinery, which lives in
`corroborate.graph.causal` and is the canonical source. When the
framework's primitives evolve, update them there; this doc just
points at what's already implemented and tells authors how to
use it.
