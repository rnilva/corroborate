# HYPOTHESIS_AS_GRAPH.md — the framework's organizing principle

## The principle

**A hypothesis is a causal graph, not a claim.** Bridges contribute
edges to that graph; the hypothesis is the graph itself.

```
Hypothesis = (V, E, evidence(E))
  V — typed measurable nodes (measurables, do-effects, outcomes)
  E — typed empirical edges (one per @claim_bridge)
  evidence(E) — Pearl-rung label per edge: unevaluated /
                correlational / causal_one_sided / refuted
```

The hypothesis file (`experiments/findings/<short>.py` or
`<short>/__init__.py`) is the container; its `BRIDGES` tuple is
the canonical artifact. Every other framework surface — findings
prose, run reports, audit tables — is interpretation of the
post-evaluation graph.

## How existing primitives realize it

| Concern | Primitive | Lives in |
|---|---|---|
| Typed node | `@measurable` / `DoEffect` / raw column | `corroborate.measurables` / `core.intervention` |
| Typed edge | `@claim_bridge` → `Bridge` | `corroborate.bridge.bridge` |
| Dataset-relative edge grouping | `(source_key, target_key, extent_hash)` | `corroborate.graph.causal.BridgeEdge` |
| Pearl rung per edge | `Tier` (associational / interventional / invariant) | `corroborate.graph.causal.Tier` |
| Post-eval evidence label | `BridgeEdge.evidentiary_level` | `corroborate.graph.causal.EvidentiaryLevel` |
| Graph topology (pre-eval) | `authored_graph(bridges)` | `corroborate.graph.causal` |
| Per-bridge string-ID-set key | `BridgeEvaluation.extent_hash` | `corroborate.bridge.bridge` |
| Evidence stamper (post-eval) | `evaluated_graph(bridges, post_eval)` | `corroborate.graph.causal` |
| Cluster query | `clusters_by_extent` + `cluster_verdict` + `ClusterVerdict` | `corroborate.graph.causal` |
| Direction composition | `compose_direction(edges)` | multiplicative algebra |
| Tier composition | `chain_tier(edges)` | min-tier along path |

No new framework primitives required — the framework's existing
machinery IS the implementation of the principle.

## Authoring discipline

### 1. A bridge tests ONE edge, not a claim

`@claim_bridge` carries `source`, `target`, optional `scope`. Each
bridge produces one `BridgeEdge` at evaluation time with one
verdict-derived `evidentiary_level`. **Causal-mechanism claims
do not live at the bridge level** — they live at the graph
level. An authored `Finding` names the bridges that compose the
claim; post-evaluation extent grouping is a dataset-relative
diagnostic, not scientific identity.

### 2. Bridge names should describe the edge, not the claim

Recommended pattern: `<predictor>_<relation>_<target>[__<scope>]`

`<relation>` ∈
- `corr` (ASSOCIATIONAL coupling)
- `do` (INTERVENTIONAL — do-effect)
- `partial_corr` (conditional independence)
- `backdoor` / `placebo` / `rcc` (identification / refutation
  sub-edges; pair with sibling bridges that share scope to form
  a refutation cluster)

Mechanism words (`mediates`, `causes`, `drives`, `attenuates`,
`amplifies`, `shadowed`, `rescues`, `refuted`) **do not belong in
bridge names**. They describe graph-level claims; a single bridge
cannot test them.

Migration target: bridges named with mechanism words should be
either (a) renamed to the edge they actually test, or (b)
sibling-bridge-paired so the cluster carries cluster-level
evidence and the mechanism name reads on the cluster, not the
individual bridge.

### 3. Refutation clusters are authored; extent grouping checks them

A claim like "X causes Y under scope S" requires a *cluster* of
bridges testing the same `(source, target)` over the intended
population. The corresponding `Finding` explicitly names those
bridges. Sharing a named scope predicate keeps that intent visible
and normally makes the bridges admit the same rows.

`extent_hash` provides a compact post-evaluation cross-check: matching
keys mean the bridges reported the same de-duplicated string-ID set
under a shared ID namespace. It does not compare row values or
multiplicity and cannot establish provenance, chronology, or semantic
scope equivalence. Author bridges with a named scope predicate (for
example a module-level constant in `_scope.py`) for code reuse and
clarity; do not treat the key as proof of the scientific claim.

The canonical 3-bridge pattern (used by
`reach_link_backdoor_ate_negative` + `_placebo_refuted` +
`_rcc_robust` in the frozen study's
`experiments/findings/ddqn/bias_correction.py`, `submission`
branch):

```python
from experiments.findings.ddqn._scope import DDQN_RELEVANT_SCOPE

@claim_bridge(source='jensen_gap', target='eval_best_burst_mean',
              scope=DDQN_RELEVANT_SCOPE, tier=Tier.ASSOCIATIONAL)
def reach_link_backdoor_ate_negative(...): ...

@claim_bridge(source='jensen_gap', target='eval_best_burst_mean',
              scope=DDQN_RELEVANT_SCOPE, tier=Tier.ASSOCIATIONAL)
def reach_link_placebo_refuted(...): ...

@claim_bridge(source='jensen_gap', target='eval_best_burst_mean',
              scope=DDQN_RELEVANT_SCOPE, tier=Tier.ASSOCIATIONAL)
def reach_link_rcc_robust(...): ...
```

All three import the same `DDQN_RELEVANT_SCOPE` constant. If they
report the same admitted string IDs, the post-evaluation graph groups
them under one `(jensen_gap, eval_best_burst_mean, extent_hash)` key.
The `Finding` remains the authored statement that they belong to one
claim.

Authors should share the named expression when they intend one scope,
but object identity is irrelevant: semantically equivalent expressions
that admit the same IDs produce the same key, while the same expression
can produce a different key as the record grows.

### 4. Scope determines evidence eligibility; the key summarises IDs

Two bridges with the same `(source, target)` but different scope
expressions often admit different string-ID sets and therefore have
different keys. Distinct scopes can nevertheless coincide on a finite
dataset, and equal ID-set keys say nothing about whether row contents
are equal. The key summarises the current admitted IDs, not the scope
expression or a persistent evidence snapshot.

Concrete examples in the frozen study's
`experiments/findings/ddqn/` (`submission` branch):
- REACH DoWhy trio: 3 bridges on `(jensen_gap, eval_best_burst_mean)`
  sharing `DDQN_RELEVANT_SCOPE` → cluster of 3 at one extent.
- Extreme-Q-div trio: 3 bridges on the same `(source, target)` but
  with `_EXTREME_Q_DIV_SCOPE` → distinct cluster of 3 at a
  different extent.
- MetaMaze γ pair (mean + median): 2 bridges sharing
  `_METAMAZE_GAMMA_SCOPE` → cluster of 2.
- Polarity-stratified `effective_horizon → outcome` bridges (GOAL
  scope vs SURVIVE scope): same `(source, target)`, syntactically
  different inline expressions, different admitted cells →
  distinct clusters (NOT corroborators, since polarity-disjoint
  sub-claims).

### 5. Empty evidence is determined by counts, not by a digest

When a bridge admits zero rows, `n_cells_in_scope == 0` is the
source of truth. Such bridges also share the grouping key for the empty
ID set. The converse is not guaranteed: a non-empty external frame
whose IDs are absent or non-string also contributes no strings to the
key. Consumers must not infer dataset emptiness, identity, or
provenance from `extent_hash` alone.

As data lands, the grouping can change. That is ordinary execution
metadata for a live record, not evidence of which snapshot came first.

## Causal claims as graph queries

The principle makes "does the bias channel hold?" a graph walk,
not a bridge lookup:

```python
def bias_channel_supported(g: CausalGraph) -> bool:
    paths = paths_between(g, source='do(DDQN)', target='outcome',
                          via='jens_gap')
    for p in paths:
        if all(e.evidentiary_level == 'causal_one_sided' for e in p):
            return True
    return False
```

Cluster-shaped claims compose the same way:

A claim like "the REACH bias-correction link cluster is
corroborated" is authored as a `Finding` module — the framework's
typed contract at `corroborate.core.finding.Finding`. The author
declares which bridges support the claim; the framework computes
the verdict via `composed_verdict`:

```python
# finding_reach_bias_link.py (frozen study, `submission` branch)
"""REACH bias-correction link is causally corroborated."""
from corroborate.graph.causal import ClusterVerdict
from experiments.findings.ddqn.bias_correction import (
    reach_link_backdoor_ate_negative,
    reach_link_placebo_refuted,
    reach_link_rcc_robust,
)

EXPECTED = ClusterVerdict.SUPPORTED
BRIDGES = (
    reach_link_backdoor_ate_negative,
    reach_link_placebo_refuted,
    reach_link_rcc_robust,
)
```

The parent hypothesis declares its findings in `FINDINGS`
(required on `Hypothesis` Protocol); the `run_hypothesis.py`
consumer iterates `hypothesis.FINDINGS`, evaluates each via
`composed_verdict(g, bridges=f.BRIDGES)`, surfaces drift when
`actual_verdict != f.EXPECTED`.

Findings narratives quote which subgraph in the graph they're
walking; the graph state is the canonical artifact. The
per-bridge `evidentiary_level` gives the Pearl-rung admit fact;
cluster-level "do all the bridges I cite admit?" is a structural
query the Finding declaratively asks, not a central aggregator on
the graph.

The framework provides `evaluated_graph`, `clusters_by_extent`,
`cluster_verdict`, `composed_verdict`, and the `ClusterVerdict`
enum in `corroborate.graph.causal`. See the frozen study's
`experiments/findings/ddqn/finding_*.py` (`submission` branch)
for hand-rolled findings covering cluster,
refutation-with-EXPECTED=REFUTED, and
asymmetric-envelope-across-scopes shapes.

### 3b. Scope clusters — pool bridge + meta-regression sibling

A claim of the shape "this effect corroborates population-wide
but heterogeneous along covariate C" requires a *scope cluster*:
a pool bridge that emits HELD_WITH_SCOPE_FLAG when between-stratum
I² ≥ 0.5, plus a meta-regression sibling whose coefficient on C
tests the cleavage. Both bridges share a NAMED scope predicate
for one authored population. Matching extent keys are a useful
execution-time cross-check; the `Finding` explicitly composes the
siblings.

The pool side uses the `stratified_arm_diff_pooled` analysis
primitive (`corroborate.analyses.stratified_arm_diff_pooled`),
which computes per-stratum **independent-samples** Cohen's d (not
paired Hedges' g — seeds aren't matched draws across arms in an
RL substrate, cf. the primitive's module docstring), DL-pools
across strata, and dispatches `random_effects_verdict` to emit
`HELD / HELD_WITH_SCOPE_FLAG / NO_EFFECT / POWER_INSUFFICIENT`.
Bridges that fixture it return the result's verdict directly:

```python
from experiments.findings.<short>._scope import SHARED_SCOPE

@claim_bridge(source=INTERVENTION, target='<outcome>',
              scope=SHARED_SCOPE, tier=Tier.ASSOCIATIONAL,
              predicted_direction='a_gt_b')
def effect_pools_across_envs(
    stratified_arm_diff_pooled: StratifiedArmDiffPooledResult,
) -> tuple[Verdict, RefutationClass | None]:
    return (
        stratified_arm_diff_pooled.verdict,
        stratified_arm_diff_pooled.refutation,
    )

@claim_bridge(source=INTERVENTION, target='<outcome>',
              scope=SHARED_SCOPE, tier=Tier.ASSOCIATIONAL)
def cleavage_at_C(meta_regression: MetaRegressionResult) -> Verdict:
    # CI on C's coefficient excludes zero → HELD; else NO_EFFECT.
    ...
```

The pool bridge's HELD_WITH_SCOPE_FLAG triggers the recipe:
"which covariate predicts the per-stratum effect?", answered by
the sibling meta-regression. The scope-cluster shape is the
empirical scope claim's structural unit. The authored `Finding`
defines that unit; a shared `(source, target, extent_hash)` is only
diagnostic agreement on the current de-duplicated string-ID set.

The methodology is in `ANALYSIS_RECIPE.md` §1.5 (`submission`
branch); the discipline against treating
heterogeneous-but-pooled-positive as plain
HELD is in `corroborate/bridge/verdict.py`. The choice of
independent-samples Cohen's d over paired Hedges' g (rejecting
seed-pairing for cross-env pooling) is documented in
`corroborate/analyses/stratified_arm_diff_pooled.py`'s module
docstring.

## Anti-patterns

- **Single bridge claiming mediation**. `X_mediates_Y` in the
  bridge name. Mediation is a graph-level claim. Rename to
  `X_partial_corr_Y__cond_Z` (the edge actually tested) and add
  refutation siblings (placebo, RCC) sharing scope for the
  mechanism story.
- **Findings prose without graph-walk justification**. A claim
  "DDQN helps via clip channel" in findings prose should
  reference the path through the graph that carries it (which
  edges? which `evidentiary_level`? which extent cluster?).
  Otherwise it's interpretation unanchored to the framework's
  evidence.
- **Duplicated inline scope when clustering is intended**. Repeating
  a predicate obscures whether sibling bridges intend the same
  population. Refactor it to a module-level constant (`_scope.py`) for
  authoring clarity. Expression object identity does not affect the
  extent key; only the admitted string IDs do.
- **`proportion_mediated` for mediation claims**. Deleted
  2026-05-18 (statistical case: ratio explodes near zero,
  lands outside [0, 1] under suppression, first-difference
  identification ≠ population slope under seed-coupled
  noise). The framework's mediation answer is graph-walk +
  cluster query, not a single ratio-of-means primitive;
  the canonical analyses are `partial_spearman` (rank-based,
  multicollinearity-robust) paired with `mediation_dowhy`
  (typed `linearity_status` diagnostic) at the same scope.
- **HP-conditioned bridges named as if HP-invariant**. A bridge
  on sync=500 scope tests an edge at sync=500. The bridge declaration,
  not an ID-set hash, carries that constraint. Don't paper over it in
  the bridge name.

## Connection to existing framework docs

(The internal design docs below other than `CLAUDE.md` are frozen
on the `submission` branch.)

- `CLAUDE.md` typing discipline applies at the node + edge level:
  no `Any`, frozen dataclasses, PEP 695 generics. Graph traversals
  use `corroborate.graph.Graph[N, M]`'s typed walks.
- `CORPUS_INTEGRITY.md` invariants protect node-level data
  (cells, measurables); this principle covers edge-level evidence.
- `SWEEP_PERSISTENCY.md` covers how edges' supporting data
  (runs/traces/measurements) flow through the corpus boundary.
- `PRIMITIVES_AUDIT.md` four-question test for primitives: this
  principle says "don't add chain-claim primitives — the existing
  graph + cluster query IS the chain primitive."
- `UNCONSUMED_PRIMITIVES_AUDIT.md` Round 3 documents the historical
  resolution that introduced extent-based grouping (replacing the
  typed-but-unwired `condition_desc` + `claim_id` fields and retiring
  auto-promotion). The grouping key is now explicitly non-authoritative.

## Honest scope

This document codifies the principle; it does not exhaustively
re-fact the existing causal-graph machinery, which lives in
`corroborate.graph.causal` and is the canonical source. When the
framework's primitives evolve, update them there; this doc just
points at what's already implemented and tells authors how to
use it.
