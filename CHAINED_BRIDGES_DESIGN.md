# CHAINED_BRIDGES_DESIGN.md — verdict-dependent scope for the bridge graph

## The gap

`HYPOTHESIS_AS_GRAPH.md` establishes that a hypothesis IS a causal
graph: bridges are typed edges, their composition expresses chains
of evidence. The framework's existing primitives realize this for
**cell-level scope** — a bridge's `scope: pl.Expr` filters which
cells contribute to its verdict.

The gap surfaces when one edge's testability depends on another
edge's verdict:

```
(disc, raw)              outcome_translation_consistent
   |                          ↓ HELD per env
   |--------- ON ENVS WHERE THIS HOLDS --------
                              ↓
(jens, eval_best_burst_raw_mean)    mech_to_outcome
```

The "ON ENVS WHERE THIS HOLDS" gate is a graph-topological
relationship, but the framework currently has no first-class
representation. The author is forced to either:

- **Collapse** the gate into a cell-level predicate:
  `pl.col('env_disc_raw_alignment') > 0.7` — works mechanically,
  but the dependency is buried in a polars expression rather than
  surfaced as graph topology.
- **Author independently** with no formal chain — the verdict-level
  dependency lives only in the dependent bridge's docstring.

Both lose the structural fact that the graph has a chain.

## Concrete instances of the gap (substrate)

| dependent bridge | precondition (currently buried as scope) | scope predicate today |
|---|---|---|
| cross_env Δ_jens → Δ_out_raw | per-env disc-raw outcome alignment | `pl.col('env_disc_raw_alignment') > 0.7` |
| polarity_conditional REACH | per-env REACH polarity | `finite_lt('env_reward_polarity', -0.3)` |
| mech→outcome at REACH | mech is operative per env | `vanilla mean(jens) > VANILLA_JENS_NOISE_FLOOR` |
| any post-training analysis | per-env convergence | `outcome.sd > epsilon` (saturation guard) |
| three-gate scope | G1 ∧ G2 ∧ G3 all HELD per env | manual AND of three predicates |

In each row the right column is **a derived measurable thresholded
to bool** — but the *substantive* precondition is a bridge claim
("this env has REACH dynamics"; "outcome-translation is internally
consistent"; "mech is operative"). The threshold-on-measurable
form loses the claim status.

## Why this matters

The framework's `feedback_scope_is_causal_bridge` discipline says
scope IS the load-bearing assumption in the mechanism's
derivation. When that assumption is itself testable evidence —
"this env's outcome-translation is consistent" is a *measurable
falsifiable claim* — then scope is doing double duty: gating the
dependent bridge AND silently asserting the precondition.

That silent assertion has two costs:

1. **No verdict surface for the precondition**. If outcome-
   translation alignment shifts (e.g., a new env with loose disc/
   raw correlation), there's nowhere for the framework to report
   "precondition refuted — dependent bridge's HELD reading no
   longer applies." It just silently filters cells out.

2. **No graph composition**. `compose_direction` chains causal
   direction across edges; `chain_tier` chains evidence rung. But
   there's no `chain_precondition` that says "this dependent
   bridge's verdict is conditional on these prior bridges' HELDs."

## Proposed primitive: `depends_on`

### Surface

```python
@claim_bridge(
    source='jensen_gap',
    target='eval_best_burst_raw_mean',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=MODULE_SCOPE,            # base cell-level scope
    depends_on=(outcome_translation_consistent,),  # NEW
    predicted_direction='a_lt_b',
)
def ddqn_outcome_scales_with_jens__xenv(...) -> Verdict:
    ...
```

`depends_on` is a tuple of bridges. Semantics:

1. **Evaluation order**: dependencies evaluate first; the
   dependent bridge sees their per-env verdicts.
2. **Effective scope**: the dependent's evaluated scope is
   `bridge.scope ∩ (per-env: all dependencies HELD)`.
3. **Verdict reporting**: the dependent bridge emits a verdict
   only on the dependency-satisfied subset. Cluster surface
   tracks which envs were excluded by a failed precondition.

### Composition with existing primitives

| Existing primitive | What it composes | What `depends_on` adds |
|---|---|---|
| `compose_direction(edges)` | causal direction along a path | independent — direction unaffected by preconditions |
| `chain_tier(edges)` | minimum evidence rung along a path | independent — tier unaffected |
| `cluster_verdict(bridges)` | AND-aggregate of member verdicts in one extent | new: `cluster_with_preconditions` that propagates SKIPPED-BY-PRECONDITION |
| `walk_subgraph(g, nodes)` | induced subgraph on nodes | new sibling: `walk_dependency_chain(bridge)` returning the precondition tree |

### New verdict semantics

Add to `ClusterVerdict`:

- `PRECONDITION_FAILED` — at least one dependency's verdict was
  not HELD on the envs that matter. The dependent didn't get a
  chance to test.
- Keep `UNDERPOWERED` for "dependent ran but didn't have power."

The renderer surfaces both distinctly: PRECONDITION_FAILED says
*the bridge couldn't be tested because evidence for its scope-
assumption is missing*; UNDERPOWERED says *the bridge ran but
n was insufficient*.

## Worked example: cross_env_mediation

Current (scope-predicate workaround):

```python
_XENV_SCOPE = DDQN_RELEVANT_SCOPE & (
    pl.col('env_disc_raw_alignment') > 0.7
)

@claim_bridge(source='jensen_gap', target='eval_best_burst_raw_mean',
              scope=_XENV_SCOPE, ...)
def ddqn_outcome_scales_with_jens__xenv(...): ...
```

Proposed (chained-bridge):

```python
# Precondition bridge — substantive edge claim.
@claim_bridge(
    source='mc_return__mean_axis_-1',
    target='eval_best_burst_raw_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=DDQN_RELEVANT_SCOPE,
    predicted_direction='a_gt_b',
)
def outcome_translation_consistent(stratified_spearman) -> Verdict:
    """Per-env Spearman ρ(disc, raw); HELD when ρ ≥ 0.7."""
    return partial_spearman_signed_verdict(
        stratified_spearman, threshold=0.7, sign=+1,
    )

# Dependent bridge — explicit chain.
@claim_bridge(
    source='jensen_gap',
    target='eval_best_burst_raw_mean',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=DDQN_RELEVANT_SCOPE,            # no special carve-out
    depends_on=(outcome_translation_consistent,),
    predicted_direction='a_lt_b',
)
def ddqn_outcome_scales_with_jens__xenv(
    cross_stratum_arm_diff_slope, ...
) -> Verdict:
    ...
```

The framework reads `depends_on`, evaluates the precondition first
per env, then evaluates the dependent only on HELD envs. The
dependency is now in the graph topology; the renderer can show:

```
finding_cross_env_mediation_chained:
  outcome_translation_consistent:  HELD (n_envs=6, ρ_pool=+0.87)
  ddqn_outcome_scales_with_jens__xenv:  HELD (n_envs=6, ρ=-0.83)
    [restricted to 6 envs where outcome_translation HELD]
    [6 envs SKIPPED-BY-PRECONDITION]
```

## Migration

- Drop-in: existing bridges with `depends_on=()` default behave
  unchanged.
- Substrate migration: convert the 4-5 "scope as silent
  precondition" patterns in the table above. Each becomes
  (precondition bridge, dependent bridge with `depends_on`).
- Polarity REACH/SURVIVE branching: each polarity-conditional
  bridge currently has `scope = pl.col('env_reward_polarity') < -0.3`
  as a cell-level filter. Under the proposal, this becomes a
  precondition bridge `env_is_reach_polarity` (per-env Spearman
  ρ(episode_length, mc_return) ≤ -0.3) that dependent REACH-cohort
  bridges declare via `depends_on=(env_is_reach_polarity,)`.

## Open design questions

1. **Verdict propagation when precondition is REFUTED**: does the
   dependent get reported as PRECONDITION_FAILED (skipped) or
   REFUTED (precondition's refutation propagates)? I'd argue
   SKIPPED — the dependent wasn't tested, only its premise was.
2. **Multi-precondition AND vs OR**: `depends_on=(A, B)` — does
   the dependent test on envs where both HELD (AND) or either
   HELD (OR)? AND is the natural default; OR is harder to motivate
   substantively.
3. **Cycle prevention**: precondition graph must be a DAG. The
   bridge registry enforces at module-load time (topological sort
   succeeds, else raise at startup).
4. **Cross-hypothesis dependencies**: a dependent in
   `experiments.findings.ddqn` referring to a precondition in
   `experiments.findings.ddqn_sweeps` — out of scope for v1; one
   hypothesis at a time.
5. **Time-varying preconditions**: a precondition that holds at
   one canonical scope but not at a sliced one — out of scope for
   v1; bridges carry one scope each.
6. **Inheritance of precondition scope**: does the dependent
   inherit the precondition's scope OR intersect with its own?
   Intersection — the dependent's scope can be tighter than the
   precondition's.

## Out of scope (for v1)

- Verdict-dependent measurables (a measurable whose definition
  switches on a bridge's verdict)
- Quantitative precondition strength (the dependent's verdict
  weighted by precondition's effect-size)
- UI / rendering changes beyond surfacing PRECONDITION_FAILED in
  the cluster table

## Naming / framing

Two readings of `depends_on`:

- **"Precondition"** — emphasizes the gating semantics ("test only
  where prereq is satisfied").
- **"Premise"** — emphasizes the epistemic dependency ("the
  dependent claim presupposes the prior claim").

Substrate authors tend to think in the first; framework readers
benefit from the second. Either name works; `depends_on` is
neutral and matches Python idiom.

## Relationship to `HYPOTHESIS_AS_GRAPH.md`

This proposal does NOT introduce new edge types or change the
Pearl rung structure. It introduces a **dependency relation
between edges** — orthogonal to the causal-direction and
evidence-rung relations.

Under the V/E framing:
- V — nodes unchanged
- E — edges unchanged, but each edge carries an additional
  attribute `depends_on: tuple[edge_id, ...]`
- evidence(E) — extended with a `SKIPPED_BY_PRECONDITION` state
  alongside the existing rung labels

The hypothesis-as-graph principle survives; the graph just
acquires a second relation on its edges.

## Not a refactor

The substrate works today with the scope-predicate workaround.
This proposal is forward-looking: as the bridge library grows and
more chained-precondition patterns recur, the workaround burden
compounds. The right time to land this is when the third or
fourth instance of "buried precondition in scope" surfaces — we
have four candidates above, which probably reaches threshold.
