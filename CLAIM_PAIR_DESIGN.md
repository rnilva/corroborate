# Claim pair — Pearl-honest interventions as paired Claim contrasts

## Status

Open design.

## Motivation

The substrate currently expresses interventions through three
related but structurally distinct primitives:

1. **`Intervention(slot_path, replacement)`** — a structural swap
   at a named slot in the Claim graph. The `replacement` is a
   `partial(claim, ...)` that takes the slot's place.
2. **`DoEffect(treatment, baseline)`** — composes `Intervention`
   tuples into treatment vs. baseline arms. `treatment=()` and
   `baseline=()` both canonicalize to the empty arm.
3. **No mechanism for HP / wrapper changes.** Reward-scale,
   sync_period, n_step, γ — all live as scope predicates on
   bridges, not as interventions. The `Intervention.wrap_with`
   gap is documented in memory `project_wrapper_intervention_
   gap` and deferred from verdict consolidation.

This three-shape split has accumulated structural tensions:

- **Pearl rung-2 conflation.** In a fully controlled simulation,
  every config knob is `do()`-able: `do(bootstrap=ddqn)`,
  `do(rs=0.1)`, `do(γ=0.99)`, `do(sync_period=10000)` are all
  formally interventional. The framework currently calls only
  Claim-graph slot swaps "interventions"; the rest are scope
  predicates. This is a *practitioner-readability* convention,
  not a Pearl-formal distinction. The `Tier.INTERVENTIONAL` enum
  value carries the convention, not the structural truth.

- **HP-as-intervention isn't expressible.** A bridge that asks
  "do(rs=0.1) effect on outcome, no algo swap" cannot be authored
  as `tier=INTERVENTIONAL` today — there's no `Intervention` for
  rs. The bridge has to embed rs in scope (`reward_scale == 0.1`)
  and use the algo-swap intervention for the source. Two
  conceptually separate manipulations get conflated.

- **`Intervention.slot_path` is bookkeeping, not structure.** The
  `slot_path='bootstrap'` string tells the sweep WHERE to apply
  the partial. But the partial itself (`partial(bootstrap,
  greedification=double_greedify)`) IS the structural change —
  the replacement Claim. The slot path is the framework's
  internal handle for the application site; structurally, the
  intervention is just "this Claim instead of that Claim".

- **Symmetric paired contrasts don't fit.** A bridge that asserts
  "do(lr=0.001) vs do(lr=0.0001) effect" wants both arms to be
  `partial(adam, lr=...)`. The current `DoEffect(treatment,
  baseline)` shape biases toward "treatment swap, baseline
  empty" because the empty baseline canonicalizes to the
  substrate default. Symmetric interventions break this pattern.

- **`Tier` axis muddles two distinctions.** Pearl's ladder
  (associational / interventional / counterfactual) is one
  axis; "did the bridge use an algo swap as source?" is another.
  The current Tier enum mixes them. ASSOCIATIONAL bridges that
  pair-test using `paired_link_per_burst` aren't observational
  in Pearl's sense — they're paired-do() with an associational
  *target*. The conflation makes Tier hard to read.

The principle we want to express:

> **An intervention is a paired sweep of two Claims (or two
> partial-applications of the same Claim). The contrast is the
> two callables, full stop. There's no slot_path indirection,
> no asymmetric "treatment vs empty baseline", no separate
> primitive for HP/wrapper interventions.**

This collapses `Intervention` + `DoEffect` + the wrapper-
intervention gap into one recursive structure: paired Claim
contrasts.

## Principle: claim_pair as the only intervention primitive

```python
@dataclass(frozen=True, slots=True)
class ClaimPair[**P, T]:
    """A paired Claim contrast. Both arms are Claims (raw or
    partial-applied). The framework runs each in a paired sweep,
    pairing on declared `pair_by` keys.

    The arm_key derives from `canonical_str(arm)` of each side —
    no slot_path, no DoEffect composition. Pearl-honest by
    construction: every bridge with this source is testing
    `do(treatment) − do(baseline)`."""
    treatment: Claim[P, T] | functools.partial[T]
    baseline: Claim[P, T] | functools.partial[T]
    name: str  # author-chosen label for the contrast

    @property
    def treatment_arm_key(self) -> str:
        return canonical_str(self.treatment)

    @property
    def baseline_arm_key(self) -> str:
        return canonical_str(self.baseline)
```

A `ClaimPair` is the only thing a bridge's `source` can be when the
bridge tests an intervention. The two arms are the contrast;
nothing else.

## Examples

### DDQN vs vanilla — Claim swap (was: `DoEffect(DDQN_SWAP, ())`)

```python
DDQN_VS_VANILLA = ClaimPair(
    name='ddqn_vs_vanilla',
    baseline=bootstrap,
    treatment=partial(bootstrap, greedification=double_greedify),
)
```

The two arms are explicit. No `slot_path='bootstrap'` because the
two Claims ARE what's running at that slot.

### Reward-scale 0.1 vs 1.0 — wrapper intervention

```python
RS_LOW_VS_DEFAULT = ClaimPair(
    name='rs_0p1_vs_1p0',
    baseline=partial(env_wrapper, reward_scale=1.0),
    treatment=partial(env_wrapper, reward_scale=0.1),
)
```

Closes the wrapper-intervention gap. A bridge sourced on this
pair tests `do(rs=0.1) − do(rs=1.0)`, not "DDQN under rs=0.1
scope". Pearl-honest: this is interventional on rs, not
associational.

### γ sweep — symmetric HP intervention

```python
GAMMA_HIGH_VS_LOW = ClaimPair(
    name='gamma_0p99_vs_0p9',
    baseline=partial(dqn, gamma=0.9),
    treatment=partial(dqn, gamma=0.99),
)
```

A first-class do(γ) intervention. Each γ sweep cell becomes a
true paired comparison rather than a stratum within a scope
predicate.

### Composed interventions — DDQN under rs=0.1

A bridge testing the DDQN swap *within* the rs=0.1 stratum still
uses the DDQN-vs-vanilla pair as its source, with rs=0.1 in
scope:

```python
@claim_bridge(
    source=DDQN_VS_VANILLA,
    target='eval_best_burst_mean',
    scope=(pl.col('reward_scale') == 0.1),
    predicted_direction='a_gt_b',
)
def ddqn_rescues_at_low_rs(paired_g: PairedGResult) -> Verdict: ...
```

Same as today. The refactor doesn't change this case; it just
removes the slot_path + DoEffect ceremony.

### Dual interventions — DDQN AND rs=0.1, jointly

For a bridge that tests do(bootstrap=ddqn AND rs=0.1) vs
do(bootstrap=vanilla AND rs=1.0), the source is a composed pair:

```python
DDQN_AT_LOW_RS = ClaimPair(
    name='ddqn_at_rs_0p1_vs_vanilla_at_rs_1p0',
    baseline=partial(dqn, bootstrap=bootstrap, reward_scale=1.0),
    treatment=partial(
        dqn,
        bootstrap=partial(bootstrap, greedification=double_greedify),
        reward_scale=0.1,
    ),
)
```

The composition is a single `partial`, not a `DoEffect` of two
`Intervention` tuples. Multi-slot interventions become natural
nested partials, not tuple compositions.

## Framework consequences

### `Tier` enum simplifies

Under the refactor, **every bridge with `source: ClaimPair` is
Pearl rung-2 by construction**. The Tier enum stops needing to
distinguish INTERVENTIONAL vs ASSOCIATIONAL at the source level.

Two natural replacements:

- **Drop `Tier` entirely.** The bridge body's analysis primitive
  carries the rung information: `paired_g` on the
  ClaimPair-derived Δ_outcome is interventional;
  `paired_link_per_burst` on observed-on-observed deltas is
  associational; DoWhy-based primitives are causal/counterfactual.
  No enum needed.

- **Reframe `Tier`** to mean "what the analysis observes":
  `Tier.OUTCOME` (paired_g on observed outcome differences),
  `Tier.LINK` (correlation between observed mediator deltas),
  `Tier.STRUCTURAL` (DoWhy / mediation decomposition). This is
  about the question being answered, not the manipulation.

The second is mildly clearer for documentation but adds enum
maintenance. The first is honest and minimal.

### `arm_key` derives uniformly

Currently arm_keys are derived through
`combined_arm_key(intervention_tuple)` which composes individual
`Intervention.slot_path + canonical_str(replacement)` into a
sorted-joined string. Under the refactor:

```python
treatment_arm_key = canonical_str(claim_pair.treatment)
baseline_arm_key = canonical_str(claim_pair.baseline)
```

`canonical_str` already handles partial-of-partial recursion and
Claim canonicalization. No bookkeeping needed beyond that.

### `walk_paths` works the same

The two arms are Claims (or partial-applied Claims). The
existing walker iterates over both at sweep time — no change.
`MEASURABLES` declarations and trace-context computation are
unaffected.

### Admission gates simplify

Current gates check `bridge.source` for typed-contract
compliance (e.g., `Intervention.replacement` is callable). Under
the refactor:

- The gate becomes "both `claim_pair.treatment` and
  `claim_pair.baseline` are callable, both have compatible
  signatures, both are `@claim`-decorated or `partial` of one".
- Arm-key collision check (treatment != baseline canonical_str)
  is the symmetric form of the existing
  `treatment_arm_key != baseline_arm_key` gate.

### Wrapper-intervention gap closes

`project_wrapper_intervention_gap` documented that wrapper / HP
changes weren't first-class interventions. Under the refactor,
they're just `ClaimPair(treatment=partial(env_wrapper, rs=0.1),
baseline=partial(env_wrapper, rs=1.0))`. The gap is the
asymmetry of `DoEffect(treatment, baseline)` favoring "treatment
swap, baseline empty"; the symmetric pair shape is natively
fine.

### HP-sweep-as-intervention becomes natural

Each γ value in a γ sweep can be authored as its own
ClaimPair against a baseline γ. This may explode bridge count if
done literally — but more honestly, it makes explicit which γ
contrasts are *causal claims* and which are HPO-style. Bridges
that argmax over γ values aren't claim_pairs; they're a
different shape (probably outside the bridge primitive — see
HPO-vs-causal-inference distinction in CLAUDE.md).

## Substrate transformation table

| Today | Refactored |
|---|---|
| `Intervention(slot_path='bootstrap', replacement=partial(bootstrap, greedification=double_greedify))` | `partial(bootstrap, greedification=double_greedify)` (raw, no wrapper) |
| `DoEffect(treatment=(SWAP,), baseline=())` | `ClaimPair(baseline=bootstrap, treatment=partial(bootstrap, ...))` |
| `INTERVENTION = DoEffect(...)` | `INTERVENTION = ClaimPair(...)` |
| `Tier.INTERVENTIONAL` on bridge w/ algo swap | (drop or rename — see Tier section) |
| `Tier.ASSOCIATIONAL` on bridge w/ paired link | (drop or rename) |
| Wrapper change as scope predicate | `ClaimPair` on the wrapper's partial — first-class intervention |
| HP sweep as scope strata | `ClaimPair` per HP value (or stay as scope if HPO not causal) |

## Migration

### Step 1 — author `ClaimPair`, deprecate `Intervention` + `DoEffect`

Add `claim_pair.py` to `corroborate/core/`. Re-export through
`corroborate.core.intervention` for back-compat. The old
`Intervention` and `DoEffect` classes stay but emit
`DeprecationWarning`.

### Step 2 — substrate constants migrate

`experiments/findings/ddqn_universe.py`'s `DDQN_SWAP` and
`INTERVENTION` constants get rewritten as `ClaimPair`. Same for
`adaptive_dqn` variants. Adaptive_dqn / expectile siblings get
their own `ClaimPair` instead of separate `DoEffect`s.

### Step 3 — bridges migrate

Bridges that reference `INTERVENTION` (the module-level
DoEffect) now reference the module-level `ClaimPair`. The
`@claim_bridge(source=...)` field accepts both types during
migration; final cleanup drops `DoEffect`.

### Step 4 — Tier handling

Either:
- Drop `Tier` entirely. Bridges remove the `tier=` field; the
  framework infers from the analysis primitive (which carries
  rung information).
- Rename Tier semantics to "what is observed" (OUTCOME / LINK /
  STRUCTURAL). Mechanical s/INTERVENTIONAL/OUTCOME/ etc.

### Step 5 — wrapper interventions

Author `env_wrapper` Claim equivalents (currently wrappers are
config dicts in YAML). The `partial(env_wrapper, rs=0.1)` form
becomes valid. Existing `reward_scale_low_fourrooms` /
`reward_scale_sweep` bridges can be re-encoded as
ClaimPair-based interventions on `env_wrapper`.

## Risks and edge cases

### `partial` of a Claim that's already partial

`partial(partial(adam, lr=0.001), b1=0.95)` — the outer partial
is over a partial. `canonical_str` already recurses through
nested partials (verified in
`tests/test_hypothesis.py::test_canonical_str_partial_*`). No
change needed; the test coverage proves it works.

### Identity case: `partial(claim)` with no kwargs

`partial(bootstrap)` ≡ `bootstrap`. canonical_str returns the
same string. Edge case, but the equivalence is honest.

### Symmetric pairs where both are partial

The `partial(adam, lr=0.001)` vs `partial(adam, lr=0.0001)`
case. Canonical strings differ, arm_keys derive correctly. The
existing `combined_arm_key` was biased toward "treatment swap,
baseline empty" (empty canonicalized to 'baseline'); the new
shape doesn't have this asymmetry — both arms are explicit.

### Multi-slot interventions

`partial(dqn, bootstrap=..., optimizer=..., replay=...)` —
multiple slots set in one partial. The composed partial is the
arm. canonical_str handles arbitrary kwargs. The DoEffect
tuple-of-Interventions pattern collapses into a single nested
partial.

### Counterfactual bridges (rung 3)

DoWhy backdoor / placebo / refutation bridges currently use
custom analyses that take a corpus and apply structural causal
modeling. Under the refactor, their `source` could be a
`ClaimPair` (the treatment contrast) and the analysis primitive
would handle the counterfactual machinery. Same shape; the
analysis carries the rung-3 logic.

## What's NOT in scope for this refactor

- **Multi-arm sweeps** (>2 claims). HPO-style "sweep many γ" is
  intentionally not unified into ClaimPair. Each γ-pair is its
  own ClaimPair if it's a causal claim; if it's HPO it shouldn't
  be a bridge at all.

- **Stochastic interventions** (e.g., do(γ ~ N(0.99, 0.01))).
  Currently not supported; out of scope.

- **Verdict-layer changes**. The verdict enum
  (HELD / NO_EFFECT / POWER_INSUFFICIENT / INADMISSIBLE) is
  unchanged.

- **Cache schema changes**. Arm keys derive differently but the
  parquet column names are stable.

## Acceptance

The refactor lands when:

1. `ClaimPair` exists, tested, documented.
2. All existing `Intervention` + `DoEffect` usages migrated
   (mechanical s/.../.../).
3. The wrapper-intervention gap memory note can be marked
   resolved.
4. The Tier decision (drop or rename) is committed.
5. At least one new bridge demonstrates a wrapper /
   HP-as-intervention test that previously couldn't be authored
   (rs=0.1 vs rs=1.0 outcome, no algo swap).
6. `findings_chain_amplifier_link_active_in_bounded_q` and
   peers run unchanged on the new arm-key derivation.

The principle of the refactor is captured in one sentence: *"The
intervention is the two Claims; nothing else."* If a future
reader can recover that from the code without reading this doc,
the refactor succeeded.
