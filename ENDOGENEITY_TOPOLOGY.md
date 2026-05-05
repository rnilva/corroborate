# Endogeneity from topology — design

Companion to `ADMISSION_GATES_DESIGN.md`. The `EXOGENOUS_SOURCE`
gate enforces "Tier.INTERVENTIONAL bridges require an endogenous
source." This doc specifies what *truly endogenous* means, and
shows the framework can derive it from existing structural
information without a substrate-supplied registry.

## Status

Open design. Two questions pinned at the bottom; resolution
needed before implementation.

## Motivation

A first cut of `EXOGENOUS_SOURCE` would treat
"source ∈ `registered_names()`" as the endogenous test — i.e., any
`@measurable`-decorated column passes the gate. This has a
loophole: an author can wrap an HP leaf in a trivial measurable
and the gate sees it as endogenous.

Today's `effective_horizon` measurable IS exactly this:

```python
@measurable(reads=('gamma',))
def effective_horizon(record):
    g = record['gamma']
    return 1.0 / (1.0 - g)
```

`effective_horizon = 1 / (1 - γ)` is a deterministic algebraic
function of the leaf `gamma`. A bridge sourced on
`effective_horizon >= 50` is observationally equivalent to one
sourced on `gamma >= 0.98`. The migration `gamma == 0.99 →
effective_horizon >= 50` is **cosmetic** unless the measurable's
closure actually touches cell-derived state.

The structurally-honest definition of endogenous: a measurable's
*transitive reads closure* must include at least one column that
came from running the cell — a trajectory output.

## Topological classification

Every column in a cell record falls into exactly one of four
categories, determined by structural information the framework
already tracks:

| category | how the framework identifies it | endogenous? |
|---|---|---|
| **Leaf scalar** | appears in `walk_paths(claim).leaves` for the cell's bound composition | no — author-chosen HP |
| **Framework metadata** | in `_STANDARD_METADATA` (`env_name`, `seed`, `id`, `arm_key`, `corpus`, …) | no — provenance, not cell-derived |
| **Registered measurable** | name in `registered_names()` (has `@measurable` decorator) | recurse via `transitive_reads`; endogenous iff at least one base case is a trajectory output |
| **Trajectory output** | by elimination: in cell record AND not a leaf AND not metadata AND not registered | yes — emitted by running the claim |

Trajectory outputs aren't declared anywhere — they're the
complement. `mc_return`, `jensen_gap`, `done`, `reward`,
`predicted_q_at_start`, etc. live in the cell record because the
substrate's `dqn(...)` Claim emits them at runtime. The framework
doesn't need to know their names ahead of time; it knows everything
that *isn't* a trajectory output (leaves, metadata, derived
measurables), and trajectory-output is the topological residual.

## Algorithm

```python
def is_truly_endogenous(
    name: str, claim: Callable,
) -> bool:
    """A column is truly endogenous iff its closure depends on
    cell-trajectory state, not just on HP leaves or env metadata.

    Resolution order:
    1. Leaf scalar (per `walk_paths`) → exogenous (False).
    2. Framework metadata (`_STANDARD_METADATA`) → False (it's
       provenance, not cell dynamics).
    3. Registered measurable: recurse over `transitive_reads`;
       endogenous iff ANY base case in the closure is endogenous.
    4. Otherwise → trajectory output by elimination → True.
    """
    leaves = set(walk_paths(claim).leaves)
    if name in leaves:
        return False
    if name in _STANDARD_METADATA:
        return False
    if name in registered_names():
        m = lookup_measurable(name)
        return any(
            is_truly_endogenous(r, claim)
            for r in transitive_reads(m)
        )
    return True  # trajectory output by elimination
```

The function operates entirely on framework primitives:
`walk_paths` (already used for arm_key construction),
`_STANDARD_METADATA` (framework constant), `registered_names()` /
`lookup_measurable` / `transitive_reads` (existing measurable
registry API).

## Examples

Without any substrate input beyond the existing `@measurable`
declarations:

| name | walk_paths leaf? | metadata? | registered? | trajectory? | truly endogenous? |
|---|---|---|---|---|---|
| `gamma` | yes | — | — | — | **no** — leaf |
| `optimizer.inner.lr` | yes | — | — | — | **no** — leaf |
| `replay.capacity` | yes | — | — | — | **no** — leaf |
| `env_name` | — | yes | — | — | **no** — metadata |
| `seed`, `id`, `arm_key`, `corpus` | — | yes | — | — | **no** — metadata |
| `done` | no | no | no | by elimination | **yes** — trajectory |
| `mc_return` | no | no | no | yes | **yes** |
| `jensen_gap` | no | no | no | yes (claim-emitted) | **yes** |
| `predicted_q_at_start` | no | no | no | yes | **yes** |
| `effective_horizon` (today: `reads=('gamma',)`) | — | — | yes | — | **no** — closure is `{gamma}` (leaf only) |
| `effective_horizon` (proposed: `reads=('gamma', 'done')` via `bootstrap_fraction`) | — | — | yes | — | **yes** — closure includes `done` (trajectory) |
| `bootstrap_fraction` (`reads=('done',)`) | — | — | yes | — | **yes** — closure is `{done}` |
| `q_divergence_score` (`reads=('jensen_gap', 'gamma', 'r_max via env_name')`) | — | — | yes | — | **yes** — `jensen_gap` is trajectory |
| `r_max` (`reads=('env_name',)`) | — | — | yes | — | **no** — closure is `{env_name}` (metadata only) |
| `log_action_dim`, `log_obs_dim`, `log_horizon` (`reads=('env_name',)`) | — | — | yes | — | **no** — env-feature lookups |
| `mc_return_first_quarter` (`reads=('mc_return',)`) | — | — | yes | — | **yes** — closure is `{mc_return}` |

The classification matches the substrate-author's intuition:
trajectory-derived measurables pass; pure HP transforms and
env-feature lookups are filtered as exogenous-equivalent. No
substrate-supplied trajectory-key list required.

## Implications

**The substrate doesn't ship anything new for the gate.** It just
ships `@measurable`-decorated functions with honest `reads=`
declarations; the framework derives endogeneity from those.

**The endogenous frontier widens by writing measurables that touch
trajectory state.** To elevate γ from exogenous-HP to
endogenous-mediator, the substrate redefines `effective_horizon`
to include `bootstrap_fraction` (which reads `done`). The
framework's classification updates automatically.

**Synthetic-wrapper measurables are exposed.** Today's
`effective_horizon`, `r_max`, `log_action_dim`, `log_obs_dim`,
`log_horizon` all close over only leaves or env metadata. Under
the topological check they're correctly exogenous-equivalent —
not endogenous — even though they're `@measurable`-registered.

**Bridge migrations using synthetic measurables need re-checking.**
Specifically, the recent migrations from `gamma == 0.99` /
`gamma == 0.999` to `effective_horizon >= 50` /
`effective_horizon >= 500` (commits 026f49a, 9c6f310) are
tautological dressing under the strict definition. They'll either
need:
1. Redefinition of `effective_horizon` to genuinely depend on cell
   dynamics (the `1 / (1 − γ × bootstrap_fraction)` form), then
   re-verification of cell sets and verdicts, OR
2. Reverting to honest scope (`gamma == X`) and acknowledging the
   `EXOGENOUS_SCOPE` WARN, OR
3. Designing a different endogenous delegate that captures the
   chain-depth amplifier intent.

## Open questions

### Q1: Where does the gate get `claim` from?

`is_truly_endogenous(name, claim)` requires a base claim
composition to resolve `walk_paths(claim).leaves`. The set of
leaves depends on which Claim the bridge runs against — the
substrate's `dqn(...)` exposes one set of leaves; a different
substrate's claim exposes another.

Today the runner constructs the composed claim via
`apply_interventions(base, treatment)` and passes it to the
substrate's `Runner` callable. The bridge framework's
`Bridge.evaluate` doesn't see the base claim — it operates on a
DataFrame of cells.

Three plausible resolutions:

1. **Pass `base` to `Bridge.evaluate`.** The runner threads it
   through. Concrete and explicit, but widens the framework's
   contract: bridges are no longer pure pl.DataFrame consumers.

2. **Cache `walk_paths(base).leaves` once at substrate-claim
   registration.** Substrate registers its base claim with the
   framework (some `register_substrate_claim(base)` call); the
   leaves are computed once and stored. Bridges look up the cached
   set at evaluate time. Avoids per-bridge plumbing but adds a new
   substrate-registration step.

3. **Persist leaves into the cell record / cache.** Each cell
   carries its own leaves as a column (or a serialized set). The
   gate reads from the cell rather than recomputing. Storage
   overhead, but the leaves are already partly encoded in the
   `arm_key` column — could be derived from there.

Recommendation TBD. Option 2 is cleanest if substrate registration
is otherwise needed; Option 1 is most explicit; Option 3 leans on
existing persistence.

### Q2: What about `Bridge.source` that's a string column not in any of the four categories?

A bridge sourced on a raw trajectory output like `'mc_return'`
(no `@measurable` wrapper, just the column name as a string):

- `walk_paths(claim).leaves`? No.
- `_STANDARD_METADATA`? No.
- `registered_names()`? No.
- → trajectory output by elimination → True (endogenous).

This passes the gate, which is correct: a bridge testing
`do(arm) → mc_return` is a legitimate causal claim sourced on a
trajectory observable.

But: there's no validation that the column actually exists in the
cell record. A typo'd source name (`'mc_returns'` instead of
`'mc_return'`) would also fall into this branch and pass the gate
— then crash later when the analysis tries to read the column. The
gate would have given false reassurance.

Two resolutions:

1. **Add a fifth category check: "in cell record at evaluate
   time."** The gate (or the orthogonal `RESOLVED_FIXTURES` /
   `KNOWN_COLUMN` gate) verifies the source name appears in the
   filtered cells' columns before passing. Catches typos at gate
   time rather than at analysis time.

2. **Accept the gate's narrow responsibility.** `EXOGENOUS_SOURCE`
   only checks the exogenous/endogenous principle; column existence
   is `RESOLVED_FIXTURES`'s job (or a sibling). Keep the gates
   single-purpose.

Recommendation TBD. Option 1 is friendlier; Option 2 is cleaner.

## Relationship to ADMISSION_GATES_DESIGN.md

This doc refines the parent's "Substrate registry (L2)" section.
The parent currently says the framework provides
`registered_names()` and the substrate populates by importing its
measurable modules. That's still true, but the endogenous frontier
is **`registered_names() filtered by topological endogeneity`**,
not `registered_names()` itself.

When this doc's design lands, the parent should be updated to
reference `is_truly_endogenous` and drop the implicit
"registered = endogenous" conflation.
