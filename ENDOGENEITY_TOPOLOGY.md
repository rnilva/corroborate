# Endogeneity from topology — design

Companion to `ADMISSION_GATES_DESIGN.md`. The `EXOGENOUS_SOURCE`
gate enforces "Tier.INTERVENTIONAL bridges require an endogenous
source." This doc specifies what *endogenous* means structurally,
and shows the framework derives it from existing primitives —
no separate registry, no metadata constants, no `@exogenous`
decorator.

## Status

Open design. Two implementation questions pinned at the bottom.

## Motivation

A first cut of `EXOGENOUS_SOURCE` would treat
"source ∈ `registered_names()`" as the endogenous test — i.e., any
`@measurable`-decorated column passes the gate. This has a
loophole: an author can wrap an HP leaf in a trivial measurable
and the gate sees it as endogenous. The Phase-1 `effective_horizon
= 1/(1-γ)` was exactly this — a registered measurable whose
closure was just `{gamma}` (a single HP leaf), no cell-trajectory
state. The Phase-1 redefinition to `1/(1-γ·bf)` fixed it (closure
now includes `done` via `bootstrap_fraction`), but only because
the substrate-author noticed; the gate didn't enforce.

The structurally honest definition: a column is endogenous iff
its dependency closure transitively touches a value the *cell
produced by running*, as opposed to a value the *author chose at
experiment-design time*.

## The principle

The substrate's outermost claim function — the function the
runner ultimately calls to produce one cell — IS the substrate's
full structural composition. Its leaves are the author-controlled
primitives: every value the experimenter set at design time.

For the corroborate_rl substrate, the outermost claim is the
runner's experimental composition that takes:

- The bound `dqn(...)` partial (HP knobs are leaves of this part)
- The grid dimensions (env_name, seed, wrappers)

and returns a trajectory record. `walk_paths(outermost_claim).leaves`
returns one set: every author-controlled primitive — HP leaves
AND grid dimensions, in one source of truth.

There is no separate "metadata" registry. There is no
`@exogenous` decorator. There is no "framework controls these
columns, substrate controls those." The substrate ships its
outermost claim function; the framework derives the author
primitive set by walking it.

## Three structural rules

Every column in a cell record is classified by structural
position in the substrate's claim graph:

| structural position | how identified | endogenous? |
|---|---|---|
| **Leaf of the outermost claim** | `walk_paths(outermost_claim).leaves` | no — author chose |
| **`@measurable`-decorated** | `name in registered_names()` | recurse via `transitive_reads`; endogenous iff at least one base case is endogenous |
| **Anything else in the cell record** | by elimination | yes — cell-controlled primitive (trajectory output, or framework provenance like `id` / `arm_key` / `cycle_id` which the gate doesn't care about because no real bridge sources on them) |

Trajectory outputs aren't enumerated. They're whatever shows up
in the cell record that's neither a leaf of the claim composition
nor a registered measurable.

Framework provenance columns (`id`, `arm_key`, `corpus`,
`cycle_id`, `parent_id`, `timestamp`, `verdict`) live in the cell
record but are wiring, not data. Under the elimination rule
they'd be classified as endogenous, but no real bridge sources on
them; if one accidentally does, `resolved_source` (Q2) catches
the column-validity question and the analysis fails downstream
on non-numeric values. Not the gate's problem.

## Algorithm

```python
import functools

@functools.cache
def is_endogenous(name: str, leaves: frozenset[str]) -> bool:
    """A column is endogenous iff its dependency closure touches
    a value the cell produced by running, not just leaves of the
    outermost claim composition.

    Resolution order:
    1. Leaf of the outermost claim → exogenous (author chose).
    2. Registered measurable → recurse over `transitive_reads`;
       endogenous iff any base case is itself endogenous.
    3. Otherwise → cell-controlled primitive by elimination
       (trajectory output) → endogenous.
    """
    if name in leaves:
        return False
    if name not in registered_names():
        return True
    closure = transitive_reads(name)
    return any(r not in leaves for r in closure)
```

`leaves` comes from `walk_paths(substrate.outermost_claim).leaves`,
cached once at substrate registration. `transitive_reads` is the
framework's existing closure walker (`measurables/measurable.py`).

The function is `functools.cache`'d on `(name, leaves)` —
deterministic mapping; recursive calls with the same `leaves`
hit the cache after first computation.

## Examples

| name | leaf of outermost claim? | registered? | truly endogenous? |
|---|---|---|---|
| `gamma`, `optimizer.inner.lr`, `replay.capacity` | yes | — | no — leaf |
| `env_name`, `seed`, `wrappers` | yes (grid dimensions ARE leaves of the outermost claim) | — | no — leaf |
| `done`, `mc_return`, `jensen_gap`, `predicted_q_at_start` | no | no | yes — cell-controlled by elimination |
| `effective_horizon` (`reads=('gamma','done')` via `bootstrap_fraction`) | — | yes | yes — closure touches `done` (trajectory) |
| `bootstrap_fraction` (`reads=('done',)`) | — | yes | yes — closure is `{done}` |
| `q_divergence_score` (`reads=('jensen_gap','gamma','env_name')`) | — | yes | yes — closure touches `jensen_gap` (trajectory) |
| `r_max` (`reads=('env_name',)`) | — | yes | no — closure is `{env_name}` (leaf) |
| `log_action_dim`, `log_obs_dim`, `log_horizon` (`reads=('env_name',)`) | — | yes | no — env-feature lookups |
| (HYPOTHETICAL) `effective_horizon_synthetic` (`reads=('gamma',)`) | — | yes | no — closure is `{gamma}` (leaf only); the gate catches this loophole |

The `r_max` / `log_action_dim` / `log_horizon` family are
synthetic in the strict sense — they're env-feature lookups, not
cell-derived. They classify as exogenous because their closure
is `{env_name}`, which is a leaf. That's correct: a Tier.INTERVENTIONAL
bridge sourced on `log_action_dim` would be making a causal
claim about an env-structural feature, not about cell dynamics.
The substrate currently uses these as covariates in
meta-regressions (correct), not as INTERVENTIONAL sources.

## Implications

**The substrate ships one thing**: its outermost claim function.
Whatever that function takes as kwargs, those become the author
leaves automatically. Adding a new HP knob → a new leaf → a new
exogenous primitive. Adding a new grid dimension (e.g., a new
wrapper kind) → similarly. No registry to keep in sync.

**The endogenous frontier widens by writing new measurables**.
A measurable whose `reads=` declaration touches a trajectory key
(directly or transitively) is endogenous; one that closes over
only leaves is exogenous. The author's lever is the `reads=`
declaration; topology decides.

**Synthetic wrappers are caught by the gate**. The Phase-1
`effective_horizon` was synthetic before redefinition; under the
strict gate it would have BLOCKed any Tier.INTERVENTIONAL bridge
sourced on it. The redefinition to include `bootstrap_fraction`
made the closure include `done`; now it passes.

**Bridge migrations using synthetic measurables surface clearly**.
If the substrate adds a measurable like `chain_depth = 1/(1-γ)`
and tries to source an INTERVENTIONAL bridge on it, the gate
BLOCKs with a clear message: "closure of `chain_depth` is `{gamma}`,
which is a leaf of the outermost claim; bridge needs a delegate
that depends on cell-trajectory state."

## Open questions

### Q1: Where does the gate get the outermost claim from?

The substrate must register its outermost claim function so the
framework can compute leaves. Pattern (matching how `@measurable`
populates the measurable registry):

```python
# corroborate_rl module init
from corroborate.bridge.admission import register_outermost_claim
from corroborate_rl.dqn.experiment import experiment   # the substrate's outermost
register_outermost_claim(experiment)
```

The substrate's `experiment` function takes env_name, seed,
wrappers, plus the bound dqn — its leaves are the full author
primitive set.

If the substrate doesn't currently HAVE an explicit outermost
function (the runner builds the composition implicitly via
intervention application + grid_point), the substrate authors
one for the framework to walk. ~1-2 hours of substrate work,
mostly cosmetic.

### Q2: What about `Bridge.source` that's a string column not in any of the categories?

A bridge sourced on a raw trajectory output like `'mc_return'`
(no `@measurable` wrapper, just the column name as a string):

- not a leaf of the outermost claim
- not in `registered_names()`
- → cell-controlled primitive by elimination → endogenous → passes

This is correct: a bridge testing `do(arm) → mc_return` is a
legitimate causal claim sourced on a trajectory observable.

But: a typo'd source name (`'mc_returns'` instead of
`'mc_return'`) also falls into this branch and passes the gate
— then crashes later when the analysis tries to read the column.
The gate would have given false reassurance.

Resolution: a separate `resolved_source` gate (BLOCK) verifies
the source name appears in the filtered cells' columns at
evaluate time. Catches typos at gate time, single-purpose, no
overlap with `EXOGENOUS_SOURCE`.

## Relationship to ADMISSION_GATES_DESIGN.md

This doc replaces the parent's "Substrate registry (L2)" section.
The parent currently says the framework provides `registered_names()`
and `_STANDARD_METADATA`; the substrate populates by importing
its measurable modules. With this doc:

- Drop `_STANDARD_METADATA` from the parent doc and from
  `bridge/admission.py`. It conflated two unrelated things
  (substrate grid dimensions like `env_name` + framework wiring
  like `arm_key`); the topological framing dissolves both.
- Replace `_endogenous_pool() = registered_names() | _STANDARD_METADATA`
  with `is_endogenous(name, leaves=_substrate_leaves)` from day
  one. No loose-vs-strict phasing.
- The parent's `EXOGENOUS_SOURCE` algorithm becomes the call
  site for `is_endogenous`.

When this design lands, the parent should be revised to remove
the metadata-registry framing; substrate-claim registration is
the single source of truth.
