# Primitives audit — pythonic discipline

Companion to `ADMISSION_GATES_DESIGN.md`. Catalogs framework
primitives (typed contracts, decorators, dataclasses) and asks
"could this be plain Python?" for each. Ships a contributor-
discipline meta-pattern: when to introduce a framework primitive
vs lean on Python composition.

## Motivation

The framework's contribution is logical strictness applied to
scientific claims. That goal IS its primitives — `Bridge`,
`DoEffect`, `Hypothesis` Protocol, `@claim_bridge`. But not
everything that calls itself a "primitive" earns its keep:
class-wrapped namespaces, cached fields that derive trivially,
authoring-boilerplate tuples that could auto-collect — these
are framework ceremony without typed-contract value.

The audit's discipline question: for each primitive, what does
it give the bridge author that plain Python doesn't?

## What's already pythonically pure

The framework's best moments are when typed contracts compose
through Python operators rather than custom combinators:

| Surface | Why it's already pythonic |
|---|---|
| `Bridge.scope: pl.Expr` | Polars expressions compose via `&` / `|` / `~` — no framework wrapper |
| `pair_by: tuple[str, ...]` | Plain tuple, no `PairKeySet` class |
| `Bridge.gates: tuple[AdmissionGate, ...]` (proposed) | Tuple, composes via `+` |
| `_FOURROOMS_REGIME = pl.col(...) & ...` (in dqn_bridges.py) | Module-level constant; bridges `&` it into per-bridge scope |
| `combined_arm_key(intervention_tuple) -> str` | Free function over a tuple; no method |
| `apply_interventions(base, interventions)` | Free function |
| `runtime_checkable Protocol` for `Hypothesis`, `Runner`, `AdmissionGate` | Structural typing — modern python, more pythonic than ABCs |
| Frozen dataclasses for typed-data-with-no-behavior (`PairedGResult`, `BridgeEvaluation`) | Slots, no methods (or `@property` derived only) |

These are the patterns to **generalize**.

## Reductions worth making

### 1. `BRIDGES: tuple[Bridge, ...]` → auto-collected from module vars

**Current authoring pattern:**

```python
BRIDGES = (
    *NSTEP_INTERVENTION_BRIDGES,
    *ACTION_DIM_BRIDGES,
    ...
)
```

**Pythonic alternative — `__all__`-style helper:**

```python
def collect_bridges(namespace: Mapping[str, object]) -> tuple[Bridge, ...]:
    return tuple(v for v in namespace.values() if isinstance(v, Bridge))

# Bridge author at the bottom of the file:
BRIDGES = collect_bridges(globals())
```

Saves ~10 LoC per bridge file. Curated subgroups
(`NSTEP_INTERVENTION_BRIDGES`) stay as named tuples for
partial-evaluation use cases — they're documentation, not the
canonical-run set.

**Trade-off**: ordering becomes module-definition-order. Cross-
bridge gates (Phase 5 of the gates design) need explicit
declarations anyway, not implicit ordering. Net win.

### 2. `Bridge.params: MappingProxyType[str, object]` → `@cached_property`-derived

The decorator originally extracted defaulted kwargs into
`Bridge.params` as a cached `MappingProxyType`. The cache is
stale-free because Bridge is frozen, but the cost of recompute
is a single `inspect.signature(...)` call (~7 µs) — once per
bridge per `runner.run`.

A naive `@property` would recompute on every access (7 µs each
× 60 accesses per runner.run ≈ 0.4 ms total — fine but wasteful
under future hot paths). `@cached_property` is the
best-of-both-worlds: first access pays the 7 µs, subsequent
accesses are 19 ns (just `__dict__` lookup) — measured 368×
speedup.

```python
@cached_property
def params(self) -> Mapping[str, object]:
    if self.holds_when is None:
        return {}
    sig = inspect.signature(self.holds_when)
    return {
        name: get_param_default(p)
        for name, p in sig.parameters.items()
        if get_param_default(p) is not inspect.Parameter.empty
    }
```

**Slots constraint**: `cached_property` writes through
`instance.__dict__`, which slotted classes don't have. So the
reduction also drops `slots=True` from Bridge. Memory cost is
~28 bytes more per instance × ~30 bridges per `runner.run` =
~840 bytes — negligible. `slots=True` wasn't earning its keep on
Bridge anyway (no per-cell hot path). Frozen-ness is preserved
— `cached_property`'s `__dict__` write bypasses `__setattr__`
without violating the frozen contract.

**Auto-registration of Measurables** at decoration time still
happens — the decorator computes params locally for that purpose
without storing the result on Bridge.

**Verdict**: small reduction with measurable performance win.
Cleaner Bridge surface.

### 3. `Scope.*` / `Gate.*` namespaces — plain modules, not class wrappers

The admission-gates design originally proposed `class Scope:` and
`class Gate:` with class-level attributes. **Class-as-namespace
is unconscious mimicry of `enum.Enum`'s pattern**, but enums
exist for `__members__` introspection, value↔name mapping, and
nominal typing — none of which apply to a polars-expression
namespace.

```python
# Class-wrapped (smell):
class Scope:
    PREMISE_ACTIVE: pl.Expr = pl.col(...) == 'held'
    CONVERGED: pl.Expr = ...

# Pythonic — plain module:
# corroborate_rl/scope.py
PREMISE_ACTIVE: pl.Expr = pl.col(...) == 'held'
CONVERGED: pl.Expr = ...

# Bridge author has either form they prefer:
from corroborate_rl import scope
scope.PREMISE_ACTIVE  # qualified

from corroborate_rl.scope import PREMISE_ACTIVE
PREMISE_ACTIVE  # bare
```

`ADMISSION_GATES_DESIGN.md` updated to reflect this convention
when the gates implementation lands.

## Where primitives ARE the right answer

Despite the audit, several "primitives" pull weight that plain
Python doesn't replicate:

- **`Hypothesis` Protocol** — typed structural shape; modules +
  classes both conform; pyright narrows it. Removing it loses
  static checking.
- **`Bridge` dataclass** — collects authoring metadata in ONE
  place. Splitting into "function with attrs" loses the typed
  discipline.
- **`DoEffect`** — typed contrast with `treatment_arm_key()` /
  `baseline_arm_key()` methods. Could become free functions on
  tuple-of-tuples, but the field-named accessors are
  documentation. The dataclass earns its keep.
- **`Intervention(slot_path, replacement)`** — typed slot-Claim
  pair. Tuple `(slot_path, replacement)` would lose the typed
  field names + the `apply()` method. Field-named is
  documentation.
- **`@claim_bridge` / `@analysis` / `@measurable` decorators** —
  extract metadata from the function's signature. Real work, not
  just labeling. The decorator IS the principled marker
  ("@claim is the single marker for 'this carries a theorem
  and records itself'" per CLAUDE.md).
- **Polars expressions for scope** — performance. A Python
  `Callable[[Mapping[str, object]], bool]` would be 100× slower
  on parquet-typed data.
- **`Verdict` / `Direction` / `Tier` enums** — IDE
  discoverability for authoring ergonomics. Switching to
  `Literal[...]` would lose `Verdict.<TAB>` completion. Keep.

## Meta-pattern: when to introduce a framework primitive

The audit's discipline summary, captured for contributor docs.

A framework primitive is the right answer when it:

1. **Encodes a typed contract** the bridge / substrate author
   should obey. Examples: `Hypothesis` Protocol (the
   verdict-time contract), `Bridge` (the authored edge),
   `Intervention` (the typed structural delta).
2. **Provides runtime narrowing** that gives pyright real
   information. Examples: `runtime_checkable` Protocols,
   `TypeIs[T]` predicates, frozen dataclasses with typed fields.
3. **Does real work beyond labeling** — extracts signature
   metadata, registers in a typed registry, walks the partial
   tree, etc. Examples: `@claim_bridge` (signature walker),
   `walk_paths` (claim-graph walker), `combined_arm_key`
   (canonical fingerprint).
4. **Hits a performance floor** Python composition can't reach.
   Example: polars expressions for parquet-column filters
   (~100× faster than per-row Python).

A framework primitive is unnecessary ceremony when it:

1. **Wraps a tuple-of-tuples without adding behavior.** A
   would-be `DoEffect`-like wrapper around already-meaningful
   structured data is a smell — the data IS the structure.
2. **Re-exports constants under a class namespace.** `class
   Foo: BAR = ...` is enum-mimicry without enum-purpose. Use a
   plain module file.
3. **Caches a derived value that's cheap to recompute.**
   `Bridge.params` is the canonical example — `inspect.signature`
   is sub-ms; the `@property` is honest.
4. **Expresses composition where Python operators already do it.**
   `And(check_a, check_b)` is what `(check_a, check_b)` (tuple)
   or `expr_a & expr_b` (polars / numpy) already are.

## Decision

Apply reductions 1 + 2 + 3 in one focused refactor PR. Add the
meta-pattern as a contributor-discipline section in CLAUDE.md
("when to introduce a framework primitive"). Update
ADMISSION_GATES_DESIGN.md to use plain-module `Scope` / `Gate`
namespaces when that work lands.

The patterns to **generalize for future framework work**:

- Tuple-`+` and polars-`&` for composition; never combinator
  types.
- Free functions over methods when there's no shared state.
- `runtime_checkable` Protocols for substrate-extensible shapes.
- Module-level constants composed via operators for module-
  scoped shared state (the `_FOURROOMS_REGIME` pattern).
- Frozen dataclasses for typed-data; slots for hot paths;
  `@property` for derived access.
