# corroborate — contributor instructions

## Purpose

`corroborate` provides three composable capabilities on top of a
`@claim`-decorated executable program: (a) intervention study,
(b) falsification, (c) causal discovery. These are substrate, not
application — orchestrators (dialectic loops, audits, data-mining)
compose them differently. See `README.md`.

## Typing discipline (load-bearing)

The framework's contribution is logical strictness applied to
scientific claims. Its own code must be the same shape — strict
types reject spurious connections and information loss before they
reach a smoke run. This is a **load-bearing** principle, not a
cosmetic one.

**The rule.** No type erasure. C++ template-mind discipline: no
`void*` equivalents anywhere; type information flows end-to-end
through call chains.

### Disallowed unless polymorphism truly requires

- `Any` — opts out of type checking entirely. Almost never
  required; nearly always a sign that the API needs redesign.
- `object` (as a parameter or container element) — accepts
  anything but provides nothing. Forces narrowing at the use
  site, which usually re-introduces erasure. Only acceptable
  inside a *generic upper-bound* container where the polymorphism
  is intrinsic (e.g. a trace holding heterogeneous Claim[P, T]
  calls — `list[Claim[..., object]]` is the upper bound at the
  container boundary, never at element use sites).
- `cast()` — circumvents the checker. Last resort with a
  one-line comment explaining the runtime invariant the type
  system can't express.
- `# type: ignore[...]` — same. Last resort with rationale.
- `getattr` / `setattr` on typed values — these return `Any` /
  accept `object` and erase types. **Redesign the API to expose
  typed attributes on a class instance** rather than dynamic
  attributes on a function. If you find yourself writing
  `setattr(fn, '__is_claim__', True)` followed by
  `getattr(fn, '__is_claim__', False)`, return a typed wrapper
  class from the decorator — its instance attributes are
  statically resolved.

### Required where polymorphism truly applies

- **PEP 695 type parameters** (`def f[T](x: T) -> T:`,
  `class Foo[**P, T]:`). Use these aggressively. ParamSpec
  (`**P`) preserves caller signature through generic wrappers;
  TypeVarTuple (`*Ts`) preserves heterogeneous tuple shape.
- **Protocol** for structural typing — capturing "anything that
  has these typed attributes" without inheritance. Read-only
  attributes via `@property` are how Protocols match
  frozen-dataclass fields (writable Protocol fields don't match
  immutable concrete fields).
- **PEP 742 `TypeIs[T]`** — the narrowing primitive of choice.
  Returns `bool` at runtime but tells pyright to narrow the
  argument both in the True AND False branches:
  ```python
  def is_mapping(v: object) -> TypeIs[Mapping[str, object]]:
      return isinstance(v, Mapping)

  if is_mapping(x):
      # x narrowed to Mapping[str, object]
      ...
  else:
      # x narrowed to (object & not Mapping[str, object])
      ...
  ```
  Prefer `TypeIs` over `TypeGuard` (one-sided narrowing) and
  over `cast` (no narrowing). It's how runtime predicates avoid
  type erasure.
- **Concrete generic types** in containers — `list[Claim[P, T]]`,
  `dict[str, FactRow]`, never bare `list` or `dict`.
- **Frozen dataclasses with `slots=True`** for typed records —
  attribute access is statically resolved; no `getattr` paths.

### Enforcement

- `pyright` strict mode runs on `src/` AND `tests/`. Configured
  in `pyproject.toml`. Both must pass before commit.
- `pytest` covers behavioral invariants. Both green is the gate.
- Standard pre-commit (manual for now): `uv run pyright && uv run pytest`.

### Heuristic before adding a `# type: ignore`

1. Can a Protocol or generic express the constraint?
2. Can a typed wrapper class replace the dynamic-attribute pattern?
3. Is the value's true type a `Literal` or `TypeGuard`-narrowable
   union?
4. If none of the above, write a one-line comment explaining
   exactly what runtime invariant justifies the escape.

If the answer to (1)-(3) is "no" three times in a row, the API
probably needs redesign before adding the ignore.

## Style

- Modern Python only. Target 3.13+; use PEP 695 generics, PEP 692
  TypedDict-Unpack where applicable.
- Frozen dataclasses by default for records. `slots=True` for hot
  paths.
- Standard library first; reach for an external dep only when
  absolutely required (cf. framework-subtraction discipline).
- Docstrings explain WHY, not WHAT — well-named identifiers do
  the WHAT.

## Vocabulary (framework-honest, not domain-borrowed)

The framework speaks of two kinds of measurables:

1. **Claim outputs** — what running the configured composition
   emits at run time (the record dict's entries: `reward[t]`,
   `loss[t]`, `td_error[t]`, …). Author-named flat keys.
2. **Leaf scalar claims** — non-recursive nodes in the graph of
   claims, observed at *composition time* by walking the bound
   `partial(...)` via `signature.walk_paths`. Dotted topology
   paths (`gamma`, `optimizer.inner.lr`,
   `bootstrap.greedification`).

RL practice calls (2) "hyperparameters". The framework does NOT.
`regime='leaf'`, `aggregate.leaf_signature`, `walk_paths`,
`flatten_leaves` — never `hp_*`. "Leaf" generalises beyond RL
configuration; "HP" leaks domain jargon into framework semantics.

Substrate code is welcome to say "HP" in its own docs/comments
(it's the reader's vocabulary). Framework code does not.

## Three-way claim taxonomy

The framework supports three categories of authored entities;
which one applies depends on whether something has theoretical
content and at what shape:

1. **Module Claim** — a frozen-dataclass `ClaimBase` subclass
   with a single `__call__` that IS the theoretical claim.
   `__call__` records itself via `record_call`; the Module bundles
   construction-time leaves. Used for things with one primary
   end-to-end operation.
   *Examples: `Adam`, `MLP`, `EpsilonGreedy`, `WarmedUpdate`.*
   `dqn` itself (a `@claim`'d free function) sits here too,
   wrapped as `FnClaim`.

2. **Free Claim** — a top-level `@claim`-decorated function. The
   `FnClaim` wrapper auto-records calls; the function IS the
   theoretical operation, no Module wrapper needed.
   *Examples: `bootstrap`, `double_greedify`, `semi_gradient`,
   `uniform_sample`, `linear_epsilon`.*

3. **Config bundle** — a frozen-dataclass that's NOT a Claim.
   Holds construction-time leaves and slot Claims (which are
   Module Claims or Free Claims), plus possibly mechanics
   methods that have no theorem (FIFO append, etc.). The walker
   surfaces its fields as topology leaves regardless of Claim
   status; mechanics methods are plain methods, no `record_call`.
   *Example: `Replay` — `capacity`/`batch_size` leaves +
   `sample` slot Claim + `init`/`add`/`sample_batch` mechanics.*

The discriminator: **does the entity have one theoretically-
meaningful primary operation?** If yes → Module Claim. If it's
itself the operation as a free function → Free Claim. If it
bundles config + slots + mechanics with no single primary
theorem → config bundle. The category is set by the entity's
nature, not by framework preference.

A method on a config bundle is just a method — it isn't a Claim
even though it's callable. Theoretical content lives on the slot
Claims (e.g., `Replay.sample` is a slot, points at
`uniform_sample` which IS a Free Claim — that's where Lin 1992
attaches). The bundle is mechanical organisation.

## Persistence shape (typed × open)

Each row store splits into two surfaces:

- **Framework-typed** — closed-set enums (`Verdict`,
  `RefutationClass`), lineage IDs (`id`, `parent_id`, `cycle_id`,
  `treatment_arm_id`, …), framework-controlled provenance
  (`timestamp`). Typed dataclass fields. Stable across substrates.
- **Open** — `measurements: Mapping[str, MeasurementLeaf]` where
  `MeasurementLeaf = str | int | float | bool`. Path-keyed
  scalars, substrate-shaped. The substrate decides what's in here.

Two stores join by UUID:

- **Trace store** (`TraceRow`) — per-cell raw observation. Outputs
  (1) as 1-D `list[float]` columns + leaves (2) as scalar columns.
  v9's `traces.parquet` analog.
- **Row store** (`RunRow` / `ArmRow` / `ComparisonRow` /
  `CorpusRow`) — provenance + framework verdicts +
  `measurements`. v9's `measurements.parquet` /
  `bridge_ledger.parquet` analog.

**Hard rule: no JSON-wrapped struct columns in parquet.** Every
heterogeneous-keyed dict (HPs, bridge stats, meta) is flattened to
top-level path-keyed columns at the parquet boundary. Polars
null-pads heterogeneous keys across rows; readers skip nulls. The
benefit is `df.filter(pl.col('optimizer.inner.lr') < 1e-3)` works
at the dataframe level — JSON wrapping kills this.

The path-keyed convention is collision-free by construction:
leaves use **dotted topology paths** (`replay.batch_size`),
trajectories use **flat author-chosen keys** (`reward`). Output
prefixes (`outcome.`, `bridge.`, `invariant.`) on the row store
namespace results separately from leaves.

## Test iteration

Tests that compile a JAX kernel and run DQN end-to-end on
CartPole are marked `@pytest.mark.slow`. Defaults:

- `uv run pytest tests/` → fast cohort only (~9 s, 213 tests).
- `uv run pytest tests/ -m slow` → slow only (~92 s, 22 tests).
- `uv run pytest tests/ -m ''` → full suite (~95 s, 235 tests).

`addopts = "-q --strict-markers -m 'not slow'"` in pyproject. The
empty `-m ''` overrides addopts to include both.

## Acceptance criteria

`v0` is acceptance-tested by reproducing the DDQN study in
`/workspace/poc_v10/PAPER_NOTES.md` §3 — mechanism HELD ↛ outcome
HELD ↛ link HELD across 17 envs, with the methodological
contribution living in keeping these three verdicts separate.

The framework's primary distinguishing feature lives at the
verdict layer: `POWER_INSUFFICIENT` is a first-class verdict
distinct from `HELD` and `NO_EFFECT`. Treating an underpowered
test as "no effect" smuggles methodological problems past the
reader (PAPER_NOTES.md §3.4); the framework refuses that smuggle.
