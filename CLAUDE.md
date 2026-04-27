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
