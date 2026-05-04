# corroborate — contributor instructions

## Purpose

`corroborate` finds the *scope* of an authored mechanism claim,
then verifies the *causal chain* that explains it. The chain
runs `env feature → invariance gap → mechanism activation →
outcome`; the invariance gap (residual of the theorem's premise,
measured per env) is the load-bearing node — both the
scope-defining feature and the causal mediator.

The framework provides three composable capabilities on top of
a `@claim`-decorated executable program: (a) intervention study,
(b) falsification, (c) causal discovery. These are substrate, not
application — orchestrators (dialectic loops, audits, data-mining)
compose them differently. Authoring an invariant per mechanism
claim is the substrate-author's primary commitment. See
`README.md`.

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
  `dict[str, RunRow]`, never bare `list` or `dict`.
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

## Two primitive shapes

Substrate authoring uses two shapes; which one applies depends on
whether the entity has theoretical content and whether it carries
construction-time configuration:

1. **Free Claim** — a top-level `@claim`-decorated function. The
   `FnClaim` wrapper auto-records calls; the function IS the
   theoretical operation, no class needed. Bake leaves at
   composition time via `functools.partial`. The default shape
   for everything that's a pure operation:
   *Examples: `bootstrap`, `double_greedify`, `semi_gradient`,
   `uniform_sample`, `linear_epsilon`, `epsilon_greedy`,
   `mlp_forward`, `cnn_forward`, `adam`, `rmsprop`,
   `warmed_update`, `periodic_copy`, `squared_error`.*
   `dqn` itself (a `@claim`'d free function) sits here too.

2. **Config bundle** — a frozen-dataclass that's NOT a Claim.
   Carries construction-time leaves as fields + slot Claims as
   fields + mechanics methods (allocation / state-update glue
   with no theorem attached). The walker surfaces its fields as
   topology leaves; mechanics methods are plain methods, no
   `record_call`. Used when stateful mechanics need to be paired
   with the configuration that parameterises them.
   *Examples: `Replay` (`capacity` / `batch_size` leaves +
   `sample` slot Claim + `init` / `add` / `sample_batch`
   mechanics); `MLP` / `CNN` (`hidden` / `obs_shape` leaves +
   `init` mechanics + `__call__` delegating to the
   `mlp_forward` / `cnn_forward` Free Claim where Hornik 1989
   attaches).*

The discriminator: **does this entity bundle stateful mechanics
that need to be paired with construction-time HPs?** If yes →
config bundle (the methods are the mechanics, the fields are the
HPs and slot Claims). If no, it's just a function → Free Claim
(decorate with `@claim`, configure via `partial`).

The `@claim` decorator is the **single marker** for "this carries
a theorem and records itself." Plain functions (no decorator) are
mechanics — paired with a Claim through delegation (`MLP.__call__`
calls `mlp_forward`) or through bundle methods (`Replay.add`).
Decorator absence IS the negation; there's no `@mechanics` marker.

A method on a config bundle is just a method — it isn't a Claim
even though it's callable. Theoretical content lives on the slot
Claims that the bundle holds (e.g., `Replay.sample` is a field
pointing at `uniform_sample`, which IS a Free Claim — that's
where Lin 1992 attaches) or on the Free Claim that
`__call__` delegates to (e.g., `MLP.__call__` calls
`mlp_forward`, which IS the Hornik 1989 Claim). The bundle is
mechanical organisation around a Claim or slot of Claims.

**Escape hatch (rare).** Substrate authors who genuinely need a
class-based Claim — stateful `__call__` with a theorem attached
directly to the instance, not delegated — write a frozen
dataclass exposing `name: str` and call `record_call(self,
args, kwargs, result)` inside `__call__`. The class structurally
satisfies `Claim[P, T]` without inheritance. Unused in the
current substrate; documented in `claim.py` and tested in
`tests/test_claim.py::test_manual_dataclass_with_record_call`.

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
- **Row store** (`RunRow`) — provenance + framework verdict +
  `measurements`. The cross-arm aggregate
  (`HypothesisComparisonRow`) is materialised on demand from
  RunRows via `from_cells`; it has no on-disk persistence pair.

A topology sidecar `<corpus>/graphs.json` is written alongside
the row stores. It carries one `ComputationGraph` per `arm_key`
so post-hoc consumers can recover the per-arm Claim-to-Claim
data flow that ran each cell.

**Hard rule: no JSON-wrapped struct columns in parquet.** Every
heterogeneous-keyed dict (HPs, derived measurables) is flattened
to top-level path-keyed columns at the parquet boundary. Polars
null-pads heterogeneous keys across rows; readers skip nulls. The
benefit is `df.filter(pl.col('optimizer.inner.lr') < 1e-3)` works
at the dataframe level — JSON wrapping kills this.

The path-keyed convention is collision-free by construction:
leaves use **dotted topology paths** (`replay.batch_size`),
trajectories use **flat author-chosen keys** (`reward`),
registered measurables use **bare names** (`jensen_gap`,
`eval_best_burst_mean`). The framework's namespace filter is the
measurable registry itself — `aggregate.leaf_signature` excludes
`registered_names()` from the configurational fingerprint.

## Canonical analyses (use these, don't reimplement)

The framework provides typed analysis primitives that should be
the default surface for mech / link / outcome analyses on
substrate corpora. Reimplementing these inline in experiments
duplicates logic that's already centralized; **prefer the
primitives** below.

### Per-cell vs per-burst — when to choose

- **Per-cell scalar** (`paired_g`, `meta_regression_paired_g`):
  pair vanilla / DDQN at the trajectory-averaged level. Fast,
  reliable when training is uniphase. **Fails silently** when
  Q dynamics are non-monotone (e.g., Q-explosion-prone envs):
  the trajectory-averaged Δ_jens combines causally opposite
  phases (early bias-correction vs late Q-explosion), washing
  the signal to ~0.

- **Per-burst** (`paired_g_per_burst`,
  `paired_link_per_burst`, `meta_regression_per_burst`,
  `mundlak_paired_g_per_burst`): compute Δ per (env, burst).
  Default for any env where Q dynamics aren't monotone. The
  panel makes phase structure visible and corroborable.

`findings_fourrooms_time_series.md` and the SpaceInvaders late-
burst attenuation history establish per-burst as the canonical
form for any analysis on Q-explosion-prone or phase-transition
envs. Cross-burst link cancellation is real; per-burst unmasks it.

### Mech / link / outcome separation

Three verdicts are kept independent (PAPER_NOTES.md §3
methodological claim). The corpus's `jensen_gap` measurable is
clamped to `max(0, mean(Q − MC))` — `0` does NOT mean
"unbiased": pair with the `jensen_dormancy_gap` measurable to
distinguish "true zero" from "underestimating (mech dormant)".
Link verdicts on dormant-mech cells are UNTESTABLE, not NULL.

### Conditioning rule

Link analyses MUST condition on `mech HELD` (Δ_jens < 0 with
the mechanism active, not just `jensen_gap > 0`). Otherwise
"link null" claims silently mix mech-dormant (bias premise
inactive) with mech-active-but-link-broken cells. The two are
different verdicts — the framework refuses to collapse them.

### Concrete primitives

| analysis | use for |
|---|---|
| `paired_g` | scalar Δ on a single measurable (outcome OR mech), pair-by seed |
| `paired_g_per_burst` | per-burst Δ on one measurable; reductions `mean` and `mc_minus_q` |
| `paired_link_per_burst` | per-burst r(Δ_target, Δ_predictor) — the empirical link, panel-typed |
| `phase_link_consistency` | scalar derived from per-burst link panel: fraction of bursts with significant negative r |
| `meta_regression_paired_g` | per-stratum Δ regressed on covariates |
| `meta_regression_per_burst` | per-(stratum, burst) panel meta-regression |
| `mundlak_paired_g_per_burst` | per-cell mediator + per-burst g (composite for moderator probes) |
| `dowhy` | DoWhy backdoor / refutation on a typed causal graph |
| `factorial_2x2` | 2×2 factorial interaction Δ |
| `tautology_audit` | three-check audit (HP shadow / partial-correlation / convergence) |
| `verdict_distribution` | corpus-level verdict count / class breakdown |
| `universe_scope` | universal scope analysis primitive |

When proposing an analysis: **check this list first**. New
inline analyses only when none of the above fits — and even then,
prefer to extend an existing primitive (or add a sibling) rather
than copying logic.

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
