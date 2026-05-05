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

## When to introduce a framework primitive

The framework's contribution is logical strictness applied to
scientific claims. Each typed primitive should earn its keep
against plain Python composition. See `PRIMITIVES_AUDIT.md` for
the full audit + meta-pattern; the four-question test before
adding a new one:

A primitive (typed dataclass, decorator, Protocol, enum) is the
right answer when it:

1. **Encodes a typed contract** the substrate / bridge author
   should obey — `Hypothesis` Protocol, `Bridge`, `Intervention`.
2. **Provides runtime narrowing** that gives pyright real
   information — `runtime_checkable` Protocols, `TypeIs[T]`,
   frozen dataclasses with typed fields.
3. **Does real work beyond labeling** — extracts signature
   metadata, registers in a typed registry, walks the partial
   tree (`@claim_bridge`, `@analysis`, `walk_paths`).
4. **Hits a performance floor** Python composition can't reach —
   polars expressions for parquet-column filters.

A primitive is unnecessary ceremony when it:

1. **Wraps a tuple-of-tuples without adding behavior.** A
   would-be `DoEffect`-like wrapper around already-meaningful
   data — the tuple IS the structure.
2. **Re-exports constants under a class namespace.**
   `class Foo: BAR = ...` is enum-mimicry without enum-purpose.
   Use a plain module file (`foo.py` with `BAR = ...`).
3. **Caches a derived value that's cheap to recompute.**
   `Bridge.params` was the canonical example —
   `inspect.signature` is sub-ms; `@property` is honest.
4. **Expresses composition where Python operators already do
   it.** `And(check_a, check_b)` is what `(check_a, check_b)`
   (tuple) or `expr_a & expr_b` (polars / numpy) already are.

The patterns to **prefer**: tuple-`+` and polars-`&` for
composition; free functions over methods when there's no shared
state; `runtime_checkable` Protocols for substrate-extensible
shapes; module-level constants composed via operators
(the `_FOURROOMS_REGIME` pattern in
`experiments/findings/dqn_bridges.py`); frozen dataclasses with
`@property` for derived access.

If a candidate primitive doesn't pass the four-question test,
the answer is to leave it as plain Python.

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
| `proportion_mediated` | linear-mediation decomposition: indirect / total share of Δ_target carried by `mediator` |
| `partial_spearman_rho` (graph.discovery) | linear-mediation Spearman form — partial-r of (X, Y) given Z; the Spearman analog of `proportion_mediated`'s direct effect |
| `stratified_partial_spearman_rho` (graph.discovery) | **JCI form**: per-env partial Spearman, Fisher-z-pooled — the canonical adjustment when env is a confound |
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

## Test principle

Tests assert framework output against an **analytical closed
form** within a **sampling-distribution-derived bound**. Four
rules, in order of importance:

1. **Closed-form, not arbitrary**. The expected value is computed
   from substrate parameters (`expected_g = mu_x · sqrt(n_steps) /
   sigma_x · c_4(n_pairs)`), not plucked. Bound size names what
   it absorbs (sample-SD CV at n=30 → 15%; cluster-robust SE
   inflation → t-critical at df=n-p; etc.). `g > 0.8` when
   structural g is 28 is a 35× slack — no.

2. **Substrate-grounded over synthetic-input**. Cells flow through
   `RunRow.as_dict()` → analysis (the production data path). The
   `tests/analytic/lg_scm/` substrate (Linear-Gaussian SCM with
   `@claim` Free Claims + frozen-dataclass config bundles) and
   `src/corroborate_rl/tests/analytic/deadly_triad/` (FQI / Q-
   divergence) make closed-form expectations possible without
   mocks. Synthetic-input tests (hand-built dicts → primitive →
   assertion) are unit-level coverage of internals only.

3. **Z-score bounds replace "doesn't reject null"**.
   `|coef / SE| < 2.5` against the framework's reported SE catches
   both inflated estimates AND collapsed SEs; `p > alpha + CI
   covers zero` passes garbage estimators with overconfident CIs.
   When framework SE comes from a CI (no `.se` field), invert
   with the framework's own t-critical (`scipy.stats.t.ppf(1-α/2,
   df=n-p)`), not `1.96`.

4. **No substrate-redundant assertions**. If a test reads back
   what it stamped (count of stamped verdicts, sum of stamped
   values), the assertion is tautological. The framework's logic
   isn't being verified. Either delete or replace with an
   assertion on transformation logic the framework actually does
   (case-folding, classification, dominance resolution).

Persistence tests pair: `tests/test_persistence.py` covers the
write/read CONTRACT (round-trip equality on hand-built rows);
`tests/analytic/lg_scm/test_parquet_round_trip.py` proves
**closed-form analyses still recover the structural answer**
after a real parquet round-trip. Both shapes are needed.

**Empirical coverage check.** `mutmut` is wired in
`pyproject.toml [tool.mutmut]` to mutate framework analysis
primitives and run only the analytic suite. Surviving mutants
are coverage gaps (closed-form bound too loose, or the line
isn't exercised). Run with `uv run mutmut run`; list all results
with `uv run mutmut results --all true` (default omits killed);
inspect a single mutant with `uv run mutmut show <name>`.

**Sharp edge:** mutmut wraps each function with a trampoline
that materializes default-arg values at the wrapper level, so
mutations of parameter defaults (e.g., `arm_field='arm_key'` →
`'XX...XX'`) never propagate through the call. Treat default-arg
mutations as wrap-broken, not real survivors.

**Workflow.** When fixing surfaced gaps:

1. Sample survivors per file: `uv run mutmut results --all true |
   grep "<file>.*survived" | head`.
2. Inspect each: `uv run mutmut show <name>`. Categorize as
   wrap-broken / equivalent / real-gap.
3. For real-gap clusters (e.g., Fisher-z formula never reached
   because tests give r=±1 short-circuit; IVW weighting never
   exercised because tests use uniform SE), add a closed-form
   test that **specifically constructs cells putting the
   framework on the unexercised code path**. Examples:
   `tests/analytic/lg_scm/test_paired_link_fisher_z.py` (moderate-r
   construction with independent ε per arm), `test_random_effects_ivw.py`
   (heterogeneous SE via varied n_pairs + mu_x extremes).
4. Re-run `uv run mutmut run`; verify the targeted mutants flip
   from `survived` to `killed`.

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
