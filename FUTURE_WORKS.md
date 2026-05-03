# Future works

Backlog of deferrable improvements surfaced during code reviews.
Each entry has: status, the rationale for deferral, and the
condition that should lift it.

Entries are ordered by *forcing function*: the higher up, the
sooner they're likely to bind.

## Refuter semantics — `is_refuter` flag deferred (2026-05-02)

**Status:** removed without replacement during Phase 1 of the
Bridge-collapse refactor. The `BridgeRole` enum's `'refuter'`
member, plus the `refuter_edge` factory, were deleted along with
the rest of `claimed_edge.py`.

**Why deferred:** the only consumer of refuter-as-Hypothesis-edge-
role was `tests/test_claimed_edge.py` (also deleted). No
experiment, smoke, or analysis currently authors a refuter edge.
The plan called out the `is_refuter: bool` distinction
(HELD-flips-interpretation: a HELD refuter *contradicts* the
hypothesis) as the one piece of role-enum semantics worth
preserving, but with zero current consumers there's nothing for
the flag to drive.

**Lift condition:** when a substrate authors a refuter edge —
i.e. an intervention-shaped Bridge whose HELD verdict should
flip the hypothesis-level interpretation. At that point: add
`is_refuter: bool = False` to `claim_bridge.Bridge`, have
`hypothesis_subgraph_verdict` invert the verdict for refuter
edges before assembling the Hypothesis-level pattern.

(Note: `bridges_dowhy.py` uses `'role': 'refuter'` strings
inside DoWhy refutation analyses — those are stats-dict labels
on placebo / random-common-cause refutations, unrelated to the
Hypothesis-edge role taxonomy. They stay untouched.)

## v10 `redundancy.py` port — explicitly deferred (2026-05-01)

**Status:** not blocking active work; deferred until the
authoring layer (file protocol + analyses + bridges) has settled
enough that R(h) over a register of authored claims is the
natural next consumer.

**Why deferred:** the file protocol shipped 32 authored claim_
bridges (~1565 LoC) and is producing real verdicts; the bottleneck
right now is *authoring ergonomics* (per-env factory, threshold
helpers, structural-field plumbing), not the absence of an axiom-
19 reward signal. A redundancy primitive built before the register
shape stabilises would have to be re-fitted afterwards — a worse
order of operations than the reverse.

**Lift condition:** the authoring layer settles AND a register/
cycle persistence path lands. At that point `compute_R_info(h,
register)` has its inputs typed and its consumer named. See item
#5 (`redundancy.py — ΔI_redundancy primitive`) in the v10 audit
section below for the implementation specs.

## Type-discipline refactorings (framework-types pass, 2026-05-03)

Surfaced during the strict-Any cleanup. Listed by structural
significance, not by typing severity — the framework now passes
pyright cleanly; these are design quirks the cleanup made visible.

### Module Claims → pure-functional

**Status:** deferred (user-flagged during review).

**Description:** `ClaimBase` subclasses with `__call__` + manual
`record_call` inside duplicate the `@claim` / `FnClaim` decorator
path. The Three-way claim taxonomy in CLAUDE.md (Module Claim /
Free Claim / Config bundle) collapses to two if Module Claims are
expressed as a config bundle (frozen dataclass, no `__call__`)
plus a top-level `@claim`'d function taking the bundle as a
kwarg. `record_call` becomes the framework's responsibility
universally; authors stop writing it.

**Why deferred:** ripples through every `ClaimBase` subclass in
`rl/dqn/claims/*` and the walker in `signature.py`. Bigger than
the type cleanup wanted to scope.

**Lift when:** the substrate-extension pass starts (so a second
substrate's Module-shaped claims are designed pure-functional
from the start), or a Claim Protocol mismatch surfaces a third
time.

### Bridge / Verdict / Direction surface still in flux

**Status:** main is iterating actively (`drop Bridge.intervention`
2026-05-03, `INVARIANT bridges`, `INTERVENTION auto-resolution`,
`bridges-on-raw`).

**Description:** `Bridge.source` accepts `str | Measurable[...] |
DoEffect | INTERVENTION` — wide tape suggesting two distinct
shapes ("interventional contrast" vs "measurement coupling")
collapsed into one type. `intervention_edges` /
`coupling_edges` was just refactored from a field-check to an
`isinstance(e.source, DoEffect)` check (post-merge in
hypothesis.py). Suggests the surface wants two Bridge subtypes,
or a sealed `BridgeKind` discriminator.

**Why deferred:** ongoing iteration; another design pass while
main is mid-refactor would conflict.

**Lift when:** `claim_bridge.py` settles (no significant changes
for a sweep cycle).

### `iterate` / `Loop[C, T, Idx]` / record_call interaction

**Status:** deferred.

**Description:** `iterate` exists primarily so the loop boundary
appears as one node in the computation graph. The mechanism is
heavyweight: every substrate that wants graph capture wraps
loops in `iterate`, which forces parametric Loop typing
gymnastics (3 casts in `rl/dqn/eval.py` for T re-binding +
aggregation narrow). Alternative: graph-extraction-time
post-processor that recognises loop boundaries from records,
without the wrapping primitive. Eliminates `iterate` and the
`Loop[C, T, Idx]` Protocol; substrates use plain `for` /
`jax.lax.scan`.

**Why deferred:** rl is the only substrate exercising graph
capture; the graph extractor's current shape works.

**Lift when:** a non-RL substrate adopts `iterate` (cost
becomes visible) or graph extraction gains new requirements
(the post-processor design becomes the natural home).

### Phantom `Analysis[R]` parameter

**Status:** deferred.

**Description:** `Analysis[R: Mapping[str, object], O]` declares
`R` but never uses it in any field — pure documentation as
type. Either drop `R` to `Analysis[O]`, or wire it through to
a method (`def __call__(self, corpus: Iterable[R], **deps) ->
O`) so it has structural meaning. Currently, the registry cast
at `analysis.py:103` exists because `R` invariance + phantom
status forces the upper-bound erasure.

**Why deferred:** documentation value of `R` is real; removing
it loses the substrate-record-shape signal at the analysis
type. Wiring it through requires settling on an
`Iterable[R]` / `Sequence[Mapping[str, object]]` shape for the
corpus argument across all `@analysis` impls.

**Lift when:** a substrate has multiple distinct record shapes
that need analyses keyed by R, OR the corpus shape stabilises
across analyses (currently varies).

### Substrate extension contract not formalised

**Status:** deferred.

**Description:** `rl/` is the only substrate. Framework hooks
(`Hypothesis.measurables`, `Bridge.source`, `Loop[C, T, Idx]`,
`Claim` Protocol) are implicitly substrate-extension points,
but no documentation says so and no test exercises the
framework with a non-RL substrate. RL-isms could be baked into
the framework root by accident.

**Why deferred:** YAGNI until a second substrate is real.

**Lift when:** a second substrate (supervised learning,
optimization, evolutionary search) is in play. Add: a
"to add a substrate, implement X / Y / Z" doc + a tiny
non-RL substrate as test fixture.

### Registry consolidation

**Status:** deferred.

**Description:** Three independent `_REGISTRY` globals
(`measurable._REGISTRY`, `analysis._REGISTRY`, `claim._FN_CACHE`),
each a `Registry[T]` singleton. Could consolidate to
`corroborate.registries` with explicit handles —
simplifies introspection ("what's registered globally?") and
testing (mocking).

**Why deferred:** isolation has worked; consolidation is
ergonomic, not load-bearing.

**Lift when:** registry introspection becomes a first-class
need (a debug command, a registry-diff tool) — or a registry
collision bug appears.

### `analyses/__init__.py` 14 side-effect imports

**Status:** deferred (small).

**Description:** `from .X import Y as _Y  # pyright: ignore
[reportUnusedImport]` × 14. Function-scoped imports inside a
`_register_default_analyses()` function don't trigger
`reportUnusedImport`. Same pattern fits `cell_runner.py`'s
`import corroborate.rl.dqn.measurables` side-effect.

**Why deferred:** 14 ignores are bounded; the function-scope
move is a stylistic choice that doesn't affect behaviour.

**Lift when:** doing another pass on `analyses/`'s public
API surface.

### `signature.py` walker functional refactor

**Status:** deferred (cosmetic).

**Description:** the recursive walker (200+ lines) uses mutable
`kwargs_out: list[KwargInfo]` state through helpers. Could be
expressed functionally — each helper returns a tuple, no
mutation. The `_introspection_boundary` already wraps the
Any-leaks; the walker on top is bookkeeping.

**Lift when:** signature.py needs other changes (extending the
regime taxonomy, supporting new claim shapes, etc.).

### Stubs maintenance procedure

**Status:** deferred.

**Description:** 5 hand-written stubs (`gymnax`, `optax`,
`scipy`, `statsmodels`, `dowhy`). On upstream API drift, stubs
silently rot. Cheap remediation: a smoke test that exercises
the stub'd surface against the real installed library; a
note documenting which version each stub matches.

**Lift when:** a dep upgrade trips a runtime AttributeError
that pyright would have caught with up-to-date stubs.

### Numpy axis-aware reduction helper

**Status:** deferred (cosmetic).

**Description:** 5 nearly-identical `cast(npt.NDArray[np
.floating], arr.X(axis=axis))` lines in `reductions.py
:reduce_axis`. A single helper collapses to one cast.

**Lift when:** more axis-aware reductions are added (the
duplication forcing function bites).

## Open primitives bundled toward v1 acceptance

v0 acceptance (§3 DDQN three-way verdict) is closed by the
typed-edge / verdict-walk path. v1 acceptance (closed dialectic
loop with axiom-19 reward) needs three primitives still
missing:

1. **`redundancy.py` — ΔI_redundancy primitive** (~240 LoC).
   4-factor jaccard·concord·intervention·identity overlap.
2. **`register.py` — append-only G register** (~120 LoC).
   Past comparisons + latest-wins fact projection.
3. **`compute_R_info` aggregator** (~80 LoC). Combines per-
   bridge ΔI with the redundancy term into `R(h)`.

**HPO-smuggle gate** is deferred: `computation_graph.signature(g)`
is the principled form (two interventions producing the same
structural signature ARE the same mechanism). Lift only if a
counterexample appears where signatures match but mechanisms
differ in a way that matters for the dialectic loop.

**Lift when:** v1 design starts; the authoring layer is settled
enough that the register's typing is forced by real consumers.

## Vectorised env support (n_envs > 1)

**Status:** deferred — modern DRL feature, not v0 blocker.

**Description:** Modern DRL (SB3, CleanRL, v9) parallelises M envs
per cell via `vmap(env.step)` — buffer adds M transitions per
step, one batched gradient step per cycle. Wall-clock benefit is
M× on slow envs (Atari ~1 ms/step). gymnax envs are
microsecond-fast so the benefit at v0 scale is marginal.

Mnih 2015 / Hasselt 2016 (DDQN) used single-env per cell; v0
matches that for paper fidelity.

**Why deferred:** v0's gymnax sweep at 50k steps doesn't need it.
Adding before a forcing function risks complicating dqn-step
structure (two-level vmap: seeds × envs) without payoff. v9
ships with `NUM_ENVS=64` because their sweep is at higher step
budgets where env-step throughput dominates.

**Lift when:** an Atari-grade env enters the sweep, OR step
budgets per cell exceed ~500k and env-step throughput becomes
the bottleneck.

**Insertion point when needed:**
- `n_envs: Annotated[int, Exogenous] = 1` on `dqn` kwargs.
- `rollout_phase` vmaps over an n_envs axis.
- `Replay.add` accepts a vector of M transitions.
- `DQNState.env_state` / `obs` become `(n_envs, ...)`-batched.
- ~150 LoC of structural change; clean extension that doesn't
  break n_envs=1 semantics.

## Replay-as-Claim Protocol mismatch — RESOLVED (2026-04-28)

**Status:** resolved. Replay is no longer a `ClaimBase` subclass.

**Resolution.** None of the three Protocol-design alternatives
were taken. Instead, the principled-but-overengineered detour
(making Replay a Claim somehow) was abandoned in favour of
acknowledging that `Replay` simply isn't a framework Claim — it's
a config bundle. The Lin 1992 theoretical claim is about the
*sampling distribution*, which lives in the `sample` slot
(an `@claim` free function: `uniform_sample`,
`prioritised_sample`, …). The slot's FnClaim records itself; the
Replay dataclass owns HPs + mechanics methods (`init`, `add`,
`sample_batch`); none of those methods are theoretical claims.

5 LoC change:
- Drop `ClaimBase` from `Replay`.
- Drop `record_call(self, ...)` from `Replay.add` (mechanics, not
  a Claim — append-to-FIFO has no paper reference).
- `sample_batch` stays as a binding wrapper around `self.sample`;
  the slot records the call.

The walker still surfaces `replay.capacity`, `replay.batch_size`,
`replay.sample` as topology leaves regardless of Claim status.
Pyright clean, no Protocol mismatch.

**Principle that survived:** *every theoretically-meaningful
operation is a Claim, but not every callable needs to be one.*
The framework supports both Module Claims (single `__call__`) and
free-function Claims (FnClaim), plus config bundles (frozen
dataclasses with no Claim status). Mechanics methods on a config
bundle are just methods.

**PER infrastructure** (post_train_update slot at the dqn-step
level, prioritised_sample, update_priorities, no_priority_update)
deferred until PER actually lands. Design without a use site is
what almost dragged this fix into a +60 LoC bundle-of-claims
rewrite — the lesson from the audit thread.

## Deferred from second-pass external review

### Typed `Powered` record on HypothesisComparisonRow

**Status:** deferred.

**Description:** `HypothesisComparisonRow.adequately_powered:
bool` collapses the gradient that derived it (n + sd + observed
g + MDE). A typed `Powered(mde_d, observed_g, achieved_power)`
would carry the inputs so two stat implementations can't
disagree about what "adequately" means.

**Why deferred:** the bool tracks the gate result; richer
power-curve diagnostics aren't a current consumer demand.

**Lift when:** a consumer needs to log per-env power-curve
diagnostics (recording WHY each cell fell HELD vs
POWER_INSUFFICIENT).

### Vector outcomes (`primary_outcome_summary: float` → `tuple[float, ...]`)

**Status:** deferred.

**Description:** `RunRow.primary_outcome_summary` is one float
per cell. Sample-efficiency-to-threshold and AUC-of-learning-curve
outcomes need the per-step trajectory; the current scalar
collapses them.

**Why deferred:** PAPER_NOTES.md §3 acceptance test uses
`final_return` (one scalar per cell). §3.9 caveat 5 explicitly
flags richer outcomes as out-of-v0-scope. v9 and v10 both store
scalar-per-cell.

**Lift when:** a paper section needs richer outcome shapes
(probably §3.9 follow-on or step 6's CorpusBridge widening).

### `CycleRef(index, id, parent_id)` for verdict-evolution tracking

**Status:** deferred.

**Description:** `cycle_id: str | None` on every row groups but
doesn't order or support typed queries like "did this hypothesis's
verdict shift HELD→NO_EFFECT between cycles 4 and 5?" A typed
`CycleRef` with integer index and parent reference is the right
primitive when the redundancy register cares about cycle
sequencing.

**Why deferred:** no dialectic loop yet exists in corroborate;
nothing populates `cycle_index` or `parent_cycle`. Speculative
until a loop consumes it.

**Lift when:** the dialectic-loop orchestrator lands (probably
post-step-7 acceptance test) AND verdict-evolution diagnostics
become a feature.

### `CorpusBridge[CorpusRow]` primitive widening for link bridges

**Status:** deferred.

**Description:** `BridgeResult.targets: tuple[str, ...]` is a flat
tuple of record keys. PAPER_NOTES.md §3.5's link bridge
(`Pearson r(stat_q, stat_f)` across envs) operates on PAIRS in a
corpus, not a record. Need either widened targets
(`tuple[tuple[str, str], ...]` carrying direction) or a sister
type `CorpusBridge[CorpusRow]`.

**Why deferred:** step 6 (verdict + corpus layer) is the natural
place to design the corpus-bridge call site. Pre-building shape
without a use site risks designing the wrong primitive.

**Lift when:** step 6 starts. The §3.5 link bridge is the first
concrete consumer.

### Theorem-gap measurables that need richer logging

**Status:** deferred — multiple gaps blocked on data the v0
StepRecord doesn't carry.

**Background.** Invariants in `corroborate` measure *gap
magnitude* from theorem conditions, not threshold-bounded
boolean tests (`invariant.py` module docstring; memory
`feedback_invariant_three_roles.md`). Several of the gaps the
DQN claim set should report aren't computable from the v0
record. Each needs a specific data extension before the gap
measurable can be implemented honestly.

**Currently shipped (v0 + step 4):** `fqi_decay_gap`,
`hasselt_covariance_gap` (Pearson r over kept
Q-values), `action_coverage_gap` (caveated Watkins floor),
`jensen_overestimation_gap` (reads EvalRecord — step 4.4 lifted
this from "needs separate eval pass" via the `train_with_eval`
infrastructure), `state_action_coverage_gap(env_spec)` (reads
per-step `state_hash` — step 4.4 lifted via `EnvSpec.state_hash`
+ rollout-phase logging).

| Gap | Theorem | Data needed | Lift gate |
|-----|---------|-------------|-----------|
| Banach contraction rate | Bertsekas-Tsitsiklis 1996 §6.3 — `r_emp = ‖Q_{t+1} − Q_t‖ / ‖Q_t − Q_{t−1}‖` should ≤ γ | Q-evaluation on a fixed probe set per step (or online_params snapshots, much more memory) | Probe-set hook in `dqn_step` (could compose with `_value_probe`); needs probe-set design |

**Banach gap remaining:** the `corroborate` discipline is to log
raw values and reduce post-hoc. Banach contraction rate needs
*more raw data* — Q-evaluations on a fixed probe set per step,
or per-step parameter snapshots. Either is a probe pass
extension to the training loop, not a logging change.

**Lift when:** an experiment in §3-§5 needs the contraction-rate
gap as a measured outcome. The probe-set hook can compose with
the existing `_value_probe` in `train_phase`.

## Cosmetic / micro-cleanups

### `signature.py` reportAny errors — RESOLVED (2026-05-03)

**Status:** resolved via `_introspection_boundary.py`
(framework-types branch). The principled-adapter option (a)
above was taken: typed wrappers `get_type_hints_obj`,
`get_param_default`, `get_param_annotation`,
`get_field_default{_factory}`, `get_typing_args`,
`get_partial_args`, `get_partial_keywords`, `get_attr_obj`,
`get_bound_arguments` narrow `Any` → `object` /
`Mapping[str, object]` at one site. `signature.py`,
`_canonical.py`, and `computation_graph.py` route through it.

### `pytest.raises` over try/raise/except

**Status:** deferred.

**Description:** `tests/test_schema.py:84-95` and similar
use `try: raise AssertionError; except TypeError: pass` instead
of `pytest.raises(TypeError)`. Cosmetic style fix.

**Why deferred:** behaviorally equivalent; not blocking any
gate.

**Lift when:** a test refactor pass cleans up idioms.

### TraceContext records in completion order, not start order

**Status:** documented (claim.py).

**Description:** `Claim.__call__` appends `self` to the trace
*after* the underlying fn runs, so `f(g())` traces as
`[g, f]`. Documented in the new `trace_context` docstring.

**Why deferred:** behavior unchanged; the documentation is the
fix. Functional-claim graph derivation will need to handle
completion-order traces (or switch to a stack-based recording
if start-order matters).

**Lift when:** functional-claim graph derivation lands and
completion-order semantics turn out to be wrong for the use case.
