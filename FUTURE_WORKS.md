# Future works

Backlog of deferrable improvements. Each entry has: status, the
rationale for deferral, and the condition that lifts it.

Triage convention:
- **LIVE** — still deferred, lift condition unmet.
- **DOABLE** — lift condition fired or close; ready to revisit.
- **RESOLVED** — kept as history pointer (delete after a release).

Ordered by *forcing function*: higher up = sooner to bind.

---

## Endogenous-variable scope predicates

**Status:** LIVE. Active design tension; HP envelopes are the
current placeholder.

**Description:** Bridges authored with only `pl.col('env_name') == X`
scope pool cells across every corpus that has that env, mixing
different sweeps' HPs. The current minimal substitute is per-bridge-
family HP envelopes (e.g. `_FOURROOMS_REGIME = lr == 1e-4`,
`_ACTION_DIM_SWEEP_REGIME = lr == 1e-3 & capacity == 50000`) — see
`experiments/findings/dqn_bridges.py`. These work but enumerate
HP regimes by name, not the principled axis.

The principled scope is **endogenous variables** — features
observed/computed from the cell's own state, not metadata about
which sweep produced it. Examples:
- `jensen_dormancy_premise_active == 'held'` — cells where the
  bias-compounding mechanism's premise is active.
- Convergence-class membership (`with_cell_class` in
  `corroborate_rl.cell_classification`).
- Effective-horizon thresholds (`effective_horizon >= 20`).

Per ANALYSIS_RECIPE.md §0, classification is the canonical
pre-flight; bridges should consume the classification verdict
in their scope predicate.

**Lift when:** a substrate author writes endogenous-scope bridges
on a real claim and demonstrates the contamination-free pool. The
HP-envelope helpers can then migrate one-by-one as their
endogenous equivalent is authored.

---

## Wrapper-as-intervention gap

**Status:** LIVE. Memory: `project_wrapper_intervention_gap`.

**Description:** The substrate's env-wrappers (`RewardClippedEnv`,
the n-step bootstrap target, etc.) AREN'T proper Claims. Wrapping
IS structurally an intervention — installing a wrapper changes
the claim graph topology — but the framework's typed `Intervention`
contract requires a slot-Claim swap, which wrappers currently
aren't. Consequence: bridges that want to test `do(wrapper=X)
vs do(no_wrapper)` can't express it as a `DoEffect`; n_step
attenuation has to ride as a moderator (the slope bridges) or
within-arm HP cleavage rather than a Pearl-rung-2 contrast.

**Lift when:** the substrate refactors wrappers into slot Claims
(or the framework grows an `Intervention.wrap_with` shape). At
that point the n-step bridges can promote from
`Tier.ASSOCIATIONAL` to `Tier.INTERVENTIONAL`, and the slope
bridges become true `do()` contrasts.

---

## v10 `redundancy.py` port — toward v1 acceptance

**Status:** LIVE. Authoring layer mostly settled (32 bridges
stable across two zoo files); register/cycle persistence still
missing.

**Description:** v1 acceptance (closed dialectic loop with
axiom-19 reward) needs three primitives:

1. **`redundancy.py` — ΔI_redundancy primitive** (~240 LoC).
   4-factor jaccard·concord·intervention·identity overlap.
2. **`register.py` — append-only G register** (~120 LoC). Past
   comparisons + latest-wins fact projection.
3. **`compute_R_info` aggregator** (~80 LoC). Combines
   per-bridge ΔI with the redundancy term into `R(h)`.

**HPO-smuggle gate** is deferred: `signature(g_treatment) -
signature(g_baseline)` is the principled form (two interventions
producing the same structural signature ARE the same mechanism).
Lift only if a counterexample appears.

**Lift when:** v1 design starts; the authoring layer is settled
enough that the register's typing is forced by real consumers.

---

## Substrate extension contract not formalised

**Status:** LIVE, but more doable post-Phase-6.

**Description:** `corroborate_rl` is the only substrate. Framework
extension points (`Hypothesis` Protocol, `Bridge.source`,
`Loop[C, T, Idx]`, `Claim` Protocol, `Runner[R]` Protocol) are
implicitly substrate-extension surfaces, but no document says so
and no test exercises the framework with a non-RL substrate. The
verdict-consolidation refactor cleaned the substrate boundary
(typed contracts in framework, substrate-coupled YAML loaders
substrate-side); adding the formal contract is straightforward
now.

**Lift when:** a second substrate (supervised learning,
optimization, evolutionary search) is in play. Add: a "to add a
substrate, implement X / Y / Z" doc + a tiny non-RL substrate as
test fixture.

---

## Vectorised env support (n_envs > 1)

**Status:** LIVE — modern DRL feature, not v0 blocker.

**Description:** Modern DRL (SB3, CleanRL, v9) parallelises M
envs per cell via `vmap(env.step)` — buffer adds M transitions
per step. Wall-clock benefit is M× on slow envs (Atari ~1
ms/step). gymnax envs are microsecond-fast so the benefit at v0
scale is marginal.

**Lift when:** an Atari-grade env enters the sweep, OR step
budgets per cell exceed ~500k.

**Insertion point:**
- `n_envs: Annotated[int, Exogenous] = 1` on `dqn` kwargs.
- `rollout_phase` vmaps over an n_envs axis.
- `Replay.add` accepts a vector of M transitions.
- `DQNState.env_state` / `obs` become `(n_envs, ...)`-batched.

---

## Type-discipline refactorings (framework-types pass)

The framework now passes pyright cleanly; these are design
quirks the cleanup made visible.

### `iterate` / `Loop[C, T, Idx]` / record_call interaction

**Status:** LIVE.

**Description:** `iterate` exists primarily so the loop boundary
appears as one node in the computation graph. The mechanism is
heavyweight: every substrate that wants graph capture wraps loops
in `iterate`, which forces parametric `Loop` typing gymnastics.
Alternative: graph-extraction-time post-processor that recognises
loop boundaries from records, without the wrapping primitive.

**Lift when:** a non-RL substrate adopts `iterate` (cost becomes
visible) or graph extraction gains new requirements.

### Phantom `Analysis[R]` parameter

**Status:** LIVE.

**Description:** `Analysis[R: Mapping[str, object], O]` declares
`R` but never uses it in any field — pure documentation as type.
Either drop `R` to `Analysis[O]`, or wire it through to
`__call__(self, corpus: Iterable[R], **deps) -> O`.

**Lift when:** a substrate has multiple distinct record shapes
that need analyses keyed by R, OR the corpus shape stabilises
across analyses.

### Registry consolidation

**Status:** LIVE.

**Description:** Three independent `_REGISTRY` globals
(`measurable._REGISTRY`, `analysis._REGISTRY`,
`claim._FN_CACHE`). Could consolidate to `corroborate.registries`
with explicit handles — simplifies introspection ("what's
registered globally?") and testing.

**Lift when:** registry introspection becomes a first-class need
(a debug command, a registry-diff tool) — or a registry collision
bug appears.

### `analyses/__init__.py` 14 side-effect imports

**Status:** LIVE (small).

**Description:** `from .X import Y as _Y  # pyright: ignore
[reportUnusedImport]` × 14. Function-scoped imports inside a
`_register_default_analyses()` would avoid the ignores.

**Lift when:** doing another pass on `analyses/`'s public API.

### `signature.py` walker functional refactor

**Status:** LIVE (cosmetic).

**Description:** the recursive walker (200+ lines) uses mutable
state through helpers. Could be expressed functionally — each
helper returns a tuple, no mutation.

**Lift when:** signature.py needs other changes.

### Stubs maintenance procedure

**Status:** LIVE.

**Description:** 5 hand-written stubs (`gymnax`, `optax`, `scipy`,
`statsmodels`, `dowhy`). On upstream API drift, stubs silently rot.

**Lift when:** a dep upgrade trips a runtime AttributeError that
pyright would have caught with up-to-date stubs.

### Numpy axis-aware reduction helper

**Status:** LIVE (cosmetic).

**Description:** 5 nearly-identical `cast(npt.NDArray[np.floating],
arr.X(axis=axis))` lines in `reductions.py:reduce_axis`. A single
helper collapses to one cast.

**Lift when:** more axis-aware reductions are added.

### `pytest.raises` over try/raise/except

**Status:** LIVE (cosmetic).

**Description:** `tests/test_schema.py:84-95` and similar use
`try: raise; except: pass` instead of `pytest.raises(...)`.

**Lift when:** a test refactor pass cleans up idioms.

### TraceContext records in completion order, not start order

**Status:** documented (claim.py).

**Description:** `Claim.__call__` appends `self` to the trace
*after* the underlying fn runs, so `f(g())` traces as `[g, f]`.

**Lift when:** functional-claim graph derivation lands and
completion-order semantics turn out to be wrong.

---

## Theorem-gap measurables that need richer logging

**Status:** LIVE. Multiple gaps blocked on data the v0 StepRecord
doesn't carry.

**Background.** Invariants in `corroborate` measure *gap
magnitude* from theorem conditions, not threshold-bounded boolean
tests. Several DQN-claim gaps aren't computable from the v0
record.

**Currently shipped:** `fqi_decay_gap`, `hasselt_covariance_gap`,
`action_coverage_gap`, `jensen_overestimation_gap`,
`state_action_coverage_gap`.

| Gap | Theorem | Data needed | Lift gate |
|-----|---------|-------------|-----------|
| Banach contraction rate | Bertsekas-Tsitsiklis 1996 §6.3 — `r_emp = ‖Q_{t+1} − Q_t‖ / ‖Q_t − Q_{t−1}‖` should ≤ γ | Q-evaluation on a fixed probe set per step | Probe-set hook in `dqn_step` (could compose with `_value_probe`) |

**Lift when:** an experiment in §3-§5 needs the contraction-rate
gap as a measured outcome.

---

## Resolved (kept as history pointers)

- **Replay-as-Claim Protocol mismatch** — RESOLVED 2026-04-28.
  Replay is a config bundle, not a Claim. Lin 1992's claim
  attaches to the `sample` slot Claim, not to Replay itself.
- **`signature.py` reportAny errors** — RESOLVED 2026-05-03 via
  `_introspection_boundary.py`. Typed wrappers narrow `Any` →
  `object` at one site.
- **Refuter semantics (`is_refuter` flag)** — RESOLVED 2026-05-04.
  The Phase-6 verdict-consolidation refactor deleted
  `hypothesis_subgraph_verdict`, removing the only call site
  that would have flipped HELD for refuter edges. The
  xfail-style `predicted_direction='null'` (commit 6770abc) is
  the natural successor for "this should be null" claims —
  HELD now uniformly means "prediction confirmed" regardless of
  whether the prediction is positive/negative/null. Refuter
  intent rides on `predicted_direction='null'` (or the future
  alternative directions); no separate flag needed.
- **Bridge / Verdict / Direction surface in flux** — RESOLVED
  2026-05-04 by the verdict-consolidation refactor.
  `Bridge.source: str | Measurable | DoEffect` is the typed
  surface; the `intervention_edges` / `coupling_edges` two-path
  ambiguity dissolved when `hypothesis_subgraph_verdict` was
  deleted in favour of per-bridge `evaluate`. No
  `BridgeKind`-discriminator needed.
- **Module Claims → pure-functional** — RESOLVED 2026-05-04.
  `ClaimBase` no longer exists in the codebase. The substrate
  uses Free Claims (`@claim` on functions) + config bundles
  (frozen dataclasses, no `__call__`). The class-based-Claim
  escape hatch is documented in `claim.py` but unused. The
  duplication this entry warned about isn't present.
- **Typed `Powered` record on HypothesisComparisonRow** —
  OBSOLETE 2026-05-04. `HypothesisComparisonRow` was renamed
  `PairedComparisonResult` in the verdict-consolidation refactor;
  verdict-deriving fields including `adequately_powered: bool`
  were dropped. `verdict_from_paired_stats` in `stats/effect_size.py`
  still has `adequately_powered_paired` as a function — that
  surface is unchanged.
- **Vector outcomes (`primary_outcome_summary` widening)** —
  OBSOLETE 2026-05-04. `primary_outcome_summary` doesn't exist
  as a column. Bridges target outcome columns directly via
  `target='eval_best_burst_mean'` etc.; vector outcomes are
  expressed as substrate-side per-burst trajectories
  (`paired_g_per_burst`, `paired_link_per_burst`) — not by
  widening a single field.
- **`CorpusBridge[CorpusRow]`** — OBSOLETE 2026-05-04. The
  framework's analyses (`paired_g`, `meta_regression_paired_g`,
  `paired_link_per_burst`) already operate cross-cell. The
  speculative `CorpusBridge` design never had a concrete
  consumer; bridges now use analysis fixtures that consume the
  cell-level corpus directly.

---

## Speculative / dialectic-loop adjacent

### `CycleRef(index, id, parent_id)` for verdict-evolution tracking

**Status:** LIVE.

**Description:** `cycle_id: str | None` on every row groups but
doesn't order or support typed queries like "did this hypothesis's
verdict shift HELD→NO_EFFECT between cycles 4 and 5?". A typed
`CycleRef` with integer index and parent reference is the right
primitive when the redundancy register cares about cycle
sequencing.

**Lift when:** the dialectic-loop orchestrator lands AND
verdict-evolution diagnostics become a feature.
