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

This is **the first admission gate** in `corroborate` — v9 / v10
shipped with plenty of them and the pattern is well-known from
those versions. The principled scope is **endogenous variables**
— features observed/computed from the cell's own state, not
metadata about which sweep produced it. Examples:
- `jensen_dormancy_premise_active == 'held'` — cells where the
  bias-compounding mechanism's premise is active.
- Convergence-class membership (`with_cell_class` in
  `corroborate_rl.cell_classification`).
- Effective-horizon thresholds (`effective_horizon >= 20`).

Per ANALYSIS_RECIPE.md §0, classification is the canonical
pre-flight; bridges should consume the classification verdict
in their scope predicate.

**What's needed:** a small `Scope.*` namespace of named
admission-gate polars expressions that bridges import and AND
into their scopes. Each named gate is a one-liner; the value is
naming + discoverability, not new computation.

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

## HP-as-intervention + n-way contrasts

**Status:** LIVE. Real but difficult — the HP/Intervention
distinction is load-bearing in the framework.

**Description.** Two coupled framing problems:

1. **HPs are technically interventions.** `do(γ=0.999)` IS a
   Pearl rung-2 manipulation if γ is treated as a manipulable
   variable. The framework currently splits HPs (cell covariates)
   from Interventions (slot Claim swaps) on the principle that
   HPs don't change the claim graph topology, only leaf scalar
   values. But the tracer's `walk_paths` already exposes HPs as
   leaf claims at composition time — the typed surface to lift
   them to Pearl rung-2 is half-built.

2. **N-way contrasts.** `DoEffect` is single-treatment-vs-single-
   baseline. Multi-arm sweeps (`expectile_3way` with 3 arms,
   `dampened_alpha_envs` with 5 α values, n-step grid with 5 n
   values) are authored as N pairwise bridges; the omnibus
   "any arm differs" or "monotone trend" claim has no first-class
   form.

**Coupling.** N-way HP-cleavage IS the n-way intervention
case once HPs are lifted to typed Interventions. A
`MultiDoEffect(arms: tuple[Intervention | LeafIntervention, ...])`
where `LeafIntervention(leaf_path: str, value: object)` carries
HP swaps would unify both. The slope/meta-regression bridges
already express the panel form for one HP axis; what's missing
is the typed-DoEffect-of-arms shape so the analyses can detect
n-way rather than being told it's a single contrast.

**Risks.**
- Blurs the structural-vs-parametric distinction the framework
  currently leans on (memory: HP swap ≠ structural change).
- Pearl's `do()` makes most sense for variables in the SCM;
  HPs are exogenous parameters of the SCM, not variables in it.
  Lifting them changes the formal interpretation.
- The existing "predicted_direction is for arm contrasts; HP
  cleavage rides on slope/meta-regression" convention has to
  be revisited.

**Lift when:** a substrate scientist needs the omnibus / monotone
claim form for a multi-arm sweep AND has a clear answer for the
HP-vs-Intervention boundary in their domain. Both `LeafIntervention`
and `MultiDoEffect` are smaller-than-they-look once the framing
is settled — but the framing needs settling first.

---

## Counterfactual mediation primitive (Pearl NDE/NIE)

**Status:** LIVE. Linear mediation now shipped as the salvaged
`mediation_dowhy` (typed `linearity_status` diagnostic);
counterfactual decomposition still deferred.

**Description.** Linear-mediation decomposition (treatment ×
mediator additive, single-slope assumption) is sufficient for
most bridge claims; `mediation_dowhy` (salvaged 2026-05-18 from
the v9 `proportion_mediated`) returns total / direct / indirect
ATEs + indirect proportion + a typed `linearity_status` field
that classifies the linear assumption's defensibility on the
corpus (RELIABLE / SIGN_FLIPPED / OUT_OF_BOUNDS /
UNIDENTIFIED / POWER_INSUFFICIENT). The *counterfactual*
decomposition — Pearl's natural-direct (NDE) and natural-
indirect (NIE) effects — re-simulates the mediator's
distribution under the counterfactual treatment, identifying
treatment×mediator interactions and nonlinearity that
linear-mediation can't.

**Lift gate is empirical, not a-priori.** Three diagnostic
signals indicate linear-mediation has broken:

1. `mediation_dowhy.linearity_status in {SIGN_FLIPPED,
   OUT_OF_BOUNDS}` (suppressor / overshoot / sign-flipped
   direct ATE — typed at the result surface).
2. Per-stratum partial-ρ heterogeneity across mediator-binned
   strata (treatment×mediator interaction).
3. LOESS RMSE materially smaller than linear-fit RMSE on
   Δ_M → Δ_Y (nonlinear functional form).

When a bridge's data fires one of these on real corpora, that's
the cue to author the counterfactual primitive. Until then,
linear is the right tool (and the `linearity_status` field tells
the bridge author when it isn't).

**Likely first triggers** (per the substrate's known findings):
- Per-burst link panels (memory `findings_minatar_link_attenuation`,
  `findings_fourrooms_time_series`) show non-stationary mech-link
  slope across phases — candidate for signal #1.
- Q-explosion regimes (`findings_q_amplification_cartpole`):
  jensen_gap blows up faster than outcome — clearly nonlinear
  M → Y, candidate for #3.
- Underlearning rescue (`findings_underlearning_rescue`):
  reward-scale-dependent suppressor effects — candidate for #2.

**What it would look like.** An `@analysis def
mediation_counterfactual(...)` returning a `MediationResult`
with NDE, NIE, and bootstrap CIs — implementable on top of
DoWhy's `mediation_estimand` (which exists in DoWhy but isn't
wrapped in `corroborate.analyses.dowhy`).

**Lift when:** a bridge author runs `mediation_dowhy` on real
data and the typed `linearity_status` returns SIGN_FLIPPED or
OUT_OF_BOUNDS at a scope where the bridge's verdict needs the
decomposition to be load-bearing (rather than just a sibling
diagnostic alongside `partial_spearman`'s rank-based answer).

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

**Status:** LIVE. Some gaps blocked on data the v0 StepRecord
doesn't carry.

**Background.** Invariants in `corroborate` measure *gap
magnitude* from theorem conditions, not threshold-bounded boolean
tests. Several DQN-claim gaps aren't computable from the v0
record.

**Currently shipped:** `fqi_decay_gap`, `hasselt_covariance_gap`,
`jensen_overestimation_gap`, `state_action_coverage_gap`,
`jensen_dormancy_gap`, `jensen_floor_late`,
`banach_contraction_gap_coarse` (eval-burst probe — added
2026-05-05).

| Gap | Theorem | Data needed | Lift gate |
|-----|---------|-------------|-----------|
| Banach contraction rate (strict) | Bertsekas-Tsitsiklis 1996 §6.3 — per-step `r_emp = ‖Q_{t+1} − Q_t‖_∞ / ‖Q_t − Q_{t−1}‖_∞` should ≤ γ | Per-step Q-evaluation on a *designed* probe set (currently the eval-rollout's init states are the probe; the probe isn't coverage-balanced and the iteration spans `eval_every` TD updates, not one) | Probe-set hook in `dqn_step` + designed probe-set construction |

**Coarse version shipped.**
`banach_contraction_gap_coarse` reads the existing burst-spaced
`predicted_q_at_start` array and computes a geometric-mean ratio
over consecutive bursts — the same shape `fqi_decay_gap` uses
for FQI's multiplicative decay. Returns `max(0, geomean_ratio −
γ)`; non-zero gap means Banach is empirically violated on
average across bursts.

**Why the strict version is still LIVE.** The coarse measurable
trades two coarse-grainings for the textbook gap:
1. Probe set is the eval-rollout init states (K per burst), not
   a designed coverage-balanced sample. For envs with
   `resample_init_pos=False` the probe collapses to a single
   state.
2. The "iteration" in the ratio spans `eval_every` TD updates,
   not a single update. The actual Bellman-step contraction
   rate is undermeasured at this granularity.

**Lift when:** an experiment needs the strict per-step Banach gap
(coverage-balanced probe set, single-step ratio). The probe-set
hook can compose with the existing `_value_probe` in
`train_phase`.

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

---

## Framework: `Tier.INVARIANT` cross-node composition

**Status:** OPEN.

**Surfaced 2026-05-22** during `hasselt_clean` hypothesis
authoring (see `experiments/findings/hasselt_clean/`).

**Problem.** `Tier.INVARIANT` is currently a *self-loop-only*
edge type: substrate axioms (e.g. Hasselt's σ-floor bound
expressed as `jensen_dormancy_premise_active_bridge`) are
modeled as self-loops on the source measurable, with
`AT_MOST` / `AT_LEAST` direction. The `chain_tier` walker in
`src/corroborate/graph/causal.py:231-248` explicitly *skips*
invariant edges, documented as "self-loops, never compose into
cross-node chains".

When a substrate author wants to express "theoretical premise
activation → mechanism activation" as a first-class chain edge,
the current framework forces an associational sibling: a
non-invariant bridge from the premise measurable to its
downstream consequence (see B1 in `hasselt_clean/chain.py`).
This is functionally equivalent but semantically a workaround —
the invariant *is* the substrate's axiom about the cross-node
relationship; modeling it as a self-loop loses the upstream-edge
structure.

**Proposed surface.** Allow `Tier.INVARIANT` cross-node edges
with the upstream measurable as `source`, the downstream
consequence as `target`, and the axiom's predicate (e.g.
`σ-floor ≥ observed_bias`) encoded via the bridge's
`threshold` + `direction` fields. The synthesized per-cell
verdict (`held` / `invariant_violation` / `power_insufficient`)
would then live on a *typed cross-node edge* in the post-eval
graph. `chain_tier` would need to be updated to skip invariants
only when they're self-loops (substrate axioms about a single
measurable) and include them as the minimum-tier link when
they're cross-node.

**Lift when:** another substrate's authoring discipline
exercises the same pattern (theoretical premise as upstream
edge), confirming the abstraction generalises beyond the DDQN
case study. Until then `hasselt_clean`'s associational B1 is
the load-bearing workaround.

---

## Substrate: 5-env `jensen_dormancy_gap` backfill regression

**Status:** OPEN engineering debt.

**Surfaced 2026-05-22.** During cluster-wide `rep_ea` backfill,
`scripts/rep_ea_serial.py` rewrote per-corpus
`measurements.parquet` files for Acrobot / MetaMaze / MC / LL
with a narrow `required=` set that did NOT include
`jensen_dormancy_gap` (or its transitive `online_std_q_per_step`
trace read). `build_measurements` overwrote the per-corpus
store with only the requested measurables + transitive deps,
truncating the dormancy column from those files.

**Cache state at time of writing:** the framework cache
(`experiments/data/cache/ddqn_sweeps.parquet`) retains
`jensen_dormancy_gap` finite values for 5 of 10 canonical-pool
envs from earlier ingests (Acrobot, LL, MetaMaze, MC, Snake);
the 5 MinAtar+FR envs have the column null. Per-corpus stores
for Acrobot / MetaMaze / MC are truncated — re-ingest from them
will *remove* dormancy from the cache.

**Lift when:** trace storage budget allows for cluster-wide
re-restore of the 5 missing envs (~5GB MinAtar + 12GB FR) to
recompute jdg under the full `required` set. The
`hasselt_clean` hypothesis is scoped to the 5 envs that
currently have jdg coverage; extending it to the full 10-env
canonical pool requires this backfill.

---

## Framework + substrate discipline: bridge verdict-body uniformity

**Status:** OPEN.

**Surfaced 2026-05-22** during `hasselt_clean` chain authoring
(see `experiments/findings/hasselt_clean/`) and the comparison
to `experiments/findings/ddqn/finding_hasselt_chain.py`'s
Stage 1 bridge `algorithm_reduces_bootstrap_gap_magnitude`.

**Problem.** Two distinct verdict-body shapes coexist for
random-effects pooled bridges:

1. **Custom threshold on `pooled_d`** (CI-like): the bridge
   body reads `result.pooled_d` and compares to a fixed
   threshold (`d ≤ -0.3 → HELD`). Bypasses I², τ², the
   prediction interval. Effectively a CI-based "average effect
   is reliably non-zero in our sample" test. Used in
   `algorithm_reduces_bootstrap_gap_magnitude`.

2. **Framework `.verdict` property** (PI-based): the bridge
   body delegates to `result.verdict`, which dispatches to
   `random_effects_verdict` (PI-based: PI excludes zero in
   predicted direction → HELD/HELD_WITH_SCOPE_FLAG; PI brackets
   zero → NO_EFFECT/NULL_EFFECT). Used in
   `intervention_reduces_bias__premise_active`.

Both are honest answers to different questions:
- (1) tests *"is the average effect across our sample reliably
  non-zero?"* — adequate when envs are exchangeable.
- (2) tests *"would a new env from this population reliably show
  this direction?"* — the framework's prediction-interval
  extrapolation discipline.

On the canonical DDQN panel (γ=0.999, 9-env), the same
underlying data fires HELD under (1) at d=-0.6 and NO_EFFECT
under (2) at d=-1.9 — because cross-env heterogeneity is
extreme (I²=0.97, MinAtar d=-8.9 vs classical d=-0.01)
and the PI brackets zero despite the CI being decisively
negative.

**The substantive issue underneath**: when environments aren't
exchangeable (they differ in network class, Q-magnitude, reward
sparsity, etc.), neither verdict shape is the right tool — the
random-effects model's homogeneity assumption is structurally
violated. The principled answer is a *moderator-aware sibling*
(meta-regression or `cross_stratum_property_slope`) per
`HYPOTHESIS_AS_GRAPH.md` §3b, which tests cleavage by env-feature
rather than pooling across heterogeneous strata blindly.

**Proposed framework discipline.** Either:
- (a) Deprecate verdict-body shape (1) — require all
  random-effects-pool bridges to use `.verdict`. Forces honest
  exposure of cross-env heterogeneity through
  `HELD_WITH_SCOPE_FLAG` or `NO_EFFECT/NULL_EFFECT` per the PI
  test; substrate authors then explicitly author the
  moderator-aware sibling when heterogeneity bites.
- (b) Keep both, but ban shape (1) without an accompanying
  moderator sibling in the same Finding. Pool-as-CI-test is fine
  iff a moderator-aware sibling answers the "why so much
  heterogeneity?" follow-up.

(b) is the lighter-touch discipline.

**Lift when:** the next substrate-author writing a cross-env
pool bridge faces the choice. Until codified, in-tree bridges
mix shapes — the original `ddqn/finding_hasselt_chain.py` uses
(1) at Stage 1; `hasselt_clean` uses (2) at B3. Both are
defensible at v1 cost.
