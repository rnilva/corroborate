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

**Status:** LIVE. Linear mediation now shipped as
`proportion_mediated`; counterfactual decomposition still
deferred.

**Description.** Linear-mediation decomposition (treatment ×
mediator additive, single-slope assumption) is sufficient for
most bridge claims; `proportion_mediated` (added 2026-05-05)
returns the indirect / total share with an `in_unit_interval`
diagnostic flag. The *counterfactual* decomposition — Pearl's
natural-direct (NDE) and natural-indirect (NIE) effects —
re-simulates the mediator's distribution under the
counterfactual treatment, identifying treatment×mediator
interactions and nonlinearity that linear-mediation can't.

**Lift gate is empirical, not a-priori.** Per ANALYSIS_RECIPE.md
§3a, three diagnostic signals indicate linear-mediation has
broken:

1. `proportion_mediated.in_unit_interval == False` (suppressor
   or overshoot).
2. Per-stratum partial-ρ heterogeneity across mediator-binned
   strata (treatment×mediator interaction).
3. LOESS RMSE materially smaller than linear-fit RMSE on
   Δ_M → Δ_Y (nonlinear functional form).

When a bridge's data fires one of these on real corpora, that's
the cue to author the counterfactual primitive. Until then,
linear is the right tool.

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

**Lift when:** a bridge author runs `proportion_mediated` on real
data and sees one of the three diagnostic signals fire, AND the
bridge's verdict needs the decomposition to be load-bearing.

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
