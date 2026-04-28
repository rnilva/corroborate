# Future works

Backlog of deferrable improvements surfaced during code reviews.
Each entry has: status, the rationale for deferral, and the
condition that should lift it.

Entries are ordered by *forcing function*: the higher up, the
sooner they're likely to bind.

## v10 audit — features and design degeneracies (2026-04-28)

**Status:** in-flight. This entry tracks the gap between
poc_v10's primitive set and corroborate's. v0 acceptance (§3
DDQN three-way verdict) does NOT require closing this gap;
v1 acceptance (closed dialectic loop with axiom-19 reward) does.

### Feature gaps (no v0 equivalent)

1. **`edges.py` — claim-graph derivation from trace** (~150 LoC).
   `extract_edges(records)` builds inter-Claim edges by `id()`
   matching args ↔ outputs. Mechanism dataflow *derived*, not
   declared. corroborate has TraceContext + CallRecord but the
   trace is currently dead-end data.
2. **`measurable_graph.py` — statistical graph** (~80 LoC).
   Pairwise Pearson over per-step record + "explained-by-claim-
   graph" diagnostic.
3. **`computation_graph.py` — graph signature** (~70 LoC).
   Hashable fingerprint for mechanism_key extension.
4. **`causal_graph.py` — Pearl-ladder typed BridgeEdges** (~270
   LoC). `direction × tier × evidentiary_level` with
   ASSOCIATIONAL → INTERVENTIONAL promotion algebra.
5. **`redundancy.py` — ΔI_redundancy primitive** (~240 LoC).
   4-factor jaccard·concord·intervention·identity overlap
   closing biases 3, 4, 5. The principled axiom-19 redundancy.
6. **`hypothesis_row.py` — single canonical aggregator** (~720
   LoC). `HypothesisComparisonRow.from_cells(h, t, b)` is THE
   one aggregation function. Source-and-view pattern.
7. **`register.py` — append-only G register** (~120 LoC). The
   dialectic-loop's register of past comparisons + latest-wins
   fact projection.
8. **HPO-smuggle gate.** *Status revised 2026-04-28.* v10's
   `admission.py` walks `partial.keywords` and buckets diffs as
   `scalar` vs `callable`; pure-scalar diffs are rejected as HPO
   smuggle. Heuristic, not principled. The principled form is
   `computation_graph.signature(g)` — two interventions
   producing the same structural signature ARE the same
   mechanism; differing signatures are real interventions. Ported
   the graph system in commit (this entry); admission as a
   separate module is **deferred** unless graph-signature equality
   proves insufficient. Lift conditions: a counterexample where
   sigs match but mechanisms differ (graph-invisible branch flip,
   for example) AND it matters for the dialectic loop.
9. **R-formula / axiom-19 implementation** (~450 LoC of
   smoke). corroborate has all the inputs (per-bridge ΔI,
   ΔI_population in stats path) but no consumer that aggregates
   them into `R(h)`.

**Tally:** ~2.0–2.4 KLoC of v10 functionality not yet ported.

### Design-level degeneracies

- **D1: Five row types vs three.** ArmRow is intermediate
  machinery no real path uses; aggregate.py's 772 LoC has
  multiple half-roads vs v10's one canonical `from_cells`.
- **D2: mechanism_key empirical vs declared.** corroborate
  derives identity via `leaf_signature(measurements)`; v10 has
  declared `Hypothesis.mechanism_key`. Probable degeneracy
  when register dedup is needed.
- **D3: Reads-set incomplete on Bridge.** `Bridge.targets ⊊
  reads`. Closure (`targets ∪ measurable transitive deps`) not
  assembled; ΔI_redundancy can't be wired without it.
- **D4: No FactRow.** Bridge results land flat-keyed in
  `RunRow.measurements`; per-fact `delta_i` /
  `natural_strength` / `evidentiary_level` not preserved.
  Loses redundancy-primitive substrate.
- **D5: No graph derivation from trace.** TraceContext logs
  but doesn't derive edges. Dead-end data.
- **D6: No HPO-smuggle admission.** Any intervention is
  admissible. Closed loop will need it.
- **D7: Three-claim taxonomy might be over-fit.** Module
  Claim / Free Claim / config bundle vs v10's "claim is a
  function, module is a dataclass." Replay-as-Claim saga
  suggests this has wobbled.
- **D8: No axiom-19 computation.** Theory in PAPER_NOTES.md;
  zero functions implement it. Reward signal of dialectic
  loop is unimplemented.

### Sequencing (post-§3-acceptance)

1. **Faithful intervention + auto-induced graphs** (LANDED
   2026-04-28). `graph.py` + `computation_graph.py` port v10's
   generic Graph[N, M] + Edge / ComputationEdge / signature.
   `extract_raw_edges` / `build_computation_graph(records)`
   derive the claim graph from `trace_context()` records by
   `id()` matching. The structural signature `signature(g)` is
   the principled HPO-prevention primitive (subsumes admission
   heuristic). 26 tests; faithful-intervention property
   demonstrated (HP tweak → empty diff; slot swap → non-empty).
2. **D3 + D4 + redundancy system** (next): FactRow lift,
   reads-set closure, redundancy.py, register.py, compute_R_info.
   ~600 LoC bundle.
3. Cleanup step 1 (move rl/dqn/ out of corroborate namespace),
   step 3 (pull cell_runner reductions to experiment), step 4
   (lift worker-pool plumbing into sweep.py).
4. Causal graph + measurable graph (Pearl-ladder typed
   BridgeEdges + statistical-graph diagnostic). Unblocks PAPER
   §3.5 in full v10 shape.

**Lift when:** v0 acceptance lands (§3 verdict table renders
on real sweep data); v1 design starts.

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

## Step 3: cell-runner trace emission

**Status:** queued. Next step in the masterplan after Steps 1-2
landed columnar persistence.

**Description:** `TraceRow` exists as schema + standalone smoke
(`test_trace_persistence.py`); `run_dqn_cell` / `run_dqn_arm`
build `RunRow`s only. The two stores aren't yet linked end-to-end
on disk. After this:

- Each cell produces both a `RunRow` and a `TraceRow` with
  `RunRow.id == TraceRow.id`.
- `run_dqn_arm` writes to two parquets via `write_runrows` and
  `write_tracerows`.
- `experiments/collect_ddqn_runs.py` writes `runs.parquet` +
  `traces.parquet` per arm.

**Why deferred:** Steps 1-2 were the design-load-bearing changes
(decide the shape; collapse the JSON-wrapping crime). Step 3 is
mechanical plumbing — modest LoC, no design open questions.

**Lift when:** ready to actually persist a multi-cell sweep with
both stores. Smallest user-visible change with largest practical
effect; once it lands `df.filter(pl.col('optimizer.inner.lr') <
1e-3)` is the workflow that exists end-to-end on disk, not just
in tests.

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

### Paired-cell comparison primitive (matched-seed Δ)

**Status:** deferred.

**Description:** `aggregate.comparison_from_arms(treatment,
baseline)` produces an unpaired `ComparisonRow` from two
ArmRows already aggregated across seeds. §3's central claim
("DDQN reduces Jensen-gap by Δ vs vanilla") is more
statistically powerful as a *paired* test on matched
(env, seed) pairs — Δ_i = treatment_i − baseline_i, then
Hedges' g of the Δ distribution vs zero. The pairing helper
(`pair_runs_by_seed(treatment, baseline) →
tuple[(RunRow, RunRow), ...]`) and the paired statistics
aren't yet implemented.

**Why deferred:** Step 5 (MDE + Hedges' g + derived q) is the
consumer for paired statistics. Pairing without paired-stats
would be a half-shipped primitive; bundling them keeps the
statistical contract coherent. Audit pass 4 confirmed the
unpaired path (`comparison_from_arms`) makes §3 expressible
end-to-end — not optimal, but viable as a v0 acceptance.

**Lift when:** Step 5 lands paired stats. The factory shape
should be: `paired_comparison_from_runs(treatment_runs,
baseline_runs) → ComparisonRow` — pairs by (env_name, seed),
computes Δ per pair, produces ComparisonRow with
`effect_size_g` of the Δ distribution.

### Typed `Powered` record on ComparisonRow

**Status:** deferred.

**Description:** `ComparisonRow.adequately_powered: bool`
collapses the gradient that derived it (n + sd + observed g + MDE).
A typed `Powered(mde_d, observed_g, achieved_power)` would carry
the inputs so two stat implementations can't disagree about what
"adequately" means.

**Why deferred:** v0's auto-detection (`_is_adequately_powered`)
matches v10's pattern; bool tracks the gate result. Step 5's
MDE+power module is the consumer that would benefit from the
richer record.

**Lift when:** step 5 lands and needs to log per-env power-curve
diagnostics (PAPER_NOTES.md §3.4 has 5 of 17 mechanism-HELD with
adequate power; the rest POWER_INSUFFICIENT — recording WHY each
fell which way is the natural next step).

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
`lin_iid_gap`, `hasselt_covariance_gap` (Pearson r over kept
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
