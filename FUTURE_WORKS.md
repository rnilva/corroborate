# Future works

Backlog of deferrable improvements surfaced during code reviews.
Each entry has: status, the rationale for deferral, and the
condition that should lift it.

Entries are ordered by *forcing function*: the higher up, the
sooner they're likely to bind.

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
