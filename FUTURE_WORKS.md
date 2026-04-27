# Future works

Backlog of deferrable improvements surfaced during code reviews.
Each entry has: status, the rationale for deferral, and the
condition that should lift it.

Entries are ordered by *forcing function*: the higher up, the
sooner they're likely to bind.

## Deferred from second-pass external review

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
