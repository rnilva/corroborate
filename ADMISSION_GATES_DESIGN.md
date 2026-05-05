# Admission gates — design

Three-tier admission protocol for `@claim_bridge`-authored bridges:
**BLOCK** (typed-contract guards), **WARN** (quality nudges),
**INFO** (informational notes). Framework runs gates before the
bridge body and surfaces results on `BridgeEvaluation`.

## Motivation

Bridges currently encode admission criteria implicitly:

- `paired_g` raises `ValueError` if `treatment_arm == baseline_arm`
  — a runtime crash, not a verdict.
- `verdict_from_paired_stats` does power-gate logic *inside* the
  analysis, hidden from the bridge author.
- `Bridge.scope` accepts any polars expression; nothing checks
  whether the predicate references endogenous variables (the
  principled axis) or just HP-envelope filters (the temporary
  substitute).
- Bridge authors writing `if paired_g.n_pairs < 30: return
  POWER_INSUFFICIENT` repeat the same boilerplate in every
  bridge body. The intent is structural — admission, not body
  logic — but the framework offers no separate surface for it.

The proposal: lift admission criteria to first-class declarations
on the bridge decorator, with three severity tiers and a layered
authoring discipline.

## Principle — exogenous vs endogenous

The load-bearing axis for what counts as a causal source is
**exogenous vs endogenous**, not "leaf scalar vs structural slot."

- **Endogenous columns** are features computed from the cell's
  own state: registered `@measurable`-decorated columns
  (`jensen_gap`, `q_divergence_score`, `bootstrap_fraction`,
  `effective_horizon`, …), framework metadata (`env_name`,
  `seed`, `id`), and claim-graph slot identities (the typed
  `DoEffect` contrast on `Intervention(slot_path, replacement)`
  for *Claim* swaps).
- **Exogenous columns** are author choices that vary per cell
  but are not produced by the cell's dynamics: HP leaves
  (`gamma`, `n_step`, `reward_scale`, `optimizer.inner.lr`,
  `replay.capacity`, `total_steps`, `eval_every`).

`Tier.INTERVENTIONAL` (Pearl rung-2 causal claim) **requires an
endogenous source.** A bridge that wants `do(γ=0.999) → outcome`
must find the endogenous *delegate* (`effective_horizon`,
`q_divergence_score`) and source the bridge through that. The
exogenous knob is the upstream cause; the endogenous delegate is
where the causal claim attaches.

Why this rule is BLOCK-strength (rather than WARN):

- Pearl's `do()` is a SCM operation on variables observed/produced
  in the system. HPs are the SCM's parameters, not its variables;
  causal language about HP swaps without an endogenous delegate
  blurs structural-vs-parametric and invites HPO-shopping.
- The recent migration work (`q_divergence_score > 1`,
  `effective_horizon >= 50`, etc.) demonstrated that the
  endogenous delegate is usually right there — the substrate just
  has to surface it. Forcing the delegate is a forcing function
  for principled bridge authoring.

`Tier.ASSOCIATIONAL` and `Tier.INVARIANT` are unaffected:
observational HP × measurable bridges (slope claims via
`meta_regression_paired_g_by_nstep`, etc.) remain valid; they
just don't claim causality.

This principle is enforced by `EXOGENOUS_SOURCE` (BLOCK, L1
auto-gate). It's the source-side counterpart to `EXOGENOUS_SCOPE`
(WARN); together they police the exogenous-vs-endogenous axis at
the two surfaces (`Bridge.scope` and `Bridge.source`) where
authors can sneak HP-knobs into causal claims.

**The framework keeps `Intervention(slot_path, replacement)` with
`replacement: Replacement = FnClaim | partial | Callable` —
Claims only. The gate makes the typing's intent explicit; we do
NOT widen `Intervention[T]` to accept leaf scalars.** A future
"LeafIntervention" or `Intervention[T<:object]` shape would
require revoking this principle and isn't on the roadmap.

## Audit: v9 / v10 / current corroborate

**v9** (`poc_v8/framework/graph/admissibility.py`) shipped 25+
admission checks: `ReachableToGoal`, `DistinctArms`,
`EmpiricallyDistinct`, `FusedCouplingDetector`, `LocalityGate`,
`PriorRungVerified`, `ChainSignCoheres`, `EvidenceBreadth`,
`NotStructuralDuplicate`, `NotSubsumedByHeld`, `And`/`Or`/`Not`
combinators, etc. Each was a `Record`-subclass with custom
`__call__`; ~20-element `RefutationClass` enum and 5 `Verdict`
literals. Lesson: high entropy. Bridge authors had to know which
checks applied to which claim shapes; proliferation was the
forcing function for v10's reset.

**v10** (`poc_v10/admission.py`) shipped one check: `hpo_gate`.
Plain Python `functools.partial`-tree walk; ~160 LoC including
tests. Lesson: minimal works, but no growth path — when the next
gate is needed, the framework has no hook to attach it to.

**corroborate now** has the typed contract (Bridge.source,
Bridge.tier, Bridge.predicted_direction, Bridge.scope) but the
admission checks are scattered:

| v9 check | corroborate's equivalent |
|---|---|
| `DistinctArms` | `paired_g` runtime check (post-decoration) |
| `EvidenceBreadth` | `verdict_from_paired_stats` internal power gate |
| `Tier`-hierarchy | `Bridge.tier` enum, not enforced |
| `ScopeBindingResolves` | `Bridge.scope: pl.Expr`, not validated |
| `EmpiricallyDistinct` | nothing |
| `FusedCouplingDetector` | nothing |
| `ChainSignCoheres` | `Direction` enum carries data; no compose check |
| (new) HP-as-causal-source | `Bridge.tier == INTERVENTIONAL` is author-trust; nothing checks the source is endogenous |

The middle path: **a small set of well-named gates declared at
the bridge decorator, with three severity tiers**.

## Architecture

### Three severity tiers

```python
class GateLevel(Enum):
    BLOCK = 'block'  # Bridge → Verdict.INADMISSIBLE; body never runs
    WARN = 'warn'    # Bridge proceeds; warning recorded on
                     # BridgeEvaluation; suppressible via
                     # gate.acknowledge(name)
    INFO = 'info'    # Bridge proceeds; informational note for the
                     # report; not normally surfaced unless asked
```

The split distinguishes **typed-contract violations** (BLOCK —
the bridge can't produce a meaningful verdict) from **quality
nudges** (WARN — the verdict is meaningful but the authoring is
non-principled) from **diagnostic chatter** (INFO).

### Four authoring layers

Different gates live at different levels of the authoring
hierarchy:

```
[L1 framework auto-gates] → [L2 substrate library] →
[L3 module shared tuple]  → [L4 per-bridge gates kwarg]
```

- **L1** — universal invariants of the typed Bridge contract.
  Always-on; bridge authors don't declare them. Examples:
  `DISTINCT_ARMS` for `DoEffect`-sourced bridges,
  `RESOLVED_FIXTURES`, `EXOGENOUS_SCOPE` (auto-WARN).
- **L2** — substrate-author conventions, exposed as discoverable
  named constants in **plain module files** (not class-wrapped
  namespaces — see PRIMITIVES_AUDIT.md, reduction #3). Bridge
  authors `from corroborate_rl import scope, gate`. Examples:
  `scope.PREMISE_ACTIVE`, `gate.MIN_PAIRS(n)`. The lower-case
  module name signals "this is a discoverable namespace, not a
  class-as-namespace smell".
- **L3** — module-level shared tuples for repetition reduction.
  Bridge authors hoist common gate-tuples / scope-expressions
  into a module constant. Examples: `_DEFAULT_GATES`,
  `_FOURROOMS_REGIME` (already exists for scope).
- **L4** — per-bridge `gates=(...)` kwarg on `@claim_bridge`.
  Explicit author choice; the open-ended growth surface.

**The framework only knows about L1 and L4.** Layers 2 and 3
emerge from plain Python tuple composition (`+`) and polars `&`
— no framework primitives needed.

### Why this layering

- **L1 is small and fixed** — typed-contract guards only. Bridge
  authors should never have to think about them.
- **L4 is the open-ended growth surface** — substrate / module /
  bridge gates all flow through `gates=(...)`.
- **L2 is the substrate's contribution** — discoverable namespaces
  (`scope.PREMISE_ACTIVE`, `gate.MIN_PAIRS`) ship as plain
  constants from substrate libraries. Substrate-coupled but
  framework-independent.
- **L3 falls out of Python** — module-level constants composed
  via tuple `+` and polars `&`. The `_FOURROOMS_REGIME` pattern
  in `dqn_bridges.py` already does this for scope predicates.

This keeps the framework primitives small (one tier of auto-gates
+ one tier of explicit declarations) while admitting unbounded
substrate growth without framework changes.

## API surface

### Core types

```python
# corroborate/bridge/admission.py (new)

class GateLevel(Enum):
    BLOCK = 'block'
    WARN = 'warn'
    INFO = 'info'


@dataclass(frozen=True, slots=True)
class GateResult:
    """Result of running one admission gate against one bridge."""
    gate_name: str
    level: GateLevel
    passed: bool
    message: str


@runtime_checkable
class AdmissionGate(Protocol):
    """An admission gate is a typed callable taking the bridge and
    its filtered cells, returning a `GateResult` (or None when the
    gate doesn't apply, e.g., `EXOGENOUS_SCOPE` on a no-scope
    bridge)."""
    name: str
    level: GateLevel

    def check(
        self, bridge: 'Bridge',
        cells: 'Sequence[Mapping[str, object]]',
    ) -> GateResult | None: ...
```

### `Bridge` extension

```python
@dataclass(frozen=True, slots=True)
class Bridge:
    # existing fields ...
    gates: tuple[AdmissionGate, ...] = ()  # NEW — L4 declarations


@dataclass(frozen=True, slots=True)
class BridgeEvaluation:
    bridge_name: str
    verdict: Verdict
    analysis_results: Mapping[str, object]
    warnings: tuple[GateResult, ...] = ()       # NEW — WARN/INFO that fired
    blocked_by: GateResult | None = None        # NEW — set when verdict == INADMISSIBLE
```

A new `Verdict.INADMISSIBLE` member is added to the enum.

### Bridge.evaluate flow

```python
def evaluate(bridge: Bridge, cells: ...) -> BridgeEvaluation:
    filtered = filter_by_scope(cells, bridge.scope)
    all_gates = _AUTO_GATES + bridge.gates
    block_results: list[GateResult] = []
    warnings: list[GateResult] = []
    for gate in all_gates:
        result = gate.check(bridge, filtered)
        if result is None:
            continue
        if not result.passed and result.level is GateLevel.BLOCK:
            block_results.append(result)
        elif not result.passed:  # WARN or INFO
            warnings.append(result)
    if block_results:
        return BridgeEvaluation(
            bridge_name=bridge.name,
            verdict=Verdict.INADMISSIBLE,
            analysis_results={},
            warnings=tuple(warnings),
            blocked_by=block_results[0],  # first failure wins
        )
    # Bridge body runs with cells; existing fixture-resolution path:
    analysis_results = resolve_for_holds_when(bridge.holds_when, filtered, ...)
    verdict = bridge.holds_when(**analysis_results)
    return BridgeEvaluation(
        bridge_name=bridge.name, verdict=verdict,
        analysis_results=analysis_results,
        warnings=tuple(warnings),
    )
```

### `gate.acknowledge`

A bridge author who knowingly violates a WARN-level gate can
suppress its warning explicitly:

```python
@claim_bridge(
    source=INTERVENTION,
    scope=_FOURROOMS_REGIME & ...,
    gates=(gate.acknowledge('exogenous_scope'),),
)
def known_hp_envelope_bridge(...): ...
```

`gate.acknowledge(name)` returns an `AdmissionGate` whose
`check()` always returns `GateResult(passed=True, level=INFO,
message=f'acknowledged: {name}')` AND whose presence on the
bridge causes the framework to filter out any WARN-level gate
with the matching `name` from the `warnings` field.

## Phase 1 — gates worth shipping first

### L1 (framework auto-gates)

| Gate | Level | Why first |
|---|---|---|
| `DISTINCT_ARMS` | BLOCK | Universal invariant of `DoEffect`-sourced bridges; today's runtime `ValueError` becomes a clean verdict |
| `RESOLVED_FIXTURES` | BLOCK | Catches typo'd analysis-fixture parameter names at evaluate-time with a clear message |
| `EXOGENOUS_SOURCE` | BLOCK | Enforces the "endogenous source for `Tier.INTERVENTIONAL`" principle (cf. § Principle); blocks causal claims sourced on HP leaves like `gamma`, `n_step`, `reward_scale` |
| `EXOGENOUS_SCOPE` | WARN | Free detection (polars expr meta) signals HP-envelope bridges that haven't migrated to endogenous predicates |
| `NO_PREDICTED_DIRECTION` | INFO | Surfaces "this bridge can't detect sign-flip refutations" without blocking |

`DISTINCT_ARMS` and `RESOLVED_FIXTURES` are ports of existing
runtime checks; `EXOGENOUS_SOURCE`, `EXOGENOUS_SCOPE` and
`NO_PREDICTED_DIRECTION` are new. `EXOGENOUS_SOURCE` and
`EXOGENOUS_SCOPE` consume the substrate's endogenous-column
registry (see § Substrate registry, below).

### L4 (per-bridge declarations) — substrate-shipped gates

| Gate | Level | What it replaces |
|---|---|---|
| `gate.MIN_PAIRS(n)` | BLOCK | The `if paired_g.n_pairs < 30: return POWER_INSUFFICIENT` boilerplate every paired-g bridge currently writes |
| `gate.SOLVED_BY_BASELINE` | WARN | Bridges scoped to envs where the baseline arm already saturates the outcome (memory: `feedback_canonical_analyses` mention of saturation masking) |

### Detection rules

`EXOGENOUS_SCOPE` algorithm:

```python
def check(bridge: Bridge, cells: ...) -> GateResult | None:
    if bridge.scope is None:
        return None
    referenced = set(bridge.scope.meta.root_names())
    endogenous = referenced & (
        registered_names()           # @measurable-registered columns
        | _STANDARD_METADATA         # 'env_name', 'seed', 'id', ...
    )
    exogenous = referenced - endogenous
    if exogenous and not endogenous:
        return GateResult(
            gate_name='exogenous_scope',
            level=GateLevel.WARN,
            passed=False,
            message=(
                f'Bridge.scope references only exogenous (HP-leaf) '
                f'columns: {sorted(exogenous)}. The principled '
                f'scope-axis is endogenous (cf. ANALYSIS_RECIPE.md '
                f'§0); HP envelopes are a temporary substitute. '
                f'See FUTURE_WORKS "Endogenous-variable scope '
                f'predicates".'
            ),
        )
    return None
```

`EXOGENOUS_SOURCE` algorithm:

```python
def check(bridge: Bridge, cells: ...) -> GateResult | None:
    if bridge.tier is not Tier.INTERVENTIONAL:
        return None  # principle only applies to causal claims
    source = bridge.source
    # DoEffect source: every Intervention.replacement must be a
    # Claim-shaped swap (callable). LeafIntervention-shaped entries
    # would fire here. Today the type system already enforces this
    # (`Replacement = FnClaim | partial | Callable`); this gate is
    # belt-and-braces in case the typing relaxes.
    if isinstance(source, DoEffect):
        offenders = [
            iv.slot_path for iv in source.treatment + source.baseline
            if not callable(iv.replacement)
        ]
        if offenders:
            return GateResult(
                gate_name='exogenous_source',
                level=GateLevel.BLOCK,
                passed=False,
                message=(
                    f'Tier.INTERVENTIONAL bridge {bridge.name!r} has '
                    f'non-Claim Intervention(s) on slot(s) {offenders}. '
                    f'Causal claims require an endogenous source — find '
                    f'the delegate (e.g., effective_horizon for γ, '
                    f'q_divergence_score for sync_period) and route the '
                    f'bridge through it. See § Principle.'
                ),
            )
        return None
    # str / Measurable source: must reference an endogenous column.
    name = source if isinstance(source, str) else source.name
    if name in (registered_names() | _STANDARD_METADATA):
        return None
    return GateResult(
        gate_name='exogenous_source',
        level=GateLevel.BLOCK,
        passed=False,
        message=(
            f'Tier.INTERVENTIONAL bridge {bridge.name!r} sourced on '
            f'{name!r} which is not a registered measurable nor '
            f'standard metadata. Causal claims require an endogenous '
            f'source — find the delegate. See § Principle.'
        ),
    )
```

`NO_PREDICTED_DIRECTION` is even simpler: check
`bridge.predicted_direction is None`.

## Substrate registry (L2)

Both `EXOGENOUS_SOURCE` and `EXOGENOUS_SCOPE` need to know what
counts as "endogenous." The framework can't decide on its own —
that distinction is substrate-specific (RL substrate's
`q_divergence_score` is endogenous; an NLP substrate's columns
will differ). The framework provides:

- `registered_names() -> frozenset[str]` — every
  `@measurable`-decorated column the substrate has registered.
- `_STANDARD_METADATA: frozenset[str] = frozenset({
    'env_name', 'seed', 'id', 'arm_key', 'corpus', 'cycle_id',
    'parent_id', 'timestamp', ...})` — framework-controlled
  provenance columns; same on every substrate.

The substrate populates `registered_names()` by importing its
measurable modules (the existing
`import corroborate_rl.dqn.measurables  # populate registry`
pattern). When the cache materialises a column produced by a
registered measurable, the gate accepts it.

This means the substrate's "endogenous frontier" widens by adding
new `@measurable`-decorated functions. To promote `gamma` from
exogenous-HP to endogenous-mediator, the substrate authors a
measurable like `effective_horizon` (which already exists) and
sources causal bridges through it. The HP-leaf `gamma` itself
stays exogenous.

`DISTINCT_ARMS` for DoEffect:

```python
def check(bridge: Bridge, cells: ...) -> GateResult | None:
    if not isinstance(bridge.source, DoEffect):
        return None
    if bridge.source.treatment_arm_key() == bridge.source.baseline_arm_key():
        return GateResult(
            gate_name='distinct_arms',
            level=GateLevel.BLOCK,
            passed=False,
            message=(
                f'DoEffect treatment and baseline arms produce '
                f'identical canonical_str. The contrast is '
                f'self-vs-self; rebuild with a non-empty treatment '
                f'or baseline tuple.'
            ),
        )
    return None
```

## Reporting integration

`scripts/run_hypothesis.py`'s verdict-table extends with a
warnings-badge column:

```
ddqn_reduces_jensen_gap__fourrooms_n1   held              g=-3.26 ...
                                          ⚠  exogenous_scope
ddqn_outcome_attenuates__fourrooms_n3   held    (null)    g=+0.09 ...
ddqn_some_broken_bridge                 inadmissible      blocked: distinct_arms
```

The `verdict_distribution` analysis adds per-bridge warning
frequency to its summary.

## Phased rollout

1. **Phase 1: framework infrastructure** — `GateLevel`,
   `GateResult`, `AdmissionGate` Protocol, `Bridge.gates`,
   `BridgeEvaluation.warnings`, `Verdict.INADMISSIBLE`. ~120 LoC,
   one PR.
2. **Phase 2: 4 high-value auto-gates** — `DISTINCT_ARMS`,
   `EXOGENOUS_SOURCE`, `EXOGENOUS_SCOPE`, `NO_PREDICTED_DIRECTION`.
   ~50 LoC each. `EXOGENOUS_SOURCE` is the BLOCK-strength source-side
   counterpart to `EXOGENOUS_SCOPE`'s WARN; together they enforce
   the principle.
3. **Phase 3: substrate `scope.*` / `gate.*` namespaces** —
   `corroborate_rl.bridges_lib` exposes `scope.PREMISE_ACTIVE`,
   `gate.MIN_PAIRS(n)`, etc. Substrate-coupled, no framework
   changes.
4. **Phase 4: `gate.acknowledge(name)` + warning suppression** —
   the explicit "I know" surface.
5. **Phase 5: optional gates as use-cases bite** —
   `gate.PRIOR_RUNG_VERIFIED(needs=...)` (cross-bridge
   constraint, depends on FUTURE_WORKS "cross-bridge constraint
   declarations"), `gate.SIGN_COHERES(chain=...)`,
   `gate.SOLVED_BY_BASELINE`.

## Open questions

### Should L1 auto-WARN gates be opt-out?

Some bridges legitimately use HP envelopes (the substrate hasn't
shipped the endogenous predicate yet for that claim). Auto-WARN
on `EXOGENOUS_SCOPE` would emit a warning every run for those
bridges. Two approaches:

- **Always warn, suppress via `gate.acknowledge('exogenous_scope')`**
  — explicit per-bridge opt-out. Migration-friendly: warnings stay
  visible until the author handles them.
- **Opt-in via `gates=(gate.EXOGENOUS_SCOPE,)`** — default off,
  bridges that want the warning add it. Quieter but reduces
  discoverability of the smell.

Recommendation: **always-warn, explicit acknowledge**. The whole
point of the WARN tier is honest signaling; opt-in defeats it.

### Should L1 auto-BLOCK gates be overridable?

`DISTINCT_ARMS` blocks self-vs-self DoEffects; legitimate use
cases are hard to imagine. `RESOLVED_FIXTURES` blocks broken
fixture references; same.

Recommendation: **L1 BLOCK gates are NOT overridable**. They're
typed-contract invariants; "I'd like to break the contract" is
a code smell, not a feature.

### How does `INFO` differ from `WARN`?

The proposal lists three tiers but only two have clear uses:
BLOCK (the bridge can't produce a meaningful verdict) and WARN
(the verdict is meaningful but non-principled). INFO is reserved
for diagnostic notes that don't suggest a fix — e.g., "this
bridge ran on 1247 cells; 89% from corpus X."

If no concrete INFO gate emerges by Phase 5, drop the level. Two
tiers (BLOCK + WARN) may be enough.

### Cross-bridge gates (Phase 5)

`gate.PRIOR_RUNG_VERIFIED(needs=mech_bridge)` requires evaluating
ONE bridge before another. The current `runner.run` evaluates
bridges in `BRIDGES`-tuple order; cross-bridge gates need either:
- A topological-sort step in `runner.run` (lift the dependency
  resolution from FUTURE_WORKS "Cross-bridge constraint
  declarations"), OR
- An explicit `dependencies: tuple[Bridge, ...]` field on the
  bridge that declares ordering, OR
- A two-pass evaluation: pass 1 collects verdicts; pass 2 runs
  cross-bridge gates over the verdict map.

Defer until Phase 5 when an actual cross-bridge gate is needed.

## Risks / non-goals

### Not goals

- **Replicating v9's 25 checks.** v9 over-built; corroborate
  ships the 3-5 highest-value ones and adds when needed.
- **Lifecycle / dialectic-loop checks.**
  `NotStructuralDuplicate`, `NotSubsumedByHeld` belong with the
  `register.py` / cycle-persistence work in FUTURE_WORKS, not
  here.
- **Embedding admission logic in analyses.** The framework's
  current pattern (e.g., power-gate inside
  `verdict_from_paired_stats`) is a smell; the gate decorator
  surface is the principled home, but migrating existing
  analyses is a separate refactor — the new gates ship alongside
  the old internal logic, not as a replacement.

### Risks

- **Warning fatigue.** If WARN gates fire on too many bridges,
  authors will tune them out. Mitigation: ship only 2-3
  high-signal warnings in Phase 1; `gate.acknowledge` lets
  explicit-bridge-level opt-out without tuning out the gate
  globally.
- **Performance.** Auto-gates run on every bridge every
  evaluation. Mitigation: gates run on metadata + filtered cells
  only (after scope), not on full corpus; should be O(scope
  cardinality) at most.
- **Substrate coupling.** `scope.*` / `gate.*` namespaces are
  substrate-specific (the substrate decides what's endogenous).
  Mitigation: namespaces ship as substrate-library modules
  (`corroborate_rl.bridges_lib`); the framework doesn't import
  them. Substrates can ship their own.

## Decision

Proceed with Phase 1 + Phase 2 (framework infrastructure + 4
auto-gates) as one focused PR. Phases 3-4 follow as small
substrate-side and acknowledge-mechanism PRs. Phase 5 deferred
until cross-bridge constraints have a concrete consumer.

### Decisions on previously-open questions

- **BLOCK vs WARN for HP-as-causal-source.** `EXOGENOUS_SOURCE`
  is BLOCK. The bridge is asserting `Tier.INTERVENTIONAL`, which
  is a typed claim that an exogenous-source bridge cannot
  meaningfully satisfy. Unlike `EXOGENOUS_SCOPE` (WARN — the
  bridge can still produce a verdict on a non-principled axis),
  an INTERVENTIONAL claim sourced on an HP knob has no clean
  causal interpretation; the gate refuses to admit it.
- **Registry location.** Substrate-side. The framework provides
  `registered_names()` (built from `@measurable`-decorated
  columns the substrate registers via import) plus
  `_STANDARD_METADATA` (framework-controlled). The substrate
  doesn't ship a separate "admitted leaves" list — the
  endogenous frontier IS the registered-measurable set. To
  elevate a leaf scalar to causal-claim sourcing, the substrate
  authors a measurable that delegates the leaf (e.g.,
  `effective_horizon` for γ, `q_divergence_score` for
  `sync_period`).
- **Why no `LeafIntervention` typing widening.** The principle
  ("INTERVENTIONAL → endogenous source") is enforced at the gate
  level, not via type system. Widening
  `Intervention[T<:object]` to accept leaf scalars would invite
  HPO-shopping disguised as designed intervention; the BLOCK
  gate is the cleaner answer. `Intervention` keeps its
  Callable-only `Replacement` constraint.
