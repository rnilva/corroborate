# Bridge primitives manifest — typed knowledge missing from the framework

## Status

Open design. Multi-round build-out. Spec derived from the 2026-05-12
trim of `experiments/findings/ddqn_universe.py` (4187 → 2584 lines):
removing prose surfaced the typed primitives the framework should
have carried in the first place.

## Motivation

CLAUDE.md prohibits multi-paragraph docstrings and multi-line comment
blocks. The DDQN universe was in heavy violation: 1090 docstring lines
and 814 banner lines across 39 bridges. The trim pass removed 1603
lines of prose, but every removal exposed a load-bearing distinction
that *should* have lived in a typed primitive — not in narrative text.

Per CLAUDE.md's primitive-introduction discipline (the four-question
test in § "When to introduce a framework primitive"), each item below
encodes a typed contract that the substrate author should obey,
provides runtime narrowing that gives pyright real information, does
real work beyond labeling, or hits a performance floor Python
composition can't reach.

The pattern: **the framework forces narrative reasoning into
verdicts and measurables, with no typed surface between bridges.**
Bridges are sealed boxes; the relationships *between* them (companion,
falsification-pair, supersedes, channel-disjoint) live as prose
because the framework has no primitive to carry them.

## Primitives proposed

### 1. `VerdictReason` — refine `POWER_INSUFFICIENT` into typed sub-classes

**Symptom**: `POWER_INSUFFICIENT` is overloaded across three
structurally distinct conditions. The trim pass surfaced 11
bridges where multi-line "Bridge returns POW_INSUF because…"
banners disambiguated reasons. The framework can't say *why* in
the verdict itself.

**Sub-classes** (initial):

- `DATA_ORPHAN` — scope filter admits 0 cells; required corpus is
  missing. Currently 6 bridges (adaptive pair, polyak pair, n=10
  Monte Carlo, eff_h FR within-env).
- `DEGENERATE_PRIMITIVE` — the analysis primitive structurally
  fails (e.g., `proportion_mediated` returns a ratio outside [0,1]
  when total effect is small). Currently
  `target_staleness_late_mediates_outcome__minatar_intermediate_sync`
  pre-migration.
- `CI_STRADDLES_THRESHOLD` — real underpower; data is present but
  the CI doesn't confidently exclude the null. The dormancy bridge
  sits here.
- `ASSUMPTION_VIOLATION` — heavy-tail kurtosis flags fired (already
  has `paired_g.assumption_violations`, but no verdict-level expression).

**Where it would surface**: `Bridge.holds_when` returns
`tuple[Verdict, RefutationClass | None, VerdictReason | None]`.

**Earns its keep**: typed contract (4 cases close-set); narrows
pyright on the reason; does real work (different sub-classes
demand different remediation — re-collect data vs widen sample vs
fix methodology); replaces ~200 lines of prose explanation.

### 2. `BridgeRelationship` — typed edges between bridges

**Symptom**: 200+ lines of "companion to / sister bridge / sibling
of X" cross-references. Bridges form structured groups the
framework doesn't represent.

**Relationship kinds** (initial):

- `FALSIFICATION_PAIR(positive=B1, null=B2)` — paired bridges
  encoding the two endpoints of a falsification curve. e.g.,
  `ddqn_helps_at_full_bootstrap__fourrooms_n1` (HELD positive) +
  `ddqn_null_under_monte_carlo__fourrooms_n10` (HELD null). Both
  must HELD for the falsification claim to hold; cutting either
  amputates the structure.
- `POLARITY_STRATIFIED_PAIR(goal=B1, survive=B2)` — same claim,
  two polarity half-planes. Currently:
  `eff_h_mediates_g_link__{goal,survival}_envs`,
  `metamaze_link_steeper_at_high_gamma__{mean,median}`,
  `staleness_{amplifies,does_not_amplify}__polyak`.
- `SUPERSEDES_BY(old, new, reason)` — when a refactor replaces an
  older bridge form, the substantive claim is preserved via a
  different test. Auto-cleared when both bridges exist in the
  registry. Currently encoded as `# CLAIM X — CUT: substance in
  CLAIM Y` prose.
- `CHANNEL_DISJOINT(channel_a=B1, channel_b=B2, scope_predicate)`
  — two parallel-causal-paths claims on disjoint scope. CLAIM 26b
  (Hasselt channel, G1-active scope) and CLAIM 3 (Q-clip channel,
  G1-dormant scope). The module docstring spent 20+ lines
  explaining "these aren't competing — they're parallel".

**Where it would surface**: `Bridge.relationships:
tuple[BridgeRelationship, ...]` field. Runner can validate (both
bridges in registry, scopes disjoint where claimed, etc.).

**Earns its keep**: typed contract on claim-graph topology;
runtime narrowing for the runner's verdict-display ("HELD as
falsification-pair: both endpoints fire"); does real work
(structural integrity checks, e.g., refuse to merge a PR that
removes one half of a falsification pair without explanation);
replaces ~200 lines of cross-reference prose.

### 3. `Threshold` — typed-with-provenance, replaces raw floats

**Symptom**: `_rescue_threshold()` is a function returning
`float = 0.35`. The 35-line docstring derived `0.5 × (0.8 − 0.1)`
from three substrate parameters: `failure_baseline`,
`optimal_ceiling`, `rescue_fraction`. Today the derivation is
function-body math + free-text docstring; the threshold value is
opaque to the runner.

**Shape**:

```python
@dataclass(frozen=True, slots=True)
class Threshold[*Ts]:
    value: float
    derivation: tuple[*Ts]  # named substrate parameters
    rule: Callable[[*Ts], float]  # recomputes value
    rationale: str  # one-line WHY (the only prose allowed)
```

Or simpler: a typed dataclass per derivation pattern (`RescueThreshold`,
`NullCeiling`, etc.) with named fields and a `@property` returning
the float.

**Where it would surface**: every threshold in
`_native_diff_ci_verdict`, `_native_diff_null_verdict`,
`_rescue_threshold`. Verdict logic reads `threshold.value`, the
typed record carries provenance.

**Earns its keep**: typed contract (the three parameters are named
substrate concepts); runtime narrowing (pyright sees the derivation
fields); does real work (runner can emit threshold-with-derivation
to the snapshot, reviewer can re-evaluate at different
parameterizations without grep'ing function bodies); replaces ~50
lines of threshold-derivation prose across 5+ bridges.

### 4. `Tier` refinement — Pearl rung-1/2/3

**Symptom**: the existing `Tier.ASSOCIATIONAL | INTERVENTIONAL`
enum is too coarse. The module docstring tried to refine into
TIER A1 / A2 / INT / B in free text. "Pearl-rung-2 do(τ)"
terminology appears in 15+ docstrings.

**Shape**:

```python
class Tier(Enum):
    ASSOCIATIONAL = 'associational'      # rung 1
    DESIGNED_INTERVENTION = 'designed'    # rung 2 — do() via sweep
    COUNTERFACTUAL = 'counterfactual'    # rung 3 — DoWhy backdoor + refutations
```

Plus an orthogonal `Scope` axis:

```python
class ScopeBreadth(Enum):
    UNIVERSAL = 'universal'  # env-feature predicate; generalizes
    SAMPLED = 'sampled'      # env_name predicate; existence proof
    WITHIN_ENV = 'within_env'  # do(γ) / do(|A|) / do(τ) within one env
```

**Where it would surface**: `@claim_bridge(tier=Tier.DESIGNED_INTERVENTION,
breadth=ScopeBreadth.WITHIN_ENV)`. Runner groups verdicts by
(tier, breadth) for the verdict-landscape report.

**Earns its keep**: typed contract (3+3 closed-set); runtime
narrowing (`if bridge.tier is Tier.COUNTERFACTUAL: …`); does real
work (DoWhy bridges should auto-gate on backdoor.identified;
within-env do() probes should auto-stratify); replaces ~50 lines
of "Pearl rung-N" tagging prose.

### 5. `MethodologyHistory` — typed migration lineage

**Symptom**: "Migrated 2026-05-11 from `paired_g` to
`arm_mean_diff` per the seed-pairing critique" appears in ~12
docstrings. This is an edge in a methodology-history DAG.

**Shape**:

```python
@dataclass(frozen=True, slots=True)
class Migration:
    date: date            # ISO date
    old_primitive: str    # analysis name
    new_primitive: str
    reason: str           # one-line WHY (rotates to PR description after merge)
```

`Bridge.history: tuple[Migration, ...]`.

**Where it would surface**: bridge declarations. Optionally
auto-populated from git history at registry-build time.

**Earns its keep**: typed contract (each migration is a structured
event); runtime narrowing (`migrations[0].old_primitive ==
'paired_g'`); does real work (audit primitive: "show all bridges
whose latest migration was after 2026-05-01"); replaces ~30 lines.

### 6. `ScopeAggregate` — typed config-level predicate constructor

**Symptom**: I added `_G1_VANILLA_CONFIG_PREMISE_ACTIVE` and
`_VANILLA_CONFIG_Q_BOUNDED` helpers as bare `pl.Expr`
constants. Their structure — "vanilla-masked partition aggregate
over config keys with predicate" — is a recurring pattern.
Plus `partition_aggregate` already exists. But the *vanilla-only*
masking pattern (`pl.when(arm == 'baseline').then(...).otherwise(None)
.over(config_keys).mean()`) is ad-hoc per usage.

**Shape**:

```python
def config_mean_of_vanilla(
    column: str,
    *,
    config_keys: tuple[str, ...] = _DEFAULT_CONFIG_KEYS,
    baseline_arm: str = 'baseline',
    arm_field: str = 'arm_key',
) -> pl.Expr: ...

# Usage:
scope = (
    config_mean_of_vanilla('jensen_gap') > 0.05
    & config_mean_of_vanilla('jensen_dormancy_gap') < 0.05
)
```

Or, more ambitious, a `Scope` dataclass typing the (config-keys,
predicate-on-vanilla-aggregate) pair explicitly.

**Earns its keep**: tightens the pyright surface for
seed-asymmetric-filter avoidance (the audit's recurring concern);
codifies the "endogenous scope per `feedback_endogenous_scope_predicates.md`"
discipline; replaces ~3-4 module-level constants with a single
typed function call.

### 7. (deferred) `ChannelDisjoint` — two-channel scope model

**Symptom**: The two-channel architecture (Channel A Hasselt vs
Channel B Q-magnitude) lives in the module-level docstring. It's
load-bearing for interpreting CLAIM 2's POW_INSUF (Channel A
inactive but Channel B firing).

**Deferred** because it might be subsumed by
`BridgeRelationship.CHANNEL_DISJOINT` (#2) without needing a
separate primitive. Reconsider after #2 is in place.

## Round structure

Each primitive is independently shippable. Suggested order:

| round | primitive | why first |
|---|---|---|
| 1 | `VerdictReason` (#1) | Highest-volume prose replacement (~200 lines). Touches verdict-emit at one site. Low risk of cross-bridge breakage. |
| 2 | `Threshold` (#3) | Self-contained per bridge. No registry-level changes. Sets the "typed substrate parameter" pattern for later primitives. |
| 3 | `Tier` refinement (#4) | Touches every bridge decorator but mechanically (search-replace). Sets up #5's tier-aware audit primitives. |
| 4 | `BridgeRelationship` (#2) | Larger surface (registry-level relationships + validation). After #1 / #3 because relationship shapes can express verdict-tier constraints. |
| 5 | `MethodologyHistory` (#5) | Optional layer on top of existing bridges. Could auto-populate from git log. |
| 6 | `ScopeAggregate` (#6) | Targeted ergonomics fix once primitives 1-5 are in. |

## Out of scope for this manifest

- The `AdmissionGate` work proposed in `ADMISSION_GATES_DESIGN.md`
  is orthogonal (it gates which cells the bridge sees; this manifest
  is about typing what the bridge says). The two compose.
- The claim-unification work in `CLAIM_UNIFICATION_DESIGN.md` is
  substrate-side; this manifest is bridge-side.
- The `PRIMITIVES_AUDIT.md` discipline applies — every primitive
  here must pass the four-question test before implementation.

## Non-goal: extracting historical numbers

Empirical specifics ("ATE=-0.018 p=0.003", "Δ=+0.638 CI=[+0.594,
+0.682]") that I trimmed from docstrings live in `findings_*.md`
notes. Reincarnating them as typed fields on bridges would
re-create the original problem. The framework's responsibility is
to recompute verdicts on the current corpus; the prose was
defending against cache regressions, which is what
`*.run.json` snapshots are for.

## Acceptance for the manifest itself

Each primitive lands with:
1. PEP 695 typed dataclass / enum / Protocol — pyright strict.
2. Round-trip test in `tests/test_<primitive>.py` (closed-form, not
   substrate-redundant per CLAUDE.md § Test principle).
3. One ddqn_universe bridge migrated to use the primitive,
   demonstrating prose → typed conversion (PR-level proof of "earns
   its keep").
4. Updated `BRIDGE_AUDIT_TABLE.md` entry citing the primitive used.
