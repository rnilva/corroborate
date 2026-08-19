"""Admission gates — typed callables checked at `evaluate()`-time.

Three severity tiers per `ADMISSION_GATES_DESIGN.md`:

- **BLOCK** — bridge can't produce a meaningful verdict. The
  framework returns `Verdict.INADMISSIBLE` with `blocked_by`
  pointing at the offending gate; the bridge body never runs.
- **WARN** — bridge proceeds, but the verdict is non-principled
  in a known way (HP-envelope scope, missing predicted_direction,
  etc.). Warning surfaces on `BridgeEvaluation.warnings`.
- **INFO** — diagnostic note; not normally surfaced.

A gate is a plain callable: `(bridge, filtered_cells, ctx) ->
GateResult | None`, with `ctx: GateContext` carrying the
evaluation's claim composition and/or leaf registry (the
framework constructs it; gates only read it). Returning None
means "this gate doesn't apply to this bridge"; returning a
`GateResult` with `passed=True` is informational; with
`passed=False` triggers the level's behavior.

The framework's auto-gates (`AUTO_GATES`) are unconditionally
prepended to every bridge's `gates` tuple at `evaluate()`-time.
Per-bridge `gates=(...)` declarations on `@claim_bridge`
add to the auto-gate list.

Native endogeneity gating (`exogenous_source`, `exogenous_scope`)
consumes the substrate's claim chain via `ctx.claim` —
`is_endogenous(name, claim)` keys on
`walk_paths(claim, regime='leaf')`. Without that topology, a plain
source cannot self-authorise an interventional bridge; external
effects declare an explicit value-based `DoEffect`, whose
declared columns must be registered configuration (`ctx.leaves`).
See `ENDOGENEITY_TOPOLOGY.md` for the three structural rules.
"""
from __future__ import annotations

import functools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Protocol

from corroborate.core.claim import Claim
from corroborate.core.intervention import DoEffect
from corroborate.core.signature import walk, walk_paths
from corroborate.measurables import registered_names, transitive_reads


@functools.cache
def _claim_leaves(claim: Claim[..., object]) -> frozenset[str]:
    """Cached union of `walk_paths(claim, regime='leaf')` and
    `walk_paths(claim, regime='exogenous')` keys — every name the
    walker recognises as a configurational primitive of the
    substrate's outermost claim. Both regimes are author-set at
    design time:

    - `regime='leaf'`: arm-level config (`gamma`, `replay.capacity`,
      `optimizer.inner.lr`, …).
    - `regime='exogenous'`: per-cell framework-injected values
      (`env`, `env_params`, `n_actions`, …) plus per-cell author
      grid dimensions (`env_name`, `seed`, `wrappers`).

    Both are exogenous from the gate's perspective — neither is
    "produced by the cell running" — so the endogeneity classifier
    needs the union. (Note: in current substrates with `Env` under
    `TYPE_CHECKING`, the runtime regime detection collapses
    everything to 'leaf' due to unresolved forward refs, so this
    union is currently equal to `walk_paths(regime='leaf')`. The
    union shape stays correct if forward-ref handling improves.)

    FnClaim is hashable (frozen dataclass), so functools.cache
    keys cleanly."""
    sig = walk(claim)
    leaves = frozenset(walk_paths(sig, regime='leaf').keys())
    exogenous = frozenset(walk_paths(sig, regime='exogenous').keys())
    return leaves | exogenous


def is_endogenous(name: str, claim: Claim[..., object]) -> bool:
    """Topological endogeneity test (cf. ENDOGENEITY_TOPOLOGY.md).

    Three structural rules:

    1. Leaf of `claim` → exogenous (author chose at design time).
    2. Registered `@measurable` → recurse via `transitive_reads`;
       endogenous iff any base case is itself outside `claim`'s
       leaves.
    3. Otherwise → cell-controlled primitive by elimination
       (trajectory output) → endogenous.

    A measurable closing only over leaves classifies as exogenous
    — this catches the Phase-1 `effective_horizon = 1/(1-γ)`
    loophole (closure was just `{gamma}`, all in leaves). The
    redefinition `1/(1-γ·bf)` adds `bootstrap_fraction` to the
    closure; that measurable reads `done` (a trajectory key, not
    in leaves) → endogenous → bridges sourced through
    `effective_horizon` clear the gate."""
    leaves = _claim_leaves(claim)
    if name in leaves:
        return False
    if name not in registered_names():
        return True
    return any(r not in leaves for r in transitive_reads(name))


if TYPE_CHECKING:
    from corroborate.bridge.bridge import Bridge


CONTRAST_ARM_FIELD = 'contrast_arm'
"""Evaluation-transient column `evaluate()` stamps on the scoped
cells selected by a value-based `DoEffect`. Its only values are the
symbolic analysis identities `baseline` and `treatment`; source
values and their display formatting never become arm identity.
Reserved: evaluation raises if the input already carries it.
Defined here so the contrast-quality gates and `bridge.py` share
one spelling without an import cycle."""


class GateLevel(Enum):
    """Severity tier for an admission-gate result.

    BLOCK fails the bridge with `Verdict.INADMISSIBLE`. WARN and
    INFO let the bridge proceed; their results accumulate on
    `BridgeEvaluation.warnings`."""
    BLOCK = 'block'
    WARN = 'warn'
    INFO = 'info'


@dataclass(frozen=True, slots=True)
class GateResult:
    """Outcome of running one admission gate against one bridge.

    `passed=True`: the gate accepts the bridge (or doesn't apply
    — gates that don't apply return `None` instead, but
    `passed=True` is also valid for "applied + accepted").
    `passed=False`: the level's behavior fires."""
    gate_name: str
    level: GateLevel
    passed: bool
    message: str


@dataclass(frozen=True, slots=True)
class GateContext:
    """Evaluation context handed to every admission gate.

    Constructed by `evaluate()` only; gate authors consume its
    fields, never build or mutate one. New context lands as a new
    field HERE — the gate signature itself never grows again (each
    kwarg added to the old signature broke every custom gate; a
    runtime signature-sniffing shim was the alternative, and both
    are worse than one stable positional parameter).

    `claim` — the substrate's outermost @claim; native endogeneity
    gating keys on its leaf walk.

    `leaves` — the external record's registered configuration
    columns (derive from the record's own config artifacts, e.g.
    `corroborate.data.config_columns` /
    `corroborate_rl.sb3.sb3_config_columns`). The registry is
    authoritative about WHAT WAS CONFIGURED: a value effect must
    source on registered columns, and a registered column that
    moves with the contrast is a confound. It does not attest
    assignment or randomisation — those are unverifiable for
    external records regardless."""
    claim: Claim[..., object] | None = None
    leaves: frozenset[str] | None = None


# Protocol for admission gates. We use Protocol (not Callable) so
# the positional `ctx` parameter is part of the typed contract.
class AdmissionGate(Protocol):
    """Typed callable: a gate runs against (bridge, cells) plus
    the evaluation's `GateContext`, returns a `GateResult` (with
    `passed=True/False` for a fired/silent verdict) or `None`
    when the gate doesn't apply to this bridge."""
    def __call__(
        self,
        bridge: 'Bridge',
        cells: Sequence[Mapping[str, object]],
        ctx: GateContext,
    ) -> GateResult | None: ...


def value_contrast_active(bridge: 'Bridge') -> bool:
    """Whether this bridge carries an explicit value-based DoEffect.

    Arm membership and orientation belong to the declaration —
    never to observed support, and never to the leaf registry
    (which answers a different question: whether the declared
    columns are configuration at all)."""
    return (
        isinstance(bridge.source, DoEffect)
        and bridge.source.is_value_based
    )


# ============ Auto-gates ============


def distinct_arms(
    bridge: 'Bridge',
    cells: Sequence[Mapping[str, object]],
    ctx: GateContext,
) -> GateResult | None:
    """BLOCK: a `DoEffect`-sourced bridge whose arms produce
    duplicate canonical_str fingerprints is structurally
    self-vs-self (binary) or has collapsed levels (N-arm).
    Replaces today's `paired_g` runtime ValueError with a clean
    `Verdict.INADMISSIBLE`."""
    del cells, ctx  # not consulted
    if not isinstance(bridge.source, DoEffect):
        return None
    arm_keys = bridge.source.arm_keys()
    if len(set(arm_keys)) != len(arm_keys):
        return GateResult(
            gate_name='distinct_arms',
            level=GateLevel.BLOCK,
            passed=False,
            message=(
                f'DoEffect arms produce duplicate canonical_str '
                f'fingerprints ({arm_keys!r}); contrast has '
                f'collapsed levels. Rebuild with non-overlapping '
                f'Intervention tuples per arm.'
            ),
        )
    return None


def resolved_source(
    bridge: 'Bridge',
    cells: Sequence[Mapping[str, object]],
    ctx: GateContext,
) -> GateResult | None:
    """BLOCK: Bridge.source string references a column not
    present in the filtered cells. Catches typo'd source names
    (e.g. `'mc_returns'` for `'mc_return'`) at gate time with a
    clear message; orthogonal to `EXOGENOUS_SOURCE`. Structural
    DoEffect sources don't apply, but value effects do name
    external source column(s) and therefore validate them."""
    del ctx
    source = bridge.source
    names: tuple[str, ...]
    if isinstance(source, DoEffect):
        source_names = source.value_source_names
        if source_names is None:
            return None
        names = source_names
    else:
        names = (source if isinstance(source, str) else source.name,)
    if not cells:
        return None  # empty corpus — let downstream surface the issue
    # Iterable[Mapping] inputs may be heterogeneous while a live record
    # grows. Presence in any scoped row is enough; value selection later
    # ignores rows that do not belong to the declared contrast.
    available = frozenset(
        key for cell in cells for key in cell.keys()
    )
    missing = [n for n in names if n not in available]
    if not missing:
        return None
    name = missing[0]
    return GateResult(
        gate_name='resolved_source',
        level=GateLevel.BLOCK,
        passed=False,
        message=(
            f'Bridge {bridge.name!r} sources on column {name!r} '
            f'which is not in the filtered cells. Likely a typo '
            f'or a measurable not materialised in this corpus; '
            f'first 20 available columns across scoped rows: '
            f'{sorted(available)[:20]}.'
        ),
    )


def distinct_units(
    bridge: 'Bridge',
    cells: Sequence[Mapping[str, object]],
    ctx: GateContext,
) -> GateResult | None:
    """WARN (BLOCK below 4): the bridge's source varies at a COARSER
    grain than the cell, so the row count overstates n.

    The failure this catches: a source that is a property of some
    unit above the cell — a bed, a map, an environment — replicated
    across the cells belonging to that unit. Each replicate then
    enters the analysis as an independent observation, and the
    p-value is computed against a sample size the design never had.
    Observed in the wild on first outside use: a bed-level covariate
    with 3 distinct values, replicated over 4 instances each, gave
    rho=+0.913 p=0.0006 from what was really n=3; aggregated to the
    unit it read rho=+0.500 p=0.667.

    The effective n is the number of DISTINCT source values, not the
    number of rows. This gate reports it. It does not aggregate for
    you — the right unit is the author's call, and the fix may be
    either to aggregate or to declare that the replicates really are
    independent (e.g. the source is measured per cell, not inherited).

    DoEffect sources don't apply: an arm indicator is meant to repeat
    across cells, and `pair_by` already carries the design there —
    which covers value-based effects too, since those are DoEffects.
    Bool sources don't apply either: a per-cell binary indicator has
    2 distinct values by construction — value cardinality says
    nothing about the grain it was measured at.

    An ASSOCIATIONAL bridge sourcing on a registered configuration
    leaf still gets this guard: used as a regressor rather than a
    condition label, a k-valued leaf really does bound effective n
    at k."""
    del ctx
    source = bridge.source
    if isinstance(source, DoEffect) or not cells:
        return None
    name = source if isinstance(source, str) else source.name
    if name not in cells[0]:
        return None  # `resolved_source` reports this first
    vals: list[float] = []
    for c in cells:
        v = c.get(name)
        if isinstance(v, bool):
            continue  # binary indicator: cardinality ≠ grain
        if isinstance(v, (int, float)) and v == v:  # finite; NaN != NaN
            vals.append(round(float(v), 12))
    if not vals:
        return None
    n_cells, n_eff = len(vals), len(set(vals))
    if n_eff >= n_cells:
        return None  # every cell its own value: nothing to report
    if n_eff * 2 > n_cells:
        return None  # mild ties (rank data); not replication
    block = n_eff < 4
    return GateResult(
        gate_name='distinct_units',
        level=GateLevel.BLOCK if block else GateLevel.WARN,
        passed=False,
        message=(
            f'Bridge {bridge.name!r} sources on {name!r}, which takes only '
            f'{n_eff} distinct values across {n_cells} cells. The effective '
            f'sample size is {n_eff}, not {n_cells}: the source varies at a '
            f'coarser grain than the cell, so the replicates are not '
            f'independent observations of it. Aggregate to that unit, or '
            f'confirm the source is measured per cell.'
        ),
    )


def exogenous_source(
    bridge: 'Bridge',
    cells: Sequence[Mapping[str, object]],
    ctx: GateContext,
) -> GateResult | None:
    """Adjudicate the claimed source of an interventional bridge.

    BLOCK: native `Tier.INTERVENTIONAL` (Pearl rung-2) bridges
    require an *endogenous* source — a registered measurable
    whose closure transitively touches a non-leaf, OR a
    DoEffect of Claim-shaped Interventions. HP knobs (`gamma`,
    `n_step`, etc.) sourced directly into a causal claim are
    blocked; the substrate must surface the endogenous delegate
    (`effective_horizon`, `q_divergence_score`) and source the
    bridge through it.

    Endogeneity is keyed on `walk_paths(claim, regime='leaf')` for
    native claim-backed sources. For an external value-based
    `DoEffect`, the checkable fact is the leaf registry: the
    declared source column(s) must be REGISTERED CONFIGURATION
    (`ctx.leaves`) — a measured or derived column cannot be the
    assigned parameter of an intervention, and that is the one
    thing the record can prove either way. Without a registry the
    check is unassessable and WARNs (register via
    `corroborate.data.config_columns` or list the columns by
    hand). What no registry can attest — assignment,
    randomisation, hidden confounding — stays author-asserted
    for external effects, registry or not.

    Per ADMISSION_GATES_DESIGN.md § Principle (exogenous vs
    endogenous)."""
    del cells
    # Lazy import to avoid Tier->bridge cycle.
    from corroborate.graph.causal import Tier
    source = bridge.source
    # Graph construction treats every DoEffect as interventional,
    # irrespective of the Bridge.tier default; the value-effect
    # branch therefore sits before the tier guard so omitting
    # `tier=INTERVENTIONAL` cannot suppress the source check.
    if isinstance(source, DoEffect) and source.is_value_based:
        source_names = source.value_source_names
        if source_names is None:  # narrowed by is_value_based
            return None
        if ctx.leaves is None:
            return GateResult(
                gate_name='exogenous_source',
                level=GateLevel.WARN,
                passed=False,
                message=(
                    f'value-based DoEffect on '
                    f'{", ".join(map(repr, source_names))}: no '
                    f'configuration registry was supplied, so the '
                    f'framework could not verify the declared '
                    f'column(s) are configuration rather than '
                    f'measurements. Pass `leaves=` (derive it from '
                    f'the record, e.g. '
                    f'corroborate.data.config_columns).'
                ),
            )
        unregistered = [
            n for n in source_names if n not in ctx.leaves
        ]
        if unregistered:
            return GateResult(
                gate_name='exogenous_source',
                level=GateLevel.BLOCK,
                passed=False,
                message=(
                    f'value-based DoEffect declares '
                    f'{", ".join(map(repr, unregistered))} as '
                    f'assigned parameter(s), but the record does not '
                    f'register them as configuration — a measured or '
                    f'derived column cannot be the assigned parameter '
                    f'of an intervention. Declare a configuration '
                    f'column, or register these in `leaves=` if the '
                    f'producer really configured them.'
                ),
            )
        return None
    if bridge.tier is not Tier.INTERVENTIONAL:
        return None
    if isinstance(source, DoEffect):
        # Every Intervention.replacement must be Claim-shaped
        # (callable). Today the type system enforces this; this
        # gate is belt-and-braces for any future relaxation.
        offenders = [
            iv.slot_path
            for arm in source.arms
            for iv in arm
            if not callable(iv.replacement)
        ]
        if offenders:
            return GateResult(
                gate_name='exogenous_source',
                level=GateLevel.BLOCK,
                passed=False,
                message=(
                    f'Tier.INTERVENTIONAL bridge {bridge.name!r} '
                    f'has non-Claim Intervention(s) on slot(s) '
                    f'{offenders}. Causal claims require an '
                    f'endogenous source — find the delegate '
                    f'(e.g., `effective_horizon` for γ, '
                    f'`q_divergence_score` for sync_period). See '
                    f'ADMISSION_GATES_DESIGN.md § Principle.'
                ),
            )
        return None
    name = source if isinstance(source, str) else source.name
    if ctx.claim is not None:
        # Native composition available: the endogenous-source
        # doctrine applies.
        if is_endogenous(name, ctx.claim):
            return None
        return GateResult(
            gate_name='exogenous_source',
            level=GateLevel.BLOCK,
            passed=False,
            message=(
                f'Tier.INTERVENTIONAL bridge {bridge.name!r} sourced '
                f'on {name!r}, which is a leaf of the outermost '
                f'claim (author-controlled at design time, not '
                f'produced by the cell). Causal claims require an '
                f'endogenous source — find the delegate (e.g., '
                f'`effective_horizon` for γ, `q_divergence_score` '
                f'for sync_period). See ADMISSION_GATES_DESIGN.md § '
                f'Principle.'
            ),
        )
    return GateResult(
        gate_name='exogenous_source',
        level=GateLevel.BLOCK,
        passed=False,
        message=(
            f'Tier.INTERVENTIONAL bridge {bridge.name!r} uses '
            f'source {name!r} without a native claim. '
            f'Observed values cannot establish an intervention or '
            f'determine its orientation. Declare the estimand '
            f'explicitly with '
            f'`DoEffect.from_values(source={name!r}, '
            f'reference=..., treatment=...)`.'
        ),
    )


def exogenous_scope(
    bridge: 'Bridge',
    cells: Sequence[Mapping[str, object]],
    ctx: GateContext,
) -> GateResult | None:
    """WARN: `Bridge.scope` references only exogenous columns
    (leaves of the substrate's outermost claim). The principled
    scope-axis is endogenous; HP envelopes (`_FOURROOMS_REGIME =
    lr == 1e-4`) are a temporary substitute until the substrate
    ships an endogenous predicate.

    Endogeneity is keyed on `walk_paths(claim, regime='leaf')`;
    when `claim` is None, the gate short-circuits. Deliberately
    NOT keyed on external `leaves`: the endogenous-scope doctrine
    is a substrate-authoring discipline — it presumes endogenous
    measurables exist to scope on, which an external record may
    simply not carry (its natural scope axes ARE configuration
    columns like `env_id`)."""
    del cells
    if bridge.scope is None:
        return None
    if ctx.claim is None:
        return None
    # DeferredScope: extract column references from the static_scope
    # half if any; the dynamic part references the stratify column
    # (a regular cell column, not a measurable).
    from corroborate.bridge.deferred_scope import DeferredScope
    if isinstance(bridge.scope, DeferredScope):
        static_expr = bridge.scope.static_scope
        if static_expr is None:
            return None
        referenced = set(static_expr.meta.root_names())
    else:
        referenced = set(bridge.scope.meta.root_names())
    endogenous = {n for n in referenced if is_endogenous(n, ctx.claim)}
    exogenous = referenced - endogenous
    if exogenous and not endogenous:
        return GateResult(
            gate_name='exogenous_scope',
            level=GateLevel.WARN,
            passed=False,
            message=(
                f'Bridge.scope references only exogenous (leaves '
                f"of the outermost claim) columns: "
                f'{sorted(exogenous)!r}. The principled scope-axis '
                f'is endogenous; HP '
                f'envelopes are a temporary substitute.'
            ),
        )
    return None


def no_predicted_direction(
    bridge: 'Bridge',
    cells: Sequence[Mapping[str, object]],
    ctx: GateContext,
) -> GateResult | None:
    """INFO: bridge didn't declare `predicted_direction`. The
    verdict can't distinguish "wrong sign" from "small effect"
    via `verdict_from_paired_stats`; sign-flip refutations are
    silently absorbed as NO_EFFECT. Author-friendly diagnostic;
    not a bug."""
    del cells, ctx
    if bridge.predicted_direction is not None:
        return None
    return GateResult(
        gate_name='no_predicted_direction',
        level=GateLevel.INFO,
        passed=False,
        message=(
            f'Bridge {bridge.name!r} declares no '
            f'`predicted_direction`. Sign-flip refutations '
            f'(observed effect with the wrong sign) cannot be '
            f"distinguished from null effects. Add e.g. "
            f"`predicted_direction='a_lt_b'` if the claim has a "
            f'sign prior; `predicted_direction='
            "'null'` for xfail-style null claims."
        ),
    )


def _normalised(value: object) -> object:
    """NaN reads as missing for constancy checks — two NaNs must
    not count as 'different values' (float NaN != NaN)."""
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def contrast_present(
    bridge: 'Bridge',
    cells: Sequence[Mapping[str, object]],
    ctx: GateContext,
) -> GateResult | None:
    """BLOCK when one of an explicit value effect's arms is absent."""
    del ctx
    if not value_contrast_active(bridge):
        return None
    effect = bridge.source
    if not isinstance(effect, DoEffect):
        return None
    expected = frozenset(effect.arm_keys())
    present = frozenset(
        label
        for cell in cells
        if isinstance((label := cell.get(CONTRAST_ARM_FIELD)), str)
    )
    missing = expected.difference(present)
    if not missing:
        return None
    source_names = effect.value_source_names or ()
    return GateResult(
        gate_name='contrast_present',
        level=GateLevel.BLOCK,
        passed=False,
        message=(
            f'explicit value effect on '
            f'{", ".join(map(repr, source_names))} is missing '
            f'declared arm(s) {sorted(missing)!r} in the scoped '
            f'record. Both '
            f'reference={dict(effect.reference_assignment)!r} and '
            f'treatment={dict(effect.treatment_assignment)!r} are '
            f'required; widen the scope or grow the record.'
        ),
    )


def _scalar_equal(left: object, right: object) -> bool:
    """Best-effort scalar equality for observable config metadata."""
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False


def _same_value_set(left: Sequence[object], right: Sequence[object]) -> bool:
    """Set equality without requiring configuration values hashable."""
    left_unique: list[object] = []
    right_unique: list[object] = []
    for value in left:
        if not any(_scalar_equal(value, seen) for seen in left_unique):
            left_unique.append(value)
    for value in right:
        if not any(_scalar_equal(value, seen) for seen in right_unique):
            right_unique.append(value)
    return (
        len(left_unique) == len(right_unique)
        and all(
            any(_scalar_equal(value, other) for other in right_unique)
            for value in left_unique
        )
    )


def contrast_isolation(
    bridge: 'Bridge',
    cells: Sequence[Mapping[str, object]],
    ctx: GateContext,
) -> GateResult | None:
    """Is anything else moving with the declared contrast?

    Two checks compose, graded by the leaf registry, over exactly
    the cells this claim admits, re-run on every evaluation as the
    record grows:

    1. **Registered balance (BLOCK).** Each configuration leaf in
       `ctx.leaves` must take the same value set on both sides of
       every complete pairing unit. A registered leaf that differs
       inside a pair is a co-varied knob — a certain confound —
       and the verdict is INADMISSIBLE. Per-pair comparison
       catches confounds per-arm constancy misses (a knob varying
       across seeds but differing within every pair). Undeclared
       lockstep partners of a genuine joint intervention also land
       here — the fix is to widen the declaration
       (`DoEffect.from_values(reference={...}, treatment={...})`).
    2. **Unregistered rider scan (WARN).** Columns OUTSIDE the
       registry that are constant within each arm but differ
       across arms — a producer label, or an unlogged knob; only
       the author can say which, so the record shows a warning
       rather than a verdict. Skipped below two cells per arm,
       where every varying column would false-positive.

    Neither check certifies assignment or the absence of hidden
    confounding — those are unverifiable for external records.
    When no registry is supplied the balance half reports itself
    unassessed (INFO); the rider scan still runs, treating every
    non-declared column as unregistered."""
    if not value_contrast_active(bridge):
        return None
    effect = bridge.source
    if not isinstance(effect, DoEffect):
        return None
    declared = frozenset(effect.value_source_names or ())
    # The bridge's target is the outcome under test — a treatment
    # effect on it is the hypothesis, not a confound; it must
    # never be reported as a rider (deterministic effects would
    # trip the constancy scan).
    ignore = {
        CONTRAST_ARM_FIELD, *declared, 'id', *bridge.pair_by,
        bridge.target_name,
    }
    expected = frozenset(effect.arm_keys())
    by_arm: dict[str, list[Mapping[str, object]]] = {}
    by_unit: dict[
        tuple[object, ...], dict[str, list[Mapping[str, object]]],
    ] = {}
    for cell in cells:
        label = cell.get(CONTRAST_ARM_FIELD)
        if isinstance(label, str):
            by_arm.setdefault(label, []).append(cell)
            unit = tuple(cell.get(key) for key in bridge.pair_by)
            by_unit.setdefault(unit, {}).setdefault(label, []).append(cell)
    if not by_arm:
        return None  # nothing matched the declaration: gate is moot

    # ---- rider scan over unregistered columns (WARN) ----
    rider_result: GateResult | None = None
    registered = ctx.leaves if ctx.leaves is not None else frozenset()
    arms = list(by_arm.values())
    if len(by_arm) == len(expected) and all(
        len(rows) >= 2 for rows in arms
    ):
        columns: set[str] = set()
        for rows in arms:
            for cell in rows:
                columns.update(cell.keys())
        riders: list[str] = []
        for column in sorted(columns - ignore - registered):
            try:
                per_arm = [
                    {_normalised(row.get(column)) for row in rows}
                    for rows in arms
                ]
            except TypeError:
                # Unhashable cell values (list-typed trajectory /
                # trace columns) aren't configuration; skip.
                continue
            constant_within = all(len(values) == 1 for values in per_arm)
            if constant_within and len(set().union(*per_arm)) > 1:
                riders.append(column)
        if riders:
            shown = ', '.join(riders[:5])
            more = (
                f' (+{len(riders) - 5} more)' if len(riders) > 5 else ''
            )
            rider_result = GateResult(
                gate_name='contrast_isolation',
                level=GateLevel.WARN,
                passed=False,
                message=(
                    f'unregistered column(s) move with the declared '
                    f'contrast: {shown}{more}. A label is harmless; '
                    f'an unlogged knob is a confound — drop the '
                    f'column, register it as a configuration leaf, '
                    f'or widen the declaration if it was assigned '
                    f'jointly.'
                ),
            )

    # ---- registered balance within pairing units (BLOCK) ----
    if ctx.leaves is None:
        balance_result = GateResult(
            gate_name='contrast_isolation',
            level=GateLevel.INFO,
            passed=False,
            message=(
                'registered-configuration balance unassessed: no '
                '`leaves=` registry was supplied'
            ),
        )
        return rider_result or balance_result
    candidates = sorted(ctx.leaves - ignore)
    if not candidates:
        # The registry exists but holds nothing beyond the declared
        # contrast columns and pairing keys — the balance check ran
        # over an empty candidate set. Vacuously balanced, not
        # unassessed.
        balance_result = GateResult(
            gate_name='contrast_isolation',
            level=GateLevel.INFO,
            passed=True,
            message=(
                'no registered configuration leaves remain to '
                'compare after the declared contrast columns and '
                'pairing keys are excluded'
            ),
        )
        return rider_result or balance_result
    complete_units = [
        (unit, unit_arms)
        for unit, unit_arms in by_unit.items()
        if expected.issubset(unit_arms)
    ]
    if not complete_units:
        balance_result = GateResult(
            gate_name='contrast_isolation',
            level=GateLevel.INFO,
            passed=False,
            message=(
                'registered-configuration balance unassessed: no '
                f'complete pairing units exist for '
                f'pair_by={bridge.pair_by!r}'
            ),
        )
        return rider_result or balance_result

    complete_rows = [
        row
        for _, unit_arms in complete_units
        for rows in unit_arms.values()
        for row in rows
    ]
    absent = [
        column for column in candidates
        if not any(column in row for row in complete_rows)
    ]
    absent_set = frozenset(absent)
    present_candidates = [
        column for column in candidates if column not in absent_set
    ]

    imbalanced: dict[str, list[tuple[object, ...]]] = {}
    baseline_key, treatment_key = effect.arm_keys()
    for unit, unit_arms in complete_units:
        baseline_rows = unit_arms[baseline_key]
        treatment_rows = unit_arms[treatment_key]
        for column in present_candidates:
            baseline_values = [
                _normalised(row.get(column)) for row in baseline_rows
            ]
            treatment_values = [
                _normalised(row.get(column)) for row in treatment_rows
            ]
            if not _same_value_set(baseline_values, treatment_values):
                imbalanced.setdefault(column, []).append(unit)
    if imbalanced:
        shown = ', '.join(
            f'{column} ({len(units)} unit(s))'
            for column, units in list(imbalanced.items())[:5]
        )
        more = (
            f' (+{len(imbalanced) - 5} more)'
            if len(imbalanced) > 5 else ''
        )
        return GateResult(
            gate_name='contrast_isolation',
            level=GateLevel.BLOCK,
            passed=False,
            message=(
                f'registered configuration leaves differ between '
                f'baseline and treatment within pair_by='
                f'{bridge.pair_by!r}: {shown}{more}. A knob moved '
                f'with the contrast — a confound, unless it was '
                f'assigned jointly, in which case widen the '
                f'declaration.'
            ),
        )
    if rider_result is not None:
        return rider_result
    if absent:
        return GateResult(
            gate_name='contrast_isolation',
            level=GateLevel.INFO,
            passed=False,
            message=(
                'registered-configuration balance unassessed for '
                f'missing registered columns: {absent!r}; all '
                'present registered leaves were balanced'
            ),
        )
    return GateResult(
        gate_name='contrast_isolation',
        level=GateLevel.BLOCK,
        passed=True,
        message=(
            f'registered configuration leaves are balanced within '
            f'{len(complete_units)} complete pairing unit(s); no '
            f'unregistered column moves with the contrast'
        ),
    )


def pair_completeness(
    bridge: 'Bridge',
    cells: Sequence[Mapping[str, object]],
    ctx: GateContext,
) -> GateResult | None:
    """WARN: pairing units of a value contrast missing one or more
    conditions. Paired analyses drop incomplete units silently;
    the gate makes the drop visible on the verdict record."""
    del ctx
    if not value_contrast_active(bridge):
        return None
    if not bridge.pair_by:
        return None
    effect = bridge.source
    if not isinstance(effect, DoEffect):
        return None
    expected = frozenset(effect.arm_keys())
    arms_by_unit: dict[tuple[object, ...], set[str]] = {}
    for cell in cells:
        label = cell.get(CONTRAST_ARM_FIELD)
        if not isinstance(label, str):
            continue
        unit = tuple(cell.get(key) for key in bridge.pair_by)
        arms_by_unit.setdefault(unit, set()).add(label)
    if not arms_by_unit:
        return None  # no derived conditions: gate doesn't apply
    incomplete = sum(
        1 for arms in arms_by_unit.values()
        if frozenset(arms) != expected
    )
    if incomplete:
        return GateResult(
            gate_name='pair_completeness',
            level=GateLevel.WARN,
            passed=False,
            message=(
                f'{incomplete} of {len(arms_by_unit)} pairing '
                f'unit(s) missing one or more conditions; paired '
                f'analyses drop them'
            ),
        )
    return None


# Auto-gates run on every bridge before its body. Per-bridge
# `gates=(...)` are appended to this tuple at evaluate-time.
# `resolved_source` runs first so a typo'd source surfaces with
# the column-existence message before `exogenous_source`'s
# leaf-test, which would otherwise classify the absent name as
# endogenous-by-elimination and silently pass. `distinct_units`
# defers to it the same way (returns None on an absent source
# column) so the typo diagnostic wins over the grain diagnostic.
# `contrast_present` sits before `distinct_units` so a
# value-contrast record with no contrast reports "no contrast in
# scope" rather than an effective-n diagnostic; `exogenous_source`
# sits before `distinct_units` for the same reason — a native
# leaf-sourced interventional bridge should hear the structural
# diagnosis ("find the endogenous delegate"), not the effective-n
# symptom.
AUTO_GATES: tuple[AdmissionGate, ...] = (
    distinct_arms,
    resolved_source,
    contrast_present,
    exogenous_source,
    distinct_units,
    exogenous_scope,
    contrast_isolation,
    pair_completeness,
    no_predicted_direction,
)


__all__ = [
    'AUTO_GATES',
    'CONTRAST_ARM_FIELD',
    'AdmissionGate',
    'GateContext',
    'GateLevel',
    'GateResult',
    'contrast_isolation',
    'contrast_present',
    'distinct_arms',
    'distinct_units',
    'exogenous_scope',
    'exogenous_source',
    'is_endogenous',
    'no_predicted_direction',
    'pair_completeness',
    'resolved_source',
    'value_contrast_active',
]
