"""Admission gates — typed callables checked at `evaluate()`-time.

Three severity tiers per `ADMISSION_GATES_DESIGN.md`:

- **BLOCK** — bridge can't produce a meaningful verdict. The
  framework returns `Verdict.INADMISSIBLE` with `blocked_by`
  pointing at the offending gate; the bridge body never runs.
- **WARN** — bridge proceeds, but the verdict is non-principled
  in a known way (HP-envelope scope, missing predicted_direction,
  etc.). Warning surfaces on `BridgeEvaluation.warnings`.
- **INFO** — diagnostic note; not normally surfaced.

A gate is a plain callable: `(bridge, filtered_cells, *, claim) ->
GateResult | None`. Returning None means "this gate doesn't
apply to this bridge"; returning a `GateResult` with
`passed=True` is informational; with `passed=False` triggers
the level's behavior.

The framework's auto-gates (`AUTO_GATES`) are unconditionally
prepended to every bridge's `gates` tuple at `evaluate()`-time.
Per-bridge `gates=(...)` declarations on `@claim_bridge`
add to the auto-gate list.

Endogeneity gating (`exogenous_source`, `exogenous_scope`)
consumes the substrate's claim chain via the `claim` kwarg
threaded by `evaluate()` — `is_endogenous(name, claim)` keys on
`walk_paths(claim, regime='leaf')`. When `claim` is None
(framework-only tests, synthetic-corpus contexts), those gates
short-circuit. See `ENDOGENEITY_TOPOLOGY.md` for the three
structural rules.
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
cells of a value-contrast bridge — the derived condition label
(`'gamma=0.99'`). Reserved: evaluation raises if the input
already carries it. Defined here so the contrast-quality gates
and `bridge.py` share one spelling without an import cycle."""


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


# Protocol for admission gates. `claim` is the substrate's
# outermost @claim threaded through evaluate() — endogeneity
# gates close over its leaf set. `leaves` is the external
# counterpart: the configuration leaves of a record produced
# outside the framework (derived from its resolved-config files,
# e.g. `corroborate.data.config_columns`), for cells that have no
# claim composition to walk. Gates that need neither ignore the
# kwargs. We use Protocol (not Callable) so the kw-only
# parameters are part of the typed contract.
class AdmissionGate(Protocol):
    """Typed callable: a gate runs against (bridge, cells) plus
    the substrate's outermost claim and/or the external record's
    registered configuration leaves, returns a `GateResult` (with
    `passed=True/False` for a fired/silent verdict) or `None`
    when the gate doesn't apply to this bridge."""
    def __call__(
        self,
        bridge: 'Bridge',
        cells: Sequence[Mapping[str, object]],
        *,
        claim: Claim[..., object] | None = None,
        leaves: frozenset[str] | None = None,
    ) -> GateResult | None: ...


def value_contrast_active(
    bridge: 'Bridge',
    *,
    claim: Claim[..., object] | None,
    leaves: frozenset[str] | None,
) -> bool:
    """Whether `evaluate()` derives conditions for this bridge from
    its source column's values — the value-contrast path for
    contrasts executed outside the framework.

    Active exactly when the bridge declares `Tier.INTERVENTIONAL`
    on a plain string source that the DATA side registers as a
    configuration leaf (`leaves`), and no native claim composition
    is present. A native `claim=` wins over `leaves=`: with the
    composition available, the endogenous-source doctrine applies
    and `exogenous_source` adjudicates instead — external records
    get value-contrast semantics precisely because they have no
    composition to hold to that doctrine."""
    from corroborate.graph.causal import Tier
    return (
        bridge.tier is Tier.INTERVENTIONAL
        and isinstance(bridge.source, str)
        and claim is None
        and leaves is not None
        and bridge.source in leaves
    )


# ============ Auto-gates ============


def distinct_arms(
    bridge: 'Bridge',
    cells: Sequence[Mapping[str, object]],
    *,
    claim: Claim[..., object] | None = None,
    leaves: frozenset[str] | None = None,
) -> GateResult | None:
    """BLOCK: a `DoEffect`-sourced bridge whose arms produce
    duplicate canonical_str fingerprints is structurally
    self-vs-self (binary) or has collapsed levels (N-arm).
    Replaces today's `paired_g` runtime ValueError with a clean
    `Verdict.INADMISSIBLE`."""
    del cells, claim, leaves  # not consulted
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
    *,
    claim: Claim[..., object] | None = None,
    leaves: frozenset[str] | None = None,
) -> GateResult | None:
    """BLOCK: Bridge.source string references a column not
    present in the filtered cells. Catches typo'd source names
    (e.g. `'mc_returns'` for `'mc_return'`) at gate time with a
    clear message; orthogonal to `EXOGENOUS_SOURCE`. DoEffect
    sources don't apply (no string column to validate)."""
    del claim, leaves
    source = bridge.source
    if isinstance(source, DoEffect):
        return None
    name = source if isinstance(source, str) else source.name
    if not cells:
        return None  # empty corpus — let downstream surface the issue
    available = frozenset(cells[0].keys())
    if name in available:
        return None
    return GateResult(
        gate_name='resolved_source',
        level=GateLevel.BLOCK,
        passed=False,
        message=(
            f'Bridge {bridge.name!r} sources on column {name!r} '
            f'which is not in the filtered cells. Likely a typo '
            f'or a measurable not materialised in this corpus; '
            f'first 20 available columns: '
            f'{sorted(available)[:20]}.'
        ),
    )


def distinct_units(
    bridge: 'Bridge',
    cells: Sequence[Mapping[str, object]],
    *,
    claim: Claim[..., object] | None = None,
    leaves: frozenset[str] | None = None,
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
    across cells, and `pair_by` already carries the design there. A
    value contrast doesn't apply for the same reason — its source is
    the condition indicator of an externally-executed contrast, and
    `pair_by` plus the contrast-quality gates carry the design. Bool
    sources don't apply either: a per-cell binary indicator has
    2 distinct values by construction — value cardinality says
    nothing about the grain it was measured at.

    An ASSOCIATIONAL bridge sourcing on a registered configuration
    leaf still gets this guard: used as a regressor rather than a
    condition label, a k-valued leaf really does bound effective n
    at k."""
    source = bridge.source
    if isinstance(source, DoEffect) or not cells:
        return None
    if value_contrast_active(bridge, claim=claim, leaves=leaves):
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
    *,
    claim: Claim[..., object] | None = None,
    leaves: frozenset[str] | None = None,
) -> GateResult | None:
    """BLOCK: `Tier.INTERVENTIONAL` (Pearl rung-2) bridges
    require an *endogenous* source — a registered measurable
    whose closure transitively touches a non-leaf, OR a
    DoEffect of Claim-shaped Interventions. HP knobs (`gamma`,
    `n_step`, etc.) sourced directly into a causal claim are
    blocked; the substrate must surface the endogenous delegate
    (`effective_horizon`, `q_divergence_score`) and source the
    bridge through it.

    Endogeneity is keyed on `walk_paths(claim, regime='leaf')`;
    when `claim` is None the gate falls back to the external
    record's registered configuration leaves (`leaves`): an
    interventional bridge there must source on a registered leaf
    — the assigned parameter of the externally-executed contrast
    — and sourcing on anything else (a measured or derived
    column) is blocked, because a measurement cannot be the
    assigned parameter of an intervention. With neither registry
    the gate short-circuits — substrates that want the rule
    enforced thread `claim=` (or `leaves=`) through `evaluate()`.

    Per ADMISSION_GATES_DESIGN.md § Principle (exogenous vs
    endogenous)."""
    del cells
    # Lazy import to avoid Tier->bridge cycle.
    from corroborate.graph.causal import Tier
    if bridge.tier is not Tier.INTERVENTIONAL:
        return None
    source = bridge.source
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
    if claim is not None:
        # Native composition available: the endogenous-source
        # doctrine applies.
        if is_endogenous(name, claim):
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
    if leaves is not None:
        # External record: the assigned parameter of an
        # externally-executed contrast must be a registered
        # configuration leaf.
        if name in leaves:
            return None
        return GateResult(
            gate_name='exogenous_source',
            level=GateLevel.BLOCK,
            passed=False,
            message=(
                f'Tier.INTERVENTIONAL bridge {bridge.name!r} sourced '
                f'on {name!r}, which is not a registered '
                f'configuration leaf of this record — a measured or '
                f'derived column cannot be the assigned parameter of '
                f'an intervention. Source on a configuration column, '
                f'or register {name!r} in `leaves=` if the producer '
                f'really configured it.'
            ),
        )
    return None  # gate doesn't apply without either registry


def exogenous_scope(
    bridge: 'Bridge',
    cells: Sequence[Mapping[str, object]],
    *,
    claim: Claim[..., object] | None = None,
    leaves: frozenset[str] | None = None,
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
    del cells, leaves
    if bridge.scope is None:
        return None
    if claim is None:
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
    endogenous = {n for n in referenced if is_endogenous(n, claim)}
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
    *,
    claim: Claim[..., object] | None = None,
    leaves: frozenset[str] | None = None,
) -> GateResult | None:
    """INFO: bridge didn't declare `predicted_direction`. The
    verdict can't distinguish "wrong sign" from "small effect"
    via `verdict_from_paired_stats`; sign-flip refutations are
    silently absorbed as NO_EFFECT. Author-friendly diagnostic;
    not a bug."""
    del cells, claim, leaves
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
    *,
    claim: Claim[..., object] | None = None,
    leaves: frozenset[str] | None = None,
) -> GateResult | None:
    """BLOCK: a value-contrast bridge whose scoped cells carry
    fewer than two distinct values of the source — there is no
    contrast in this record to test. The claim states which
    parameter was contrasted; the record must actually vary it
    within scope, or the comparison is undefined."""
    if not value_contrast_active(bridge, claim=claim, leaves=leaves):
        return None
    values: set[object] = set()
    for cell in cells:
        value = _normalised(cell.get(bridge.source_name))
        if value is not None:
            values.add(value)
    if len(values) >= 2:
        return None
    return GateResult(
        gate_name='contrast_present',
        level=GateLevel.BLOCK,
        passed=False,
        message=(
            f'source {bridge.source_name!r} takes {len(values)} '
            f'value(s) across the scoped cells — no contrast is '
            f'present in this record. An interventional claim on a '
            f'configuration leaf needs at least two of its values '
            f'in scope (widen the scope, or grow the record).'
        ),
    )


def contrast_isolation(
    bridge: 'Bridge',
    cells: Sequence[Mapping[str, object]],
    *,
    claim: Claim[..., object] | None = None,
    leaves: frozenset[str] | None = None,
) -> GateResult | None:
    """A value-contrast whose scoped cells differ in some OTHER
    column that is constant within each condition — the signature
    of something that changed together with the claimed
    parameter. Checked over exactly the cells this claim admits,
    at every evaluation, so a confound entering the growing
    record flips the verdict instead of passing silently. Quality
    control lives here — per claim, per extent, on the verdict
    record — rather than at a data-loading door, because "the
    contrast is isolated" is a property of the cells a claim
    admits, not of a file format.

    Severity is decided by the leaf registry: a rider that IS a
    registered configuration leaf is a co-varied knob — a certain
    confound — and BLOCKs; a rider outside the registry (a
    producer label column, an unregistered field) WARNs, because
    only the author can say whether it is a knob or a name.
    Downgraded to WARN entirely when any condition has fewer than
    two cells — at n=1 per condition nothing distinguishes the
    contrast from any co-varying column."""
    del claim
    by_arm: dict[str, list[Mapping[str, object]]] = {}
    for cell in cells:
        label = cell.get(CONTRAST_ARM_FIELD)
        if isinstance(label, str):
            by_arm.setdefault(label, []).append(cell)
    if not by_arm:
        return None  # no derived conditions: gate doesn't apply
    if len(by_arm) < 2 or any(len(rows) < 2 for rows in by_arm.values()):
        return GateResult(
            gate_name='contrast_isolation',
            level=GateLevel.WARN,
            passed=False,
            message=(
                'contrast isolation unverifiable: fewer than two '
                'cells per condition — nothing distinguishes the '
                'contrast from a co-varying column at n=1'
            ),
        )
    ignore = {
        CONTRAST_ARM_FIELD, bridge.source_name, 'id', *bridge.pair_by,
    }
    arms = list(by_arm.values())
    columns: set[str] = set()
    for rows in arms:
        for cell in rows:
            columns.update(cell.keys())
    riders: list[str] = []
    for column in sorted(columns.difference(ignore)):
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
        registered = leaves if leaves is not None else frozenset()
        knob_riders = [r for r in riders if r in registered]
        # Registry unknown → conservative: every rider blocks.
        blocking = knob_riders if leaves is not None else riders
        warning_only = [r for r in riders if r not in set(blocking)]
        if blocking:
            shown = ', '.join(blocking[:5])
            more = (
                f' (+{len(blocking) - 5} more)'
                if len(blocking) > 5 else ''
            )
            return GateResult(
                gate_name='contrast_isolation',
                level=GateLevel.BLOCK,
                passed=False,
                message=(
                    f'configuration leaf/leaves constant within each '
                    f'condition but different across them — a '
                    f'confound rides the {bridge.source_name!r} '
                    f'contrast: {shown}{more}'
                ),
            )
        shown = ', '.join(warning_only[:5])
        more = (
            f' (+{len(warning_only) - 5} more)'
            if len(warning_only) > 5 else ''
        )
        return GateResult(
            gate_name='contrast_isolation',
            level=GateLevel.WARN,
            passed=False,
            message=(
                f'unregistered column(s) move with the '
                f'{bridge.source_name!r} contrast: {shown}{more}. '
                f'A label is harmless; an unregistered knob is a '
                f'confound — drop the column or register it as a '
                f'configuration leaf.'
            ),
        )
    return GateResult(
        gate_name='contrast_isolation',
        level=GateLevel.BLOCK,
        passed=True,
        message='scoped cells differ only in the contrast parameter',
    )


def pair_completeness(
    bridge: 'Bridge',
    cells: Sequence[Mapping[str, object]],
    *,
    claim: Claim[..., object] | None = None,
    leaves: frozenset[str] | None = None,
) -> GateResult | None:
    """WARN: pairing units of a value contrast missing one or more
    conditions. Paired analyses drop incomplete units silently;
    the gate makes the drop visible on the verdict record."""
    del claim, leaves
    if not bridge.pair_by:
        return None
    all_labels: set[str] = set()
    arms_by_unit: dict[tuple[object, ...], set[str]] = {}
    for cell in cells:
        label = cell.get(CONTRAST_ARM_FIELD)
        if not isinstance(label, str):
            continue
        all_labels.add(label)
        unit = tuple(cell.get(key) for key in bridge.pair_by)
        arms_by_unit.setdefault(unit, set()).add(label)
    if not arms_by_unit:
        return None  # no derived conditions: gate doesn't apply
    incomplete = sum(
        1 for arms in arms_by_unit.values() if arms != all_labels
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
