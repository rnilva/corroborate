"""Admission gates — typed callables checked at `evaluate()`-time.

Three severity tiers per `ADMISSION_GATES_DESIGN.md`:

- **BLOCK** — bridge can't produce a meaningful verdict. The
  framework returns `Verdict.INADMISSIBLE` with `blocked_by`
  pointing at the offending gate; the bridge body never runs.
- **WARN** — bridge proceeds, but the verdict is non-principled
  in a known way (HP-envelope scope, missing predicted_direction,
  etc.). Warning surfaces on `BridgeEvaluation.warnings`.
- **INFO** — diagnostic note; not normally surfaced.

A gate is a plain callable: `(bridge, filtered_cells) ->
GateResult | None`. Returning None means "this gate doesn't
apply to this bridge"; returning a `GateResult` with
`passed=True` is informational; with `passed=False` triggers
the level's behavior.

The framework's auto-gates (`AUTO_GATES`) are unconditionally
prepended to every bridge's `gates` tuple at `evaluate()`-time.
Per-bridge `gates=(...)` declarations on `@claim_bridge`
add to the auto-gate list.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from corroborate.core.intervention import DoEffect
from corroborate.measurables import registered_names


def _endogenous_pool() -> frozenset[str]:
    """Substrate-extensible endogenous frontier: registered
    measurables ∪ framework-controlled metadata. Tuple →
    frozenset so the gate's set-test stays O(1) amortised."""
    return frozenset(registered_names()) | _STANDARD_METADATA

if TYPE_CHECKING:
    from corroborate.bridge.bridge import Bridge


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


# Plain-callable Protocol for admission gates. A gate is just a
# function; tuple-of-gates composes via `+`. No combinator class.
type AdmissionGate = Callable[
    ['Bridge', Sequence[Mapping[str, object]]],
    GateResult | None,
]


# Framework-controlled provenance / metadata columns shared
# across substrates. Always counted as endogenous for
# scope/source admission. Substrates extend the endogenous
# frontier by registering more `@measurable` columns.
_STANDARD_METADATA: frozenset[str] = frozenset({
    'env_name',
    'seed',
    'id',
    'arm_key',
    'corpus',
    'cycle_id',
    'parent_id',
    'timestamp',
    'verdict',
    'intervention_name',  # legacy column kept for back-compat
    'parent_cycle_id',
})


# ============ Auto-gates ============


def distinct_arms(
    bridge: 'Bridge',
    cells: Sequence[Mapping[str, object]],
) -> GateResult | None:
    """BLOCK: a `DoEffect`-sourced bridge whose treatment and
    baseline arms produce identical canonical_str is structurally
    self-vs-self. Replaces today's `paired_g` runtime ValueError
    with a clean Verdict.INADMISSIBLE."""
    del cells  # not consulted
    if not isinstance(bridge.source, DoEffect):
        return None
    if bridge.source.treatment_arm_key() == bridge.source.baseline_arm_key():
        return GateResult(
            gate_name='distinct_arms',
            level=GateLevel.BLOCK,
            passed=False,
            message=(
                f'DoEffect treatment and baseline arms produce '
                f'identical canonical_str '
                f'({bridge.source.treatment_arm_key()!r}). The '
                f'contrast is self-vs-self; rebuild with a '
                f'non-empty treatment or baseline tuple.'
            ),
        )
    return None


def exogenous_source(
    bridge: 'Bridge',
    cells: Sequence[Mapping[str, object]],
) -> GateResult | None:
    """BLOCK: `Tier.INTERVENTIONAL` (Pearl rung-2) bridges
    require an *endogenous* source — a registered measurable or
    standard metadata, OR a DoEffect of Claim-shaped
    Interventions. HP knobs (`gamma`, `n_step`, etc.) sourced
    directly into a causal claim are blocked; the substrate
    must surface the endogenous delegate (`effective_horizon`,
    `q_divergence_score`) and source the bridge through it.

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
            for iv in (*source.treatment, *source.baseline)
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
    # str / Measurable source: must reference a registered
    # measurable or standard metadata.
    name = source if isinstance(source, str) else source.name
    if name in _endogenous_pool():
        return None
    return GateResult(
        gate_name='exogenous_source',
        level=GateLevel.BLOCK,
        passed=False,
        message=(
            f'Tier.INTERVENTIONAL bridge {bridge.name!r} sourced '
            f'on {name!r} which is not a registered measurable '
            f'nor standard metadata. Causal claims require an '
            f'endogenous source — find the delegate. See '
            f'ADMISSION_GATES_DESIGN.md § Principle.'
        ),
    )


def exogenous_scope(
    bridge: 'Bridge',
    cells: Sequence[Mapping[str, object]],
) -> GateResult | None:
    """WARN: `Bridge.scope` references only exogenous (HP-leaf)
    columns. The principled scope-axis is endogenous; HP envelopes
    (`_FOURROOMS_REGIME = lr == 1e-4`) are a temporary substitute
    until the substrate ships an endogenous predicate.

    Detection is free: walks the polars expression's referenced
    column names and cross-references against the substrate's
    registered measurables + standard metadata."""
    del cells
    if bridge.scope is None:
        return None
    referenced = set(bridge.scope.meta.root_names())
    pool = _endogenous_pool()
    endogenous = referenced & pool
    exogenous = referenced - pool
    if exogenous and not endogenous:
        return GateResult(
            gate_name='exogenous_scope',
            level=GateLevel.WARN,
            passed=False,
            message=(
                f'Bridge.scope references only exogenous (HP-leaf) '
                f'columns: {sorted(exogenous)!r}. The principled '
                f'scope-axis is endogenous (cf. ANALYSIS_RECIPE.md '
                f'§0); HP envelopes are a temporary substitute. '
                f'See FUTURE_WORKS "Endogenous-variable scope '
                f'predicates".'
            ),
        )
    return None


def no_predicted_direction(
    bridge: 'Bridge',
    cells: Sequence[Mapping[str, object]],
) -> GateResult | None:
    """INFO: bridge didn't declare `predicted_direction`. The
    verdict can't distinguish "wrong sign" from "small effect"
    via `verdict_from_paired_stats`; sign-flip refutations are
    silently absorbed as NO_EFFECT. Author-friendly diagnostic;
    not a bug."""
    del cells
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


# Auto-gates run on every bridge before its body. Per-bridge
# `gates=(...)` are appended to this tuple at evaluate-time.
AUTO_GATES: tuple[AdmissionGate, ...] = (
    distinct_arms,
    exogenous_source,
    exogenous_scope,
    no_predicted_direction,
)


__all__ = [
    'AUTO_GATES',
    'AdmissionGate',
    'GateLevel',
    'GateResult',
    'distinct_arms',
    'exogenous_scope',
    'exogenous_source',
    'no_predicted_direction',
]
