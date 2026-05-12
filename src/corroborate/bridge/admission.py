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
# gates close over its leaf set; gates that don't need it
# ignore the kwarg. We use Protocol (not Callable) so the
# kw-only `claim` parameter is part of the typed contract.
class AdmissionGate(Protocol):
    """Typed callable: a gate runs against (bridge, cells) plus
    the substrate's outermost claim, returns a `GateResult` (with
    `passed=True/False` for a fired/silent verdict) or `None`
    when the gate doesn't apply to this bridge."""
    def __call__(
        self,
        bridge: 'Bridge',
        cells: Sequence[Mapping[str, object]],
        *,
        claim: Claim[..., object] | None = None,
    ) -> GateResult | None: ...


# ============ Auto-gates ============


def distinct_arms(
    bridge: 'Bridge',
    cells: Sequence[Mapping[str, object]],
    *,
    claim: Claim[..., object] | None = None,
) -> GateResult | None:
    """BLOCK: a `DoEffect`-sourced bridge whose arms produce
    duplicate canonical_str fingerprints is structurally
    self-vs-self (binary) or has collapsed levels (N-arm).
    Replaces today's `paired_g` runtime ValueError with a clean
    `Verdict.INADMISSIBLE`."""
    del cells, claim  # not consulted
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
) -> GateResult | None:
    """BLOCK: Bridge.source string references a column not
    present in the filtered cells. Catches typo'd source names
    (e.g. `'mc_returns'` for `'mc_return'`) at gate time with a
    clear message; orthogonal to `EXOGENOUS_SOURCE`. DoEffect
    sources don't apply (no string column to validate)."""
    del claim
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


def exogenous_source(
    bridge: 'Bridge',
    cells: Sequence[Mapping[str, object]],
    *,
    claim: Claim[..., object] | None = None,
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
    when `claim` is None (framework-only tests, synthetic
    contexts), the gate short-circuits — substrates that want
    the rule enforced thread `claim=` through `evaluate()`.

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
    # str / Measurable source: needs claim to test endogeneity.
    if claim is None:
        return None  # gate doesn't apply without substrate context
    name = source if isinstance(source, str) else source.name
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


def exogenous_scope(
    bridge: 'Bridge',
    cells: Sequence[Mapping[str, object]],
    *,
    claim: Claim[..., object] | None = None,
) -> GateResult | None:
    """WARN: `Bridge.scope` references only exogenous columns
    (leaves of the substrate's outermost claim). The principled
    scope-axis is endogenous; HP envelopes (`_FOURROOMS_REGIME =
    lr == 1e-4`) are a temporary substitute until the substrate
    ships an endogenous predicate.

    Endogeneity is keyed on `walk_paths(claim, regime='leaf')`;
    when `claim` is None, the gate short-circuits."""
    del cells
    if bridge.scope is None:
        return None
    if claim is None:
        return None
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
                f'is endogenous (cf. ANALYSIS_RECIPE.md §0); HP '
                f'envelopes are a temporary substitute. See '
                f'FUTURE_WORKS "Endogenous-variable scope '
                f'predicates".'
            ),
        )
    return None


def no_predicted_direction(
    bridge: 'Bridge',
    cells: Sequence[Mapping[str, object]],
    *,
    claim: Claim[..., object] | None = None,
) -> GateResult | None:
    """INFO: bridge didn't declare `predicted_direction`. The
    verdict can't distinguish "wrong sign" from "small effect"
    via `verdict_from_paired_stats`; sign-flip refutations are
    silently absorbed as NO_EFFECT. Author-friendly diagnostic;
    not a bug."""
    del cells, claim
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
# `resolved_source` runs first so a typo'd source surfaces with
# the column-existence message before `exogenous_source`'s
# leaf-test, which would otherwise classify the absent name as
# endogenous-by-elimination and silently pass.
AUTO_GATES: tuple[AdmissionGate, ...] = (
    distinct_arms,
    resolved_source,
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
    'is_endogenous',
    'no_predicted_direction',
    'resolved_source',
]
