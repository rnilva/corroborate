"""Intervention — typed structural delta to a theory.

A Hypothesis describes a CHANGE to the theory; Interventions are
the typed primitives describing each individual swap. The empty
tuple is the baseline arm; a non-empty tuple is a treatment arm
whose identity is derived from the canonical fingerprint of the
swaps.

Intervention vs HP variation. HPs (γ, lr, batch_size, total_steps)
MUST NOT appear as Interventions — they are cell covariates that
meta-regression cleaves on (cf. v9's aggregation reframing). Slot
swaps (`bootstrap`, `replay.sample`, `action_select`) ARE
Interventions — they describe the structural mechanism delta.

Two hypotheses with the same `intervention_arms` but different HP
grid points share an arm key and pair as same-arm cells; the HP
difference becomes a covariate, not an arm distinguisher.

Identity is via static fingerprint (`canonical_str`). The runtime
graph-signature delta check (`signature(g_treatment) -
signature(g_baseline)`) — the principled HPO-smuggle gate — is
deferred (`FUTURE_WORKS.md` line 41). A static fingerprint cannot
distinguish "scalar-only diff to a partial's keywords" (HP smuggle
that ought to be rejected) from "structural replacement swap"
(real intervention). Both produce different `arm_key()` values;
the runtime check is what would reject the former when the
dialectic loop forces it."""
from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeIs

from corroborate._internals.canonical import canonical_str
from corroborate.core.claim import FnClaim


class ArmRole(StrEnum):
    """Typed sentinel for "which side of the DoEffect contrast" —
    bridge authors use this in place of literal arm_key strings,
    so the bridge body stays decoupled from the persistence-side
    canonical fingerprint of the unmodified or treatment arm.

    The runner resolves an `ArmRole` value to the actual arm_key
    string via `DoEffect.role_arm_key(role)` before forwarding to
    analyses; analyses themselves see only resolved strings."""
    BASELINE = 'baseline'
    TREATMENT = 'treatment'

type Replacement = (
    FnClaim[..., object]
    | functools.partial[object]
    | Callable[..., object]
)
"""Runtime universe of slot replacements: a FnClaim free-function
wrapper, a `functools.partial` binding, or a plain callable
(covers config-bundle instances and class-based Claims using the
`record_call` escape hatch — they're callable). Each canonicalises
distinctly via `canonical_str` so two structurally-equal
replacements produce the same fingerprint across processes."""


def is_replacement(v: object) -> TypeIs[Replacement]:
    """Narrow `v` to `Replacement` when it's callable. Every union
    arm is callable (FnClaim wrappers, partials, plain
    callables / config bundles); a runtime `callable()` check is
    therefore sufficient and the TypeIs is honest about the
    narrowing."""
    return callable(v)


@dataclass(frozen=True, slots=True)
class Intervention:
    """A typed structural delta: at `slot_path`, swap the parent
    claim for `replacement`.

    `slot_path` is a dotted topology path (`bootstrap`,
    `replay.sample`, `action_select`) addressing a kwarg slot of
    the bound theory. `replacement` is the substitute Claim,
    callable, or `functools.partial`.

    Two Interventions with the same `slot_path` and a
    canonically-equal `replacement` produce the same `arm_key()`.
    HP variation does NOT appear at this layer."""
    slot_path: str
    replacement: Replacement

    def arm_key(self) -> str:
        return f'{self.slot_path}={canonical_str(self.replacement)}'

    def apply[T](
        self, base: Callable[..., T],
    ) -> functools.partial[T]:
        """Apply this do() to a base composition.

        Pearl: `do(slot_path = replacement)` on the SCM `base`.
        Returns the post-intervention SCM with `slot_path` pinned
        to `replacement` and every other mechanism preserved.
        Implemented as `functools.partial(base, slot_path=
        replacement)` — single-slot surgical replacement, all
        other slots fall through to base's defaults / outer
        partial bindings.

        Nested `slot_path` (containing `.`) is NOT supported —
        substituting at depth would require knowing the parent
        claim's structure to reconstruct it. Authors with nested
        intent construct the parent claim explicitly with the
        substituted child:

            # Wrong (not supported):
            Intervention(slot_path='replay.sample', replacement=X)
            # Right:
            Intervention(slot_path='replay',
                         replacement=Replay(sample=X))

        The framework's config bundles are frozen dataclasses;
        constructing the substituted parent is the substrate
        author's idiom."""
        if '.' in self.slot_path:
            raise ValueError(
                f'nested slot_path {self.slot_path!r} not supported '
                f'by Intervention.apply(); construct the parent '
                f'claim with the nested override explicitly.',
            )
        return functools.partial(
            base, **{self.slot_path: self.replacement},
        )


def apply_interventions[T](
    base: Callable[..., T],
    interventions: Sequence[Intervention],
) -> Callable[..., T]:
    """Sequentially apply a tuple of `do()`s to a base composition.

    Pearl: `do()`s on disjoint variables commute; on overlapping
    variables, the later one shadows the earlier (last-wins via
    `functools.partial`'s kwarg merge). Order is preserved so the
    overlapping case is deterministic, but callers should design
    intervention tuples with disjoint `slot_path`s — overlapping
    interventions on the same slot are a code smell."""
    composed: Callable[..., T] = base
    for iv in interventions:
        composed = iv.apply(composed)
    return composed


def combined_arm_key(interventions: tuple[Intervention, ...]) -> str:
    """Stable arm fingerprint for a tuple of Interventions.

    Sorted by slot_path so order-of-construction does not change
    identity. Empty tuple → `'baseline'`; non-empty → `'+'`-joined
    arm keys."""
    if not interventions:
        return 'baseline'
    return '+'.join(sorted(i.arm_key() for i in interventions))


@dataclass(frozen=True, slots=True)
class DoEffect:
    """The typed contrast carried on a Bridge: two arms expressed
    as Intervention tuples on the claim graph.

    Pearl-rung-2 edges in the causal graph have an *intervention*
    as the source node, NOT a measurable. `DoEffect` carries the
    treatment / baseline as `tuple[Intervention, ...]` — typed
    structural deltas on the claim graph. Arm identity for cell
    matching derives from `combined_arm_key` (canonical_str of
    each tuple), accessible via the `treatment_arm_key()` /
    `baseline_arm_key()` methods.

    The empty tuple is the baseline arm by convention (no
    structural deltas applied = the substrate's vanilla
    composition). A non-empty `baseline` is a treatment-vs-treatment
    contrast (e.g. DDQN-3step vs DDQN-1step).

    The `Intervention` tuple shape unifies with
    `Hypothesis.intervention_arms` — DoEffect IS the
    contrast specification; what was previously two type-erased
    arm-name strings is now structurally connected to the typed
    claim graph."""
    treatment: tuple[Intervention, ...]
    baseline: tuple[Intervention, ...] = ()

    def treatment_arm_key(self) -> str:
        """Canonical fingerprint of the treatment arm. Empty tuple
        → `'baseline'`; non-empty → `'+'`-joined slot=replacement
        keys."""
        return combined_arm_key(self.treatment)

    def baseline_arm_key(self) -> str:
        """Canonical fingerprint of the baseline arm. Empty tuple
        → `'baseline'`; non-empty → `'+'`-joined slot=replacement
        keys."""
        return combined_arm_key(self.baseline)

    def role_arm_key(self, role: ArmRole) -> str:
        """Resolve an ArmRole sentinel to the canonical arm_key
        string for that side of the contrast. Lets bridge authors
        write typed `ArmRole.BASELINE` / `ArmRole.TREATMENT`
        instead of stringifying the persistence fingerprint."""
        if role is ArmRole.BASELINE:
            return self.baseline_arm_key()
        return self.treatment_arm_key()

    def node_key(self) -> str:
        """Canonical string identity for the do-node in
        `CausalGraph`. `'do(treatment|vs=baseline)'` form makes
        the contrast explicit when the graph is rendered."""
        return f'do({self.treatment_arm_key()}|vs={self.baseline_arm_key()})'
