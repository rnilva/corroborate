"""Intervention — typed structural delta to a theory.

A Hypothesis describes CHANGES to the theory; `Intervention`s are
the typed primitives describing each individual slot swap. A
`DoEffect` has two declaration forms:

1. Structural effects carry N arms, each a tuple of `Intervention`s —
   Pearl's `do(·)` operator for that arm. Binary contrast is the
   special case N=2, often with one empty-tuple "no intervention" arm.
2. Value effects declare an exact binary contrast over an external
   measurable, via `DoEffect.from_values(...)`. They identify rows as
   baseline or treatment without inferring either arm from the observed
   data. This declaration fixes the estimand; it does not by itself
   attest that the external values were experimentally assigned.

Pearl-coherence: each arm in `DoEffect.arms` is a joint
intervention `do(slot_1 = v_1, slot_2 = v_2, ...)`. Multi-level
sweeps (dose-response), factorial designs, and N-way contrasts
are first-class — the framework dispatches one cell per arm; the
analysis-side primitives (per-pair `paired_g`, dose-response,
`factorial_2x2`) pick the reference structure separately.

Intervention vs HP variation. HPs (γ, lr, batch_size, total_steps)
MUST NOT appear as Interventions — they are cell covariates that
meta-regression cleaves on (cf. v9's aggregation reframing). Slot
swaps (`bootstrap`, `replay.sample`, `action_select`) ARE
Interventions — they describe the structural mechanism delta.

Two hypotheses with the same arm contents but different HP grid
points share an arm key and pair as same-arm cells; the HP
difference becomes a covariate, not an arm distinguisher.

Identity is via static fingerprint (`canonical_str`). The runtime
graph-signature delta check (per-arm `signature(g_arm_i) -
signature(g_arm_j)`) — the principled HPO-smuggle gate — is
deferred (`FUTURE_WORKS.md` line 41). A static fingerprint cannot
distinguish "scalar-only diff to a partial's keywords" (HP smuggle
that ought to be rejected) from "structural replacement swap"
(real intervention). Both produce different `arm_key()` values;
the runtime check is what would reject the former when the
dialectic loop forces it."""
from __future__ import annotations

import functools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Self, TypeIs, override

from corroborate._internals.canonical import canonical_str


class ArmRole(StrEnum):
    """Typed sentinel for "which side of a BINARY DoEffect
    contrast" — bridge authors use this in place of literal
    arm_key strings, so the bridge body stays decoupled from the
    persistence-side canonical fingerprint of the unmodified or
    treatment arm.

    Binary-only by design. When `DoEffect` carries N > 2 arms,
    bridges use `arm_keys[i]` indexing (or explicit string keys)
    instead — the `BASELINE` / `TREATMENT` semantics don't extend
    naturally to multi-level interventions. The bridge dispatcher
    raises if an `ArmRole` sentinel appears in a non-binary
    bridge's params.

    Resolution lives in `bridge.py`'s evaluate path: when a
    bridge's source is a binary DoEffect, `ArmRole.TREATMENT`
    resolves to `arm_keys[1]` and `ArmRole.BASELINE` to
    `arm_keys[0]`."""
    BASELINE = 'baseline'
    TREATMENT = 'treatment'


type AssignedValue = str | int | float | bool
"""Scalar value admissible in an external value-based `DoEffect`.

The deliberately small union matches scalar dataframe condition
columns while keeping equality and canonical rendering unambiguous.
"""


type Replacement = object
"""Runtime universe of slot replacements: any value the substrate
admits at a `slot_path`. Two shapes are common:

1. **Callable slot swaps** — `FnClaim` wrappers, `functools.partial`
   bindings, plain callables (covers class-based Claims via the
   `record_call` escape hatch). The canonical case the framework
   was originally designed for; `bootstrap`, `action_select`,
   `replay.sample` all live here.
2. **Config-bundle slot swaps** — frozen-dataclass instances
   (`Replay(capacity=5000)`, `MLP(hidden=[64])`, etc.). The slot
   value carries construction-time HPs + slot-Claim refs, all in
   one immutable record.

`canonical_str` handles both — two structurally-equal
replacements produce the same fingerprint, so arm identity is
preserved across processes. The `Replacement` alias is `object`
because the framework's substrate-agnostic shape can't enforce
"this slot accepts a callable" without knowing the substrate's
type contract; the substrate's own type-checker catches slot/
replacement mismatches at the call site."""


def is_replacement(v: object) -> TypeIs[Replacement]:
    """Narrow `v` to `Replacement`. With `Replacement = object`
    this is trivially True — kept as a typed gate so YAML
    parsers and other ingest sites have a single named predicate
    to call rather than dropping the check."""
    del v
    return True


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
        constructing the substituted parent is the implementation
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


def _is_assigned_value(value: object) -> TypeIs[AssignedValue]:
    """Narrow an external object to the supported arm-value union."""
    return isinstance(value, (str, int, float, bool))


def _scalar_values_equal(
    left: AssignedValue,
    right: AssignedValue,
) -> bool:
    """Compare two declared/observed arm values by scalar `==`.

    This deliberately follows Python's numeric equality semantics:
    numerically equivalent values compare equal even across compatible
    numeric types (`1 == 1.0` and `True == 1`). That prevents one row
    value from belonging to both arms. Comparisons whose result cannot
    be reduced to one boolean are not valid arm comparisons.
    """
    try:
        comparison = left == right
        return bool(comparison)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            'DoEffect value arms require scalar values with an '
            'unambiguous equality comparison.',
        ) from exc


def _assigned_value_key(value: AssignedValue) -> str:
    """Canonical identity consistent with scalar arm equality.

    Python considers ``True``, ``1``, and ``1.0`` equal. Integral
    floats therefore use the integer spelling too; non-integral floats
    keep Python's round-trip representation. Strings remain quoted and
    cannot collide with numeric values.
    """
    if isinstance(value, str):
        return canonical_str(str(value))
    if isinstance(value, bool):
        return repr(int(value))
    if isinstance(value, int):
        return repr(int(value))
    number = float(value)
    if math.isfinite(number) and number.is_integer():
        return repr(int(number))
    return repr(number)


def _require_value_source(value: object) -> str:
    """Runtime validation for unchecked callers of `from_values`."""
    if not isinstance(value, str):
        raise TypeError('DoEffect value source must be a string.')
    if not value.strip():
        raise ValueError('DoEffect value source must not be empty.')
    return str(value)


def _require_arm_value(value: object, role: str) -> AssignedValue:
    """Validate one declared arm value: supported scalar, not
    NaN-like (a non-reflexive value could never be matched by
    equality)."""
    if not _is_assigned_value(value):
        raise TypeError(
            f'DoEffect {role} must be a string, integer, float, '
            f'or boolean scalar.',
        )
    if not _scalar_values_equal(value, value):
        raise ValueError(
            f'DoEffect {role} value must not be NaN-like.',
        )
    return value


@dataclass(frozen=True, slots=True)
class _ValueContrast:
    """Exact external-value assignment underlying a binary DoEffect.

    `assignments` holds one `(column, reference, treatment)` triple
    per declared column, sorted by column name — a single-knob
    contrast is the one-triple case, a joint intervention carries
    several. Dataclass equality compares by scalar `==` per entry,
    so equality-equivalent numeric declarations (1 vs 1.0) denote
    the same contrast."""
    assignments: tuple[tuple[str, AssignedValue, AssignedValue], ...]


@dataclass(frozen=True, slots=True, repr=False)
class DoEffect:
    """A structural or exact-value contrast on the claim graph.

    The existing constructor declares a structural, potentially
    multi-arm effect. Each arm is a tuple of slot-replacement
    `Intervention`s; arm identity is the `combined_arm_key` of that
    tuple::

        DoEffect(arms=((), (Intervention(...),)))

    `from_values` instead declares an external binary contrast whose
    orientation is independent of the observed support::

        DoEffect.from_values(
            source='gamma', reference=0.80, treatment=0.99,
        )

    Exact declared values classify rows into the symbolic `baseline`
    and `treatment` arms. Those symbols, rather than formatted value
    labels, are the internal arm identities. Values outside the two
    declared arms are unclassified.

    Pearl-rung-2 edges in the causal graph have an *intervention*
    as the source node, NOT a measurable. `DoEffect` carries N
    arms — each a `do(joint_intervention)` operation in Pearl's
    calculus. The framework dispatches one cell per (grid_point,
    arm) and tags each `RunRow.arm_key` with the canonical
    fingerprint of its arm.

    The empty-tuple arm (when present) is the "no intervention"
    control — the implementation's vanilla composition. Non-empty arms
    are interventional structural deltas. Binary contrast is the
    special case `arms=(control, treatment)` with one tuple
    typically empty. Multi-level (dose-response) is
    `arms=((), (level_1,), (level_2,))`. Factorial 2×2 is
    `arms=((), (a,), (b,), (a, b))`.

    Pearl-coherence: each arm is a joint `do(·)` operation;
    `DoEffect.arms` declares the structural intervention
    conditions. Analyses (per-pair `paired_g` /
    `stratified_arm_diff_pooled`, dose-response, factorial) pick
    the contrast structure separately from the arm declaration.
    The framework's job is to dispatch arms; analyses choose
    references."""
    arms: tuple[tuple[Intervention, ...], ...]
    _value_contrast: _ValueContrast | None = field(
        default=None, repr=False,
    )

    @classmethod
    def from_values(
        cls,
        *,
        source: str | None = None,
        reference: AssignedValue | Mapping[str, AssignedValue],
        treatment: AssignedValue | Mapping[str, AssignedValue],
    ) -> Self:
        """Declare an exact, oriented binary effect over measurables.

        Single-knob form — `source` names the column, the two
        scalars are its arm values::

            DoEffect.from_values(
                source='gamma', reference=0.80, treatment=0.99,
            )

        Joint form — a multi-knob intervention (values that were
        assigned together, or one logical setting surfacing as
        several config fields) declares every co-assigned column,
        mirroring the structural side's joint `do(slot_1=v_1,
        slot_2=v_2)` arms::

            DoEffect.from_values(
                reference={'gamma': 0.80, 'n_step': 1},
                treatment={'gamma': 0.99, 'n_step': 3},
            )

        A row belongs to an arm only when EVERY declared column
        matches that arm's value. Every declared column must vary
        between the arms — a column with one value in both arms is
        held fixed, which is scope's job (`scope=pl.col(...) ==
        v`), not part of the contrast.

        Equality is scalar Python equality, so compatible numeric
        representations which compare equal denote the same arm. In
        particular, `True` and `1` cannot be declared as separate
        arms. NaN-like (non-reflexive) declarations are rejected
        because no observed value could be matched to them by
        equality.

        The placeholder empty intervention tuples preserve the
        public `arms` shape, but value-based effects are data-side
        declarations and must not be dispatched as structural
        runner interventions.
        """
        if isinstance(reference, Mapping) or isinstance(treatment, Mapping):
            if not (
                isinstance(reference, Mapping)
                and isinstance(treatment, Mapping)
            ):
                raise TypeError(
                    'DoEffect.from_values: reference and treatment must '
                    'both be mappings (joint form) or both be scalars '
                    '(single-knob form).',
                )
            if source is not None:
                raise TypeError(
                    'DoEffect.from_values: the joint form takes its '
                    'columns from the mapping keys; `source` must be '
                    'omitted.',
                )
            reference_columns = {
                _require_value_source(name) for name in reference
            }
            treatment_columns = {
                _require_value_source(name) for name in treatment
            }
            if not reference_columns:
                raise ValueError(
                    'DoEffect.from_values: joint form needs at least '
                    'one column.',
                )
            if reference_columns != treatment_columns:
                raise ValueError(
                    'DoEffect.from_values: reference and treatment must '
                    'assign the same columns; got '
                    f'{sorted(reference_columns)!r} vs '
                    f'{sorted(treatment_columns)!r}.',
                )
            pairs = {
                name: (reference[name], treatment[name])
                for name in reference
            }
        else:
            if source is None:
                raise TypeError(
                    'DoEffect.from_values: the single-knob form '
                    'requires `source`.',
                )
            pairs = {source: (reference, treatment)}
        assignments: list[tuple[str, AssignedValue, AssignedValue]] = []
        for name in sorted(pairs):
            raw_reference, raw_treatment = pairs[name]
            column = _require_value_source(name)
            column_reference = _require_arm_value(
                raw_reference, f'reference[{column}]',
            )
            column_treatment = _require_arm_value(
                raw_treatment, f'treatment[{column}]',
            )
            if _scalar_values_equal(column_reference, column_treatment):
                raise ValueError(
                    f'DoEffect.from_values: column {column!r} does not '
                    f'vary between arms ({column_reference!r} in both). '
                    f'A held-fixed setting is scope, not contrast — pin '
                    f'it with `scope=` instead.',
                )
            assignments.append((column, column_reference, column_treatment))
        return cls(
            arms=((), ()),
            _value_contrast=_ValueContrast(
                assignments=tuple(assignments),
            ),
        )

    @property
    def is_value_based(self) -> bool:
        """Whether this effect was declared with `from_values`."""
        return self._value_contrast is not None

    @property
    def value_source_names(self) -> tuple[str, ...] | None:
        """Declared column names (column-sorted), or None for
        structural effects."""
        contrast = self._value_contrast
        if contrast is None:
            return None
        return tuple(name for name, _, _ in contrast.assignments)

    @property
    def value_source_name(self) -> str | None:
        """The single declared column, or None for structural
        effects. Raises on a joint effect — use
        `value_source_names`."""
        contrast = self._value_contrast
        if contrast is None:
            return None
        return self._single_assignment()[0]

    @property
    def reference_assignment(self) -> Mapping[str, AssignedValue]:
        """Declared baseline values per column; invalid for a
        structural effect."""
        contrast = self._require_value_contrast()
        return {
            name: reference
            for name, reference, _ in contrast.assignments
        }

    @property
    def treatment_assignment(self) -> Mapping[str, AssignedValue]:
        """Declared treatment values per column; invalid for a
        structural effect."""
        contrast = self._require_value_contrast()
        return {
            name: treatment
            for name, _, treatment in contrast.assignments
        }

    @property
    def reference_value(self) -> AssignedValue:
        """Declared baseline value of a single-knob effect; raises
        for structural and joint effects."""
        return self._single_assignment()[1]

    @property
    def treatment_value(self) -> AssignedValue:
        """Declared treatment value of a single-knob effect; raises
        for structural and joint effects."""
        return self._single_assignment()[2]

    def classify_value(self, value: object) -> ArmRole | None:
        """Return the declared symbolic arm for a single-knob
        effect's `value`, if either. Raises for joint effects —
        classify whole rows with `classify_row`.

        Unmatched, NaN-like, or non-scalar observed values are
        simply unclassified. Construction has already proved that
        no scalar value can compare equal to both declared arms."""
        column, reference, treatment = self._single_assignment()
        del column
        if not _is_assigned_value(value):
            return None
        try:
            if _scalar_values_equal(value, reference):
                return ArmRole.BASELINE
            if _scalar_values_equal(value, treatment):
                return ArmRole.TREATMENT
        except TypeError:
            return None
        return None

    def classify_row(self, row: Mapping[str, object]) -> ArmRole | None:
        """Return the symbolic arm a row belongs to, if either.

        A row is baseline only when EVERY declared column matches
        its reference value, treatment only when every column
        matches its treatment value. Mixed, unmatched, missing,
        NaN-like, or non-scalar values leave the row unclassified.
        Per-column arm distinctness (validated at construction)
        makes the two memberships mutually exclusive."""
        contrast = self._require_value_contrast()
        all_reference = True
        all_treatment = True
        for column, reference, treatment in contrast.assignments:
            value = row.get(column)
            if not _is_assigned_value(value):
                return None
            try:
                matches_reference = _scalar_values_equal(value, reference)
                matches_treatment = _scalar_values_equal(value, treatment)
            except TypeError:
                return None
            all_reference = all_reference and matches_reference
            all_treatment = all_treatment and matches_treatment
            if not (all_reference or all_treatment):
                return None
        if all_reference:
            return ArmRole.BASELINE
        if all_treatment:
            return ArmRole.TREATMENT
        return None

    def _single_assignment(
        self,
    ) -> tuple[str, AssignedValue, AssignedValue]:
        contrast = self._require_value_contrast()
        if len(contrast.assignments) != 1:
            raise TypeError(
                'joint value DoEffect: use value_source_names / '
                'reference_assignment / treatment_assignment / '
                'classify_row for multi-column effects.',
            )
        return contrast.assignments[0]

    def _require_value_contrast(self) -> _ValueContrast:
        contrast = self._value_contrast
        if contrast is None:
            raise TypeError(
                'Value-arm metadata is unavailable on a structural '
                'DoEffect.',
            )
        return contrast

    @override
    def __repr__(self) -> str:
        """Expose the authored declaration rather than private storage."""
        contrast = self._value_contrast
        if contrast is None:
            return f'DoEffect(arms={self.arms!r})'
        if len(contrast.assignments) == 1:
            column, reference, treatment = contrast.assignments[0]
            return (
                'DoEffect.from_values('
                f'source={column!r}, '
                f'reference={reference!r}, '
                f'treatment={treatment!r})'
            )
        return (
            'DoEffect.from_values('
            f'reference={self.reference_assignment!r}, '
            f'treatment={self.treatment_assignment!r})'
        )

    def arm_keys(self) -> tuple[str, ...]:
        """Canonical fingerprint per arm. Empty tuple → `'baseline'`;
        non-empty → `'+'`-joined slot=replacement keys. The string
        flows to `RunRow.arm_key` per cell, and analyses filter
        cells by string match on these keys."""
        if self.is_value_based:
            return (ArmRole.BASELINE.value, ArmRole.TREATMENT.value)
        return tuple(combined_arm_key(a) for a in self.arms)

    def node_key(self) -> str:
        """Canonical string identity for the do-node in
        `CausalGraph`. Renders as `do(arm0|arm1|...|armN)` — the
        multi-arm contrast made explicit at the intervention
        node. Value-based effects render each arm's joint
        assignment `+`-joined in column order, mirroring
        `combined_arm_key`."""
        contrast = self._value_contrast
        if contrast is not None:
            reference = '+'.join(
                f'{name}={_assigned_value_key(ref)}'
                for name, ref, _ in contrast.assignments
            )
            treatment = '+'.join(
                f'{name}={_assigned_value_key(treat)}'
                for name, _, treat in contrast.assignments
            )
            return f'do({reference}|{treatment})'
        return f"do({'|'.join(self.arm_keys())})"
