"""Tests for `Hypothesis[R]` and leaf-value canonicalization.
Strict typing exercised: bridges and hypothesis share R.

`MechanismKey` no longer exists as a framework artifact; the
configurational identity of a hypothesis is recovered from its
runs' `measurements` via `aggregate.leaf_signature`. These tests
cover the data-class shape + the `canonical_str` helper used for
leaf-value serialization."""
from __future__ import annotations

from collections.abc import Mapping

from corroborate._internals.canonical import canonical_str
from corroborate.core.claim import claim
from corroborate.core.hypothesis import Hypothesis
from corroborate.core.intervention import Intervention


# ============ Construction ============

def test_hypothesis_minimal() -> None:
    h: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='baseline', intervention={},
    )
    assert h.name == 'baseline'
    assert h.intervention == {}
    assert h.measurables == ()
    assert h.predicted_direction is None


def test_hypothesis_predicted_direction() -> None:
    h: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='h', intervention={}, predicted_direction='a_gt_b',
    )
    assert h.predicted_direction == 'a_gt_b'


# ============ HP-value canonicalization ============

def test_canonical_str_distinguishes_value_types() -> None:
    """Different types canonicalise distinctly so '1' (int)
    doesn't collide with '1' (str)."""
    assert canonical_str(1) != canonical_str('1')


def test_canonical_str_handles_claims() -> None:
    """Claim-typed values canonicalise via `Claim:<name>`."""
    @claim
    def my_alternative(x: int) -> int:
        return x * 2

    assert canonical_str(my_alternative) == 'Claim:my_alternative'


def test_canonical_str_distinguishes_different_claims() -> None:
    @claim
    def alt_a(x: int) -> int:
        return x

    @claim
    def alt_b(x: int) -> int:
        return x

    assert canonical_str(alt_a) != canonical_str(alt_b)


def test_canonical_str_handles_plain_callables() -> None:
    """Plain callables canonicalise via their `__name__`."""
    def plain_alternative(x: int) -> int:
        return x

    s = canonical_str(plain_alternative)
    assert 'plain_alternative' in s


def test_canonical_str_bool_distinct_from_int() -> None:
    """bool is a subclass of int; the canonicaliser handles them
    separately so True/False don't collide with 1/0."""
    assert canonical_str(True) != canonical_str(1)
    assert canonical_str(False) != canonical_str(0)


def test_canonical_str_partial_canonicalises_keywords() -> None:
    """`functools.partial(fn, kw=value)` canonicalises by
    recursing into `.func` and lex-encoding `.keywords`. Two
    independently-constructed partials with the same wrapped
    callable + same kwargs are equal."""
    from functools import partial

    def fn(x: int, *, kw: int = 0) -> int:
        return x + kw

    p1 = partial(fn, kw=5)
    p2 = partial(fn, kw=5)
    assert canonical_str(p1) == canonical_str(p2)


def test_canonical_str_partial_distinguishes_kwargs() -> None:
    """Partials with different baked kwargs canonicalise
    distinctly."""
    from functools import partial

    def fn(*, kw: int = 0) -> int:
        return kw

    p1 = partial(fn, kw=5)
    p2 = partial(fn, kw=10)
    assert canonical_str(p1) != canonical_str(p2)


def test_canonical_str_dataclass_field_expansion() -> None:
    """Frozen-dataclass instances canonicalise by sorted-field
    expansion."""
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class HP:
        a: int = 1
        b: float = 2.0

    s_default = canonical_str(HP())
    s_changed = canonical_str(HP(a=2))
    assert s_default != s_changed
    # Same values → same canonical string.
    assert canonical_str(HP(a=1, b=2.0)) == canonical_str(HP(a=1, b=2.0))


def test_canonical_str_tuple_recurses() -> None:
    """Tuples of scalars get a stable canonical form via element-
    wise recursion."""
    s = canonical_str((1, 2, 3))
    assert s == '(1,2,3)'


# ============ intervention_arms + arm_key ============

@claim
def _alt_a(x: int) -> int:
    return x


@claim
def _alt_b(x: int) -> int:
    return x * 2


def test_hypothesis_default_arms_baseline() -> None:
    """An empty `intervention_arms` tuple yields the baseline arm
    key, regardless of what the runtime `intervention` dict
    contains."""
    h: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='vanilla', intervention={'gamma': 0.99, 'lr': 1e-3},
    )
    assert h.intervention_arms == ()
    assert h.arm_key() == 'baseline'


def test_hypothesis_arm_key_reflects_arms() -> None:
    """A non-empty `intervention_arms` produces a fingerprint
    derived from the typed swaps."""
    h: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='ddqn', intervention={'gamma': 0.99},
        intervention_arms=(
            Intervention(slot_path='bootstrap', replacement=_alt_a),
        ),
    )
    assert h.arm_key() == 'bootstrap=Claim:_alt_a'


def test_arm_key_invariant_under_hp_change() -> None:
    """Different HP grid points with the same `intervention_arms`
    produce the same arm key — the framework's load-bearing
    promise that HPs are covariates, not arm distinguishers."""
    arms = (Intervention(slot_path='bootstrap', replacement=_alt_a),)
    h_lo: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='ddqn',
        intervention={'gamma': 0.99, 'lr': 1e-3},
        intervention_arms=arms,
    )
    h_hi: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='ddqn',
        intervention={'gamma': 0.95, 'lr': 1e-4},
        intervention_arms=arms,
    )
    assert h_lo.arm_key() == h_hi.arm_key()


def test_arm_key_distinguishes_different_arms() -> None:
    """Two hypotheses with different arms produce different arm
    keys."""
    h_a: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='a', intervention={},
        intervention_arms=(
            Intervention(slot_path='bootstrap', replacement=_alt_a),
        ),
    )
    h_b: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='b', intervention={},
        intervention_arms=(
            Intervention(slot_path='bootstrap', replacement=_alt_b),
        ),
    )
    assert h_a.arm_key() != h_b.arm_key()


