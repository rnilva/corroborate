"""Tests for the `Hypothesis` Protocol + canonical_str leaf
fingerprinting + combined_arm_key.

After Phase 6 the Hypothesis dataclass is gone — the framework's
verdict-time contract is a runtime_checkable Protocol with three
attributes (INTERVENTION + BRIDGES + MEASURABLES). Substrate
authoring uses module-level constants OR class-with-ClassVars
to satisfy it. These tests cover:
- The Protocol's runtime_checkable shape.
- `canonical_str` leaf-value fingerprinting.
- `combined_arm_key` over Intervention tuples (HPs don't
  perturb arm identity)."""
from __future__ import annotations

from corroborate._internals.canonical import canonical_str
from corroborate.core.claim import claim
from corroborate.core.hypothesis import Hypothesis
from corroborate.core.intervention import (
    DoEffect, Intervention, combined_arm_key,
)


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
    """`functools.partial(fn, kw=value)` canonicalises by recursing
    into `.func` and lex-encoding `.keywords`. Two independently-
    constructed partials with the same wrapped callable + same
    kwargs are equal."""
    from functools import partial

    def fn(x: int, *, kw: int = 0) -> int:
        return x + kw

    p1 = partial(fn, kw=5)
    p2 = partial(fn, kw=5)
    assert canonical_str(p1) == canonical_str(p2)


def test_canonical_str_partial_distinguishes_kwargs() -> None:
    """Partials with different baked kwargs canonicalise distinctly."""
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
    assert canonical_str(HP(a=1, b=2.0)) == canonical_str(HP(a=1, b=2.0))


def test_canonical_str_tuple_recurses() -> None:
    """Tuples of scalars get a stable canonical form via element-
    wise recursion."""
    s = canonical_str((1, 2, 3))
    assert s == '(1,2,3)'


# ============ Intervention arm_key fingerprinting ============

@claim
def _alt_a(x: int) -> int:
    return x


@claim
def _alt_b(x: int) -> int:
    return x * 2


def test_combined_arm_key_empty_tuple_baseline() -> None:
    """Empty Intervention tuple → `'baseline'`."""
    assert combined_arm_key(()) == 'baseline'


def test_combined_arm_key_reflects_swap() -> None:
    """Non-empty Intervention tuple produces a fingerprint derived
    from the typed swap's slot_path + canonical_str(replacement)."""
    arms = (Intervention(slot_path='bootstrap', replacement=_alt_a),)
    assert combined_arm_key(arms) == 'bootstrap=Claim:_alt_a'


def test_combined_arm_key_distinguishes_different_replacements() -> None:
    """Same slot_path, different replacements → different arm_keys."""
    arms_a = (Intervention(slot_path='bootstrap', replacement=_alt_a),)
    arms_b = (Intervention(slot_path='bootstrap', replacement=_alt_b),)
    assert combined_arm_key(arms_a) != combined_arm_key(arms_b)


# ============ Hypothesis Protocol shape ============

def test_module_satisfies_protocol() -> None:
    """A module-level INTERVENTION + BRIDGES + MEASURABLES makes
    the module conform structurally — this test imports the
    framework's `core.hypothesis` module which doesn't have those
    attrs, so it should NOT be a Hypothesis."""
    import corroborate.core.hypothesis as mod
    assert not isinstance(mod, Hypothesis)


def test_doeffect_arm_keys() -> None:
    """`DoEffect.treatment_arm_key()` derives from
    `combined_arm_key(treatment)`; baseline same."""
    de = DoEffect(
        treatment=(Intervention(slot_path='bootstrap', replacement=_alt_a),),
        baseline=(),
    )
    assert de.treatment_arm_key() == 'bootstrap=Claim:_alt_a'
    assert de.baseline_arm_key() == 'baseline'
