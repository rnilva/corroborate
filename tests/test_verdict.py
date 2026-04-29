"""Tests for Verdict and RefutationClass — the type-and-property
level invariants the rest of the framework relies on."""
from __future__ import annotations

from corroborate.verdict import RefutationClass, Verdict


# ============ Verdict ============

def test_verdict_has_five_members() -> None:
    assert len(list(Verdict)) == 5


def test_verdict_string_values_are_stable() -> None:
    """String values are persisted to parquet and used in v9-row
    interop; renames here cascade to the corpus on disk."""
    assert Verdict.HELD.value == 'held'
    assert Verdict.HELD_WITH_SCOPE_FLAG.value == 'held_with_scope_flag'
    assert Verdict.NO_EFFECT.value == 'no_effect'
    assert Verdict.POWER_INSUFFICIENT.value == 'power_insufficient'
    assert Verdict.INVARIANT_VIOLATION.value == 'invariant_violation'


def test_terminality() -> None:
    """POWER_INSUFFICIENT is the only non-terminal verdict —
    explicitly says 'rerun at higher n.'"""
    assert Verdict.HELD.is_terminal()
    assert Verdict.HELD_WITH_SCOPE_FLAG.is_terminal()
    assert Verdict.NO_EFFECT.is_terminal()
    assert Verdict.INVARIANT_VIOLATION.is_terminal()
    assert not Verdict.POWER_INSUFFICIENT.is_terminal()


def test_corroboration_includes_held_and_scope_flag() -> None:
    """Both HELD and HELD_WITH_SCOPE_FLAG attest to population-
    level corroboration; the latter additionally signals
    heterogeneity. NO_EFFECT, POWER_INSUFFICIENT, and
    INVARIANT_VIOLATION are not corroborations."""
    assert Verdict.HELD.is_corroboration()
    assert Verdict.HELD_WITH_SCOPE_FLAG.is_corroboration()
    assert not Verdict.NO_EFFECT.is_corroboration()
    assert not Verdict.POWER_INSUFFICIENT.is_corroboration()
    assert not Verdict.INVARIANT_VIOLATION.is_corroboration()


def test_uniform_is_held_only() -> None:
    """`is_uniform()` distinguishes plain HELD (low I²) from
    HELD_WITH_SCOPE_FLAG (high I²). Only HELD is uniform."""
    assert Verdict.HELD.is_uniform()
    assert not Verdict.HELD_WITH_SCOPE_FLAG.is_uniform()
    assert not Verdict.NO_EFFECT.is_uniform()
    assert not Verdict.POWER_INSUFFICIENT.is_uniform()
    assert not Verdict.INVARIANT_VIOLATION.is_uniform()


def test_refutation_is_no_effect_only() -> None:
    """POWER_INSUFFICIENT is NOT a refutation — that's the
    methodological point. INVARIANT_VIOLATION is also not a
    refutation; it's an out-of-scope verdict. HELD_WITH_SCOPE_FLAG
    corroborates with caveat — also not a refutation."""
    assert Verdict.NO_EFFECT.is_refutation()
    assert not Verdict.HELD.is_refutation()
    assert not Verdict.HELD_WITH_SCOPE_FLAG.is_refutation()
    assert not Verdict.POWER_INSUFFICIENT.is_refutation()
    assert not Verdict.INVARIANT_VIOLATION.is_refutation()


def test_corroboration_and_refutation_are_disjoint() -> None:
    """No verdict is simultaneously corroboration and refutation."""
    for v in Verdict:
        assert not (v.is_corroboration() and v.is_refutation())


# ============ RefutationClass ============

def test_refutation_class_has_four_members() -> None:
    assert len(list(RefutationClass)) == 4


def test_refutation_class_values_are_stable() -> None:
    assert RefutationClass.NULL_EFFECT.value == 'null_effect'
    assert RefutationClass.SIGN_FLIP.value == 'sign_flip'
    assert RefutationClass.UNDERPOWERED.value == 'underpowered'
    assert RefutationClass.TIME_BUDGET_DORMANT.value == 'time_budget_dormant'


def test_refutation_class_values_are_distinct() -> None:
    """No collisions in the string-value space (parquet keys)."""
    values = {m.value for m in RefutationClass}
    assert len(values) == len(list(RefutationClass))
