"""Tests for the typed Intervention primitive."""
from __future__ import annotations

import functools

import pytest

from corroborate.claim import claim
from corroborate.intervention import (
    Intervention,
    apply_interventions,
    combined_arm_key,
)


@claim
def _fn_a(x: int) -> int:
    return x


@claim
def _fn_b(x: int) -> int:
    return x * 2


def test_arm_key_includes_slot_path_and_replacement() -> None:
    i = Intervention(slot_path='bootstrap', replacement=_fn_a)
    assert i.arm_key() == 'bootstrap=Claim:_fn_a'


def test_equal_replacement_yields_equal_arm_key() -> None:
    i1 = Intervention(slot_path='bootstrap', replacement=_fn_a)
    i2 = Intervention(slot_path='bootstrap', replacement=_fn_a)
    assert i1.arm_key() == i2.arm_key()


def test_different_replacement_yields_different_arm_key() -> None:
    i1 = Intervention(slot_path='bootstrap', replacement=_fn_a)
    i2 = Intervention(slot_path='bootstrap', replacement=_fn_b)
    assert i1.arm_key() != i2.arm_key()


def test_different_slot_path_yields_different_arm_key() -> None:
    i1 = Intervention(slot_path='bootstrap', replacement=_fn_a)
    i2 = Intervention(slot_path='action_select', replacement=_fn_a)
    assert i1.arm_key() != i2.arm_key()


def test_partial_replacement_canonicalises_by_keywords() -> None:
    i1 = Intervention(
        slot_path='bootstrap', replacement=functools.partial(_fn_a, x=5),
    )
    i2 = Intervention(
        slot_path='bootstrap', replacement=functools.partial(_fn_a, x=5),
    )
    assert i1.arm_key() == i2.arm_key()
    i3 = Intervention(
        slot_path='bootstrap', replacement=functools.partial(_fn_a, x=6),
    )
    assert i1.arm_key() != i3.arm_key()


def test_combined_arm_key_baseline_for_empty_tuple() -> None:
    assert combined_arm_key(()) == 'baseline'


def test_combined_arm_key_single_arm_equals_arm_key() -> None:
    i = Intervention(slot_path='bootstrap', replacement=_fn_a)
    assert combined_arm_key((i,)) == i.arm_key()


def test_combined_arm_key_order_invariance() -> None:
    i1 = Intervention(slot_path='bootstrap', replacement=_fn_a)
    i2 = Intervention(slot_path='action_select', replacement=_fn_a)
    assert combined_arm_key((i1, i2)) == combined_arm_key((i2, i1))


def test_combined_arm_key_contains_each_arm() -> None:
    i1 = Intervention(slot_path='bootstrap', replacement=_fn_a)
    i2 = Intervention(slot_path='action_select', replacement=_fn_b)
    key = combined_arm_key((i1, i2))
    assert i1.arm_key() in key
    assert i2.arm_key() in key


# ============ apply / apply_interventions ============

def _theory(*, bootstrap: object = 'default_bs',
            action_select: object = 'default_as') -> dict[str, object]:
    """Tiny stand-in for a `dqn`-shape kwarg-only function. The
    return value tells us which slot got which value."""
    return {'bootstrap': bootstrap, 'action_select': action_select}


def test_apply_returns_partial_with_slot_pinned() -> None:
    """`Intervention.apply(base)` returns `partial(base, slot_path
    =replacement)` — the post-do() SCM with that slot pinned."""
    iv = Intervention(slot_path='bootstrap', replacement='ddqn_bs')
    do_theory = iv.apply(_theory)
    out = do_theory()
    assert out['bootstrap'] == 'ddqn_bs'
    # All other slots fall through to base's defaults.
    assert out['action_select'] == 'default_as'


def test_apply_does_not_touch_other_slots() -> None:
    """Pearl: do() on X doesn't alter Y's mechanism. Verify by
    applying do(bootstrap=X) and checking action_select is
    unchanged."""
    iv = Intervention(slot_path='bootstrap', replacement='X')
    out = iv.apply(_theory)()
    assert out['action_select'] == 'default_as'


def test_apply_rejects_nested_slot_path() -> None:
    """Nested slot_paths require parent-claim reconstruction; not
    supported by .apply(). Author constructs the parent claim
    with the substituted child explicitly."""
    iv = Intervention(slot_path='replay.sample', replacement='X')
    with pytest.raises(ValueError, match='nested slot_path'):
        iv.apply(_theory)


def test_apply_interventions_sequential_composition() -> None:
    """Sequential do()s on disjoint slots produce a composition
    where each slot is pinned to its respective replacement."""
    arms = (
        Intervention(slot_path='bootstrap', replacement='ddqn_bs'),
        Intervention(slot_path='action_select', replacement='boltzmann'),
    )
    composed = apply_interventions(_theory, arms)
    out = composed()
    assert out['bootstrap'] == 'ddqn_bs'
    assert out['action_select'] == 'boltzmann'


def test_apply_interventions_overlapping_slots_last_wins() -> None:
    """Overlapping do()s on the same slot: later one wins (Pearl:
    the second do() shadows the first, since both pin the same
    variable)."""
    arms = (
        Intervention(slot_path='bootstrap', replacement='first'),
        Intervention(slot_path='bootstrap', replacement='second'),
    )
    composed = apply_interventions(_theory, arms)
    out = composed()
    assert out['bootstrap'] == 'second'


def test_apply_interventions_empty_returns_base_unchanged() -> None:
    """Baseline arm: empty interventions → base composition is
    used as-is, all slots resolve to defaults."""
    composed = apply_interventions(_theory, ())
    out = composed()
    assert out['bootstrap'] == 'default_bs'
    assert out['action_select'] == 'default_as'


def test_apply_pearl_honest_pair_shares_base() -> None:
    """Demonstrates the Pearl-honest contrast: treatment and
    baseline applied to the same base produce compositions that
    differ ONLY on the do()'d slots."""
    treatment_arms = (
        Intervention(slot_path='bootstrap', replacement='treatment_bs'),
    )
    baseline_arms = ()  # baseline = no do()
    treatment = apply_interventions(_theory, treatment_arms)
    baseline = apply_interventions(_theory, baseline_arms)
    t_out, b_out = treatment(), baseline()
    # Treatment's bootstrap is the do()'d value.
    assert t_out['bootstrap'] == 'treatment_bs'
    # Baseline's bootstrap is the natural default.
    assert b_out['bootstrap'] == 'default_bs'
    # Every other slot agrees — only the do()'d slot differs.
    assert t_out['action_select'] == b_out['action_select']
