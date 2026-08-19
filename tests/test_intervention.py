"""Tests for the typed Intervention primitive."""
from __future__ import annotations

import functools
from enum import IntEnum, StrEnum

import pytest

from corroborate.core.claim import FnClaim, claim
from corroborate.core.intervention import (
    ArmRole,
    AssignedValue,
    DoEffect,
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


# ============ value-based DoEffect ============

def test_value_doeffect_declares_exact_oriented_binary_arms() -> None:
    effect = DoEffect.from_values(
        source='gamma', reference=0.80, treatment=0.99,
    )

    assert effect.is_value_based
    assert effect.value_source_name == 'gamma'
    assert effect.reference_value == 0.80
    assert effect.treatment_value == 0.99
    assert effect.arm_keys() == ('baseline', 'treatment')
    assert effect.classify_value(0.80) is ArmRole.BASELINE
    assert effect.classify_value(0.99) is ArmRole.TREATMENT


def test_value_doeffect_does_not_infer_unobserved_arm_values() -> None:
    effect = DoEffect.from_values(
        source='gamma', reference=0.80, treatment=0.99,
    )

    assert effect.classify_value(0.90) is None
    assert effect.classify_value(None) is None
    assert effect.classify_value(float('nan')) is None


def test_value_doeffect_preserves_declared_orientation_without_sorting() -> None:
    effect = DoEffect.from_values(
        source='gamma', reference=0.99, treatment=0.80,
    )

    assert effect.classify_value(0.99) is ArmRole.BASELINE
    assert effect.classify_value(0.80) is ArmRole.TREATMENT


def test_value_doeffect_identity_does_not_use_short_display_format() -> None:
    effect = DoEffect.from_values(
        source='gamma', reference=0.8000001, treatment=0.8000002,
    )

    assert effect.classify_value(0.8000001) is ArmRole.BASELINE
    assert effect.classify_value(0.8000002) is ArmRole.TREATMENT
    assert '0.8000001' in effect.node_key()
    assert '0.8000002' in effect.node_key()


def test_value_doeffect_node_identity_matches_scalar_equality() -> None:
    """Equality-equivalent numeric declarations share one node key."""
    integer = DoEffect.from_values(
        source='condition', reference=1, treatment=2,
    )
    floating = DoEffect.from_values(
        source='condition', reference=1.0, treatment=2.0,
    )
    boolean = DoEffect.from_values(
        source='condition', reference=True, treatment=2,
    )

    assert integer == floating == boolean
    assert integer.node_key() == floating.node_key() == boolean.node_key()


def test_value_doeffect_normalises_scalar_enum_subclasses() -> None:
    class StringCondition(StrEnum):
        SOURCE = 'condition'
        REFERENCE = 'a'
        TREATMENT = 'b'

    class NumericCondition(IntEnum):
        REFERENCE = 1
        TREATMENT = 2

    string_enum = DoEffect.from_values(
        source=StringCondition.SOURCE,
        reference=StringCondition.REFERENCE,
        treatment=StringCondition.TREATMENT,
    )
    string_primitive = DoEffect.from_values(
        source='condition', reference='a', treatment='b',
    )
    numeric_enum = DoEffect.from_values(
        source='condition',
        reference=NumericCondition.REFERENCE,
        treatment=NumericCondition.TREATMENT,
    )
    numeric_primitive = DoEffect.from_values(
        source='condition', reference=1, treatment=2,
    )

    assert string_enum == string_primitive
    assert string_enum.node_key() == string_primitive.node_key()
    assert numeric_enum == numeric_primitive
    assert numeric_enum.node_key() == numeric_primitive.node_key()


def test_value_doeffect_repr_shows_the_declaration() -> None:
    effect = DoEffect.from_values(
        source='gamma', reference=0.8, treatment=0.99,
    )

    assert repr(effect) == (
        "DoEffect.from_values(source='gamma', reference=0.8, "
        'treatment=0.99)'
    )


@pytest.mark.parametrize(
    ('reference', 'treatment'),
    [
        (True, 1),
        (1, 1.0),
        ('same', 'same'),
    ],
)
def test_value_doeffect_rejects_equivalent_arms(
    reference: AssignedValue,
    treatment: AssignedValue,
) -> None:
    with pytest.raises(ValueError, match='does not vary between arms'):
        DoEffect.from_values(
            source='condition',
            reference=reference,
            treatment=treatment,
        )


@pytest.mark.parametrize(
    ('reference', 'treatment', 'invalid_role'),
    [
        (float('nan'), 1.0, 'reference'),
        (0.0, float('nan'), 'treatment'),
    ],
)
def test_value_doeffect_rejects_nan_declarations(
    reference: AssignedValue,
    treatment: AssignedValue,
    invalid_role: str,
) -> None:
    with pytest.raises(ValueError, match=invalid_role):
        DoEffect.from_values(
            source='condition',
            reference=reference,
            treatment=treatment,
        )


def test_value_doeffect_rejects_empty_source_name() -> None:
    with pytest.raises(ValueError, match='source'):
        DoEffect.from_values(source='  ', reference=0, treatment=1)


def test_value_doeffect_rejects_non_string_source_name() -> None:
    with pytest.raises(TypeError, match='source'):
        DoEffect.from_values(
            source=1,  # type: ignore[arg-type]
            reference=0,
            treatment=1,
        )


def test_value_doeffect_rejects_non_scalar_declarations() -> None:
    with pytest.raises(TypeError, match='reference'):
        DoEffect.from_values(
            source='condition',
            reference=['baseline'],  # type: ignore[arg-type]
            treatment='treatment',
        )


def test_joint_value_doeffect_declares_multi_column_arms() -> None:
    """A multi-knob intervention (or one logical knob surfacing as
    several config fields) declares every co-assigned column; a row
    belongs to an arm only when ALL declared columns match."""
    effect = DoEffect.from_values(
        reference={'gamma': 0.80, 'n_step': 1},
        treatment={'gamma': 0.99, 'n_step': 3},
    )

    assert effect.is_value_based
    assert effect.value_source_names == ('gamma', 'n_step')
    assert effect.reference_assignment == {'gamma': 0.80, 'n_step': 1}
    assert effect.treatment_assignment == {'gamma': 0.99, 'n_step': 3}
    assert effect.arm_keys() == ('baseline', 'treatment')
    assert effect.node_key() == 'do(gamma=0.8+n_step=1|gamma=0.99+n_step=3)'

    assert effect.classify_row(
        {'gamma': 0.80, 'n_step': 1},
    ) is ArmRole.BASELINE
    assert effect.classify_row(
        {'gamma': 0.99, 'n_step': 3},
    ) is ArmRole.TREATMENT
    # A partial match is neither arm — the joint assignment is the
    # unit of membership.
    assert effect.classify_row({'gamma': 0.99, 'n_step': 1}) is None
    assert effect.classify_row({'gamma': 0.99}) is None


def test_joint_value_doeffect_single_only_accessors_raise() -> None:
    effect = DoEffect.from_values(
        reference={'gamma': 0.80, 'n_step': 1},
        treatment={'gamma': 0.99, 'n_step': 3},
    )
    with pytest.raises(TypeError, match='joint value DoEffect'):
        _ = effect.reference_value
    with pytest.raises(TypeError, match='joint value DoEffect'):
        effect.classify_value(0.80)


def test_joint_value_doeffect_rejects_held_fixed_columns() -> None:
    """A column with one value in both arms is scope, not contrast."""
    with pytest.raises(ValueError, match='does not vary between arms'):
        DoEffect.from_values(
            reference={'gamma': 0.80, 'n_step': 3},
            treatment={'gamma': 0.99, 'n_step': 3},
        )


def test_joint_value_doeffect_rejects_mismatched_columns() -> None:
    with pytest.raises(ValueError, match='same columns'):
        DoEffect.from_values(
            reference={'gamma': 0.80},
            treatment={'gamma': 0.99, 'n_step': 3},
        )


def test_joint_value_doeffect_rejects_mixed_forms() -> None:
    with pytest.raises(TypeError, match='must be omitted'):
        DoEffect.from_values(
            source='gamma',
            reference={'gamma': 0.80},
            treatment={'gamma': 0.99},
        )
    with pytest.raises(TypeError, match='both be mappings'):
        DoEffect.from_values(
            source='gamma',
            reference={'gamma': 0.80},
            treatment=0.99,
        )


def test_structural_doeffect_preserves_existing_api() -> None:
    effect = DoEffect(arms=((),))

    assert not effect.is_value_based
    assert effect.value_source_name is None
    assert effect.arm_keys() == ('baseline',)
    with pytest.raises(TypeError, match='structural DoEffect'):
        _ = effect.reference_value
    with pytest.raises(TypeError, match='structural DoEffect'):
        effect.classify_value(0)


# ============ apply / apply_interventions ============

def _make_claim(label: str) -> FnClaim[..., str]:
    """Build a unique `@claim`-wrapped callable carrying `label` as
    its `__name__` — satisfies the `Replacement` callable contract
    while remaining identity-comparable in assertions. The shape
    matches how authored implementations compose: `replacement=` always
    receives a Claim, never a stringly-typed sentinel.

    The typed first overload of `claim` returns `FnClaim[P, T]`;
    cast through `Callable` keeps the helper's return narrow."""
    def _r() -> str:
        return label
    _r.__name__ = label
    return claim(_r)


_DEFAULT_BS = _make_claim('default_bs')
_DEFAULT_AS = _make_claim('default_as')


def _theory(
    *,
    bootstrap: object = _DEFAULT_BS,
    action_select: object = _DEFAULT_AS,
) -> dict[str, object]:
    """Tiny stand-in for a `dqn`-shape kwarg-only function. The
    return value tells us which slot got which value."""
    return {'bootstrap': bootstrap, 'action_select': action_select}


def test_apply_returns_partial_with_slot_pinned() -> None:
    """`Intervention.apply(base)` returns `partial(base, slot_path
    =replacement)` — the post-do() SCM with that slot pinned."""
    ddqn_bs = _make_claim('ddqn_bs')
    iv = Intervention(slot_path='bootstrap', replacement=ddqn_bs)
    do_theory = iv.apply(_theory)
    out = do_theory()
    assert out['bootstrap'] is ddqn_bs
    # All other slots fall through to base's defaults.
    assert out['action_select'] is _DEFAULT_AS


def test_apply_does_not_touch_other_slots() -> None:
    """Pearl: do() on X doesn't alter Y's mechanism. Verify by
    applying do(bootstrap=X) and checking action_select is
    unchanged."""
    iv = Intervention(slot_path='bootstrap', replacement=_make_claim('X'))
    out = iv.apply(_theory)()
    assert out['action_select'] is _DEFAULT_AS


def test_apply_rejects_nested_slot_path() -> None:
    """Nested slot_paths require parent-claim reconstruction; not
    supported by .apply(). Author constructs the parent claim
    with the substituted child explicitly."""
    iv = Intervention(
        slot_path='replay.sample', replacement=_make_claim('X'),
    )
    with pytest.raises(ValueError, match='nested slot_path'):
        iv.apply(_theory)


def test_apply_interventions_sequential_composition() -> None:
    """Sequential do()s on disjoint slots produce a composition
    where each slot is pinned to its respective replacement."""
    ddqn_bs = _make_claim('ddqn_bs')
    boltzmann = _make_claim('boltzmann')
    arms = (
        Intervention(slot_path='bootstrap', replacement=ddqn_bs),
        Intervention(slot_path='action_select', replacement=boltzmann),
    )
    composed = apply_interventions(_theory, arms)
    out = composed()
    assert out['bootstrap'] is ddqn_bs
    assert out['action_select'] is boltzmann


def test_apply_interventions_overlapping_slots_last_wins() -> None:
    """Overlapping do()s on the same slot: later one wins (Pearl:
    the second do() shadows the first, since both pin the same
    variable)."""
    second = _make_claim('second')
    arms = (
        Intervention(slot_path='bootstrap', replacement=_make_claim('first')),
        Intervention(slot_path='bootstrap', replacement=second),
    )
    composed = apply_interventions(_theory, arms)
    out = composed()
    assert out['bootstrap'] is second


def test_apply_interventions_empty_returns_base_unchanged() -> None:
    """Baseline arm: empty interventions → base composition is
    used as-is, all slots resolve to defaults."""
    composed = apply_interventions(_theory, ())
    out = composed()
    assert out['bootstrap'] is _DEFAULT_BS
    assert out['action_select'] is _DEFAULT_AS


def test_apply_pearl_honest_pair_shares_base() -> None:
    """Demonstrates the Pearl-honest contrast: treatment and
    baseline applied to the same base produce compositions that
    differ ONLY on the do()'d slots."""
    treatment_bs = _make_claim('treatment_bs')
    treatment_arms = (
        Intervention(slot_path='bootstrap', replacement=treatment_bs),
    )
    baseline_arms = ()  # baseline = no do()
    treatment = apply_interventions(_theory, treatment_arms)
    baseline = apply_interventions(_theory, baseline_arms)
    t_out, b_out = treatment(), baseline()
    # Treatment's bootstrap is the do()'d value.
    assert t_out['bootstrap'] is treatment_bs
    # Baseline's bootstrap is the natural default.
    assert b_out['bootstrap'] is _DEFAULT_BS
    # Every other slot agrees — only the do()'d slot differs.
    assert t_out['action_select'] is b_out['action_select']
