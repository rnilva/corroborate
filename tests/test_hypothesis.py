"""Tests for `Hypothesis[R]`, `MechanismKey`, and the canonical
intervention signature. Strict typing exercised: bridges and
hypothesis share R; mechanism_key is structurally hashable."""
from __future__ import annotations

from collections.abc import Mapping

from corroborate.bridge import BridgeResult, bridge
from corroborate.claim import claim
from corroborate.hypothesis import Hypothesis, MechanismKey
from corroborate.verdict import Verdict


# ============ Construction ============

def test_hypothesis_minimal() -> None:
    h: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='baseline',
        intervention={},
    )
    assert h.name == 'baseline'
    assert h.intervention == {}
    assert h.bridges == ()
    assert h.predicted_direction is None


def test_hypothesis_with_bridges() -> None:
    @bridge(targets=('x',))
    def b(record: Mapping[str, object]) -> BridgeResult:
        del record
        return BridgeResult(
            verdict=Verdict.HELD, reason='', stats={},
            name='', targets=(),
        )

    h: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='with_bridges',
        intervention={'slot': 'value'},
        bridges=(b,),
    )
    assert len(h.bridges) == 1
    assert h.bridges[0] is b


# ============ MechanismKey ============

def test_mechanism_key_returns_mechanism_key_dataclass() -> None:
    h: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='h', intervention={'slot': 'value'},
    )
    assert isinstance(h.mechanism_key, MechanismKey)


def test_mechanism_key_intervention_signature_is_sorted() -> None:
    """The intervention signature is sorted by slot name so two
    hypotheses with reordered intervention dicts produce the same
    key (dicts in Python 3.7+ are insertion-ordered, but order
    isn't structural)."""
    h1: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='a',
        intervention={'b': 1, 'a': 2, 'c': 3},
    )
    h2: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='b',  # different name — name doesn't affect key
        intervention={'c': 3, 'a': 2, 'b': 1},
    )
    assert h1.mechanism_key == h2.mechanism_key


def test_mechanism_key_distinguishes_different_interventions() -> None:
    h1: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='h', intervention={'slot': 'value_a'},
    )
    h2: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='h', intervention={'slot': 'value_b'},
    )
    assert h1.mechanism_key != h2.mechanism_key


def test_mechanism_key_includes_direction() -> None:
    h_no_dir: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='h', intervention={},
    )
    h_a_gt: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='h', intervention={}, predicted_direction='a_gt_b',
    )
    h_a_lt: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='h', intervention={}, predicted_direction='a_lt_b',
    )

    assert h_no_dir.mechanism_key != h_a_gt.mechanism_key
    assert h_a_gt.mechanism_key != h_a_lt.mechanism_key


def test_mechanism_key_bridge_names_are_order_independent() -> None:
    @bridge(targets=('x',))
    def b1(record: Mapping[str, object]) -> BridgeResult:
        del record
        return BridgeResult(
            verdict=Verdict.HELD, reason='', stats={},
            name='', targets=(),
        )

    @bridge(targets=('y',))
    def b2(record: Mapping[str, object]) -> BridgeResult:
        del record
        return BridgeResult(
            verdict=Verdict.HELD, reason='', stats={},
            name='', targets=(),
        )

    h1: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='h', intervention={}, bridges=(b1, b2),
    )
    h2: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='h', intervention={}, bridges=(b2, b1),
    )
    assert h1.mechanism_key == h2.mechanism_key


def test_mechanism_key_intervention_only_drops_bridge_names() -> None:
    """`MechanismKey.intervention_only()` projects to
    `InterventionKey` — preserves intervention_signature and
    direction, drops bridge_names. Two MechanismKeys with same
    intervention but different bridge sets project to the same
    InterventionKey (the projection causal discovery wants for
    binary intervention variables)."""
    from corroborate.hypothesis import InterventionKey, MechanismKey

    mk_a = MechanismKey(
        intervention_signature=(('slot', 'value'),),
        bridge_names=frozenset({'bridge_x', 'bridge_y'}),
        direction='a_gt_b',
    )
    mk_b = MechanismKey(
        intervention_signature=(('slot', 'value'),),
        bridge_names=frozenset({'completely', 'different'}),
        direction='a_gt_b',
    )

    iv_a = mk_a.intervention_only()
    iv_b = mk_b.intervention_only()
    assert isinstance(iv_a, InterventionKey)
    assert iv_a == iv_b


def test_mechanism_key_intervention_only_preserves_direction() -> None:
    from corroborate.hypothesis import MechanismKey

    mk = MechanismKey(
        intervention_signature=(('slot', 'value'),),
        bridge_names=frozenset(),
        direction='a_lt_b',
    )
    iv = mk.intervention_only()
    assert iv.direction == 'a_lt_b'
    assert iv.intervention_signature == (('slot', 'value'),)


def test_mechanism_key_intervention_only_distinguishes_interventions() -> None:
    from corroborate.hypothesis import MechanismKey

    mk_x = MechanismKey(
        intervention_signature=(('slot', 'value_x'),),
        bridge_names=frozenset(),
        direction=None,
    )
    mk_y = MechanismKey(
        intervention_signature=(('slot', 'value_y'),),
        bridge_names=frozenset(),
        direction=None,
    )
    assert mk_x.intervention_only() != mk_y.intervention_only()


def test_mechanism_key_is_hashable() -> None:
    """MechanismKey is hashable so it can be used as a dict key
    or set member — anti-laundering registry uses it that way."""
    h: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='h', intervention={'a': 1}, predicted_direction='a_gt_b',
    )
    keys: set[MechanismKey] = {h.mechanism_key}
    assert h.mechanism_key in keys


# ============ Intervention value canonicalization ============

def test_canonical_str_distinguishes_value_types() -> None:
    """Intervention values of different types canonicalise
    distinctly so a 'value=1' (int) doesn't collide with 'value=1'
    (str)."""
    h_int: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='h', intervention={'v': 1},
    )
    h_str: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='h', intervention={'v': '1'},
    )
    assert h_int.mechanism_key != h_str.mechanism_key


def test_canonical_str_handles_claims() -> None:
    """Claim-typed intervention values canonicalise via
    `Claim:<name>` so two interventions referencing the same Claim
    by name produce the same key."""
    @claim
    def my_alternative(x: int) -> int:
        return x * 2

    h1: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='h', intervention={'mechanism': my_alternative},
    )
    h2: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='h2', intervention={'mechanism': my_alternative},
    )
    assert h1.mechanism_key == h2.mechanism_key


def test_canonical_str_distinguishes_different_claims() -> None:
    @claim
    def alt_a(x: int) -> int:
        return x

    @claim
    def alt_b(x: int) -> int:
        return x

    h_a: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='h', intervention={'mechanism': alt_a},
    )
    h_b: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='h', intervention={'mechanism': alt_b},
    )
    assert h_a.mechanism_key != h_b.mechanism_key


def test_canonical_str_handles_plain_callables() -> None:
    """Plain (non-@claim) callables are canonicalised via their
    `__name__` when available, falling through to `repr()`. Allows
    interventions that haven't been promoted to @claim yet to still
    be structurally distinguishable."""
    def plain_alternative(x: int) -> int:
        return x

    h: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='h', intervention={'mechanism': plain_alternative},
    )
    sig = h.mechanism_key.intervention_signature
    assert any('plain_alternative' in pair[1] for pair in sig)


def test_canonical_str_bool_distinct_from_int() -> None:
    """bool is a subclass of int in Python; the canonicaliser
    handles them separately so True/False don't collide with
    1/0."""
    h_true: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='h', intervention={'flag': True},
    )
    h_one: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='h', intervention={'flag': 1},
    )
    assert h_true.mechanism_key != h_one.mechanism_key
