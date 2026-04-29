"""Tests for `Hypothesis[R]` and leaf-value canonicalization.
Strict typing exercised: bridges and hypothesis share R.

`MechanismKey` no longer exists as a framework artifact; the
configurational identity of a hypothesis is recovered from its
runs' `measurements` via `aggregate.leaf_signature`. These tests
cover the data-class shape + the `canonical_str` helper used for
leaf-value serialization."""
from __future__ import annotations

from collections.abc import Mapping

from corroborate.bridge import BridgeResult, bridge
from corroborate._canonical import canonical_str
from corroborate.claim import claim
from corroborate.hypothesis import Hypothesis
from corroborate.intervention import Intervention
from corroborate.verdict import Verdict


# ============ Construction ============

def test_hypothesis_minimal() -> None:
    h: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='baseline', intervention={},
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


# ============ edges + mechanism_edge accessor (E2) ============

def test_hypothesis_default_edges_empty() -> None:
    """Backward-compat: a Hypothesis without typed edges has an
    empty `edges` tuple. The flat `bridges` tuple stays usable."""
    h: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='legacy', intervention={},
    )
    assert h.edges == ()
    assert h.mechanism_edge() is None


def test_hypothesis_mechanism_edge_returns_role_mechanism() -> None:
    """`mechanism_edge()` returns the first edge with role
    'mechanism' or None if none declared."""
    from corroborate.bridge import BridgeResult, bridge as bridge_decorator
    from corroborate.claimed_edge import (
        link_edge,
        mechanism_edge,
        outcome_edge,
    )

    @bridge_decorator(targets=('m',))
    def _stub_m(record: Mapping[str, object]) -> BridgeResult:
        del record
        return BridgeResult(
            verdict=Verdict.HELD, reason='', stats={},
            name='_stub_m', targets=('m',),
        )

    @bridge_decorator(targets=('o',))
    def _stub_o(record: Mapping[str, object]) -> BridgeResult:
        del record
        return BridgeResult(
            verdict=Verdict.HELD, reason='', stats={},
            name='_stub_o', targets=('o',),
        )

    @bridge_decorator(targets=('o',))
    def _stub_l(record: Mapping[str, object]) -> BridgeResult:
        del record
        return BridgeResult(
            verdict=Verdict.HELD, reason='', stats={},
            name='_stub_l', targets=('o',),
        )

    h: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='ddqn', intervention={},
        edges=(
            outcome_edge(
                target='o', predicted_direction='a_gt_b',
                bridge=_stub_o,
            ),
            mechanism_edge(
                target='m', predicted_direction='a_lt_b',
                bridge=_stub_m,
            ),
            link_edge(
                source='m', target='o',
                predicted_direction='a_gt_b', bridge=_stub_l,
            ),
        ),
    )
    me = h.mechanism_edge()
    assert me is not None
    assert me.role == 'mechanism'
    assert me.target == 'm'
    assert me.predicted_direction == 'a_lt_b'


def test_hypothesis_edges_by_role() -> None:
    """`edges_by_role` returns all edges with the given role."""
    from corroborate.bridge import BridgeResult, bridge as bridge_decorator
    from corroborate.claimed_edge import outcome_edge

    @bridge_decorator(targets=('o1',))
    def _b1(record: Mapping[str, object]) -> BridgeResult:
        del record
        return BridgeResult(
            verdict=Verdict.HELD, reason='', stats={},
            name='_b1', targets=('o1',),
        )

    @bridge_decorator(targets=('o2',))
    def _b2(record: Mapping[str, object]) -> BridgeResult:
        del record
        return BridgeResult(
            verdict=Verdict.HELD, reason='', stats={},
            name='_b2', targets=('o2',),
        )

    h: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='multi', intervention={},
        edges=(
            outcome_edge(
                target='o1', predicted_direction='a_gt_b',
                bridge=_b1,
            ),
            outcome_edge(
                target='o2', predicted_direction='a_gt_b',
                bridge=_b2,
            ),
        ),
    )
    outs = h.edges_by_role('outcome')
    assert len(outs) == 2
    assert h.edges_by_role('mechanism') == ()


def test_hypothesis_edges_invariant_under_arm_key() -> None:
    """`arm_key` derives from `intervention_arms` only — adding
    or changing `edges` doesn't perturb arm identity."""
    from corroborate.bridge import BridgeResult, bridge as bridge_decorator
    from corroborate.claimed_edge import mechanism_edge

    @bridge_decorator(targets=('m',))
    def _b(record: Mapping[str, object]) -> BridgeResult:
        del record
        return BridgeResult(
            verdict=Verdict.HELD, reason='', stats={},
            name='_b', targets=('m',),
        )

    arms = (
        Intervention(slot_path='bootstrap', replacement=_alt_a),
    )
    h_no_edges: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='ddqn', intervention={}, intervention_arms=arms,
    )
    h_with_edges: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='ddqn', intervention={}, intervention_arms=arms,
        edges=(
            mechanism_edge(
                target='m', predicted_direction='a_lt_b', bridge=_b,
            ),
        ),
    )
    assert h_no_edges.arm_key() == h_with_edges.arm_key()
