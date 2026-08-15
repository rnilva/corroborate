"""Tests for the `Finding` Protocol + `composed_verdict` + the
`_validate_hypothesis` subset check that enforces
`Finding.BRIDGES ⊆ Hypothesis.BRIDGES`.

Parallel coverage to `test_hypothesis.py`: the `runtime_checkable`
Protocol shape for findings, and the runtime-composed verdict
semantics over post-evaluated graphs.

`composed_verdict` ignores edge-shape (cluster vs envelope vs
chain) by design — the test set exercises shape-invariance via
fixtures that present each shape and verify the same admit/
refute/underpowered semantics across all of them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import cast, ClassVar

import pytest

from corroborate.bridge.bridge import Bridge, claim_bridge
from corroborate.bridge.verdict import Verdict
from corroborate.core.claim import claim
from corroborate.core.finding import Finding
from corroborate.core.intervention import DoEffect, Intervention
from corroborate.graph.causal import (
    stable_extent_hash,
    BridgeEdge, ClusterVerdict, Direction, Tier,
    composed_verdict, evaluated_graph,
)
from corroborate.graph.graph import Graph
from corroborate.runner.runner import _validate_hypothesis


# ============ Helpers: build typed bridges via @claim_bridge ============


@claim
def _alt(x: int) -> int:
    return x


_DOEFFECT = DoEffect(arms=(
    (),
    (Intervention(slot_path='op', replacement=_alt),),
))


@claim_bridge(source='m_source', target='m_target')
def _bridge_a(
    *, source: str = 'm_source', target: str = 'm_target',
    direction: Direction = Direction.DIRECT, tier: Tier = Tier.ASSOCIATIONAL,
) -> Verdict:
    del source, target, direction, tier
    return Verdict.HELD


@claim_bridge(source='m_source', target='m_target')
def _bridge_b(
    *, source: str = 'm_source', target: str = 'm_target',
    direction: Direction = Direction.DIRECT, tier: Tier = Tier.ASSOCIATIONAL,
) -> Verdict:
    del source, target, direction, tier
    return Verdict.HELD


@claim_bridge(source='m_source', target='m_other')
def _bridge_c(
    *, source: str = 'm_source', target: str = 'm_other',
    direction: Direction = Direction.DIRECT, tier: Tier = Tier.ASSOCIATIONAL,
) -> Verdict:
    del source, target, direction, tier
    return Verdict.HELD


# ============ Finding Protocol conformance ============


def test_class_with_classvars_satisfies_finding_protocol() -> None:
    """A class with `EXPECTED`, `BRIDGES`, `BLOCKED_ON`, `__name__`
    structurally satisfies the `runtime_checkable` Protocol."""
    @dataclass(frozen=True)
    class MyFinding:
        EXPECTED: ClassVar[ClusterVerdict] = ClusterVerdict.SUPPORTED
        BRIDGES: ClassVar[tuple[Bridge, ...]] = (_bridge_a, _bridge_b)
        BLOCKED_ON: ClassVar[str | None] = None
    assert isinstance(MyFinding, Finding)


def test_class_missing_expected_is_not_finding() -> None:
    @dataclass(frozen=True)
    class Broken:
        BRIDGES: ClassVar[tuple[Bridge, ...]] = ()
        BLOCKED_ON: ClassVar[str | None] = None
    assert not isinstance(Broken, Finding)


def test_class_missing_bridges_is_not_finding() -> None:
    @dataclass(frozen=True)
    class Broken:
        EXPECTED: ClassVar[ClusterVerdict] = ClusterVerdict.SUPPORTED
        BLOCKED_ON: ClassVar[str | None] = None
    assert not isinstance(Broken, Finding)


def test_class_missing_blocked_on_is_not_finding() -> None:
    @dataclass(frozen=True)
    class Broken:
        EXPECTED: ClassVar[ClusterVerdict] = ClusterVerdict.SUPPORTED
        BRIDGES: ClassVar[tuple[Bridge, ...]] = ()
    assert not isinstance(Broken, Finding)


def test_object_missing_name_is_not_finding() -> None:
    """`__name__` is a required Protocol attribute. Modules + classes
    carry it for free; plain instances lacking it fail conformance."""
    @dataclass(frozen=True)
    class NoName:
        EXPECTED: ClassVar[ClusterVerdict] = ClusterVerdict.SUPPORTED
        BRIDGES: ClassVar[tuple[Bridge, ...]] = ()
        BLOCKED_ON: ClassVar[str | None] = None
    instance = NoName()
    # Class HAS __name__; instance doesn't. Both are reachable from
    # the substrate (modules and class objects). Instances aren't.
    assert isinstance(NoName, Finding)
    assert not isinstance(instance, Finding)


# ============ composed_verdict over a post-eval graph ============


def _evaluated_with(
    bridges: tuple[Bridge, ...],
    verdicts: dict[str, tuple[Verdict, int]],
) -> Graph[str, BridgeEdge]:
    """Build a post-eval graph by stamping each bridge with the
    given (Verdict, extent_hash) pair via `evaluated_graph`."""
    from corroborate.graph.causal import PostEvalEntry
    post = {
        name: PostEvalEntry(verdict=v, extent_hash=h)
        for name, (v, h) in verdicts.items()
    }
    return evaluated_graph(bridges, post)


def test_composed_verdict_all_admit_supported() -> None:
    """All members admit (HELD on shared extent) → SUPPORTED."""
    g = _evaluated_with(
        (_bridge_a, _bridge_b),
        {'_bridge_a': (Verdict.HELD, 42), '_bridge_b': (Verdict.HELD, 42)},
    )
    assert composed_verdict(g, bridges=(_bridge_a, _bridge_b)) is (
        ClusterVerdict.SUPPORTED
    )


def test_composed_verdict_any_refuted_refuted() -> None:
    """Any member with `NO_EFFECT` → REFUTED."""
    g = _evaluated_with(
        (_bridge_a, _bridge_b),
        {
            '_bridge_a': (Verdict.HELD, 42),
            '_bridge_b': (Verdict.NO_EFFECT, 42),
        },
    )
    assert composed_verdict(g, bridges=(_bridge_a, _bridge_b)) is (
        ClusterVerdict.REFUTED
    )


def test_composed_verdict_mix_admit_unevaluated_underpowered() -> None:
    """Admit + unevaluated mix (e.g. POWER_INSUFFICIENT) →
    UNDERPOWERED."""
    g = _evaluated_with(
        (_bridge_a, _bridge_b),
        {
            '_bridge_a': (Verdict.HELD, 42),
            '_bridge_b': (Verdict.POWER_INSUFFICIENT, 42),
        },
    )
    assert composed_verdict(g, bridges=(_bridge_a, _bridge_b)) is (
        ClusterVerdict.UNDERPOWERED
    )


def test_composed_verdict_all_empty_extent_empty_extent() -> None:
    """All members admit zero cells (extent_hash == stable_extent_hash(()))
    → EMPTY_EXTENT. The corpus can't distinguish them."""
    empty = stable_extent_hash(())
    g = _evaluated_with(
        (_bridge_a, _bridge_b),
        {
            '_bridge_a': (Verdict.POWER_INSUFFICIENT, empty),
            '_bridge_b': (Verdict.POWER_INSUFFICIENT, empty),
        },
    )
    assert composed_verdict(g, bridges=(_bridge_a, _bridge_b)) is (
        ClusterVerdict.EMPTY_EXTENT
    )


def test_composed_verdict_envelope_shape_admits() -> None:
    """Envelope (same source/target, different extents) composes
    identically to a cluster — all admit → SUPPORTED.
    `composed_verdict` is shape-invariant by design."""
    g = _evaluated_with(
        (_bridge_a, _bridge_b),
        {
            '_bridge_a': (Verdict.HELD, 11),
            '_bridge_b': (Verdict.HELD, 22),  # distinct extent
        },
    )
    assert composed_verdict(g, bridges=(_bridge_a, _bridge_b)) is (
        ClusterVerdict.SUPPORTED
    )


def test_composed_verdict_missing_bridge_raises_assertion() -> None:
    """If a declared bridge isn't in the graph, `composed_verdict`
    raises `AssertionError` — the impossible state.
    `_validate_hypothesis` prevents this from firing in production
    by enforcing `Finding.BRIDGES ⊆ Hypothesis.BRIDGES` at
    startup. This test pins the loud failure that should fire if
    the validator is bypassed or the graph is corrupted."""
    g = _evaluated_with(
        (_bridge_a,),
        {'_bridge_a': (Verdict.HELD, 42)},
    )
    # Ask for _bridge_b too, which isn't in the graph.
    with pytest.raises(AssertionError, match='_bridge_b'):
        composed_verdict(g, bridges=(_bridge_a, _bridge_b))


# ============ _validate_hypothesis subset invariant ============


def test_validate_accepts_finding_bridges_subset() -> None:
    """A Hypothesis whose Findings cite only bridges from its own
    BRIDGES tuple validates successfully."""
    @dataclass(frozen=True)
    class GoodFinding:
        EXPECTED: ClassVar[ClusterVerdict] = ClusterVerdict.SUPPORTED
        BRIDGES: ClassVar[tuple[Bridge, ...]] = (_bridge_a,)
        BLOCKED_ON: ClassVar[str | None] = None

    @dataclass(frozen=True)
    class H:
        INTERVENTION: ClassVar[DoEffect] = _DOEFFECT
        BRIDGES: ClassVar[tuple[Bridge, ...]] = (_bridge_a, _bridge_b)
        FINDINGS: ClassVar[tuple[Finding, ...]] = (
            cast(Finding, GoodFinding),  # class satisfies the Protocol structurally
        )
    out = _validate_hypothesis(H)
    assert out is H


def test_validate_rejects_finding_with_bridge_not_in_parent() -> None:
    """Finding citing a bridge not in Hypothesis.BRIDGES → TypeError
    at validation time. Programming-error catchable at startup,
    not surfaced as a runtime verdict."""
    @dataclass(frozen=True)
    class BadFinding:
        EXPECTED: ClassVar[ClusterVerdict] = ClusterVerdict.SUPPORTED
        # Cites _bridge_c, which isn't in the parent's BRIDGES.
        BRIDGES: ClassVar[tuple[Bridge, ...]] = (_bridge_a, _bridge_c)
        BLOCKED_ON: ClassVar[str | None] = None

    @dataclass(frozen=True)
    class H:
        INTERVENTION: ClassVar[DoEffect] = _DOEFFECT
        BRIDGES: ClassVar[tuple[Bridge, ...]] = (_bridge_a, _bridge_b)
        FINDINGS: ClassVar[tuple[Finding, ...]] = (
            cast(Finding, BadFinding),  # class satisfies the Protocol structurally
        )
    with pytest.raises(TypeError, match='not in.*BRIDGES'):
        _validate_hypothesis(H)


def test_validate_empty_findings_accepts() -> None:
    """A Hypothesis with FINDINGS=() validates trivially (the
    subset loop is no-op)."""
    @dataclass(frozen=True)
    class H:
        INTERVENTION: ClassVar[DoEffect] = _DOEFFECT
        BRIDGES: ClassVar[tuple[Bridge, ...]] = ()
        FINDINGS: ClassVar[tuple[Finding, ...]] = ()
    assert _validate_hypothesis(H) is H


def test_validate_rejects_blocked_on_with_terminal_expected() -> None:
    """Author contradiction: `BLOCKED_ON` non-None paired with a
    terminal `EXPECTED` (SUPPORTED / REFUTED) is a hard invariant
    violation. `_validate_hypothesis` rejects at startup so
    operators can't accidentally ship a finding where they forgot
    to clear BLOCKED_ON after data landed."""
    @dataclass(frozen=True)
    class ContradictoryFinding:
        EXPECTED: ClassVar[ClusterVerdict] = ClusterVerdict.SUPPORTED
        BRIDGES: ClassVar[tuple[Bridge, ...]] = (_bridge_a,)
        BLOCKED_ON: ClassVar[str | None] = 'stale gap note'

    @dataclass(frozen=True)
    class H:
        INTERVENTION: ClassVar[DoEffect] = _DOEFFECT
        BRIDGES: ClassVar[tuple[Bridge, ...]] = (_bridge_a,)
        FINDINGS: ClassVar[tuple[Finding, ...]] = (
            cast(Finding, ContradictoryFinding),  # class satisfies the Protocol structurally
        )
    with pytest.raises(TypeError, match='BLOCKED_ON'):
        _validate_hypothesis(H)


def test_validate_accepts_blocked_on_with_non_terminal_expected() -> None:
    """`BLOCKED_ON` paired with UNDERPOWERED or EMPTY_EXTENT is
    legitimate — the empirical state is pinned to a non-terminal
    verdict pending data. Validator accepts."""
    @dataclass(frozen=True)
    class PendingFinding:
        EXPECTED: ClassVar[ClusterVerdict] = ClusterVerdict.UNDERPOWERED
        BRIDGES: ClassVar[tuple[Bridge, ...]] = (_bridge_a,)
        BLOCKED_ON: ClassVar[str | None] = 'waiting for n=240 cells'

    @dataclass(frozen=True)
    class H:
        INTERVENTION: ClassVar[DoEffect] = _DOEFFECT
        BRIDGES: ClassVar[tuple[Bridge, ...]] = (_bridge_a,)
        FINDINGS: ClassVar[tuple[Finding, ...]] = (
            cast(Finding, PendingFinding),  # class satisfies the Protocol structurally
        )
    assert _validate_hypothesis(H) is H
