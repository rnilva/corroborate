"""Variance probe — pyright must reject mixing `Bridge[R]` of
different `R` into the same `Hypothesis[R].bridges` tuple.

This test exercises the type system at compile time. Each block
that should be a pyright error is annotated with `# pyright:
ignore[<rule>]` to mark the *expected* error: if the rule
becomes unnecessary (i.e. pyright stops flagging the line),
basedpyright's `reportUnnecessaryTypeIgnoreComment` setting
fires and the test 'fails' at type-check time. So the gate
catches both:

- a future refactor accidentally relaxing variance (the line
  starts type-checking → reportUnnecessaryTypeIgnoreComment),
- an unrelated regression that breaks something else (any
  other type error fires).

Runtime-wise this file does nothing — pytest collects it as
no-op functions. The discipline is in basedpyright's pass."""
from __future__ import annotations

from collections.abc import Mapping
from typing import TypedDict

from corroborate.bridge import Bridge, BridgeResult, bridge
from corroborate.hypothesis import Hypothesis
from corroborate.verdict import Verdict


class _DQNRecord(TypedDict):
    final_return: float


class _DiffRecord(TypedDict):
    something_else: int


@bridge(targets=('final_return',))
def _dqn_bridge(record: _DQNRecord) -> BridgeResult:
    del record
    return BridgeResult(
        verdict=Verdict.HELD, reason='', stats={},
        name='', targets=(),
    )


@bridge(targets=('something_else',))
def _diff_bridge(record: _DiffRecord) -> BridgeResult:
    del record
    return BridgeResult(
        verdict=Verdict.HELD, reason='', stats={},
        name='', targets=(),
    )


@bridge(targets=('x',))
def _generic_bridge(record: Mapping[str, object]) -> BridgeResult:
    del record
    return BridgeResult(
        verdict=Verdict.HELD, reason='', stats={},
        name='', targets=(),
    )


def test_homogeneous_bridges_compose() -> None:
    """A Hypothesis[R] with all bridges typed against the same
    R type-checks cleanly. The negative cases (mixing R types)
    are below — those should fail pyright."""
    h_dqn: Hypothesis[_DQNRecord] = Hypothesis(
        name='dqn',
        intervention={},
        bridges=(_dqn_bridge,),
    )
    h_generic: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='generic',
        intervention={},
        bridges=(_generic_bridge,),
    )
    assert h_dqn.name == 'dqn'
    assert h_generic.name == 'generic'


def test_mixed_bridges_rejected_at_type_level() -> None:
    """Mixing Bridge[_DQNRecord] and Bridge[_DiffRecord] in a
    single Hypothesis tuple is a type error. Construct the
    offending tuple as a typed local and assign — the line
    that fails type-check carries the ignore comment.

    If the variance discipline is silently relaxed, the line
    will start type-checking and the
    `reportUnnecessaryTypeIgnoreComment` rule will fire."""
    bad_tuple: tuple[Bridge[_DQNRecord], Bridge[_DiffRecord]] = (
        _dqn_bridge,
        _diff_bridge,
    )
    h: Hypothesis[_DQNRecord] = Hypothesis(
        name='mixed',
        intervention={},
        bridges=bad_tuple,  # pyright: ignore[reportArgumentType]
    )
    assert h.name == 'mixed'


def test_widened_hypothesis_not_assignable_to_specific() -> None:
    """A Hypothesis[Mapping[str, object]] cannot be assigned to
    Hypothesis[_DQNRecord] — Hypothesis is invariant in R.
    Bridge typed for the wider signature would accept records
    missing DQN's required keys; pyright correctly rejects."""
    h_wide: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='widened',
        intervention={},
        bridges=(_generic_bridge,),
    )
    h_narrow: Hypothesis[_DQNRecord] = h_wide  # pyright: ignore[reportAssignmentType]
    assert h_narrow.name == 'widened'
