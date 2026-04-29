"""Tests for `redundancy_check` — outcome-jaccard + HP-R² tautology
detection."""
from __future__ import annotations

import math
from collections.abc import Mapping

from corroborate.measurable import measurable
from corroborate.redundancy_check import (
    TautologyReport, audit_mediator_panel, is_hp_tautological,
    is_outcome_tautological, jaccard, reads_overlap,
)
from corroborate.schema import RunRow
from corroborate.verdict import Verdict


# ============ jaccard ============

def test_jaccard_full_overlap_is_one() -> None:
    assert jaccard(frozenset({'a', 'b'}), frozenset({'a', 'b'})) == 1.0


def test_jaccard_no_overlap_is_zero() -> None:
    assert jaccard(frozenset({'a'}), frozenset({'b'})) == 0.0


def test_jaccard_partial_overlap() -> None:
    """{a,b} ∩ {b,c} = {b}; {a,b} ∪ {b,c} = {a,b,c}; jaccard = 1/3."""
    j = jaccard(frozenset({'a', 'b'}), frozenset({'b', 'c'}))
    assert math.isclose(j, 1/3, rel_tol=1e-9)


def test_jaccard_both_empty_is_zero() -> None:
    """Vacuous overlap convention — empty sets aren't informative."""
    assert jaccard(frozenset(), frozenset()) == 0.0


# ============ reads_overlap (Measurable level) ============

def test_reads_overlap_identical_measurables() -> None:
    @measurable(reads=('mc_return', 'episode_length'))
    def a(record: Mapping[str, object]) -> float:
        del record
        return 1.0

    @measurable(reads=('mc_return', 'episode_length'))
    def b(record: Mapping[str, object]) -> float:
        del record
        return 2.0

    assert reads_overlap(a, b) == 1.0


def test_reads_overlap_disjoint() -> None:
    @measurable(reads=('mc_return',))
    def a(record: Mapping[str, object]) -> float:
        del record
        return 1.0

    @measurable(reads=('online_argmax',))
    def b(record: Mapping[str, object]) -> float:
        del record
        return 2.0

    assert reads_overlap(a, b) == 0.0


# ============ is_outcome_tautological ============

def test_outcome_tautological_when_full_overlap() -> None:
    @measurable(reads=('mc_return',))
    def m(record: Mapping[str, object]) -> float:
        del record
        return 1.0

    assert is_outcome_tautological(m, frozenset({'mc_return'}))


def test_outcome_tautological_when_disjoint() -> None:
    @measurable(reads=('online_argmax', 'target_argmax'))
    def m(record: Mapping[str, object]) -> float:
        del record
        return 1.0

    assert not is_outcome_tautological(m, frozenset({'mc_return'}))


def test_outcome_tautological_with_partial_overlap_below_threshold() -> None:
    """Mediator reads {mc_return, td_error}; outcome reads
    {mc_return}. Jaccard = 1/2 = 0.5 = threshold by default.
    Inclusive comparison flags it; below threshold it doesn't."""
    @measurable(reads=('mc_return', 'td_error'))
    def m(record: Mapping[str, object]) -> float:
        del record
        return 1.0

    # Default threshold = 0.5; jaccard = 1/2 → flagged.
    assert is_outcome_tautological(m, frozenset({'mc_return'}))
    # Stricter threshold → not flagged.
    assert not is_outcome_tautological(
        m, frozenset({'mc_return'}), threshold=0.6,
    )


# ============ is_hp_tautological ============

def test_hp_tautological_when_deterministic() -> None:
    """Mediator = 0.5 * HP exactly → R² = 1.0 → flagged."""
    hp = [10.0, 20.0, 30.0, 40.0, 50.0]
    mediator = [5.0, 10.0, 15.0, 20.0, 25.0]
    assert is_hp_tautological(mediator, hp)


def test_hp_tautological_when_independent() -> None:
    """Mediator uncorrelated with HP → low R² → not flagged."""
    hp = [10.0, 20.0, 30.0, 40.0, 50.0]
    mediator = [3.0, 1.0, 5.0, 2.0, 4.0]  # no relationship
    assert not is_hp_tautological(mediator, hp)


def test_hp_tautological_when_partially_correlated() -> None:
    """Strong but imperfect correlation — depends on threshold."""
    hp = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    # y = 0.5 * x + small noise → R² high but < 1.
    mediator = [5.1, 9.8, 15.2, 19.9, 25.3, 29.7]
    # Default threshold = 0.95; tight relationship may or may not flag.
    # Lower threshold should flag.
    assert is_hp_tautological(mediator, hp, threshold=0.9)


# ============ audit_mediator_panel ============

def _row(
    cell_id: str, *, capacity: int, batch_size: int,
    mediator_outcome_taut: float,
    mediator_hp_taut: float,
    mediator_clean: float,
) -> RunRow:
    return RunRow(
        id=cell_id, parent_id=None, cycle_id=None,
        timestamp='ts', verdict=Verdict.HELD, arm_key='baseline',
        measurements={
            'replay.capacity': capacity,
            'replay.batch_size': batch_size,
            'mediator.mc_return_based': mediator_outcome_taut,
            'mediator.deterministic_in_hp': mediator_hp_taut,
            'mediator.independent': mediator_clean,
        },
    )


def test_audit_panel_flags_outcome_tautological() -> None:
    """A measurable with reads={mc_return} is flagged when outcome
    reads also include mc_return."""
    @measurable(reads=('mc_return',))
    def mc_return_based(record: Mapping[str, object]) -> float:
        del record
        return 1.0

    @measurable(reads=('online_argmax',))
    def independent(record: Mapping[str, object]) -> float:
        del record
        return 1.0

    runs = [
        _row(f'c{i}', capacity=10000 + i * 1000, batch_size=32,
             mediator_outcome_taut=1.0, mediator_hp_taut=0.0,
             mediator_clean=float(i % 2))
        for i in range(8)
    ]
    reports = audit_mediator_panel(
        [mc_return_based, independent], runs,
        outcome_reads=frozenset({'mc_return'}),
        hp_axes=('replay.capacity', 'replay.batch_size'),
    )
    by_name = {r.measurable_name: r for r in reports}
    assert by_name['mc_return_based'].flagged_outcome is True
    assert by_name['independent'].flagged_outcome is False


def test_audit_panel_flags_hp_tautological() -> None:
    """A measurable whose value is f(capacity) gets HP-flagged on
    that axis, while a constant mediator doesn't (R² is undefined
    when x has no variance, but the mediator-on-HP regression is
    NaN there — falls through as not flagged)."""
    @measurable(reads=('online_argmax',))
    def deterministic_in_hp(record: Mapping[str, object]) -> float:
        del record
        return 1.0

    @measurable(reads=('online_argmax',))
    def independent(record: Mapping[str, object]) -> float:
        del record
        return 1.0

    # Per-cell mediator values: deterministic_in_hp = capacity / 2;
    # independent = pseudorandom independent of capacity.
    rng_vals = [3.2, 7.1, 4.5, 1.9, 8.3, 5.4, 2.7, 6.1]
    runs = []
    for i in range(8):
        cap = 10000 + i * 5000
        runs.append(_row(
            f'c{i}', capacity=cap, batch_size=32,
            mediator_outcome_taut=0.0,
            mediator_hp_taut=cap / 2,           # deterministic
            mediator_clean=rng_vals[i],          # independent
        ))
    reports = audit_mediator_panel(
        [deterministic_in_hp, independent], runs,
        outcome_reads=frozenset({'mc_return'}),
        hp_axes=('replay.capacity', 'replay.batch_size'),
        mediator_path_for={
            'deterministic_in_hp': 'mediator.deterministic_in_hp',
            'independent': 'mediator.independent',
        },
    )
    by_name = {r.measurable_name: r for r in reports}
    assert 'replay.capacity' in by_name['deterministic_in_hp'].flagged_hp
    assert 'replay.capacity' not in by_name['independent'].flagged_hp


def test_audit_panel_clean_property() -> None:
    """A measurable that's neither outcome- nor HP-tautological has
    `is_clean=True`."""
    @measurable(reads=('online_argmax',))
    def clean(record: Mapping[str, object]) -> float:
        del record
        return 1.0

    rng_vals = [3.2, 7.1, 4.5, 1.9, 8.3, 5.4, 2.7, 6.1]
    runs = []
    for i in range(8):
        cap = 10000 + i * 5000
        runs.append(_row(
            f'c{i}', capacity=cap, batch_size=32,
            mediator_outcome_taut=0.0, mediator_hp_taut=0.0,
            mediator_clean=rng_vals[i],
        ))
    reports = audit_mediator_panel(
        [clean], runs,
        outcome_reads=frozenset({'mc_return'}),
        hp_axes=('replay.capacity',),
        mediator_path_for={'clean': 'mediator.independent'},
    )
    assert len(reports) == 1
    r = reports[0]
    assert r.is_clean
    assert r.flagged_outcome is False
    assert r.flagged_hp == ()


def test_audit_panel_returns_typed_dataclass() -> None:
    @measurable(reads=('mc_return',))
    def m(record: Mapping[str, object]) -> float:
        del record
        return 1.0

    runs = [_row(
        'c0', capacity=10000, batch_size=32,
        mediator_outcome_taut=0.0, mediator_hp_taut=0.0,
        mediator_clean=0.0,
    )]
    reports = audit_mediator_panel(
        [m], runs, outcome_reads=frozenset({'mc_return'}),
        hp_axes=('replay.capacity',),
    )
    assert isinstance(reports[0], TautologyReport)
