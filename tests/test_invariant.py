"""Tests for `@invariant(of=claim, targets=...)` — tautological-
tagged bridges attached to Claims (axiom 18) — and the `bounded`
theorem-condition invariant factory."""
from __future__ import annotations

from collections.abc import Mapping

from corroborate.bridge import Bridge, BridgeResult
from corroborate.claim import claim
from corroborate.invariant import at_most, invariant
from corroborate.measurable import Measurable
from corroborate.verdict import Verdict


# ============ Decoration produces a Bridge ============

def test_invariant_returns_bridge() -> None:
    @claim
    def some_claim(x: int) -> int:
        return x

    @invariant(of=some_claim, targets=('q_max',))
    def q_bounded(record: Mapping[str, object]) -> BridgeResult:
        return BridgeResult(
            verdict=Verdict.HELD, reason='|q|<1000',
            stats={}, name='', targets=(),
        )

    assert isinstance(q_bounded, Bridge)
    assert q_bounded.targets == ('q_max',)


def test_invariant_default_name_includes_claim_name() -> None:
    @claim
    def my_step(x: int) -> int:
        return x

    @invariant(of=my_step, targets=('x',))
    def my_test(record: Mapping[str, object]) -> BridgeResult:
        return BridgeResult(
            verdict=Verdict.HELD, reason='', stats={},
            name='', targets=(),
        )

    assert 'my_test' in my_test.name
    assert 'my_step' in my_test.name


def test_invariant_explicit_name_overrides() -> None:
    @claim
    def some_claim(x: int) -> int:
        return x

    @invariant(of=some_claim, targets=('x',), name='custom_inv')
    def fn(record: Mapping[str, object]) -> BridgeResult:
        return BridgeResult(
            verdict=Verdict.HELD, reason='', stats={},
            name='', targets=(),
        )

    assert fn.name == 'custom_inv'


# ============ Tag injection ============

def test_invariant_injects_tautological_kind() -> None:
    """The wrapper auto-tags every returned BridgeResult's stats
    with `kind='tautological'`. The author does not have to set
    it (and existing stats are preserved)."""
    @claim
    def step(x: int) -> int:
        return x

    @invariant(of=step, targets=('x',))
    def inv(record: Mapping[str, object]) -> BridgeResult:
        return BridgeResult(
            verdict=Verdict.HELD, reason='ok',
            stats={'value': 0.5, 'threshold': 1.0},
            name='', targets=(),
        )

    record: Mapping[str, object] = {'x': 0}
    result = inv(record)
    assert result.stats['kind'] == 'tautological'
    assert result.stats['value'] == 0.5
    assert result.stats['threshold'] == 1.0


def test_invariant_injects_of_claim_name() -> None:
    @claim
    def my_specific_claim(x: int) -> int:
        return x

    @invariant(of=my_specific_claim, targets=('x',))
    def inv(record: Mapping[str, object]) -> BridgeResult:
        return BridgeResult(
            verdict=Verdict.HELD, reason='', stats={},
            name='', targets=(),
        )

    record: Mapping[str, object] = {'x': 0}
    result = inv(record)
    assert result.stats['of_claim'] == 'my_specific_claim'


def test_invariant_tags_added_even_on_reject() -> None:
    """Whether HELD or REJECT, the invariant tag is added — that's
    how aggregate_verdict will distinguish 'invariant violation'
    from 'normal refutation' downstream."""
    @claim
    def step(x: int) -> int:
        return x

    @invariant(of=step, targets=('x',))
    def inv(record: Mapping[str, object]) -> BridgeResult:
        return BridgeResult(
            verdict=Verdict.NO_EFFECT, reason='violated', stats={},
            name='', targets=(),
        )

    record: Mapping[str, object] = {'x': 0}
    result = inv(record)
    assert result.verdict is Verdict.NO_EFFECT
    assert result.stats['kind'] == 'tautological'


# ============ Stats preservation ============

def test_invariant_preserves_existing_stats() -> None:
    @claim
    def step(x: int) -> int:
        return x

    @invariant(of=step, targets=('x',))
    def inv(record: Mapping[str, object]) -> BridgeResult:
        return BridgeResult(
            verdict=Verdict.HELD, reason='',
            stats={'a': 1, 'b': 'foo', 'c': True, 'd': 3.14},
            name='', targets=(),
        )

    record: Mapping[str, object] = {'x': 0}
    result = inv(record)
    assert result.stats['a'] == 1
    assert result.stats['b'] == 'foo'
    assert result.stats['c'] is True
    assert result.stats['d'] == 3.14
    assert result.stats['kind'] == 'tautological'  # added on top


def test_invariant_uses_decorator_targets_when_inner_empty() -> None:
    """The wrapper backfills targets from the decorator if the
    inner BridgeResult left them empty — convenience for authors
    who don't bother setting them inside the body."""
    @claim
    def step(x: int) -> int:
        return x

    @invariant(of=step, targets=('q_max', 'epsilon'))
    def inv(record: Mapping[str, object]) -> BridgeResult:
        return BridgeResult(
            verdict=Verdict.HELD, reason='', stats={},
            name='', targets=(),  # empty — decorator backfills
        )

    record: Mapping[str, object] = {'x': 0}
    result = inv(record)
    assert result.targets == ('q_max', 'epsilon')


# ============ at_most(gap, threshold, of_claim) factory ============

# Helper: a tiny fake gap measurable returning record['v'] as
# float. Avoids depending on the full reductions stack here.
def _fake_gap(name: str = 'fake_gap') -> Measurable[Mapping[str, object], float]:
    def fn(record: Mapping[str, object]) -> float:
        v = record['v']
        return float(v) if isinstance(v, (int, float)) else 0.0
    return Measurable(fn=fn, name=name, reads=('v',))


def test_at_most_returns_bridge_with_gap_reads_as_targets() -> None:
    @claim
    def some_step(x: int) -> int:
        return x

    inv = at_most(_fake_gap(), threshold=1.0, of_claim=some_step)
    assert isinstance(inv, Bridge)
    assert inv.targets == ('v',)


def test_at_most_held_when_gap_under_threshold() -> None:
    @claim
    def some_step(x: int) -> int:
        return x

    inv = at_most(_fake_gap(), threshold=1.0, of_claim=some_step)
    record: Mapping[str, object] = {'v': 0.5}
    result = inv(record)
    assert result.verdict is Verdict.HELD
    assert result.stats['kind'] == 'tautological'
    assert result.stats['of_claim'] == 'some_step'
    assert result.stats['gap_value'] == 0.5
    assert result.stats['threshold'] == 1.0


def test_at_most_invariant_violation_when_gap_over_threshold() -> None:
    @claim
    def some_step(x: int) -> int:
        return x

    inv = at_most(_fake_gap(), threshold=1.0, of_claim=some_step)
    record: Mapping[str, object] = {'v': 5.0}
    result = inv(record)
    # gap = 5 > 1 → theorem out of scope → INVARIANT_VIOLATION
    assert result.verdict is Verdict.INVARIANT_VIOLATION
    assert result.stats['kind'] == 'tautological'
    assert result.stats['gap_value'] == 5.0


def test_at_most_held_at_exact_threshold() -> None:
    """gap == threshold counts as HELD (`<=`, not strict `<`)."""
    @claim
    def some_step(x: int) -> int:
        return x

    inv = at_most(_fake_gap(), threshold=1.0, of_claim=some_step)
    record: Mapping[str, object] = {'v': 1.0}
    assert inv(record).verdict is Verdict.HELD


def test_at_most_power_insufficient_when_gap_is_nan() -> None:
    """NaN gap (no data — replay never filled, etc.) maps to
    POWER_INSUFFICIENT, NOT silent HELD. Treating NaN as HELD
    would be a false-confirmation of scope from a run that
    couldn't produce evidence."""
    @claim
    def some_step(x: int) -> int:
        return x

    inv = at_most(_fake_gap(), threshold=1.0, of_claim=some_step)
    record: Mapping[str, object] = {'v': float('nan')}
    result = inv(record)
    assert result.verdict is Verdict.POWER_INSUFFICIENT
    assert result.stats['kind'] == 'tautological'


def test_at_most_default_name_includes_gap_and_threshold() -> None:
    @claim
    def step(x: int) -> int:
        return x

    inv = at_most(_fake_gap('jensen_gap'), threshold=0.5, of_claim=step)
    assert 'jensen_gap' in inv.name
    assert '0.5' in inv.name


# ============ Discovery mode (threshold=None) ============

def test_at_most_discovery_mode_held_when_finite() -> None:
    """`threshold=None` is discovery mode: any finite gap_value
    yields HELD with `stats['gap_value']` recorded. Never
    INVARIANT_VIOLATION — the author hasn't committed scope yet."""
    @claim
    def some_step(x: int) -> int:
        return x

    inv = at_most(_fake_gap(), threshold=None, of_claim=some_step)
    record: Mapping[str, object] = {'v': 0.42}
    result = inv(record)
    assert result.verdict is Verdict.HELD
    assert result.stats['gap_value'] == 0.42
    assert result.stats['kind'] == 'tautological'


def test_at_most_discovery_mode_held_for_huge_finite_value() -> None:
    """In discovery mode, even a large gap_value yields HELD —
    there's no threshold yet to violate."""
    @claim
    def some_step(x: int) -> int:
        return x

    inv = at_most(_fake_gap(), threshold=None, of_claim=some_step)
    record: Mapping[str, object] = {'v': 1e9}
    result = inv(record)
    assert result.verdict is Verdict.HELD


def test_at_most_discovery_mode_power_insufficient_on_nan() -> None:
    """NaN still maps to POWER_INSUFFICIENT in discovery mode —
    we can't record a `gap_value` that doesn't exist. Treating
    NaN as HELD would silently record stale-data verdicts as
    confirmation."""
    @claim
    def some_step(x: int) -> int:
        return x

    inv = at_most(_fake_gap(), threshold=None, of_claim=some_step)
    record: Mapping[str, object] = {'v': float('nan')}
    result = inv(record)
    assert result.verdict is Verdict.POWER_INSUFFICIENT


def test_at_most_discovery_mode_default_threshold() -> None:
    """`threshold` defaults to None — discovery is the default
    regime, since for novel mechanisms the author rarely knows
    the threshold a priori."""
    @claim
    def some_step(x: int) -> int:
        return x

    inv = at_most(_fake_gap(), of_claim=some_step)
    record: Mapping[str, object] = {'v': 100.0}
    assert inv(record).verdict is Verdict.HELD


def test_at_most_discovery_name_uses_asterisk_marker() -> None:
    """The bridge name uses '*' to signal discovery mode — visible
    in stdout and in flattened parquet column names so a reader
    can tell at-a-glance whether scope was committed or not."""
    @claim
    def step(x: int) -> int:
        return x

    inv = at_most(_fake_gap('jensen_gap'), threshold=None, of_claim=step)
    assert 'jensen_gap' in inv.name
    assert '*' in inv.name
