"""Tests for `Measurable[T]` and `@measurable`. Verifies T is
preserved through the typed wrapper end-to-end."""
from __future__ import annotations

from collections.abc import Mapping

from corroborate.bridge import Record
from corroborate.measurable import Measurable, measurable


# ============ Construction + basic call ============

def test_measurable_returns_measurable_instance() -> None:
    @measurable()
    def constant_one(record: Record) -> float:
        del record
        return 1.0

    assert isinstance(constant_one, Measurable)
    assert constant_one.name == 'constant_one'


def test_measurable_call_returns_native_T() -> None:
    @measurable()
    def count(record: Record) -> int:
        return len(record)

    record: Mapping[str, object] = {'a': 1, 'b': 2, 'c': 3}
    result = count(record)
    # Pyright sees `result: int`, not `object` — T is preserved.
    assert result == 3
    assert isinstance(result, int)


def test_measurable_preserves_diverse_return_types() -> None:
    @measurable()
    def as_float(record: Record) -> float:
        del record
        return 0.5

    @measurable()
    def as_str(record: Record) -> str:
        del record
        return 'hello'

    @measurable()
    def as_bool(record: Record) -> bool:
        del record
        return True

    record: Mapping[str, object] = {}
    assert as_float(record) == 0.5
    assert as_str(record) == 'hello'
    assert as_bool(record) is True


# ============ Name handling ============

def test_measurable_name_default_is_fn_name() -> None:
    @measurable()
    def some_specific_measurable(record: Record) -> int:
        del record
        return 0

    assert some_specific_measurable.name == 'some_specific_measurable'


def test_measurable_explicit_name_overrides() -> None:
    @measurable(name='custom_metric')
    def some_fn(record: Record) -> int:
        del record
        return 0

    assert some_fn.name == 'custom_metric'


def test_measurable_empty_string_name_kept() -> None:
    @measurable(name='')
    def some_fn(record: Record) -> int:
        del record
        return 0

    assert some_fn.name == ''


# ============ Factory pattern ============

def test_measurable_factory_pattern() -> None:
    """Factories returning Measurable[T] — the framework's pattern
    for parameterized measurables. Implemented with closures, no
    special framework support."""
    def late_window_mean(target: str) -> Measurable[float]:
        @measurable(name=f'{target}_late_mean')
        def fn(record: Record) -> float:
            v = record.get(target, 0.0)
            return v if isinstance(v, (int, float)) else 0.0

        return fn

    m1 = late_window_mean('q_max')
    m2 = late_window_mean('ep_return')

    assert m1.name == 'q_max_late_mean'
    assert m2.name == 'ep_return_late_mean'
    assert m1 != m2  # different fn closures


# ============ Realistic narrowing pattern ============

def test_record_value_narrowing() -> None:
    """Values from a Record are typed `object`; the measurable
    body narrows via `isinstance` (no cast) before using the
    value. This is the canonical narrowing pattern bridges and
    measurables follow."""
    @measurable(name='scalar_via_x')
    def from_x(record: Record) -> float:
        v = record['x']
        if isinstance(v, (int, float)):
            return float(v)
        return 0.0

    record: Mapping[str, object] = {'x': 5.0}
    assert from_x(record) == 5.0

    fallback: Mapping[str, object] = {'x': 'not-a-number'}
    assert from_x(fallback) == 0.0


# ============ Equality ============

def test_measurable_equality_by_field() -> None:
    def f(record: Record) -> int:
        del record
        return 0

    m1 = Measurable(fn=f, name='m')
    m2 = Measurable(fn=f, name='m')
    assert m1 == m2


def test_measurable_inequality_on_different_fn() -> None:
    def f1(record: Record) -> int:
        del record
        return 0

    def f2(record: Record) -> int:
        del record
        return 0

    m1 = Measurable(fn=f1, name='same')
    m2 = Measurable(fn=f2, name='same')
    assert m1 != m2
