"""Tests for `Measurable[T]` and `@measurable`. Verifies T is
preserved through the typed wrapper end-to-end."""
from __future__ import annotations

from collections.abc import Mapping

from corroborate.measurable import Measurable, measurable


# ============ Construction + basic call ============

def test_measurable_no_parens_form() -> None:
    """`@measurable` without parens is the canonical short form
    when no parameters are needed."""
    @measurable
    def constant_one(record: Mapping[str, object]) -> float:
        del record
        return 1.0

    assert isinstance(constant_one, Measurable)
    assert constant_one.name == 'constant_one'


def test_measurable_paren_form() -> None:
    """`@measurable()` with empty parens is also valid (same
    behavior as no parens; useful before adding `name=...`)."""
    @measurable()
    def some_metric(record: Mapping[str, object]) -> float:
        del record
        return 1.0

    assert isinstance(some_metric, Measurable)
    assert some_metric.name == 'some_metric'


def test_measurable_call_returns_native_T() -> None:
    @measurable()
    def count(record: Mapping[str, object]) -> int:
        return len(record)

    record: Mapping[str, object] = {'a': 1, 'b': 2, 'c': 3}
    result = count(record)
    # Pyright sees `result: int`, not `object` — T is preserved.
    assert result == 3
    assert isinstance(result, int)


def test_measurable_preserves_diverse_return_types() -> None:
    @measurable()
    def as_float(record: Mapping[str, object]) -> float:
        del record
        return 0.5

    @measurable()
    def as_str(record: Mapping[str, object]) -> str:
        del record
        return 'hello'

    @measurable()
    def as_bool(record: Mapping[str, object]) -> bool:
        del record
        return True

    record: Mapping[str, object] = {}
    assert as_float(record) == 0.5
    assert as_str(record) == 'hello'
    assert as_bool(record) is True


# ============ Name handling ============

def test_measurable_name_default_is_fn_name() -> None:
    @measurable()
    def some_specific_measurable(record: Mapping[str, object]) -> int:
        del record
        return 0

    assert some_specific_measurable.name == 'some_specific_measurable'


def test_measurable_explicit_name_overrides() -> None:
    @measurable(name='custom_metric')
    def some_fn(record: Mapping[str, object]) -> int:
        del record
        return 0

    assert some_fn.name == 'custom_metric'


def test_measurable_empty_string_name_kept() -> None:
    @measurable(name='')
    def some_fn(record: Mapping[str, object]) -> int:
        del record
        return 0

    assert some_fn.name == ''


# ============ Factory pattern ============

def test_measurable_factory_pattern() -> None:
    """Factories returning Measurable[T] — the framework's pattern
    for parameterized measurables. Implemented with closures, no
    special framework support."""
    def late_window_mean(target: str) -> Measurable[Mapping[str, object], float]:
        @measurable(name=f'{target}_late_mean')
        def fn(record: Mapping[str, object]) -> float:
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
    def from_x(record: Mapping[str, object]) -> float:
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
    def f(record: Mapping[str, object]) -> int:
        del record
        return 0

    m1 = Measurable(fn=f, name='m')
    m2 = Measurable(fn=f, name='m')
    assert m1 == m2


def test_measurable_inequality_on_different_fn() -> None:
    def f1(record: Mapping[str, object]) -> int:
        del record
        return 0

    def f2(record: Mapping[str, object]) -> int:
        del record
        return 0

    m1 = Measurable(fn=f1, name='same')
    m2 = Measurable(fn=f2, name='same')
    assert m1 != m2


# ============ reads field ============

def test_measurable_reads_default_empty() -> None:
    @measurable
    def m(record: Mapping[str, object]) -> int:
        del record
        return 0

    assert m.reads == ()


def test_measurable_reads_explicit() -> None:
    @measurable(name='leaf', reads=('q_max',))
    def m(record: Mapping[str, object]) -> int:
        del record
        return 0

    assert m.reads == ('q_max',)


def test_measurable_reads_via_direct_construction() -> None:
    def fn(record: Mapping[str, object]) -> int:
        del record
        return 0

    m: Measurable[Mapping[str, object], int] = Measurable(
        fn=fn, name='foo', reads=('a', 'b'),
    )
    assert m.reads == ('a', 'b')


# ============ Name-keyed registry + transitive resolution ============

def test_registry_indexes_by_name() -> None:
    """`@measurable` registers the instance in `_REGISTRY` keyed
    by the resolved name."""
    from corroborate.measurable import get_registered

    @measurable
    def _reg_demo_q(record: Mapping[str, object]) -> float:
        del record
        return 42.0

    looked_up = get_registered('_reg_demo_q')
    assert looked_up is _reg_demo_q


def test_evaluate_with_measurables_no_deps() -> None:
    """A function with NO measurable params resolves trivially —
    the resolver passes the record through unchanged."""
    from corroborate.measurable import evaluate_with_measurables

    def f(record: Mapping[str, object]) -> str:
        return 'ok'

    assert evaluate_with_measurables(f, {}) == 'ok'


def test_evaluate_with_measurables_one_dep() -> None:
    """A function declaring one measurable as a parameter gets
    that measurable's value injected."""
    from corroborate.measurable import evaluate_with_measurables

    @measurable
    def base_value(record: Mapping[str, object]) -> int:
        v = record['x']
        assert isinstance(v, int)
        return v * 2

    def consumer(record: Mapping[str, object], base_value: int) -> int:
        del record
        return base_value + 100

    out = evaluate_with_measurables(consumer, {'x': 5})
    assert out == 110  # 5*2 + 100


def test_evaluate_with_measurables_transitive() -> None:
    """Measurable A depends on measurable B; B depends on the
    record. Resolver walks the dep tree and computes B once,
    then A."""
    from corroborate.measurable import evaluate_with_measurables

    @measurable
    def _trans_x(record: Mapping[str, object]) -> int:
        v = record['raw']
        assert isinstance(v, int)
        return v + 1

    @measurable
    def _trans_y(record: Mapping[str, object], _trans_x: int) -> int:
        del record
        return _trans_x * 10

    def consumer(record: Mapping[str, object], _trans_y: int) -> int:
        del record
        return _trans_y + 1

    out = evaluate_with_measurables(consumer, {'raw': 4})
    # _trans_x = 5; _trans_y = 50; consumer = 51.
    assert out == 51


def test_evaluate_memoizes_within_one_record() -> None:
    """A measurable is computed exactly once per record even when
    multiple dependents read it. The cache is shared across all
    deps in one `evaluate_with_measurables` call."""
    from corroborate.measurable import evaluate_with_measurables

    call_count = {'n': 0}

    @measurable
    def _memo_base(record: Mapping[str, object]) -> int:
        call_count['n'] += 1
        v = record['x']
        assert isinstance(v, int)
        return v

    @measurable
    def _memo_a(record: Mapping[str, object], _memo_base: int) -> int:
        del record
        return _memo_base + 1

    @measurable
    def _memo_b(record: Mapping[str, object], _memo_base: int) -> int:
        del record
        return _memo_base + 2

    def consumer(
        record: Mapping[str, object],
        _memo_a: int, _memo_b: int,
    ) -> int:
        del record
        return _memo_a + _memo_b

    out = evaluate_with_measurables(consumer, {'x': 10})
    # _memo_a = 11; _memo_b = 12; sum = 23.
    assert out == 23
    # _memo_base ran exactly once despite both _memo_a and _memo_b
    # reading it.
    assert call_count['n'] == 1


def test_evaluate_unknown_measurable_param_passes_through() -> None:
    """A param name that isn't a registered measurable is left
    for the caller to supply. Python will raise TypeError if the
    caller doesn't pass it (normal function-call semantics)."""
    from corroborate.measurable import evaluate_with_measurables

    def f(record: Mapping[str, object], not_a_measurable: int) -> int:
        del record
        return not_a_measurable

    # Resolver does NOT inject not_a_measurable; standard call
    # raises TypeError for the missing positional/kw arg.
    try:
        evaluate_with_measurables(f, {})
        raise AssertionError('expected TypeError')
    except TypeError:
        pass


def test_unknown_dep_in_intermediate_raises_typeerror() -> None:
    """An intermediate measurable declaring a param NOT in the
    registry is left unfilled by the resolver. Python's normal
    call-site checking raises TypeError when the framework
    invokes it without that arg. Consistent with
    `evaluate_with_measurables` only auto-injecting registered
    names — unknowns pass through, the caller must supply or it
    errors."""
    from corroborate.measurable import evaluate_with_measurables

    @measurable
    def _ku_intermediate(record: Mapping[str, object], not_registered: int) -> int:
        del record, not_registered
        return 0

    def consumer(
        record: Mapping[str, object], _ku_intermediate: int,
    ) -> int:
        del record
        return _ku_intermediate

    try:
        evaluate_with_measurables(consumer, {})
        raise AssertionError('expected TypeError')
    except TypeError:
        pass
