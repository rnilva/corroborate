"""Tests for `Measurable[T]` and `@measurable`. Verifies T is
preserved through the typed wrapper end-to-end."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest

from corroborate.measurables import Measurable, measurable


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
    from corroborate.measurables import get_registered

    @measurable
    def _reg_demo_q(record: Mapping[str, object]) -> float:
        del record
        return 42.0

    looked_up = get_registered('_reg_demo_q')
    assert looked_up is _reg_demo_q


def test_register_idempotent_on_signature_equal_distinct_instances() -> None:
    """Two distinct `Measurable` instances with the same name +
    matching `signature()` must NOT raise — re-registration is
    accepted silently. Catches the cross-findings-module pattern
    where each module composes the same reduction by value
    (`reduce_axis(...)`) and gets a fresh Measurable instance with
    auto-name `mc_return__mean_axis_-1` on each call."""
    from corroborate.measurables import register
    from corroborate.measurables.measurable import Measurable, get_registered

    def _make() -> Measurable[Mapping[str, object], float]:
        # Same closure body, distinct identity each call.
        def _fn(record: Mapping[str, object]) -> float:
            del record
            return 1.0
        return Measurable(
            name='_idempotent_dup_test', reads=('x',), fn=_fn,
        )

    first = _make()
    second = _make()
    assert first is not second  # distinct instances
    assert first.signature() == second.signature()  # same closure hash

    register(first)
    register(second)  # must not raise

    # First-registered identity wins (no last-write-wins replacement).
    assert get_registered('_idempotent_dup_test') is first


def test_register_as_preserves_compose_of() -> None:
    """`register_as(m, name='alias')` constructs a renamed Measurable
    that threads `m.compose_of` through — the source lineage stays
    intact across the rename, so `signature()` recursion at
    `measurable.py:240-246` reaches the leaf through the aliased
    Measurable just as it would through the original."""
    from corroborate.measurables import (
        from_key,
        get_registered,
        mean_window,
        register_as,
    )

    source = from_key('_register_as_test_key')
    composed = mean_window(source, 0.5, 1.0)
    aliased = register_as(composed, name='_register_as_test_alias')

    assert aliased.name == '_register_as_test_alias'
    assert aliased.reads == composed.reads == ('_register_as_test_key',)
    # `compose_of` threaded through — the renamed Measurable's lineage
    # reaches back to the original `mean_window` operand.
    assert aliased.compose_of == composed.compose_of
    # Registered under the alias.
    assert get_registered('_register_as_test_alias') is aliased
    # Functional behavior preserved.
    record: Mapping[str, object] = {
        '_register_as_test_key': np.asarray([1.0, 2.0, 3.0, 4.0]),
    }
    assert aliased(record) == pytest.approx(3.5)  # mean of [3, 4]


def test_register_as_reads_override() -> None:
    """`register_as(m, name=..., reads=(...))` overrides the operand's
    reads when the persistence contract differs from the auto-derived
    set."""
    from corroborate.measurables import from_key, register_as

    source = from_key('_register_as_reads_override_key')
    aliased = register_as(
        source,
        name='_register_as_reads_override_alias',
        reads=('explicit_key',),
    )
    assert aliased.reads == ('explicit_key',)


def test_signature_flips_on_constant_change() -> None:
    """Constant-only edit must change the signature so the cache
    invalidates. Was a real bug pre-2026-05-06: `co_code` alone
    doesn't capture literal values (they live in `co_consts`)."""
    from corroborate.measurables.measurable import Measurable

    def _v1(record: Mapping[str, object]) -> float:
        del record
        return 0.5  # threshold v1

    def _v2(record: Mapping[str, object]) -> float:
        del record
        return 0.7  # threshold v2 — same opcodes, different const

    m1 = Measurable(name='_const_test', reads=(), fn=_v1)
    m2 = Measurable(name='_const_test', reads=(), fn=_v2)
    assert m1.signature() != m2.signature(), (
        'constant edit (0.5 → 0.7) must flip signature'
    )


def test_signature_flips_on_external_name_change() -> None:
    """Switching from `np.mean` to `np.nanmean` flips co_names but
    leaves co_code identical at the call site (LOAD_ATTR uses an
    index into co_names). Cache must invalidate."""
    import numpy as np
    from corroborate.measurables.measurable import Measurable

    def _uses_mean(record: Mapping[str, object]) -> float:
        x = record.get('x')
        return float(np.mean(x))  # type: ignore[arg-type]

    def _uses_nanmean(record: Mapping[str, object]) -> float:
        x = record.get('x')
        return float(np.nanmean(x))  # type: ignore[arg-type]

    m1 = Measurable(name='_name_test', reads=('x',), fn=_uses_mean)
    m2 = Measurable(name='_name_test', reads=('x',), fn=_uses_nanmean)
    assert m1.signature() != m2.signature(), (
        'np.mean → np.nanmean must flip signature'
    )


def test_signature_does_not_flip_on_local_var_rename() -> None:
    """Cosmetic local-var rename should NOT bust cache. `co_varnames`
    is deliberately excluded from the signature."""
    from corroborate.measurables.measurable import Measurable

    def _v1(record: Mapping[str, object]) -> float:
        x = record.get('count')
        return float(x or 0)  # type: ignore[arg-type]

    def _v2(record: Mapping[str, object]) -> float:
        n = record.get('count')  # renamed local x → n
        return float(n or 0)  # type: ignore[arg-type]

    m1 = Measurable(name='_rename_test', reads=('count',), fn=_v1)
    m2 = Measurable(name='_rename_test', reads=('count',), fn=_v2)
    assert m1.signature() == m2.signature(), (
        'local var rename should NOT flip signature'
    )


def test_signature_flips_on_constant_inside_nested_lambda() -> None:
    """Nested code objects (lambdas, comprehensions) live in
    `co_consts`; `_hash_code` recurses into them. A const edit
    inside a lambda body must propagate to the outer signature."""
    from corroborate.measurables.measurable import Measurable

    def _v1(record: Mapping[str, object]) -> float:
        xs = record.get('xs', ())
        return float(sum(map(lambda v: v * 2.0, xs)))  # type: ignore[arg-type]

    def _v2(record: Mapping[str, object]) -> float:
        xs = record.get('xs', ())
        return float(sum(map(lambda v: v * 3.0, xs)))  # type: ignore[arg-type]

    m1 = Measurable(name='_nested_test', reads=('xs',), fn=_v1)
    m2 = Measurable(name='_nested_test', reads=('xs',), fn=_v2)
    assert m1.signature() != m2.signature(), (
        'constant edit inside nested lambda (× 2.0 → × 3.0) '
        'must flip outer signature'
    )


def test_register_raises_on_signature_mismatch() -> None:
    """Different closures with the same name still raise — that's a
    real authoring conflict, NOT covered by the idempotency rule.
    Constant-only edits (`return 1.0` vs `return 2.0`) now flip
    the signature thanks to the `co_consts` term in `_hash_code`."""
    from corroborate.measurables import register
    from corroborate.measurables.measurable import Measurable

    def _fn_a(record: Mapping[str, object]) -> float:
        del record
        return 1.0

    def _fn_b(record: Mapping[str, object]) -> float:
        del record
        return 2.0  # constant-only edit

    a = Measurable(name='_conflict_test', reads=('x',), fn=_fn_a)
    b = Measurable(name='_conflict_test', reads=('x',), fn=_fn_b)
    assert a.signature() != b.signature()

    register(a)
    with pytest.raises(ValueError, match='already registered'):
        register(b)


def test_evaluate_with_measurables_no_deps() -> None:
    """A function with NO measurable params resolves trivially —
    the resolver passes the record through unchanged."""
    from corroborate.measurables import evaluate_with_measurables

    def f(record: Mapping[str, object]) -> str:
        return 'ok'

    assert evaluate_with_measurables(f, {}) == 'ok'


def test_evaluate_with_measurables_one_dep() -> None:
    """A function declaring one measurable as a parameter gets
    that measurable's value injected."""
    from corroborate.measurables import evaluate_with_measurables

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
    from corroborate.measurables import evaluate_with_measurables

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
    from corroborate.measurables import evaluate_with_measurables

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
    from corroborate.measurables import evaluate_with_measurables

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
    from corroborate.measurables import evaluate_with_measurables

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
