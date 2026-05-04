"""Measurable — typed scalar derivation from a record.

A `Measurable[R, T]` is a named function `(record, **deps) -> T`
that produces a summary quantity (late-window mean, q_max,
td_error_norm, etc.) bridges or other measurables consume. Two
type parameters:

- `R: Mapping[str, object]` — the record schema.
- `T` — the scalar return type.

The framework's post-hoc analytical layer. Distinct from Claims
(steps in the algorithm) and Bridges (verdict-producing tests).

`@measurable` ALSO registers each instance in a name-keyed
registry. Bridges and other measurables declare dependencies by
parameter name; `evaluate_with_measurables(bridge_fn, record)`
inspects the function's signature, looks up each parameter in
the registry, computes once per record (memoized), and injects.
Pytest-fixture-style transitive resolution.

Two surface shapes coexist:

1. **Value-based composition** — pass `Measurable` instances
   around explicitly, call them on a record. The classical form
   for ad-hoc reductions.
2. **Name-based registry resolution** — declare a parameter
   matching a registered measurable's `name`; the framework
   injects the computed value. Useful for the `record →
   {q_mean, q_max, q_std, q_gap, ...}` reduction graph.

`reads` is the LEAF record keys this measurable ultimately
depends on — the union of its own direct reads and the
transitive reads of any measurable deps. Populated declaratively
when known; defaults to `()` otherwise. The framework's
reads-set discipline (axiom 19's redundancy primitive) consumes
this for fingerprinting."""
from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Mapping
from typing import cast, overload, override

import polars as pl

from corroborate._internals.registry import Registry


class Measurable[R: Mapping[str, object], T]:
    """Typed generic wrapper. Behaves as `Callable[..., T]` over
    `(record, **deps)`; carries `name` and `reads` as typed
    attributes. Both `R` (record type) and `T` (return type) are
    preserved through `__call__`.

    `reads` defaults to `()`. Reductions and the `from_key`
    primitive populate it explicitly so downstream invariant
    factories can derive `Bridge.targets` from it.

    The function may take ONLY a record (`fn(record) -> T`) — the
    classical shape — OR a record plus named parameters that match
    other registered measurables (`fn(record, q_mean, q_std) -> T`).
    Resolver finds and injects the dep values.

    `fallbacks` declare alternative-input siblings tried in order
    when the primary's `reads` are absent (typically: the
    pre-reduced form of a column dropped at persistence). Each
    fallback is itself a full Measurable with its OWN `name` —
    callers that need to record provenance (e.g. invariant
    persistence) `dispatch(record)` to pick whichever's reads are
    present and use that Measurable's `name`. A fallback CAN be
    an approximation (different reduction order, different
    sufficient statistic), so persisting under its own name keeps
    aliased observations distinct from primary ones rather than
    silently filing both into the canonical column.

    **Variance.** Field access is via `@property` (not bare
    dataclass attrs) so PEP 695 inference lands `R` contravariant
    and `T` covariant — the natural variance for "function reading
    R, returning T". A `Measurable[Mapping[str, object], float]`
    is therefore assignable to a slot expecting
    `Measurable[Mapping[str, NDArray], float]` (wider record
    acceptance is fine where narrower is expected) and a
    `Measurable[..., float]` to a slot expecting
    `Measurable[..., object]` (more specific T is fine where
    wider is expected). Frozen-dataclass form would invariance both
    parameters because dataclass-auto `__init__` puts fields in
    contravariant input position; the read-only property form is
    the variance-friendly shape."""

    __slots__ = (
        '_fn', '_name', '_reads', '_fallbacks', '_compose_of',
    )

    _fn: Callable[..., T]
    _name: str
    _reads: tuple[str, ...]
    _fallbacks: tuple['Measurable[R, T]', ...]
    _compose_of: tuple['Measurable[Mapping[str, object], object]', ...]

    def __init__(
        self,
        fn: Callable[..., T],
        name: str,
        reads: tuple[str, ...] = (),
        fallbacks: tuple['Measurable[R, T]', ...] = (),
        compose_of: tuple[
            'Measurable[Mapping[str, object], object]', ...,
        ] = (),
    ) -> None:
        self._fn = fn
        self._name = name
        self._reads = reads
        self._fallbacks = fallbacks
        # Closure-captured operand measurables. Reduction factories
        # (`reduce_axis(of, ...)`, `slice_axis(of, ...)`,
        # `mean_window(of, ...)`, etc.) pass `compose_of=(of,)` so
        # `signature()` can walk into the operand recursively. The
        # parameter-name dependency walk (`_measurable_param_names`)
        # only sees explicit-parameter deps; closures are invisible
        # to it. Without `compose_of`, editing the source measurable
        # of a composition wouldn't invalidate the composed
        # measurable's signature — a real cache-staleness bug.
        self._compose_of = compose_of

    @property
    def fn(self) -> Callable[..., T]:
        return self._fn

    @property
    def name(self) -> str:
        return self._name

    @property
    def reads(self) -> tuple[str, ...]:
        return self._reads

    @property
    def fallbacks(self) -> tuple['Measurable[R, T]', ...]:
        return self._fallbacks

    @property
    def compose_of(
        self,
    ) -> tuple['Measurable[Mapping[str, object], object]', ...]:
        """Closure-captured operand measurables (factory-recorded).
        `()` for plain `@measurable`-decorated leaves and explicit
        `Measurable(...)` constructions; non-empty for outputs of
        reduction factories (`reduce_axis(of, ...)`, etc.)."""
        return self._compose_of

    def __call__(self, record: R, **deps: object) -> T:
        return self.dispatch(record).fn(record, **deps)

    def signature(self) -> str:
        """Closure hash: SHA-256 of this Measurable's bytecode plus
        the sorted bytecode hashes of every transitive measurable
        dependency. Detects "user edited a measurable's body, cache
        is stale" — changing any function in the closure flips the
        resulting hex. 16-hex-char output; collisions are irrelevant
        against the user-edit baseline.

        Walks two dependency channels:

        - **Parameter-name registered deps** (the
          `pytest-fixture-style` resolution graph traced by
          `transitive_measurables`). Captures explicit `def
          foo(record, q_max, q_min)` style deps.
        - **Closure-captured operands** (`compose_of`). Captures
          deps that factories like `reduce_axis(of, ...)` hold via
          their generated `fn`'s closure. Without this, editing a
          source measurable wouldn't invalidate composed
          downstreams — a real cache-staleness bug.

        Cache-invalidation lives on the Measurable itself rather
        than in the runner because the closure IS a property of the
        function (its bytecode + dep bytecodes), not of the run.
        Runner-side logic that wants to detect stale columns just
        calls `m.signature()` and compares to a stored hex."""
        import hashlib
        seen: set[str] = set()
        parts: list[str] = []

        def _walk(m: 'Measurable[Mapping[str, object], object]') -> None:
            if m._name in seen:
                return
            seen.add(m._name)
            bc = bytes(m._fn.__code__.co_code)
            parts.append(f'{m._name}:{hashlib.sha256(bc).hexdigest()[:16]}')
            # Closure-captured operands.
            for sub in m._compose_of:
                _walk(sub)

        _walk(cast(
            'Measurable[Mapping[str, object], object]', self,
        ))
        # Parameter-name registered deps. `transitive_measurables`
        # raises KeyError if `self.name` isn't registered; for
        # ad-hoc measurables that the caller built but never
        # registered, the closure walk above is the only signal.
        try:
            for d in sorted(transitive_measurables(self._name)):
                if d in seen:
                    continue
                m = get_registered(d)
                if m is None:
                    parts.append(f'{d}:unregistered')
                    continue
                _walk(m)
        except KeyError:
            pass
        return hashlib.sha256(
            '\n'.join(sorted(parts)).encode(),
        ).hexdigest()[:16]

    def dispatch(self, record: R) -> 'Measurable[R, T]':
        """Pick the Measurable whose `reads` are present in
        `record`: primary first, else the first matching
        fallback, else the primary as final fallback (so a
        downstream `.fn(record)` raises naturally on the first
        missing key)."""
        if not self._fallbacks or all(k in record for k in self._reads):
            return self
        for alt in self._fallbacks:
            if all(k in record for k in alt._reads):
                return alt
        return self

    @override
    def __repr__(self) -> str:
        return (
            f'Measurable(name={self._name!r}, reads={self._reads!r}, '
            f'fn={self._fn!r})'
        )

    @override
    def __eq__(self, other: object) -> bool:
        # Field-wise equality (mirrors the prior frozen-dataclass
        # behavior). Two Measurables are equal iff they wrap the
        # same fn (identity) under the same name + reads +
        # fallbacks + compose_of. Diverging fn at the same name
        # compares unequal — that's how
        # `test_measurable_inequality_on_different_fn` catches
        # accidental fixture redefinition.
        if not isinstance(other, Measurable):
            return NotImplemented
        return (
            self._fn == other._fn
            and self._name == other._name
            and self._reads == other._reads
            and self._fallbacks == other._fallbacks
            and self._compose_of == other._compose_of
        )

    @override
    def __hash__(self) -> int:
        return hash((
            self._fn, self._name, self._reads,
            self._fallbacks, self._compose_of,
        ))


# ============ Name-keyed registry + resolver ============

_REGISTRY: Registry[Measurable[Mapping[str, object], object]] = Registry()


def get_registered(
    name: str,
) -> Measurable[Mapping[str, object], object] | None:
    """Look up a measurable by its registered name. Returns None
    if not registered. Typed accessor for the global registry."""
    return _REGISTRY.get(name)


def register[R: Mapping[str, object], T](
    m: Measurable[R, T],
) -> Measurable[R, T]:
    """Register `m` in the global name-keyed registry, idempotent
    on (name, identity). Returns the same instance.

    Two registration paths converge here:

    - The `@measurable` decorator at function-definition time.
    - The `@claim_bridge` decoder at bridges-import time, which
      auto-registers `Measurable` instances passed by value as a
      bridge's `source` / `target` (so a value-composed reduction
      like `mean_window(from_key('q_max'), 0.5, 1.0)` becomes
      cache-buildable without the author writing a separate
      `@measurable` wrapper).

    Same-name re-registration with the same identity is a no-op;
    same-name with a *different* instance raises `ValueError` to
    catch authoring mistakes (two reductions colliding on an
    auto-generated name)."""
    # Same-identity re-entry is idempotent; same-name with a
    # different instance raises `ValueError` via `Registry.register`.
    _REGISTRY.register(
        m.name,
        cast('Measurable[Mapping[str, object], object]', m),
    )
    return m


def registered_names() -> tuple[str, ...]:
    """Sorted tuple of all currently-registered measurable names.
    Useful for debug / diagnostic output."""
    return _REGISTRY.names()


def transitive_measurables(name: str) -> frozenset[str]:
    """All registered-measurable names reachable from `name` via
    the parameter-name dependency graph (inclusive of `name`).

    Walks `_measurable_param_names` recursively. Cycle-safe via a
    visited set. Used by cache builders to materialise every
    measurable a bridge file consumes — declared name + every
    transitive measurable dep — so analyses can read pre-computed
    scalars rather than recomputing per cell.

    Loud KeyError if `name` isn't registered."""
    visited: set[str] = set()

    def _close(n: str) -> None:
        if n in visited:
            return
        visited.add(n)
        m = _REGISTRY.get(n)
        if m is None:
            raise KeyError(
                f'no measurable named {n!r}. Registered: '
                f'{_REGISTRY.names()}',
            )
        for dep_name in _measurable_param_names(m.fn):
            _close(dep_name)

    _close(name)
    return frozenset(visited)


def transitive_reads(name: str) -> frozenset[str]:
    """The leaf record-keys a registered measurable ultimately
    depends on, closing over the measurable graph.

    Walks: starting from `name`, recurse into each measurable's
    parameter-named deps (via `_measurable_param_names`), union
    each transitively-reached measurable's `reads` field plus any
    `aliases[i].reads` (alias paths consume disjoint leaves and
    the redundancy primitive needs the conservative upper bound).
    Cycle-safe via a visited set.

    Loud KeyError if `name` isn't registered — callers asking
    about an unknown measurable get a useful failure rather than
    silent empty.

    Returns the union over the closure. Used by `Bridge.transitive_
    reads` to compute the leaf-key reads-set the redundancy
    primitive consumes."""
    visited: set[str] = set()

    def _close(n: str) -> frozenset[str]:
        if n in visited:
            return frozenset()
        visited.add(n)
        m = _REGISTRY.get(n)
        if m is None:
            raise KeyError(
                f'no measurable named {n!r}. Registered: '
                f'{_REGISTRY.names()}',
            )
        out: set[str] = set(m.reads)
        for alt in m.fallbacks:
            out.update(alt.reads)
        for dep_name in _measurable_param_names(m.fn):
            out.update(_close(dep_name))
        return frozenset(out)

    return _close(name)


def _measurable_param_names(fn: Callable[..., object]) -> tuple[str, ...]:
    """Parameters of `fn` (after the first positional `record`)
    whose names are registered measurables. Used by the resolver
    to decide which params get auto-injected."""
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        return ()
    params = list(sig.parameters.values())
    # Skip the first positional — that's the record (or a
    # zero-arg measurable; we still don't treat it as a dep).
    deps = [
        p.name for p in params[1:]
        if p.name in _REGISTRY
    ]
    return tuple(deps)


def _resolve_one(
    name: str, record: Mapping[str, object],
    cache: dict[str, object],
) -> object:
    """Resolve one measurable by name; recurses on its deps.
    Memoizes in `cache` so each measurable computes at most once
    per record. Loud KeyError if name isn't registered — caller
    asked for something that doesn't exist."""
    if name in cache:
        return cache[name]
    m = _REGISTRY.get(name)
    if m is None:
        raise KeyError(
            f'no measurable named {name!r}. Registered: '
            f'{_REGISTRY.names()}',
        )
    dep_names = _measurable_param_names(m.fn)
    deps = {d: _resolve_one(d, record, cache) for d in dep_names}
    cache[name] = m(record, **deps)
    return cache[name]


def evaluate_with_measurables[T](
    fn: Callable[..., T],
    record: Mapping[str, object],
    *,
    cache: dict[str, object] | None = None,
) -> T:
    """Inspect `fn`'s signature, resolve parameter-named
    measurables from the registry, call `fn(record, **deps)`.

    `cache` may be shared across multiple `fn` evaluations on the
    same record so that each measurable computes at most once
    overall — pass an empty dict the first time, the same dict
    on subsequent calls.

    Parameters whose names are NOT registered measurables are
    left for the caller to supply; if `fn` has unmatched required
    params and none are passed, Python raises the usual
    TypeError. The resolver only auto-injects what it can find."""
    if cache is None:
        cache = {}
    dep_names = _measurable_param_names(fn)
    deps = {d: _resolve_one(d, record, cache) for d in dep_names}
    return fn(record, **deps)


def compute_missing_columns(
    df: pl.DataFrame,
    names: Iterable[str],
) -> pl.DataFrame:
    """For each name in `names` that's not already a column of
    `df` and resolves in the @measurable registry, compute the
    measurable per-cell and add it as a column. Names that don't
    resolve (or aren't registered) are silently skipped — callers
    that need null-padding for unresolvable names handle that
    themselves.

    Single source of truth for the "raw cells → cached scalars"
    transform: the runner uses this to populate the per-module
    cache; `claim_bridge.evaluate()` uses the same path when a
    bridge's `scope` references a measurable column the input
    DataFrame doesn't carry yet (raw runs.parquet, no cache).

    Per-cell evaluator caches transitive measurables across
    same-cell calls (one pass populates all `names` for one cell
    before moving on), so dep-shared measurables compute once per
    cell."""
    if df.height == 0:
        return df
    have = set(df.columns)
    pending: list[tuple[str, Measurable[Mapping[str, object], object]]] = []
    for name in names:
        if name in have:
            continue
        m = get_registered(name)
        if m is None:
            continue
        pending.append((name, m))
    if not pending:
        return df

    cells = cast(list[dict[str, object]], df.to_dicts())
    new_cols: dict[str, list[object]] = {n: [] for n, _ in pending}
    for cell in cells:
        per_cell_cache: dict[str, object] = {}
        for name, m in pending:
            try:
                v = evaluate_with_measurables(
                    m.fn, cell, cache=per_cell_cache,
                )
            except Exception:  # noqa: BLE001
                # Record-level evaluation failure (missing inputs,
                # etc.) maps to None for this cell + measurable.
                # Downstream analyses NaN-skip these cells.
                v = None
            new_cols[name].append(v)

    return df.with_columns(
        [_to_polars_series(name, vals) for name, vals in new_cols.items()],
    )


def _to_polars_series(name: str, vals: list[object]) -> pl.Series:
    """Construct a polars Series from a heterogeneous-shape value
    list, tolerating None entries for measurables that return
    list/array types.

    polars's default dtype inference treats `[None, [1, 2], [3, 4]]`
    as needing-uniform-length sequences and crashes on `len(None)`.
    For sequence-typed measurables, replace None with an empty list
    so the constructor sees a uniform List shape; downstream
    consumers null-check the per-row length anyway."""
    import polars as pl
    has_seq = any(
        v is not None and not isinstance(
            v, (str, bytes, int, float, bool),
        )
        for v in vals
    )
    if not has_seq:
        return pl.Series(name, vals)
    normalized: list[object] = [
        v if v is not None else [] for v in vals
    ]
    return pl.Series(name, normalized)


# ============ Decorator ============

@overload
def measurable[R: Mapping[str, object], T](
    fn: Callable[..., T], /,
) -> Measurable[R, T]: ...


@overload
def measurable[R: Mapping[str, object], T](
    *,
    name: str | None = None,
    reads: tuple[str, ...] = (),
) -> Callable[[Callable[..., T]], Measurable[R, T]]: ...


def measurable[R: Mapping[str, object], T](
    fn: Callable[..., T] | None = None,
    /,
    *,
    name: str | None = None,
    reads: tuple[str, ...] = (),
) -> Measurable[R, T] | Callable[[Callable[..., T]], Measurable[R, T]]:
    """Register `fn` as a typed `Measurable[R, T]`. The instance
    is returned (so ad-hoc value-composition still works) AND
    indexed in a name-keyed registry (so other measurables /
    bridges can declare deps by parameter name).

    Two decorator forms:

        @measurable
        def q_mean(record: DQNRecord) -> float: ...

        @measurable(reads=('online_q_per_action',))
        def q_max(record: DQNRecord) -> float: ...

    Pytest-fixture-style transitive deps:

        @measurable
        def q_gap(record: DQNRecord, q_max: float, q_second: float) -> float:
            return q_max - q_second

    The framework reads `inspect.signature(q_gap.fn)`, sees the
    `q_max` and `q_second` parameter names, looks them up in the
    registry, computes recursively (memoized per record), and
    injects.

    `reads` is the LEAF record-key set the measurable ultimately
    depends on. For a measurable that itself reads a record key,
    declare it explicitly. For a measurable that ONLY depends on
    other measurables, leave `reads=()` and the framework can
    derive its reads from the union of its deps' reads (the
    `redundancy` primitive does this lookup transitively)."""
    if fn is not None:
        instance: Measurable[R, T] = Measurable(
            fn=fn, name=fn.__name__, reads=(),
        )
        # `@measurable` uses `replace` (last-write-wins) rather
        # than `register`. The decorator runs once per
        # module-import, but pytest fixtures redefine measurables
        # with the same name across tests; strict registration
        # would raise. The public `register(m)` function (above)
        # IS strict — it's the path `@claim_bridge` uses to
        # enforce uniqueness on auto-named composed measurables.
        _REGISTRY.replace(
            instance.name,
            cast(
                'Measurable[Mapping[str, object], object]',
                instance,
            ),
        )
        return instance

    def decorator(inner: Callable[..., T]) -> Measurable[R, T]:
        resolved_name = name if name is not None else inner.__name__
        instance: Measurable[R, T] = Measurable(
            fn=inner, name=resolved_name, reads=reads,
        )
        _REGISTRY.replace(
            instance.name,
            cast(
                'Measurable[Mapping[str, object], object]',
                instance,
            ),
        )
        return instance
    return decorator
