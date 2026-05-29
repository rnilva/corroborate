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
from corroborate.core.signature import bytecode_source_hash


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
        """Closure hash: SHA-256 over each Measurable's bytecode
        + literal constants + referenced names, including every
        transitive dependency. Detects "user edited a measurable's
        body, cache is stale" — changing any function in the
        closure flips the resulting hex. 16-hex-char output;
        collisions are irrelevant against the user-edit baseline.

        Hashes three CodeType fields per function:
        - `co_code`: opcode stream (control flow + operations).
        - `co_consts`: literal table — numbers, strings, nested
          code objects from comprehensions / lambdas. Nested code
          objects are recursively walked (their consts + names
          + opcodes also hashed) so a constant-only edit inside an
          inner lambda still invalidates the outer signature.
        - `co_names`: external name references (globals,
          attributes, free names) — changes when the body switches
          which `np.nanmean` vs `np.mean` it calls, etc.

        Deliberately NOT hashed:
        - `co_varnames` (local variable names) — renaming a local
          shouldn't bust cache.
        - `co_argcount` / signature info — captured indirectly via
          opcodes when the args are referenced.

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
            parts.append(f'{m._name}:{bytecode_source_hash(m._fn.__code__)}')
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


# Modules that registered a measurable whose `fn.__module__` does NOT
# name the registering site. `register_as` aliases carry the factory's
# module (`reductions`), not the substrate that aliased them — recorded
# here at alias time so `registry_source_modules()` (the forkserver /
# spawn worker re-import set) re-runs the aliasing module. Without it a
# `register_as`-only module is silently absent from a fresh worker's
# registry and its aliases null-pad.
_EXTRA_SOURCE_MODULES: set[str] = set()


def get_registered(
    name: str,
) -> Measurable[Mapping[str, object], object] | None:
    """Look up a measurable by its registered name. Returns None
    if not registered. Typed accessor for the global registry."""
    return _REGISTRY.get(name)


def register[R: Mapping[str, object], T](
    m: Measurable[R, T],
) -> Measurable[R, T]:
    """Register `m` in the global name-keyed registry. Returns the
    same instance.

    Two registration paths converge here:

    - The `@measurable` decorator at function-definition time.
    - The `@claim_bridge` decoder at bridges-import time, which
      auto-registers `Measurable` instances passed by value as a
      bridge's `source` / `target` (so a value-composed reduction
      like `mean_window(from_key('q_max'), 0.5, 1.0)` becomes
      cache-buildable without the author writing a separate
      `@measurable` wrapper).

    Idempotency rules:

    - Same-name re-registration with the *same identity* is a
      no-op (the historical case — same module imported twice).
    - Same-name re-registration with a *different identity but
      equal `signature()`* is also a no-op. This catches the
      common authoring pattern where two findings modules each
      compose the same reduction by value (each call to
      `reduce_axis(from_key('mc_return'), axis=-1, op='mean')`
      mints a fresh `Measurable` instance with auto-name
      `mc_return__mean_axis_-1` but identical closure semantics).
      The closure hash is the same authority that powers cache-
      drift detection — if it's good enough to detect "formula
      changed, invalidate cache", it's good enough to detect
      "same formula, two instances".
    - Same-name with a *different signature* raises `ValueError`
      — that's a real conflict that needs an author rename.
    """
    existing = _REGISTRY.get(m.name)
    if existing is not None and existing is not m:
        if existing.signature() == m.signature():
            # Same closure, distinct instances — accept silently.
            # Don't replace: keep the first-registered identity
            # so downstream `is`-checks against the existing
            # reference stay valid.
            return m
        # Fall through to `Registry.register`, which will raise
        # ValueError with the existing-vs-new mismatch surfaced.
    _REGISTRY.register(
        m.name,
        cast('Measurable[Mapping[str, object], object]', m),
    )
    return m


def register_as[R: Mapping[str, object], T](
    m: Measurable[R, T],
    *,
    name: str,
    reads: tuple[str, ...] | None = None,
) -> Measurable[R, T]:
    """Register `m` under an aliased `name`, preserving `compose_of`
    and `reads`.

    Use when binding a stable hand-picked name to a composition of
    framework reductions whose auto-derived name would be verbose
    (e.g., `'online_max_q_per_step__mean_50_100'`) — the cached
    parquet column contract and `pl.col('q_late_mean')` scope
    predicates depend on stable hand-picked names.

    Threads `compose_of=m.compose_of` so `signature()` recursion at
    `measurable.py:240-246` reaches the source lineage through the
    rename, preserving structural cache invalidation. The earlier
    substrate idiom `Measurable(fn=composition.fn, name='alias',
    reads=(...))` silently dropped `compose_of`; `register_as` is
    the discipline that keeps the lineage honest.

    `reads` defaults to `m.reads`. Override only when the
    composition's auto-derived reads don't match the desired
    persistence contract (rare).
    """
    aliased: Measurable[R, T] = Measurable(
        fn=m.fn,
        name=name,
        reads=reads if reads is not None else m.reads,
        compose_of=m.compose_of,
    )
    # `aliased.fn.__module__` points at the factory that minted `m`
    # (e.g. `corroborate.measurables.reductions`), NOT the substrate
    # module calling `register_as` here. Record the caller's module so
    # the forkserver / spawn re-import set (`registry_source_modules`)
    # re-runs it — a `register_as`-only module (no plain `@measurable`)
    # is otherwise absent from a fresh worker and its aliases null-pad.
    # `inspect.getmodule(frame)` resolves via the frame's code file →
    # typed `ModuleType | None`, no `Any` leak.
    _frame = inspect.currentframe()
    _caller_mod = (
        inspect.getmodule(_frame.f_back) if _frame is not None else None
    )
    if _caller_mod is not None and _caller_mod.__name__ != '__main__':
        _EXTRA_SOURCE_MODULES.add(_caller_mod.__name__)
    register(aliased)
    return aliased


def registered_names() -> tuple[str, ...]:
    """Sorted tuple of all currently-registered measurable names.
    Useful for debug / diagnostic output."""
    return _REGISTRY.names()


def registry_source_modules() -> tuple[str, ...]:
    """Sorted distinct `fn.__module__` of every registered
    measurable — the import set that, re-executed in a fresh
    interpreter, re-runs the `@measurable` decorators that
    populated the registry.

    The substrate-agnostic recovery surface for fork-unsafe
    parallel workers (`runner._load_directory`). A worker spawned
    under `forkserver` / `spawn` starts with an EMPTY registry;
    re-importing exactly these modules re-establishes it without
    any per-substrate CLI plumbing.

    Modules whose name can't be re-imported by string are
    excluded:
    - `__main__` — a worker re-importing `__main__` would re-run
      the parent's entry script, not the registration site.
    - dunder-only / empty names — defensive against synthetic
      functions with a stripped `__module__`.

    Factory-composed measurables (`from_key`, `reduce_axis`, …)
    carry `fn.__module__ == 'corroborate.measurables.reductions'`
    rather than the substrate module that composed them. When such a
    composition is bound to a stable name via `register_as`, that call
    records its *caller's* module in `_EXTRA_SOURCE_MODULES` (unioned
    in below) — so a `register_as`-only module (one with no plain
    `@measurable`, e.g. `corroborate_rl.dqn.trace_reductions`) is still
    re-imported in a fresh worker. Without that record its aliases
    would be silently absent from the worker registry and come back
    all-null."""
    mods: set[str] = set()
    for name in _REGISTRY.names():
        m = _REGISTRY.get(name)
        if m is None:
            continue
        mod = m.fn.__module__
        if mod and mod != '__main__':
            mods.add(mod)
    # `register_as` aliases carry the factory's module, not the
    # aliasing substrate's; that module is recorded at alias time.
    mods.update(_EXTRA_SOURCE_MODULES)
    return tuple(sorted(mods))


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


def _measurable_extra_param_names(fn: Callable[..., object]) -> tuple[str, ...]:
    """Parameters of `fn` (after the first positional `record`)
    whose names are NOT registered measurables. Used by the P1+P4
    startup validator to flag likely typos: an unregistered
    parameter name is almost always either (a) a typo of a real
    measurable name, or (b) a leftover from a renamed dep.

    Either way: the framework will raise `TypeError: missing
    required argument` at call time, but the author wants to know
    at registration time — not when the bridge eval finally
    triggers a per-cell call months after the typo landed."""
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        return ()
    params = list(sig.parameters.values())
    extras = [
        p.name for p in params[1:]
        if p.name not in _REGISTRY
        # Ignore conventional kwargs that aren't measurable deps.
        and p.name not in ('record', '_record', 'self')
    ]
    return tuple(extras)


def audit_measurable_registry(
    *, strict: bool = False,
) -> tuple[str, ...]:
    """P1+P4 startup validator. Returns a tuple of warning
    messages — one per measurable whose declared `reads` tuple
    diverges from the framework's auto-derived expectation.

    Three failure modes flagged:

    1. **Empty `reads=()` with no parameter-injected deps**: the
       measurable reads NOTHING transitively. Usually means it's a
       constant or has hard-coded values — almost certainly a
       substrate author bug (the measurable will only ever return
       the same value regardless of input).

    2. **Parameter-injected name not in registry**: a measurable
       declares `def fn(record, mc_return_raw_episodes)` but
       `mc_return_raw_episodes` isn't registered. The framework
       can't inject; the call will TypeError at runtime. Author
       likely forgot to import the substrate module that
       registers the dep, or typo'd the name.

    3. **`reads` declares cols that aren't on any actual record
       path**: NOT flagged here (would require a corpus to test
       against). The validator catches structural drift, not
       cell-level mismatch.

    `strict=True` raises `ValueError` instead of returning
    warnings — useful from a CLI startup gate.

    Pure read; no side effects. Safe to call multiple times."""
    warnings: list[str] = []
    for name in _REGISTRY.names():
        m = _REGISTRY.get(name)
        if m is None:
            continue
        deps = _measurable_param_names(m.fn)
        extras = _measurable_extra_param_names(m.fn)
        if extras:
            warnings.append(
                f'measurable {name!r}: parameter(s) {extras!r} '
                f'are not in the registry — framework cannot inject. '
                f'Possible causes: typo, or substrate module that '
                f'registers them not imported. Will TypeError on '
                f'first call.',
            )
        if not deps and not m.reads:
            # Truly read-nothing measurable. Likely a constant or
            # an authoring bug. Flag.
            warnings.append(
                f'measurable {name!r}: declares `reads=()` AND has '
                f'no parameter-injected deps — measurable reads '
                f'nothing transitively. Hard-coded value? Likely a '
                f'substrate author bug.',
            )
    if strict and warnings:
        raise ValueError(
            'measurable registry audit failed:\n  '
            + '\n  '.join(warnings),
        )
    return tuple(warnings)


def _resolve_one(
    name: str, record: Mapping[str, object],
    cache: dict[str, object],
) -> object:
    """Resolve one measurable by name; recurses on its deps.
    Memoizes in `cache` so each measurable computes at most once
    per record. Loud KeyError if name isn't registered — caller
    asked for something that doesn't exist.

    **P1 fix — record-as-precomputed-cache**: when a measurable's
    value is already present in `record` (e.g. a list column
    previously computed and persisted to `measurements.parquet`,
    then joined onto the cell DataFrame), use that value rather
    than recomputing from scratch. This is what closes the lie at
    `_missing_for_restore`: a measurable whose transitive reads
    are trace cols (cloud-evicted) but whose own value is already
    in the cache as a list col should be USABLE without re-
    restoring traces.

    Skips substitution when the cached value is None/NaN (treated
    as 'not actually computed') — partial-NaN recompute then fires
    through the normal path."""
    if name in cache:
        return cache[name]
    # Record-as-precomputed-cache: prefer the persisted value when
    # present + non-missing. The `is None` short-circuit handles
    # the None-stamped case from `compute_missing_columns`'s
    # cell-injection cascade — those entries are sentinels that
    # the framework couldn't compute on this cell, NOT a real
    # cached value.
    if name in record:
        val = record[name]
        if val is not None and not _is_scalar_nan(val):
            cache[name] = val
            return val
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


def _is_scalar_nan(v: object) -> bool:
    """True iff `v` is a float NaN scalar. Returns False for any
    non-float (list / ndarray / mapping / etc.). Used by
    `_resolve_one` to decide whether a record-side value counts
    as 'already computed': a NaN scalar is a sentinel, not a real
    cached value."""
    if not isinstance(v, float):
        return False
    return v != v   # NaN-NaN-self-inequality, the standard idiom


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


def _topo_sort_pending(
    pending: list[tuple[
        str,
        'Measurable[Mapping[str, object], object]',
        list[object] | None,
    ]],
) -> list[tuple[
    str,
    'Measurable[Mapping[str, object], object]',
    list[object] | None,
]]:
    """Topological sort of `pending` by inter-measurable
    dependencies. A pending name `b` depends on `a` iff `a` is in
    pending AND `a` appears in `b.reads` directly OR transitively
    through another pending name. Leaves (no deps within pending)
    come first.

    The walk uses `m.reads` (authored direct-reads tuple) rather
    than `_measurable_param_names` — this captures both styles
    of dep authoring: param-injection (`def fn(record, dep: T)`)
    AND `record.get('dep')`. The cascade fix targets the latter;
    param-injected deps additionally cascade via `_resolve_one`
    in the per-cell evaluator, independent of this ordering.

    Tie-breaking is alphabetical (Kahn's algorithm sorts the
    leaf-set at each step). When multiple unrelated leaves sit
    at the same dependency depth, the alphabetical order — not
    the input order in `pending` — wins. Bridge-author
    consequence: if downstream resolution depends on tie-break
    determinism for two name-independent measurables, the order
    is the alphabetical lexicographic one.

    Falls back to the input order on cycles (defensive; valid
    `@measurable` graphs are DAGs)."""
    by_name: dict[str, tuple[
        str,
        Measurable[Mapping[str, object], object],
        list[object] | None,
    ]] = {n: (n, m, ex) for n, m, ex in pending}
    pending_names: set[str] = set(by_name)
    deps: dict[str, set[str]] = {n: set() for n in pending_names}
    # Transitive closure: walk m.reads recursively, intersecting
    # with pending_names at every step. Names that aren't
    # registered or aren't in pending get skipped.
    for n in pending_names:
        stack = [n]
        seen_here: set[str] = set()
        while stack:
            cur = stack.pop()
            cur_m = _REGISTRY.get(cur)
            if cur_m is None:
                continue
            for r in cur_m.reads:
                if r in pending_names and r != n and r not in seen_here:
                    seen_here.add(r)
                    deps[n].add(r)
                    stack.append(r)
    # Kahn's algorithm with stable (alphabetical) tie-breaking.
    ordered: list[str] = []
    remaining = {n: set(d) for n, d in deps.items()}
    while remaining:
        leaves = sorted(n for n, d in remaining.items() if not d)
        if not leaves:
            # Cycle — append remaining in stable order (defensive;
            # @measurable graphs should be DAGs).
            ordered.extend(sorted(remaining))
            break
        ordered.extend(leaves)
        for leaf in leaves:
            del remaining[leaf]
        for d in remaining.values():
            d -= set(leaves)
    return [by_name[n] for n in ordered if n in by_name]


def compute_missing_columns(
    df: pl.DataFrame,
    names: Iterable[str],
) -> pl.DataFrame:
    """For each name in `names` that resolves in the @measurable
    registry, compute the measurable per-cell. Two cases:

    - **Column missing** from `df`: compute for every cell, add as
      a new column.
    - **Column present but partially null** (some cells null,
      others filled): compute only for the null cells, preserve
      existing values, replace the column. This handles the
      diagonal_relaxed-concat-of-corpora case where some
      corpora's runs.parquet pre-computes a measurable and others
      don't — without per-cell fill, the null subset stays null
      because the column-level "have it / don't" check skips it.

    Names that don't resolve (or aren't registered) are silently
    skipped — callers that need null-padding for unresolvable
    names handle that themselves.

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
    # `pending` items: (name, measurable, existing_values).
    # `existing_values` is None when the column is fully missing
    # (compute every cell); a list[object | None] when partially
    # null (compute only for cells where the entry is None).
    pending: list[tuple[
        str,
        Measurable[Mapping[str, object], object],
        list[object] | None,
    ]] = []
    seen: set[str] = set()
    # Dedupe `names` — `pl.Expr.meta.root_names()` returns one entry
    # per reference in the expression, so a scope that mentions
    # `q_divergence_score` six times yields six duplicates here.
    # Without dedup the inner loop appends the same value six
    # times per cell, blowing the column up to 6 × df.height and
    # tripping a polars ShapeError at `with_columns` time.
    for name in names:
        if name in seen:
            continue
        m = get_registered(name)
        if m is None:
            continue
        if name not in have:
            seen.add(name)
            pending.append((name, m, None))
            continue
        # Column already present — recompute cells with missing
        # values (null OR NaN). NaN ≠ null in polars but both
        # signal "no real value for this cell"; an all-NaN column
        # (e.g. trace-dependent measurable was added to defaults
        # AFTER a sweep ran with no traces — sweep persistence
        # writes NaN for cells that couldn't be computed) should
        # be re-evaluated when the data is now available, not
        # kept as fossilised NaN. Float dtypes get the
        # null|NaN check; non-float dtypes (Int / Bool / String)
        # only support is_null.
        col = df[name]
        if col.dtype.is_float():
            missing_mask = col.is_null() | col.is_nan()
        else:
            missing_mask = col.is_null()
        if not missing_mask.any():
            continue
        seen.add(name)
        # `existing_values` per-cell entry is None where missing
        # (so the eval loop recomputes), original value where
        # present. NaN counts as "missing" for the recompute
        # decision; we set those entries to None so the same
        # branch fires per-cell.
        existing_vals: list[object] = [
            None if missing else v
            for v, missing in zip(col.to_list(), missing_mask.to_list())
        ]
        pending.append((name, m, existing_vals))
    if not pending:
        # **C3 fast-path** (CACHE_BUILD.md): no measurables to
        # compute → return the input frame WITHOUT materialising
        # `to_dicts()`. The repeated-cache-load case lands here:
        # a cache that already has every required column should
        # not pay the cost of converting hundreds-of-cells × tens-
        # of-columns into Python dicts just to discover there's
        # nothing to do.
        return df

    # Topologically order `pending` by inter-measurable deps so a
    # measurable that reads another pending measurable's value
    # (via param injection OR `record.get(...)`) sees the
    # just-computed value rather than the pre-pass null/NaN.
    # Without this, a single `compute_missing_columns` pass with
    # required={leaf, dependent} leaves the `dependent` reading
    # the absent column and silently returning NaN — author has
    # to call build twice in the right order (today's gotcha:
    # `lambda_a_late` reads `q_action_std_late` /
    # `q_argmax_margin_late` via `record.get` and didn't
    # cascade-compute in the same pass).
    #
    # Param-injected deps already cascade via `_resolve_one` per
    # cell; this fixes the `record.get(...)` pattern too. Order
    # uses `m.reads` (the authored direct-reads tuple); names not
    # in pending are ignored (they're trace / runs cols already
    # on the cell dict).
    pending = _topo_sort_pending(pending)
    cells = cast(list[dict[str, object]], df.to_dicts())
    new_cols: dict[str, list[object]] = {n: [] for n, _, _ in pending}
    # Track measurables that ALWAYS failed across all cells — those
    # are authoring bugs, not "missing inputs," and should surface
    # as a stderr warning so the substrate author sees them. Per-cell
    # failures (legitimately missing inputs on subset of cells)
    # remain silent NaN-mapped via the existing path.
    # Counts denominator is per-measurable (full set for added cols,
    # null-cell subset for partial cols).
    fail_counts: dict[str, int] = {n: 0 for n, _, _ in pending}
    eval_counts: dict[str, int] = {n: 0 for n, _, _ in pending}
    last_exception: dict[str, BaseException] = {}
    for i, cell in enumerate(cells):
        per_cell_cache: dict[str, object] = {}
        for name, m, existing in pending:
            if existing is not None and existing[i] is not None:
                # Already filled — preserve.
                new_cols[name].append(existing[i])
                # Cascade: downstream pending measurables that read
                # this via `record.get(...)` need it on the cell dict.
                cell[name] = existing[i]
                continue
            eval_counts[name] += 1
            try:
                v = evaluate_with_measurables(
                    m.fn, cell, cache=per_cell_cache,
                )
            except (KeyError, TypeError, ValueError, ZeroDivisionError) as e:
                # Expected per-cell failure modes:
                # - KeyError: required record key absent on this cell
                # - TypeError: wrong shape / type passed to numpy op
                # - ValueError: shape mismatch / numpy-derived
                # - ZeroDivisionError: degenerate-input arithmetic
                # Map to None for this cell; downstream NaN-skips.
                #
                # Anything else (AttributeError, RuntimeError,
                # NameError, etc.) propagates with full traceback
                # — those are authoring bugs that should fail
                # loudly with a line number, not silently map to
                # None and rely on a stderr warning at session
                # end. Pinned by
                # `test_unrelated_exception_propagates`.
                v = None
                fail_counts[name] += 1
                last_exception[name] = e
            new_cols[name].append(v)
            # Cascade: inject just-computed value into the cell so
            # downstream pending measurables that read it via
            # `record.get(name)` see the new value rather than the
            # pre-pass null/NaN. Param-injected deps already cascade
            # via the per-cell evaluator cache.
            cell[name] = v
    # If a measurable failed for EVERY cell, that's an authoring bug
    # (typo in measurable.fn body, broken signature, etc.) — not
    # "missing inputs on a subset." Emit a stderr warning so it
    # doesn't disappear into a silent all-null column.
    if cells:
        import sys as _sys
        for name, count in fail_counts.items():
            denom = eval_counts[name]
            if denom > 0 and count == denom:
                exc = last_exception.get(name, RuntimeError('unknown'))
                _sys.stderr.write(
                    f'WARNING: measurable {name!r} raised '
                    f'{type(exc).__name__} on ALL {count} cells '
                    f'({exc}); column will be all-null. Authoring '
                    f'bug or schema mismatch?\n',
                )

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
    consumers null-check the per-row length anyway.

    Multi-dimensional ndarrays (e.g. a measurable returning shape
    `(n_bursts, n_episodes)` per cell) are converted to nested
    Python lists before construction. Polars supports
    `List(List(...))` natively but its inference path can't go
    from `[ndarray2D, ndarray2D, ...]` directly — it falls back to
    object dtype and raises. Pre-converting via `.tolist()` lets
    the inference path see uniform nested-list shape."""
    import polars as pl
    import numpy as np

    has_seq = any(
        v is not None and not isinstance(
            v, (str, bytes, int, float, bool),
        )
        for v in vals
    )
    if not has_seq:
        return pl.Series(name, vals)
    # Convert ndarrays (any dim) to nested Python lists so polars'
    # inference sees uniform list-of-list shape rather than a list
    # of ndarray objects.
    normalized: list[object] = [
        v.tolist() if isinstance(v, np.ndarray)
        else (v if v is not None else [])
        for v in vals
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
