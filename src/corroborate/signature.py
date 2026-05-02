"""Signature introspection — `Exogenous` marker + walker.

Two pieces:

1. **`Exogenous`** — a sentinel class used as PEP 593 `Annotated`
   metadata on claim kwargs. `Annotated[int, Exogenous]` declares
   that a kwarg is something we generalize *over*, not intervene
   on. Anything not so marked is implicitly a `leaf` — a
   configurational scalar claim, interventionable by default.
   Authors hide a leaf from intervention by baking it in via
   `functools.partial`; the bake-in records honestly via
   `canonical_str`'s partial branch.

2. **Walker** — `walk(claim) → ClaimSignature`. Recursively
   descends into:
   - free-function claims (`FnClaim`): walk the wrapped fn's
     signature.
   - frozen-dataclass instances (Modules, config bundles): walk
     `dataclasses.fields`.
   - `functools.partial` over a claim: walk the wrapped claim's
     signature, but with each baked kwarg's default replaced by
     the bound value. Lets `partial(linear_epsilon, anneal_steps=
     50_000)` surface `anneal_steps=50_000` instead of the
     original default.

The walker also feeds two consumers: `flatten_leaves` /
`flatten_exogenous` (configurational-leaf discovery for parquet
columns + intervention surface), and `collect_invariants` (auto-
discovery of invariants attached to any claim in the composition
tree).

"Leaf" terminology: a leaf-regime kwarg is a configurational
scalar claim — a non-recursive value in the graph of claims
that's observed at composition time. RL practice calls these
"hyperparameters"; the framework uses "leaf" because the same
shape covers any non-RL configuration too. Authors who want to
hide a leaf from intervention bake it in via `functools.partial`."""
from __future__ import annotations

import functools
import hashlib
import inspect
import typing
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from typing import (
    Annotated, Literal, Protocol, TypeIs,
    get_args, get_origin, runtime_checkable,
)

from corroborate.claim import FnClaim


@runtime_checkable
class _Named(Protocol):
    """Anything with a string `__name__` — captures `def`-ed
    functions, lambdas (`<lambda>`), classes, and most builtins.
    Replacing `getattr(value, '__name__', ...)` with an isinstance
    check against this Protocol satisfies the typing-discipline
    rule against dynamic-attribute access on typed values."""
    @property
    def __name__(self) -> str: ...


def _callable_name(value: object) -> str:
    """Best-effort name of a callable for hash/error messages.
    Returns the `__name__` attribute when present (the common
    case for any function / class / builtin), otherwise the
    repr — which is stable for identity-based callables like
    `object.__init__`."""
    if isinstance(value, _Named):
        return value.__name__
    return repr(value)


# ============ Marker ============

class Exogenous:
    """Sentinel marker for `Annotated[T, Exogenous]`.

    Class identity (or subclass) is the test. No instances, no
    state. Authors mark a kwarg exogenous by writing
    `seed: Annotated[int, Exogenous]` — anything else is a leaf."""


# ============ Walked-signature shape ============

type Regime = Literal['exogenous', 'leaf']


@dataclass(frozen=True, slots=True)
class KwargInfo:
    """One kwarg's introspected description.

    `is_at_default` is true iff the bound `default` value equals
    the function's signature default by `==`. This catches the
    simple case (user explicitly bound the same scalar/Claim as
    the default).

    `sig_default_inner` carries the walked structure of the
    function's signature default (when it's recursable: a
    callable / partial / dataclass). At canonicalization time, if
    `is_at_default` is False but `canonical(inner) == canonical(
    sig_default_inner)`, the binding is canonically a no-op and
    the kwarg is elided. This covers the `partial(claim_default)`
    wrapping case where the partial object isn't `==` the FnClaim
    but their canonical forms match."""
    name: str
    regime: Regime
    annotation: object
    default: object
    inner: ClaimSignature | None = None
    is_at_default: bool = True
    sig_default_inner: ClaimSignature | None = None


@dataclass(frozen=True, slots=True)
class ClaimSignature:
    """Recursive description of a claim's introspectable surface."""
    name: str
    kwargs: tuple[KwargInfo, ...]


# ============ Walker (single recursion) ============

def walk(claim: object) -> ClaimSignature:
    """Walk a claim's signature, classifying each kwarg by regime
    and recursing into structured defaults.

    Outermost level: every kwarg is reported (required HPs without
    defaults are still HPs).

    Nested levels (depth ≥ 1): kwargs without a default AND without
    `Exogenous` annotation are DATA FLOW (runtime composition
    inputs), not configuration — skipped.

    Recurses into:
    - `FnClaim` defaults → walk wrapped fn's signature.
    - frozen-dataclass instances → walk `dataclasses.fields`.
    - `functools.partial` over a claim → walk wrapped, overlay
      baked kwargs as defaults."""
    return _walk(claim, depth=0)


def _walk(value: object, *, depth: int) -> ClaimSignature:
    """Internal recursive walker. Dispatches on value shape.
    Returns an empty ClaimSignature for unrecognised shapes."""
    if isinstance(value, functools.partial):
        return _walk_partial(value, depth=depth)
    if isinstance(value, FnClaim):
        return _walk_fn(value.fn, name=value.name, depth=depth, baked={})
    if is_dataclass(value) and not isinstance(value, type):
        return _walk_dataclass(value, depth=depth)
    if callable(value):
        # Plain function (not @claim'd) or builtin — treat like
        # a function-claim: walk its signature with empty baked.
        return _walk_fn(
            value, name=_callable_name(value), depth=depth, baked={},
        )
    return ClaimSignature(name='<leaf>', kwargs=())


def _walk_partial(p: functools.partial[object], *, depth: int) -> ClaimSignature:
    """`functools.partial` over a claim: walk the wrapped callable,
    overlay keyword baked args.

    Positional baked args are NOT supported for HP overlay — the
    framework's intervention pattern names HPs by keyword (the
    `**intervention` spread into `partial(claim, **kwargs)`). If
    `p.args` is non-empty, those values are passed at call-time
    but don't show as HP defaults in the walker output. Authors
    who need positional bake-in are using partial outside the
    documented intervention surface."""
    wrapped: object = p.func
    baked = dict(p.keywords) if p.keywords else {}

    if isinstance(wrapped, FnClaim):
        return _walk_fn(wrapped.fn, name=wrapped.name, depth=depth, baked=baked)
    if is_dataclass(wrapped) and not isinstance(wrapped, type):
        # Rare: partial over a Module instance. Module fields are
        # construction-time so partial.keywords would be runtime
        # call-args. Walk the dataclass and ignore baked here.
        return _walk_dataclass(wrapped, depth=depth)
    if callable(wrapped):
        return _walk_fn(
            wrapped, name=_callable_name(wrapped),
            depth=depth, baked=baked,
        )
    return ClaimSignature(name='<leaf>', kwargs=())


def _walk_fn(
    fn: Callable[..., object],
    *,
    name: str,
    depth: int,
    baked: Mapping[str, object],
) -> ClaimSignature:
    """Walk a plain function's signature. `baked` overlays defaults
    (used by `_walk_partial` to surface bake-in values as HP
    defaults)."""
    try:
        sig = inspect.signature(fn)
        hints = typing.get_type_hints(fn, include_extras=True)
    except (TypeError, NameError, ValueError):
        return ClaimSignature(name=name, kwargs=())

    kwargs_out: list[KwargInfo] = []
    for param_name, param in sig.parameters.items():
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if param_name in ('self', 'cls'):
            continue
        annotation = hints.get(param_name, param.annotation)
        regime = _regime_from_annotation(annotation)
        base_annotation = _strip_annotated(annotation)

        baked_value = baked.get(param_name, _MISSING)
        has_default = (
            baked_value is not _MISSING
            or param.default is not inspect.Parameter.empty
        )
        # Data-flow filter at depth >= 1.
        if depth >= 1 and not has_default and regime != 'exogenous':
            continue

        default = baked_value if baked_value is not _MISSING else param.default
        inner: ClaimSignature | None = (
            _walk(default, depth=depth + 1) if _is_recursable(default) else None
        )
        sig_default = param.default
        # Walk the signature default itself (separate from `default`
        # which may be baked-over) so `canonical()` can compare
        # canonical forms when the user wraps the default in a partial.
        sig_default_inner: ClaimSignature | None = (
            _walk(sig_default, depth=depth + 1)
            if sig_default is not inspect.Parameter.empty
            and _is_recursable(sig_default)
            else None
        )
        if baked_value is _MISSING:
            is_at_default = True
        elif sig_default is inspect.Parameter.empty:
            is_at_default = False
        else:
            try:
                is_at_default = (
                    baked_value is sig_default
                    or bool(baked_value == sig_default)
                )
            except (TypeError, ValueError):
                is_at_default = baked_value is sig_default
        kwargs_out.append(KwargInfo(
            name=param_name, regime=regime,
            annotation=base_annotation, default=default, inner=inner,
            is_at_default=is_at_default,
            sig_default_inner=sig_default_inner,
        ))
    return ClaimSignature(name=name, kwargs=tuple(kwargs_out))


def _walk_dataclass(instance: object, *, depth: int) -> ClaimSignature:
    """Walk a frozen-dataclass instance's fields."""
    cls = type(instance)
    try:
        hints = typing.get_type_hints(cls, include_extras=True)
    except (TypeError, NameError):
        hints = {}

    kwargs_out: list[KwargInfo] = []
    for f in fields(instance):
        annotation = hints.get(f.name, f.type)
        regime = _regime_from_annotation(annotation)
        base_annotation = _strip_annotated(annotation)
        value = getattr(instance, f.name)
        inner: ClaimSignature | None = (
            _walk(value, depth=depth + 1) if _is_recursable(value) else None
        )
        # Dataclass field's declared default — used by canonical
        # form elision. `dataclasses.MISSING` is the sentinel for
        # "no default declared".
        from dataclasses import MISSING as _DC_MISSING
        if f.default is not _DC_MISSING:
            sig_default = f.default
            try:
                is_at_default = value is sig_default or bool(value == sig_default)
            except (TypeError, ValueError):
                is_at_default = value is sig_default
        elif f.default_factory is not _DC_MISSING:
            try:
                sig_default = f.default_factory()
                is_at_default = value is sig_default or bool(value == sig_default)
            except (TypeError, ValueError, Exception):
                is_at_default = False
        else:
            is_at_default = False
        kwargs_out.append(KwargInfo(
            name=f.name, regime=regime,
            annotation=base_annotation, default=value, inner=inner,
            is_at_default=is_at_default,
        ))
    return ClaimSignature(name=cls.__name__, kwargs=tuple(kwargs_out))


def _is_recursable(value: object) -> bool:
    """True if `_walk(value)` would surface non-empty structure."""
    if isinstance(value, functools.partial):
        return True
    if isinstance(value, FnClaim):
        return True
    if is_dataclass(value) and not isinstance(value, type):
        return True
    return False


# Internal sentinel for "no baked value at this position." `None`
# is a valid HP default, so we can't use it as the "missing" marker.
_MISSING: object = object()


# ============ Flatten helpers ============

def flatten_exogenous(sig: ClaimSignature) -> dict[str, KwargInfo]:
    """All Exogenous kwargs across the tree, flattened by name.
    Outermost wins on collision."""
    out: dict[str, KwargInfo] = {}
    _flatten(sig, out, regime='exogenous')
    return out


def flatten_leaves(sig: ClaimSignature) -> dict[str, KwargInfo]:
    """All leaf-regime kwargs across the tree, flattened by name.

    "Leaf" here means: a configurational scalar claim (a non-
    recursive node of the configured composition graph). What RL
    practice calls a "hyperparameter"; the framework-honest term
    is `leaf` since the value is observed at composition time, not
    a claim that runs."""
    out: dict[str, KwargInfo] = {}
    _flatten(sig, out, regime='leaf')
    return out


def walk_paths(sig: ClaimSignature, *, regime: Regime) -> dict[str, KwargInfo]:
    """Path-keyed projection of the configured tree.

    Returns `{path: KwargInfo}` where `path` is the dotted path
    from the outermost claim to each kwarg of the given regime.
    Distinct from `flatten_leaves` (flat-keyed, last-wins on
    collision): two leaves with the same name at different
    positions in the tree get distinct keys
    (`optimizer.lr` vs `q_network.lr`).

    The outermost claim's name is *not* included in the prefix —
    it's a per-experiment constant and adds noise. So the top-
    level kwargs of the configured claim appear at depth 1
    (e.g. `gamma`, `optimizer`, `bootstrap`) and their children
    extend the path (`optimizer.lr`, `optimizer.inner.warmup_steps`)."""
    out: dict[str, KwargInfo] = {}
    _walk_paths(sig, out, regime=regime, prefix='')
    return out


def _flatten(
    sig: ClaimSignature, acc: dict[str, KwargInfo], *, regime: Regime,
) -> None:
    for kw in sig.kwargs:
        if kw.regime == regime and kw.name not in acc:
            acc[kw.name] = kw
        if kw.inner is not None:
            _flatten(kw.inner, acc, regime=regime)


def _walk_paths(
    sig: ClaimSignature,
    acc: dict[str, KwargInfo],
    *,
    regime: Regime,
    prefix: str,
) -> None:
    for kw in sig.kwargs:
        path = f'{prefix}.{kw.name}' if prefix else kw.name
        if kw.regime == regime:
            acc[path] = kw
        if kw.inner is None:
            continue
        # Exogenous propagates through structural descent: when
        # the parent is something we generalize *over* (env_params,
        # rng_key, etc.), its sub-fields are too — they describe
        # the variation surface, not intervention surface. So when
        # collecting leaf paths, don't descend into an exogenous
        # parent. Collecting exogenous paths still descends (an
        # author querying the exogenous surface wants the full
        # tree)."""
        if regime == 'leaf' and kw.regime == 'exogenous':
            continue
        _walk_paths(kw.inner, acc, regime=regime, prefix=path)


# ============ Annotation helpers ============

def _is_exogenous_marker(meta: object) -> TypeIs[type[Exogenous]]:
    return isinstance(meta, type) and issubclass(meta, Exogenous)


def _regime_from_annotation(ann: object) -> Regime:
    if get_origin(ann) is Annotated:
        for meta in get_args(ann)[1:]:
            if _is_exogenous_marker(meta):
                return 'exogenous'
    return 'leaf'


def _strip_annotated(ann: object) -> object:
    if get_origin(ann) is Annotated:
        return get_args(ann)[0]
    return ann


# ============ Canonical form for graph identity ============

CANONICAL_VERSION: str = 'v1'


def canonical(sig: ClaimSignature) -> ClaimSignature:
    """Strip kwargs whose binding is canonically equivalent to the
    signature default.

    Two equivalence rules:
    1. Direct: bound value `==` signature default → elide.
    2. Canonical: canonical(walk(bound)) == canonical(walk(sig_default))
       → elide. Catches the `partial(claim_default)` wrapper case
       where the partial object isn't `==` the FnClaim but their
       walked-and-canonicalised structures match.

    The second rule gives the framework forward-compatibility: a
    new Protocol axis added with a default implementation doesn't
    perturb canonical forms of corpora that pre-date the axis,
    even if those corpora wrapped the default in partials.

    Recursive: child ClaimSignatures are canonicalised too. A
    divergent grandchild keeps the parent kwarg in the canonical
    form even if the parent itself looks at-default."""
    out: list[KwargInfo] = []
    for k in sig.kwargs:
        inner = canonical(k.inner) if k.inner is not None else None
        sig_inner = (
            canonical(k.sig_default_inner)
            if k.sig_default_inner is not None else None
        )
        if k.is_at_default and (inner is None or len(inner.kwargs) == 0):
            continue
        if (
            inner is not None and sig_inner is not None
            and inner == sig_inner
        ):
            continue
        out.append(KwargInfo(
            name=k.name, regime=k.regime,
            annotation=k.annotation, default=k.default,
            inner=inner, is_at_default=k.is_at_default,
            sig_default_inner=k.sig_default_inner,
        ))
    return ClaimSignature(name=sig.name, kwargs=tuple(out))


def _stable_repr(value: object) -> str:
    """Hash-friendly stringification for leaf values. Handles the
    closed set of canonical-form leaf types: scalars, tuples, lists,
    frozen dataclasses (by class name + fields), partials and FnClaims
    (by name; their inner structure is captured in the recursive
    walk, not at this level)."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return repr(value)
    if isinstance(value, (tuple, list)):
        return '[' + ','.join(_stable_repr(v) for v in value) + ']'
    if isinstance(value, FnClaim):
        return f'Claim:{value.name}'
    if isinstance(value, functools.partial):
        wrapped: object = value.func
        wname = (
            wrapped.name if isinstance(wrapped, FnClaim)
            else _callable_name(wrapped)
        )
        return f'partial({wname})'
    if is_dataclass(value) and not isinstance(value, type):
        return f'dataclass:{type(value).__name__}'
    if callable(value):
        return f'callable:{_callable_name(value)}'
    return repr(value)


def claim_graph_signature(claim: object) -> str:
    """Deterministic structural hash of `claim`'s canonical form.

    Two bound partials with the same canonical form (after default-
    elision) produce the same signature, regardless of syntactic
    bake-in differences. Used as the program-instance identity
    column on RunRow / BridgeRow.

    Versioned: the `CANONICAL_VERSION` prefix means future changes
    to the canonicalisation rule produce different hashes, so old
    corpora's signatures don't silently collide with new ones."""
    sig = canonical(walk(claim))
    parts: list[str] = [CANONICAL_VERSION, sig.name]
    _emit_signature(sig, parts)
    blob = '|'.join(parts).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _emit_signature(sig: ClaimSignature, parts: list[str]) -> None:
    """Recursive emit for hash. Sorted by name for determinism."""
    sorted_kwargs = sorted(sig.kwargs, key=lambda k: k.name)
    for k in sorted_kwargs:
        if k.inner is not None:
            parts.append(f'{k.name}=>{k.inner.name}')
            _emit_signature(k.inner, parts)
        else:
            parts.append(f'{k.name}={_stable_repr(k.default)}')
