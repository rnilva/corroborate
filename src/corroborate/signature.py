"""Signature introspection — `Exogenous` marker + walker.

Two pieces:

1. **`Exogenous`** — a sentinel class used as PEP 593 `Annotated`
   metadata on claim kwargs. `Annotated[int, Exogenous]` declares
   that a kwarg is something we generalize *over*, not intervene
   on. Anything not so marked is implicitly a `leaf` — a
   configurational scalar claim, interventionable by default.
   Authors hide a leaf from intervention by baking it in via
   `functools.partial`; the bake-in records honestly via
   `_canonical_str`'s partial branch.

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
import inspect
import typing
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from typing import Annotated, Literal, TypeIs, get_args, get_origin

from corroborate.bridge import Bridge
from corroborate.claim import FnClaim, iter_invariants


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
    """One kwarg's introspected description."""
    name: str
    regime: Regime
    annotation: object
    default: object
    inner: ClaimSignature | None = None


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
        name = getattr(value, '__name__', '<callable>')
        if not isinstance(name, str):
            name = '<callable>'
        return _walk_fn(value, name=name, depth=depth, baked={})
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
        name = getattr(wrapped, '__name__', '<callable>')
        if not isinstance(name, str):
            name = '<callable>'
        return _walk_fn(wrapped, name=name, depth=depth, baked=baked)
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
        kwargs_out.append(KwargInfo(
            name=param_name, regime=regime,
            annotation=base_annotation, default=default, inner=inner,
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
        kwargs_out.append(KwargInfo(
            name=f.name, regime=regime,
            annotation=base_annotation, default=value, inner=inner,
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


# ============ Invariant collection ============

def collect_invariants(claim: object) -> tuple[Bridge[Mapping[str, object]], ...]:
    """Walk the composition tree starting from `claim`; union all
    `invariants` from each Claim node encountered. De-duplicated
    by `id`.

    Implementation: traverse the `walk(claim)` output. Each
    `KwargInfo.default` is the effective sub-claim AFTER bake
    (the walker already overlays partial.keywords as defaults),
    so bake-shadowing falls out for free — no separate accounting
    needed. For each visited default we collect its invariants
    via `iter_invariants` (typed predicate) and recurse via
    `kw.inner`."""
    bridges: list[Bridge[Mapping[str, object]]] = []
    seen: set[int] = set()

    # Top-level: invariants attached to `claim` itself. For
    # `partial(claim, ...)`, the wrapped claim's invariants live
    # on `.func`; drill through partials to find them.
    target = claim
    while isinstance(target, functools.partial):
        target = target.func
    _add(iter_invariants(target), bridges, seen)

    # Recurse: each kwarg's default is an effective sub-claim
    # (post-bake). Walker handles partial-overlay; we just visit
    # the resulting tree.
    sig = walk(claim)
    _walk_collect(sig, bridges, seen)
    return tuple(bridges)


def _walk_collect(
    sig: ClaimSignature,
    bridges: list[Bridge[Mapping[str, object]]],
    seen: set[int],
) -> None:
    """Visit each kwarg's default + recurse via `kw.inner`."""
    for kw in sig.kwargs:
        _add(iter_invariants(kw.default), bridges, seen)
        if kw.inner is not None:
            _walk_collect(kw.inner, bridges, seen)


def _add(
    invs: tuple[Bridge[Mapping[str, object]], ...],
    bridges: list[Bridge[Mapping[str, object]]],
    seen: set[int],
) -> None:
    """Append unseen invariants (de-dup by id)."""
    for inv in invs:
        if id(inv) not in seen:
            seen.add(id(inv))
            bridges.append(inv)


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
