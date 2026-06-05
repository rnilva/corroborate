"""Signature introspection + source-fingerprint primitives.

The module hosts two related surfaces:

A. **Source-fingerprint primitives** — three hashes, picked by the
*level of abstraction* the caller wants to identify:

| primitive | level | granularity | use case |
|---|---|---|---|
| `bytecode_source_hash` | Python bytecode | `co_code` + `co_consts` + `co_names` | Measurable cache invalidation; closure-hash |
| `claim_graph_signature` | signature tree | dataclass fields + bound partial defaults | Program-instance identity; arm fingerprint |
| `bridge_source_hash` | source text (canonical) | AST-dump (docstring-stripped) + decorator kwargs (JSON-sorted) | Pre-registration commitment artifacts |

Three distinct APIs rather than one polymorphic `style=...`
dispatcher: the choice of which level a caller wants is a
*semantic* decision (cache-invalidation vs commitment artifact vs
program-instance identity), not a parameter of one operation.

B. **Signature walker** — `Exogenous` marker + `walk(claim)
ClaimSignature` + canonical-form helpers. Feeds the
`claim_graph_signature` primitive above and the framework's
`flatten_leaves` / `flatten_exogenous` consumers.

`Exogenous` is a sentinel class used as PEP 593 `Annotated`
metadata on claim kwargs. `Annotated[int, Exogenous]` declares
that a kwarg is something we generalize *over*, not intervene
on. Anything not so marked is implicitly a `leaf` — a
configurational scalar claim, interventionable by default.
Authors hide a leaf from intervention by baking it in via
`functools.partial`; the bake-in records honestly via
`canonical_str`'s partial branch.

The walker descends into:
- free-function claims (`FnClaim`): walk the wrapped fn's
  signature.
- frozen-dataclass instances (Modules, config bundles): walk
  `dataclasses.fields`.
- `functools.partial` over a claim: walk the wrapped claim's
  signature, but with each baked kwarg's default replaced by
  the bound value. Lets `partial(linear_epsilon, anneal_steps=
  50_000)` surface `anneal_steps=50_000` instead of the
  original default.

"Leaf" terminology: a leaf-regime kwarg is a configurational
scalar claim — a non-recursive value in the graph of claims
that's observed at composition time. RL practice calls these
"hyperparameters"; the framework uses "leaf" because the same
shape covers any non-RL configuration too. Authors who want to
hide a leaf from intervention bake it in via `functools.partial`."""
from __future__ import annotations

import ast
import dataclasses
import functools
import hashlib
import inspect
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from types import CodeType
from typing import (
    TYPE_CHECKING, Annotated, Literal, Protocol, TypeIs,
    get_origin, runtime_checkable,
)

from corroborate._internals.introspection import (
    get_attr_obj,
    get_field_default,
    get_field_default_factory,
    get_param_annotation,
    get_param_default,
    get_type_hints_obj,
    get_typing_args,
)
from corroborate.core.claim import Claim, FnClaim

if TYPE_CHECKING:
    from corroborate.bridge.bridge import Bridge


def root_claim_name(value: object) -> str | None:
    """The name of the root `@claim` at the bottom of a composed
    callable. A cell's program identity: the runner receives
    `claim = apply_interventions(partial(reg.fn(program), **hps),
    arm_iv)` — a `functools.partial` chain bottoming out in the root
    program's `FnClaim`. Unwrap the `.func` chain and return that
    claim's `name` (`'dqn'`, `'paired_dqn'`, …). `None` if the chain
    doesn't end in a named `Claim` (e.g. a bare lambda) — callers
    stamp the typed-but-absent program as `None`."""
    root: object = value
    while isinstance(root, functools.partial):
        root = root.func
    if isinstance(root, Claim):
        return root.name
    return None


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
    except (TypeError, ValueError):
        return ClaimSignature(name=name, kwargs=())
    hints = get_type_hints_obj(fn)

    kwargs_out: list[KwargInfo] = []
    for param_name, param in sig.parameters.items():
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if param_name in ('self', 'cls'):
            continue
        annotation: object = hints.get(param_name, get_param_annotation(param))
        regime = _regime_from_annotation(annotation)
        base_annotation = _strip_annotated(annotation)

        baked_value = baked.get(param_name, _MISSING)
        sig_default = get_param_default(param)
        has_default = (
            baked_value is not _MISSING
            or sig_default is not inspect.Parameter.empty
        )
        # Data-flow filter at depth >= 1.
        if depth >= 1 and not has_default and regime != 'exogenous':
            continue

        default = baked_value if baked_value is not _MISSING else sig_default
        inner: ClaimSignature | None = (
            _walk(default, depth=depth + 1) if _is_recursable(default) else None
        )
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
    if not is_dataclass(instance) or isinstance(instance, type):
        # Defensive: callers (`_walk`) gate this with the same
        # check, but inline narrowing here lets `fields(instance)`
        # accept the typed argument without falling back to
        # `DataclassInstance`-incompatible `object`.
        return ClaimSignature(name=type(instance).__name__, kwargs=())
    cls = type(instance)
    hints = get_type_hints_obj(cls)

    kwargs_out: list[KwargInfo] = []
    for f in fields(instance):
        annotation: object = hints.get(f.name, f.type)
        regime = _regime_from_annotation(annotation)
        base_annotation = _strip_annotated(annotation)
        value: object = get_attr_obj(instance, f.name)
        inner: ClaimSignature | None = (
            _walk(value, depth=depth + 1) if _is_recursable(value) else None
        )
        # Dataclass field's declared default — used by canonical
        # form elision. `MISSING` is the sentinel for "no default
        # declared"; the boundary returns it (or the factory's
        # output) typed as `object`, so `is`-comparison narrows
        # cleanly.
        field_default = get_field_default(f)
        if field_default is not dataclasses.MISSING:
            sig_default: object = field_default
            try:
                is_at_default = value is sig_default or bool(value == sig_default)
            except (TypeError, ValueError):
                is_at_default = value is sig_default
        elif (factory := get_field_default_factory(f)) is not None:
            try:
                sig_default = factory()
                is_at_default = value is sig_default or bool(value == sig_default)
            except (TypeError, ValueError):
                # Narrow to the equality-comparison failure modes
                # we actually expect (TypeError on non-comparable
                # types, ValueError on numpy-array shape mismatch).
                # A bare `Exception` here would silently mask
                # authoring bugs in the factory itself
                # (ZeroDivisionError, ImportError, AttributeError);
                # those should propagate so canonical-form
                # fingerprints don't silently diverge for
                # structurally-equal claims.
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
        for meta in get_typing_args(ann)[1:]:
            if _is_exogenous_marker(meta):
                return 'exogenous'
    return 'leaf'


def _strip_annotated(ann: object) -> object:
    if get_origin(ann) is Annotated:
        return get_typing_args(ann)[0]
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


# ============ Bytecode source hash (closure cache-invalidation) ============


def bytecode_source_hash(code: CodeType) -> str:
    """16-hex-char SHA-256 prefix over (`co_code`, `co_consts`,
    `co_names`). Recurses into nested code objects in `co_consts`
    so a constant edit inside a lambda / comprehension / inner def
    still flips the hash.

    Choose this when: you have a Python `CodeType` (function /
    lambda / comprehension bytecode) and want to detect any
    bytecode-level edit — including literal-constant changes that
    don't affect the AST shape — without paying the source-text
    parse cost.

    Why these three fields: `co_code` alone misses constant-only
    edits (`return 1.0` → `return 2.0` produces identical opcode
    streams; the literal lives in `co_consts` indexed by an opcode
    arg). `co_names` misses changes to external references
    (`np.mean` → `np.nanmean`). Local var renames live in
    `co_varnames` and are deliberately NOT hashed — cosmetic edits
    shouldn't bust the cache.

    Limit: Python compiler optimisations (constant folding) may
    affect the hash across Python versions; pin the interpreter
    version in `pyproject.toml`.

    Used by `Measurable.signature()` per-function and itself
    recursively for nested code."""
    h = hashlib.sha256()
    h.update(code.co_code)
    h.update(b'|names=')
    h.update('\x00'.join(code.co_names).encode())
    h.update(b'|consts=')
    for const in code.co_consts:
        if isinstance(const, CodeType):
            h.update(b'<code:' + bytecode_source_hash(const).encode() + b'>')
        else:
            # `repr` is stable across Python versions for the
            # primitives that show up in `co_consts` (numbers,
            # strings, tuples, frozensets, None, bytes).
            h.update(repr(const).encode())
        h.update(b'\x00')
    return h.hexdigest()[:16]


# ============ Bridge source hash (pre-registration manifests) ============


def bridge_source_hash(bridge: 'Bridge') -> str:
    """Hex-string SHA-256 over (AST of bridge `holds_when` source
    with docstrings stripped) + (JSON of decorator kwargs sorted
    by key).

    Choose this when: you want to identify the source-text-
    canonical content of a `@claim_bridge`-decorated bridge for
    pre-registration manifests or similar commitment artifacts.
    The AST canonicalisation makes it robust to reformat passes
    (black / ruff) and docstring edits; the decorator-kwargs
    serialisation captures the bound predicate parameters so a
    `harm_floor=0.3 → 0.5` literal edit OR a scope-expression
    edit flips the hash.

    Decorator kwargs serialised: `predicted_direction`, `source` /
    `target` (canonicalised via the framework's `endpoint_name`
    helper — strings pass through, Measurables surface `.name`,
    DoEffects surface the canonical `do(treatment|vs=baseline)`
    graph-render string), `direction`, `tier`, `scope`
    (`str(pl.Expr)`).

    Limit: `str(pl.Expr)` is not formally stable across polars
    versions. The framework's pyproject pins polars; consumers who
    upgrade polars MUST re-write their manifest (the audit's
    source-hash check will otherwise drift on a no-op upgrade)."""
    if bridge.holds_when is None:
        raise ValueError(
            f'bridge_source_hash: bridge {bridge.name!r} has no '
            f'holds_when body; bridges constructed via '
            f'`@claim_bridge` always carry one. Refusing to hash '
            f'a body-less Bridge.',
        )
    # Deferred import breaks the circular chain `core.signature` ←
    # `bridge.bridge` ← `measurables` ← `core.signature` (via
    # `Measurable.signature()` consuming `bytecode_source_hash`).
    # The deferred resolution only fires when `bridge_source_hash`
    # is actually called, by which time the full module graph is
    # constructed.
    from corroborate.bridge.bridge import endpoint_name

    src = inspect.getsource(bridge.holds_when)
    parsed = ast.parse(src)
    # Strip docstrings from every FunctionDef / ClassDef / Module
    # body before dumping — docstrings are documentation, not
    # behaviour. The contract is that cosmetic edits (whitespace,
    # comments, docstrings) must not bust the hash while semantic
    # edits (literal values, expressions, control flow) must.
    _strip_docstrings(parsed)
    ast_repr = ast.dump(
        parsed, annotate_fields=True, include_attributes=False,
    )
    pd = bridge.predicted_direction
    decorator_kwargs: dict[str, str] = {
        'predicted_direction': pd if pd is not None else 'null',
        # `endpoint_name` is the single laundering point for the
        # `BridgeEndpoint = str | Measurable | DoEffect` union —
        # str passthrough; Measurable → `.name`; DoEffect →
        # `node_key()` (the canonical `do(treatment|vs=baseline)`
        # graph-render string). Routing through it keeps the
        # serialised form in lockstep with every other consumer of
        # the union (graph builder, cache walker).
        'source': endpoint_name(bridge.source),
        'target': endpoint_name(bridge.target),
        'direction': bridge.direction.value,
        # Tier is an IntEnum — `.value` returns int. Use `.name`
        # for a stable string ('INVARIANT' / 'ASSOCIATIONAL' /
        # 'INTERVENTIONAL') so the json.dumps payload's dict type
        # stays `dict[str, str]`.
        'tier': bridge.tier.name,
        # Polars Expr / DeferredScope / None — str() is the only
        # stable serialisation available without polars-version
        # coupling we don't already inherit.
        'scope': str(bridge.scope),
    }
    payload = ast_repr + '\n' + json.dumps(decorator_kwargs, sort_keys=True)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _strip_docstrings(tree: ast.AST) -> None:
    """Walk an AST and drop the leading docstring node from every
    `Module`, `FunctionDef`, `AsyncFunctionDef`, and `ClassDef`
    body. Mutates in place — caller passes the parsed tree
    directly, then calls `ast.dump`.

    The contract: a docstring edit is a documentation change and
    must NOT flip the source hash. A code change (literal,
    expression, control flow) does flip it. This walk is the
    canonicalisation step that makes both true."""
    for node in ast.walk(tree):
        if isinstance(node, (
            ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
        )):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                node.body = body[1:]


