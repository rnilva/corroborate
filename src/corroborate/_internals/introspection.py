"""Python introspection boundary — laundering point for stdlib's
reflection APIs (`typing.get_type_hints`, `typing.get_args`,
`inspect.Parameter.default`, `dataclasses.Field.default`,
`functools.partial.args/keywords`).

Each of these returns `Any` because Python's runtime reflection
traffics in arbitrary user-supplied type expressions and values
— the framework's `object`-as-upper-bound discipline is the
honest type, narrower than `Any` and forbidding casual reads.

Same shape as `_json_boundary.py` / `_polars_boundary.py` /
`_yaml_boundary.py`. Each function laundering an `Any`-typed
return is the framework's ONE allowed escape hatch for that
reflection surface — the runtime invariant (arbitrary user types
/ values) cannot be expressed in stdlib's static contract.

Many returns drop the ignore because covariant containers
(`Mapping[str, object]` over `dict[str, Any]`,
`tuple[object, ...]` over `tuple[Any, ...]`) absorb the `Any` at
the function boundary. The remaining ignores cover positions
where the value flows through directly (`p.default` is `Any`
in the non-`empty` arm, etc.).

Module name is underscore-prefixed to signal **internal use
only**. External users should `import inspect` / `import typing` /
`import dataclasses` directly."""
from __future__ import annotations

import dataclasses
import functools
import inspect
import typing
from collections.abc import Callable, Mapping


def get_type_hints_obj(
    obj: Callable[..., object] | type, /,
) -> Mapping[str, object]:
    """`typing.get_type_hints(obj, include_extras=True)` returns
    `dict[str, Any]`. The widen to `Mapping[str, object]` lands
    cleanly via `Mapping`'s value covariance. Returns an empty
    mapping when introspection raises (unresolvable forward refs)."""
    try:
        return typing.get_type_hints(obj, include_extras=True)
    except (TypeError, NameError, ValueError):
        return {}


def get_param_annotation(p: inspect.Parameter) -> object:
    """`inspect.Parameter.annotation` is `Any` — the source-level
    annotation expression is arbitrary. Returns
    `inspect.Parameter.empty` when no annotation is declared, so
    callers can compare with `is`. Single-site Any laundering at
    the read; same shape as `get_param_default`."""
    annotation: object = p.annotation
    if annotation is inspect.Parameter.empty:
        return inspect.Parameter.empty
    return annotation


def get_param_default(p: inspect.Parameter) -> object:
    """`inspect.Parameter.default` is `Any`; returns
    `inspect.Parameter.empty` (a typed sentinel `type[_empty]`)
    when no default is declared. Callers compare against
    `inspect.Parameter.empty` with `is` to detect the no-default
    case before consuming the value.

    `p.default`'s `Any` type triggers `reportAny` even on the `is`
    comparison. Single-site laundering: capture once, then narrow
    via the sentinel check."""
    default: object = p.default
    if default is inspect.Parameter.empty:
        return inspect.Parameter.empty
    return default


def get_field_default(f: dataclasses.Field[object], /) -> object:
    """`dataclasses.Field.default` is `Any |
    Literal[_MISSING_TYPE.MISSING]`. Returns `dataclasses.MISSING`
    when no default is declared. Callers compare against
    `dataclasses.MISSING` with `is` to detect the no-default case."""
    if f.default is dataclasses.MISSING:
        return dataclasses.MISSING
    # After narrowing-out MISSING, the remaining union arm is
    # `Any` — pyright happens to widen it to `object` at the return
    # boundary here, no ignore needed.
    return f.default


def get_field_default_factory(
    f: dataclasses.Field[object], /,
) -> Callable[[], object] | None:
    """`Field.default_factory` is `_DefaultFactory[Any] |
    Literal[_MISSING_TYPE.MISSING]`. Returns `None` when no factory
    is declared so callers can branch on a typed `is None`."""
    factory = f.default_factory
    if factory is dataclasses.MISSING:
        return None
    return factory


def get_typing_args(ann: object) -> tuple[object, ...]:
    """`typing.get_args` returns `tuple[Any, ...]`; covariant
    `tuple[object, ...]` widening lands without an ignore."""
    return typing.get_args(ann)


def get_partial_args(
    p: functools.partial[object], /,
) -> tuple[object, ...]:
    """`functools.partial.args` is `tuple[Any, ...]` — bound
    positional args carry the wrapped callable's parameter types,
    which the partial type itself doesn't preserve."""
    return p.args


def get_partial_keywords(
    p: functools.partial[object], /,
) -> Mapping[str, object]:
    """`functools.partial.keywords` is `dict[str, Any]`. Covariant
    widen to `Mapping[str, object]` lands without an ignore."""
    return p.keywords


def get_bound_arguments(
    bound: inspect.BoundArguments,
) -> Mapping[str, object]:
    """`BoundArguments.arguments` is `OrderedDict[str, Any]` —
    bound argument values carry the wrapped function's parameter
    types, which `BoundArguments` itself doesn't preserve. Covariant
    widen to `Mapping[str, object]` lands without an ignore."""
    return bound.arguments


def get_attr_obj(obj: object, name: str) -> object:
    """`getattr(obj, name)` returns `Any` because the runtime
    attribute set isn't statically known. The framework needs this
    only for dynamic field-name lookup on dataclass instances
    (`fields(instance)` yields `Field.name: str`, not a literal);
    direct `instance.<name>` syntax is preferred everywhere else
    per CLAUDE.md's getattr/setattr rule."""
    return getattr(obj, name)
