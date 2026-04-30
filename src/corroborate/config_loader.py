"""YAML → Hypothesis builder, registry-resolved.

Two recursive value forms are recognised; everything else is a
leaf (passes through unchanged for scalars, recurses element-wise
for lists and plain mappings).

- `{class: <Name>, **kwargs}` — instantiate a registered Module
  Claim or container class with recursively-resolved kwargs.
- `{fn: <Name>, **kwargs}` — resolve a registered `FnClaim`. Bare
  `{fn: X}` (no kwargs) returns the FnClaim itself; with kwargs
  returns `partial(fn, **kwargs)`. Mirrors the Python-authoring
  pattern `partial(bootstrap, greedification=double_greedify)`.

The two magic keys (`class`, `fn`) are reserved at any depth; YAML
authors never need string-prefix sigils. List literals tuple-ify
to match `frozen=True, slots=True` ClaimBase fields like
`MLP.hidden: tuple[int, ...]`.

Round-trip contract: a YAML-loaded Hypothesis's slot values pass
`claim_graph_signature` equality with the equivalent Python-
authored Hypothesis. The signature is the canonical name of the
configured composition; if the YAML and Python paths produce
different signatures, the YAML schema is wrong and the loader
should refuse before the sweep launches."""
from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import TypeIs, cast

import yaml

from corroborate.hypothesis import Hypothesis, PredictedDirection
from corroborate.intervention import Intervention, Replacement
from corroborate.registry import Registry


_CLASS_KEY = 'class'
_FN_KEY = 'fn'


def _is_str_keyed_mapping(v: object) -> TypeIs[Mapping[str, object]]:
    """Narrow YAML's `Any`-ish output to a typed mapping. PyYAML's
    `safe_load` returns nested `dict | list | scalar`; without
    narrowing, downstream walks lose all element types."""
    if not isinstance(v, Mapping):
        return False
    return all(isinstance(k, str) for k in v.keys())


def _is_predicted_direction(v: object) -> TypeIs[PredictedDirection]:
    return isinstance(v, str) and v in (
        'a_gt_b', 'a_lt_b', 'two_sided',
    )


def _construct(cls: type, kwargs: Mapping[str, object]) -> object:
    """Boundary call: instantiate `cls` with YAML-resolved kwargs.
    Constructor signatures are dynamic at this layer (the loader
    doesn't know e.g. `Replay.__init__`'s arity), so the return
    type widens to `object`. Callers narrow at use-site
    (Intervention.replacement is typed as `Replacement`, etc.)."""
    return cast(object, cls(**kwargs))


def resolve(value: object, *, reg: Registry) -> object:
    """Recursive YAML node → Python value.

    Mappings with `class` or `fn` keys dispatch to the registry;
    other mappings recurse element-wise. Lists tuple-ify and
    recurse. Scalars pass through unchanged."""
    if _is_str_keyed_mapping(value):
        if _CLASS_KEY in value:
            return _resolve_class(value, reg=reg)
        if _FN_KEY in value:
            return _resolve_fn(value, reg=reg)
        return {k: resolve(v, reg=reg) for k, v in value.items()}
    if isinstance(value, list):
        # Tuple-ify: ClaimBase fields like MLP.hidden are typed
        # `tuple[int, ...]`; YAML lists must coerce.
        elements: list[object] = list(value)  # narrow Any → object
        return tuple(resolve(v, reg=reg) for v in elements)
    return value


def _resolve_class(
    node: Mapping[str, object], *, reg: Registry,
) -> object:
    name = node[_CLASS_KEY]
    if not isinstance(name, str):
        raise TypeError(
            f'`class` key must be a string token; got '
            f'{type(name).__name__}',
        )
    kwargs = {
        k: resolve(v, reg=reg)
        for k, v in node.items() if k != _CLASS_KEY
    }
    if name in reg.module_classes:
        return _construct(reg.module_class(name), kwargs)
    if name in reg.containers:
        return _construct(reg.container(name), kwargs)
    raise KeyError(
        f'no Module Claim or container named {name!r}; '
        f'modules={sorted(reg.module_classes)}, '
        f'containers={sorted(reg.containers)}',
    )


def _resolve_fn(
    node: Mapping[str, object], *, reg: Registry,
) -> object:
    name = node[_FN_KEY]
    if not isinstance(name, str):
        raise TypeError(
            f'`fn` key must be a string token; got '
            f'{type(name).__name__}',
        )
    fn = reg.fn(name)
    kwargs = {
        k: resolve(v, reg=reg)
        for k, v in node.items() if k != _FN_KEY
    }
    if not kwargs:
        return fn
    return partial(fn, **kwargs)


def load_hypothesis(
    path: Path, *, reg: Registry,
) -> Hypothesis[Mapping[str, object]]:
    """Build a Hypothesis from a YAML file. The file is one
    hypothesis per `path`; multi-hypothesis manifests are loaded
    by `load_hypotheses`."""
    with path.open() as f:
        raw = cast(object, yaml.safe_load(f))
    if not _is_str_keyed_mapping(raw):
        raise TypeError(
            f'top-level YAML must be a string-keyed mapping; got '
            f'{type(raw).__name__}',
        )
    return _build_hypothesis(raw, reg=reg)


def _build_hypothesis(
    node: Mapping[str, object], *, reg: Registry,
) -> Hypothesis[Mapping[str, object]]:
    name = node.get('name')
    if not isinstance(name, str):
        raise TypeError(
            f'hypothesis.name must be a string; got '
            f'{type(name).__name__}',
        )

    intervention_raw = node.get('intervention', {})
    if not _is_str_keyed_mapping(intervention_raw):
        raise TypeError(
            f'intervention must be a string-keyed mapping; got '
            f'{type(intervention_raw).__name__}',
        )
    intervention: dict[str, object] = {
        k: resolve(v, reg=reg)
        for k, v in intervention_raw.items()
    }

    direction_raw = node.get('predicted_direction')
    direction: PredictedDirection | None
    if direction_raw is None:
        direction = None
    elif _is_predicted_direction(direction_raw):
        direction = direction_raw
    else:
        raise ValueError(
            f'predicted_direction must be null/a_gt_b/a_lt_b/'
            f'two_sided; got {direction_raw!r}',
        )

    arms_raw = node.get('intervention_arms', [])
    if not isinstance(arms_raw, list):
        raise TypeError(
            f'intervention_arms must be a list; got '
            f'{type(arms_raw).__name__}',
        )
    arms_typed: list[object] = list(arms_raw)
    arms = tuple(_build_arm(a, reg=reg) for a in arms_typed)

    return Hypothesis(
        name=name,
        intervention=intervention,
        bridges=(),
        predicted_direction=direction,
        intervention_arms=arms,
    )


def _build_arm(node: object, *, reg: Registry) -> Intervention:
    if not _is_str_keyed_mapping(node):
        raise TypeError(
            f'intervention_arm must be a mapping; got '
            f'{type(node).__name__}',
        )
    slot_path = node.get('slot_path')
    if not isinstance(slot_path, str):
        raise TypeError(
            f'intervention_arm.slot_path must be a string; got '
            f'{type(slot_path).__name__}',
        )
    if 'replacement' not in node:
        raise KeyError('intervention_arm missing `replacement`')
    raw_repl = resolve(node['replacement'], reg=reg)
    # Replacement = ClaimBase | FnClaim | partial | Callable.
    # Resolve always returns a callable for class/fn-keyed nodes;
    # if YAML authored a non-callable replacement (a leaf scalar),
    # that's a config error caught at construction time.
    if not callable(raw_repl):
        raise TypeError(
            f'intervention_arm.replacement must resolve to a '
            f'callable; got {type(raw_repl).__name__}',
        )
    replacement = cast(Replacement, raw_repl)
    return Intervention(slot_path=slot_path, replacement=replacement)


__all__ = ['load_hypothesis', 'resolve']
