"""YAML → InterventionConfig builder, registry-resolved.

Substrate-coupled: the YAML schema (`name`, `base`, `arms`) is a
substrate-authoring convention, not a framework typed contract.
The framework's hypothesis surface is the `Hypothesis` Protocol
(`INTERVENTION: DoEffect`, `BRIDGES: tuple[Bridge, ...]`);
`InterventionConfig` is the intermediate the substrate's
`dispatch_sweep` decomposes into a Protocol-conformer + a `base`
callable.

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
to match `frozen=True, slots=True` config-bundle fields like
`MLP.hidden: tuple[int, ...]`.

Round-trip contract: a YAML-loaded InterventionConfig's slot values
are structurally equal (frozen-dataclass `==` on config bundles,
identity on FnClaim references) to the equivalent Python-authored
InterventionConfig.
The smokes assert this; drift means the YAML schema diverged from
the Python authoring shape and the loader should refuse before
the sweep launches."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import TypeIs

from corroborate._internals.yaml import safe_load as _yaml_load
from corroborate.core.intervention import DoEffect, Intervention, is_replacement
from corroborate.runner.registry import Registry


@dataclass(frozen=True, slots=True)
class InterventionConfig:
    """YAML-loaded intervention configuration. A substrate-coupled
    intermediate — the substrate's `dispatch_sweep` decomposes it
    into a Hypothesis Protocol-conformer + a `base` Callable.

    Carries:
    - `name`: substrate-chosen short label (for arm_tag output naming).
    - `base`: HP scalars + slot-Claim bindings, as a flat dict.
      Becomes the bound kwargs of `partial(dqn, **base)` — the SCM
      the `do_effect`'s `do(·)` operators act on. Empty arms inherit
      these values; non-empty arms override at the configured slot
      path.
    - `do_effect`: typed multi-arm contrast (framework primitive).
      `do_effect.arms` is the `tuple[tuple[Intervention, ...], ...]`
      of slot replacements; empty-tuple arm is the Pearl-style "no
      intervention" control. Default `DoEffect(arms=((),))` (single
      empty arm) supports the shared-mode "this template is one arm
      in a multi-template sweep" pattern."""
    name: str
    base: Mapping[str, object]
    do_effect: DoEffect = field(
        default_factory=lambda: DoEffect(arms=((),)),
    )


_CLASS_KEY = 'class'
_FN_KEY = 'fn'
_FROM_ENV_KEY = 'from_env'


def is_str_keyed_mapping(v: object) -> TypeIs[Mapping[str, object]]:
    """Narrow YAML's `Any`-ish output to a typed mapping. PyYAML's
    `safe_load` returns nested `dict | list | scalar`; without
    narrowing, downstream walks lose all element types."""
    if not isinstance(v, Mapping):
        return False
    return all(isinstance(k, str) for k in v.keys())


def _construct(cls: type, kwargs: Mapping[str, object]) -> object:
    """Boundary call: instantiate `cls` with YAML-resolved kwargs.
    Constructor signatures are dynamic at this layer (the loader
    doesn't know e.g. `Replay.__init__`'s arity), so the return
    type widens to `object`. Callers narrow at use-site
    (Intervention.replacement is typed as `Replacement`, etc.).

    `type.__call__` returns `Any` in typeshed because the
    constructor return type is bound to the unparameterised `type`
    here; widen to `object` once at this boundary."""
    return cls(**kwargs)  # pyright: ignore[reportAny]


def resolve(
    value: object,
    *,
    reg: Registry,
    env_attrs: Mapping[str, object] | None = None,
) -> object:
    """Recursive YAML node → Python value.

    Mappings with `class`, `fn`, or `from_env` keys dispatch to
    the registry / env-attribute lookup; other mappings recurse
    element-wise. Lists tuple-ify and recurse. Scalars pass
    through unchanged.

    `env_attrs` is the per-env attribute map used to resolve
    `{from_env: <attr>}` placeholders in per-env mode. When
    `None` (shared default), encountering a `from_env` mapping
    raises — placeholders only make sense inside per-env dispatch
    where the env context is known."""
    if is_str_keyed_mapping(value):
        if _FROM_ENV_KEY in value:
            return _resolve_from_env(value, env_attrs=env_attrs)
        if _CLASS_KEY in value:
            return _resolve_class(
                value, reg=reg, env_attrs=env_attrs,
            )
        if _FN_KEY in value:
            return _resolve_fn(value, reg=reg, env_attrs=env_attrs)
        return {
            k: resolve(v, reg=reg, env_attrs=env_attrs)
            for k, v in value.items()
        }
    if isinstance(value, list):
        # Tuple-ify: bundle fields like MLP.hidden are typed
        # `tuple[int, ...]`; YAML lists must coerce.
        elements: list[object] = list(value)  # narrow Any → object
        return tuple(
            resolve(v, reg=reg, env_attrs=env_attrs) for v in elements
        )
    return value


def _resolve_from_env(
    node: Mapping[str, object],
    *,
    env_attrs: Mapping[str, object] | None,
) -> object:
    """Resolve a `{from_env: <attr>}` placeholder against
    `env_attrs`. The mapping must contain ONLY the `from_env`
    key — no peer kwargs — so the substitution is unambiguous."""
    if env_attrs is None:
        raise ValueError(
            '`from_env` reference encountered but no env context '
            'provided; this placeholder is only valid in '
            "per-env dispatch (`env_binding: 'per_env'`).",
        )
    if len(node) != 1:
        raise TypeError(
            f'`from_env` mapping must contain exactly one key; '
            f'got {sorted(node)}',
        )
    attr = node[_FROM_ENV_KEY]
    if not isinstance(attr, str):
        raise TypeError(
            f'`from_env` value must be a string; got '
            f'{type(attr).__name__}',
        )
    if attr not in env_attrs:
        raise KeyError(
            f'env attribute {attr!r} not in env_attrs '
            f'(known: {sorted(env_attrs)})',
        )
    return env_attrs[attr]


def _resolve_class(
    node: Mapping[str, object],
    *,
    reg: Registry,
    env_attrs: Mapping[str, object] | None,
) -> object:
    name = node[_CLASS_KEY]
    if not isinstance(name, str):
        raise TypeError(
            f'`class` key must be a string token; got '
            f'{type(name).__name__}',
        )
    kwargs = {
        k: resolve(v, reg=reg, env_attrs=env_attrs)
        for k, v in node.items() if k != _CLASS_KEY
    }
    return _construct(reg.cls(name), kwargs)


def _resolve_fn(
    node: Mapping[str, object],
    *,
    reg: Registry,
    env_attrs: Mapping[str, object] | None,
) -> object:
    name = node[_FN_KEY]
    if not isinstance(name, str):
        raise TypeError(
            f'`fn` key must be a string token; got '
            f'{type(name).__name__}',
        )
    fn = reg.fn(name)
    kwargs = {
        k: resolve(v, reg=reg, env_attrs=env_attrs)
        for k, v in node.items() if k != _FN_KEY
    }
    if not kwargs:
        return fn
    return partial(fn, **kwargs)


def load_intervention(
    path: Path, *, reg: Registry,
) -> InterventionConfig:
    """Build an InterventionConfig from a YAML file. The file is
    one intervention per `path`; multi-intervention sweeps are
    loaded by the substrate's own dispatcher."""
    with path.open() as f:
        raw = _yaml_load(f)
    if not is_str_keyed_mapping(raw):
        raise TypeError(
            f'top-level YAML must be a string-keyed mapping; got '
            f'{type(raw).__name__}',
        )
    return build_intervention_from_mapping(raw, reg=reg)


def build_intervention_from_mapping(
    node: Mapping[str, object],
    *,
    reg: Registry,
    env_attrs: Mapping[str, object] | None = None,
) -> InterventionConfig:
    """Public path-into-loader for callers (e.g. the RL sweep
    dispatcher) that already have the parsed mapping in hand and
    just need it turned into an InterventionConfig.
    `load_intervention` delegates here after `yaml.safe_load`.

    `env_attrs` is forwarded to `resolve` so the loader can
    substitute `{from_env: <attr>}` placeholders during per-env
    dispatch.

    `name` field supports string-template substitution: occurrences
    of `{from_env: <attr>}` are replaced with the corresponding
    env_attr value at per-env dispatch time. Required when
    `env_binding: per_env` produces multiple InterventionConfig
    instances that would otherwise collide on `cfg.name` (one
    config per env)."""
    import re
    name = node.get('name')
    if not isinstance(name, str):
        raise TypeError(
            f'intervention.name must be a string; got '
            f'{type(name).__name__}',
        )
    if env_attrs is not None and '{from_env:' in name:
        def _sub_match(match: 're.Match[str]') -> str:
            key = match.group(1).strip()
            if key not in env_attrs:
                raise KeyError(
                    f'intervention.name template references env attr '
                    f'{key!r} not in env_attrs '
                    f'(known: {sorted(env_attrs)})',
                )
            return str(env_attrs[key])
        name = re.sub(r'\{from_env:\s*([^}]+)\}', _sub_match, name)

    base_raw = node.get('base', {})
    if not is_str_keyed_mapping(base_raw):
        raise TypeError(
            f'intervention.base must be a string-keyed mapping; got '
            f'{type(base_raw).__name__}',
        )
    base: dict[str, object] = {
        k: resolve(v, reg=reg, env_attrs=env_attrs)
        for k, v in base_raw.items()
    }

    arms_raw = node.get('arms')
    arms: tuple[tuple[Intervention, ...], ...]
    if arms_raw is None:
        arms = ((),)
    else:
        if not isinstance(arms_raw, list):
            raise TypeError(
                f'intervention.arms must be a list of arms (each '
                f'arm a list of slot-replacement dicts, or [] for '
                f'the empty control arm); got '
                f'{type(arms_raw).__name__}',
            )
        arms_list: list[tuple[Intervention, ...]] = []
        for i, arm_raw in enumerate(arms_raw):
            if not isinstance(arm_raw, list):
                raise TypeError(
                    f'intervention.arms[{i}] must be a list of '
                    f'slot-replacement dicts (or [] for empty '
                    f'control); got {type(arm_raw).__name__}',
                )
            arm = tuple(
                _build_arm(a, reg=reg, env_attrs=env_attrs)
                for a in arm_raw
            )
            arms_list.append(arm)
        arms = tuple(arms_list)

    return InterventionConfig(
        name=name,
        base=base,
        do_effect=DoEffect(arms=arms),
    )


def _build_arm(
    node: object,
    *,
    reg: Registry,
    env_attrs: Mapping[str, object] | None,
) -> Intervention:
    if not is_str_keyed_mapping(node):
        raise TypeError(
            f'arm entry must be a mapping; got '
            f'{type(node).__name__}',
        )
    slot_path = node.get('slot_path')
    if not isinstance(slot_path, str):
        raise TypeError(
            f'arm.slot_path must be a string; got '
            f'{type(slot_path).__name__}',
        )
    if 'replacement' not in node:
        raise KeyError('arm missing `replacement`')
    raw_repl = resolve(node['replacement'], reg=reg, env_attrs=env_attrs)
    if not is_replacement(raw_repl):
        raise TypeError(
            f'arm.replacement must resolve to a callable; got '
            f'{type(raw_repl).__name__}',
        )
    return Intervention(slot_path=slot_path, replacement=raw_repl)


__all__ = [
    'InterventionConfig',
    'build_intervention_from_mapping',
    'is_str_keyed_mapping',
    'load_intervention',
    'resolve',
]
