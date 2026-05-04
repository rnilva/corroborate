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
to match `frozen=True, slots=True` config-bundle fields like
`MLP.hidden: tuple[int, ...]`.

Round-trip contract: a YAML-loaded Hypothesis's slot values are
structurally equal (frozen-dataclass `==` on config bundles,
identity on FnClaim references) to the equivalent Python-authored
Hypothesis.
The smokes assert this; drift means the YAML schema diverged from
the Python authoring shape and the loader should refuse before
the sweep launches."""
from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import TypeIs

from collections.abc import Mapping as _Mapping
from dataclasses import dataclass, field

from corroborate._internals.yaml import safe_load as _yaml_load
from corroborate.core.hypothesis import PredictedDirection
from corroborate.core.intervention import Intervention, is_replacement
from corroborate.runner.registry import Registry


@dataclass(frozen=True, slots=True)
class HypothesisConfig:
    """YAML-loaded hypothesis configuration. A substrate-coupled
    intermediate — the substrate's `dispatch_sweep` decomposes it
    into a Hypothesis Protocol-conformer + a `base` Callable.

    Carries:
    - `name`: substrate-chosen short label (for arm_tag output naming).
    - `intervention`: HP scalars + slot-Claim swaps, as a flat dict.
      The substrate splits HPs (slot paths NOT in `intervention_arms`)
      into the `base` callable's bound kwargs; mechanism swaps come
      via `intervention_arms`.
    - `intervention_arms`: typed structural deltas (mechanism swaps
      only). Becomes `DoEffect.treatment` on the substrate's
      Hypothesis Protocol-conformer.
    - `predicted_direction`: optional sign-prior; per-bridge
      `Bridge.predicted_direction` is the canonical home, but
      substrate authors can default it here for legacy YAML configs."""
    name: str
    intervention: _Mapping[str, object]
    predicted_direction: PredictedDirection | None = None
    intervention_arms: tuple[Intervention, ...] = field(default_factory=tuple)

    def arm_key(self) -> str:
        """Canonical fingerprint of `intervention_arms`. Empty tuple →
        `'baseline'`. Substrate's `dispatch_sweep` derives the same
        value via `DoEffect(treatment=intervention_arms,
        baseline=()).treatment_arm_key()` — both flow through
        `combined_arm_key`."""
        from corroborate.core.intervention import combined_arm_key
        return combined_arm_key(self.intervention_arms)


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


def _is_predicted_direction(v: object) -> TypeIs[PredictedDirection]:
    return isinstance(v, str) and v in (
        'a_gt_b', 'a_lt_b', 'two_sided',
    )


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
    `{from_env: <attr>}` placeholders in paired mode. When
    `None` (chunked default), encountering a `from_env` mapping
    raises — placeholders only make sense inside paired-mode
    dispatch where the env context is known."""
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
            "paired-mode dispatch (`arms_shape: 'paired'`).",
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


def load_hypothesis(
    path: Path, *, reg: Registry,
) -> HypothesisConfig:
    """Build a Hypothesis from a YAML file. The file is one
    hypothesis per `path`; multi-hypothesis sweeps are loaded
    by the substrate's own dispatcher."""
    with path.open() as f:
        raw = _yaml_load(f)
    if not is_str_keyed_mapping(raw):
        raise TypeError(
            f'top-level YAML must be a string-keyed mapping; got '
            f'{type(raw).__name__}',
        )
    return build_hypothesis_from_mapping(raw, reg=reg)


def build_hypothesis_from_mapping(
    node: Mapping[str, object],
    *,
    reg: Registry,
    env_attrs: Mapping[str, object] | None = None,
) -> HypothesisConfig:
    """Public path-into-loader for callers (e.g. the RL sweep
    dispatcher) that already have the parsed mapping in hand and
    just need it turned into a Hypothesis. `load_hypothesis`
    delegates here after `yaml.safe_load`.

    `env_attrs` is forwarded to `resolve` so the loader can
    substitute `{from_env: <attr>}` placeholders during
    paired-mode dispatch."""
    name = node.get('name')
    if not isinstance(name, str):
        raise TypeError(
            f'hypothesis.name must be a string; got '
            f'{type(name).__name__}',
        )

    intervention_raw = node.get('intervention', {})
    if not is_str_keyed_mapping(intervention_raw):
        raise TypeError(
            f'intervention must be a string-keyed mapping; got '
            f'{type(intervention_raw).__name__}',
        )
    intervention: dict[str, object] = {
        k: resolve(v, reg=reg, env_attrs=env_attrs)
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
    arms = tuple(
        _build_arm(a, reg=reg, env_attrs=env_attrs)
        for a in arms_typed
    )

    return HypothesisConfig(
        name=name,
        intervention=intervention,
        predicted_direction=direction,
        intervention_arms=arms,
    )


def _build_arm(
    node: object,
    *,
    reg: Registry,
    env_attrs: Mapping[str, object] | None,
) -> Intervention:
    if not is_str_keyed_mapping(node):
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
    raw_repl = resolve(node['replacement'], reg=reg, env_attrs=env_attrs)
    if not is_replacement(raw_repl):
        raise TypeError(
            f'intervention_arm.replacement must resolve to a '
            f'callable; got {type(raw_repl).__name__}',
        )
    return Intervention(slot_path=slot_path, replacement=raw_repl)


__all__ = [
    'build_hypothesis_from_mapping',
    'is_str_keyed_mapping',
    'load_hypothesis',
    'resolve',
]
