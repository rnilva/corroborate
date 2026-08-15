"""Substrate-agnostic name → typed-handle registry.

YAML- or config-driven sweep authoring needs to map string tokens
(`'double_greedify'`, `'Replay'`) back to the typed Python handle
the implementation's `Claim` graph holds. This module is that map.

Two surfaces:

- `fns` — `Registry[FnClaim[..., object]]`. `@claim`-decorated
  free functions. Key is `FnClaim._name` (== wrapped function
  `__name__`). YAML uses these to fill slot bindings, e.g.
  `bootstrap.greedification: double_greedify`.

- `classes` — `Registry[type]`. Frozen-dataclass config bundles
  (`Replay`, `MLP`, `CNN`), ready to instantiate with YAML
  kwargs.

Both surfaces share `corroborate._registry.Registry[T]`; this
module is the substrate-facing facade that adds `add_module` /
`add_modules` walker convenience plus the `fn(name)` / `cls(name)`
loud-`KeyError` accessors YAML loaders surface as config errors.

`add_module(module)` walks `vars(module)` and indexes every
`FnClaim` instance and every frozen-dataclass class it finds
(config bundles). NamedTuple record types (`Transition`, `Batch`,
...) are filtered out via the `tuple`-subclass check. Authors who
keep their bundles in the same module as their `@claim` functions
get auto-discovery for free; explicit `add_class` remains for
cross-module imports.

Name collisions raise `ValueError` at registration time; the
implementation fixes the ambiguity by renaming. Lookups raise
`KeyError` with the missing name; YAML loaders surface this as
a config error pointing at the offending token.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from dataclasses import dataclass, field
from importlib import import_module
from types import ModuleType

from corroborate._internals.registry import Registry as _Registry
from corroborate.core.claim import FnClaim


@dataclass(slots=True)
class Registry:
    """Two typed maps from string token to Python handle.

    `fns` and `classes` each wrap `corroborate._registry.Registry`
    so collision and lookup discipline is centralised in one
    place; this class adds the `add_module` walker and the
    loud-`KeyError` accessors implementation code consumes."""

    fns: _Registry[FnClaim[..., object]] = field(
        default_factory=_Registry,
    )
    classes: _Registry[type] = field(default_factory=_Registry)

    def add_module(self, module: ModuleType) -> None:
        """Index every `FnClaim` instance AND every frozen-
        dataclass class found in `vars(module)`. Skips
        dunder/private names and NamedTuple record types
        (filtered via the `tuple`-subclass check).

        Config bundles (`Replay`, `MLP`, `CNN`) auto-discover via
        `dataclasses.is_dataclass`. Pure record types
        (`Transition`, `Batch`, `ReplayState`) are NamedTuples
        and skipped.

        Re-adding the same value at the same name is a no-op;
        adding a *different* value at an already-present name
        raises `ValueError`."""
        # `vars()` is `dict[str, Any]` — narrow via `object` and
        # let isinstance flow into the typed branches below.
        module_namespace: dict[str, object] = dict(vars(module))
        for attr_name, value in module_namespace.items():
            if attr_name.startswith('_'):
                continue
            if isinstance(value, FnClaim):
                # Type narrowing: value is an `FnClaim[Any, Any]`
                # instance after isinstance; `name` is a typed
                # field. Annotating locally makes pyright happy.
                fn_value: FnClaim[..., object] = value
                self.fns.register(fn_value.name, fn_value)
            elif (
                isinstance(value, type)
                and dataclasses.is_dataclass(value)
                and not issubclass(value, tuple)
            ):
                # Frozen-dataclass config bundle (e.g. `Replay`,
                # `MLP`, `CNN`). NamedTuples (`Transition`,
                # `Batch`) are tuple-subclasses and skipped.
                self.classes.register(value.__name__, value)

    def add_modules(self, module_names: Iterable[str]) -> None:
        """Convenience: import each name and `add_module`. Order
        is irrelevant — collisions across modules raise either
        way."""
        for name in module_names:
            self.add_module(import_module(name))

    def add_class(self, cls: type) -> None:
        """Register a class at its `__name__`. Idempotent on the
        same class; raises on a different class at the same key.

        Used both by `add_module` (auto-discovered config-bundle
        classes) and by the implementation (manual cross-module
        registration when a bundle lives outside the implementation's
        claim modules)."""
        self.classes.register(cls.__name__, cls)

    def fn(self, name: str) -> FnClaim[..., object]:
        """Resolve `name` to the `FnClaim` registered for it.
        `KeyError` if absent."""
        v = self.fns.get(name)
        if v is None:
            raise KeyError(
                f'no FnClaim named {name!r}; '
                f'known: {self.fns.names()}',
            )
        return v

    def cls(self, name: str) -> type:
        """Resolve `name` to a registered config-bundle class.
        `KeyError` if absent."""
        v = self.classes.get(name)
        if v is None:
            raise KeyError(
                f'no class named {name!r}; '
                f'known: {self.classes.names()}',
            )
        return v


__all__ = ['Registry']
