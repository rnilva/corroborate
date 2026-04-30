"""Substrate-agnostic name → typed-handle registry.

YAML- or config-driven sweep authoring needs to map string tokens
(`'double_greedify'`, `'Replay'`) back to the typed Python handle
the substrate's `Claim` graph holds. The registry is that map.

Three surfaces, kept separate so callers don't widen the return
type to a union and re-introduce the type erasure CLAUDE.md treats
as load-bearing:

- `fns` — `dict[str, FnClaim[..., object]]`. `@claim`-decorated
  free functions. Key is `FnClaim._name` (== wrapped function
  `__name__`). YAML uses these to fill slot bindings, e.g.
  `bootstrap.greedification: double_greedify`.

- `module_classes` — `dict[str, type[ClaimBase]]`. Module-Claim
  *classes* (the type, not an instance) ready to instantiate with
  YAML kwargs, e.g. `q_network: {class: MLP, hidden: [64, 64]}`.

- `containers` — `dict[str, type]`. Config-bundle classes that
  are NOT Claims themselves but hold slot Claims (e.g. `Replay`).
  CLAUDE.md's three-way taxonomy distinguishes these from Module
  Claims; the registry mirrors that split so the YAML loader
  knows to instantiate-with-kwargs (same as `module_classes`)
  but doesn't promise Claim semantics.

`add_module(module)` walks `vars(module)` and indexes every
`FnClaim` instance and `ClaimBase` subclass it finds. Containers
are registered explicitly via `add_container` — they have no
unique structural marker (just `@dataclass(frozen=True)` with
slot Claim fields, which would also catch unrelated frozen
dataclasses), so we don't auto-detect.

Name collisions raise `ValueError` at registration time; the
substrate fixes the ambiguity by renaming. Lookups raise
`KeyError` with the missing name; YAML loaders surface this as
a config error pointing at the offending token.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from importlib import import_module
from types import ModuleType

from corroborate.claim import ClaimBase, FnClaim


@dataclass(slots=True)
class Registry:
    """Three typed maps from string token to Python handle.

    `fns`, `module_classes`, `containers` are public for
    inspection but mutate via the `add_*` methods so collisions
    raise consistently. Lookups via `fn`, `module_class`,
    `container` — typed return shape per surface."""

    fns: dict[str, FnClaim[..., object]] = field(default_factory=dict)
    module_classes: dict[str, type[ClaimBase]] = field(default_factory=dict)
    containers: dict[str, type] = field(default_factory=dict)

    def add_module(self, module: ModuleType) -> None:
        """Index every `FnClaim` instance and `ClaimBase` subclass
        found in `vars(module)`. Skips dunder/private names and
        `ClaimBase` itself.

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
                self._add_fn(value)
            elif (
                isinstance(value, type)
                and issubclass(value, ClaimBase)
                and value is not ClaimBase
            ):
                self._add_module_class(value)

    def add_modules(self, module_names: Iterable[str]) -> None:
        """Convenience: import each name and `add_module`. Order
        is irrelevant — collisions across modules raise either
        way."""
        for name in module_names:
            self.add_module(import_module(name))

    def add_container(self, cls: type) -> None:
        """Register a config-bundle class. Caller picks the key
        (uses `cls.__name__`). Idempotent on the same class;
        raises on a different class at the same key."""
        key = cls.__name__
        existing = self.containers.get(key)
        if existing is not None and existing is not cls:
            raise ValueError(
                f'container name {key!r} already registered to '
                f'{existing!r}; cannot rebind to {cls!r}',
            )
        self.containers[key] = cls

    def _add_fn(self, fn: FnClaim[..., object]) -> None:
        key = fn.name
        existing = self.fns.get(key)
        if existing is not None and existing.fn is not fn.fn:
            raise ValueError(
                f'FnClaim name {key!r} already registered to a '
                f'different underlying function',
            )
        self.fns[key] = fn

    def _add_module_class(self, cls: type[ClaimBase]) -> None:
        key = cls.__name__
        existing = self.module_classes.get(key)
        if existing is not None and existing is not cls:
            raise ValueError(
                f'ClaimBase subclass name {key!r} already '
                f'registered to {existing!r}; cannot rebind to '
                f'{cls!r}',
            )
        self.module_classes[key] = cls

    def fn(self, name: str) -> FnClaim[..., object]:
        """Resolve `name` to the `FnClaim` registered for it.
        `KeyError` if absent."""
        try:
            return self.fns[name]
        except KeyError:
            raise KeyError(
                f'no FnClaim named {name!r}; '
                f'known: {sorted(self.fns)}',
            ) from None

    def module_class(self, name: str) -> type[ClaimBase]:
        """Resolve `name` to a `ClaimBase` subclass. `KeyError`
        if absent."""
        try:
            return self.module_classes[name]
        except KeyError:
            raise KeyError(
                f'no ClaimBase subclass named {name!r}; '
                f'known: {sorted(self.module_classes)}',
            ) from None

    def container(self, name: str) -> type:
        """Resolve `name` to a registered container class.
        `KeyError` if absent."""
        try:
            return self.containers[name]
        except KeyError:
            raise KeyError(
                f'no container named {name!r}; '
                f'known: {sorted(self.containers)}',
            ) from None


__all__ = ['Registry']
