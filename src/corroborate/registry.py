"""Substrate-agnostic name → typed-handle registry.

YAML- or config-driven sweep authoring needs to map string tokens
(`'double_greedify'`, `'Replay'`) back to the typed Python handle
the substrate's `Claim` graph holds. The registry is that map.

Two surfaces:

- `fns` — `dict[str, FnClaim[..., object]]`. `@claim`-decorated
  free functions. Key is `FnClaim._name` (== wrapped function
  `__name__`). YAML uses these to fill slot bindings, e.g.
  `bootstrap.greedification: double_greedify`.

- `classes` — `dict[str, type]`. Module-Claim *classes* (subclasses
  of `ClaimBase`) and config-bundle classes (e.g. `Replay`),
  ready to instantiate with YAML kwargs. The Module-vs-bundle
  taxonomy distinction is a Claim semantics question reasserted
  by `is_claim()` at use-site, not by the registry — at the YAML
  resolution boundary both shapes are interchangeable
  ("instantiate with kwargs"), so encoding the split here just
  doubled the surface without preventing any error.

`add_module(module)` walks `vars(module)` and indexes every
`FnClaim` instance and `ClaimBase` subclass it finds. Containers
are registered explicitly via `add_class` — they have no unique
structural marker (just `@dataclass(frozen=True)` with slot
Claim fields, which would also catch unrelated frozen
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
    """Two typed maps from string token to Python handle.

    `fns` and `classes` are public for inspection but mutate via
    the `add_*` methods so collisions raise consistently. Lookups
    via `fn` and `cls` — typed return shape per surface."""

    fns: dict[str, FnClaim[..., object]] = field(default_factory=dict)
    classes: dict[str, type] = field(default_factory=dict)

    def add_module(self, module: ModuleType) -> None:
        """Index every `FnClaim` instance and `ClaimBase` subclass
        found in `vars(module)`. Skips dunder/private names and
        `ClaimBase` itself.

        Re-adding the same value at the same name is a no-op;
        adding a *different* value at an already-present name
        raises `ValueError`. Config-bundle classes (e.g. `Replay`)
        aren't auto-detected — register via `add_class`."""
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
                self._add_class(value)

    def add_modules(self, module_names: Iterable[str]) -> None:
        """Convenience: import each name and `add_module`. Order
        is irrelevant — collisions across modules raise either
        way."""
        for name in module_names:
            self.add_module(import_module(name))

    def add_class(self, cls: type) -> None:
        """Register a class at its `__name__`. Idempotent on the
        same class; raises on a different class at the same key.

        Used both by `add_module` (auto-discovered ClaimBase
        subclasses) and by the substrate (manual config-bundle
        registration)."""
        self._add_class(cls)

    def _add_fn(self, fn: FnClaim[..., object]) -> None:
        key = fn.name
        existing = self.fns.get(key)
        if existing is not None and existing.fn is not fn.fn:
            raise ValueError(
                f'FnClaim name {key!r} already registered to a '
                f'different underlying function',
            )
        self.fns[key] = fn

    def _add_class(self, cls: type) -> None:
        key = cls.__name__
        existing = self.classes.get(key)
        if existing is not None and existing is not cls:
            raise ValueError(
                f'class name {key!r} already registered to '
                f'{existing!r}; cannot rebind to {cls!r}',
            )
        self.classes[key] = cls

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

    def cls(self, name: str) -> type:
        """Resolve `name` to a registered class (Module Claim or
        config bundle). `KeyError` if absent."""
        try:
            return self.classes[name]
        except KeyError:
            raise KeyError(
                f'no class named {name!r}; '
                f'known: {sorted(self.classes)}',
            ) from None


__all__ = ['Registry']
