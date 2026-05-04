"""Generic name-keyed registry primitive.

Three framework registries (`analysis._REGISTRY`, `measurable.
_REGISTRY`, the dual maps inside `registry.ClaimRegistry`) share
the same shape: a `dict[str, T]` plus collision-detecting
register, presence check, sorted-names introspection, and
get-or-None lookup. This module owns the single implementation;
each consumer holds a typed `Registry[T]` instance.

Re-registering the same `(name, value)` pair is idempotent (the
`@analysis` / `@measurable` decorators end up running once at
import + once if a module is reloaded); re-registering a
*different* value at an already-present name raises `ValueError`
with the existing-vs-new mismatch surfaced — substrate authors
fix the ambiguity by renaming.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field


@dataclass(slots=True)
class Registry[T]:
    """One registry instance — a typed name → value map. The
    type parameter `T` is the registered handle's type
    (`Analysis[...]`, `Measurable[...]`, `FnClaim[...]`, etc.)."""
    _entries: dict[str, T] = field(default_factory=dict)

    def register(self, name: str, value: T) -> None:
        """Register `value` under `name`. Same-value re-registration
        is a no-op (handles import-cycle re-entry); different-value
        re-registration raises `ValueError`."""
        existing = self._entries.get(name)
        if existing is not None and existing is not value:
            raise ValueError(
                f'{name!r} already registered to a different value '
                f'(existing={existing!r}, new={value!r})',
            )
        self._entries[name] = value

    def replace(self, name: str, value: T) -> None:
        """Last-write-wins registration. Used where the registration
        site is intentionally re-entrant (test fixtures redefining
        a measurable with the same name across tests, hot-reload
        scenarios). Strict callers use `register` instead."""
        self._entries[name] = value

    def get(self, name: str) -> T | None:
        """Return the registered value or `None` if absent.
        Caller-side discipline: raise loudly when a None is
        unexpected."""
        return self._entries.get(name)

    def names(self) -> tuple[str, ...]:
        """Sorted tuple of registered names — for diagnostic output."""
        return tuple(sorted(self._entries))

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[str]:
        """Iterate registered names (insertion order). Lets
        `set(registry)` / `for name in registry` work without
        needing to call `.names()` explicitly — matches the
        former `dict[str, T]` shape the framework used to expose."""
        return iter(self._entries)


__all__ = ['Registry']
