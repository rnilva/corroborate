"""Gymnax spaces stub — typed `Box` / `Discrete` for the substrate.

The runtime classes live at `gymnax.environments.spaces.{Box,Discrete}`;
the substrate imports them from this path for `ActionDuplicatedEnv`
constructing a fresh `Discrete(n)` for the inflated action space.
Re-exported from `gymnax/__init__.pyi` as `gymnax.Box` /
`gymnax.Discrete` so type annotations can import from the top
level even though the runtime module only exposes them at this
nested path."""
from __future__ import annotations

from gymnax import Box as Box, Discrete as Discrete, Space as Space

__all__ = ['Box', 'Discrete', 'Space']
