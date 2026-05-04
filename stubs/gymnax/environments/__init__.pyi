"""Gymnax environments subpackage stub. Re-exports `spaces` so
substrate imports `from gymnax.environments import spaces` resolve
to the typed surface declared in `spaces.pyi`."""
from __future__ import annotations

from gymnax.environments import spaces as spaces

__all__ = ['spaces']
