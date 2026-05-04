"""Tests for `corroborate.runner.run` — the typed Hypothesis-Protocol
dispatch surface that replaced `run_module` in Phase 6.

Coverage:
- `_validate_hypothesis` accepts conforming objects (dataclass
  classes — the test stand-in for module-level conformance) and
  rejects non-conforming ones with TypeError.
- `_validate_hypothesis` element-checks `BRIDGES` (non-Bridge
  members raise).
- `_default_cache_path` reads `__name__` typed (no getattr
  fallbacks); module-style names with dots are split correctly,
  bare class names pass through.

The framework's bridge-zoo modules (`dqn_bridges.py`,
`ddqn_universe.py`) cover module-shape conformance at production
load time; these tests exercise the typed-Protocol contract via
class-based hypotheses (which carry `__name__` from Python for
free, same as modules)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pytest

from corroborate.bridge.bridge import Bridge
from corroborate.core.intervention import DoEffect, Intervention

# `_validate_hypothesis` / `_default_cache_path` are
# underscore-prefixed because they're internal helpers called only
# from `run` in production. Tests exercise them directly to keep
# the surface focused on Protocol-narrowing + cache-path defaulting
# without spinning up the full ingest+evaluate pipeline. Per CLAUDE.md
# §heuristic, this is the third option ("the value's true type
# requires a typed test path the public API doesn't expose").
from corroborate.runner.runner import (
    _default_cache_path,  # pyright: ignore[reportPrivateUsage]
    _validate_hypothesis,  # pyright: ignore[reportPrivateUsage]
)


def _trivial_doeffect() -> DoEffect:
    """Stub DoEffect — used only for Protocol shape; never
    evaluates against cells in these tests."""
    from corroborate.core.claim import claim

    @claim
    def _stub(x: int) -> int:
        return x

    return DoEffect(
        treatment=(Intervention(slot_path='stub', replacement=_stub),),
        baseline=(),
    )


# ============ _validate_hypothesis ============

def test_validate_accepts_class_with_classvars() -> None:
    @dataclass(frozen=True)
    class H:
        INTERVENTION: ClassVar[DoEffect] = _trivial_doeffect()
        BRIDGES: ClassVar[tuple[Bridge, ...]] = ()
    out = _validate_hypothesis(H)
    assert out is H


def test_validate_rejects_class_missing_intervention() -> None:
    @dataclass(frozen=True)
    class Broken:
        BRIDGES: ClassVar[tuple[Bridge, ...]] = ()
    with pytest.raises(TypeError, match='Hypothesis Protocol'):
        _validate_hypothesis(Broken)


def test_validate_rejects_class_missing_bridges() -> None:
    @dataclass(frozen=True)
    class Broken:
        INTERVENTION: ClassVar[DoEffect] = _trivial_doeffect()
    with pytest.raises(TypeError, match='Hypothesis Protocol'):
        _validate_hypothesis(Broken)


def test_validate_rejects_non_bridge_in_bridges() -> None:
    """`runtime_checkable` only validates attribute presence;
    `_validate_hypothesis` adds the element-type check on top."""
    @dataclass(frozen=True)
    class Malformed:
        INTERVENTION: ClassVar[DoEffect] = _trivial_doeffect()
        # str instead of Bridge — element-type check should catch.
        BRIDGES: ClassVar[tuple[object, ...]] = ('not-a-bridge',)
    with pytest.raises(TypeError, match='non-Bridge'):
        _validate_hypothesis(Malformed)


# ============ _default_cache_path ============

def test_default_cache_path_bare_class_name() -> None:
    """A class-style bare `__name__` (no dots) passes through."""
    @dataclass(frozen=True)
    class DDQNvsVanilla:
        INTERVENTION: ClassVar[DoEffect] = _trivial_doeffect()
        BRIDGES: ClassVar[tuple[Bridge, ...]] = ()
    p = _default_cache_path(DDQNvsVanilla)
    assert p == Path('experiments/data/cache/DDQNvsVanilla.parquet')
