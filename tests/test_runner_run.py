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


def test_default_cache_path_dotted_module_name_uses_last_segment() -> None:
    """Module-style names with dots: `<short>` is the LAST segment.
    The hypothesis runner uses `__name__.split('.')[-1]` so a module
    `experiments.findings.dqn_bridges` resolves to `dqn_bridges.parquet`,
    NOT to `experiments.findings.dqn_bridges.parquet`.

    Pin the dotted-path-stripping branch — the bare-name test above
    only exercises the `[-1]` fallback when there's no dot. A
    regression to `__name__` (no split) would pass the bare test
    and breach this one."""
    # Module-style __name__: the runner accepts anything with
    # `__name__: str + INTERVENTION + BRIDGES`. ModuleType is a
    # natural surrogate; types.ModuleType lets us set `__name__`
    # directly. Cast for the typed Hypothesis Protocol.
    import types
    from typing import cast
    from corroborate.core.hypothesis import Hypothesis
    mod = types.ModuleType('experiments.findings.dqn_bridges')
    # Module attribute writes via setattr — direct assignment trips
    # pyright's strict module-attribute-access mode.
    setattr(mod, 'INTERVENTION', _trivial_doeffect())
    setattr(mod, 'BRIDGES', ())
    p = _default_cache_path(cast(Hypothesis, mod))
    assert p == Path('experiments/data/cache/dqn_bridges.parquet'), (
        f'cache_path = {p}; expected the last-segment short name '
        f'`dqn_bridges.parquet`, not the full dotted name.'
    )


# ============ CACHE_ADDITIVITY.md CA1-CA3 ============


def test_load_data_returns_none_when_data_is_none() -> None:
    """**CA2**: `_load_data(None, ...)` returns None — no walk,
    no DataFrame construction. The cache-only default path
    relies on this."""
    from corroborate.runner.runner import _load_data
    out = _load_data(
        None, restore_from_cloud=False, required=(), bridges=(),
    )
    assert out is None


def test_load_data_dispatches_sequence_path_to_named_corpora(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """**CA3**: passing `Sequence[Path]` routes to the
    named-corpora ingest (via `_load_directory(corpus_dirs=...)`),
    not the directory-walk path. Pin by constructing two real
    corpus dirs and verifying both flow through."""
    # Force sequential processing — the parallel ProcessPoolExecutor
    # path uses fork() and stalls under pytest's stderr capture.
    monkeypatch.setenv('CORROBORATE_CACHE_WORKERS', '1')

    import polars as pl
    from corroborate.runner.runner import _load_data

    root = tmp_path / 'data'
    for i, name in enumerate(('a', 'b', 'unrelated')):
        d = root / name
        d.mkdir(parents=True)
        # Distinct env_name per corpus so CI4 content-dedup
        # doesn't collapse them.
        df = pl.DataFrame({
            'id': [f'{name}-0', f'{name}-1'],
            'env_name': [f'Env_{name}'] * 2,
            'arm_key': ['baseline'] * 2,
            'seed': [0, 1],
            'reward_scale': [float(i)] * 2,
        })
        df.write_parquet(d / 'runs.parquet')

    # Ingest just `a` and `b`; `unrelated` should not contribute.
    out = _load_data(
        [root / 'a', root / 'b'],
        restore_from_cloud=False, required=(), bridges=(),
    )
    assert out is not None
    ids = sorted(out['id'].to_list())
    assert ids == ['a-0', 'a-1', 'b-0', 'b-1'], (
        f'expected only a + b cells; got {ids}. CA3 may have '
        f'fallen through to a directory walk.'
    )


def test_load_data_named_corpora_empty_sequence_is_noop(
    tmp_path: Path,
) -> None:
    """**CA3 edge case**: empty list — return None, don't walk
    any default root."""
    from corroborate.runner.runner import _load_data
    out = _load_data(
        [], restore_from_cloud=False, required=(), bridges=(),
    )
    assert out is None


def test_load_data_named_corpora_missing_dir_raises(
    tmp_path: Path,
) -> None:
    """**CA3 edge case**: a named dir that doesn't exist raises
    FileNotFoundError loudly. No silent skip — the user typed
    the name explicitly."""
    from corroborate.runner.runner import _load_data
    with pytest.raises(FileNotFoundError, match='--ingest dir not found'):
        _load_data(
            [tmp_path / 'does_not_exist'],
            restore_from_cloud=False, required=(), bridges=(),
        )
