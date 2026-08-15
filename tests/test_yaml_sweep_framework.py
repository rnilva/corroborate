"""Framework-side yaml_sweep primitives — substrate-agnostic.

Exercises `corroborate.runner.yaml_sweep` against a minimal
`Sweep`-Protocol-satisfying dataclass that doesn't depend on the
DQN substrate. The substrate's `DQNSweep` keeps its own
substrate-level coverage in
`src/corroborate_rl/tests/test_pre_registration_at_sweep_launch.py`."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pytest

# Importing analyses populates the registry so any test-fixture
# bridge can resolve its analyses at import time.
import corroborate.analyses  # noqa: F401  # pyright: ignore[reportUnusedImport]

from corroborate.bridge.bridge import Direction, Tier, claim_bridge
from corroborate.bridge.verdict import Verdict
from corroborate.core.claim import claim
from corroborate.core.intervention import DoEffect, Intervention
from corroborate.core.pre_registration import (
    BridgeCommitmentInput,
    MANIFEST_NAME,
    PreRegistrationManifest,
    read_manifest,
)
from corroborate.runner.yaml_sweep import (
    Sweep,
    assert_unique_cfg_names,
    build_archive_remote,
    build_merge_top_level,
    build_pre_registered_bridges,
    require_predicted_direction,
    require_predicted_verdict,
    require_sweep_str,
    write_pre_registration_manifest_for_sweep,
)


# ============ Minimal Sweep-satisfying dataclass ============


@dataclass(frozen=True, slots=True)
class _MinimalSweep:
    """Bare frozen dataclass with exactly the Sweep Protocol's
    five fields. No implementation-specific add-ons. Tests use this to
    prove the framework primitives work without DQN coupling."""
    name: str
    out_dir: Path
    archive_remote: str | None
    merge_top_level: bool
    pre_registered_bridges: tuple[BridgeCommitmentInput, ...]


# ============ Fixture bridge (resolvable by import path) ============


@claim
def _t_op_framework_pre_reg(x: int) -> int:
    return x


@claim
def _b_op_framework_pre_reg(x: int) -> int:
    return x


_INTERVENTION = DoEffect(arms=(
    (Intervention(slot_path='op', replacement=_b_op_framework_pre_reg),),
    (Intervention(slot_path='op', replacement=_t_op_framework_pre_reg),),
))


@claim_bridge(
    source=_INTERVENTION,
    target='outcome',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    predicted_direction='a_gt_b',
)
def _framework_fixture_bridge(
    *,
    treatment_arm: str = '',
    baseline_arm: str = '',
) -> Verdict:
    """Body-bearing bridge so `compute_bridge_source_hash` has an
    AST to hash. Returns HELD unconditionally — the test never
    runs the bridge, only resolves it at sweep-launch time."""
    del treatment_arm, baseline_arm
    return Verdict.HELD


_BRIDGE_NAME = f'{__name__}._framework_fixture_bridge'


# ============ Test 1: Protocol satisfaction ============


def test_minimal_sweep_satisfies_protocol(tmp_path: Path) -> None:
    """A frozen dataclass with the five required fields satisfies
    the `Sweep` Protocol both at runtime (`isinstance` via
    `runtime_checkable`) and structurally for pyright. Proves the
    Protocol is read-only-compatible with frozen-dataclass fields
    (CLAUDE.md: writable Protocol fields would NOT match
    immutable concrete fields)."""
    sweep = _MinimalSweep(
        name='test',
        out_dir=tmp_path / 'sweep',
        archive_remote=None,
        merge_top_level=True,
        pre_registered_bridges=(),
    )
    assert isinstance(sweep, Sweep)


# ============ Test 2: empty-commitments no-op ============


def test_write_manifest_skipped_when_empty(tmp_path: Path) -> None:
    """`pre_registered_bridges=()` → returns None, no file
    written. The pre-registration manifest is opt-in; sweeps
    without explicit commitments behave identically to the
    pre-feature baseline (no `pre_registration.json` on disk)."""
    out_dir = tmp_path / 'no_commit'
    out_dir.mkdir()
    sweep = _MinimalSweep(
        name='no_commit',
        out_dir=out_dir,
        archive_remote=None,
        merge_top_level=True,
        pre_registered_bridges=(),
    )
    written = write_pre_registration_manifest_for_sweep(sweep)
    assert written is None
    assert not (out_dir / MANIFEST_NAME).exists()


# ============ Test 3: write + round-trip with one commitment ============


def test_write_manifest_round_trips(tmp_path: Path) -> None:
    """A `Sweep` with one commitment writes `pre_registration.json`
    that round-trips through `PreRegistrationManifest.from_dict`.
    Exercises `build_commitments` + `compute_sweep_config_hash` +
    `get_git_head_sha` + `write_manifest` together."""
    out_dir = tmp_path / 'with_commit'
    out_dir.mkdir()
    sweep = _MinimalSweep(
        name='with_commit',
        out_dir=out_dir,
        archive_remote=None,
        merge_top_level=True,
        pre_registered_bridges=(
            BridgeCommitmentInput(
                bridge_name=_BRIDGE_NAME,
                predicted_direction='a_gt_b',
                predicted_verdict=Verdict.HELD,
            ),
        ),
    )
    written = write_pre_registration_manifest_for_sweep(sweep)
    assert written is not None
    assert written == out_dir / MANIFEST_NAME
    assert written.exists()

    manifest = read_manifest(out_dir)
    assert isinstance(manifest, PreRegistrationManifest)
    assert len(manifest.bridge_commitments) == 1
    commitment = manifest.bridge_commitments[0]
    assert commitment.bridge_name == _BRIDGE_NAME
    assert commitment.predicted_direction == 'a_gt_b'
    assert commitment.predicted_verdict == Verdict.HELD
    # Framework HEAD SHA at test time, sha256 hex digests.
    assert len(manifest.git_commit_hash) == 40
    assert len(manifest.sweep_config_hash) == 64
    assert len(commitment.source_hash) == 64


# ============ Test 4: second write raises FileExistsError ============


def test_second_write_raises_immutable(tmp_path: Path) -> None:
    """Manifests are immutable per spec §5. A second
    `write_pre_registration_manifest_for_sweep` against the same
    `out_dir` MUST raise `FileExistsError` with a message
    referencing the immutability invariant — silent overwrite
    would let a HARKing operator delete + rewrite the manifest
    mid-sweep."""
    out_dir = tmp_path / 'immutable'
    out_dir.mkdir()
    sweep = _MinimalSweep(
        name='immutable',
        out_dir=out_dir,
        archive_remote=None,
        merge_top_level=True,
        pre_registered_bridges=(
            BridgeCommitmentInput(
                bridge_name=_BRIDGE_NAME,
                predicted_direction='a_gt_b',
                predicted_verdict=Verdict.HELD,
            ),
        ),
    )
    _ = write_pre_registration_manifest_for_sweep(sweep)
    with pytest.raises(FileExistsError, match='immutable'):
        _ = write_pre_registration_manifest_for_sweep(sweep)


# ============ Test 5: YAML helpers exercise ============


def test_yaml_helpers_round_trip() -> None:
    """The framework's YAML scalar parsers each accept the
    expected shapes and reject the unexpected ones with clear
    typed errors. Drives `require_sweep_str`,
    `build_archive_remote`, `build_merge_top_level`,
    `build_pre_registered_bridges`, `require_predicted_direction`,
    `require_predicted_verdict` directly."""
    # require_sweep_str
    assert require_sweep_str({'name': 'foo'}, 'name') == 'foo'
    with pytest.raises(TypeError, match='sweep.name must be a string'):
        _ = require_sweep_str({'name': 42}, 'name')

    # build_archive_remote
    assert build_archive_remote({}) is None
    assert build_archive_remote(
        {'archive_remote': 's3://bucket/path'},
    ) == 's3://bucket/path'
    with pytest.raises(TypeError, match='archive_remote'):
        _ = build_archive_remote({'archive_remote': 42})

    # build_merge_top_level (default True, strict bool)
    assert build_merge_top_level({}) is True
    assert build_merge_top_level({'merge_top_level': False}) is False
    with pytest.raises(TypeError, match='merge_top_level'):
        _ = build_merge_top_level({'merge_top_level': 'yes'})

    # build_pre_registered_bridges — empty/absent
    assert build_pre_registered_bridges({}) == ()
    assert build_pre_registered_bridges(
        {'pre_registered_bridges': []},
    ) == ()

    # build_pre_registered_bridges — one entry round-trips
    entry: Mapping[str, object] = {
        'bridge': _BRIDGE_NAME,
        'predicted_direction': 'a_gt_b',
        'predicted_verdict': 'held',
    }
    built = build_pre_registered_bridges(
        {'pre_registered_bridges': [entry]},
    )
    assert len(built) == 1
    assert built[0].bridge_name == _BRIDGE_NAME
    assert built[0].predicted_direction == 'a_gt_b'
    assert built[0].predicted_verdict == Verdict.HELD

    # Typo'd verdict / direction fail loud
    bad_verdict: Mapping[str, object] = {
        'bridge': _BRIDGE_NAME,
        'predicted_direction': 'a_gt_b',
        'predicted_verdict': 'definitely_a_typo',
    }
    with pytest.raises(ValueError, match='predicted_verdict'):
        _ = build_pre_registered_bridges(
            {'pre_registered_bridges': [bad_verdict]},
        )

    bad_direction: Mapping[str, object] = {
        'bridge': _BRIDGE_NAME,
        'predicted_direction': 'sideways',
        'predicted_verdict': 'held',
    }
    with pytest.raises(ValueError, match='predicted_direction'):
        _ = build_pre_registered_bridges(
            {'pre_registered_bridges': [bad_direction]},
        )

    # Standalone direction / verdict narrowers (the dispatch
    # surface used by build_pre_registered_bridges).
    assert require_predicted_direction(
        {'predicted_direction': 'a_lt_b'},
    ) == 'a_lt_b'
    assert require_predicted_direction(
        {'predicted_direction': 'two_sided'},
    ) == 'two_sided'
    assert require_predicted_direction(
        {'predicted_direction': 'null'},
    ) == 'null'
    assert require_predicted_verdict(
        {'predicted_verdict': 'no_effect'},
    ) == Verdict.NO_EFFECT


# ============ Test 6: assert_unique_cfg_names ============


@dataclass(frozen=True, slots=True)
class _NamedStub:
    name: str


def test_assert_unique_cfg_names_passes_on_unique() -> None:
    """Three distinct names → no raise. The post-expansion
    uniqueness check is what `dispatch_sweep` calls before any
    cell runs to refuse silent overwrites at
    `<out_dir>/<cfg.name>/`."""
    configs = (
        _NamedStub(name='vanilla'),
        _NamedStub(name='ddqn'),
        _NamedStub(name='maxmin'),
    )
    assert_unique_cfg_names(configs)


def test_assert_unique_cfg_names_raises_on_collision() -> None:
    """Two configs share `name` → `ValueError` naming the
    collision count. Author-facing message points at the
    `{from_env: env_name}` substitution fix."""
    configs = (
        _NamedStub(name='dup'),
        _NamedStub(name='unique'),
        _NamedStub(name='dup'),
    )
    with pytest.raises(ValueError, match="share output paths"):
        assert_unique_cfg_names(configs)
