"""Sweep-launch pre-registration manifest — substrate test.

Lives here (not in the framework's top-level `tests/`) because it
imports `corroborate_rl.dqn.yaml_sweep.DQNSweep` — the substrate's
typed sweep shape with the new `pre_registered_bridges` field.

Per-spec test 5 from
`docs/IMPLEMENTATION_SPEC_pre_registration_manifest.md` §8: the
manifest-write helper is called at sweep launch, before any cell
runs; the on-disk JSON round-trips through `read_manifest`;
double-write raises `FileExistsError` (manifest immutability)."""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

# Importing analyses populates the registry so the bridge's
# referenced analyses resolve at import time.
import corroborate.analyses  # noqa: F401  # pyright: ignore[reportUnusedImport]

from corroborate.bridge.bridge import Direction, Tier, claim_bridge
from corroborate.bridge.verdict import Verdict
from corroborate.core.claim import claim
from corroborate.core.intervention import DoEffect, Intervention
from corroborate.core.pre_registration import (
    BridgeCommitmentInput,
    MANIFEST_NAME,
    read_manifest,
)
from corroborate_rl.dqn.collect import EnvConfig
from corroborate_rl.dqn.yaml_sweep import (
    DQNSweep, load_sweep, write_pre_registration_manifest_for_sweep,
)


# ============ Fixture bridge (module-scope, resolvable by import) ============


@claim
def _t_op_pre_reg_substrate(x: int) -> int:
    return x


@claim
def _b_op_pre_reg_substrate(x: int) -> int:
    return x


_INTERVENTION = DoEffect(arms=(
    (Intervention(slot_path='op', replacement=_b_op_pre_reg_substrate),),
    (Intervention(slot_path='op', replacement=_t_op_pre_reg_substrate),),
))


@claim_bridge(
    source=_INTERVENTION,
    target='outcome',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    predicted_direction='a_gt_b',
)
def fixture_bridge(
    *,
    treatment_arm: str = '',
    baseline_arm: str = '',
) -> Verdict:
    """Resolvable bridge for the sweep-launch manifest tests."""
    del treatment_arm, baseline_arm
    return Verdict.HELD


# pytest imports the test file as a top-level module (see
# substrate conftest); pyright doesn't know that, so we read the
# module's actual `__name__` at runtime rather than hard-coding.
_BRIDGE_NAME = f'{__name__}.fixture_bridge'


# ============ Test 5: manifest written at sweep launch ============


def test_manifest_written_at_sweep_launch(tmp_path: Path) -> None:
    """The runner-commit contract: a `DQNSweep` declaring
    `pre_registered_bridges` writes `<out_dir>/pre_registration.json`
    before any cell runs. Exercises the helper directly rather
    than dispatching the full sweep loop (which would compile a
    JAX kernel and take minutes)."""
    out_dir = tmp_path / 'fixture_sweep'
    out_dir.mkdir()
    sweep = DQNSweep(
        name='fixture',
        out_dir=out_dir,
        envs=(EnvConfig(env_name='TestEnv', n_seeds=2, chunk_size=2),),
        intervention_templates=(),
        env_binding='shared',
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
    assert manifest is not None
    assert len(manifest.bridge_commitments) == 1
    commitment = manifest.bridge_commitments[0]
    assert commitment.bridge_name == _BRIDGE_NAME
    assert commitment.predicted_direction == 'a_gt_b'
    assert commitment.predicted_verdict == Verdict.HELD
    # git_commit_hash should be a 40-char SHA (framework HEAD at
    # test time).
    assert len(manifest.git_commit_hash) == 40
    # Source-hash is the sha256 hex digest (64 chars).
    assert len(commitment.source_hash) == 64
    # Sweep-config hash same shape.
    assert len(manifest.sweep_config_hash) == 64

    # Immutability: a second call against the same out_dir must
    # raise FileExistsError rather than silently overwriting.
    with pytest.raises(FileExistsError, match='immutable'):
        _ = write_pre_registration_manifest_for_sweep(sweep)


def test_manifest_skipped_when_no_pre_registered_bridges(
    tmp_path: Path,
) -> None:
    """Sweeps without `pre_registered_bridges` declared MUST NOT
    write a manifest. The feature is opt-in; existing sweeps
    behave identically to the pre-feature baseline (no
    `pre_registration.json` on disk, no surprise audit drift)."""
    out_dir = tmp_path / 'no_commitment_sweep'
    out_dir.mkdir()
    sweep = DQNSweep(
        name='no_commit',
        out_dir=out_dir,
        envs=(EnvConfig(env_name='TestEnv', n_seeds=2, chunk_size=2),),
        intervention_templates=(),
        env_binding='shared',
    )
    written = write_pre_registration_manifest_for_sweep(sweep)
    assert written is None
    assert not (out_dir / MANIFEST_NAME).exists()


# ============ YAML schema parsing ============


def test_yaml_loads_pre_registered_bridges(tmp_path: Path) -> None:
    """The `pre_registered_bridges` key in YAML round-trips into
    a typed `tuple[BridgeCommitmentInput, ...]` on `DQNSweep`."""
    yaml_path = tmp_path / 'sweep.yaml'
    _ = yaml_path.write_text(
        'name: yaml_fixture\n'
        f'out_dir: {tmp_path / "yaml_out"}\n'
        'env_binding: shared\n'
        'envs:\n'
        '  - {name: TestEnv, n_seeds: 2, chunk_size: 2}\n'
        'interventions:\n'
        '  - name: vanilla_dqn\n'
        'pre_registered_bridges:\n'
        f'  - bridge: {_BRIDGE_NAME}\n'
        '    predicted_direction: a_gt_b\n'
        '    predicted_verdict: held\n',
        encoding='utf-8',
    )
    from corroborate_rl.dqn.yaml_sweep import default_dqn_registry
    sweep = load_sweep(yaml_path, reg=default_dqn_registry())
    assert len(sweep.pre_registered_bridges) == 1
    entry = sweep.pre_registered_bridges[0]
    assert entry.bridge_name == _BRIDGE_NAME
    assert entry.predicted_direction == 'a_gt_b'
    assert entry.predicted_verdict == Verdict.HELD


def test_yaml_rejects_unknown_verdict(tmp_path: Path) -> None:
    """A typo'd verdict at YAML load fails loud — we won't burn
    sweep compute on a typed mistake."""
    yaml_path = tmp_path / 'sweep_bad.yaml'
    _ = yaml_path.write_text(
        'name: bad\n'
        f'out_dir: {tmp_path / "bad_out"}\n'
        'env_binding: shared\n'
        'envs:\n'
        '  - {name: TestEnv, n_seeds: 2, chunk_size: 2}\n'
        'interventions:\n'
        '  - name: vanilla_dqn\n'
        'pre_registered_bridges:\n'
        f'  - bridge: {_BRIDGE_NAME}\n'
        '    predicted_direction: a_gt_b\n'
        '    predicted_verdict: definitely_a_typo\n',
        encoding='utf-8',
    )
    from corroborate_rl.dqn.yaml_sweep import default_dqn_registry
    with pytest.raises(ValueError, match='predicted_verdict'):
        _ = load_sweep(yaml_path, reg=default_dqn_registry())


# ============ Suppression of import-time warnings ============


# `polars as pl` is imported but unused; keep it referenced so
# downstream linting doesn't strip the import (test runners
# sometimes need polars available for pyarrow round-trips even
# when the test body doesn't use it).
_ = pl
