"""YAML schema tests for `keep_q_checkpoint_final` /
`keep_q_checkpoint_per_burst`.

Fast cohort — no JAX, no DQN, no sweep dispatch. Just verifies
the loader's typed contract: default-false, explicit true,
non-bool rejected with a recognisable message.

Mirrors `test_yaml_sweep.py::test_keep_q_per_action_*` and the
`gradient_probes` field-parsing pattern."""
from __future__ import annotations

from pathlib import Path

import pytest

from corroborate.runner.registry import Registry
from corroborate_rl.dqn.yaml_sweep import default_dqn_registry, load_sweep


# Match `test_yaml_sweep.py::_minimal_sweep_yaml` shape so
# `_build_envs` / `_build_interventions` succeed without us
# duplicating their assertions here.
_MINIMAL_BODY = (
    'name: ckpt_test\n'
    'out_dir: /tmp/ckpt_test\n'
    'envs:\n'
    '  - name: Catch-bsuite\n'
    '    n_seeds: 1\n'
    'interventions:\n'
    '  - name: van\n'
    '    base: {}\n'
)


@pytest.fixture
def reg() -> Registry:
    return default_dqn_registry()


def test_keep_q_checkpoint_final_defaults_false(
    tmp_path: Path, reg: Registry,
) -> None:
    """Existing YAMLs that don't mention the flag continue to run
    with checkpoint persistence OFF — no surprise disk-usage
    explosion for users who upgrade through this change."""
    p = tmp_path / 'sweep.yaml'
    _ = p.write_text(_MINIMAL_BODY)
    s = load_sweep(p, reg=reg)
    assert s.keep_q_checkpoint_final is False
    assert s.keep_q_checkpoint_per_burst is False


def test_keep_q_checkpoint_final_explicit_true(
    tmp_path: Path, reg: Registry,
) -> None:
    p = tmp_path / 'sweep.yaml'
    _ = p.write_text(_MINIMAL_BODY + 'keep_q_checkpoint_final: true\n')
    s = load_sweep(p, reg=reg)
    assert s.keep_q_checkpoint_final is True
    assert s.keep_q_checkpoint_per_burst is False


def test_keep_q_checkpoint_per_burst_explicit_true(
    tmp_path: Path, reg: Registry,
) -> None:
    p = tmp_path / 'sweep.yaml'
    _ = p.write_text(
        _MINIMAL_BODY + 'keep_q_checkpoint_per_burst: true\n',
    )
    s = load_sweep(p, reg=reg)
    assert s.keep_q_checkpoint_final is False
    assert s.keep_q_checkpoint_per_burst is True


def test_both_flags_can_co_exist(
    tmp_path: Path, reg: Registry,
) -> None:
    """Final + per-burst are independent axes; the YAML can opt
    into both. Useful when an analysis wants the full per-burst
    trajectory PLUS the canonical final-state record."""
    p = tmp_path / 'sweep.yaml'
    _ = p.write_text(
        _MINIMAL_BODY
        + 'keep_q_checkpoint_final: true\n'
        + 'keep_q_checkpoint_per_burst: true\n',
    )
    s = load_sweep(p, reg=reg)
    assert s.keep_q_checkpoint_final is True
    assert s.keep_q_checkpoint_per_burst is True


def test_keep_q_checkpoint_final_rejects_non_bool(
    tmp_path: Path, reg: Registry,
) -> None:
    p = tmp_path / 'sweep.yaml'
    _ = p.write_text(
        _MINIMAL_BODY + 'keep_q_checkpoint_final: maybe\n',
    )
    with pytest.raises(
        TypeError, match='keep_q_checkpoint_final must be bool',
    ):
        _ = load_sweep(p, reg=reg)


def test_keep_q_checkpoint_per_burst_rejects_non_bool(
    tmp_path: Path, reg: Registry,
) -> None:
    p = tmp_path / 'sweep.yaml'
    _ = p.write_text(
        _MINIMAL_BODY + 'keep_q_checkpoint_per_burst: 1\n',
    )
    with pytest.raises(
        TypeError, match='keep_q_checkpoint_per_burst must be bool',
    ):
        _ = load_sweep(p, reg=reg)
