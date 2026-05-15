"""Smokes for `scripts/run_sweep.py` — primarily the `--dry-run`
flag, which validates a YAML through the dispatch path
(`expand_sweep`) without touching JAX or writing cells.

The dispatch path itself is exercised in
`test_yaml_sweep_per_env.py`; this file verifies the CLI surface
threads through correctly."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_SWEEP_PATH = REPO_ROOT / 'scripts' / 'run_sweep.py'


@pytest.fixture(autouse=True)
def _add_script_to_syspath() -> 'Iterator[None]':  # noqa: F821
    """`scripts/run_sweep.py` isn't a package; import it directly
    via the path. Restore syspath after to avoid leaking into
    sibling tests."""
    sys.path.insert(0, str(RUN_SWEEP_PATH.parent))
    try:
        yield
    finally:
        sys.path.remove(str(RUN_SWEEP_PATH.parent))


def test_dry_run_shared_sweep(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """`--dry-run` on a `shared`-mode YAML resolves the
    `interventions:` templates, validates them, and prints the
    expanded set without touching JAX. Exits 0."""
    import run_sweep
    cfg = tmp_path / 'shared.yaml'
    cfg.write_text(
        'name: shared_demo\n'
        f'out_dir: {tmp_path / "out"}\n'
        'env_binding: shared\n'
        'envs:\n'
        '  - {name: FourRooms-misc, n_seeds: 2}\n'
        'defaults:\n'
        '  total_steps: 1000\n'
        '  gamma: 0.99\n'
        'interventions:\n'
        '  - name: vanilla\n'
        '  - name: ddqn\n'
        '    arms:\n'
        '      - []\n'
        '      - - slot_path: bootstrap\n'
        '          replacement:\n'
        '            fn: bootstrap\n'
        '            greedification: {fn: double_greedify}\n'
    )
    run_sweep.main([str(cfg), '--dry-run'])
    out = capsys.readouterr().out
    assert 'sweep: \'shared_demo\'' in out
    assert 'env_binding   : shared' in out
    assert 'interventions (expanded): 2' in out
    assert '- vanilla (1 arms)' in out
    assert '- ddqn (2 arms)' in out
    assert 'dry-run: OK' in out


def test_dry_run_per_env_collision_raises(tmp_path: Path) -> None:
    """`--dry-run` exercises CI9 via `expand_sweep` — a per_env
    YAML with bare template names produces post-expansion
    duplicates and raises before any dispatch happens."""
    import run_sweep
    cfg = tmp_path / 'broken.yaml'
    cfg.write_text(
        'name: broken\n'
        f'out_dir: {tmp_path / "out"}\n'
        'env_binding: per_env\n'
        'envs:\n'
        '  - {name: FourRooms-misc, n_seeds: 2}\n'
        '  - {name: Acrobot-v1, n_seeds: 2}\n'
        'defaults:\n'
        '  total_steps: 1000\n'
        '  gamma: 0.99\n'
        'interventions:\n'
        '  - name: bare_name_collides\n'  # missing {from_env}
    )
    with pytest.raises(ValueError, match='share output paths'):
        run_sweep.main([str(cfg), '--dry-run'])


def test_dry_run_unknown_measurable_raises(tmp_path: Path) -> None:
    """`required_measurables:` is validated at YAML-load time.
    `--dry-run` surfaces the validation error without touching
    JAX."""
    import run_sweep
    cfg = tmp_path / 'bad_meas.yaml'
    cfg.write_text(
        'name: bad_meas\n'
        f'out_dir: {tmp_path / "out"}\n'
        'env_binding: shared\n'
        'envs:\n'
        '  - {name: FourRooms-misc, n_seeds: 2}\n'
        'interventions:\n'
        '  - name: arm\n'
        '    required_measurables: [not_a_real_measurable]\n'
    )
    with pytest.raises(KeyError, match='not_a_real_measurable'):
        run_sweep.main([str(cfg), '--dry-run'])
