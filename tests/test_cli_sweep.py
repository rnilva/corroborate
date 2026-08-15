"""Framework-side `corroborate sweep run` CLI smokes.

Exercises implementation resolution + `--dry-run` against the in-tree
RL substrate. The JAX-touching paths (real dispatch) are covered
by the implementation's `tests/test_run_sweep_cli.py`; this file
proves the framework wiring works.

Routes dispatch through `corroborate.__main__.main` so the full
parser tree is exercised (sweep subcommand wiring + argparse
defaults + the `sweep_subcmd` narrow)."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from corroborate.__main__ import main as corroborate_main
from corroborate.runner.yaml_sweep import SweepEntryPoints


# ============ Implementation `SWEEP_ENTRY_POINTS` shape sanity ============


def test_in_tree_substrate_exposes_entry_points() -> None:
    """The in-tree RL implementation's lightweight entry module
    `corroborate_rl.dqn_sweep` exports `SWEEP_ENTRY_POINTS:
    SweepEntryPoints`. The lightweight module — top-level under
    `corroborate_rl`, NOT under the JAX-pulling
    `corroborate_rl.dqn` namespace — is the implementation's
    framework-facing surface; the heavy
    `corroborate_rl.dqn.yaml_sweep` is implementation detail
    behind the lightweight module's lazy proxies."""
    from corroborate_rl import dqn_sweep
    ep = dqn_sweep.SWEEP_ENTRY_POINTS
    assert isinstance(ep, SweepEntryPoints)
    # All five callables present (implementation provides full surface).
    assert ep.load_sweep is not None
    assert ep.dispatch_sweep is not None
    assert ep.default_registry is not None
    assert ep.expand_sweep is not None
    assert ep.format_dry_run_summary is not None


# ============ Dry-run path: end-to-end via __main__ ============


@pytest.fixture
def _shared_sweep_yaml(tmp_path: Path) -> Iterator[Path]:
    """A minimal `shared`-mode DQN sweep with one intervention.
    The body matches `tests/test_run_sweep_cli.py::test_dry_run_
    shared_sweep` so cross-test divergence stays visible."""
    cfg = tmp_path / 'shared.yaml'
    _ = cfg.write_text(
        'name: cli_smoke\n'
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
    yield cfg


def test_sweep_run_dry_run(
    _shared_sweep_yaml: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`corroborate sweep run <yaml> --dry-run` resolves the
    in-tree implementation, expands templates, and prints the summary
    + intervention count. Exit code 0; no JAX-platform touch."""
    rc = corroborate_main([
        'sweep', 'run', str(_shared_sweep_yaml), '--dry-run',
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "sweep: 'cli_smoke'" in out
    assert 'interventions (expanded): 2' in out
    assert '- vanilla' in out
    assert '- ddqn' in out
    assert 'dry-run: OK' in out


# ============ Substrate-resolution failure modes ============


def test_sweep_run_missing_substrate_module(
    _shared_sweep_yaml: Path,
) -> None:
    """A implementation name that doesn't import raises SystemExit
    with a message naming the missing module. The dispatch
    surfaces the import error before touching JAX or the YAML
    contents."""
    with pytest.raises(SystemExit, match='could not import'):
        _ = corroborate_main([
            'sweep', 'run', str(_shared_sweep_yaml),
            '--substrate', 'definitely.not.a.real.module',
            '--dry-run',
        ])


def test_sweep_run_substrate_without_entry_points(
    _shared_sweep_yaml: Path,
) -> None:
    """A implementation that imports but doesn't expose
    `SWEEP_ENTRY_POINTS` fails loud at the typed-shape check —
    not at the YAML load or dispatch step. The error message
    points at the missing attribute name so the implementation
    author knows where to fix it."""
    # `corroborate.bridge.verdict` is a real importable module
    # that has no `SWEEP_ENTRY_POINTS` — use it as the stand-in.
    with pytest.raises(SystemExit, match='SWEEP_ENTRY_POINTS'):
        _ = corroborate_main([
            'sweep', 'run', str(_shared_sweep_yaml),
            '--substrate', 'corroborate.bridge.verdict',
            '--dry-run',
        ])


def test_sweep_run_missing_yaml(tmp_path: Path) -> None:
    """A nonexistent YAML path exits 1 with a stderr message —
    not an import error, not an uncaught FileNotFoundError."""
    missing = tmp_path / 'no_such_file.yaml'
    rc = corroborate_main([
        'sweep', 'run', str(missing), '--dry-run',
    ])
    assert rc == 1
