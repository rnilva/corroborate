"""Substrate-extends-CLI mechanism — framework-side tests.

Exercises `corroborate.cli.sweep`'s two-phase argparse flow:
`peek_substrate` parses `--substrate <name>` out of argv,
`load_substrate` imports the substrate module and reads both
`SWEEP_ENTRY_POINTS` (required) and `SWEEP_CLI_EXTENSIONS`
(optional) from a single import, and the extensions'
`add_args` / `pre_import_setup` callbacks are wired into the
dispatch flow.

The in-tree RL substrate's `corroborate_rl.dqn_sweep` is the
canonical lightweight-module example — top-level under
`corroborate_rl` (NOT under `corroborate_rl.dqn`) so the eager
`from corroborate_rl.dqn import measurables` side-effect (which
pulls JAX) doesn't fire on `corroborate_rl.dqn_sweep` import.
Its `SWEEP_ENTRY_POINTS` Callables are lazy proxies that import
the heavy `corroborate_rl.dqn.yaml_sweep` on first invocation —
by which time `pre_import_setup` has stamped `JAX_PLATFORMS`."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pytest

from corroborate.cli.sweep import (
    load_substrate,
    peek_substrate,
)
from corroborate.runner.yaml_sweep import (
    SweepCliExtensions,
    SweepEntryPoints,
)


# ============ Load-bearing JAX-free invariant (subprocess) ============


def test_in_tree_substrate_module_does_not_pull_jax_at_import() -> None:
    """The in-tree DQN substrate's lightweight entry module
    `corroborate_rl.dqn_sweep` MUST stay JAX-free at module-load
    time. If a future edit accidentally imports JAX (or anything
    that pulls JAX, like `corroborate_rl.dqn.measurables`) at
    module load, the framework's `pre_import_setup` hook
    becomes a no-op — JAX latches the backend BEFORE
    `JAX_PLATFORMS` is stamped.

    Run in a fresh subprocess because `sys.modules` persists
    across pytest's own tree (some prior test in the suite has
    almost certainly loaded JAX already)."""
    import subprocess
    import sys
    result = subprocess.run(
        [
            sys.executable, '-c',
            'import corroborate_rl.dqn_sweep; import sys; '
            'assert "jax" not in sys.modules, '
            '"JAX leaked at corroborate_rl.dqn_sweep module load — '
            'lazy-proxy design defeated"; '
            'assert "corroborate_rl.dqn" not in sys.modules, '
            '"heavy substrate package leaked at module load"',
        ],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, (
        f'subprocess failed:\n'
        f'stdout: {result.stdout}\n'
        f'stderr: {result.stderr}'
    )


# ============ peek_substrate ============


def test_peek_substrate_flag_form() -> None:
    """`--substrate name` parses correctly."""
    result = peek_substrate(
        ['run', 'cfg.yaml', '--substrate', 'foo.bar', '--dry-run'],
    )
    assert result == 'foo.bar'


def test_peek_substrate_equals_form() -> None:
    """`--substrate=name` parses correctly."""
    result = peek_substrate(
        ['run', 'cfg.yaml', '--substrate=foo.bar', '--dry-run'],
    )
    assert result == 'foo.bar'


def test_peek_substrate_falls_back_to_default() -> None:
    """argv without `--substrate` returns the documented default
    (resolved through the same precedence as `dispatch`'s
    `_resolve_substrate_name` so the peek matches downstream)."""
    result = peek_substrate(['run', 'cfg.yaml', '--dry-run'])
    assert result == 'corroborate_rl.dqn_sweep'


def test_peek_substrate_respects_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`$CORROBORATE_SWEEP_SUBSTRATE` takes precedence over the
    documented default when argv carries no explicit flag."""
    monkeypatch.setenv(
        'CORROBORATE_SWEEP_SUBSTRATE', 'env.override.module',
    )
    result = peek_substrate(['run', 'cfg.yaml'])
    assert result == 'env.override.module'


def test_peek_substrate_flag_wins_over_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit `--substrate` argv flag wins over env var."""
    monkeypatch.setenv(
        'CORROBORATE_SWEEP_SUBSTRATE', 'env.override',
    )
    result = peek_substrate(['run', '--substrate', 'cli.override'])
    assert result == 'cli.override'


def test_peek_substrate_env_var_with_equals_form_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit `--substrate=name` (equals form) wins over env var.
    Test gap flagged in review #10 — was previously only covered
    for the space-separated form."""
    monkeypatch.setenv(
        'CORROBORATE_SWEEP_SUBSTRATE', 'env.override',
    )
    result = peek_substrate(['run', '--substrate=cli.override'])
    assert result == 'cli.override'


def test_peek_substrate_last_wins_on_duplicate() -> None:
    """argparse takes the LAST `--substrate` value on duplicates;
    `peek_substrate` must match — reviewer caught the prior
    first-wins divergence that produced confusing failures
    (peek loaded foo_cli, dispatch tried to import bar). Now
    both follow argparse's last-wins."""
    result = peek_substrate([
        'run', 'cfg.yaml',
        '--substrate', 'first.module',
        '--substrate', 'second.module',
        '--dry-run',
    ])
    assert result == 'second.module'


# ============ load_substrate ============


def testload_substrate_in_tree_returns_both_attrs() -> None:
    """The in-tree RL substrate `corroborate_rl.dqn_sweep` exports
    BOTH `SWEEP_ENTRY_POINTS` and `SWEEP_CLI_EXTENSIONS` —
    `load_substrate` returns both from a single import."""
    ep, ext = load_substrate('corroborate_rl.dqn_sweep')
    assert isinstance(ep, SweepEntryPoints)
    assert isinstance(ext, SweepCliExtensions)


def testload_substrate_missing_module_raises_with_hint() -> None:
    """A typo'd `--substrate` produces a SystemExit with a
    parent-module-imports-cleanly hint pointing at the leaf
    name. Closes review major #3."""
    with pytest.raises(SystemExit, match='check the leaf name'):
        _ = load_substrate('corroborate_rl.definitely_typoed_leaf')


def testload_substrate_missing_parent_raises_with_different_hint() -> None:
    """A substrate with a missing parent package gets a distinct
    hint ('also missing — check that the substrate package is
    installed') — different failure mode from a typo."""
    with pytest.raises(SystemExit, match='package is\\s+installed'):
        _ = load_substrate('definitely_not_a_real.substrate.module')


def testload_substrate_without_entry_points_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A substrate module that imports cleanly but doesn't
    export `SWEEP_ENTRY_POINTS` fails loud."""
    _ = (tmp_path / 'missing_ep_substrate.py').write_text(
        '# Substrate stub without SWEEP_ENTRY_POINTS\n',
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(SystemExit, match='SWEEP_ENTRY_POINTS'):
        _ = load_substrate('missing_ep_substrate')


def testload_substrate_wrong_type_entry_points_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`SWEEP_ENTRY_POINTS` of the wrong type fails loud."""
    _ = (tmp_path / 'badep_substrate.py').write_text(
        'SWEEP_ENTRY_POINTS = {"load_sweep": None}\n',
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(SystemExit, match='SweepEntryPoints'):
        _ = load_substrate('badep_substrate')


def _write_minimal_substrate(
    tmp_path: Path, name: str, extra_lines: str = '',
) -> None:
    """Write a minimal substrate-fixture module to tmp_path."""
    _ = (tmp_path / f'{name}.py').write_text(
        'from pathlib import Path\n'
        'from corroborate.runner.registry import Registry\n'
        'from corroborate.runner.yaml_sweep import (\n'
        '    Sweep, SweepEntryPoints,\n'
        ')\n'
        '\n'
        f'{extra_lines}'
        '\n'
        'def _load(path: Path, *, reg: Registry) -> Sweep:\n'
        '    raise NotImplementedError\n'
        '\n'
        'def _dispatch(sweep: Sweep) -> tuple[Path, Path]:\n'
        '    raise NotImplementedError\n'
        '\n'
        'def _registry() -> Registry:\n'
        '    return Registry()\n'
        '\n'
        'SWEEP_ENTRY_POINTS: SweepEntryPoints[Sweep] = SweepEntryPoints[Sweep](\n'
        '    load_sweep=_load,\n'
        '    dispatch_sweep=_dispatch,\n'
        '    default_registry=_registry,\n'
        ')\n'
    )


def testload_substrate_without_cli_extensions_returns_none(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A substrate that exports `SWEEP_ENTRY_POINTS` but NOT
    `SWEEP_CLI_EXTENSIONS` is valid — the framework just doesn't
    register substrate-specific args. Closes review #10's
    'no-cli-module' gap."""
    _write_minimal_substrate(tmp_path, 'no_cli_substrate')
    monkeypatch.syspath_prepend(str(tmp_path))
    ep, ext = load_substrate('no_cli_substrate')
    assert isinstance(ep, SweepEntryPoints)
    assert ext is None


def testload_substrate_with_none_cli_extensions_returns_none(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A substrate that intentionally sets `SWEEP_CLI_EXTENSIONS =
    None` (rather than omitting) is also accepted; treated the
    same as missing. Distinguishes the `_MISSING` sentinel's role
    (separates missing-from-None only in the wrong-type error
    path). Closes review #10's '=None' gap."""
    _write_minimal_substrate(
        tmp_path, 'none_cli_substrate',
        extra_lines='SWEEP_CLI_EXTENSIONS = None\n',
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    ep, ext = load_substrate('none_cli_substrate')
    assert isinstance(ep, SweepEntryPoints)
    assert ext is None


def testload_substrate_wrong_type_cli_extensions_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`SWEEP_CLI_EXTENSIONS` of the wrong type fails loud."""
    _write_minimal_substrate(
        tmp_path, 'badext_substrate',
        extra_lines='SWEEP_CLI_EXTENSIONS = {"add_args": None}\n',
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(SystemExit, match='SweepCliExtensions'):
        _ = load_substrate('badext_substrate')


# ============ SweepCliExtensions runtime shape check ============


def test_sweep_cli_extensions_post_init_rejects_non_callable() -> None:
    """`__post_init__` fail-loud check (nit #11c) rejects a
    non-Callable add_args at construction, not at first
    invocation."""
    def _good_setup(args: argparse.Namespace) -> None:
        del args

    with pytest.raises(TypeError, match='add_args must be callable'):
        _ = SweepCliExtensions(
            add_args='not a callable',  # pyright: ignore[reportArgumentType]
            pre_import_setup=_good_setup,
        )


def test_sweep_cli_extensions_post_init_rejects_non_callable_setup() -> None:
    """Same fail-loud check on `pre_import_setup`."""
    def _good_add_args(parser: argparse.ArgumentParser) -> None:
        del parser

    with pytest.raises(
        TypeError, match='pre_import_setup must be callable',
    ):
        _ = SweepCliExtensions(
            add_args=_good_add_args,
            pre_import_setup=42,  # pyright: ignore[reportArgumentType]
        )


# ============ End-to-end: substrate args appear in --help ============


def test_substrate_args_appear_in_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`corroborate sweep run --help` includes the in-tree
    substrate's `--device {cpu,gpu}` option — proves the
    framework's `add_args(p_run, argv=...)` correctly threaded
    argv to the substrate's `add_args(p_run)` callback."""
    from corroborate.cli.sweep import add_args
    parser = argparse.ArgumentParser(prog='corroborate sweep')
    add_args(parser, argv=['run', 'cfg.yaml', '--help'])

    with pytest.raises(SystemExit) as exc_info:
        _ = parser.parse_args(['run', 'cfg.yaml', '--help'])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert '--device' in out
    assert '{cpu,gpu}' in out


# ============ Load-bearing invariant: pre_import_setup BEFORE heavy import ============


def test_pre_import_setup_runs_before_entry_point_callable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A substrate exposes lazy `SWEEP_ENTRY_POINTS` Callables;
    the framework's `dispatch` must call `pre_import_setup`
    BEFORE any of them fires. Verified by routing through
    `corroborate.cli.sweep.dispatch` against a fake substrate
    whose `default_registry()` lazy proxy asserts a sentinel
    env var (set in `pre_import_setup`) is present. If a future
    refactor reordered the calls in `dispatch`, the assert
    inside `default_registry()` would fire."""
    _ = (tmp_path / 'ordering_test_substrate.py').write_text(
        'import argparse\n'
        'import os\n'
        'from pathlib import Path\n'
        'from corroborate.runner.registry import Registry\n'
        'from corroborate.runner.yaml_sweep import (\n'
        '    Sweep, SweepCliExtensions, SweepEntryPoints,\n'
        ')\n'
        '\n'
        '_SENTINEL_ENV = "ORDERING_TEST_SENTINEL"\n'
        '\n'
        'def _add_args(parser: argparse.ArgumentParser) -> None:\n'
        '    del parser\n'
        '\n'
        'def _pre_import_setup(args: argparse.Namespace) -> None:\n'
        '    del args\n'
        '    os.environ[_SENTINEL_ENV] = "1"\n'
        '\n'
        'SWEEP_CLI_EXTENSIONS = SweepCliExtensions(\n'
        '    add_args=_add_args,\n'
        '    pre_import_setup=_pre_import_setup,\n'
        ')\n'
        '\n'
        'def _default_registry_lazy() -> Registry:\n'
        '    # If `dispatch` swaps the order (calls this before\n'
        '    # `pre_import_setup`), the sentinel is missing.\n'
        '    assert os.environ.get(_SENTINEL_ENV) == "1", (\n'
        '        "pre_import_setup did not run before "\n'
        '        "default_registry() — call order bug in "\n'
        '        "corroborate.cli.sweep.dispatch"\n'
        '    )\n'
        '    return Registry()\n'
        '\n'
        'def _load_sweep_lazy(path: Path, *, reg: Registry) -> Sweep:\n'
        '    raise NotImplementedError(\n'
        '        "ordering_test_substrate is a setup-order probe"\n'
        '    )\n'
        '\n'
        'def _dispatch_lazy(sweep: Sweep) -> tuple[Path, Path]:\n'
        '    raise NotImplementedError\n'
        '\n'
        'SWEEP_ENTRY_POINTS: SweepEntryPoints[Sweep] = SweepEntryPoints[Sweep](\n'
        '    load_sweep=_load_sweep_lazy,\n'
        '    dispatch_sweep=_dispatch_lazy,\n'
        '    default_registry=_default_registry_lazy,\n'
        ')\n'
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delenv('ORDERING_TEST_SENTINEL', raising=False)

    cfg_file = tmp_path / 'cfg.yaml'
    _ = cfg_file.write_text('# unused\n')

    # Drive dispatch via the public CLI surface so we exercise
    # the actual call ordering: parser build → parse → dispatch.
    from corroborate.cli.sweep import add_args, dispatch
    parser = argparse.ArgumentParser(prog='corroborate sweep')
    # No --dry-run: the dispatch path goes default_registry()
    # (substrate's assert) → load_sweep() (NotImplementedError).
    # --dry-run would short-circuit at the "expand_sweep is None"
    # check since this fake substrate doesn't provide one.
    argv = [
        'run', str(cfg_file),
        '--substrate', 'ordering_test_substrate',
    ]
    add_args(parser, argv=argv)
    ns = parser.parse_args(argv)

    # The substrate's `default_registry()` has an internal assert
    # that fires if `dispatch` swapped the call order. We expect
    # dispatch to fail at `load_sweep` (NotImplementedError) AFTER
    # `default_registry()` succeeded (i.e., the assert passed).
    with pytest.raises(NotImplementedError):
        _ = dispatch(ns)
    # Sentinel confirms pre_import_setup ran.
    assert os.environ.get('ORDERING_TEST_SENTINEL') == '1'
    monkeypatch.delenv('ORDERING_TEST_SENTINEL', raising=False)


# Suppress unused-import lints; `sys` is referenced inside
# fixture write_text strings only.
_ = sys
