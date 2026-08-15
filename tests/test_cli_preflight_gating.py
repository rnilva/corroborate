"""CLI-level preflight gating tests for `run_sweep.py` and the
`corroborate hypothesis` subcommand (also reachable via the
`scripts/run_hypothesis.py` back-compat shim). The unit-level
preflight tests cover `preflight()` in isolation; these cover the
gating logic that decides WHEN to call it.
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest


# ============ corroborate.cli.hypothesis — _check_and_report uses preflight ============

# We exercise `main()` directly with a mock for `_preflight` so no
# cloud / JAX work happens. The `_preflight` import in
# `corroborate.cli.hypothesis` is function-local; patch the source
# path so any indirect import resolves to the stub.

_HYP_PREFLIGHT_PATH = (
    'corroborate._internals.cloud_auth.preflight'
)
_HYP_PRINT_PATH = 'corroborate.cli.hypothesis._print_verdicts'


def _write_remote_json(d: Path, remote_root: str) -> None:
    """Hand-write a valid `_remote.json` so the hypothesis script's
    preflight gate fires."""
    import json
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        'remote_root': remote_root,
        'files': [],
    }
    _ = (d / '_remote.json').write_text(json.dumps(payload, indent=2))


def test_hypothesis_skips_preflight_when_no_ingest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """No --ingest mode → no cloud touch → preflight NOT called."""
    # Avoid actually running the hypothesis pipeline (heavy JAX).
    # We only care that preflight isn't invoked.
    called: list[object] = []

    def _fake_preflight(remote: str, *, profile: str | None = None) -> None:
        called.append(remote)

    monkeypatch.setattr(_HYP_PREFLIGHT_PATH, _fake_preflight)

    # Patch `run` so we don't actually evaluate bridges.
    with patch('corroborate.cli.hypothesis.run', return_value={}), \
         patch(_HYP_PRINT_PATH):
        from corroborate.cli.hypothesis import main
        # Use --check mode which returns before the bridge eval +
        # before the preflight block (check is early-return).
        # Instead, use --no-report --no-cache to skip writes.
        # Actually, the cleanest no-cloud no-ingest path is just
        # "no args after module" with --no-report.
        _ = main([
            'tests.probes.stub_hypothesis',
            '--no-report', '--no-cache',
        ])
    # We care only that preflight wasn't called; the run-level
    # return code is irrelevant.
    assert not called, (
        f'preflight should NOT be called without --ingest; got '
        f'calls: {called}'
    )


def test_hypothesis_skips_preflight_when_no_remote_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """--ingest <corpus> but the corpus has no `_remote.json` →
    no cloud touch possible → preflight NOT called."""
    corpus = tmp_path / 'local_only_corpus'
    corpus.mkdir()
    # No _remote.json.

    called: list[object] = []

    def _fake_preflight(remote: str, *, profile: str | None = None) -> None:
        called.append(remote)

    monkeypatch.setattr(_HYP_PREFLIGHT_PATH, _fake_preflight)
    with patch('corroborate.cli.hypothesis.run', return_value={}), \
         patch(_HYP_PRINT_PATH):
        from corroborate.cli.hypothesis import main
        _ = main([
            'tests.probes.stub_hypothesis',
            '--ingest', str(corpus),
            '--no-report', '--no-cache',
        ])
    assert not called, (
        f'preflight should NOT be called without _remote.json; '
        f'got calls: {called}'
    )


def test_hypothesis_runs_preflight_when_remote_json_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """--ingest <corpus> + corpus has `_remote.json` → preflight
    IS called with the bucket from the manifest."""
    corpus = tmp_path / 'cloud_backed_corpus'
    _write_remote_json(
        corpus, 's3://my-bucket/cloud_backed_corpus',
    )

    called: list[str] = []

    def _fake_preflight(remote: str, *, profile: str | None = None) -> None:
        called.append(remote)

    monkeypatch.setattr(_HYP_PREFLIGHT_PATH, _fake_preflight)
    with patch('corroborate.cli.hypothesis.run', return_value={}), \
         patch(_HYP_PRINT_PATH):
        from corroborate.cli.hypothesis import main
        _ = main([
            'tests.probes.stub_hypothesis',
            '--ingest', str(corpus),
            '--no-report', '--no-cache',
        ])
    assert called == ['s3://my-bucket/cloud_backed_corpus'], (
        f'expected exactly one preflight call for the bucket URI; '
        f'got: {called}'
    )


def test_hypothesis_skip_preflight_flag_disables_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """--skip-preflight bypasses the check even when a
    cloud-backed corpus would otherwise trigger it."""
    corpus = tmp_path / 'cloud_backed_corpus'
    _write_remote_json(corpus, 's3://my-bucket/x')

    called: list[object] = []

    def _fake_preflight(remote: str, *, profile: str | None = None) -> None:
        called.append(remote)

    monkeypatch.setattr(_HYP_PREFLIGHT_PATH, _fake_preflight)
    with patch('corroborate.cli.hypothesis.run', return_value={}), \
         patch(_HYP_PRINT_PATH):
        from corroborate.cli.hypothesis import main
        _ = main([
            'tests.probes.stub_hypothesis',
            '--ingest', str(corpus),
            '--skip-preflight',
            '--no-report', '--no-cache',
        ])
    assert not called, (
        f'--skip-preflight should suppress the preflight call; '
        f'got: {called}'
    )


def test_hypothesis_no_restore_skips_preflight(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """--no-restore → runner won't pull from cloud → preflight
    skipped (no cloud touch to guard)."""
    corpus = tmp_path / 'cloud_backed_corpus'
    _write_remote_json(corpus, 's3://my-bucket/x')

    called: list[object] = []

    def _fake_preflight(remote: str, *, profile: str | None = None) -> None:
        called.append(remote)

    monkeypatch.setattr(_HYP_PREFLIGHT_PATH, _fake_preflight)
    with patch('corroborate.cli.hypothesis.run', return_value={}), \
         patch(_HYP_PRINT_PATH):
        from corroborate.cli.hypothesis import main
        _ = main([
            'tests.probes.stub_hypothesis',
            '--ingest', str(corpus),
            '--no-restore',
            '--no-report', '--no-cache',
        ])
    assert not called


def test_hypothesis_profile_exported_to_env_independent_of_preflight(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Regression: --profile must export AWS_PROFILE to os.environ
    even when preflight is skipped (--skip-preflight). Otherwise
    the downstream cloud op falls back to the default chain."""
    corpus = tmp_path / 'cloud_backed_corpus'
    _write_remote_json(corpus, 's3://my-bucket/x')

    monkeypatch.delenv('AWS_PROFILE', raising=False)
    monkeypatch.setattr(_HYP_PREFLIGHT_PATH, lambda *_, **__: None)
    with patch('corroborate.cli.hypothesis.run', return_value={}), \
         patch(_HYP_PRINT_PATH):
        from corroborate.cli.hypothesis import main
        _ = main([
            'tests.probes.stub_hypothesis',
            '--ingest', str(corpus),
            '--profile', 'r2',
            '--skip-preflight',  # CRITICAL: profile should still export
            '--no-report', '--no-cache',
        ])
    assert os.environ.get('AWS_PROFILE') == 'r2', (
        f'--profile must export AWS_PROFILE even with --skip-preflight; '
        f'got: {os.environ.get("AWS_PROFILE")!r}'
    )


def test_hypothesis_nested_corpus_triggers_preflight(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Regression: `--ingest-all <root>` where the cloud-backed
    corpus is NESTED two levels deep (e.g.
    `k_sweep_acrobot/ddqn_vs_vanilla/_remote.json`) must trigger
    preflight. The earlier one-level `iterdir()` walk missed
    nested corpora and silently skipped preflight."""
    root = tmp_path / 'data'
    nested = root / 'parent_sweep' / 'child_corpus'
    _write_remote_json(nested, 's3://my-bucket/nested')

    called: list[str] = []

    def _fake_preflight(remote: str, *, profile: str | None = None) -> None:
        called.append(remote)

    monkeypatch.setattr(_HYP_PREFLIGHT_PATH, _fake_preflight)
    with patch('corroborate.cli.hypothesis.run', return_value={}), \
         patch(_HYP_PRINT_PATH):
        from corroborate.cli.hypothesis import main
        _ = main([
            'tests.probes.stub_hypothesis',
            '--ingest-all', str(root),
            '--no-report', '--no-cache',
        ])
    assert called == ['s3://my-bucket/nested'], (
        f'nested-corpus walk should reach 2 levels deep; got: {called}'
    )


def test_hypothesis_named_ingest_nested_corpus_triggers_preflight(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Same nested case but via `--ingest <parent>` (Sequence[Path]
    code path). Should reach the child's _remote.json."""
    parent = tmp_path / 'parent_sweep'
    child = parent / 'child_corpus'
    _write_remote_json(child, 's3://my-bucket/named-nested')

    called: list[str] = []

    def _fake_preflight(remote: str, *, profile: str | None = None) -> None:
        called.append(remote)

    monkeypatch.setattr(_HYP_PREFLIGHT_PATH, _fake_preflight)
    with patch('corroborate.cli.hypothesis.run', return_value={}), \
         patch(_HYP_PRINT_PATH):
        from corroborate.cli.hypothesis import main
        _ = main([
            'tests.probes.stub_hypothesis',
            '--ingest', str(parent),
            '--no-report', '--no-cache',
        ])
    assert called == ['s3://my-bucket/named-nested'], (
        f'named-ingest nested-corpus walk should reach 2 levels; '
        f'got: {called}'
    )


def test_hypothesis_corrupt_remote_json_fails_cleanly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Corrupt _remote.json should surface as a clean preflight
    error (exit 1 + diagnostic stderr) rather than an unhandled
    JSONDecodeError / TypeError trace."""
    corpus = tmp_path / 'corrupt_corpus'
    corpus.mkdir()
    _ = (corpus / '_remote.json').write_text('not-json{{')

    monkeypatch.setattr(_HYP_PREFLIGHT_PATH, lambda *_, **__: None)
    with patch('corroborate.cli.hypothesis.run', return_value={}), \
         patch(_HYP_PRINT_PATH):
        from corroborate.cli.hypothesis import main
        rc = main([
            'tests.probes.stub_hypothesis',
            '--ingest', str(corpus),
            '--no-report', '--no-cache',
        ])
    assert rc == 1
    err = capsys.readouterr().err
    assert 'corrupt manifest' in err.lower()


# ============ corroborate sweep run preflight gating ============

# Implementation dispatch is heavier — `corroborate_rl.dqn.yaml_sweep`
# pulls JAX via the eager `from corroborate_rl.dqn import
# measurables` in `corroborate_rl.dqn.__init__`. Tests here mock
# the substrate's `load_sweep` / `dispatch_sweep` via the lazy
# proxies in `corroborate_rl.dqn_sweep` (which the framework's
# `SWEEP_ENTRY_POINTS` Callables route through). A real
# minimal `DQNSweep` instance is used to satisfy the proxy's
# defensive isinstance narrow.


def _make_stub_sweep(archive_remote: str | None) -> object:
    """Construct a minimal real `DQNSweep` for these mock tests.

    Uses the substrate's actual dataclass (which structurally
    satisfies the framework `Sweep` Protocol) rather than a
    duck-typed stub, because `corroborate_rl.dqn_sweep`'s lazy
    proxies do a defensive `isinstance(sweep, DQNSweep)` narrow
    before delegating to the heavy module."""
    from corroborate_rl.dqn.collect import EnvConfig
    from corroborate_rl.dqn.yaml_sweep import DQNSweep
    return DQNSweep(
        name='stub_sweep',
        out_dir=Path('/tmp/stub_out'),
        envs=(EnvConfig(env_name='TestEnv', n_seeds=2, chunk_size=2),),
        intervention_templates=(),
        env_binding='shared',
        archive_remote=archive_remote,
    )


def _patch_substrate_callables(
    monkeypatch: pytest.MonkeyPatch,
    archive_remote: str | None,
) -> None:
    """Monkey-patch the substrate's heavy `load_sweep` /
    `dispatch_sweep` so the framework's CLI dispatch doesn't
    actually run a sweep. Patches the underlying functions; the
    substrate's lazy proxies `from corroborate_rl.dqn.yaml_sweep
    import load_sweep` resolves to the patched attribute at
    proxy-call time."""
    stub_sweep = _make_stub_sweep(archive_remote)
    monkeypatch.setattr(
        'corroborate_rl.dqn.yaml_sweep.load_sweep',
        lambda _path, reg=None: stub_sweep,
    )
    monkeypatch.setattr(
        'corroborate_rl.dqn.yaml_sweep.dispatch_sweep',
        lambda _sweep: (Path('/tmp/runs.parquet'),
                        Path('/tmp/traces.parquet')),
    )


def _run_cli_sweep(argv: list[str]) -> int:
    """Drive `corroborate.cli.sweep.dispatch` through the
    public CLI surface — same flow as `corroborate sweep run`."""
    import argparse as _argparse
    from corroborate.cli.sweep import add_args, dispatch
    parser = _argparse.ArgumentParser(prog='corroborate sweep')
    add_args(parser, argv=argv)
    ns = parser.parse_args(argv)
    return dispatch(ns)


def test_sweep_skips_preflight_when_archive_remote_is_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """No `archive_remote` in the sweep config → no cloud touch →
    no preflight."""
    cfg = tmp_path / 'sweep.yaml'
    _ = cfg.write_text('# stub')

    called: list[object] = []

    def _fake_preflight(remote: str, *, profile: str | None = None) -> None:
        called.append(remote)

    monkeypatch.setattr(
        'corroborate._internals.cloud_auth.preflight', _fake_preflight,
    )
    _patch_substrate_callables(monkeypatch, archive_remote=None)
    _ = _run_cli_sweep(['run', str(cfg)])
    assert not called


def test_sweep_runs_preflight_when_archive_remote_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """`archive_remote` set + no --skip-preflight → preflight IS
    called with the configured remote."""
    cfg = tmp_path / 'sweep.yaml'
    _ = cfg.write_text('# stub')

    called: list[str] = []

    def _fake_preflight(remote: str, *, profile: str | None = None) -> None:
        called.append(remote)

    monkeypatch.setattr(
        'corroborate._internals.cloud_auth.preflight', _fake_preflight,
    )
    _patch_substrate_callables(
        monkeypatch, archive_remote='s3://my-bucket/x',
    )
    _ = _run_cli_sweep(['run', str(cfg)])
    assert called == ['s3://my-bucket/x']


def test_sweep_skip_preflight_disables_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    cfg = tmp_path / 'sweep.yaml'
    _ = cfg.write_text('# stub')

    called: list[object] = []

    def _fake_preflight(remote: str, *, profile: str | None = None) -> None:
        called.append(remote)

    monkeypatch.setattr(
        'corroborate._internals.cloud_auth.preflight', _fake_preflight,
    )
    _patch_substrate_callables(
        monkeypatch, archive_remote='s3://my-bucket/x',
    )
    _ = _run_cli_sweep(['run', str(cfg), '--skip-preflight'])
    assert not called


def test_sweep_profile_exported_with_skip_preflight(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Regression: --profile must export AWS_PROFILE even when
    --skip-preflight is set."""
    cfg = tmp_path / 'sweep.yaml'
    _ = cfg.write_text('# stub')

    monkeypatch.delenv('AWS_PROFILE', raising=False)
    monkeypatch.setattr(
        'corroborate._internals.cloud_auth.preflight',
        lambda *_, **__: None,
    )
    _patch_substrate_callables(
        monkeypatch, archive_remote='s3://my-bucket/x',
    )
    _ = _run_cli_sweep([
        'run', str(cfg), '--profile', 'r2', '--skip-preflight',
    ])
    assert os.environ.get('AWS_PROFILE') == 'r2'
