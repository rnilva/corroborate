"""Cross-config lint for the sweep YAMLs.

Catches authoring mistakes that span multiple YAMLs and dispatch-
time conditions that the build-only paths can't see:

- **Duplicate `archive_remote`.** Two YAMLs pushing to the same S3
  prefix silently overwrite each other's merged top-level files;
  the local `assert_unique_remote_root` check (`integrity.py:278`)
  only scans `sweep_dir.parent` for siblings, so it doesn't catch
  YAMLs whose `out_dir`s sit in different parents.
- **Post-expansion `cfg.name` collisions.** `env_binding: per_env`
  templates without `{from_env: ...}` substitution in the `name`
  field produce duplicate config names after env expansion —
  CI9 raises at dispatch, but the build-only tests
  (`build_per_env`) don't trigger it. Lint runs the same check
  (`expand_sweep`) on every YAML at test time.

Failures here fail the PR before any sweep dispatch."""
from __future__ import annotations

from pathlib import Path

import pytest

from corroborate_rl.dqn.yaml_sweep import (
    default_dqn_registry, expand_sweep, load_sweep,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = REPO_ROOT / 'experiments' / 'configs'


def test_archive_remote_unique_across_configs() -> None:
    """No two YAMLs in `experiments/configs/` may declare the same
    `archive_remote`. The framework's per-sweep cell namespacing
    relies on `<archive_remote>/<cfg.name>/...` being unique per
    sweep; a shared `archive_remote` lets two sweeps' merged
    top-level files collide on the same S3 path with last-writer-
    wins semantics. See SWEEP_PERSISTENCY.md §I1."""
    reg = default_dqn_registry()
    by_remote: dict[str, list[str]] = {}
    for p in sorted(CONFIGS_DIR.glob('*.yaml')):
        sweep = load_sweep(p, reg=reg)
        if sweep.archive_remote is None:
            continue
        by_remote.setdefault(sweep.archive_remote, []).append(p.name)
    collisions = {
        remote: files for remote, files in by_remote.items()
        if len(files) > 1
    }
    assert not collisions, (
        f'archive_remote collisions across configs:\n'
        + '\n'.join(
            f'  {remote!r} ← {sorted(files)}'
            for remote, files in collisions.items()
        )
        + '\n\nTwo YAMLs sharing an `archive_remote` push to the '
        'same S3 prefix. Their merged top-level `runs.parquet` / '
        '`traces.parquet` collide on upload (last-writer-wins). '
        'Pick distinct prefixes per YAML.'
    )


def test_every_config_dispatches_cleanly() -> None:
    """For every YAML in `experiments/configs/`, `expand_sweep`
    must succeed — i.e., templates resolve through the registry
    AND post-expansion `cfg.name` values are unique.

    Catches dispatch-time failures the build-only tests miss:
    `env_binding: per_env` templates with bare `name: ddqn`
    (no `{from_env: ...}`) expand to N copies sharing the same
    name and would fail CI9 at dispatch. The build-only path
    (`build_per_env`) returns these without complaint."""
    reg = default_dqn_registry()
    failures: list[str] = []
    for p in sorted(CONFIGS_DIR.glob('*.yaml')):
        try:
            sweep = load_sweep(p, reg=reg)
            _ = expand_sweep(sweep, reg=reg)
        except Exception as e:
            failures.append(f'{p.name}: {type(e).__name__}: {e}')
    assert not failures, (
        'YAMLs that fail to expand cleanly:\n  '
        + '\n  '.join(failures)
    )
