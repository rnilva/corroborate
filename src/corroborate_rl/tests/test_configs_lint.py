"""Cross-config lint for the sweep YAMLs.

Catches authoring mistakes that span multiple YAMLs and that the
single-config loader can't see:

- **Duplicate `archive_remote`.** Two YAMLs pushing to the same S3
  prefix silently overwrite each other's merged top-level files;
  the local `assert_unique_remote_root` check (`integrity.py:278`)
  only scans `sweep_dir.parent` for siblings, so it doesn't catch
  YAMLs whose `out_dir`s sit in different parents.

The lint runs at test time so PRs that introduce a collision fail
fast, before any sweep dispatch."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from corroborate_rl.dqn.yaml_sweep import default_dqn_registry, load_sweep


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


def test_cfg_name_unique_within_each_config() -> None:
    """Within each YAML, every intervention's `name` must be unique.

    `dispatch_sweep` writes per-intervention parquets to
    `<out_dir>/<cfg.name>/`. Duplicate names overwrite each other
    at merge time, which CI9 raises on. This lint surfaces the
    same condition statically — useful when a YAML adds a new
    intervention and copy-pastes the `name` from a sibling."""
    reg = default_dqn_registry()
    failures: list[str] = []
    for p in sorted(CONFIGS_DIR.glob('*.yaml')):
        sweep = load_sweep(p, reg=reg)
        # Only `shared` mode resolves templates once; `per_env`
        # mode generates one InterventionConfig per env (the
        # template's `name` must include `{from_env: ...}` to
        # differentiate). We check the template-level names —
        # within-sweep N×envs collisions are CI9's job at dispatch.
        names = [
            t.get('name', '') for t in sweep.intervention_templates
        ]
        counts = Counter(names)
        dupes = {n: c for n, c in counts.items() if c > 1}
        if dupes:
            failures.append(f'{p.name}: {dupes}')
    assert not failures, (
        'duplicate intervention `name`s within a YAML:\n  '
        + '\n  '.join(failures)
    )
