"""Generic YAML-driven DQN sweep runner.

Replaces per-experiment Python scripts: pass a manifest YAML, the
runner loads it through the auto-registry, builds the arms, and
forwards to `run_hypotheses`. The manifest is the single
authoring surface; the corpus shape (parquets, archive, arm
tags) is unchanged from the per-script path.

Usage:
  uv run python experiments/run_yaml_sweep.py \\
      experiments/configs/expectile_3way.yaml
"""
from __future__ import annotations

import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.9')

import argparse
from pathlib import Path
from typing import cast

from corroborate.rl.dqn.yaml_sweep import (
    default_dqn_registry, dispatch_manifest, load_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _ = parser.add_argument(
        'manifest', type=Path,
        help='Path to the experiment YAML manifest.',
    )
    args = parser.parse_args()
    # argparse types attribute access as Any — cast at the boundary.
    manifest_path = cast(Path, args.manifest)

    reg = default_dqn_registry()
    manifest = load_manifest(manifest_path, reg=reg)
    print(
        f'manifest: {manifest.name} '
        f'({len(manifest.hypotheses)} hypotheses × '
        f'{len(manifest.envs)} envs, '
        f'arms_shape={manifest.arms_shape})',
        flush=True,
    )
    runs, traces = dispatch_manifest(manifest)
    print(f'corpus → {runs}, {traces}', flush=True)


if __name__ == '__main__':
    main()
