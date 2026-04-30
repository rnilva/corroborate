"""Generic YAML-driven DQN sweep runner.

Replaces per-experiment Python scripts: pass a sweep YAML, the
runner loads it through the auto-registry, builds the arms, and
forwards to `run_hypotheses`. The YAML is the single authoring
surface; the corpus shape (parquets, archive, arm tags) is
unchanged from the per-script path.

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

from corroborate.rl.dqn.yaml_sweep import dispatch_sweep, load_sweep


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _ = parser.add_argument(
        'manifest', type=str,
        help='Path to the sweep YAML.',
    )
    # `argparse.Namespace` attribute access is `Any` by stub
    # convention; cast at the boundary, then runtime-check to
    # narrow before constructing the Path.
    manifest_attr = cast(object, parser.parse_args().manifest)
    if not isinstance(manifest_attr, str):
        raise TypeError(
            f'manifest path must be a string; got '
            f'{type(manifest_attr).__name__}',
        )

    from corroborate.rl.dqn.yaml_sweep import default_dqn_registry
    sweep = load_sweep(Path(manifest_attr), reg=default_dqn_registry())
    print(
        f'sweep: {sweep.name} '
        f'({len(sweep.hypothesis_templates)} hypotheses × '
        f'{len(sweep.envs)} envs, '
        f'arms_shape={sweep.arms_shape})',
        flush=True,
    )
    runs, traces = dispatch_sweep(sweep)
    print(f'corpus → {runs}, {traces}', flush=True)


if __name__ == '__main__':
    main()
