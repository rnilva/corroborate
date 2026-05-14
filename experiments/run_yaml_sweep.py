"""Generic YAML-driven DQN sweep runner.

Replaces per-experiment Python scripts: pass a sweep YAML, the
runner loads it through the auto-registry, builds the arms, and
forwards to `run_intervention`. The YAML is the single authoring
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

from corroborate_rl.dqn.yaml_sweep import dispatch_sweep, load_sweep


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _ = parser.add_argument(
        'sweep', type=str,
        help='Path to the sweep YAML.',
    )
    # `argparse.Namespace` attribute access is `Any` by stub
    # convention. Annotate at assignment so the type narrows to
    # `object`, then `isinstance` narrows further.
    sweep_path_attr: object = parser.parse_args().sweep
    if not isinstance(sweep_path_attr, str):
        raise TypeError(
            f'sweep path must be a string; got '
            f'{type(sweep_path_attr).__name__}',
        )

    from corroborate_rl.dqn.yaml_sweep import default_dqn_registry
    sweep = load_sweep(Path(sweep_path_attr), reg=default_dqn_registry())
    print(
        f'sweep: {sweep.name} '
        f'({len(sweep.intervention_templates)} interventions × '
        f'{len(sweep.envs)} envs, '
        f'env_binding={sweep.env_binding})',
        flush=True,
    )
    runs, traces = dispatch_sweep(sweep)
    print(f'corpus → {runs}, {traces}', flush=True)


if __name__ == '__main__':
    main()
