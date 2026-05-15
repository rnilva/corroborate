"""Run a YAML-configured DQN sweep end-to-end.

Usage:
    set -a && . .env && set +a   # AWS creds for archive
    PYTHONPATH=. uv run python scripts/run_sweep.py \\
        experiments/configs/<sweep>.yaml

`--dry-run` validates the YAML against the dispatch path
(`expand_sweep`) and prints the resolved config list without
touching JAX/GPU or writing any cells. Use it to catch authoring
mistakes (template typos, env catalogue misses, unknown
measurables, post-expansion `cfg.name` collisions) before
committing GPU time.

Forces JAX to CPU by default (set `JAX_PLATFORMS=gpu` to override).
The sweep config decides `out_dir` and `archive_remote`; the
dispatcher writes per-arm sub-corpora under `out_dir/<name>/`
then merges to top-level `out_dir/runs.parquet` +
`traces.parquet`.

Per-arm sub-corpora persist after the merge — CORPUS_INTEGRITY.md
CI1 forbids them at runner ingest time. Run
`rm -rf <out_dir>/{<arm>}/` after a successful sweep, OR flatten
into sibling top-level dirs if you want each arm as a distinct
corpus.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog='run_sweep',
        description='Run a YAML-configured DQN sweep on the local '
                    'machine (CPU by default).',
    )
    _ = parser.add_argument(
        'config', type=Path,
        help='Path to the YAML sweep config to run.',
    )
    _ = parser.add_argument(
        '--device', choices=['cpu', 'gpu'], default='cpu',
        help='JAX platform. CPU is the safe default for small envs '
             '(FourRooms-class); flip to GPU if it exists + '
             'matters.',
    )
    _ = parser.add_argument(
        '--dry-run', action='store_true',
        help='Validate the YAML through the dispatch path '
             '(`expand_sweep`) and print the resolved configs '
             'without touching JAX or writing any cells.',
    )
    args = parser.parse_args(argv)

    cfg_path_attr: object = args.config
    if not isinstance(cfg_path_attr, Path):
        raise TypeError(
            f'config path must be a Path; got '
            f'{type(cfg_path_attr).__name__}',
        )
    cfg_path: Path = cfg_path_attr
    if not cfg_path.exists():
        raise SystemExit(f'config not found: {cfg_path}')

    dry_run_attr: object = args.dry_run
    if not isinstance(dry_run_attr, bool):
        raise TypeError(
            f'--dry-run must be bool; got '
            f'{type(dry_run_attr).__name__}',
        )

    if dry_run_attr:
        _dry_run(cfg_path)
        return

    device_attr: object = args.device
    if not isinstance(device_attr, str):
        raise TypeError(
            f'--device must be a string; got '
            f'{type(device_attr).__name__}',
        )

    # JAX prefers explicit backend names: 'cpu' / 'cuda' / 'rocm' / 'tpu'.
    # The CLI surface is cpu/gpu for ergonomics; map gpu→cuda since that's
    # what's installed on this machine. Caller can override via
    # JAX_PLATFORMS env var (e.g. for ROCm).
    platform = 'cuda' if device_attr == 'gpu' else device_attr
    if 'JAX_PLATFORMS' not in os.environ:
        os.environ['JAX_PLATFORMS'] = platform
    # Avoid JAX's default ~80% GPU prealloc — sweeps frequently
    # share a GPU with the env vectorisation, and a 90% cap leaves
    # headroom without thrashing.
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.9')
    print(
        f'run_sweep: JAX_PLATFORMS={os.environ["JAX_PLATFORMS"]}',
        file=sys.stderr,
    )

    # Imports AFTER JAX_PLATFORMS so jax picks the device up.
    from corroborate_rl.dqn.yaml_sweep import (
        default_dqn_registry, dispatch_sweep, load_sweep,
    )

    sweep = load_sweep(cfg_path, reg=default_dqn_registry())
    print(
        f'run_sweep: loaded {sweep.name!r} → out_dir={sweep.out_dir} '
        f'({len(sweep.intervention_templates)} interventions × '
        f'{len(sweep.envs)} envs, env_binding={sweep.env_binding})',
        file=sys.stderr,
    )
    runs_path, traces_path = dispatch_sweep(sweep)
    print(
        f'run_sweep: done → {runs_path}, {traces_path}',
        file=sys.stderr,
    )


def _dry_run(cfg_path: Path) -> None:
    """Validate + print without dispatching. Imports the YAML
    machinery only — no JAX, no env initialisation."""
    from corroborate_rl.dqn.yaml_sweep import (
        default_dqn_registry, expand_sweep, load_sweep,
    )
    sweep = load_sweep(cfg_path, reg=default_dqn_registry())
    print(
        f'sweep: {sweep.name!r}\n'
        f'  out_dir       : {sweep.out_dir}\n'
        f'  archive_remote: {sweep.archive_remote}\n'
        f'  env_binding   : {sweep.env_binding}\n'
        f'  envs          : {len(sweep.envs)}'
    )
    for ec in sweep.envs:
        print(f'    - {ec.env_name} (n_seeds={ec.n_seeds}, chunk={ec.chunk_size})')
    configs = expand_sweep(sweep, reg=default_dqn_registry())
    print(f'  interventions (expanded): {len(configs)}')
    for cfg in configs:
        arm_keys = cfg.do_effect.arm_keys()
        extras = (
            f' +measurables={list(cfg.required_measurables)}'
            if cfg.required_measurables else ''
        )
        print(f'    - {cfg.name} ({len(arm_keys)} arms){extras}')
    print('dry-run: OK')


if __name__ == '__main__':
    main()
