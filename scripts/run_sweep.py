"""Run a YAML-configured DQN sweep end-to-end.

Usage:
    set -a && . .env && set +a   # AWS creds for archive
    PYTHONPATH=. uv run python scripts/run_sweep.py \\
        experiments/configs/<sweep>.yaml

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
    parser.add_argument(
        'config', type=Path,
        help='YAML sweep config (e.g. '
             'experiments/configs/capacity_sweep_fourrooms.yaml)',
    )
    parser.add_argument(
        '--device', choices=['cpu', 'gpu'], default='cpu',
        help='JAX platform. CPU is the safe default for small envs '
             '(FourRooms-class); flip to GPU if it exists + '
             'matters.',
    )
    args = parser.parse_args(argv)

    # JAX prefers explicit backend names: 'cpu' / 'cuda' / 'rocm' / 'tpu'.
    # The CLI surface is cpu/gpu for ergonomics; map gpu→cuda since that's
    # what's installed on this machine. Caller can override via
    # JAX_PLATFORMS env var (e.g. for ROCm).
    platform = 'cuda' if args.device == 'gpu' else args.device
    if 'JAX_PLATFORMS' not in os.environ:
        os.environ['JAX_PLATFORMS'] = platform
    # Avoid JAX's default ~80% GPU prealloc — sweeps frequently
    # share a GPU with the env vectorisation, and a 90% cap leaves
    # headroom without thrashing.
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.9')
    print(f'run_sweep: JAX_PLATFORMS={os.environ["JAX_PLATFORMS"]}', file=sys.stderr)

    # Imports AFTER JAX_PLATFORMS so jax picks the device up.
    from corroborate_rl.dqn.yaml_sweep import (
        default_dqn_registry, dispatch_sweep, load_sweep,
    )

    cfg_path: Path = args.config
    if not cfg_path.exists():
        raise SystemExit(f'config not found: {cfg_path}')
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


if __name__ == '__main__':
    main()
