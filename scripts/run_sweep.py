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
    _ = parser.add_argument(
        '--profile', dest='profile', default=None,
        help='AWS profile name for the cloud preflight when the '
             'sweep config has `archive_remote` set. Falls back to '
             'AWS_PROFILE env var, then the default credential '
             'chain (env vars → ~/.aws/credentials → IAM role).',
    )
    _ = parser.add_argument(
        '--skip-preflight', action='store_true',
        help='Skip the upfront cloud-auth check before the sweep '
             'runs. Use when iterating against a known-good profile '
             'and the ~100-300ms head_bucket round-trip becomes '
             'friction. Off by default — preflight protects against '
             'wasting hours of compute then failing at the archive '
             'step.',
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
    # XLA deterministic matmul / scatter ops. Without this, GPU
    # thread-scheduling jitter introduces per-matmul ~1e-7 noise
    # that compounds chaotically over 1M training steps on Q-
    # explosion-prone vanilla DQN — manifests as ~8σ cross-realization
    # population-mean drift on Asterix / SI canonical-verify
    # (memory `findings_substrate_realization_variance`). Measured
    # negligible perf overhead at MinAtar scale (CNN[16]/FC[128]
    # 1M-step Asterix: 271s deterministic ≈ 273s non-deterministic).
    # Set as a `setdefault` so callers can override with explicit
    # `XLA_FLAGS=` if they need non-deterministic kernels for some
    # specific reason.
    if 'XLA_FLAGS' not in os.environ:
        os.environ['XLA_FLAGS'] = '--xla_gpu_deterministic_ops=true'
    elif '--xla_gpu_deterministic_ops' not in os.environ['XLA_FLAGS']:
        os.environ['XLA_FLAGS'] = (
            os.environ['XLA_FLAGS'].rstrip()
            + ' --xla_gpu_deterministic_ops=true'
        )
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

    # Cloud preflight + AWS_PROFILE export.
    # Preflight is gated on whether the sweep config actually
    # uploads (`archive_remote` set) AND the user hasn't opted out
    # via --skip-preflight. The AWS_PROFILE export is INDEPENDENT
    # of preflight — if --profile is passed, downstream cloud ops
    # (cloud.archive at sweep end, is_archived during resume) need
    # it on the env, regardless of whether preflight ran.
    profile_attr_raw: object = args.profile  # pyright: ignore[reportAny]
    skip_attr_raw: object = args.skip_preflight  # pyright: ignore[reportAny]
    profile_arg: str | None = (
        profile_attr_raw if isinstance(profile_attr_raw, str) else None
    )
    skip_preflight: bool = bool(skip_attr_raw) if isinstance(
        skip_attr_raw, bool,
    ) else False
    if profile_arg is not None:
        os.environ['AWS_PROFILE'] = profile_arg
    if sweep.archive_remote is not None and not skip_preflight:
        from corroborate._internals.cloud_auth import (
            CloudAuthError, preflight,
        )
        try:
            preflight(sweep.archive_remote, profile=profile_arg)
        except CloudAuthError as e:
            print(
                f'run_sweep: cloud preflight FAILED — aborting before '
                f'sweep loop kicks off.\n  {e}',
                file=sys.stderr,
            )
            raise SystemExit(1) from e

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
