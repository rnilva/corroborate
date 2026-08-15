"""CLI for `reeval_corpus` — re-evaluate a `*_ckpt` corpus's saved
per-burst Q-checkpoints at a chosen `n_episodes`, producing a NEW
corpus that ingests through the canonical pipeline.

Top-level under `corroborate_rl` (NOT under `corroborate_rl.dqn`)
so importing this module does NOT pull JAX before the device
env-stamp runs — same lazy-import discipline as `dqn_sweep.py`.
`main()` stamps `JAX_PLATFORMS` / XLA flags via `set_jax_env`
BEFORE the heavy `corroborate_rl.dqn.reeval` import latches the
backend.

Console entry: `corroborate-rl-reeval` (registered in
`pyproject.toml [project.scripts]`). Also runnable as
`python -m corroborate_rl.reeval_cli`.

Example — Breakout γ=0.99 n_eps1 → n_eps20:

    set -a && . .env && set +a   # AWS creds if restoring first
    uv run --package corroborate_rl corroborate-rl-reeval \\
        experiments/data/breakout_g099_canonical_n_eps1_ckpt \\
        --n-episodes 20 \\
        --out-dir experiments/data/breakout_g099_canonical_reeval_n20_ckpt \\
        --device gpu

This module is implementation code; the framework knows nothing about
re-eval. The reeval transform is corpus-shaped, not sweep-shaped
(no Hypothesis, no arms grid), so it gets its own console entry
rather than a `corroborate sweep run` hook."""
from __future__ import annotations

import argparse
from pathlib import Path

from collections.abc import Mapping

from corroborate._internals.argparse import to_mapping
from corroborate._internals.narrow import (
    require_bool,
    require_float,
    require_int,
    require_str,
)
from corroborate_rl.dqn_sweep import Device, set_jax_env


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='corroborate-rl-reeval',
        description=(
            're-evaluate a *_ckpt corpus at a chosen n_episodes from '
            'saved per-burst Q-checkpoints (no retraining)'
        ),
    )
    _ = parser.add_argument(
        'corpus_dir', type=str,
        help='Source corpus dir (runs.parquet + traces.parquet + '
             'q_checkpoints/ must be local — restore from cloud first '
             'if evicted).',
    )
    _ = parser.add_argument(
        '--n-episodes', type=int, required=True,
        help='Greedy eval episodes per (cell, burst) to re-roll.',
    )
    _ = parser.add_argument(
        '--out-dir', type=str, required=True,
        help='Destination for the new corpus (runs.parquet + '
             'traces.parquet + _remote.json q-checkpoint reference).',
    )
    _ = parser.add_argument(
        '--eval-seed-base', type=int, default=0,
        help='Base seed for the paired eval RNG '
             '(fold_in(PRNGKey(base), seed*stride + burst)). '
             'Default 0.',
    )
    _ = parser.add_argument(
        '--eval-keying', choices=['paired', 'original'], default='paired',
        help="'paired' (default): per-(seed, burst) eval key SHARED "
             'across arms (low-variance V−D). \'original\': reproduce '
             'the training-time per-seed scheme (round-trip / debug).',
    )
    _ = parser.add_argument(
        '--q-checkpoints-subdir', type=str, default='q_checkpoints',
        help="Checkpoint sidecar subdir under the corpus "
             "(default 'q_checkpoints').",
    )
    _ = parser.add_argument(
        '--eval-cache-dir', type=str, default=None,
        help='Streaming path: persist each cell\'s re-evaled eval '
             'arrays to <dir>/cell{NNN}.npz as the cell completes, and '
             'skip cached cells on a re-run. Makes a long (~75-min) '
             'streaming re-eval RESUMABLE (a kill loses only the '
             'in-flight cell, not the whole eval) and LOW-RAM at the '
             'trace write (streams arrays from the npz cache instead of '
             'holding the corpus-wide ~9 GB in memory). Recommended for '
             'the snake 3M re-eval.',
    )
    _ = parser.add_argument(
        '--stream-checkpoints', action='store_true',
        help='Disk-bounded path: restore each per-cell checkpoint '
             'bundle from cloud one at a time, re-eval the runs it '
             'covers, then purge it before the next — peak ckpt disk = '
             'one bundle. For bundle-layout corpora whose full '
             'q_checkpoints/ tree (e.g. snake 3M ~11 GB) does not fit '
             'locally. Requires a _remote.json manifest on the source '
             'corpus. Default (off) assumes all checkpoints are already '
             'local (small per-file corpora).',
    )
    _ = parser.add_argument(
        '--match-tol', type=float, default=1e-3,
        help='L∞ tolerance for the per-burst predicted-Q arm-fingerprint '
             'match (streaming path only). Default 1e-3 suits ≤1M-step '
             'corpora; relax (e.g. 5e-2) for deep / high-Q corpora '
             '(snake 3M) where float32 drift between the original eval '
             'and the recompute exceeds 1e-3 while the wrong arm sits '
             '~1000× further away.',
    )
    _ = parser.add_argument(
        '--device', choices=['cpu', 'gpu'], default='cpu',
        help='JAX platform. CPU default; GPU for MinAtar CNN evals.',
    )
    _ = parser.add_argument(
        '--no-deterministic', action='store_true',
        help='Skip the --xla_gpu_deterministic_ops=true XLA flag. A '
             're-eval is a MEASUREMENT (fixed eval key → deterministic '
             'policy + rollout); only FP reduction ORDER varies (~1e-6), '
             'negligible for eval statistics. Determinism BLOCKS CUDA '
             'Graph capture on Jumanji-class envs (scatter-heavy '
             'obs_extract), making them 10-30× slower (see set_jax_env '
             "docs). Pass this for Snake / PacMan-jumanji re-evals; the "
             'MinAtar-scale envs are unaffected either way.',
    )
    _ = parser.add_argument(
        '--host-checkpoints-on-cpu', action='store_true',
        help='Streaming path: hold each decoded checkpoint bundle on '
             'the CPU device and migrate only the per-(seed, burst) '
             'param SLICE to the GPU. ONLY for bundles too large to fit '
             'GPU memory even with preallocation — it makes the '
             'arm-fingerprint MATCH ~20× slower (per-probe CPU→GPU '
             'transfers). The default streaming+GPU path keeps the '
             'bundle on GPU under a preallocated pool (PREALLOCATE=true, '
             'which avoids the on-demand fragmentation that OOMs the '
             '~2.7 GB snake bundle) — fast match + fast slicing. Reach '
             'for this flag only when a single bundle exceeds the GPU.',
    )
    return parser


def _require_device(m: Mapping[str, object]) -> Device:
    raw = require_str(m, 'device')
    if raw == 'gpu':
        return 'gpu'
    if raw == 'cpu':
        return 'cpu'
    raise ValueError(f"device must be 'cpu'|'gpu'; got {raw!r}")


def main(argv: list[str] | None = None) -> int:
    """Parse args, stamp the JAX device env BEFORE importing the heavy
    `reeval` module, then run `reeval_corpus`. Returns 0 on success.

    Argparse `choices=` validates `--device` / `--eval-keying` at
    parse time; the narrows below are the type-level dispatch (raw
    Namespace attribute access would erase to `Any`)."""
    import os

    parser = _build_parser()
    args = parser.parse_args(argv)
    args_map = to_mapping(args)
    device = _require_device(args_map)
    stream = require_bool(args_map, 'stream_checkpoints')
    no_deterministic = require_bool(args_map, 'no_deterministic')
    host_checkpoints_on_cpu = require_bool(args_map, 'host_checkpoints_on_cpu')
    # Stamp JAX env BEFORE the heavy import (mirrors dqn_sweep's
    # pre_import_setup discipline). `--no-deterministic` drops the
    # CUDA-Graph-blocking deterministic XLA flag (10-30× faster on
    # Jumanji-class envs; eval is a fixed-key measurement so FP-order
    # nondeterminism is immaterial).
    set_jax_env(device, deterministic=not no_deterministic)

    if stream and device == 'gpu':
        if host_checkpoints_on_cpu:
            # Bundle too large for GPU even preallocated: host it on the
            # CPU device, migrate per-burst slices to GPU. Needs the CPU
            # backend ALONGSIDE CUDA (set_jax_env set JAX_PLATFORMS=cuda;
            # append `,cpu` so CUDA stays the default compute device).
            os.environ['JAX_PLATFORMS'] = 'cuda,cpu'
        else:
            # Default: keep the bundle on GPU but PREALLOCATE the memory
            # pool. `set_jax_env` setdefault'd PREALLOCATE=false, whose
            # on-demand allocator FRAGMENTS and OOMs decoding the ~2.7 GB
            # snake bundle even though it fits in 16 GB. A preallocated
            # pool fits the bundle (≈12 GB resident) AND keeps the
            # arm-fingerprint match fast (GPU-resident slicing, no
            # per-probe host↔device transfers). Override the setdefault.
            os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'

    # Lazy import — JAX latches the backend here, after the env stamp.
    from corroborate_rl.dqn.reeval import (
        CloudCheckpointRestorer,
        EvalKeying,
        reeval_corpus,
        reeval_corpus_streaming,
    )

    keying_raw = require_str(args_map, 'eval_keying')
    keying: EvalKeying = 'paired' if keying_raw == 'paired' else 'original'

    corpus_dir = Path(require_str(args_map, 'corpus_dir'))
    out_dir = Path(require_str(args_map, 'out_dir'))
    n_episodes = require_int(args_map, 'n_episodes')
    eval_seed_base = require_int(args_map, 'eval_seed_base')
    subdir = require_str(args_map, 'q_checkpoints_subdir')
    match_tol = require_float(args_map, 'match_tol')
    eval_cache_raw = args_map.get('eval_cache_dir')
    eval_cache_dir = (
        Path(eval_cache_raw) if isinstance(eval_cache_raw, str) else None
    )

    if stream:
        import sys

        def _progress(msg: str) -> None:
            # Flush so the operator sees per-bundle motion in real time
            # on a long (multi-GB, multi-cell) streaming re-eval —
            # Python block-buffers stdout when not a TTY.
            _ = sys.stderr.write(f'reeval: {msg}\n')
            sys.stderr.flush()

        restorer = CloudCheckpointRestorer.from_corpus(
            corpus_dir, q_checkpoints_subdir=subdir,
        )
        out = reeval_corpus_streaming(
            corpus_dir,
            n_episodes=n_episodes,
            out_dir=out_dir,
            restorer=restorer,
            eval_seed_base=eval_seed_base,
            eval_keying=keying,
            match_tol=match_tol,
            host_checkpoints_on_cpu=host_checkpoints_on_cpu,
            eval_cache_dir=eval_cache_dir,
            progress=_progress,
        )
    else:
        out = reeval_corpus(
            corpus_dir,
            n_episodes=n_episodes,
            out_dir=out_dir,
            eval_seed_base=eval_seed_base,
            eval_keying=keying,
            q_checkpoints_subdir=subdir,
        )
    print(
        f'reeval: wrote {out} (n_episodes={n_episodes}, keying={keying}, '
        f'stream={stream})',
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


__all__ = ['main']
