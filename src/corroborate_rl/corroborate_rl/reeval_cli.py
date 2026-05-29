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

This module is substrate code; the framework knows nothing about
re-eval. The reeval transform is corpus-shaped, not sweep-shaped
(no Hypothesis, no arms grid), so it gets its own console entry
rather than a `corroborate sweep run` hook."""
from __future__ import annotations

import argparse
from pathlib import Path

from collections.abc import Mapping

from corroborate._internals.argparse import to_mapping
from corroborate._internals.narrow import require_int, require_str
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
        '--device', choices=['cpu', 'gpu'], default='cpu',
        help='JAX platform. CPU default; GPU for MinAtar CNN evals.',
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
    parser = _build_parser()
    args = parser.parse_args(argv)
    args_map = to_mapping(args)
    device = _require_device(args_map)
    # Stamp JAX env BEFORE the heavy import (mirrors dqn_sweep's
    # pre_import_setup discipline).
    set_jax_env(device)

    # Lazy import — JAX latches the backend here, after the env stamp.
    from corroborate_rl.dqn.reeval import EvalKeying, reeval_corpus

    keying_raw = require_str(args_map, 'eval_keying')
    keying: EvalKeying = 'paired' if keying_raw == 'paired' else 'original'

    corpus_dir = Path(require_str(args_map, 'corpus_dir'))
    out_dir = Path(require_str(args_map, 'out_dir'))
    n_episodes = require_int(args_map, 'n_episodes')
    eval_seed_base = require_int(args_map, 'eval_seed_base')
    subdir = require_str(args_map, 'q_checkpoints_subdir')

    out = reeval_corpus(
        corpus_dir,
        n_episodes=n_episodes,
        out_dir=out_dir,
        eval_seed_base=eval_seed_base,
        eval_keying=keying,
        q_checkpoints_subdir=subdir,
    )
    print(f'reeval: wrote {out} (n_episodes={n_episodes}, keying={keying})')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


__all__ = ['main']
