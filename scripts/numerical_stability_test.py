"""Tests the "DDQN's clip is a numerical-stability filter" hypothesis.

Run Asterix-MinAtar 1M seed=0 SIX times — 3× vanilla, 3× DDQN —
WITHOUT `--xla_gpu_deterministic_ops=true` (so GPU
non-determinism produces ~1e-7 per-matmul thread-scheduling
jitter that compounds chaotically).

If the hypothesis holds:
  - σ_vanilla(eval_best across reruns) >> σ_DDQN(eval_best across reruns)
  - DDQN's bootstrap clip `min(target_q_max_a', target_q_at_online_argmax)`
    filters per-step numerical noise when target_q_at_online_argmax is
    the smaller term (the typical DDQN regime, by construction).
  - Vanilla DQN's `max_a' target_q(s', a')` integrates the noise →
    chaos amplifies it into trajectory-level divergence.

Also times each run for the perf-cost-of-determinism comparison.
"""
from __future__ import annotations

import os
# Explicitly DO NOT set deterministic_ops here — we want the
# non-deterministic baseline.
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import sys
import time
from functools import partial
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'src/corroborate_rl'))

import jax  # noqa: F401  pin JAX flags before later imports
import numpy as np

from corroborate_rl.dqn.claims import (
    Replay, periodic_copy, squared_error,
)
from corroborate_rl.dqn.claims.action_select import (
    epsilon_greedy, linear_epsilon,
)
from corroborate_rl.dqn.claims.bootstrap import (
    bootstrap, double_greedify,
)
from corroborate_rl.dqn.claims.optimizer import adam, warmed_update
from corroborate_rl.dqn.claims.q_network import CNN
from corroborate_rl.dqn.dqn import dqn
from corroborate_rl.env_catalogue import (
    ENV_REGISTRY, image_downsample_hash, get, make_env,
)


def build_partial(*, ddqn: bool):
    spec = get('Asterix-MinAtar')
    env, env_params = make_env(spec)
    bootstrap_partial = (
        partial(bootstrap, greedification=double_greedify)
        if ddqn else partial(bootstrap)
    )
    sh, _ = image_downsample_hash(
        spec.observation_shape, pool_size=3, n_buckets_per_dim=2,
        channel_agg='sum', feature_low=0.0, feature_high=2.0,
    )
    return partial(
        dqn,
        env_name='Asterix-MinAtar',
        env=env, env_params=env_params,
        n_actions=spec.n_actions, obs_shape=spec.observation_shape,
        total_steps=1_000_000, eval_every=20_000,
        n_episodes=5, gamma=0.99, sync_period=1000, n_step=1,
        eval_episode_cap=spec.eval_episode_cap or 100_000,
        q_network=CNN(
            obs_shape=spec.observation_shape, channels=(16,),
            hidden=(128,), kernel_size=3,
        ),
        replay=Replay(capacity=100_000, batch_size=32),
        optimizer=partial(
            warmed_update,
            inner=partial(adam, lr=1e-4),
            warmup_steps=100,
        ),
        action_select=partial(
            epsilon_greedy,
            schedule=partial(
                linear_epsilon, eps_init=1.0, eps_final=0.05,
                anneal_steps=100_000,
            ),
        ),
        bootstrap=bootstrap_partial,
        loss_fn=squared_error,
        target_sync=periodic_copy,
        state_hash=sh,
    )


def run_trial(label: str, ddqn: bool) -> dict[str, float]:
    print(f'\n--- {label} ---', flush=True)
    t0 = time.time()
    out = build_partial(ddqn=ddqn)(seed=0)
    wall = time.time() - t0
    mc = np.asarray(out['mc_return'])
    print(f'  wall: {wall:.1f}s', flush=True)
    return {
        'eval_best_burst_mean': float(mc.mean(axis=1).max()),
        'eval_final_mean': float(mc[-1].mean()),
        'wall_seconds': wall,
    }


def main() -> None:
    print('XLA_FLAGS =', os.environ.get('XLA_FLAGS', '(unset)'), flush=True)
    vanilla = [run_trial(f'vanilla #{i + 1}', ddqn=False) for i in range(3)]
    ddqn = [run_trial(f'DDQN #{i + 1}', ddqn=True) for i in range(3)]

    print('\n=== Results ===')
    print(f'{"trial":<18s} {"eval_best":>14s} {"eval_final":>14s} {"wall_s":>10s}')
    for i, r in enumerate(vanilla):
        print(f'vanilla #{i + 1:<10d} {r["eval_best_burst_mean"]:>14.6f} '
              f'{r["eval_final_mean"]:>14.6f} {r["wall_seconds"]:>10.1f}')
    for i, r in enumerate(ddqn):
        print(f'DDQN #{i + 1:<13d} {r["eval_best_burst_mean"]:>14.6f} '
              f'{r["eval_final_mean"]:>14.6f} {r["wall_seconds"]:>10.1f}')

    print('\n=== Cross-run statistics (same seed, different GPU realizations) ===')
    for arm, rs in [('vanilla', vanilla), ('DDQN', ddqn)]:
        eb = np.array([r['eval_best_burst_mean'] for r in rs])
        ef = np.array([r['eval_final_mean'] for r in rs])
        print(f'{arm}:')
        print(f'  eval_best:  mean={eb.mean():+.4f} std={eb.std(ddof=1):+.6f} range=[{eb.min():+.4f}, {eb.max():+.4f}]')
        print(f'  eval_final: mean={ef.mean():+.4f} std={ef.std(ddof=1):+.6f} range=[{ef.min():+.4f}, {ef.max():+.4f}]')


if __name__ == '__main__':
    main()
