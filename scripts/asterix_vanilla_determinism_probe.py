"""Controlled determinism + state_hash isolation probe.

Three Asterix vanilla seed=0 runs to isolate what drives the
old-vs-new cross-corpus 8σ vanilla shift:

  A:  state_hash = image_downsample_hash (current/new behavior)
  A': same as A, second run                (determinism check)
  B:  state_hash = default_state_hash (constant 0; old behavior)

All under `XLA_FLAGS=--xla_gpu_deterministic_ops=true`. Compare:

  - A vs A' byte-identical? → GPU determinism works
  - A vs B byte-identical?  → state_hash op is NOT the perturbation source
  - A ≠ B?                  → state_hash op IS the perturbation source
                              (confirms investigator's hypothesis)

Direct `dqn()` calls (bypass sweep machinery) so we control
exactly what state_hash is bound to. 1M steps each — ~5-7 min
per run on GPU.
"""
from __future__ import annotations

import os
# Must be set BEFORE jax imports.
os.environ['XLA_FLAGS'] = (
    os.environ.get('XLA_FLAGS', '')
    + ' --xla_gpu_deterministic_ops=true'
).strip()
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import sys
from functools import partial
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'src/corroborate_rl'))

import jax
import jax.numpy as jnp
import numpy as np

from corroborate_rl.dqn.claims import (
    MLP, Replay, bootstrap, double_greedify,
    linear_epsilon, periodic_copy, squared_error,
    uniform_sample,
)
# 4 import action_select / epsilon_greedy / linear_epsilon...
# Build everything from the env_catalogue's MinAtar.
from corroborate_rl.dqn.claims.action_select import linear_epsilon, epsilon_greedy
from corroborate_rl.dqn.claims.bootstrap import bootstrap
from corroborate_rl.dqn.claims.optimizer import adam, warmed_update
from corroborate_rl.dqn.claims.q_network import CNN
from corroborate_rl.dqn.dqn import dqn, default_state_hash
from corroborate_rl.env_catalogue import (
    ENV_REGISTRY, image_downsample_hash, get, make_env,
)


def build_partial(*, state_hash_fn):
    spec = get('Asterix-MinAtar')
    env, env_params = make_env(spec)
    return partial(
        dqn,
        env_name='Asterix-MinAtar',
        env=env, env_params=env_params,
        n_actions=spec.n_actions,
        obs_shape=spec.observation_shape,
        total_steps=1_000_000,
        eval_every=20_000,
        n_episodes=5,
        gamma=0.99,
        sync_period=1000,
        n_step=1,
        eval_episode_cap=spec.eval_episode_cap or 100_000,
        q_network=CNN(
            obs_shape=spec.observation_shape,
            channels=(16,),
            hidden=(128,),
            kernel_size=3,
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
                linear_epsilon,
                eps_init=1.0, eps_final=0.05,
                anneal_steps=100_000,
            ),
        ),
        bootstrap=partial(bootstrap),
        loss_fn=squared_error,
        target_sync=periodic_copy,
        state_hash=state_hash_fn,
    )


def run(label: str, state_hash_fn) -> dict[str, float]:
    print(f'\n=== {label} ===', flush=True)
    print(f'state_hash callable: {state_hash_fn}', flush=True)
    dqn_partial = build_partial(state_hash_fn=state_hash_fn)
    out = dqn_partial(seed=0)
    mc = np.asarray(out['mc_return'])  # (n_bursts, n_episodes)
    q_at_start = np.asarray(out['predicted_q_at_start'])  # (n_bursts, n_episodes)
    return {
        'eval_best_burst_mean': float(mc.mean(axis=1).max()),
        'eval_final_mean': float(mc[-1].mean()),
        'mc_return_burst_0_mean': float(mc[0].mean()),
        'mc_return_burst_-1_mean': float(mc[-1].mean()),
        'q_at_start_burst_0': float(q_at_start[0].mean()),
        'q_at_start_final': float(q_at_start[-1].mean()),
    }


def main() -> None:
    # State_hash callables we'll bind.
    sh_image = image_downsample_hash(
        ENV_REGISTRY['Asterix-MinAtar'].observation_shape,
        pool_size=3, n_buckets_per_dim=2, channel_agg='sum',
        feature_low=0.0, feature_high=2.0,
    )[0]
    sh_default = default_state_hash

    a = run('A: state_hash=image_downsample_hash', sh_image)
    a2 = run("A': same as A (rerun)", sh_image)
    b = run('B: state_hash=default (const 0)', sh_default)

    print('\n=== Comparison ===')
    print(f'{"key":<30s} {"A":>16s} {"A_rerun":>16s} {"B":>16s} {"A=A?":>6s} {"A=B?":>6s}')
    for k in a:
        match_aa = '✓' if a[k] == a2[k] else f'Δ{a[k]-a2[k]:+.4e}'
        match_ab = '✓' if a[k] == b[k] else f'Δ{a[k]-b[k]:+.4e}'
        print(f'{k:<30s} {a[k]:>16.6f} {a2[k]:>16.6f} {b[k]:>16.6f} {match_aa:>6s} {match_ab:>6s}')


if __name__ == '__main__':
    main()
