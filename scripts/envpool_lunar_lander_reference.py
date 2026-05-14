"""Reference rollout stats on envpool's gymnasium-faithful
LunarLander-v2 to validate any JAX port against.

Captures dynamics-level distributions (random policy):
  - episode length distribution
  - episode return distribution
  - termination cause breakdown
  - per-step reward distribution
  - obs / action statistics

These are what a JAX-native port should approximately match. Box2D
trajectory replication is not a goal (impossible without Box2D
verbatim); statistical fidelity of the distributions IS.

Output: prints summary table; writes per-episode records as JSON
to `experiments/data/lunar_lander_reference.json` for the JAX-port
validation suite to consume.
"""
from __future__ import annotations

import json
from pathlib import Path

import envpool
import numpy as np


def main() -> None:
    NUM_ENVS = 32
    MAX_EPISODES = 200
    MAX_STEPS_PER_EPISODE = 1000
    SEED = 42
    rng = np.random.default_rng(SEED)

    env = envpool.make(
        'LunarLander-v2', env_type='gymnasium',
        num_envs=NUM_ENVS, seed=SEED,
    )
    print(f'LunarLander-v2 (envpool/gymnasium-faithful, num_envs={NUM_ENVS})')
    print(f'  action_space: {env.action_space}')
    print(f'  obs_space: {env.observation_space}')

    n_actions = env.action_space.n

    ep_returns: list[float] = []
    ep_lengths: list[int] = []
    ep_terminations: list[str] = []
    all_step_rewards: list[float] = []
    obs_mins = np.full(8, np.inf)
    obs_maxs = np.full(8, -np.inf)

    # Track per-env state
    env_returns = np.zeros(NUM_ENVS)
    env_steps = np.zeros(NUM_ENVS, dtype=np.int32)

    obs, _info = env.reset()
    obs_mins = np.minimum(obs_mins, obs.min(axis=0))
    obs_maxs = np.maximum(obs_maxs, obs.max(axis=0))

    step = 0
    while len(ep_returns) < MAX_EPISODES and step < MAX_STEPS_PER_EPISODE * 4:
        actions = rng.integers(0, n_actions, size=NUM_ENVS).astype(np.int32)
        obs, reward, terminated, truncated, info = env.step(actions)
        obs_mins = np.minimum(obs_mins, obs.min(axis=0))
        obs_maxs = np.maximum(obs_maxs, obs.max(axis=0))
        env_returns += reward
        env_steps += 1
        all_step_rewards.extend(reward.tolist())
        done = terminated | truncated
        if done.any():
            for i, d in enumerate(done):
                if not d:
                    continue
                ep_returns.append(float(env_returns[i]))
                ep_lengths.append(int(env_steps[i]))
                # heuristic termination classification — envpool's
                # info dict may or may not have a discriminator,
                # so we infer from return + termination flag.
                if terminated[i]:
                    if reward[i] >= 90.0:
                        ep_terminations.append('success')
                    elif reward[i] <= -90.0:
                        ep_terminations.append('crash')
                    else:
                        ep_terminations.append('other_terminal')
                else:
                    ep_terminations.append('truncated')
                env_returns[i] = 0.0
                env_steps[i] = 0
        step += 1

    arr_returns = np.asarray(ep_returns)
    arr_lengths = np.asarray(ep_lengths)
    arr_step_rewards = np.asarray(all_step_rewards)
    termination_counts: dict[str, int] = {}
    for t in ep_terminations:
        termination_counts[t] = termination_counts.get(t, 0) + 1

    print(f'\n--- Random-policy rollout summary ---')
    print(f'  n_episodes:          {len(arr_returns)}')
    print(f'  episode return       mean={arr_returns.mean():+8.2f} '
          f'sd={arr_returns.std():.2f} min={arr_returns.min():+8.2f} '
          f'max={arr_returns.max():+8.2f}')
    print(f'  episode length       mean={arr_lengths.mean():6.1f} '
          f'sd={arr_lengths.std():.1f} min={arr_lengths.min()} '
          f'max={arr_lengths.max()}')
    print(f'  per-step reward      mean={arr_step_rewards.mean():+.4f} '
          f'sd={arr_step_rewards.std():.4f}')
    print(f'  obs range (per dim):')
    obs_names = ('x', 'y', 'vx', 'vy', 'angle', 'ang_vel',
                 'leg1', 'leg2')
    for i, n in enumerate(obs_names):
        print(f'    {n:<8} [{obs_mins[i]:+.3f}, {obs_maxs[i]:+.3f}]')
    print(f'  termination breakdown:')
    for cause, count in sorted(termination_counts.items()):
        pct = 100.0 * count / len(ep_terminations)
        print(f'    {cause:<16} {count:>4}  ({pct:.1f}%)')

    out_path = Path('experiments/data/lunar_lander_reference.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'env': 'LunarLander-v2 (envpool/gymnasium-faithful)',
        'seed': SEED,
        'num_envs': NUM_ENVS,
        'n_episodes': len(arr_returns),
        'returns': {
            'mean': float(arr_returns.mean()),
            'sd': float(arr_returns.std()),
            'min': float(arr_returns.min()),
            'max': float(arr_returns.max()),
            'all': arr_returns.tolist(),
        },
        'lengths': {
            'mean': float(arr_lengths.mean()),
            'sd': float(arr_lengths.std()),
            'min': int(arr_lengths.min()),
            'max': int(arr_lengths.max()),
            'all': arr_lengths.tolist(),
        },
        'per_step_reward': {
            'mean': float(arr_step_rewards.mean()),
            'sd': float(arr_step_rewards.std()),
        },
        'obs_range': {
            n: [float(obs_mins[i]), float(obs_maxs[i])]
            for i, n in enumerate(obs_names)
        },
        'termination_counts': termination_counts,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f'\nwrote {out_path}')


if __name__ == '__main__':
    main()
