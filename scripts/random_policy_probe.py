"""Random-policy probe — env-only characteristic measurables.

Run a uniform-random policy for `n_steps` steps on each env in our
panel. Compute env-only characteristics that don't require running
DDQN OR vanilla DQN:

  - random_return_mean  : mean undiscounted episode return (random)
  - random_return_sigma : SD of episode returns across rollouts
  - random_return_cv    : coefficient of variation = sigma/|mean|
  - episode_len_mean    : mean episode length under random policy
  - episode_len_sigma   : SD of episode length
  - reward_per_step     : mean per-step reward (density proxy)
  - reward_nonzero_frac : fraction of steps with reward != 0 (sparsity)

These are candidates for predicting v_jens (substantive premise
activation under vanilla DQN) without running anything beyond
random exploration.

Outputs `experiments/data/random_policy_probe.parquet` keyed by env_name.
"""
from __future__ import annotations

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
from corroborate_rl.env_catalogue import ENV_REGISTRY, make_env

# Force CPU for reproducibility (matches user's CPU directive)
jax.config.update('jax_platforms', 'cpu')

ENVS_TO_PROBE = [
    ('Acrobot-v1', 'gymnax'),
    ('MountainCar-v0', 'gymnax'),
    ('FourRooms-misc', 'gymnax'),
    ('MetaMaze-misc', 'gymnax'),
    ('CartPole-v1', 'gymnax'),
    ('Asterix-MinAtar', 'gymnax'),
    ('Breakout-MinAtar', 'gymnax'),
    ('Freeway-MinAtar', 'gymnax'),
    ('SpaceInvaders-MinAtar', 'gymnax'),
    ('PacMan-jumanji', 'jumanji'),
]

N_STEPS_PER_ENV = 10_000  # ~10s per env on CPU
N_EPISODES_TARGET = 30  # sample at least this many episodes per env


def probe_env(env_name: str, backend: str, n_steps: int) -> dict[str, float]:
    """Run a uniform-random policy for `n_steps` steps; collect
    episode returns, episode lengths, per-step rewards."""
    if backend == 'jumanji':
        # Trigger jumanji registration (registers PacMan-jumanji etc.)
        import corroborate_rl.jumanji_envs as _  # noqa: F401
    spec = ENV_REGISTRY[env_name]
    env, env_params = make_env(spec)
    n_actions = spec.n_actions

    rng = jax.random.PRNGKey(0)
    rng, reset_key = jax.random.split(rng)
    obs, state = env.reset(reset_key, env_params)

    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    rewards_all: list[float] = []
    cur_return = 0.0
    cur_len = 0

    @jax.jit
    def step_fn(rng_step, state, action):
        return env.step(rng_step, state, action, env_params)

    for step in range(n_steps):
        rng, action_key, step_key = jax.random.split(rng, 3)
        action = jax.random.randint(action_key, (), 0, n_actions)
        obs, state, reward, done, _ = step_fn(step_key, state, action)
        r = float(reward)
        rewards_all.append(r)
        cur_return += r
        cur_len += 1
        if bool(done):
            episode_returns.append(cur_return)
            episode_lengths.append(cur_len)
            cur_return = 0.0
            cur_len = 0
            rng, reset_key = jax.random.split(rng)
            obs, state = env.reset(reset_key, env_params)

    rewards_arr = np.asarray(rewards_all)
    eps_returns = np.asarray(episode_returns) if episode_returns else np.array([cur_return])
    eps_lengths = np.asarray(episode_lengths) if episode_lengths else np.array([cur_len])
    return dict(
        env_name=env_name,
        n_episodes=int(len(eps_returns)),
        random_return_mean=float(eps_returns.mean()),
        random_return_sigma=float(eps_returns.std(ddof=1)) if len(eps_returns) > 1 else 0.0,
        random_return_cv=float(eps_returns.std(ddof=1) / max(abs(eps_returns.mean()), 1e-6))
                          if len(eps_returns) > 1 else 0.0,
        episode_len_mean=float(eps_lengths.mean()),
        episode_len_sigma=float(eps_lengths.std(ddof=1)) if len(eps_lengths) > 1 else 0.0,
        reward_per_step=float(rewards_arr.mean()),
        # Mean-absolute-reward per step. Combines density (fraction
        # of steps with reward != 0) and per-event magnitude into a
        # single env-only feature. Best single predictor of v_jens
        # in the n=10 panel: Pearson r=+0.88, p=0.001 vs density's
        # r=+0.74. See `findings_scope_density.md`.
        reward_per_step_abs=float(np.abs(rewards_arr).mean()),
        reward_nonzero_frac=float((np.abs(rewards_arr) > 1e-9).mean()),
        reward_max=float(rewards_arr.max()),
        reward_min=float(rewards_arr.min()),
    )


def main() -> None:
    rows: list[dict[str, float]] = []
    for env_name, backend in ENVS_TO_PROBE:
        t0 = time.time()
        try:
            row = probe_env(env_name, backend, N_STEPS_PER_ENV)
            elapsed = time.time() - t0
            print(f'{env_name}: n_eps={row["n_episodes"]}, '
                  f'<R>={row["random_return_mean"]:+.2f} ± {row["random_return_sigma"]:.2f}, '
                  f'<len>={row["episode_len_mean"]:.0f}, r_per_step={row["reward_per_step"]:+.4f}, '
                  f'r_nonzero={row["reward_nonzero_frac"]:.3f} ({elapsed:.0f}s)')
            rows.append(row)
        except Exception as e:
            print(f'{env_name}: ERROR {e}')

    out = pl.DataFrame(rows)
    out_path = Path('experiments/data/random_policy_probe.parquet')
    out.write_parquet(out_path)
    print(f'\nSaved {out.shape[0]} envs to {out_path}')


if __name__ == '__main__':
    main()
