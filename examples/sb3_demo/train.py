# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "stable-baselines3>=2.3",
#   "gymnasium>=0.29",
# ]
# ///
"""Ordinary SB3 training — nothing else.

This script is the *user's* side of the boundary, and it is
exactly what an SB3 tutorial writes: construct DQN, train with an
`EvalCallback`, save the model. No corroborate imports, no
recording code, no extra files. What lands on disk is SB3's own
output:

    runs/
      <run>/model.zip           model.save()
      <run>/evaluations.npz     EvalCallback's evaluation log

corroborate's side (`analyze.py`) reads those artifacts directly.

Run (uv resolves the SB3 deps from the header, ~1 min/run on CPU):

    uv run examples/sb3_demo/train.py --seeds 3 --steps 25000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

GAMMAS = (0.99, 0.80)
ENV_ID = 'CartPole-v1'
OUT = Path(__file__).parent / 'runs'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--steps', type=int, default=25_000)
    args = ap.parse_args()

    for gamma in GAMMAS:
        for seed in range(args.seeds):
            run_dir = OUT / f'gamma{gamma * 100:03.0f}-s{seed}'
            run_dir.mkdir(parents=True, exist_ok=True)

            eval_env = Monitor(gym.make(ENV_ID))
            eval_env.reset(seed=1_000 + seed)

            print(f'training {run_dir.name} (gamma={gamma}) ...',
                  flush=True)
            model = DQN(
                'MlpPolicy', ENV_ID, verbose=0,
                gamma=gamma, seed=seed,
                learning_rate=1e-3, buffer_size=50_000,
                learning_starts=1_000, train_freq=4,
                target_update_interval=500,
                exploration_fraction=0.2,
            )
            model.learn(
                total_timesteps=args.steps,
                callback=EvalCallback(
                    eval_env,
                    log_path=str(run_dir),
                    eval_freq=args.steps // 5,
                    n_eval_episodes=5,
                    deterministic=True,
                    verbose=0,
                ),
            )
            model.save(run_dir / 'model')
    print(f'runs written to {OUT}')


if __name__ == '__main__':
    main()
