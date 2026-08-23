# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "stable-baselines3>=2.3",
#   "gymnasium>=0.29",
# ]
# ///
"""Ordinary SB3 training, logged through SB3's own Logger.

This script is the *producer's* side of the boundary: construct
DQN, point its Logger at a folder, train with an `EvalCallback`.
The one line of contact with the analysis side is the loop that
records the constructor arguments under ``config/`` — the
producer stating what was configured, next to the records SB3
already logs (``rollout/*``, ``train/*``, ``eval/*``,
``time/*``). What lands on disk is the Logger's own output:

    runs/
      <run>/progress.json     one record per Logger dump()

corroborate's side (`analyze.py`) reads those files directly.

Run (uv resolves the SB3 deps from the header, ~1 min/run on CPU):

    uv run examples/sb3_demo/train.py --seeds 3 --steps 25000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

GAMMAS = (0.99, 0.80)
OUT = Path(__file__).parent / 'runs'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--steps', type=int, default=25_000)
    args = ap.parse_args()

    for gamma in GAMMAS:
        for seed in range(args.seeds):
            run_dir = OUT / f'gamma{gamma * 100:03.0f}-s{seed}'
            config = dict(
                env_id='CartPole-v1', gamma=gamma, seed=seed,
                learning_rate=1e-3, buffer_size=50_000,
                learning_starts=1_000, train_freq=4,
                target_update_interval=500,
                exploration_fraction=0.2,
            )

            print(f'training {run_dir.name} (gamma={gamma}) ...',
                  flush=True)
            env_id = config.pop('env_id')
            model = DQN('MlpPolicy', env_id, verbose=0, **config)
            model.set_logger(configure(str(run_dir), ['json']))
            model.logger.record('config/env_id', env_id)
            for key, value in config.items():
                model.logger.record(f'config/{key}', value)

            eval_env = Monitor(gym.make(env_id))
            eval_env.reset(seed=1_000 + seed)
            model.learn(
                total_timesteps=args.steps,
                callback=EvalCallback(
                    eval_env,
                    eval_freq=args.steps // 5,
                    n_eval_episodes=5,
                    deterministic=True,
                    verbose=0,
                ),
            )
    print(f'runs written to {OUT}')


if __name__ == '__main__':
    main()
