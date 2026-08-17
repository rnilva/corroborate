# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "stable-baselines3>=2.3",
#   "gymnasium>=0.29",
# ]
# ///
"""Produce training runs corroborate can load — ordinary SB3.

This script is the *user's* side of the boundary: plain
stable-baselines3, **zero corroborate imports** and zero
corroborate-specific files. It trains DQN on CartPole-v1 at two
gamma values (0.99 vs 0.80) over paired seeds, evaluates each run
at fixed checkpoints with fixed evaluation seeds, and writes what
any careful experimenter records anyway:

    runs/
      runs.jsonl           one line per run: id + which config ran it
      configs/<run>.json   the resolved configuration actually used
      evaluations.jsonl    one record per (run, checkpoint, eval seed)
      provenance.json      who produced this, and how (optional)

Run (uv resolves the SB3 deps from the header, ~1 min/run on CPU):

    uv run examples/sb3_demo/train.py --seeds 3 --steps 25000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

GAMMAS = (0.99, 0.80)
ENV_ID = 'CartPole-v1'
EVAL_SEEDS = (1000, 1001, 1002, 1003, 1004)
OUT = Path(__file__).parent / 'runs'


def evaluate(model: DQN, eval_seed: int) -> float:
    """One greedy episode return under a fixed evaluation seed."""
    env = gym.make(ENV_ID)
    obs, _ = env.reset(seed=eval_seed)
    total, done = 0.0, False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total += float(reward)
        done = terminated or truncated
    env.close()
    return total


class CheckpointEval(BaseCallback):
    """At each checkpoint timestep, record one evaluation episode
    per evaluation seed — one JSONL record each."""

    def __init__(self, run_id: str, checkpoints: list[int], sink: list[dict]):
        super().__init__()
        self.run_id, self.checkpoints, self.sink = run_id, checkpoints, sink
        self._next = 0

    def _on_step(self) -> bool:
        if self._next < len(self.checkpoints) \
                and self.num_timesteps >= self.checkpoints[self._next]:
            cp = self.checkpoints[self._next]
            for es in EVAL_SEEDS:
                self.sink.append({
                    'run_id': self.run_id, 'checkpoint': cp,
                    'eval_seed': es,
                    'return': evaluate(self.model, es),
                })
            self._next += 1
        return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--steps', type=int, default=25_000)
    args = ap.parse_args()
    checkpoints = [args.steps // 5 * k for k in range(1, 6)]

    (OUT / 'configs').mkdir(parents=True, exist_ok=True)
    runs, evals = [], []

    for gamma in GAMMAS:
        for seed in range(args.seeds):
            run_id = f'gamma{gamma * 100:03.0f}-s{seed}'
            config = {
                'algo': 'DQN', 'env_id': ENV_ID, 'gamma': gamma,
                'seed': seed, 'total_timesteps': args.steps,
                'learning_rate': 1e-3, 'buffer_size': 50_000,
                'learning_starts': 1_000, 'train_freq': 4,
                'target_update_interval': 500,
                'exploration_fraction': 0.2,
            }
            cfg_rel = f'configs/{run_id}.json'
            (OUT / cfg_rel).write_text(json.dumps(config, indent=1))

            print(f'training {run_id} (gamma={gamma}) ...', flush=True)
            model = DQN(
                'MlpPolicy', ENV_ID, verbose=0,
                **{k: config[k] for k in (
                    'gamma', 'seed', 'learning_rate', 'buffer_size',
                    'learning_starts', 'train_freq',
                    'target_update_interval', 'exploration_fraction',
                )},
            )
            model.learn(
                total_timesteps=args.steps,
                callback=CheckpointEval(run_id, checkpoints, evals),
            )
            runs.append({'run_id': run_id, 'config_path': cfg_rel})

    (OUT / 'runs.jsonl').write_text(
        ''.join(json.dumps(r) + '\n' for r in runs))
    (OUT / 'evaluations.jsonl').write_text(
        ''.join(json.dumps(e) + '\n' for e in evals))
    (OUT / 'provenance.json').write_text(json.dumps({
        'producer': 'stable-baselines3 DQN demo producer',
        'command': ' '.join(sys.argv),
    }, indent=1))
    print(f'runs written to {OUT}')


if __name__ == '__main__':
    main()
