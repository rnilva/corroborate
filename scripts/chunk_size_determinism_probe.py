"""Rule out chunk_size as the culprit for the MetaMaze probe-vs-postfix
discrepancy.

Hypothesis 1 (the one to falsify): at fixed substrate_commit_sha,
running the same `seed=0` under different vmap batch sizes (chunk
size 1 vs 2) yields meaningfully different final-eval values for
seed 0 — XLA matmul tilings under different batch shapes produce
~1e-7 per-step numerical drift that compounds chaotically over
training.

Hypothesis 2: chunk_size is float-precision innocent at the seed
level; the postfix-vs-probe discrepancy is purely a code-state
(sha) effect.

Test: run MetaMaze γ=0.999 MLP[64,64] short training (100k steps)
at the current sha, comparing seed 0 across three configurations:
  A. chunk_size=1, n_seeds=1 (just seed 0, no vmap batching)
  B. chunk_size=2, n_seeds=2 (vmap over seeds 0, 1 — seed 0 shares
     a vmap with seed 1)
  C. chunk_size=1, n_seeds=1, second run (byte-identity check
     against A — confirms no non-determinism bug)

Report: per-config seed-0 final-state Q values + eval_best.
- A == C (byte): determinism intact within fixed chunk_size.
- A != B: chunk_size IS RNG-effective; we have to treat
  chunk_size as part of the replicate identity, alongside sha.
- A == B: chunk_size is innocent; the postfix-probe gap is
  purely sha + sampling noise.

Runs on CPU for fastest signal — GPU has its own variant of the
same XLA-tiling-by-batch effect, so a CPU positive is conservative
(GPU effect ≥ CPU effect typically).
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'src/corroborate_rl'))

# Force CPU for reproducibility within a configuration.
jax.config.update('jax_platform_name', 'cpu')

from corroborate_rl.dqn.claims import (
    MLP,
    Replay,
    bootstrap,
    linear_epsilon,
    periodic_copy,
    squared_error,
)
from corroborate_rl.dqn.claims.optimizer import adam, warmed_update
from corroborate_rl.dqn.dqn import dqn
from corroborate_rl.env_catalogue import get, make_env


def build_and_run(seed_or_seeds: int | jax.Array) -> dict[str, float | jax.Array]:
    """Build the DQN partial for MetaMaze γ=0.999 MLP[64,64],
    100k steps, run for given seed(s). Returns final state Q sum +
    eval_best for shape audit."""
    spec = get('MetaMaze-misc')
    env, env_params = make_env(spec)
    n_actions = spec.n_actions
    obs_shape = spec.observation_shape

    args = dict(
        env=env, env_params=env_params, n_actions=n_actions,
        obs_shape=obs_shape, total_steps=100_000, eval_every=20_000,
        n_episodes=5, gamma=0.999, sync_period=100, n_step=1,
        eval_episode_cap=spec.eval_episode_cap or 200,
        q_network=MLP(hidden=(64, 64)),
        replay=Replay(capacity=50000, batch_size=32),
        optimizer=warmed_update(
            inner=adam(lr=0.0001), warmup_steps=100,
        ),
        action_select=linear_epsilon(),
        bootstrap=bootstrap(),
        loss_fn=squared_error,
        target_sync=periodic_copy,
        state_hash=spec.state_hash,
    )
    if isinstance(seed_or_seeds, int):
        # chunk_size=1: direct call, no vmap.
        out = dqn(seed=seed_or_seeds, **args)
        return {
            'q_sum': float(jnp.sum(out.final_state.online_params.params[-1])),
            'eval_best': float(jnp.max(out.eval_outs.episode_returns.mean(axis=-1))),
        }
    # chunk_size>1: vmap over seeds, return slot 0.
    vmapped = jax.vmap(lambda s: dqn(seed=s, **args))
    out = vmapped(seed_or_seeds)
    return {
        'q_sum': float(jnp.sum(out.final_state.online_params.params[-1][0])),
        'eval_best': float(jnp.max(out.eval_outs.episode_returns[0].mean(axis=-1))),
    }


def main() -> None:
    print('=== chunk_size determinism probe ===')
    print('MetaMaze γ=0.999 MLP[64,64] 100k steps, CPU')
    print()
    print('config A: seed=0 alone (chunk_size=1)')
    a = build_and_run(0)
    print(f'  q_sum   = {a["q_sum"]:.10f}')
    print(f'  eval_best = {a["eval_best"]:.10f}')

    print('\nconfig B: vmap over [0, 1] (chunk_size=2), report slot 0')
    b = build_and_run(jnp.array([0, 1]))
    print(f'  q_sum   = {b["q_sum"]:.10f}')
    print(f'  eval_best = {b["eval_best"]:.10f}')

    print('\nconfig C: seed=0 alone, second run (byte-identity check)')
    c = build_and_run(0)
    print(f'  q_sum   = {c["q_sum"]:.10f}')
    print(f'  eval_best = {c["eval_best"]:.10f}')

    print('\n=== Diagnostics ===')
    ac_eq = (a['q_sum'] == c['q_sum']) and (a['eval_best'] == c['eval_best'])
    ab_eq = (a['q_sum'] == b['q_sum']) and (a['eval_best'] == b['eval_best'])
    print(f'A == C (determinism): {ac_eq}')
    print(f'A == B (chunk-size innocent): {ab_eq}')
    if ac_eq and ab_eq:
        print('VERDICT: chunk_size is innocent at the seed level. Postfix-probe gap is sha+sampling.')
    elif ac_eq and not ab_eq:
        print('VERDICT: chunk_size IS RNG-effective. Treat (seed, sha, chunk_size) as replicate identity.')
        print(f'  |A - B| q_sum: {abs(a["q_sum"] - b["q_sum"]):.6e}')
        print(f'  |A - B| eval_best: {abs(a["eval_best"] - b["eval_best"]):.6e}')
    else:
        print('VERDICT: non-determinism bug — A != C.')
        print(f'  |A - C| q_sum: {abs(a["q_sum"] - c["q_sum"]):.6e}')


if __name__ == '__main__':
    main()
