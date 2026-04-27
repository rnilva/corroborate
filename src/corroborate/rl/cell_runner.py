"""Cell runner — `run_dqn_cell(env_spec, seed, hypothesis) →
RunRow`. The bridge between the DQN training/eval substrate
(`dqn_step` + `train_with_eval`) and the schema layer
(`RunRow` + `FactRow`).

One cell = one (env, seed, hypothesis) execution. The cell
runner:

1. Resolves `env, env_params` from `gymnax` for `env_spec.name`.
2. Builds `step_fn = partial(dqn_step, ..., **hypothesis.intervention)`
   so the hypothesis's intervention applies as slot swaps.
3. Builds the `eval_fn` closure for `train_with_eval` (fresh
   greedy rollouts every `eval_config.eval_every` training steps).
4. Runs `train_with_eval` → `(final_state, record)`. The record
   is a single dict mixing per-step training fields (shape
   `(total_steps, ...)`) and per-burst eval fields (shape
   `(n_bursts, K, ...)`) — see `train_with_eval`'s docstring.
5. Projects the late-window outcome from `record`.
6. Runs each `Hypothesis.bridge` against the merged record;
   bridges target whichever keys they care about. Converts each
   `BridgeResult` to a `FactRow` (with `kind='invariant'` when
   `stats['kind']=='tautological'`).
7. Aggregates the per-cell verdict (axiom 18 precedence:
   INVARIANT_VIOLATION dominates).
8. Returns the `RunRow`."""
from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from typing import Literal

import gymnax
import jax
import optax

from corroborate.aggregate import aggregate_cell_verdict
from corroborate.bridge import Bridge, BridgeResult
from corroborate.hypothesis import Hypothesis
from corroborate.reductions import masked_window_mean
from corroborate.rl.dqn.claims.q_network import mlp_q
from corroborate.rl.dqn.dqn import default_state_hash, dqn_step, init_state
from corroborate.rl.dqn.eval import (
    EvalBurstOut,
    eval_burst,
    train_with_eval,
)
from corroborate.rl.dqn.invariants import DQNTrajectoryRecord
from corroborate.rl.dqn.state import DQNState
from corroborate.rl.dqn.types import QNetwork
from corroborate.rl.env_catalogue import EnvSpec
from corroborate.schema import FactRow, RunRow
from corroborate.verdict import Verdict


# ============ EvalConfig ============

@dataclass(frozen=True, slots=True)
class EvalConfig:
    """Eval-loop scheduling for `run_dqn_cell`. `eval_every` is in
    training-step units; `n_episodes` is K (Hasselt 2016 default
    20). `total_steps` must be a multiple of `eval_every`
    (`train_with_eval` enforces this).

    Use `EvalConfig.n_evals(total_steps, n_evals)` to construct a
    config that gives `n_evals` evenly-spaced bursts across the
    training run — the natural cross-env-scaling shape since
    total_steps varies wildly (CartPole 50k, MinAtar 10M)."""
    eval_every: int
    n_episodes: int = 20

    @classmethod
    def n_evals(
        cls,
        total_steps: int,
        n_evals: int = 20,
        n_episodes: int = 20,
    ) -> EvalConfig:
        """Construct an EvalConfig with `n_evals` evenly-spaced
        eval bursts across `total_steps`. Requires `total_steps`
        divisible by `n_evals` (so each super-step is a clean
        chunk and `train_with_eval` accepts the resulting
        eval_every without remainder)."""
        if total_steps <= 0:
            raise ValueError(f'total_steps must be positive; got {total_steps}')
        if n_evals <= 0:
            raise ValueError(f'n_evals must be positive; got {n_evals}')
        if n_evals > total_steps:
            raise ValueError(
                f'n_evals ({n_evals}) larger than total_steps ({total_steps}); '
                f'cannot fit even one super-step.',
            )
        if total_steps % n_evals != 0:
            raise ValueError(
                f'EvalConfig.n_evals: total_steps ({total_steps}) must be '
                f'divisible by n_evals ({n_evals}); got remainder '
                f'{total_steps % n_evals}.',
            )
        eval_every = total_steps // n_evals
        return cls(eval_every=eval_every, n_episodes=n_episodes)


# ============ The cell runner ============

def run_dqn_cell(
    env_spec: EnvSpec,
    seed: int,
    hypothesis: Hypothesis[DQNTrajectoryRecord],
    *,
    total_steps: int,
    optimizer: optax.GradientTransformation,
    eval_config: EvalConfig,
    q_network: QNetwork = mlp_q,
    gamma: float = 0.99,
    batch_size: int = 64,
    buffer_capacity: int = 10_000,
    warmup_steps: int = 1_000,
    sync_period: int = 100,
    outcome_fraction: float = 0.1,
    cycle_id: str | None = None,
) -> RunRow:
    """Run one (env, seed, hypothesis) cell. Returns a `RunRow`
    summarising the cell.

    Eval IS part of training: `train_with_eval` produces ONE
    record dict mixing per-step training fields and per-burst
    eval fields. Hypothesis bridges read whichever keys they
    target (jensen_overestimation_gap reads
    `predicted_q_at_start` + `mc_return`; fqi_decay_gap reads
    `td_error`; etc.) — the framework doesn't distinguish "train
    bridges" from "eval bridges" in any layer.

    `q_network` is required as a separate kwarg (default `mlp_q`)
    because the eval pass needs it to compute predicted-Q-at-start
    on greedy rollouts. If the hypothesis swaps `q_network` via
    `intervention={'q_network': ...}`, callers should pass the
    same value here so eval rollouts use the matching network."""
    env, env_params = gymnax.make(env_spec.name)

    init: DQNState = init_state(
        env=env, env_params=env_params,
        obs_dim=env_spec.obs_dim, n_actions=env_spec.n_actions,
        seed=seed, optimizer=optimizer, buffer_capacity=buffer_capacity,
    )

    # Wire env-specific state_hash; default sentinel for envs
    # that don't declare one (image envs). The (s, a)-coverage
    # gap measurable detects the no-data case from env_spec, not
    # from inspecting the record values.
    state_hash = (
        env_spec.state_hash
        if env_spec.state_hash is not None
        else default_state_hash
    )

    step_fn = partial(
        dqn_step,
        env=env, env_params=env_params,
        n_actions=env_spec.n_actions,
        optimizer=optimizer,
        state_hash=state_hash,
        gamma=gamma, batch_size=batch_size,
        buffer_capacity=buffer_capacity,
        warmup_steps=warmup_steps, sync_period=sync_period,
        **hypothesis.intervention,
    )

    eval_seed = seed * 1000 + 1  # deterministic, distinct from training seed
    base_eval_key = jax.random.PRNGKey(eval_seed)

    def eval_fn(s: DQNState, idx: jax.Array) -> EvalBurstOut:
        return eval_burst(
            online_params=s.online_params,
            env=env, env_params=env_params,
            rng_key=jax.random.fold_in(base_eval_key, idx),
            q_network=q_network, gamma=gamma,
            episode_cap=env_spec.eval_episode_cap,
            n_episodes=eval_config.n_episodes,
        )

    _final_state, record = train_with_eval(
        step_fn, init, total_steps,
        eval_fn=eval_fn, eval_every=eval_config.eval_every,
    )

    # Outcome projection — late-window mean of *episode-end*
    # returns (filtered to done==1 in the late window). Plain
    # `late_window_mean('ep_return', ...)` would average over a
    # cumulative-within-episode sawtooth; this one filters to
    # episode terminations so the value is the per-episode return.
    outcome = masked_window_mean(
        value_key='ep_return', mask_key='done',
        fraction=outcome_fraction,
    )(record)

    # Run all hypothesis bridges against the merged record.
    # Bridges target arbitrary keys regardless of which sub-pass
    # produced them (training fields, eval-burst fields, etc.).
    intervention_sig: frozenset[str] = frozenset(
        slot for slot, _ in hypothesis.mechanism_key.intervention_signature
    )
    facts = tuple(
        _bridge_result_to_fact(
            bridge=b,
            result=b(record),
            intervention_signature=intervention_sig,
        )
        for b in hypothesis.bridges
    )

    reads_set: frozenset[str] = frozenset()
    for f in facts:
        reads_set = reads_set | f.reads

    run_row = RunRow(
        id=str(uuid.uuid4()),
        parent_id=None,
        intervention_name=hypothesis.name,
        cycle_id=cycle_id,
        timestamp=datetime.now(UTC).isoformat(timespec='seconds'),
        env_name=env_spec.name,
        total_steps=total_steps,
        seed=seed,
        mechanism_key=hypothesis.mechanism_key,
        primary_outcome_summary=outcome,
        record_keys=tuple(record.keys()),
        facts=facts,
        reads_set=reads_set,
        verdict=aggregate_cell_verdict(facts),
    )
    return run_row


# ============ Helpers ============

def _bridge_result_to_fact[R: Mapping[str, object]](
    *,
    bridge: Bridge[R],
    result: BridgeResult,
    intervention_signature: frozenset[str],
) -> FactRow:
    """Convert a BridgeResult to a FactRow at cell-level
    granularity. Generic over the bridge's record type so the
    same helper handles both primary-record bridges (over
    `DQNTrajectoryRecord`) and secondary-record bridges (over the
    eval record). `kind` is read off `stats['kind']`: tautological
    → 'invariant', otherwise → 'bridge'.

    `natural_strength` is a binary placeholder (1.0 for HELD, 0.0
    otherwise) — step 5 (statistics module) replaces this with
    real q values from Hedges' g / sample sizes. `delta_i` stays
    0.0 at cell level; populated at the comparison level by the
    aggregation pipeline."""
    return FactRow(
        name=bridge.name,
        kind=_classify_kind(result.stats),
        targets=bridge.targets,
        verdict=result.verdict,
        natural_strength=1.0 if result.verdict is Verdict.HELD else 0.0,
        delta_i=0.0,
        evidentiary_level='cell',
        stats=dict(result.stats),
        intervention_signature=intervention_signature,
    )


def _classify_kind(
    stats: Mapping[str, float | int | bool | str],
) -> Literal['bridge', 'invariant']:
    """Read `stats['kind']` and project to FactRow's
    `Literal['bridge', 'invariant']`. The `@invariant` decorator
    sets `stats['kind']='tautological'`; everything else is a
    plain bridge."""
    kind_raw = stats.get('kind')
    if kind_raw == 'tautological':
        return 'invariant'
    return 'bridge'


