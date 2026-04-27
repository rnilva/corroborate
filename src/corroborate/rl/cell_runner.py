"""Cell runner — `run_dqn_cell(env_spec, seed, hypothesis) →
(RunRow, EvalTrajectoryRecord)`. The bridge between the DQN
training/eval substrate (`dqn_step` + `train_with_eval`) and the
schema layer (`RunRow` + `FactRow`).

One cell = one (env, seed, hypothesis) execution. The cell
runner:

1. Resolves `env, env_params` from `gymnax` for `env_spec.name`.
2. Builds `step_fn = partial(dqn_step, ..., **hypothesis.intervention)`
   so the hypothesis's intervention applies as slot swaps.
3. Builds the `eval_fn` closure for `train_with_eval` (fresh
   greedy rollouts every `eval_config.eval_every` training steps).
4. Runs `train_with_eval` → `(final_state, ComposedTrace)`.
5. Projects the late-window outcome from `trace.train`.
6. Runs each `Hypothesis.bridge` against `trace.train`, converts
   each `BridgeResult` to a `FactRow` (with `kind='invariant'`
   when `stats['kind']=='tautological'`).
7. Aggregates the per-cell verdict (axiom 18 precedence:
   INVARIANT_VIOLATION dominates).
8. Returns `(RunRow, eval_trace)` — the eval trace is paired
   alongside the RunRow (not embedded) because eval data is the
   Hasselt-overestimation-gap consumer's input, not part of the
   schema-layer corpus."""
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

from corroborate.bridge import Bridge, BridgeResult
from corroborate.hypothesis import Hypothesis
from corroborate.reductions import late_window_mean
from corroborate.rl.dqn.claims.q_network import mlp_q
from corroborate.rl.dqn.dqn import default_state_hash, dqn_step, init_state
from corroborate.rl.dqn.eval import (
    EvalBurstOut,
    EvalTrajectoryRecord,
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
        eval bursts across `total_steps`. Round down — the actual
        eval count will be `total_steps // (total_steps // n_evals)`,
        which equals `n_evals` when total_steps divides evenly."""
        if total_steps <= 0:
            raise ValueError(f'total_steps must be positive; got {total_steps}')
        if n_evals <= 0:
            raise ValueError(f'n_evals must be positive; got {n_evals}')
        eval_every = total_steps // n_evals
        if eval_every <= 0:
            raise ValueError(
                f'n_evals ({n_evals}) larger than total_steps ({total_steps}); '
                f'cannot fit even one super-step.',
            )
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
    outcome_key: str = 'ep_return',
    outcome_fraction: float = 0.1,
    cycle_id: str | None = None,
) -> tuple[RunRow, EvalTrajectoryRecord]:
    """Run one (env, seed, hypothesis) cell. Returns the RunRow
    summarising the cell + the eval trajectory record (paired,
    not embedded — eval data is the Hasselt-gap consumer's input,
    distinct from the schema-layer corpus).

    `q_network` is required as a separate kwarg (default `mlp_q`)
    because the eval pass needs it to compute predicted-Q-at-start
    on greedy rollouts. If the hypothesis swaps `q_network` via
    `intervention={'q_network': ...}`, callers should pass the
    same value here so eval rollouts use the matching network."""
    env, env_params = gymnax.make(env_spec.name)

    init: DQNState = init_state(
        env=env, env_params=env_params,
        obs_dim=env_spec.obs_dim, n_actions=env_spec.action_dim,
        seed=seed, optimizer=optimizer, buffer_capacity=buffer_capacity,
    )

    # Wire env-specific state_hash; default sentinel for envs
    # that don't declare one (image envs). The (s, a)-coverage
    # gap measurable detects the no-data case from env_spec, not
    # from inspecting the record values.
    state_hash = env_spec.state_hash or default_state_hash

    step_fn = partial(
        dqn_step,
        env=env, env_params=env_params,
        n_actions=env_spec.action_dim,
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

    _final_state, trace = train_with_eval(
        step_fn, init, total_steps,
        eval_fn=eval_fn, eval_every=eval_config.eval_every,
    )

    # Outcome projection — late-window mean of the chosen record key.
    outcome = late_window_mean(outcome_key, outcome_fraction)(trace.train)

    # Run hypothesis bridges → FactRows.
    intervention_sig: frozenset[str] = frozenset(
        slot for slot, _ in hypothesis.mechanism_key.intervention_signature
    )
    facts = tuple(
        _bridge_result_to_fact(
            bridge=b,
            result=b(trace.train),
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
        record_keys=tuple(trace.train.keys()),
        facts=facts,
        reads_set=reads_set,
        verdict=_aggregate_cell_verdict(facts),
    )
    return run_row, trace.eval


# ============ Helpers ============

def _bridge_result_to_fact(
    *,
    bridge: Bridge[DQNTrajectoryRecord],
    result: BridgeResult,
    intervention_signature: frozenset[str],
) -> FactRow:
    """Convert a BridgeResult to a FactRow at cell-level
    granularity. `kind` is read off `stats['kind']`: tautological
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


def _aggregate_cell_verdict(facts: tuple[FactRow, ...]) -> Verdict:
    """Aggregate per-cell facts into one run-level verdict.

    Axiom 18 precedence: any `INVARIANT_VIOLATION` dominates —
    the run sat outside a theorem's domain, so the outcome
    verdict is out of scope. Otherwise: HELD if all bridges
    held, NO_EFFECT if all rejected, else POWER_INSUFFICIENT
    (mixed signal — needs more data to resolve).

    Empty facts → POWER_INSUFFICIENT (no claims to test → can't
    say HELD)."""
    if not facts:
        return Verdict.POWER_INSUFFICIENT
    if any(f.verdict is Verdict.INVARIANT_VIOLATION for f in facts):
        return Verdict.INVARIANT_VIOLATION
    n = len(facts)
    n_held = sum(1 for f in facts if f.verdict is Verdict.HELD)
    n_rejected = sum(1 for f in facts if f.verdict is Verdict.NO_EFFECT)
    if n_held == n:
        return Verdict.HELD
    if n_rejected == n:
        return Verdict.NO_EFFECT
    return Verdict.POWER_INSUFFICIENT


