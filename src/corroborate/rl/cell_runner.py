"""Cell runner — bridges the `dqn` outermost claim to the schema
layer. One cell = one (env, seed, hypothesis) execution.

The runner is thin:

1. Resolves `env, env_params` from `gymnax`.
2. Binds the cell's exogenous knobs (env, dims, eval-episode-cap,
   state_hash) and the hypothesis's intervention into `dqn` via
   `functools.partial`. Intervention mirrors `dqn`'s signature, so
   `**hypothesis.intervention` spreads directly — no broadcast,
   no flatten, no validation. Pyright catches signature mismatches
   at the swap site.
3. vmap-over-seeds: each seed becomes a `jax.random.PRNGKey`; the
   batched call runs `dqn` once jit-compiled and produces a record
   pytree where each leaf has a leading `(n_seeds, ...)` axis.
4. Per-seed Python-side: project the late-window outcome,
   evaluate each hypothesis bridge, build the RunRow.

The DQN algorithm itself lives entirely in the `dqn` claim
(`rl/dqn/dqn.py`) — composition of init_state, nested scan, and
record assembly. The cell runner has no knowledge of training-step
semantics; it's a generic vmap-and-build-RunRow harness."""
from __future__ import annotations

import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from functools import partial
from typing import Literal

import gymnax
import jax
import jax.numpy as jnp

from corroborate.aggregate import aggregate_cell_verdict
from corroborate.bridge import Bridge, BridgeResult
from corroborate.hypothesis import Hypothesis
from corroborate.reductions import masked_window_mean
from corroborate.rl.dqn.claims.optimizer import Adam
from corroborate.rl.dqn.dqn import default_state_hash, dqn
from corroborate.rl.dqn.invariants import DQNTrajectoryRecord
from corroborate.rl.dqn.types import OptimizerFactory
from corroborate.rl.env_catalogue import EnvSpec
from corroborate.schema import FactRow, RunRow
from corroborate.signature import collect_invariants
from corroborate.verdict import Verdict


# `total_steps` default — must match `dqn`'s default. Read from
# intervention when present, fall back to this when absent.
_DEFAULT_TOTAL_STEPS: int = 50_000


def _read_total_steps(intervention: Mapping[str, object]) -> int:
    """Read `total_steps` from intervention, defaulting when absent.
    Used only to populate `RunRow.total_steps` — the value also
    flows to `dqn` itself via `**intervention` if the author set
    it. Loud error on wrong-typed override."""
    if 'total_steps' not in intervention:
        return _DEFAULT_TOTAL_STEPS
    v = intervention['total_steps']
    if isinstance(v, bool) or not isinstance(v, int):
        raise TypeError(
            f"intervention['total_steps'] must be int, "
            f"got {type(v).__name__}",
        )
    return v


def run_dqn_arm(
    env_spec: EnvSpec,
    seeds: tuple[int, ...],
    hypothesis: Hypothesis[DQNTrajectoryRecord],
    *,
    optimizer: OptimizerFactory = Adam(),
    outcome_fraction: float = 0.1,
    cycle_id: str | None = None,
) -> tuple[RunRow, ...]:
    """Run one (env, hypothesis) arm across `seeds` in parallel via
    `jax.vmap` of the `dqn` outermost claim. Returns one `RunRow`
    per seed.

    `optimizer` is also an HP in `dqn`'s signature; the runner
    accepts it as a kwarg for the common case where the experiment
    threads one optimizer choice across an arm. If a hypothesis
    intervenes on `optimizer`, that intervention wins (intervention
    ordering mirrors `partial`'s kwarg-merge semantics)."""
    if not seeds:
        raise ValueError('seeds must be non-empty')

    intervention = hypothesis.intervention
    total_steps = _read_total_steps(intervention)

    env, env_params = gymnax.make(env_spec.name)
    state_hash = (
        env_spec.state_hash
        if env_spec.state_hash is not None
        else default_state_hash
    )

    # Compose cell-level exogenous + intervention into dqn via
    # `functools.partial`. The walker / `collect_invariants` /
    # `_canonical_str` all unwrap partials, so intervention
    # overrides shadow defaults in every downstream consumer:
    # `collect_invariants(configured)` sees only the effective
    # sub-claims (no leakage from defaults that intervention
    # swapped out).
    cell_kwargs: dict[str, object] = {
        'env': env, 'env_params': env_params,
        'obs_dim': env_spec.obs_dim, 'n_actions': env_spec.n_actions,
        'eval_episode_cap': env_spec.eval_episode_cap,
        'state_hash': state_hash,
        'optimizer': optimizer,
    }
    configured = partial(dqn, **{**cell_kwargs, **intervention})

    def by_key(rng_key: jax.Array) -> dict[str, jax.Array]:
        return configured(rng_key=rng_key)

    keys = jax.vmap(jax.random.PRNGKey)(
        jnp.asarray(seeds, dtype=jnp.uint32),
    )
    batched_record = jax.vmap(by_key)(keys)

    intervention_sig: frozenset[str] = frozenset(
        slot for slot, _ in hypothesis.mechanism_key.intervention_signature
    )
    outcome_proj = masked_window_mean(
        value_key='ep_return', mask_key='done',
        fraction=outcome_fraction,
    )

    # Author-declared bridges + composition-discovered invariants.
    # Walk the BOUND `configured` tree (with intervention applied)
    # — this surfaces invariants attached only to the effective
    # sub-claims, not stale ones from defaults that were swapped
    # out. De-dup by id.
    auto_invariants: list[Bridge[DQNTrajectoryRecord]] = []
    seen_ids: set[int] = set()
    for inv in collect_invariants(configured):
        if id(inv) not in seen_ids:
            seen_ids.add(id(inv))
            auto_invariants.append(inv)  # pyright: ignore[reportArgumentType]
    effective_bridges: tuple[Bridge[DQNTrajectoryRecord], ...] = (
        tuple(hypothesis.bridges) + tuple(auto_invariants)
    )

    rows: list[RunRow] = []
    for i, seed in enumerate(seeds):
        per_seed_record: dict[str, jax.Array] = {
            k: v[i] for k, v in batched_record.items()
        }
        outcome = outcome_proj(per_seed_record)
        facts = tuple(
            _bridge_result_to_fact(
                bridge=b,
                result=b(per_seed_record),
                intervention_signature=intervention_sig,
            )
            for b in effective_bridges
        )
        reads_set: frozenset[str] = frozenset()
        for f in facts:
            reads_set = reads_set | f.reads

        rows.append(RunRow(
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
            record_keys=tuple(per_seed_record.keys()),
            facts=facts,
            reads_set=reads_set,
            verdict=aggregate_cell_verdict(facts),
            meta={},
        ))
    return tuple(rows)


def run_dqn_cell(
    env_spec: EnvSpec,
    seed: int,
    hypothesis: Hypothesis[DQNTrajectoryRecord],
    *,
    optimizer: OptimizerFactory = Adam(),
    outcome_fraction: float = 0.1,
    cycle_id: str | None = None,
) -> RunRow:
    """Run one (env, seed, hypothesis) cell. Thin convenience
    wrapper around `run_dqn_arm` for the single-seed case;
    multi-seed callers should use `run_dqn_arm` directly to avoid
    per-call vmap re-compilation."""
    rows = run_dqn_arm(
        env_spec, (seed,), hypothesis,
        optimizer=optimizer,
        outcome_fraction=outcome_fraction,
        cycle_id=cycle_id,
    )
    return rows[0]


def _bridge_result_to_fact[R: Mapping[str, object]](
    *,
    bridge: Bridge[R],
    result: BridgeResult,
    intervention_signature: frozenset[str],
) -> FactRow:
    """Convert a BridgeResult to a FactRow at cell-level
    granularity. `kind` is read off `stats['kind']`: tautological
    → 'invariant', otherwise → 'bridge'.

    `natural_strength` is a binary placeholder (1.0 for HELD, 0.0
    otherwise) — step 5 (statistics module) replaces this with real
    q values from Hedges' g / sample sizes. `delta_i` stays 0.0 at
    cell level; populated at the comparison level by the
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
    sets `stats['kind']='tautological'`; everything else is a plain
    bridge."""
    kind_raw = stats.get('kind')
    if kind_raw == 'tautological':
        return 'invariant'
    return 'bridge'
