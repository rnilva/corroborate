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
   evaluate each hypothesis bridge (plus composition-discovered
   invariants), build the RunRow with measurements at HP topology
   paths + bridge result paths.

The DQN algorithm itself lives entirely in the `dqn` claim
(`rl/dqn/dqn.py`). The cell runner has no knowledge of training-
step semantics; it's a generic vmap-and-build-RunRow harness."""
from __future__ import annotations

import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from functools import partial

import gymnax
import jax
import jax.numpy as jnp

from corroborate.aggregate import aggregate_cell_verdict
from corroborate.bridge import Bridge, BridgeResult
from corroborate.hypothesis import (
    Hypothesis,
    _canonical_str,  # pyright: ignore[reportPrivateUsage]
)
from corroborate.reductions import masked_window_mean
from corroborate.rl.dqn.claims.optimizer import Adam
from corroborate.rl.dqn.dqn import default_state_hash, dqn
from corroborate.rl.dqn.invariants import DQNTrajectoryRecord
from corroborate.rl.dqn.types import OptimizerFactory
from corroborate.rl.env_catalogue import EnvSpec
from corroborate.schema import MeasurementLeaf, RunRow
from corroborate.signature import collect_invariants, walk, walk_paths


# `total_steps` default — must match `dqn`'s default. Read from
# intervention when present, fall back to this when absent.
_DEFAULT_TOTAL_STEPS: int = 50_000


def _read_total_steps(intervention: Mapping[str, object]) -> int:
    """Read `total_steps` from intervention, defaulting when absent.
    Used only to populate the `total_steps` measurement — the
    value also flows to `dqn` itself via `**intervention` if the
    author set it. Loud error on wrong-typed override."""
    if 'total_steps' not in intervention:
        return _DEFAULT_TOTAL_STEPS
    v = intervention['total_steps']
    if isinstance(v, bool) or not isinstance(v, int):
        raise TypeError(
            f"intervention['total_steps'] must be int, "
            f"got {type(v).__name__}",
        )
    return v


def _leaf_scalar(value: object) -> MeasurementLeaf:
    """Coerce a leaf value to a scalar measurement. Primitives pass
    through; structured values (Modules, partials, FnClaims)
    canonicalise to string via `_canonical_str`."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, str)):
        return value
    return _canonical_str(value)


def _leaf_measurements(configured: object) -> dict[str, MeasurementLeaf]:
    """Topology walk → dotted-path leaves. Each `walk_paths`
    KwargInfo's default contributes one measurement at its dotted
    path. Leaves are non-recursive scalar claims of the configured
    composition (RL practice's "hyperparameters")."""
    paths = walk_paths(walk(configured), regime='leaf')
    return {path: _leaf_scalar(kw.default) for path, kw in paths.items()}


def _bridge_result_to_measurements(
    result: BridgeResult,
) -> dict[str, MeasurementLeaf]:
    """Flatten a BridgeResult into path-keyed measurements.

    `bridge.<name>.verdict` (or `invariant.<name>.verdict` when
    `stats['kind'] == 'tautological'`) carries the verdict; each
    scalar entry of `result.stats` lands under
    `<prefix>.<name>.stats.<key>`."""
    is_invariant = result.stats.get('kind') == 'tautological'
    prefix = f'invariant.{result.name}' if is_invariant else f'bridge.{result.name}'
    out: dict[str, MeasurementLeaf] = {
        f'{prefix}.verdict': result.verdict.value,
    }
    # `BridgeResult.stats` is typed `Mapping[str, float | int |
    # bool | str]` — every value already satisfies
    # MeasurementLeaf. Forward each entry verbatim.
    for stat_key, stat_value in result.stats.items():
        out[f'{prefix}.stats.{stat_key}'] = stat_value
    return out


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

    # Configurational fingerprint — the leaf measurements that
    # `aggregate.leaf_signature` projects to as the group-by key.
    # Walks the BOUND `configured` so intervention overrides
    # surface at their dotted topology paths.
    leaf_measurements = _leaf_measurements(configured)

    def by_key(rng_key: jax.Array) -> dict[str, jax.Array]:
        return configured(rng_key=rng_key)

    keys = jax.vmap(jax.random.PRNGKey)(
        jnp.asarray(seeds, dtype=jnp.uint32),
    )
    batched_record = jax.vmap(by_key)(keys)

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
        bridge_results = tuple(b(per_seed_record) for b in effective_bridges)
        verdict = aggregate_cell_verdict(
            tuple(r.verdict for r in bridge_results),
        )

        measurements: dict[str, MeasurementLeaf] = {
            'intervention_name': hypothesis.name,
            'env_name': env_spec.name,
            'seed': seed,
            'total_steps': total_steps,
            'outcome.late_window_mean': outcome,
            **leaf_measurements,
        }
        for result in bridge_results:
            measurements.update(_bridge_result_to_measurements(result))

        rows.append(RunRow(
            id=str(uuid.uuid4()),
            parent_id=None,
            cycle_id=cycle_id,
            timestamp=datetime.now(UTC).isoformat(timespec='seconds'),
            verdict=verdict,
            measurements=measurements,
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
