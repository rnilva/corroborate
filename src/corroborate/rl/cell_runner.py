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
4. Per-seed Python-side: project the late-window outcome, evaluate
   each hypothesis bridge (plus composition-discovered invariants),
   build a `RunRow` (verdict-side: leaves + bridge result paths)
   and a `TraceRow` (raw-data-side: leaves + 1-D trajectories) with
   shared id. Returns `CellResult(run, trace)` per seed.

The DQN algorithm itself lives entirely in the `dqn` claim
(`rl/dqn/dqn.py`). The cell runner has no knowledge of training-
step semantics; it's a generic vmap-and-build-records harness."""
from __future__ import annotations

import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from functools import partial
from typing import NamedTuple

import gymnax
import jax
import jax.numpy as jnp
import numpy as np

from corroborate import canonical_str
from corroborate.core.claim import trace_context
from corroborate.graph.computation import ComputationGraph, build_computation_graph
from corroborate.core.hypothesis import Hypothesis
from corroborate.rl.dqn.dqn import default_state_hash, dqn
from corroborate.rl.dqn.invariants import DQNTrajectoryRecord
from corroborate.rl.env_catalogue import EnvSpec, EnvWrapper, HasN
from corroborate.schema import MeasurementLeaf, RunRow, TraceLeaf, TraceRow
from corroborate.core.signature import walk, walk_paths
from corroborate.verdict import Verdict


class CellResult(NamedTuple):
    """One cell's pair of records — the verdict-side `RunRow` (with
    derived measurements: leaves + outcome reduction + bridge
    verdicts/stats) and the raw-data-side `TraceRow` (with
    leaves + 1-D trajectories from the configured-claim's record).

    `run.id == trace.id` — the two stores join on this UUID, so a
    consumer reading `runs.parquet` can pull the matching trace from
    `traces.parquet` and re-evaluate any bridge post-hoc."""
    run: RunRow
    trace: TraceRow


class ArmResult(NamedTuple):
    """One arm's per-seed cells plus the computation graph captured
    during the arm's vmap call.

    `graph` is a `ComputationGraph` derived from `@claim` records
    fired during JAX's abstract-trace pass — structurally constant
    across seeds since vmap traces the body once. Held in memory,
    not persisted to parquet (parquets are for measurable values;
    the graph is a runtime artifact of the bound hypothesis).

    Use `result.cells` for the per-seed CellResults; use
    `result.graph` to access the static call graph (for downstream
    redundancy / register / mechanism-key consumers)."""
    cells: tuple[CellResult, ...]
    graph: ComputationGraph


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
    canonicalise to string via `canonical_str`."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, str)):
        return value
    return canonical_str(value)


def _leaf_measurements(configured: object) -> dict[str, MeasurementLeaf]:
    """Topology walk → dotted-path leaves. Each `walk_paths`
    KwargInfo's default contributes one measurement at its dotted
    path. Leaves are non-recursive scalar claims of the configured
    composition (RL practice's "hyperparameters")."""
    paths = walk_paths(walk(configured), regime='leaf')
    return {path: _leaf_scalar(kw.default) for path, kw in paths.items()}


def _trajectory_leaves(
    record: Mapping[str, jax.Array],
) -> dict[str, TraceLeaf]:
    """Project the per-cell record to trajectory entries keyed by
    the substrate-author's record key. 0-D arrays surface as
    Python scalars; 1-D+ arrays surface as numpy arrays preserving
    JAX's source dtype.

    Going through `np.asarray(arr)` (not `arr.tolist()`) preserves
    `float32` / `int32`: polars infers `List(Float32)` /
    `List(Int32)` from the numpy dtype at DataFrame construction,
    avoiding the parquet-side round-trip-to-Python-float upcast
    that doubled per-step series storage. 0-D values must be
    Python scalars — polars rejects 0-D ndarrays in row-dict
    construction."""
    out: dict[str, TraceLeaf] = {}
    for key, arr in record.items():
        np_arr: np.ndarray = np.asarray(arr)
        if np_arr.ndim == 0:
            out[key] = np_arr.item()
        else:
            out[key] = np_arr
    return out


# `_bridge_result_to_measurements` lives in `aggregate` (the data
# layer between live BridgeResults and persisted RunRow.measurements).
# This sweep-path call site uses the canonical impl from there.


def run_dqn_arm(
    env_spec: EnvSpec,
    seeds: tuple[int, ...],
    hypothesis: Hypothesis[DQNTrajectoryRecord],
    *,
    cycle_id: str | None = None,
    wrappers: tuple[EnvWrapper, ...] = (),
) -> ArmResult:
    """Run one (env, hypothesis) arm across `seeds` in parallel via
    `jax.vmap` of the `dqn` outermost claim. Returns
    `ArmResult(cells, graph)` — per-seed `CellResult`s plus the
    `ComputationGraph` captured from the bound hypothesis.

    Graph capture: the vmap call is wrapped in `trace_context()`,
    so JAX's first-call abstract-trace pass fires `@claim` records
    once. `build_computation_graph(records)` derives the static
    call graph from those records. This is structurally constant
    across seeds (vmap traces the body once); the graph is a
    property of the bound hypothesis, not of any single seed.

    Substrate-internal knobs (optimizer, outcome window fraction,
    etc.) are NOT runner kwargs. `dqn`'s signature carries its
    own defaults; experiments that want to override them put the
    override in `Hypothesis.intervention`. Keeping the runner
    surface to (env_spec, seeds, hypothesis, cycle_id) makes
    Hypothesis the sole experiment specification."""
    if not seeds:
        raise ValueError('seeds must be non-empty')

    intervention = hypothesis.intervention
    total_steps = _read_total_steps(intervention)

    env, env_params = gymnax.make(env_spec.name)
    # Apply wrappers in order. Each is a frozen-dataclass with a
    # `wrap(inner)` method (the `EnvWrapper` Protocol). Tuple
    # parameter is statically typed `tuple[EnvWrapper, ...]` so
    # the runtime check is defensive against caller errors that
    # bypass the type system — drop in favour of trusting the
    # contract.
    for w in wrappers:
        env = w.wrap(env)
    state_hash = (
        env_spec.state_hash
        if env_spec.state_hash is not None
        else default_state_hash
    )

    # Compose cell-level exogenous + intervention into dqn via
    # `functools.partial`. The walker / `collect_invariants` /
    # `canonical_str` all unwrap partials, so intervention
    # overrides shadow defaults in every downstream consumer:
    # `collect_invariants(configured)` sees only the effective
    # sub-claims (no leakage from defaults that intervention
    # swapped out).
    # Read n_actions from the wrapped env's action_space, not the
    # static catalogue, so wrappers like ActionDuplicate that
    # inflate |A| are reflected in the Q-network output dim and
    # the epsilon-greedy sampling range.
    wrapped_action_space = env.action_space(env_params)
    if not isinstance(wrapped_action_space, HasN):
        raise TypeError(
            f"wrapped env action_space lacks `.n` (Discrete); "
            f'got {type(wrapped_action_space).__name__}',
        )
    n_actions = int(wrapped_action_space.n)
    cell_kwargs: dict[str, object] = {
        'env': env, 'env_params': env_params,
        'obs_shape': env_spec.observation_shape, 'n_actions': n_actions,
        'eval_episode_cap': env_spec.eval_episode_cap,
        'state_hash': state_hash,
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
    # Wrap the vmap call in trace_context so JAX's first-call
    # abstract-trace pass fires @claim records once; that single
    # pass IS the structural graph (per-(theory, intervention),
    # constant across seeds). build_computation_graph derives the
    # static call graph from the records.
    with trace_context() as records:
        batched_record = jax.vmap(by_key)(keys)
    graph = build_computation_graph(records)

    # Side-effect import: registers DDQN measurables (q_mean,
    # q_max, ..., pearson_r_online_target, late_window_mean) so
    # measurables declaring them as deps auto-resolve via the
    # registry. Substrate-side `dqn_default_measurables()` is
    # how authors enumerate the standard set on each Hypothesis.
    import corroborate.rl.dqn.measurables  # noqa: F401  # pyright: ignore[reportUnusedImport]
    from corroborate.measurables import evaluate_with_measurables

    cells: list[CellResult] = []
    for i, seed in enumerate(seeds):
        per_seed_record: dict[str, jax.Array] = {
            k: v[i] for k, v in batched_record.items()
        }
        # Per-cell measurable cache — shared across `hypothesis.
        # measurables` so dep-measurables (q_mean, q_std, etc.)
        # compute once per cell.
        cache: dict[str, object] = {}

        # Pre-registered measurables: walk `hypothesis.measurables`
        # and persist each at its bare measurable name as the
        # column key. Phase 5 of the Bridge-collapse refactor
        # normalised the substrate-paper-narrative prefixes
        # (`outcome.` / `mechanism.` / `invariant.`) — measurable
        # names are now bare (`eval_final_mean`, `jensen_gap`,
        # `late_window_mean`).
        measurable_cols: dict[str, MeasurementLeaf] = {}
        for m in hypothesis.measurables:
            value = evaluate_with_measurables(
                m.fn, per_seed_record, cache=cache,
            )
            scalar = _leaf_scalar(value)
            measurable_cols[m.name] = scalar

        # Each wrapper declares its own measurement keys via
        # `measurement_keys()` — no central isinstance chain.
        # Adding a new wrapper just adds the method; cell_runner
        # is invariant under wrapper-type extension.
        wrapper_cols: dict[str, MeasurementLeaf] = {}
        for w in wrappers:
            wrapper_cols.update(w.measurement_keys())
        measurements: dict[str, MeasurementLeaf] = {
            'intervention_name': hypothesis.name,
            'env_name': env_spec.name,
            'seed': seed,
            'total_steps': total_steps,
            **wrapper_cols,
            **leaf_measurements,
            **measurable_cols,
        }
        # Per-cell verdict: a successfully-completed cell is HELD;
        # the per-cell `Bridge[R]` aggregation that produced
        # Popperian verdicts is gone (Phase 4C). Verdicts now
        # emerge post-hoc from corpus-side `claim_bridge.Bridge`
        # declarations consuming the persisted columns.
        verdict = Verdict.HELD

        # Shared id between the two stores: a downstream consumer
        # reads `runs.parquet`, picks an id, fetches the matching
        # trace row by `id` from `traces.parquet`, and re-evaluates
        # any bridge against the persisted trajectory.
        cell_id = str(uuid.uuid4())
        timestamp = datetime.now(UTC).isoformat(timespec='seconds')

        run = RunRow(
            id=cell_id, parent_id=None,
            cycle_id=cycle_id, timestamp=timestamp,
            verdict=verdict, arm_key=hypothesis.arm_key(),
            measurements=measurements,
        )
        # Trace leaves: configurational leaves (shared with the
        # RunRow's measurements) + 1-D trajectories from the
        # per-seed record. Heterogeneous types per leaf — scalars
        # for leaves, lists for trajectories.
        trace_leaves: dict[str, TraceLeaf] = {}
        trace_leaves.update(leaf_measurements)
        trace_leaves.update(_trajectory_leaves(per_seed_record))
        trace = TraceRow(
            id=cell_id, cycle_id=cycle_id, timestamp=timestamp,
            leaves=trace_leaves,
        )
        cells.append(CellResult(run=run, trace=trace))
    return ArmResult(cells=tuple(cells), graph=graph)


def run_dqn_cell(
    env_spec: EnvSpec,
    seed: int,
    hypothesis: Hypothesis[DQNTrajectoryRecord],
    *,
    cycle_id: str | None = None,
) -> CellResult:
    """Run one (env, seed, hypothesis) cell. Thin convenience
    wrapper around `run_dqn_arm` for the single-seed case;
    multi-seed callers should use `run_dqn_arm` directly to avoid
    per-call vmap re-compilation. Discards the graph; callers
    that want it should use `run_dqn_arm` directly."""
    arm = run_dqn_arm(
        env_spec, (seed,), hypothesis,
        cycle_id=cycle_id,
    )
    return arm.cells[0]
