"""Cell runner — bridges the `dqn` outermost claim to the schema
layer. One cell = one (env, seed, hypothesis) execution.

The runner is thin:

1. Resolves `env, env_params` via `env_catalogue.make_env`
   (gymnax for classic_control / minatar / bsuite, jumanji for
   `*-jumanji` registered envs).
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
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from corroborate import trace_context
from corroborate.core import canonical_str
from corroborate.graph.computation import ComputationGraph, build_computation_graph
from corroborate_rl.dqn.dqn import default_state_hash
from corroborate_rl.dqn.invariants import DQNTrajectoryRecord
from corroborate_rl.env_catalogue import EnvSpec, EnvWrapper, make_env
from corroborate.corpus.schema import MeasurementLeaf, RunRow, TraceLeaf, TraceRow
from corroborate.core.signature import walk, walk_paths
from corroborate.bridge.verdict import Verdict
from corroborate.measurables import Measurable


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
    claim: Callable[..., DQNTrajectoryRecord],
    arm_key: str,
    measurables: tuple[Measurable[DQNTrajectoryRecord, object], ...],
    *,
    cycle_id: str | None = None,
    wrappers: tuple[EnvWrapper, ...] = (),
) -> ArmResult:
    """Run one (env, arm) arm across `seeds` in parallel via
    `jax.vmap` of the composed `claim`. Returns
    `ArmResult(cells, graph)` — per-seed `CellResult`s plus the
    `ComputationGraph` captured from the bound claim.

    `claim` is the substrate's theory composed with one arm's
    Intervention tuple — typically `apply_interventions(base,
    intervention.treatment)` where `base = partial(dqn, **HPs)`.
    The framework's `run_intervention` builds it; substrate
    callers pass it through.

    `arm_key` is the framework-derived canonical fingerprint of
    the Intervention tuple (`combined_arm_key`). Cells are tagged
    with this; arm identity flows from the typed structural
    deltas, not from substrate-chosen short labels.

    `measurables` are the typed Measurable instances from the
    Hypothesis Protocol's `MEASURABLES` tuple. The runner
    computes each per cell and persists the scalar at the
    measurable's `.name`.

    Graph capture: the vmap call is wrapped in `trace_context()`,
    so JAX's first-call abstract-trace pass fires `@claim` records
    once. `build_computation_graph(records)` derives the static
    call graph. Structurally constant across seeds (vmap traces
    the body once)."""
    if not seeds:
        raise ValueError('seeds must be non-empty')

    env, env_params = make_env(env_spec)
    for w in wrappers:
        env = w.wrap(env)
    state_hash = (
        env_spec.state_hash
        if env_spec.state_hash is not None
        else default_state_hash
    )

    # Compose cell-level exogenous on top of the already-bound
    # `claim`. Read n_actions from the wrapped env's action_space
    # so wrappers like ActionDuplicate that inflate |A| reflect
    # in the Q-network output dim.
    #
    # `env_name` and `wrappers` are bound here as **author
    # primitives** of dqn (post-Phase-A0 refactor — they're plain
    # kwargs, not Annotated[Exogenous]); `walk_paths` surfaces them
    # as topology leaves in `_leaf_measurements(configured)` below.
    # `env`, `env_params`, `obs_shape`, `n_actions`,
    # `eval_episode_cap`, `state_hash` are framework-injected
    # (Annotated[Exogenous]) — the runner builds them Python-side.
    wrapped_action_space = env.action_space(env_params)
    n_actions = int(wrapped_action_space.n)
    cell_kwargs: dict[str, object] = {
        'env_name': env_spec.name,
        'wrappers': wrappers,
        'env': env, 'env_params': env_params,
        'obs_shape': env_spec.observation_shape, 'n_actions': n_actions,
        'eval_episode_cap': env_spec.eval_episode_cap,
        'state_hash': state_hash,
    }
    configured = partial(claim, **cell_kwargs)

    # Configurational fingerprint — the leaf measurements that
    # `aggregate.leaf_signature` projects to as the group-by key.
    # Walks the BOUND `configured` so intervention overrides
    # surface at their dotted topology paths. Post-Phase-A0
    # includes `env_name`, `seed` (default 0; per-cell value is
    # restamped below), `wrappers`, and `total_steps` alongside
    # the HP/slot-Claim leaves.
    leaf_measurements = _leaf_measurements(configured)

    # vmap over seeds (uint32 → dqn derives PRNGKey internally
    # post-Phase-A0). `seed` is the sole vmap dimension; all other
    # kwargs (env_name, wrappers, env, ...) are bound in
    # `configured`.
    def by_seed(seed: jax.Array) -> dict[str, jax.Array]:
        return configured(seed=seed)

    seeds_arr = jnp.asarray(seeds, dtype=jnp.uint32)
    # Wrap the vmap call in trace_context so JAX's first-call
    # abstract-trace pass fires @claim records once; that single
    # pass IS the structural graph (per-(theory, intervention),
    # constant across seeds). build_computation_graph derives the
    # static call graph from the records.
    with trace_context() as records:
        batched_record = jax.vmap(by_seed)(seeds_arr)
    graph = build_computation_graph(records)

    # Side-effect import: registers DDQN measurables (q_mean,
    # q_max, ..., pearson_r_online_target, late_window_mean) so
    # measurables declaring them as deps auto-resolve via the
    # registry. Substrate-side `dqn_default_measurables()` is
    # how authors enumerate the standard set on each Hypothesis.
    import corroborate_rl.dqn.measurables  # noqa: F401  # pyright: ignore[reportUnusedImport]
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

        # Pre-registered measurables: walk the typed `measurables`
        # tuple from the Hypothesis Protocol and persist each at
        # its bare measurable name. Names are bare (`jensen_gap`,
        # `eval_final_mean`, etc.) — substrate-paper-narrative
        # prefixes were normalised earlier.
        measurable_cols: dict[str, MeasurementLeaf] = {}
        for m in measurables:
            value = evaluate_with_measurables(
                m.fn, per_seed_record, cache=cache,
            )
            scalar = _leaf_scalar(value)
            measurable_cols[m.name] = scalar

        wrapper_cols: dict[str, MeasurementLeaf] = {}
        for w in wrappers:
            wrapper_cols.update(w.measurement_keys())
        # Post-Phase-A0: `env_name`, `seed`, `wrappers`,
        # `total_steps` are surfaced by `walk_paths` and live in
        # `leaf_measurements`. The per-cell `seed` value is
        # re-stamped here to override the partial's default-0
        # (vmap binds seed at call-time, not partial-bind-time, so
        # walk_paths sees the default; we restate the actual
        # per-cell value for the persisted column).
        measurements: dict[str, MeasurementLeaf] = {
            **leaf_measurements,
            'seed': seed,
            **wrapper_cols,
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
            verdict=verdict, arm_key=arm_key,
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
    claim: Callable[..., DQNTrajectoryRecord],
    arm_key: str,
    measurables: tuple[Measurable[DQNTrajectoryRecord, object], ...],
    *,
    cycle_id: str | None = None,
) -> CellResult:
    """Run one (env, seed, claim) cell. Thin convenience wrapper
    around `run_dqn_arm` for the single-seed case; multi-seed
    callers should use `run_dqn_arm` directly to avoid per-call
    vmap re-compilation. Discards the graph; callers that want
    it should use `run_dqn_arm` directly."""
    arm = run_dqn_arm(
        env_spec, (seed,), claim, arm_key, measurables,
        cycle_id=cycle_id,
    )
    return arm.cells[0]
