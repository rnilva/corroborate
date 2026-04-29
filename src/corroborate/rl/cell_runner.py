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

import math
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from functools import partial
from typing import NamedTuple

import gymnax
import jax
import jax.numpy as jnp
import numpy as np

from corroborate.aggregate import (
    _bridge_result_to_measurements,
    aggregate_cell_verdict,
)
from corroborate.bridge import Bridge, BridgeResult
from corroborate._canonical import canonical_str
from corroborate.claim import trace_context
from corroborate.computation_graph import ComputationGraph, build_computation_graph
from corroborate.hypothesis import Hypothesis
from corroborate.reductions import masked_window_mean
from corroborate.rl.dqn.dqn import default_state_hash, dqn
from corroborate.rl.dqn.invariants import (
    DQNTrajectoryRecord,
    jensen_overestimation_gap,
)
from corroborate.rl.env_catalogue import EnvSpec
from corroborate.schema import MeasurementLeaf, RunRow, TraceLeaf, TraceRow
from corroborate.signature import collect_invariants, walk, walk_paths


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


def _mechanism_measurements(
    record: Mapping[str, jax.Array],
) -> dict[str, MeasurementLeaf]:
    """Per-cell mechanism-side measurements — theorem-gap scalars
    that DDQN's algorithmic intervention specifically targets.

    Currently:
    - `mechanism.jensen_gap` — `max(0, mean(predicted_q_at_start −
      mc_return))` over eval bursts. Hasselt 2010/2016: vanilla
      DQN's positive Jensen bias is what DDQN reduces by
      decoupling action selection (online) from value evaluation
      (target). Smaller is better.

    These scalars feed `paired_comparison_from_runs(outcome_path=
    'mechanism.jensen_gap', predicted_direction='a_lt_b')` to
    produce a per-env *mechanism* verdict — distinct from the
    *outcome* verdict on `outcome.late_window_mean`. The §3
    acceptance test wants both; their joint pattern (mechanism
    HELD ↛ outcome HELD ↛ link HELD) is the methodological claim."""
    out: dict[str, MeasurementLeaf] = {}
    if 'predicted_q_at_start' not in record or 'mc_return' not in record:
        return out
    gap = jensen_overestimation_gap()(record)
    if not math.isnan(gap):
        out['mechanism.jensen_gap'] = float(gap)
    return out


def _eval_outcomes(
    record: Mapping[str, jax.Array],
) -> dict[str, MeasurementLeaf]:
    """Extract robust eval-based outcome scalars from the per-cell
    record. Three reductions, each appropriate to a different
    research question on an unstable algorithm like DQN:

    - `outcome.eval_final_mean` — `mean(mc_return[-1, :])`. The
      LAST eval burst's mean MC return. Honest "final policy
      performance" — greedy, no exploration noise. Vulnerable to
      late-training instability (the network may have just had a
      bad gradient).
    - `outcome.eval_best_burst_mean` — `max_i(mean(mc_return[i, :]))`.
      The best burst seen during training. Robust to instability,
      slightly optimistic but standard for unstable-RL evaluation.
    - `outcome.eval_best_burst_step` — provenance: which training
      step produced the best burst. Lets consumers see whether
      'best' is at convergence or an early lucky checkpoint.

    All three computed cheaply from the eval-burst arrays already
    in the record. The trace store carries the raw arrays for any
    further post-hoc reduction; this helper bakes the standard
    three so `paired_comparison_from_runs(outcome_path=...)` can
    pick a reduction without cracking open the trace store."""
    out: dict[str, MeasurementLeaf] = {}
    if 'mc_return' not in record:
        return out
    mc = record['mc_return']  # (n_super_steps, K)
    if mc.ndim != 2 or mc.size == 0:
        return out
    burst_means = jnp.mean(mc, axis=1)        # (n_super_steps,)
    out['outcome.eval_final_mean'] = float(burst_means[-1])
    best_idx = int(jnp.argmax(burst_means))
    out['outcome.eval_best_burst_mean'] = float(burst_means[best_idx])
    if 'eval_step_index' in record:
        eval_steps = record['eval_step_index']
        if eval_steps.ndim == 1 and eval_steps.size > best_idx:
            out['outcome.eval_best_burst_step'] = int(eval_steps[best_idx])
    return out


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
        np_arr: np.ndarray = np.asarray(arr)  # pyright: ignore[reportMissingTypeArgument]
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
    cell_kwargs: dict[str, object] = {
        'env': env, 'env_params': env_params,
        'obs_shape': env_spec.observation_shape, 'n_actions': env_spec.n_actions,
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

    # Late-window 10% of training is the codebase's standard outcome
    # reduction. Researchers wanting a different window should
    # author a different `outcome.<name>` reduction (e.g.
    # `outcome.mid_window_mean`) rather than tweak this constant —
    # different windows aren't the same outcome.
    outcome_proj = masked_window_mean(
        value_key='ep_return', mask_key='done', fraction=0.1,
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

    # Side-effect import: registers DDQN measurables (q_mean,
    # q_max, ..., pearson_r_online_target) so bridges declaring
    # them as deps auto-resolve via the registry.
    import corroborate.rl.dqn.measurables  # noqa: F401
    from corroborate.measurable import evaluate_with_measurables

    cells: list[CellResult] = []
    for i, seed in enumerate(seeds):
        per_seed_record: dict[str, jax.Array] = {
            k: v[i] for k, v in batched_record.items()
        }
        outcome = outcome_proj(per_seed_record)
        # Per-cell measurable cache — shared across all bridges
        # in this cell so each measurable computes at most once.
        cache: dict[str, object] = {}
        bridge_results = tuple(
            evaluate_with_measurables(b.fn, per_seed_record, cache=cache)
            for b in effective_bridges
        )
        verdict = aggregate_cell_verdict(
            tuple(r.verdict for r in bridge_results),
        )

        measurements: dict[str, MeasurementLeaf] = {
            'intervention_name': hypothesis.name,
            'env_name': env_spec.name,
            'seed': seed,
            'total_steps': total_steps,
            'outcome.late_window_mean': outcome,
            **_eval_outcomes(per_seed_record),
            **_mechanism_measurements(per_seed_record),
            **leaf_measurements,
        }
        for result in bridge_results:
            measurements.update(_bridge_result_to_measurements(result))

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
