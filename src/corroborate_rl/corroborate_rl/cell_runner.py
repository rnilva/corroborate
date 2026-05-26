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

import os
import subprocess
import uuid
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from functools import lru_cache, partial
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from corroborate import trace_context
from corroborate.core import canonical_str
from corroborate.graph.computation import ComputationGraph, build_computation_graph
from corroborate_rl.dqn.dqn import default_state_hash
from corroborate_rl.dqn.invariants import DQNTrajectoryRecord
from corroborate_rl.dqn.q_checkpoint import (
    CHECKPOINT_KEY_PREFIX,
    QCheckpoint,
    checkpoint_path,
    parse_checkpoint_key,
    save as save_checkpoint,
)
from corroborate_rl.env_catalogue import EnvSpec, EnvWrapper, make_env
from corroborate.corpus.schema import MeasurementLeaf, RunRow, TraceLeaf, TraceRow
from corroborate.core.signature import walk, walk_paths
from corroborate.bridge.verdict import Verdict
from corroborate.measurables import Measurable


@lru_cache(maxsize=1)
def _git_head_sha() -> str | None:
    """Current substrate commit SHA, stamped on every RunRow this
    process emits. Cached at module load — git HEAD shouldn't change
    mid-sweep; if it did, a fresh runner invocation would re-read it.

    Returns `None` when the substrate isn't in a git checkout (e.g.,
    pip-install, source tarball). Bridges that need a specific
    substrate fix scope by `pl.col('substrate_commit_sha').is_in([
    sha1, sha2, ...])`; cells with `None` are pre-versioning and
    fall outside any such scope. See `docs/SUBSTRATE_FIXES.md`."""
    try:
        out = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL, text=True, timeout=2,
        ).strip()
        return out or None
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def _xla_deterministic_ops() -> bool:
    """Read the active `--xla_gpu_deterministic_ops` setting from
    `XLA_FLAGS` at cell-run time. Persisted in every RunRow's
    `measurements` so downstream consumers can filter / stratify
    by determinism mode (corpora mixing det / non-det runs are
    valid but the analyzer needs to know).

    Returns True iff `XLA_FLAGS` contains the substring
    `--xla_gpu_deterministic_ops=true`. Returns False if it
    contains `--xla_gpu_deterministic_ops=false` OR is silent on
    the flag (XLA's default is non-deterministic — see
    `corroborate_rl.dqn_sweep.set_jax_env`'s docstring for the
    perf / reproducibility trade-off this records)."""
    flags = os.environ.get('XLA_FLAGS', '')
    return '--xla_gpu_deterministic_ops=true' in flags


def _xla_command_buffer_enabled() -> bool:
    """Read the EFFECTIVE `--xla_gpu_enable_command_buffer` setting
    at cell-run time. CUDA Graph (command buffer) capture is the
    dominant systematic-bias source we measured — same env, same
    seeds, cmdbuf-on vs cmdbuf-off can flip the DDQN-vs-baseline
    outcome verdict on γ=0.999 MinAtar workloads (see
    REPRODUCIBILITY.md). Stamped per cell so cross-mode analyses
    can stratify / condition on this knob alongside
    `xla_deterministic_ops`.

    **Effective semantics** (XLA's actual runtime state, not the
    raw flag-string):

    1. **Explicit disable wins**: `--xla_gpu_enable_command_buffer=`
       (empty value) → False, regardless of det mode.
    2. **Explicit enable wins**: `--xla_gpu_enable_command_buffer=
       FUSION,...` (non-empty) → True, even when det=True (the
       explicit re-enable recovers cmdbuf from det's implicit
       disable).
    3. **det=True with no explicit cmdbuf flag**: XLA IMPLICITLY
       DISABLES cmdbuf in deterministic mode → False. This is the
       correction relative to a prior implementation that returned
       True here based on "explicit-disable-flag absent" alone.
    4. **det=False with no explicit cmdbuf flag**: XLA default ON
       (FUSION,CUSTOM_CALL,CUBLAS,CUDNN modes) → True. This is the
       worst-drift configuration the user empirically flagged.

    The corrected semantics matter for honest provenance: a
    `xla_command_buffer_enabled=True` row should mean the cmdbuf
    actually fired during training, not just "the disable flag was
    absent". Without this fix, det=True runs falsely advertised
    cmdbuf=True even though XLA disabled it at runtime."""
    flags = os.environ.get('XLA_FLAGS', '')
    # Step 1+2: explicit value (empty or non-empty) is authoritative.
    key = '--xla_gpu_enable_command_buffer='
    if key in flags:
        i = flags.index(key)
        rest = flags[i + len(key):]
        end = len(rest)
        for j, ch in enumerate(rest):
            if ch.isspace():
                end = j
                break
        return bool(rest[:end])
    # Step 3: no explicit flag → check whether det mode implicitly
    # disabled cmdbuf. `_xla_deterministic_ops` checks the same
    # XLA_FLAGS env var; reading both stays consistent.
    if _xla_deterministic_ops():
        return False
    # Step 4: no explicit flag, no det mode → XLA's default ON.
    return True


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
    construction.

    Sentinel-prefixed checkpoint keys (`__q_checkpoint__*`) are
    filtered out — their multi-dim parameter shapes don't fit
    polars `List` columns, and the cell runner persists them as
    msgpack sidecars via `_save_q_checkpoints` before this helper
    runs."""
    out: dict[str, TraceLeaf] = {}
    for key, arr in record.items():
        if key.startswith(CHECKPOINT_KEY_PREFIX):
            continue
        np_arr: np.ndarray = np.asarray(arr)
        if np_arr.ndim == 0:
            out[key] = np_arr.item()
        else:
            out[key] = np_arr
    return out


def _save_q_checkpoints(
    record: Mapping[str, jax.Array],
    *,
    q_checkpoint_dir: Path,
    cell_idx: int,
    seed: int,
) -> int:
    """Extract every `__q_checkpoint__<arm>__<role>__<param_key>`
    entry from the per-cell record, reconstruct the underlying
    `QCheckpoint` shapes, and write one msgpack file per snapshot
    via `q_checkpoint.save`.

    Per-burst payloads carry a leading `(n_super_steps, *param_shape)`
    axis; this function indexes that axis to emit one file per
    burst. Final payloads carry the bare `(*param_shape)` and emit
    one file with role `final`. Online + target params for the
    same snapshot share a file (the `QCheckpoint` record bundles
    both arms together).

    Returns the count of msgpack files written. Returns 0 when no
    sentinel keys are present (the off-by-default path — zero I/O,
    no parent-dir creation)."""
    # Partition checkpoint keys by (role, arm) so we can rebuild
    # the `Params` dicts before serialising.
    grouped: dict[
        tuple[str, str],  # (role, arm)
        dict[str, jax.Array],  # param_key -> array
    ] = {}
    for key, arr in record.items():
        parts = parse_checkpoint_key(key)
        if parts is None:
            continue
        bucket = grouped.setdefault((parts.role, parts.arm), {})
        bucket[parts.param_key] = arr

    if not grouped:
        return 0

    # `final` snapshots: both arms must be present (the producer
    # gates them with a single flag, so partial-emission is a bug).
    n_written = 0
    online_final = grouped.get(('final', 'online'))
    target_final = grouped.get(('final', 'target'))
    if online_final is not None and target_final is not None:
        # Use first leaf's leading dim as the global_step proxy —
        # actually we just pass `-1` and let the consumer derive
        # the step from training config. `burst=-1` is the
        # sentinel for "post-final-step" (no burst index applies).
        ckpt = QCheckpoint(
            online_params=dict(online_final),
            target_params=dict(target_final),
            burst=-1,
            global_step=-1,
        )
        save_checkpoint(
            checkpoint_path(
                q_checkpoint_dir, cell_idx=cell_idx, seed=seed,
                role='final',
            ),
            ckpt,
        )
        n_written += 1
    elif (online_final is None) != (target_final is None):
        # One arm without the other: producer-side bug. Surface
        # loudly rather than write a half-checkpoint that won't
        # round-trip.
        raise ValueError(
            'q_checkpoint final: one arm present, the other absent '
            f'(online={online_final is not None}, '
            f'target={target_final is not None}). Both arms must '
            'emit together — check train_with_eval gating.',
        )

    # `per_burst` snapshots: the leading axis is the burst index.
    online_pb = grouped.get(('per_burst', 'online'))
    target_pb = grouped.get(('per_burst', 'target'))
    if online_pb is not None and target_pb is not None:
        # All leaves under one arm share the same leading dim; we
        # peek at one to determine `n_super_steps`.
        first_online_leaf = next(iter(online_pb.values()))
        n_bursts = int(first_online_leaf.shape[0])
        for b in range(n_bursts):
            ckpt = QCheckpoint(
                online_params={k: v[b] for k, v in online_pb.items()},
                target_params={k: v[b] for k, v in target_pb.items()},
                burst=b,
                global_step=-1,
            )
            save_checkpoint(
                checkpoint_path(
                    q_checkpoint_dir, cell_idx=cell_idx, seed=seed,
                    role='per_burst', burst=b,
                ),
                ckpt,
            )
            n_written += 1
    elif (online_pb is None) != (target_pb is None):
        raise ValueError(
            'q_checkpoint per_burst: one arm present, the other '
            f'absent (online={online_pb is not None}, '
            f'target={target_pb is not None}). Both arms must emit '
            'together — check train_with_eval gating.',
        )

    return n_written


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
    q_checkpoint_dir: Path | None = None,
    cell_idx: int = 0,
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
    the body once).

    `q_checkpoint_dir` + `cell_idx`: when the bound claim's
    `keep_q_checkpoint_*` flags emit `__q_checkpoint__*` keys in
    the per-cell record, the runner extracts the param arrays and
    writes them to `<q_checkpoint_dir>/cell<NNN>_<seed>_*.msgpack`
    via `_save_q_checkpoints`. `cell_idx` is the framework-counted
    cell index (matches the `cell{cell_idx:03d}__{tag}` shard
    naming in `run_intervention`); the runner-side `DQNRunner` is
    responsible for threading the right value through. When
    `q_checkpoint_dir is None`, sentinel keys are still filtered
    from the trace but no checkpoint files are written —
    `train_with_eval` is expected to gate emission via its
    `keep_q_checkpoint_*` kwargs."""
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
        # Cardinality for the per-state visit-count array
        # (DQNState.state_hash_count). Falls back to 1 (single bucket)
        # for envs without a registered hash, in which case the count
        # is a global step counter — harmless for count-weighted-loss
        # interventions that just set α=0 in that scope.
        'state_hash_cardinality': (
            env_spec.state_hash_cardinality
            if env_spec.state_hash_cardinality is not None
            else 1
        ),
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

        # Persist Q-network checkpoints (one msgpack per snapshot)
        # BEFORE the trace-leaf projection — `_save_q_checkpoints`
        # reads from the un-filtered record, `_trajectory_leaves`
        # below drops sentinel-prefixed keys from the trace.
        # No-op when `q_checkpoint_dir is None` or no sentinel keys
        # are present (the default off path, no I/O).
        if q_checkpoint_dir is not None:
            _ = _save_q_checkpoints(
                per_seed_record,
                q_checkpoint_dir=q_checkpoint_dir,
                cell_idx=cell_idx,
                seed=int(seed),
            )

        # Per-cell measurable cache — shared across `hypothesis.
        # measurables` so dep-measurables (q_mean, q_std, etc.)
        # compute once per cell.
        cache: dict[str, object] = {}

        # Pre-registered measurables: walk the typed `measurables`
        # tuple from the Hypothesis Protocol and persist each at
        # its bare measurable name. Names are bare (`jensen_gap`,
        # `eval_final_mean`, etc.) — substrate-paper-narrative
        # prefixes were normalised earlier.
        #
        # Non-scalar (NDArray) returns are skipped: cell_runner
        # only persists scalar `MeasurementLeaf` (per the
        # persistence-shape rule in CLAUDE.md); array-returning
        # measurables are computed at ingestion time via
        # `build_measurements` from joined trace columns. Without
        # this skip, `_leaf_scalar` stringifies the array which
        # then traps `compute_missing_columns`'s partial-null
        # branch — non-null strings get preserved as "already
        # computed", silently shadowing the correct array value
        # that build_measurements would produce.
        measurable_cols: dict[str, MeasurementLeaf] = {}
        for m in measurables:
            value = evaluate_with_measurables(
                m.fn, per_seed_record, cache=cache,
            )
            if isinstance(value, (np.ndarray, jax.Array)) and value.ndim > 0:
                continue
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
            'xla_deterministic_ops': _xla_deterministic_ops(),
            'xla_command_buffer_enabled': _xla_command_buffer_enabled(),
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
            substrate_commit_sha=_git_head_sha(),
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
    q_checkpoint_dir: Path | None = None,
    cell_idx: int = 0,
) -> CellResult:
    """Run one (env, seed, claim) cell. Thin convenience wrapper
    around `run_dqn_arm` for the single-seed case; multi-seed
    callers should use `run_dqn_arm` directly to avoid per-call
    vmap re-compilation. Discards the graph; callers that want
    it should use `run_dqn_arm` directly."""
    arm = run_dqn_arm(
        env_spec, (seed,), claim, arm_key, measurables,
        cycle_id=cycle_id,
        q_checkpoint_dir=q_checkpoint_dir, cell_idx=cell_idx,
    )
    return arm.cells[0]
