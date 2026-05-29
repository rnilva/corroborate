"""Re-evaluate saved per-burst Q-checkpoints at a chosen
`n_episodes`, producing a NEW corpus that ingests through the
canonical pipeline.

**Why this exists.** A canonical `*_ckpt` sweep persists, per
(arm, seed, burst), the online Q-network params AND a single
greedy eval rollout (`n_episodes=1`). Some envs were swept at
`n_episodes=1` to keep wall-clock down (Breakout / Freeway /
SpaceInvaders γ=0.99). Their per-burst policies ARE trained and
checkpointed, so the eval estimate can be sharpened to
`n_episodes=20` WITHOUT retraining — only the greedy eval
rollout is re-rolled at the new K. Training is byte-identical
(same checkpoints), so every training-derived trace column is
copied verbatim; only the eval-derived columns are recomputed.

**The eval-derived vs training-derived split (EXACT).** The
eval-derived set is precisely the six `EvalBurstOut` fields
emitted by `eval_burst` — `predicted_q_at_start`, `mc_return`,
`episode_length`, `predicted_q_per_step`, `mc_return_from_step`,
`active_per_step`. Everything else in the original traces schema
(`reward`, `done`, `td_error`, `online_*_per_step`,
`target_*_per_step`, `state_hash_per_step`, `pearson_stats`,
`eval_step_index`, …) is training-derived and copied byte-for-
byte from the source corpus's `traces.parquet`.

**Eval seeding (PAIRED, locked design).** Each (train_seed,
burst) gets a deterministic eval RNG
`fold_in(PRNGKey(eval_seed_base), train_seed*_SEED_STRIDE + burst)`
that is SHARED across arms — V and D at the same (train_seed,
burst) evaluate on the SAME `n_episodes` env instances, so the
paired V−D difference has lower variance. The arm is NOT folded
into the key. This `eval_keying='paired'` is the default for
production output.

`eval_keying='original'` reproduces the canonical per-seed
scheme `fold_in(split(PRNGKey(seed), 2)[1], burst)` that the
training-time `dqn` loop used (`dqn.eval_fn`). It exists for the
round-trip correctness test (re-eval at `n_episodes=1` with the
original keying must reproduce the source corpus's eval traces
bit-for-bit) — it is NOT the production scheme.

**Checkpoint layout.** Supports both the legacy per-file layout
(`<subdir>/cell{NNN}_{seed}_burst{BB}.msgpack`) and the bundle
layout (`<subdir>/cell{NNN}.msgpack`). Layout is auto-detected
from the checkpoint directory contents. The canonical
`*_n_eps1_ckpt` corpora (Breakout / Freeway) use the per-file
layout.

**Cell→arm→seed mapping.** A run row carries `(id, arm_key,
seed)`; a checkpoint cell carries `(cell_idx, seeds)`. The map
from run row to checkpoint slot is recovered from the DATA, not
the fragile `cell_idx = chunk*n_arms + arm` convention: for each
checkpoint cell we compute `predicted_q_at_start` at burst 0
under the original eval key and match it (per seed) against the
source trace's burst-0 value. The exact float match (max-Q at a
deterministic reset state is identical given identical params)
disambiguates which arm each checkpoint cell belongs to. This
makes the mapping self-verifying — a mismatch raises rather than
silently mis-pairing arms."""
from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, TypeIs

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
from gymnax import EnvParams

from corroborate.corpus.cloud import RemoteFile, RemoteManifest, load_manifest
from corroborate.corpus.persistence import atomic_write_parquet, atomic_write_text
from corroborate_rl.dqn.claims.q_network import CNN, MLP, Params, QFunction
from corroborate_rl.dqn.eval import EvalBurstOut, eval_burst
from corroborate_rl.dqn.q_checkpoint import checkpoint_path
from corroborate_rl.dqn.q_checkpoint import load as load_qcheckpoint
from corroborate_rl.dqn.q_checkpoint_bundle import (
    QCheckpointBundle,
    bundle_path,
    extract_qcheckpoint,
    load_bundle,
)
from corroborate_rl.env_catalogue import get as get_env_spec, make_env

if TYPE_CHECKING:
    # `Env` is a stub-only Protocol — gymnax's runtime exposes
    # `Environment`, not `Env`. With `from __future__ import
    # annotations`, the annotation is stringified and never resolved
    # at runtime; pyright sees the typed surface from the stub. Same
    # rationale as `env_catalogue.py`'s TYPE_CHECKING import.
    from gymnax import Env


# ============ The exact eval-derived column set ============

EVAL_DERIVED_COLUMNS: Final[frozenset[str]] = frozenset(EvalBurstOut._fields)
"""The six trace columns recomputed by a re-eval — exactly the
`EvalBurstOut` NamedTuple fields. Every OTHER trace column is
training-derived and copied verbatim from the source corpus.

Pinned to `EvalBurstOut._fields` (not a hand-listed tuple) so the
split auto-tracks the eval record shape: adding a field to
`EvalBurstOut` extends this set without a second edit site."""


type EvalKeying = Literal['paired', 'original']
"""`paired` — production scheme: deterministic eval key per
(seed, burst), SHARED across arms (lower-variance V−D).
`original` — reproduces the training-time per-seed scheme for
the round-trip correctness test."""


_SEED_STRIDE: Final[int] = 1_000_003
"""Multiplier mixing `train_seed` into the paired eval key's
fold-in counter (`seed*_SEED_STRIDE + burst`). A large prime keeps
distinct (seed, burst) pairs from colliding in the fold-in integer
for the burst counts (≤ a few hundred) and seed counts (≤ a few
hundred) the substrate runs."""


def eval_key(
    *, keying: EvalKeying, eval_seed_base: int, seed: int, burst: int,
) -> jax.Array:
    """The per-(seed, burst) eval RNG key for one rollout burst.

    `paired`: `fold_in(PRNGKey(eval_seed_base), seed*_SEED_STRIDE +
    burst)` — independent of arm, so paired across arms.

    `original`: `fold_in(split(PRNGKey(seed), 2)[1], burst)` — the
    training loop's `run_key`-derived scheme (`dqn.eval_fn`), which
    re-rolls the canonical eval bit-for-bit at the same K."""
    if keying == 'original':
        _, run_key = jax.random.split(jax.random.PRNGKey(seed), 2)
        return jax.random.fold_in(run_key, burst)
    return jax.random.fold_in(
        jax.random.PRNGKey(eval_seed_base), seed * _SEED_STRIDE + burst,
    )


# ============ Config → objects reconstruction ============

def _parse_int_tuple(s: str) -> tuple[int, ...]:
    """Parse polars' tuple-string render (`'(16)'`, `'(10,10,4)'`,
    `'(128,)'`) back into `tuple[int, ...]`. Handles the single-
    element no-trailing-comma form polars emits."""
    inner = s.strip().removeprefix('(').removesuffix(')').strip()
    if not inner:
        return ()
    return tuple(int(p.strip()) for p in inner.split(',') if p.strip())


def _is_finite_str(v: object) -> TypeIs[str]:
    """A config cell carries a meaningful string (non-null,
    non-empty). Polars null-pads absent leaves across heterogeneous
    rows; an MLP corpus has no `q_network.channels`, so the column
    is either absent or all-null."""
    return isinstance(v, str) and bool(v.strip())


@dataclass(frozen=True, slots=True)
class CellConfig:
    """The homogeneous-across-cells configuration a re-eval needs to
    rebuild env + Q-network. Asserted identical across every run row
    of one source corpus (a re-eval re-rolls one env / network for
    the whole corpus; heterogeneous config would need per-cell
    rebuilds and is out of scope — raise instead)."""
    env_name: str
    gamma: float
    n_actions: int
    eval_episode_cap: int
    q_network: QFunction


def _build_q_network(row: Mapping[str, object]) -> QFunction:
    """Reconstruct the Q-network bundle from the leaf-walk columns.

    Discriminator: a CNN bundle surfaces `q_network.channels` /
    `q_network.kernel_size` / `q_network.obs_shape` leaves; an MLP
    surfaces only `q_network.hidden`. Reading the column presence
    (finite string) is the type-honest detector — the substrate
    walks the bundle's fields into these dotted paths at sweep
    time."""
    channels_raw = row.get('q_network.channels')
    hidden_raw = row.get('q_network.hidden')
    if not _is_finite_str(hidden_raw):
        raise ValueError(
            'reeval: runs.parquet row missing q_network.hidden — '
            'cannot reconstruct the Q-network. Columns present: '
            f'{sorted(k for k in row if k.startswith("q_network"))}',
        )
    hidden = _parse_int_tuple(hidden_raw)
    if _is_finite_str(channels_raw):
        obs_shape_raw = row.get('q_network.obs_shape')
        kernel_raw = row.get('q_network.kernel_size')
        if not _is_finite_str(obs_shape_raw):
            raise ValueError(
                'reeval: CNN config has q_network.channels but no '
                'q_network.obs_shape',
            )
        if not isinstance(kernel_raw, int) or isinstance(kernel_raw, bool):
            raise ValueError(
                'reeval: CNN config q_network.kernel_size must be int; '
                f'got {type(kernel_raw).__name__}',
            )
        return CNN(
            obs_shape=_parse_int_tuple(obs_shape_raw),
            channels=_parse_int_tuple(channels_raw),
            kernel_size=kernel_raw,
            hidden=hidden,
        )
    return MLP(hidden=hidden)


def _require_float(row: Mapping[str, object], key: str) -> float:
    v = row.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(
            f'reeval: runs.parquet missing numeric {key!r} '
            f'(got {type(v).__name__})',
        )
    return float(v)


def _require_int(row: Mapping[str, object], key: str) -> int:
    v = row.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(
            f'reeval: runs.parquet missing integer {key!r} '
            f'(got {type(v).__name__})',
        )
    return int(v)


def _build_cell_config(runs: pl.DataFrame) -> CellConfig:
    """Project the config columns, assert homogeneity, and build the
    env-name / network / cap bundle. The env itself is built lazily
    by the caller (it isn't picklable / hashable into this frozen
    record)."""
    required = (
        'env_name', 'gamma', 'n_actions', 'eval_episode_cap',
        'q_network.hidden',
    )
    for col in required:
        if col not in runs.columns:
            raise ValueError(
                f'reeval: runs.parquet has no {col!r} column — cannot '
                'reconstruct the cell config.',
            )
    # Homogeneity over the config leaves (everything that defines the
    # env + network + cap). `arm_key` / `seed` / `id` legitimately
    # vary; the q_network + env + cap must not.
    config_cols = [
        c for c in runs.columns
        if c.startswith('q_network.')
        or c in ('env_name', 'gamma', 'n_actions', 'eval_episode_cap')
    ]
    uniq = runs.select(config_cols).unique()
    if uniq.height != 1:
        raise ValueError(
            'reeval: heterogeneous config across cells (env / network / '
            f'cap not uniform):\n{uniq}',
        )
    # Env wrappers change eval dynamics (reward scaling, action
    # inflation, ep-cap, …). Re-eval rebuilds the bare env from the
    # catalogue; reconstructing the wrapper chain from leaf columns
    # is out of scope. Refuse a wrapped corpus rather than silently
    # evaluating on the unwrapped env. `'()'` is polars' render of
    # the empty wrapper tuple.
    if 'wrappers' in runs.columns:
        wrapper_vals = {
            str(v) for v in runs.get_column('wrappers').unique().to_list()
        }
        if wrapper_vals - {'()'}:
            raise ValueError(
                'reeval: source corpus has env wrappers '
                f'{sorted(wrapper_vals)} — re-eval rebuilds the bare '
                'env and would evaluate on different dynamics. Wrapped '
                'corpora are out of scope.',
            )
    row: dict[str, object] = runs.row(0, named=True)
    return CellConfig(
        env_name=str(row['env_name']),
        gamma=_require_float(row, 'gamma'),
        n_actions=_require_int(row, 'n_actions'),
        eval_episode_cap=_require_int(row, 'eval_episode_cap'),
        q_network=_build_q_network(row),
    )


# ============ Checkpoint directory discovery ============

@dataclass(frozen=True, slots=True)
class CheckpointSource:
    """Resolved per-cell checkpoint accessor — abstracts the
    per-file vs bundle on-disk layout behind one typed
    `params_for(cell_idx, seed, burst)`.

    `cell_seeds[cell_idx]` is the seed list that checkpoint cell
    covers (a single chunk; multi-chunk corpora spread seeds over
    several cells). `n_bursts` is the per-burst snapshot count."""
    base_dir: Path
    layout: Literal['per_file', 'bundle']
    cell_seeds: Mapping[int, tuple[int, ...]]
    n_bursts: int

    def load_cell(self, cell_idx: int) -> CellCheckpoints:
        """Materialise a per-cell accessor.

        For the bundle layout this reads the (multi-GB) per-cell
        msgpack ONCE — the returned accessor slices in-memory for
        every (seed, burst), avoiding the re-decode-per-burst cost a
        per-call `load_bundle` would incur. For the per-file layout
        each (seed, burst) is its own small file, so the accessor
        loads lazily on access (no benefit to pre-reading all 50×15
        files of a cell into memory at once)."""
        if self.layout == 'bundle':
            bundle = load_bundle(bundle_path(self.base_dir, cell_idx=cell_idx))
            return CellCheckpoints(
                base_dir=self.base_dir, cell_idx=cell_idx, _bundle=bundle,
            )
        return CellCheckpoints(
            base_dir=self.base_dir, cell_idx=cell_idx, _bundle=None,
        )


@dataclass(frozen=True, slots=True)
class CellCheckpoints:
    """One checkpoint cell's params accessor — bundle held in memory
    (bundle layout) or lazily read per file (per-file layout). Built
    via `CheckpointSource.load_cell` so the bundle decode happens
    once per cell."""
    base_dir: Path
    cell_idx: int
    _bundle: QCheckpointBundle | None

    def online_params(self, *, seed: int, burst: int) -> Params:
        """Online Q-params for one (seed, burst) of this cell."""
        bundle = self._bundle
        if bundle is not None:
            ckpt = extract_qcheckpoint(
                bundle, seed=seed, role='per_burst', burst=burst,
            )
            return ckpt.online_params
        # `checkpoint_path` carries the legacy-layout @deprecated
        # marker; it remains the canonical path builder for the
        # per-file corpora a re-eval consumes (Breakout / Freeway
        # n_eps1), so the call is intentional. Suppress the per-call
        # DeprecationWarning locally — a re-eval makes thousands of
        # these (one per (seed, burst)); the warning is for NEW code,
        # which this read path is not.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            path = checkpoint_path(
                self.base_dir, cell_idx=self.cell_idx, seed=seed,
                role='per_burst', burst=burst,
            )
        return load_qcheckpoint(path).online_params


_PER_FILE_RE: Final[re.Pattern[str]] = re.compile(
    r'cell(\d+)_(\d+)_burst(\d+)\.msgpack$',
)
_BUNDLE_RE: Final[re.Pattern[str]] = re.compile(r'cell(\d+)\.msgpack$')


def discover_checkpoint_source(q_checkpoints_dir: Path) -> CheckpointSource:
    """Walk a corpus's `q_checkpoints/` tree (one subdir per
    arm-config, or a single subdir for a single-config sweep) and
    resolve the layout + per-cell seed coverage + burst count.

    Raises `FileNotFoundError` if no checkpoint files are present
    (the corpus's bundles weren't restored from cloud yet) and
    `ValueError` on a mixed / ambiguous layout."""
    if not q_checkpoints_dir.is_dir():
        raise FileNotFoundError(
            f'reeval: no q_checkpoints dir at {q_checkpoints_dir}',
        )
    per_file_cells: dict[int, set[int]] = {}
    per_file_bursts: set[int] = set()
    bundle_cells: set[int] = set()
    base_dir: Path | None = None
    for path in sorted(q_checkpoints_dir.rglob('*.msgpack')):
        name = path.name
        m_pf = _PER_FILE_RE.search(name)
        if m_pf is not None:
            cell, seed, burst = (int(m_pf.group(i)) for i in (1, 2, 3))
            per_file_cells.setdefault(cell, set()).add(seed)
            per_file_bursts.add(burst)
            base_dir = path.parent
            continue
        m_b = _BUNDLE_RE.search(name)
        if m_b is not None:
            bundle_cells.add(int(m_b.group(1)))
            base_dir = path.parent
    if base_dir is None:
        raise FileNotFoundError(
            f'reeval: no checkpoint .msgpack files under '
            f'{q_checkpoints_dir} — restore the q_checkpoint bundles '
            'from cloud first (corroborate restore / recover_local_'
            'manifest).',
        )
    if per_file_cells and bundle_cells:
        raise ValueError(
            f'reeval: mixed checkpoint layout under {q_checkpoints_dir} '
            '(both per-file and bundle .msgpack present) — ambiguous.',
        )
    if per_file_cells:
        return CheckpointSource(
            base_dir=base_dir, layout='per_file',
            cell_seeds={
                c: tuple(sorted(s)) for c, s in per_file_cells.items()
            },
            n_bursts=len(per_file_bursts),
        )
    # Bundle layout: read each bundle's typed seeds + n_bursts.
    cell_seeds: dict[int, tuple[int, ...]] = {}
    n_bursts_seen: set[int] = set()
    for cell in sorted(bundle_cells):
        bundle = load_bundle(bundle_path(base_dir, cell_idx=cell))
        cell_seeds[cell] = bundle.seeds
        n_bursts_seen.add(bundle.n_bursts)
    if len(n_bursts_seen) != 1:
        raise ValueError(
            f'reeval: bundles under {q_checkpoints_dir} disagree on '
            f'n_bursts ({sorted(n_bursts_seen)})',
        )
    return CheckpointSource(
        base_dir=base_dir, layout='bundle',
        cell_seeds=cell_seeds, n_bursts=next(iter(n_bursts_seen)),
    )


# ============ Run-row → checkpoint-cell mapping ============

@dataclass(frozen=True, slots=True)
class RowCheckpointMap:
    """Per run-row resolution of (cell_idx, seed) for checkpoint
    lookup, plus the burst count to re-roll. `by_id[run_id] =
    (cell_idx, seed)`."""
    by_id: Mapping[str, tuple[int, int]]
    n_bursts: int


def _probe_bursts(n_bursts: int, *, k: int = 6) -> tuple[int, ...]:
    """Pick up to `k` burst indices spread across [0, n_bursts) for
    the arm-disambiguation fingerprint. Burst 0 alone is degenerate
    (two arms share early-training Q); a spread of LATE bursts (where
    DDQN's clip has diverged the arms) plus a couple of early ones
    distinguishes arms while keeping the mapping's forward-pass count
    O(k) per (cell, seed) instead of O(n_bursts) — the cumulative
    eager-forward cost was a memory-pressure source on 50-burst
    corpora. Always includes the LAST burst (max divergence)."""
    if n_bursts <= k:
        return tuple(range(n_bursts))
    # Evenly spaced including endpoints (0 .. n_bursts-1).
    idxs = {round(i * (n_bursts - 1) / (k - 1)) for i in range(k)}
    idxs.add(n_bursts - 1)
    return tuple(sorted(idxs))


def _trace_predicted_q_at_bursts(
    traces: pl.DataFrame, run_id: str, bursts: Sequence[int],
) -> np.ndarray:
    """The source trace's `predicted_q_at_start[b, 0]` at the probe
    `bursts` (episode-0 max-Q at each burst's start state). The
    cell's Q-evolution fingerprint at those bursts — distinct across
    arms once their training diverges."""
    sub = traces.filter(pl.col('id') == run_id)
    if sub.height != 1:
        raise ValueError(
            f'reeval: expected exactly one trace row for id={run_id}, '
            f'got {sub.height}',
        )
    raw = sub.get_column('predicted_q_at_start').to_list()[0]
    arr = np.asarray(raw, dtype=np.float64)
    if arr.size == 0:
        raise ValueError(
            f'reeval: empty predicted_q_at_start for id={run_id}',
        )
    per_burst = arr.reshape(arr.shape[0], -1)[:, 0]  # (n_bursts,)
    return per_burst[np.asarray(bursts, dtype=np.intp)]


def build_row_checkpoint_map(
    runs: pl.DataFrame,
    traces: pl.DataFrame,
    source: CheckpointSource,
    cfg: CellConfig,
    *,
    match_tol: float = 1e-3,
) -> RowCheckpointMap:
    """Resolve every run row to its checkpoint (cell_idx, seed) by
    DATA, not by the fragile `cell_idx = chunk*n_arms + arm`
    convention.

    For each (checkpoint cell, seed) we recompute `max_a Q(s_0^b)` at
    a SPREAD of probe bursts under the burst-b params at the burst's
    canonical (original-keyed) reset state, and match that vector
    against the source trace's `predicted_q_at_start[b, 0]` of the
    run rows with that seed. The probe-burst spread (not just burst
    0, which two arms share early in training before DDQN's clip
    diverges them) uniquely identifies the arm. `traces` here is the
    `id` + `predicted_q_at_start`-only projection (the heavy training
    columns are never materialised for the mapping).

    Refuses to proceed if a (cell, seed) can't be matched to a run
    row within `match_tol` — a silent mis-pair would scramble V/D."""
    env, env_params = make_env(get_env_spec(cfg.env_name))
    q_net = cfg.q_network
    n_bursts = source.n_bursts
    bursts = _probe_bursts(n_bursts)

    # `max_a Q(params, obs)` jitted ONCE so XLA compiles a single
    # executable reused across every (cell, seed, burst) probe —
    # avoids the per-eager-call compilation churn that accumulated
    # device memory on 50-burst corpora.
    @jax.jit
    def _max_q(params: Params, obs: jax.Array) -> jax.Array:
        return jnp.max(q_net(params, obs))

    # Index source rows + their Q-fingerprint at the probe bursts.
    rows_by_seed: dict[int, list[tuple[str, str, np.ndarray]]] = {}
    for r in runs.iter_rows(named=True):
        seed = _require_int(r, 'seed')
        run_id = str(r['id'])
        arm_key = str(r['arm_key'])
        traj = _trace_predicted_q_at_bursts(traces, run_id, bursts)
        rows_by_seed.setdefault(seed, []).append((run_id, arm_key, traj))

    def predicted_q_trajectory(cell: CellCheckpoints, seed: int) -> np.ndarray:
        # Recompute max_a Q(s_0^b) at the probe bursts under the
        # burst-b params. The canonical burst-b reset uses the
        # ORIGINAL eval key (how the source trace's value was
        # produced); eval_burst splits then resets per episode, so
        # episode 0's reset key is split(key, K)[0] → reset =
        # split(., 2)[0].
        out = np.empty(len(bursts), dtype=np.float64)
        for i, burst in enumerate(bursts):
            params = cell.online_params(seed=seed, burst=burst)
            key_b = eval_key(
                keying='original', eval_seed_base=0, seed=seed, burst=burst,
            )
            ep_keys = jax.random.split(key_b, 1)
            reset_key, _run = jax.random.split(ep_keys[0])
            obs0, _state = env.reset(reset_key, env_params)
            out[i] = float(_max_q(params, obs0))
        return out

    def l_inf(a: np.ndarray, b: np.ndarray) -> float:
        if a.shape != b.shape:
            return float('inf')
        return float(np.max(np.abs(a - b)))

    # Per-seed bipartite assignment: each (cell, seed) is matched to
    # the NEAREST unused run-row (with that seed) by L∞ over the
    # per-burst Q-trajectory, requiring the best match within
    # `match_tol`. Greedy nearest-match resolves ties harmlessly:
    # when two arms share an identical trajectory (e.g. before DDQN's
    # clip diverges them) their checkpoint params are identical, so
    # whichever run-row each cell binds to yields the SAME re-eval
    # output — the bijection is what matters, not which identical
    # twin gets which. Only a >tol residual (no run-row reproduces
    # the cell's Q-trajectory) is a real error.
    by_id: dict[str, tuple[int, int]] = {}
    used_ids: set[str] = set()
    for cell_idx, seeds in source.cell_seeds.items():
        cell = source.load_cell(cell_idx)
        for seed in seeds:
            recomputed = predicted_q_trajectory(cell, seed)
            best_id: str | None = None
            best_dist = float('inf')
            for (run_id, _ak, traj) in rows_by_seed.get(seed, []):
                if run_id in used_ids:
                    continue
                d = l_inf(traj, recomputed)
                if d < best_dist:
                    best_dist, best_id = d, run_id
            if best_id is None or best_dist > match_tol:
                raise ValueError(
                    f'reeval: cell {cell_idx} seed {seed} has no unused '
                    f'run row whose per-burst predicted_q trajectory '
                    f'matches within tol={match_tol} (best L∞='
                    f'{best_dist:.6g}). The Q-trajectory fingerprint '
                    'should identify the (arm, seed) of a checkpoint '
                    'cell — a >tol residual means the checkpoints or '
                    'the source traces are inconsistent.',
                )
            by_id[best_id] = (cell_idx, seed)
            used_ids.add(best_id)

    missing = set(runs.get_column('id').to_list()) - set(by_id)
    if missing:
        raise ValueError(
            f'reeval: {len(missing)} run rows had no checkpoint match '
            f'(e.g. {sorted(missing)[:3]}). Checkpoint cells cover '
            f'seeds {sorted({s for ss in source.cell_seeds.values() for s in ss})}.',
        )
    return RowCheckpointMap(by_id=by_id, n_bursts=n_bursts)


# ============ Per-cell re-eval ============

type EvalBurstFn = Callable[[Params, jax.Array], EvalBurstOut]
"""A `(online_params, rng_key) -> EvalBurstOut` closure — env /
q_network / gamma / cap / n_episodes baked in, jitted ONCE."""


def _make_eval_burst_fn(
    *,
    env: Env,
    env_params: EnvParams,
    q_net: QFunction,
    gamma: float,
    episode_cap: int,
    n_episodes: int,
) -> EvalBurstFn:
    """Build a `jax.jit`'d `eval_burst` closure with everything
    except `(online_params, rng_key)` baked in.

    Jitting ONCE is load-bearing for the full-corpus loop: a re-eval
    makes `n_cells × n_bursts` calls (Breakout: 60 × 50 = 3000). The
    un-jitted `eval_burst` re-traces the inner vmap+scan PER CALL
    (it closes over `online_params`), and the CPU XLA backend mmaps
    a fresh executable section each trace — thousands of distinct
    dylibs exhaust the contiguous executable-memory arena
    (`LLVM ERROR: Unable to allocate section memory`). With params +
    key as JITTED ARGS, XLA compiles a single executable and reuses
    it for every (cell, seed, burst)."""
    def _run(online_params: Params, rng_key: jax.Array) -> EvalBurstOut:
        return eval_burst(
            online_params=online_params,
            env=env, env_params=env_params,
            rng_key=rng_key,
            q_network=q_net,
            gamma=gamma,
            episode_cap=episode_cap,
            n_episodes=n_episodes,
        )
    return jax.jit(_run)


def _reeval_one_cell(
    *,
    cell: CellCheckpoints,
    eval_fn: EvalBurstFn,
    seed: int,
    n_bursts: int,
    eval_seed_base: int,
    keying: EvalKeying,
) -> dict[str, np.ndarray]:
    """Re-roll greedy eval for all bursts of one (cell, seed) and
    stack to `(n_bursts, n_episodes, ...)` per `EvalBurstOut` field
    (the `n_episodes` axis is fixed by `eval_fn`'s baked-in K).

    `eval_fn` is the jit-once `(params, key) -> EvalBurstOut` closure
    from `_make_eval_burst_fn` — reused across every burst so the
    eval kernel compiles exactly once for the whole corpus."""
    per_burst: list[EvalBurstOut] = []
    for burst in range(n_bursts):
        params = cell.online_params(seed=seed, burst=burst)
        key = eval_key(
            keying=keying, eval_seed_base=eval_seed_base,
            seed=seed, burst=burst,
        )
        per_burst.append(eval_fn(params, key))

    # Stack per field explicitly — no `getattr` (NamedTuple fields are
    # statically known; the typing discipline forbids attribute
    # erasure). Each leaf gains the leading n_bursts axis.
    def _stack(leaves: Sequence[jax.Array]) -> np.ndarray:
        return np.stack([np.asarray(x) for x in leaves], axis=0)

    return {
        'predicted_q_at_start': _stack(
            [b.predicted_q_at_start for b in per_burst]),
        'mc_return': _stack([b.mc_return for b in per_burst]),
        'episode_length': _stack([b.episode_length for b in per_burst]),
        'predicted_q_per_step': _stack(
            [b.predicted_q_per_step for b in per_burst]),
        'mc_return_from_step': _stack(
            [b.mc_return_from_step for b in per_burst]),
        'active_per_step': _stack([b.active_per_step for b in per_burst]),
    }


def reeval_corpus(
    corpus_dir: Path,
    *,
    n_episodes: int,
    out_dir: Path,
    eval_seed_base: int = 0,
    eval_keying: EvalKeying = 'paired',
    q_checkpoints_subdir: str = 'q_checkpoints',
) -> Path:
    """Re-evaluate `corpus_dir`'s per-burst Q-checkpoints at
    `n_episodes`, writing a NEW corpus at `out_dir`.

    Reads `corpus_dir/runs.parquet` + `corpus_dir/traces.parquet`
    (both must be local — restore from cloud first if evicted) and
    `corpus_dir/<q_checkpoints_subdir>/` (per-file or bundle
    layout). For every run row:

    1. Resolve its (cell_idx, seed) checkpoint slot by matching the
       recomputed per-burst predicted-Q TRAJECTORY against the
       source trace (self-verifying arm assignment — see
       `build_row_checkpoint_map`).
    2. Re-roll greedy `eval_burst` at `n_episodes` for each burst,
       with the paired (or original) eval key.
    3. Stack bursts → eval traces `(n_bursts, n_episodes, ...)`.
    4. Replace the six eval-derived columns; copy every other
       column verbatim.
    5. Restamp `n_episodes` in runs.parquet.

    Writes `out_dir/{runs,traces}.parquet`. Preserves the original
    `id` / lineage columns so the new corpus cross-references the
    source and ingests cleanly via `corroborate hypothesis
    --ingest`. Mirrors the source `_remote.json` (q-checkpoint
    references at the source remote root) so the new corpus stays
    restorable / archivable.

    Memory-bounded for multi-GB source traces: only the six eval
    columns are held in memory; the heavy training columns stream
    source→dest in Arrow via a lazy scan + sink (never materialised
    as Python objects).

    Returns `out_dir`."""
    if n_episodes < 1:
        raise ValueError(f'reeval: n_episodes must be >= 1; got {n_episodes}')
    runs_path = corpus_dir / 'runs.parquet'
    traces_path = corpus_dir / 'traces.parquet'
    if not runs_path.is_file():
        raise FileNotFoundError(f'reeval: no runs.parquet at {runs_path}')
    if not traces_path.is_file():
        raise FileNotFoundError(
            f'reeval: no traces.parquet at {traces_path} — restore the '
            'source corpus traces from cloud first.',
        )

    runs = pl.read_parquet(runs_path)
    cfg = _build_cell_config(runs)
    source = discover_checkpoint_source(corpus_dir / q_checkpoints_subdir)

    # Memory discipline: the source traces.parquet can be multi-GB
    # (Breakout / Freeway ~3 GB). NEVER materialise its heavy
    # training-derived columns through Python. The mapping needs only
    # `id` + `predicted_q_at_start` (tiny) — project those; the
    # column copy happens via a lazy scan → join → sink that keeps
    # the training columns in Arrow the whole way.
    traces_schema = set(pl.scan_parquet(traces_path).collect_schema().names())
    missing_eval_cols = EVAL_DERIVED_COLUMNS - traces_schema
    if missing_eval_cols:
        raise ValueError(
            f'reeval: source traces missing eval-derived columns '
            f'{sorted(missing_eval_cols)} — not a re-eval target.',
        )
    traces_q0 = pl.read_parquet(
        traces_path, columns=['id', 'predicted_q_at_start'],
    )
    row_map = build_row_checkpoint_map(runs, traces_q0, source, cfg)

    env, env_params = make_env(get_env_spec(cfg.env_name))
    # Jit the eval kernel ONCE (params + key dynamic, everything else
    # baked in) — reused across all (cell, seed, burst) so XLA
    # compiles a single executable instead of one per call.
    eval_fn = _make_eval_burst_fn(
        env=env, env_params=env_params, q_net=cfg.q_network,
        gamma=cfg.gamma, episode_cap=cfg.eval_episode_cap,
        n_episodes=n_episodes,
    )

    # Re-roll per (cell, seed). The new eval arrays are accumulated
    # as numpy stacks keyed by id; only the six eval columns live in
    # memory (Arrow-native after the Series build), never the heavy
    # training columns.
    ids_in_order = [str(i) for i in traces_q0.get_column('id').to_list()]
    new_eval: dict[str, dict[str, np.ndarray]] = {
        col: {} for col in EVAL_DERIVED_COLUMNS
    }
    # Cache the per-cell accessor: for the bundle layout this keeps
    # the (multi-GB) decode to once per cell even as rows of
    # different cells interleave in the id order.
    cell_cache: dict[int, CellCheckpoints] = {}
    for rid in ids_in_order:
        cell_idx, seed = row_map.by_id[rid]
        cell = cell_cache.get(cell_idx)
        if cell is None:
            cell = source.load_cell(cell_idx)
            cell_cache[cell_idx] = cell
        stacked = _reeval_one_cell(
            cell=cell, eval_fn=eval_fn, seed=seed,
            n_bursts=row_map.n_bursts,
            eval_seed_base=eval_seed_base, keying=eval_keying,
        )
        for col in EVAL_DERIVED_COLUMNS:
            new_eval[col][rid] = stacked[col]

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_reeval_traces(
        traces_path, out_dir / 'traces.parquet',
        new_eval=new_eval, ids_in_order=ids_in_order,
    )

    # Restamp n_episodes in runs (scalar column). The eval-derived
    # measurables recompute downstream from the new traces at
    # ingest; the runs row only needs the n_episodes provenance.
    new_runs = runs.with_columns(
        pl.lit(n_episodes).cast(pl.Int64).alias('n_episodes'),
    )
    atomic_write_parquet(new_runs, out_dir / 'runs.parquet')
    _mirror_manifest_reference(corpus_dir, out_dir, n_episodes=n_episodes)
    return out_dir


def _write_reeval_traces(
    source_traces: Path,
    out_traces: Path,
    *,
    new_eval: Mapping[str, Mapping[str, np.ndarray]],
    ids_in_order: Sequence[str],
) -> None:
    """Write the new traces.parquet: every training-derived column
    copied verbatim from `source_traces`, the six eval-derived
    columns replaced by the re-rolled arrays in `new_eval`.

    Memory-bounded: the heavy training columns flow through a polars
    LAZY scan joined to a small in-memory eval frame, streamed to
    parquet via `sink_parquet`. The training columns are never
    materialised as Python objects — they stay Arrow from disk to
    disk. Only the six eval columns (re-rolled) are built in memory,
    and only those are dropped + re-attached.

    The join is on `id`; the original column ORDER is preserved by
    selecting the source schema's column list (with the eval columns
    sourced from the joined-in frame)."""
    source_cols = pl.scan_parquet(source_traces).collect_schema().names()
    # Build the small eval frame (id + 6 re-rolled columns). Each
    # column is a list-of-arrays → polars infers the nested
    # List/Array dtype matching the source schema's `large_list`.
    eval_frame_data: dict[str, list[object]] = {'id': list(ids_in_order)}
    for col in EVAL_DERIVED_COLUMNS:
        eval_frame_data[col] = [
            new_eval[col][rid].tolist() for rid in ids_in_order
        ]
    eval_df = pl.DataFrame(eval_frame_data)
    # Lazy scan of the source with the OLD eval columns dropped, then
    # join the re-rolled eval columns, then reselect the original
    # column order. `sink_parquet` streams the heavy training columns
    # straight through.
    lazy = (
        pl.scan_parquet(source_traces)
        .drop(EVAL_DERIVED_COLUMNS)
        .join(eval_df.lazy(), on='id', how='left')
        .select(source_cols)
    )
    tmp = out_traces.with_suffix(out_traces.suffix + '.partial')
    try:
        lazy.sink_parquet(tmp)
        tmp.replace(out_traces)
    except BaseException:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


def _mirror_manifest_reference(
    source_dir: Path, out_dir: Path, *, n_episodes: int,
) -> None:
    """Write `out_dir/_remote.json` referencing the SOURCE corpus's
    q-checkpoint cloud objects so the re-eval corpus stays
    restorable / archivable.

    The checkpoints are byte-identical to the source's (same trained
    params), so the new corpus does NOT re-upload ~3 GB of msgpacks;
    instead its manifest points at the source's `remote_root` and
    lists only the `q_checkpoints/` entries. `restore(out_dir)` then
    recovers the checkpoints from where they actually live. The new
    corpus's own freshly-written `runs.parquet` / `traces.parquet`
    are NOT in this manifest — they are local and get archived (with
    their own fresh entries) by a later `corroborate archive
    <out_dir> <remote>` when the operator chooses to push the
    re-eval corpus.

    A `RemoteManifest` carries a single `remote_root`; per-file
    roots don't exist. Keeping `remote_root = <source root>` and
    only the q-checkpoint relpaths is the honest shape — every
    listed file genuinely lives under that root. `n_episodes` is
    accepted for symmetry with the corpus naming but the manifest
    body is provenance-only (no n_episodes field on the schema)."""
    del n_episodes  # provenance lives in the corpus dir name, not the manifest
    manifest = load_manifest(source_dir)
    if manifest is None:
        return
    kept: tuple[RemoteFile, ...] = tuple(
        f for f in manifest.files if f.relpath.startswith('q_checkpoints/')
    )
    if not kept:
        return
    new_manifest = RemoteManifest(
        remote_root=manifest.remote_root, files=kept,
    )
    import json
    atomic_write_text(
        out_dir / '_remote.json',
        json.dumps(dict(new_manifest.as_dict()), indent=2),
    )


__all__ = [
    'CellCheckpoints',
    'CellConfig',
    'CheckpointSource',
    'EVAL_DERIVED_COLUMNS',
    'EvalKeying',
    'RowCheckpointMap',
    'build_row_checkpoint_map',
    'discover_checkpoint_source',
    'eval_key',
    'reeval_corpus',
]
