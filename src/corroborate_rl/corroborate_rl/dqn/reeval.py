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
from typing import TYPE_CHECKING, Final, Literal, Protocol, TypeIs, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
from gymnax import EnvParams

from corroborate.corpus.cloud import (
    RemoteFile,
    RemoteManifest,
    load_manifest,
    purge as cloud_purge,
    restore as cloud_restore,
)
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


# ============ Disk-bounded checkpoint restore (streaming) ============

@runtime_checkable
class CheckpointRestorer(Protocol):
    """Per-cell checkpoint restore / release for the disk-bounded
    streaming re-eval path.

    `reeval_corpus` assumes every checkpoint is already local — fine
    for the small per-file Breakout / Freeway n_eps1 corpora
    (~MB-scale cells). The bundle-layout 3M corpora are different:
    each cell's bundle is multi-GB (snake_g099_canonical_3M_ckpt:
    ~2.7 GB × 4 cells = ~11 GB) and restoring all of them at once can
    overflow a tight local disk. `reeval_corpus_streaming` restores
    ONE bundle, re-evals the runs it covers, then releases it before
    restoring the next — peak ckpt disk = one cell's bundle.

    The Protocol keeps the streaming core cloud-agnostic (and unit-
    testable with a local no-op restorer): the driver only needs
    "make cell N's checkpoint present" / "free cell N's checkpoint".
    `relpaths()` exposes the bundle file relpaths the driver discovers
    cells from (so it needn't pre-download anything to enumerate the
    work)."""

    def relpaths(self) -> tuple[str, ...]:
        """The `q_checkpoints/.../cell{NNN}.msgpack` relpaths (relative
        to the corpus dir) this restorer can materialise — the cell
        enumeration source."""
        ...

    def restore(self, relpath: str) -> None:
        """Make `relpath` present locally (download if evicted; no-op
        if already present with the right bytes)."""
        ...

    def release(self, relpath: str) -> None:
        """Free the local copy of `relpath` (it stays cloud-restorable
        via the manifest). Idempotent — a no-op if already absent."""
        ...


_BUNDLE_RELPATH_RE: Final[re.Pattern[str]] = re.compile(
    r'(?:^|/)cell(\d+)\.msgpack$',
)
"""Matches a bundle-layout checkpoint relpath (`q_checkpoints/<sub>/
cell{NNN}.msgpack`) and captures the cell index."""


@dataclass(frozen=True, slots=True)
class CloudCheckpointRestorer:
    """`CheckpointRestorer` backed by a corpus's `_remote.json`
    manifest: `restore` / `release` map to `cloud.restore` /
    `cloud.purge` on the single bundle relpath.

    `purge` (not `rm`) is used for release per CLAUDE.md's cloud
    operator discipline — it validates the file is in the manifest
    before deleting, preserving the manifest so the bundle stays
    restorable for a subsequent run / the eventual cache build.

    Built via `from_corpus`, which reads the manifest and selects the
    bundle relpaths matching `q_checkpoints_subdir`. Raises if the
    corpus has no manifest (the streaming path needs the cloud
    address to fetch bundles one at a time) or no bundle entries
    under the subdir."""
    corpus_dir: Path
    bundle_relpaths: tuple[str, ...]

    @classmethod
    def from_corpus(
        cls, corpus_dir: Path, *, q_checkpoints_subdir: str = 'q_checkpoints',
    ) -> CloudCheckpointRestorer:
        manifest = load_manifest(corpus_dir)
        if manifest is None:
            raise FileNotFoundError(
                f'reeval streaming: {corpus_dir} has no _remote.json — '
                'the disk-bounded path restores checkpoint bundles one '
                'at a time from cloud, which needs the manifest. Restore '
                'the manifest first (corroborate.corpus.cloud.'
                'recover_local_manifest) or use reeval_corpus with the '
                'bundles already local.',
            )
        prefix = f'{q_checkpoints_subdir}/'
        relpaths = tuple(
            f.relpath for f in manifest.files
            if f.relpath.startswith(prefix)
            and _BUNDLE_RELPATH_RE.search(f.relpath) is not None
        )
        if not relpaths:
            raise ValueError(
                f'reeval streaming: manifest at {corpus_dir} lists no '
                f'bundle-layout checkpoints under {prefix!r} '
                '(cell{NNN}.msgpack). The streaming path is bundle-only; '
                'per-file corpora are small enough for reeval_corpus.',
            )
        return cls(corpus_dir=corpus_dir, bundle_relpaths=relpaths)

    def relpaths(self) -> tuple[str, ...]:
        return self.bundle_relpaths

    def restore(self, relpath: str) -> None:
        # `cloud.restore` is idempotent: a local file whose sha256
        # matches the manifest is skipped, so a re-run that crashed
        # mid-corpus doesn't re-download completed bundles.
        _ = cloud_restore(self.corpus_dir, files=[relpath])

    def release(self, relpath: str) -> None:
        # `purge` validates manifest membership before deleting and
        # keeps the manifest intact (restore stays available). A bundle
        # already absent locally is a no-op inside purge.
        _ = cloud_purge(self.corpus_dir, files=[relpath])


def _cell_idx_from_relpath(relpath: str) -> int:
    """Extract the cell index from a bundle relpath. Raises on a
    relpath that isn't bundle-shaped (the streaming enumeration only
    ever passes matched relpaths, so a miss is a programming error)."""
    m = _BUNDLE_RELPATH_RE.search(relpath)
    if m is None:
        raise ValueError(
            f'reeval streaming: relpath {relpath!r} is not a bundle '
            'cell{NNN}.msgpack path',
        )
    return int(m.group(1))


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


@dataclass(frozen=True, slots=True)
class _RowFingerprintMatcher:
    """Resolves checkpoint (cell_idx, seed) → run_id by matching the
    recomputed per-burst max-Q TRAJECTORY against the source trace's
    `predicted_q_at_start[b, 0]`.

    Built once (`build`) over the full runs + trace-Q projection; its
    `match_cell` method is reused per checkpoint cell so the eager-
    monolith path (`build_row_checkpoint_map`) and the per-bundle
    streaming path (`reeval_corpus_streaming`) share ONE matching
    implementation. Holding the jitted `_max_q` + env on the instance
    keeps XLA compilation to once across every (cell, seed, burst)
    probe.

    The probe `bursts` are fixed at construction from `n_bursts` (the
    spread that disambiguates arms — see `_probe_bursts`); every cell
    must report the same `n_bursts`, asserted by the caller."""
    env: Env
    env_params: EnvParams
    q_net: QFunction
    bursts: tuple[int, ...]
    _max_q: Callable[[Params, jax.Array], jax.Array]
    # run_id → its source-trace Q-fingerprint at the probe bursts.
    rows_by_seed: Mapping[int, tuple[tuple[str, np.ndarray], ...]]

    @classmethod
    def build(
        cls, runs: pl.DataFrame, traces: pl.DataFrame, cfg: CellConfig,
        *, n_bursts: int,
    ) -> _RowFingerprintMatcher:
        env, env_params = make_env(get_env_spec(cfg.env_name))
        q_net = cfg.q_network
        bursts = _probe_bursts(n_bursts)

        # `max_a Q(params, obs)` jitted ONCE so XLA compiles a single
        # executable reused across every (cell, seed, burst) probe —
        # avoids the per-eager-call compilation churn that accumulated
        # device memory on 50-burst corpora.
        @jax.jit
        def _max_q(params: Params, obs: jax.Array) -> jax.Array:
            return jnp.max(q_net(params, obs))

        rows_by_seed: dict[int, list[tuple[str, np.ndarray]]] = {}
        for r in runs.iter_rows(named=True):
            seed = _require_int(r, 'seed')
            run_id = str(r['id'])
            traj = _trace_predicted_q_at_bursts(traces, run_id, bursts)
            rows_by_seed.setdefault(seed, []).append((run_id, traj))
        return cls(
            env=env, env_params=env_params, q_net=q_net,
            bursts=bursts, _max_q=_max_q,
            rows_by_seed={s: tuple(v) for s, v in rows_by_seed.items()},
        )

    def _recompute_trajectory(
        self, cell: CellCheckpoints, seed: int,
    ) -> np.ndarray:
        # Recompute max_a Q(s_0^b) at the probe bursts under the
        # burst-b params. The canonical burst-b reset uses the
        # ORIGINAL eval key (how the source trace's value was
        # produced); eval_burst splits then resets per episode, so
        # episode 0's reset key is split(key, K)[0] → reset =
        # split(., 2)[0].
        out = np.empty(len(self.bursts), dtype=np.float64)
        for i, burst in enumerate(self.bursts):
            params = cell.online_params(seed=seed, burst=burst)
            key_b = eval_key(
                keying='original', eval_seed_base=0, seed=seed, burst=burst,
            )
            ep_keys = jax.random.split(key_b, 1)
            reset_key, _run = jax.random.split(ep_keys[0])
            obs0, _state = self.env.reset(reset_key, self.env_params)
            out[i] = float(self._max_q(params, obs0))
        return out

    def match_cell(
        self,
        cell: CellCheckpoints,
        seeds: Sequence[int],
        *,
        used_ids: set[str],
        match_tol: float,
    ) -> dict[str, tuple[int, int]]:
        """Match each (cell.cell_idx, seed) to the NEAREST unused run
        row (by L∞ over the per-burst Q-trajectory), within
        `match_tol`. Mutates `used_ids` so subsequent cells (e.g. the
        SAME seed under the other arm's bundle) bind the remaining
        rows. Greedy nearest-match resolves ties harmlessly: identical
        twins (pre-divergence params) yield identical re-eval output,
        so which twin binds which doesn't matter — only the bijection.
        A >tol residual (no run row reproduces the cell's trajectory)
        is a real error and raises."""
        resolved: dict[str, tuple[int, int]] = {}
        for seed in seeds:
            recomputed = self._recompute_trajectory(cell, seed)
            best_id: str | None = None
            best_dist = float('inf')
            for (run_id, traj) in self.rows_by_seed.get(seed, ()):
                if run_id in used_ids:
                    continue
                d = _l_inf(traj, recomputed)
                if d < best_dist:
                    best_dist, best_id = d, run_id
            if best_id is None or best_dist > match_tol:
                raise ValueError(
                    f'reeval: cell {cell.cell_idx} seed {seed} has no '
                    'unused run row whose per-burst predicted_q '
                    f'trajectory matches within tol={match_tol} (best '
                    f'L∞={best_dist:.6g}). The Q-trajectory fingerprint '
                    'should identify the (arm, seed) of a checkpoint '
                    'cell — a >tol residual means the checkpoints or '
                    'the source traces are inconsistent.',
                )
            resolved[best_id] = (cell.cell_idx, seed)
            used_ids.add(best_id)
        return resolved


def _l_inf(a: np.ndarray, b: np.ndarray) -> float:
    """L∞ distance, treating shape mismatch as +inf (so a wrong-shape
    fingerprint never spuriously matches)."""
    if a.shape != b.shape:
        return float('inf')
    return float(np.max(np.abs(a - b)))


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
    row within `match_tol` — a silent mis-pair would scramble V/D.

    Eager-monolith path: every checkpoint cell must be present locally
    (`source.load_cell` reads each bundle). The per-bundle streaming
    path (`reeval_corpus_streaming`) reuses the same
    `_RowFingerprintMatcher` but restores one bundle at a time."""
    matcher = _RowFingerprintMatcher.build(
        runs, traces, cfg, n_bursts=source.n_bursts,
    )
    by_id: dict[str, tuple[int, int]] = {}
    used_ids: set[str] = set()
    for cell_idx, seeds in source.cell_seeds.items():
        cell = source.load_cell(cell_idx)
        by_id.update(matcher.match_cell(
            cell, seeds, used_ids=used_ids, match_tol=match_tol,
        ))

    missing = set(runs.get_column('id').to_list()) - set(by_id)
    if missing:
        raise ValueError(
            f'reeval: {len(missing)} run rows had no checkpoint match '
            f'(e.g. {sorted(missing)[:3]}). Checkpoint cells cover '
            f'seeds {sorted({s for ss in source.cell_seeds.values() for s in ss})}.',
        )
    return RowCheckpointMap(by_id=by_id, n_bursts=source.n_bursts)


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

    _finalize_reeval_output(
        corpus_dir=corpus_dir, out_dir=out_dir, runs=runs,
        traces_path=traces_path, new_eval=new_eval,
        ids_in_order=ids_in_order, n_episodes=n_episodes,
    )
    return out_dir


def _finalize_reeval_output(
    *,
    corpus_dir: Path,
    out_dir: Path,
    runs: pl.DataFrame,
    traces_path: Path,
    new_eval: Mapping[str, Mapping[str, np.ndarray]],
    ids_in_order: Sequence[str],
    n_episodes: int,
) -> None:
    """Write the new corpus's traces + runs + manifest reference.

    Shared by `reeval_corpus` (eager) and `reeval_corpus_streaming`
    (disk-bounded): both assemble the six eval-derived arrays keyed by
    id into `new_eval`, then this writes them out — training columns
    streamed verbatim from `traces_path`, runs restamped to
    `n_episodes`, and the source's q-checkpoint cloud references
    mirrored so the new corpus stays restorable."""
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


def reeval_corpus_streaming(
    corpus_dir: Path,
    *,
    n_episodes: int,
    out_dir: Path,
    restorer: CheckpointRestorer,
    eval_seed_base: int = 0,
    eval_keying: EvalKeying = 'paired',
) -> Path:
    """Disk-bounded re-eval for bundle-layout corpora whose per-cell
    checkpoints are too large to all fit locally at once.

    Identical OUTPUT to `reeval_corpus` (same paired eval keying, same
    eval-derived / training-derived split, same self-verifying arm
    mapping) — but restores ONE checkpoint bundle at a time via
    `restorer`, re-evals the runs that bundle covers, then RELEASES it
    before restoring the next. Peak checkpoint disk = a single cell's
    bundle (e.g. ~2.7 GB for snake_g099_canonical_3M_ckpt) instead of
    the whole `q_checkpoints/` tree (~11 GB).

    Per bundle:

    1. `restorer.restore(relpath)` — download cell N's bundle.
    2. `load_bundle` → its `seeds` + `n_bursts`. The first bundle pins
       `n_bursts`; later bundles must agree (uniform source corpus).
    3. Match the bundle's (cell, seed) entries to the still-unused run
       rows by the per-burst predicted-Q TRAJECTORY fingerprint (the
       SAME `_RowFingerprintMatcher` the eager path uses) — so the two
       bundles covering one seed under different arms bind the correct
       run rows.
    4. Re-roll greedy `eval_burst` at `n_episodes` for every matched
       (seed, burst); stash the six eval arrays in memory (tiny —
       greedy-eval episode arrays compress to ~MB/cell).
    5. `restorer.release(relpath)` — free the bundle.

    After all bundles: every run row must be matched (else raise), and
    the output is written exactly as `reeval_corpus` does. The eval
    arrays for the WHOLE corpus live in memory simultaneously (they're
    small); only the multi-GB CHECKPOINTS are streamed one at a time.

    `restorer` is the disk-management injection point — typically
    `CloudCheckpointRestorer.from_corpus(corpus_dir)` for production,
    or a local no-op restorer in tests. Returns `out_dir`."""
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

    # Eval-derived column presence guard (same as the eager path) —
    # fail before any multi-GB bundle download if this isn't a re-eval
    # target.
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
    ids_in_order = [str(i) for i in traces_q0.get_column('id').to_list()]

    bundle_relpaths = sorted(
        restorer.relpaths(), key=_cell_idx_from_relpath,
    )
    if not bundle_relpaths:
        raise ValueError(
            'reeval streaming: restorer enumerated no checkpoint '
            'bundles — nothing to re-eval.',
        )

    env, env_params = make_env(get_env_spec(cfg.env_name))
    eval_fn = _make_eval_burst_fn(
        env=env, env_params=env_params, q_net=cfg.q_network,
        gamma=cfg.gamma, episode_cap=cfg.eval_episode_cap,
        n_episodes=n_episodes,
    )

    new_eval: dict[str, dict[str, np.ndarray]] = {
        col: {} for col in EVAL_DERIVED_COLUMNS
    }
    by_id: dict[str, tuple[int, int]] = {}
    used_ids: set[str] = set()
    matcher: _RowFingerprintMatcher | None = None
    n_bursts: int | None = None

    for relpath in bundle_relpaths:
        cell_idx = _cell_idx_from_relpath(relpath)
        restorer.restore(relpath)
        try:
            bundle = load_bundle(bundle_path(
                corpus_dir / relpath.rsplit('/', 1)[0], cell_idx=cell_idx,
            ))
            if n_bursts is None:
                n_bursts = bundle.n_bursts
                matcher = _RowFingerprintMatcher.build(
                    runs, traces_q0, cfg, n_bursts=n_bursts,
                )
            elif bundle.n_bursts != n_bursts:
                raise ValueError(
                    f'reeval streaming: bundle cell{cell_idx:03d} reports '
                    f'n_bursts={bundle.n_bursts} but an earlier bundle '
                    f'had {n_bursts} — non-uniform source corpus, '
                    'unsupported.',
                )
            assert matcher is not None  # set in lockstep with n_bursts
            cell = CellCheckpoints(
                base_dir=corpus_dir / relpath.rsplit('/', 1)[0],
                cell_idx=cell_idx, _bundle=bundle,
            )
            resolved = matcher.match_cell(
                cell, bundle.seeds, used_ids=used_ids, match_tol=1e-3,
            )
            by_id.update(resolved)
            for rid, (_ci, seed) in resolved.items():
                stacked = _reeval_one_cell(
                    cell=cell, eval_fn=eval_fn, seed=seed,
                    n_bursts=n_bursts,
                    eval_seed_base=eval_seed_base, keying=eval_keying,
                )
                for col in EVAL_DERIVED_COLUMNS:
                    new_eval[col][rid] = stacked[col]
        finally:
            # Free the bundle's local copy whether or not it processed
            # cleanly — the disk-bounded invariant (peak = one bundle)
            # must hold even on a mid-corpus raise.
            restorer.release(relpath)

    missing = set(ids_in_order) - set(by_id)
    if missing:
        raise ValueError(
            f'reeval streaming: {len(missing)} run rows had no checkpoint '
            f'match (e.g. {sorted(missing)[:3]}). The bundles enumerated '
            f'by the restorer ({bundle_relpaths}) do not cover every run '
            'row — check the manifest / chunk coverage.',
        )

    _finalize_reeval_output(
        corpus_dir=corpus_dir, out_dir=out_dir, runs=runs,
        traces_path=traces_path, new_eval=new_eval,
        ids_in_order=ids_in_order, n_episodes=n_episodes,
    )
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
    'CheckpointRestorer',
    'CheckpointSource',
    'CloudCheckpointRestorer',
    'EVAL_DERIVED_COLUMNS',
    'EvalKeying',
    'RowCheckpointMap',
    'build_row_checkpoint_map',
    'discover_checkpoint_source',
    'eval_key',
    'reeval_corpus',
    'reeval_corpus_streaming',
]
