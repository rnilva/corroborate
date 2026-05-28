"""Per-cell Q-checkpoint bundle — one msgpack file per cell holding
all (seed × burst|final) snapshots instead of one file per snapshot.

**Why this exists.** The per-file layout (`cell{NNN}_{seed}_burst{BB}.msgpack`)
emits 51 files per (cell, seed) — 765 files per cell at 15 seeds.
A 4-cell sweep produces 3060 files; archiving them serially over
TLS dominates wall-clock (~2.5h on R2 for a typical MinAtar sweep,
limited by per-PUT RTT rather than bandwidth).

A `QCheckpointBundle` collapses one cell's snapshots into ONE
msgpack file, mirroring `traces.parquet`'s "one artifact per cell"
shape. Archive cost drops to 1 PUT per cell.

**Storage shape.** The substrate's `batched_record` already carries
arrays with the seed axis at position 0:

- `per_burst_*` arrays: shape `(n_seeds, n_bursts, *param_shape)`
- `final_*` arrays:     shape `(n_seeds, *param_shape)`

The bundle stores these directly — no reshape, no concatenation.
Indexing a single (seed, role[, burst]) `QCheckpoint` slices axis 0
(and axis 1 for per_burst) at decode time.

**Format.** msgpack via `flax.serialization`, same primitive used
by the per-file layout. Top-level dict:

```
{
    'cell_idx': int,
    'seeds': list[int],             # source seed values, batch-axis-aligned
    'n_bursts': int,                # 0 when no per_burst payload
    'per_burst_online': {k: ndarray (n_seeds, n_bursts, *)} | None,
    'per_burst_target': {k: ndarray (n_seeds, n_bursts, *)} | None,
    'final_online':     {k: ndarray (n_seeds, *)}           | None,
    'final_target':     {k: ndarray (n_seeds, *)}           | None,
}
```

**Four-question test** (CLAUDE.md "When to introduce a primitive"):

1. Typed contract — `QCheckpointBundle` field shapes are the
   substrate's batched-record shapes; pyright checks at every
   producer / consumer site.
2. Runtime narrowing — `bundle.per_burst_online is None` narrows
   the consumer's branch (final-only bundles vs full bundles).
3. Real work — collapses 765 file PUTs per cell into 1, mirrors
   `runs.parquet` / `traces.parquet` per-cell bundling, and gives
   `extract_qcheckpoint(...)` a typed surface where today the
   filename pattern is the contract.
4. Performance floor — per-file HTTP RTT is the floor the
   per-file layout can't compose around without changing the
   archival unit. This primitive IS that change."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import jax
import numpy as np
from flax import serialization as _fs

from corroborate_rl.dqn.claims.q_network import Params
from corroborate_rl.dqn.init_override import InitOverride
from corroborate_rl.dqn.q_checkpoint import (
    CheckpointRole,
    QCheckpoint,
    _stack_per_seed_params,
)


# ============ Bundle dataclass ============

@dataclass(frozen=True, slots=True)
class QCheckpointBundle:
    """All Q-network checkpoints for one cell, indexed by
    (seed, role[, burst]).

    `seeds` is the batch-aligned source seed list. Axis 0 of every
    array in `per_burst_*` / `final_*` corresponds to `seeds[i]`.

    `n_bursts == 0` means the bundle has no per-burst payload (the
    sweep set `keep_q_checkpoint_per_burst=False`). Likewise
    `final_online is None` means no final snapshot.

    Both arms (`online`, `target`) must be present together for a
    given role — partial pairs are a producer-side bug and surfaced
    at construction in `from_batched_record`. The `__post_init__`
    cross-validates: (a) seeds uniqueness, (b) (n_bursts > 0) iff
    per_burst_* populated, (c) axis-0 of every payload array
    matches len(seeds), (d) per_burst axis-1 matches n_bursts.
    These keep contradictory states (e.g. `n_bursts=4` with
    `per_burst_online=None`) unconstructible — load_bundle and
    from_batched_record both depend on the invariants for safe
    indexing."""
    cell_idx: int
    seeds: tuple[int, ...]
    n_bursts: int
    per_burst_online: Params | None
    per_burst_target: Params | None
    final_online: Params | None
    final_target: Params | None

    def __post_init__(self) -> None:
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError(
                f'QCheckpointBundle.seeds must be unique; got '
                f'{self.seeds!r}',
            )
        if self.n_bursts < 0:
            raise ValueError(
                f'QCheckpointBundle.n_bursts must be >= 0; got '
                f'{self.n_bursts}',
            )
        per_burst_present = (
            self.per_burst_online is not None
            and self.per_burst_target is not None
        )
        if per_burst_present and self.n_bursts == 0:
            raise ValueError(
                'QCheckpointBundle: per_burst_* populated but '
                'n_bursts=0 — inconsistent state.',
            )
        if not per_burst_present and self.n_bursts > 0:
            raise ValueError(
                f'QCheckpointBundle: n_bursts={self.n_bursts} but '
                'per_burst_* is None — inconsistent state.',
            )
        n_seeds = len(self.seeds)
        _validate_payload_axis(
            self.per_burst_online, 'per_burst_online', n_seeds,
            burst_axis_len=self.n_bursts,
        )
        _validate_payload_axis(
            self.per_burst_target, 'per_burst_target', n_seeds,
            burst_axis_len=self.n_bursts,
        )
        _validate_payload_axis(
            self.final_online, 'final_online', n_seeds,
            burst_axis_len=None,
        )
        _validate_payload_axis(
            self.final_target, 'final_target', n_seeds,
            burst_axis_len=None,
        )


def _validate_payload_axis(
    payload: Params | None, field: str, n_seeds: int,
    *, burst_axis_len: int | None,
) -> None:
    """Cross-leaf shape validation for a QCheckpointBundle payload
    field. Every leaf must share axis-0 == n_seeds, and for
    per_burst payloads axis-1 == burst_axis_len. Cross-key
    inconsistency (e.g. w0 with 50 bursts and b0 with 49) is a
    producer-side bug; surface at construction rather than letting
    extract_qcheckpoint raise an IndexError mid-slice."""
    if payload is None:
        return
    for key, leaf in payload.items():
        shape = tuple(int(d) for d in leaf.shape)
        if not shape or shape[0] != n_seeds:
            raise ValueError(
                f'QCheckpointBundle.{field}[{key!r}] axis-0='
                f'{shape[0] if shape else "<scalar>"} != '
                f'len(seeds)={n_seeds}; expected '
                f'(n_seeds, ...) leading shape.',
            )
        if burst_axis_len is not None:
            if len(shape) < 2 or shape[1] != burst_axis_len:
                raise ValueError(
                    f'QCheckpointBundle.{field}[{key!r}] axis-1='
                    f'{shape[1] if len(shape) >= 2 else "<missing>"} '
                    f'!= n_bursts={burst_axis_len}; expected '
                    f'(n_seeds, n_bursts, *) leading shape.',
                )


# ============ Path convention ============

def bundle_path(base_dir: Path, *, cell_idx: int) -> Path:
    """Canonical on-disk path for one cell's bundle.

    Layout: `<base_dir>/cell{NNN}.msgpack`. Lives under the same
    `q_checkpoints/` sidecar dir the per-file layout used, so the
    archive walker (`cloud._default_files`) picks it up unchanged."""
    return base_dir / f'cell{cell_idx:03d}.msgpack'


# ============ Serialization ============

def save_bundle(path: Path, bundle: QCheckpointBundle) -> None:
    """Write `bundle` to `path` as msgpack. Creates parent dir,
    atomic via tmp + rename so a crashed write leaves no truncated
    file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _fs.msgpack_serialize(_bundle_as_dict(bundle))
    tmp = path.with_suffix(path.suffix + '.tmp')
    _ = tmp.write_bytes(payload)
    tmp.replace(path)


def _bundle_as_dict(bundle: QCheckpointBundle) -> dict[str, object]:
    """Plain-Python representation used by msgpack. Arrays kept as
    numpy ndarrays (msgpack's native binary path; flax stamps the
    dtype/shape header so decode is dtype-safe)."""
    def _params_as_np(p: Params | None) -> dict[str, np.ndarray] | None:
        if p is None:
            return None
        return {k: np.asarray(v) for k, v in p.items()}
    return {
        'cell_idx': int(bundle.cell_idx),
        'seeds': [int(s) for s in bundle.seeds],
        'n_bursts': int(bundle.n_bursts),
        'per_burst_online': _params_as_np(bundle.per_burst_online),
        'per_burst_target': _params_as_np(bundle.per_burst_target),
        'final_online': _params_as_np(bundle.final_online),
        'final_target': _params_as_np(bundle.final_target),
    }


def load_bundle(path: Path) -> QCheckpointBundle:
    """Read a msgpack bundle back into a typed `QCheckpointBundle`.

    Raises `FileNotFoundError` if `path` doesn't exist;
    `ValueError` if the bytes decode to an unexpected shape."""
    raw = path.read_bytes()
    decoded: object = _fs.msgpack_restore(raw)
    if not isinstance(decoded, Mapping):
        raise ValueError(
            f'{path}: expected msgpack to decode to a mapping; '
            f'got {type(decoded).__name__}',
        )
    cell_idx_raw = decoded.get('cell_idx')
    seeds_raw = decoded.get('seeds')
    n_bursts_raw = decoded.get('n_bursts')
    # `isinstance(True, int) is True` in Python — reject bool
    # explicitly so a corrupt / hand-crafted bundle with cell_idx=True
    # or n_bursts=False can't silently populate the int fields as 0/1.
    if not isinstance(cell_idx_raw, int) or isinstance(cell_idx_raw, bool):
        raise ValueError(
            f'{path}: bundle missing int cell_idx '
            f'(got {type(cell_idx_raw).__name__})',
        )
    if not isinstance(n_bursts_raw, int) or isinstance(n_bursts_raw, bool):
        raise ValueError(
            f'{path}: bundle missing int n_bursts '
            f'(got {type(n_bursts_raw).__name__})',
        )
    if not isinstance(seeds_raw, Sequence) or isinstance(seeds_raw, (str, bytes)):
        raise ValueError(
            f'{path}: bundle missing seeds sequence '
            f'(got {type(seeds_raw).__name__})',
        )
    seeds_tuple = tuple(int(s) for s in seeds_raw)
    return QCheckpointBundle(
        cell_idx=cell_idx_raw,
        seeds=seeds_tuple,
        n_bursts=n_bursts_raw,
        per_burst_online=_decode_params(
            decoded.get('per_burst_online'), path, 'per_burst_online',
        ),
        per_burst_target=_decode_params(
            decoded.get('per_burst_target'), path, 'per_burst_target',
        ),
        final_online=_decode_params(
            decoded.get('final_online'), path, 'final_online',
        ),
        final_target=_decode_params(
            decoded.get('final_target'), path, 'final_target',
        ),
    )


def _decode_params(
    raw: object, path: Path, field: str,
) -> Params | None:
    """Coerce a msgpack-decoded entry into a `Params` (dict of
    jax.Array). Returns None for absent fields. Raises on shape
    mismatch (typed at construction time, so a bad dict here is a
    corrupt file)."""
    import jax.numpy as jnp
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError(
            f'{path}: bundle field {field!r} must be a mapping or None; '
            f'got {type(raw).__name__}',
        )
    return {
        str(k): jnp.asarray(v) for k, v in raw.items()
    }


# ============ Construction from batched record ============

def from_batched_record(
    batched_record: Mapping[str, jax.Array],
    *,
    cell_idx: int,
    seeds: Sequence[int],
) -> QCheckpointBundle | None:
    """Build a bundle from the substrate's batched per-cell record.

    The record carries `__q_checkpoint__<arm>__<role>__<param_key>`
    entries with the seed axis at position 0 and (for per_burst) the
    burst axis at position 1. We strip the sentinel prefix, group by
    `(role, arm)`, and stack the param-key sub-dicts into the
    bundle's `per_burst_*` / `final_*` fields.

    Returns `None` when the record carries no checkpoint sentinel
    keys — the off-by-default path. Returns a bundle with both
    `per_burst_*` AND `final_*` set when both flags are on; partial
    coverage is OK (e.g. `keep_q_checkpoint_final=True` without
    `keep_q_checkpoint_per_burst`)."""
    from corroborate_rl.dqn.q_checkpoint import parse_checkpoint_key
    grouped: dict[
        tuple[CheckpointRole, str],
        dict[str, jax.Array],
    ] = {}
    for key, arr in batched_record.items():
        parts = parse_checkpoint_key(key)
        if parts is None:
            continue
        bucket = grouped.setdefault((parts.role, parts.arm), {})
        bucket[parts.param_key] = arr

    if not grouped:
        return None

    final_online = grouped.get(('final', 'online'))
    final_target = grouped.get(('final', 'target'))
    if (final_online is None) != (final_target is None):
        raise ValueError(
            'q_checkpoint_bundle.from_batched_record: final has one '
            f'arm but not the other (online={final_online is not None}, '
            f'target={final_target is not None}). Producer must emit '
            'both arms together — check train_with_eval gating.',
        )

    per_burst_online = grouped.get(('per_burst', 'online'))
    per_burst_target = grouped.get(('per_burst', 'target'))
    if (per_burst_online is None) != (per_burst_target is None):
        raise ValueError(
            'q_checkpoint_bundle.from_batched_record: per_burst has '
            f'one arm but not the other (online='
            f'{per_burst_online is not None}, target='
            f'{per_burst_target is not None}). Producer must emit '
            'both arms together — check train_with_eval gating.',
        )

    if per_burst_online is not None:
        first_leaf = next(iter(per_burst_online.values()))
        n_bursts = int(first_leaf.shape[1])
    else:
        n_bursts = 0

    return QCheckpointBundle(
        cell_idx=cell_idx,
        seeds=tuple(int(s) for s in seeds),
        n_bursts=n_bursts,
        per_burst_online=(
            dict(per_burst_online) if per_burst_online is not None else None
        ),
        per_burst_target=(
            dict(per_burst_target) if per_burst_target is not None else None
        ),
        final_online=(
            dict(final_online) if final_online is not None else None
        ),
        final_target=(
            dict(final_target) if final_target is not None else None
        ),
    )


# ============ Extraction ============

def extract_qcheckpoint(
    bundle: QCheckpointBundle,
    *,
    seed: int,
    role: CheckpointRole,
    burst: int | None = None,
) -> QCheckpoint:
    """Slice one `QCheckpoint` out of a bundle.

    `role='final'` requires the bundle's `final_*` to be populated;
    `role='per_burst'` requires `per_burst_*` AND a valid `burst`
    index in `[0, bundle.n_bursts)`. Raises `KeyError` on a missing
    seed and `ValueError` on shape/burst mismatches."""
    try:
        seed_idx = bundle.seeds.index(seed)
    except ValueError as e:
        raise KeyError(
            f'seed {seed} not in bundle.seeds={bundle.seeds}',
        ) from e
    if role == 'final':
        if bundle.final_online is None or bundle.final_target is None:
            raise ValueError(
                f'bundle (cell {bundle.cell_idx}) has no final '
                'snapshot; was keep_q_checkpoint_final=True at sweep time?',
            )
        online = {
            k: v[seed_idx] for k, v in bundle.final_online.items()
        }
        target = {
            k: v[seed_idx] for k, v in bundle.final_target.items()
        }
        return QCheckpoint(
            online_params=online, target_params=target,
            burst=-1, global_step=-1,
        )
    if burst is None:
        raise ValueError(
            "extract_qcheckpoint: role='per_burst' requires a "
            'burst index',
        )
    if bundle.per_burst_online is None or bundle.per_burst_target is None:
        raise ValueError(
            f'bundle (cell {bundle.cell_idx}) has no per_burst payload; '
            'was keep_q_checkpoint_per_burst=True at sweep time?',
        )
    if burst < 0 or burst >= bundle.n_bursts:
        raise ValueError(
            f'extract_qcheckpoint: burst={burst} out of range '
            f'[0, {bundle.n_bursts}) for cell {bundle.cell_idx}',
        )
    online = {
        k: v[seed_idx, burst]
        for k, v in bundle.per_burst_online.items()
    }
    target = {
        k: v[seed_idx, burst]
        for k, v in bundle.per_burst_target.items()
    }
    return QCheckpoint(
        online_params=online, target_params=target,
        burst=burst, global_step=-1,
    )


# ============ Batched-loading for vmap-over-seeds ============

def extract_batched_init_override(
    bundle: QCheckpointBundle,
    seeds: Sequence[int],
    *,
    role: CheckpointRole,
    burst: int | None = None,
    load_target: bool,
) -> InitOverride:
    """Slice an already-loaded bundle's batched arrays for the
    requested seed list and re-stack along axis 0.

    Splitting load from extract lets the dispatch loop in
    `yaml_sweep.dispatch_sweep` load the bundle ONCE (sweep-wide) and
    reuse it across per-(env, chunk) grid points, avoiding the
    redundant ~1.5 GB msgpack decode per iteration.

    Pre-flight validation:
    - `seeds` must be non-empty.
    - `seeds` must be internally unique (duplicates would produce
      identical vmap lanes — pseudo-replication that downstream
      parquet records as independent runs).
    - `seeds` must be a subset of `bundle.seeds` (refuse a
      partial-success-then-KeyError mid-extract; a clear error
      naming the missing seeds is more debuggable)."""
    if not seeds:
        raise ValueError('seeds must be non-empty')
    if len(set(seeds)) != len(seeds):
        raise ValueError(
            'extract_batched_init_override: seeds must be unique; '
            f'got duplicates in {tuple(seeds)!r}. Duplicate '
            'requested seeds would produce identical vmap lanes — '
            'pseudo-replication that downstream parquet would record '
            'as independent runs.',
        )
    missing = set(seeds) - set(bundle.seeds)
    if missing:
        raise ValueError(
            f'extract_batched_init_override: requested seeds '
            f'{sorted(missing)!r} not present in bundle '
            f'(bundle.seeds={bundle.seeds!r}, cell_idx='
            f'{bundle.cell_idx}). The bundle was written for a '
            'different chunk; resume from the bundle file whose '
            'cell_idx covers these seeds.',
        )
    per_seed_online: list[Params] = []
    per_seed_target: list[Params] = []
    for s in seeds:
        ckpt = extract_qcheckpoint(
            bundle, seed=s, role=role, burst=burst,
        )
        per_seed_online.append(ckpt.online_params)
        if load_target:
            per_seed_target.append(ckpt.target_params)
    online_stacked = _stack_per_seed_params(
        per_seed_online,
        field_label='extract_batched_init_override.online_params',
        ref_seed=seeds[0], ref_seeds=seeds,
    )
    target_stacked = (
        _stack_per_seed_params(
            per_seed_target,
            field_label='extract_batched_init_override.target_params',
            ref_seed=seeds[0], ref_seeds=seeds,
        )
        if load_target else None
    )
    return InitOverride(
        online_params=online_stacked,
        target_params=target_stacked,
    )


def load_batched_init_override_from_bundle(
    path: Path,
    seeds: Sequence[int],
    *,
    role: CheckpointRole,
    burst: int | None = None,
    load_target: bool,
) -> InitOverride:
    """Load a bundle and extract a batched InitOverride for the
    requested seeds. Thin wrapper around `load_bundle` +
    `extract_batched_init_override` for single-shot resume paths
    (e.g. unit tests, ad-hoc loaders); the YAML dispatch path uses
    the split form so the load is amortized across chunks."""
    bundle = load_bundle(path)
    return extract_batched_init_override(
        bundle, seeds, role=role, burst=burst, load_target=load_target,
    )


__all__ = [
    'QCheckpointBundle',
    'bundle_path',
    'extract_batched_init_override',
    'extract_qcheckpoint',
    'from_batched_record',
    'load_batched_init_override_from_bundle',
    'load_bundle',
    'save_bundle',
]
