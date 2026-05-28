"""Q-network checkpoint persistence — typed serialize / deserialize
for `Params` pytrees, plus the in-record sentinel-key conventions
the cell runner uses to extract checkpoints from the per-cell record
dict.

**Why this exists.** Substrate authors increasingly want post-hoc
per-state per-action Q evaluation at arbitrary observations (e.g.
re-evaluating the trained policy on a held-out probe set, comparing
Q surfaces across the burst trajectory). Storing the full per-step
per-action Q vector in the per-cell trace blows up parquet size to
O(n_actions × total_steps × seeds × cells). A 50-burst × 60-cell
sweep at CNN[16ch, 128-hidden] saves ~240 MB of checkpoint data
total — manageable, and re-derives every per-state Q probe needed
post-hoc without keeping the JAX kernel resident.

**On-disk format.** msgpack (via `flax.serialization`) of the
plain-dict `Params` pytree. Two pytrees per checkpoint
(`online_params`, `target_params`) plus scalar bookkeeping
(`burst`, `global_step`). msgpack is portable across JAX versions
and language runtimes (any msgpack reader can decode the bytes
into nested dicts of typed arrays), unlike pickle's
Python-version-specific bytecode dependency.

**In-record key convention.** The substrate's `train_with_eval`
returns a flat `dict[str, jax.Array]` record. To pipe stacked
checkpoint arrays through the same channel without breaking the
existing trace-column / measurable boundaries, we reserve the
sentinel key prefix `__q_checkpoint__` for checkpoint payloads:

  `__q_checkpoint__<arm>__<role>__<param_key>`

where `<arm>` is `online` or `target`, `<role>` is `final` (one
snapshot at training end) or `per_burst` (stacked across
`(n_super_steps, ...)`), and `<param_key>` is the original
`Params` dict key (e.g. `w0`, `kw1`).

The cell runner detects these keys, reconstructs the Params
pytree by stripping the prefix and grouping by (arm, role), writes
the msgpack files, and filters the keys out of the trace columns
so they don't blow up `traces.parquet`. Measurables ignore unknown
record keys, so the sentinel-prefixed keys are transparent to the
existing analysis pipeline."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from warnings import deprecated

import jax
import numpy as np
from flax import serialization as _fs

from corroborate_rl.dqn.claims.q_network import Params
from corroborate_rl.dqn.init_override import InitOverride

_LEGACY_DEPRECATION = (
    'Per-(seed, burst|final) msgpack sidecar layout is superseded by '
    'QCheckpointBundle (one msgpack per cell). The bundle path emits '
    '~1 PUT per cell instead of ~765 (15 seeds × 51 snapshots), and '
    'load_bundle / extract_qcheckpoint replace this entry. Use '
    'corroborate_rl.dqn.q_checkpoint_bundle for new code; this entry '
    'is kept for backward compatibility with legacy per-file corpora.'
)


# ============ In-record sentinel keys ============

CHECKPOINT_KEY_PREFIX = '__q_checkpoint__'
"""Sentinel prefix marking a record-dict entry as a Q-checkpoint
payload. The cell runner uses `key.startswith(CHECKPOINT_KEY_PREFIX)`
to partition the record into trace columns vs checkpoint arrays."""


type CheckpointRole = Literal['final', 'per_burst']
"""Whether a checkpoint is the single-snapshot final state or the
per-burst stack of snapshots."""


type CheckpointArm = Literal['online', 'target']
"""Which parameter set the checkpoint payload comes from."""


def checkpoint_key(
    arm: CheckpointArm, role: CheckpointRole, param_key: str,
) -> str:
    """Build the in-record sentinel key for one checkpoint leaf.

    `(arm, role, param_key) -> "__q_checkpoint__<arm>__<role>__<param_key>"`.
    Round-trips with `parse_checkpoint_key`."""
    return f'{CHECKPOINT_KEY_PREFIX}{arm}__{role}__{param_key}'


@dataclass(frozen=True, slots=True)
class CheckpointKeyParts:
    """Parsed components of a checkpoint sentinel key."""
    arm: CheckpointArm
    role: CheckpointRole
    param_key: str


def parse_checkpoint_key(key: str) -> CheckpointKeyParts | None:
    """Inverse of `checkpoint_key`. Returns `None` if `key` is not a
    checkpoint sentinel — caller uses this to discriminate trace
    columns from checkpoint payloads in a single pass.

    Recognises `online` / `target` for arm and `final` / `per_burst`
    for role; any other shape returns `None` (defensive — silently
    skipping malformed keys is preferable to crashing the trace
    write, since the checkpoint payload is auxiliary)."""
    if not key.startswith(CHECKPOINT_KEY_PREFIX):
        return None
    rest = key[len(CHECKPOINT_KEY_PREFIX):]
    parts = rest.split('__', 2)
    if len(parts) != 3:
        return None
    arm_str, role_str, param_key = parts
    if arm_str not in ('online', 'target'):
        return None
    if role_str not in ('final', 'per_burst'):
        return None
    # Both `arm_str` and `role_str` have been narrowed to their
    # respective Literal unions by the membership tests above —
    # pyright auto-narrows them to the typed-union shape, so they
    # pass into the dataclass without an explicit cast.
    return CheckpointKeyParts(
        arm=arm_str, role=role_str, param_key=param_key,
    )


# ============ On-disk record ============

@dataclass(frozen=True, slots=True)
class QCheckpoint:
    """One persisted Q-network checkpoint.

    `online_params` / `target_params` are the materialised
    parameter pytrees at this point in training. `burst` is the
    0-indexed super-step (eval-burst) number (or `-1` for the
    `final` snapshot taken after the last training step). `global_step`
    is the training step count (or `total_steps` for `final`)."""
    online_params: Params
    target_params: Params
    burst: int
    global_step: int

    def as_dict(self) -> dict[str, object]:
        """Plain-Python representation used by msgpack. Arrays are
        kept as numpy ndarrays (msgpack's native binary path)."""
        return {
            'online_params': {
                k: np.asarray(v) for k, v in self.online_params.items()
            },
            'target_params': {
                k: np.asarray(v) for k, v in self.target_params.items()
            },
            'burst': int(self.burst),
            'global_step': int(self.global_step),
        }


# ============ File path conventions ============

@deprecated(_LEGACY_DEPRECATION)
def checkpoint_path(
    base_dir: Path, *, cell_idx: int, seed: int,
    role: CheckpointRole, burst: int | None = None,
) -> Path:
    """Canonical on-disk path for one checkpoint file.

    Layout: `<base_dir>/cell<NNN>_<seed>_<final|burst<BB>>.msgpack`.
    `burst` is required for `role='per_burst'` and ignored for
    `role='final'`."""
    cell_str = f'cell{cell_idx:03d}'
    if role == 'final':
        suffix = 'final'
    else:
        if burst is None:
            raise ValueError(
                'checkpoint_path: role=per_burst requires burst index',
            )
        suffix = f'burst{burst:02d}'
    return base_dir / f'{cell_str}_{seed}_{suffix}.msgpack'


# ============ Serialization ============

@deprecated(_LEGACY_DEPRECATION)
def save(path: Path, ckpt: QCheckpoint) -> None:
    """Write `ckpt` to `path` as msgpack. Creates parent dir if
    needed; atomic via tmp + rename so a crashed write doesn't
    leave a truncated file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _fs.msgpack_serialize(ckpt.as_dict())
    tmp = path.with_suffix(path.suffix + '.tmp')
    _ = tmp.write_bytes(payload)
    tmp.replace(path)


def load(path: Path) -> QCheckpoint:
    """Read a msgpack checkpoint back into a typed `QCheckpoint`.

    msgpack decodes the nested dict structure; we re-cast leaves
    to numpy ndarrays of float32 / int32 (their training-time
    dtypes) so callers can feed them straight into a `q_network`
    forward call without dtype coercion at the use site.

    Raises `FileNotFoundError` if `path` doesn't exist;
    `ValueError` if the bytes decode to an unexpected shape."""
    raw_bytes = path.read_bytes()
    decoded: object = _fs.msgpack_restore(raw_bytes)
    if not isinstance(decoded, Mapping):
        raise ValueError(
            f'{path}: expected msgpack to decode to a mapping; '
            f'got {type(decoded).__name__}',
        )
    online_raw = decoded.get('online_params')
    target_raw = decoded.get('target_params')
    burst_raw = decoded.get('burst')
    step_raw = decoded.get('global_step')
    if (
        not isinstance(online_raw, Mapping)
        or not isinstance(target_raw, Mapping)
        or not isinstance(burst_raw, int)
        or not isinstance(step_raw, int)
    ):
        raise ValueError(
            f'{path}: msgpack payload missing required fields '
            f"(online_params, target_params, burst, global_step) or "
            f"of wrong type: keys={sorted(decoded.keys())!r}",
        )
    online_params: Params = {
        str(k): _as_jax_array(v) for k, v in online_raw.items()
    }
    target_params: Params = {
        str(k): _as_jax_array(v) for k, v in target_raw.items()
    }
    return QCheckpoint(
        online_params=online_params,
        target_params=target_params,
        burst=int(burst_raw),
        global_step=int(step_raw),
    )


def _as_jax_array(v: object) -> jax.Array:
    """Coerce a msgpack-decoded leaf to a `jax.Array`. msgpack
    delivers numpy ndarrays for binary payloads; we wrap them via
    `jnp.asarray` so the returned pytree is consumable by JAX
    forward passes directly."""
    import jax.numpy as jnp
    if isinstance(v, (np.ndarray, jax.Array)):
        return jnp.asarray(v)
    # Scalar / nested-list fallback — msgpack's primitive types
    # round-trip through `jnp.asarray` without dtype loss for
    # uniform-dtype lists.
    return jnp.asarray(v)


# ============ Batched-loading for vmap-over-seeds ============

def _stack_per_seed_params(
    per_seed: Sequence[Params], *, field_label: str,
    ref_seed: int, ref_seeds: Sequence[int],
) -> Params:
    """Structural-uniformity check + leaf-wise stack along axis 0.

    Shared body of `load_batched_online_params` /
    `load_batched_init_override` — the param-dict keys + per-key
    shapes must match across the per-seed loads, otherwise
    `jnp.stack` would raise an opaque error mid-loop. The
    `field_label` makes the error message name which override slot
    (online vs target) diverges."""
    import jax.numpy as jnp
    ref_keys = set(per_seed[0].keys())
    ref_shapes = {k: per_seed[0][k].shape for k in ref_keys}
    for i, params in enumerate(per_seed[1:], start=1):
        keys = set(params.keys())
        if keys != ref_keys:
            raise ValueError(
                f'{field_label}: param-dict keys differ between '
                f'seed {ref_seed} ({sorted(ref_keys)}) '
                f'and seed {ref_seeds[i]} ({sorted(keys)}). '
                f"Checkpoints don't share architecture.",
            )
        for k in ref_keys:
            if params[k].shape != ref_shapes[k]:
                raise ValueError(
                    f'{field_label}: param {k!r} shape differs '
                    f'between seed {ref_seed} ({ref_shapes[k]}) '
                    f'and seed {ref_seeds[i]} ({params[k].shape}).',
                )
    return {
        k: jnp.stack([params[k] for params in per_seed], axis=0)
        for k in ref_keys
    }


@deprecated(_LEGACY_DEPRECATION)
def load_batched_online_params(
    path_template: str, seeds: Sequence[int],
) -> Params:
    """Load one checkpoint per seed, stack `online_params` along
    a leading seed-axis. Returns one `Params` pytree whose leaves
    each carry a `(len(seeds), *param_shape)` leading dim — the
    shape `jax.vmap(..., in_axes=(0, ...))` consumes.

    `path_template` carries a `{seed}` placeholder substituted
    per call. Pytree leaves across the per-seed `online_params`
    dicts must share keys + per-key shapes (the vmap caller
    expects a structurally-uniform batched pytree); a mismatch
    raises `ValueError` naming the divergence so the operator
    sees it before training starts.

    The substrate's `dispatch_sweep` uses this to materialise
    the per-cell init-params pytree for "continue from saved
    checkpoint" interventions; the loaded pytree is threaded
    through `grid_point['init_online_params_batched']` to the
    DQN runner."""
    if not seeds:
        raise ValueError('seeds must be non-empty')
    per_seed: list[Params] = []
    for s in seeds:
        path = Path(path_template.format(seed=s))
        ckpt = load(path)
        per_seed.append(ckpt.online_params)
    return _stack_per_seed_params(
        per_seed, field_label='load_batched_online_params',
        ref_seed=seeds[0], ref_seeds=seeds,
    )


@deprecated(_LEGACY_DEPRECATION)
def load_batched_init_override(
    path_template: str, seeds: Sequence[int], *, load_target: bool,
) -> InitOverride:
    """Load one checkpoint per seed, return an `InitOverride` whose
    `online_params` (and optionally `target_params`) carry the
    per-seed pytrees stacked along a leading seed-axis.

    `load_target=False` is the default — produces an InitOverride
    matching the legacy `load_batched_online_params` semantic
    (target_params left None, so init_state's "target mirrors
    online" path fires).

    `load_target=True` populates both fields from the SAME msgpack
    file's `online_params` + `target_params` entries. The semantic
    shift: the resumed cell starts with the source-trajectory's
    actual (online, target) pair preserving the τ-step staleness
    that DDQN's bias-reduction premise depends on. Use this when
    the experiment asks about the steady-state operator's effect
    on a paired (online, target) attractor, not the early-dynamics
    question (which is answered by load_target=False).

    Architecture-uniformity validation mirrors
    `load_batched_online_params` — under load_target=True, both
    online and target are validated independently."""
    if not seeds:
        raise ValueError('seeds must be non-empty')
    per_seed_online: list[Params] = []
    per_seed_target: list[Params] = []
    for s in seeds:
        path = Path(path_template.format(seed=s))
        ckpt = load(path)
        per_seed_online.append(ckpt.online_params)
        if load_target:
            per_seed_target.append(ckpt.target_params)
    online_stacked = _stack_per_seed_params(
        per_seed_online,
        field_label='load_batched_init_override.online_params',
        ref_seed=seeds[0], ref_seeds=seeds,
    )
    target_stacked = (
        _stack_per_seed_params(
            per_seed_target,
            field_label='load_batched_init_override.target_params',
            ref_seed=seeds[0], ref_seeds=seeds,
        )
        if load_target else None
    )
    return InitOverride(
        online_params=online_stacked,
        target_params=target_stacked,
    )


__all__ = [
    'CHECKPOINT_KEY_PREFIX',
    'CheckpointArm',
    'CheckpointKeyParts',
    'CheckpointRole',
    'QCheckpoint',
    'checkpoint_key',
    'checkpoint_path',
    'load',
    'load_batched_init_override',
    'load_batched_online_params',
    'parse_checkpoint_key',
    'save',
]
