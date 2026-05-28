"""Q-checkpoint bundle — round-trip + extraction + batched-load.

Three test layers:

1. **Round-trip** (fast). Construct a `QCheckpointBundle` from a
   synthetic batched record, save, load, assert leaf-wise array
   equality + bookkeeping fields.

2. **Extraction** (fast). Given a bundle, slice out individual
   `QCheckpoint` instances by (seed, role[, burst]) and assert
   the right shapes / values come out.

3. **Batched loader** (fast). The `load_batched_init_override_from_bundle`
   re-stacks per-seed pytrees the same way the legacy per-file loader
   does — assert the returned `InitOverride` carries the right
   leading-axis shape for `(n_seeds, *param_shape)` per leaf.

All three use synthetic-input bundles (constructed in-Python from
small np.arrays). The end-to-end "substrate emits a bundle and a
new sweep can resume from it" path is covered by `test_cell_runner.py`'s
integration setup — kept separate so this fast cohort stays under
a second."""
from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from corroborate_rl.dqn.q_checkpoint_bundle import (
    QCheckpointBundle,
    bundle_path,
    extract_batched_init_override,
    extract_qcheckpoint,
    from_batched_record,
    load_batched_init_override_from_bundle,
    load_bundle,
    save_bundle,
)


# ============ Synthetic-bundle fixtures ============

def _synthetic_bundle(
    *,
    cell_idx: int = 0,
    n_seeds: int = 3,
    n_bursts: int = 4,
    n_features: int = 5,
) -> QCheckpointBundle:
    """Build a bundle whose arrays carry deterministic content so
    extraction can assert exact equality at every (seed, burst).

    Per-burst online[s, b, i] = 1000*s + 10*b + i (uniquely
    identifies the slice). Target gets +0.5 to keep online/target
    distinguishable. Final online gets a different offset to keep
    final/per_burst slices distinguishable."""
    rng = np.arange  # deterministic deterministically — every test seed identical
    per_burst_online = {
        'w0': (
            1000 * rng(n_seeds).reshape(-1, 1, 1)
            + 10 * rng(n_bursts).reshape(1, -1, 1)
            + rng(n_features).reshape(1, 1, -1)
        ).astype(np.float32),
    }
    per_burst_target = {
        'w0': per_burst_online['w0'] + 0.5,
    }
    final_online = {
        'w0': (
            1_000_000 + 1000 * rng(n_seeds).reshape(-1, 1)
            + rng(n_features).reshape(1, -1)
        ).astype(np.float32),
    }
    final_target = {
        'w0': final_online['w0'] + 0.5,
    }
    return QCheckpointBundle(
        cell_idx=cell_idx,
        seeds=tuple(int(s) for s in rng(n_seeds)),
        n_bursts=n_bursts,
        per_burst_online={k: jnp.asarray(v) for k, v in per_burst_online.items()},
        per_burst_target={k: jnp.asarray(v) for k, v in per_burst_target.items()},
        final_online={k: jnp.asarray(v) for k, v in final_online.items()},
        final_target={k: jnp.asarray(v) for k, v in final_target.items()},
    )


# ============ Round-trip ============

def test_save_load_round_trip(tmp_path: Path) -> None:
    """A bundle written then re-read must compare leaf-wise equal —
    no msgpack quirk silently truncating or upcasting an ndarray."""
    bundle = _synthetic_bundle()
    path = bundle_path(tmp_path, cell_idx=bundle.cell_idx)
    save_bundle(path, bundle)
    loaded = load_bundle(path)

    assert loaded.cell_idx == bundle.cell_idx
    assert loaded.seeds == bundle.seeds
    assert loaded.n_bursts == bundle.n_bursts
    assert bundle.per_burst_online is not None
    assert loaded.per_burst_online is not None
    np.testing.assert_array_equal(
        np.asarray(loaded.per_burst_online['w0']),
        np.asarray(bundle.per_burst_online['w0']),
    )
    assert bundle.final_online is not None
    assert loaded.final_online is not None
    np.testing.assert_array_equal(
        np.asarray(loaded.final_online['w0']),
        np.asarray(bundle.final_online['w0']),
    )


def test_save_load_no_per_burst(tmp_path: Path) -> None:
    """A bundle with only `final_*` set (per_burst flag was off)
    round-trips with `n_bursts=0` and `per_burst_* is None`."""
    bundle = QCheckpointBundle(
        cell_idx=7,
        seeds=(0, 1),
        n_bursts=0,
        per_burst_online=None,
        per_burst_target=None,
        final_online={'w0': jnp.asarray(np.array([[1.0], [2.0]], dtype=np.float32))},
        final_target={'w0': jnp.asarray(np.array([[3.0], [4.0]], dtype=np.float32))},
    )
    path = bundle_path(tmp_path, cell_idx=bundle.cell_idx)
    save_bundle(path, bundle)
    loaded = load_bundle(path)
    assert loaded.per_burst_online is None
    assert loaded.per_burst_target is None
    assert loaded.n_bursts == 0


# ============ Bundle path convention ============

def test_bundle_path_layout(tmp_path: Path) -> None:
    """Bundle file name is `cell{NNN}.msgpack` under the base dir —
    same naming shape the archive walker picks up under SIDECAR_DIRS."""
    p = bundle_path(tmp_path, cell_idx=42)
    assert p.name == 'cell042.msgpack'
    assert p.parent == tmp_path


# ============ Extraction ============

def test_extract_per_burst(tmp_path: Path) -> None:
    """`extract_qcheckpoint(role='per_burst', burst=b)` must slice
    axis 0 by seed and axis 1 by burst — the deterministic content
    `1000*s + 10*b + i` lets us assert the right slice came out."""
    bundle = _synthetic_bundle(n_seeds=3, n_bursts=4, n_features=5)
    ckpt = extract_qcheckpoint(bundle, seed=2, role='per_burst', burst=3)
    expected_online = (1000 * 2 + 10 * 3 + np.arange(5)).astype(np.float32)
    np.testing.assert_array_equal(
        np.asarray(ckpt.online_params['w0']),
        expected_online,
    )
    np.testing.assert_array_equal(
        np.asarray(ckpt.target_params['w0']),
        expected_online + 0.5,
    )
    assert ckpt.burst == 3


def test_extract_final(tmp_path: Path) -> None:
    """`role='final'` slices axis 0 by seed only; the deterministic
    content `1_000_000 + 1000*s + i` confirms the right seed slice."""
    bundle = _synthetic_bundle(n_seeds=3, n_features=5)
    ckpt = extract_qcheckpoint(bundle, seed=1, role='final')
    expected = (1_000_000 + 1000 * 1 + np.arange(5)).astype(np.float32)
    np.testing.assert_array_equal(
        np.asarray(ckpt.online_params['w0']),
        expected,
    )
    assert ckpt.burst == -1


def test_extract_missing_seed_raises() -> None:
    """Requesting a seed not in `bundle.seeds` raises KeyError —
    catches caller typos before they propagate as a silent
    wrong-slice."""
    bundle = _synthetic_bundle(n_seeds=3)
    with pytest.raises(KeyError, match='seed 99 not in bundle.seeds'):
        _ = extract_qcheckpoint(bundle, seed=99, role='final')


def test_extract_per_burst_out_of_range_raises() -> None:
    """`burst >= bundle.n_bursts` raises ValueError naming the range."""
    bundle = _synthetic_bundle(n_bursts=4)
    with pytest.raises(ValueError, match='out of range'):
        _ = extract_qcheckpoint(bundle, seed=0, role='per_burst', burst=4)


def test_extract_per_burst_without_burst_raises() -> None:
    """`role='per_burst'` requires `burst` — None raises."""
    bundle = _synthetic_bundle()
    with pytest.raises(ValueError, match='burst index'):
        _ = extract_qcheckpoint(bundle, seed=0, role='per_burst')


def test_extract_final_when_bundle_lacks_final_raises() -> None:
    bundle = QCheckpointBundle(
        cell_idx=0, seeds=(0,), n_bursts=1,
        per_burst_online={'w0': jnp.zeros((1, 1, 1))},
        per_burst_target={'w0': jnp.zeros((1, 1, 1))},
        final_online=None, final_target=None,
    )
    with pytest.raises(ValueError, match='no final snapshot'):
        _ = extract_qcheckpoint(bundle, seed=0, role='final')


# ============ Batched-load (the resume path) ============

def test_load_batched_init_override_from_bundle(tmp_path: Path) -> None:
    """`load_batched_init_override_from_bundle` must stack the
    requested seeds along axis 0 with the same leaf shape the
    per-file loader produces. Subset + reorder are both supported."""
    bundle = _synthetic_bundle(n_seeds=4, n_bursts=2, n_features=3)
    path = bundle_path(tmp_path, cell_idx=bundle.cell_idx)
    save_bundle(path, bundle)

    override = load_batched_init_override_from_bundle(
        path, seeds=(2, 0, 3), role='per_burst', burst=1,
        load_target=True,
    )
    assert override.online_params is not None
    assert override.target_params is not None
    assert override.online_params['w0'].shape == (3, 3)
    # Row 0 in the override is seed 2 burst 1
    np.testing.assert_array_equal(
        np.asarray(override.online_params['w0'][0]),
        (1000 * 2 + 10 * 1 + np.arange(3)).astype(np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(override.online_params['w0'][1]),
        (1000 * 0 + 10 * 1 + np.arange(3)).astype(np.float32),
    )


def test_load_batched_without_target(tmp_path: Path) -> None:
    """`load_target=False` leaves `target_params` None (legacy
    semantics: target mirrors online via init_state's fresh path)."""
    bundle = _synthetic_bundle()
    path = bundle_path(tmp_path, cell_idx=bundle.cell_idx)
    save_bundle(path, bundle)
    override = load_batched_init_override_from_bundle(
        path, seeds=(0, 1), role='final',
        load_target=False,
    )
    assert override.online_params is not None
    assert override.target_params is None


# ============ Construction from batched record ============

def test_from_batched_record_empty_returns_none() -> None:
    """`from_batched_record` returns None when the record carries
    no checkpoint sentinel keys — the off-by-default path matches
    the legacy `_save_q_checkpoints` no-op semantics."""
    record = {'reward': jnp.zeros((3, 10)), 'td_error': jnp.zeros((3, 10))}
    result = from_batched_record(record, cell_idx=0, seeds=(0, 1, 2))
    assert result is None


def test_from_batched_record_partial_arm_raises() -> None:
    """One arm of `final` without the other is a producer-side bug;
    surface loudly rather than write a half-checkpoint that won't
    round-trip into the (online, target) pair InitOverride expects."""
    from corroborate_rl.dqn.q_checkpoint import checkpoint_key
    record = {
        checkpoint_key('online', 'final', 'w0'): jnp.zeros((3, 5)),
        # target missing
    }
    with pytest.raises(ValueError, match='final has one arm'):
        _ = from_batched_record(record, cell_idx=0, seeds=(0, 1, 2))


def test_from_batched_record_builds_bundle() -> None:
    """A record carrying online+target × final+per_burst is grouped
    into a complete bundle with both `n_bursts > 0` and `final_*`
    set. Axis shapes are preserved verbatim from the input record."""
    from corroborate_rl.dqn.q_checkpoint import checkpoint_key
    n_seeds, n_bursts, n_features = 3, 4, 5
    pb = np.arange(
        n_seeds * n_bursts * n_features, dtype=np.float32,
    ).reshape(n_seeds, n_bursts, n_features)
    fn = np.arange(
        n_seeds * n_features, dtype=np.float32,
    ).reshape(n_seeds, n_features)
    record = {
        checkpoint_key('online', 'per_burst', 'w0'): jnp.asarray(pb),
        checkpoint_key('target', 'per_burst', 'w0'): jnp.asarray(pb + 0.5),
        checkpoint_key('online', 'final', 'w0'): jnp.asarray(fn),
        checkpoint_key('target', 'final', 'w0'): jnp.asarray(fn + 0.5),
    }
    bundle = from_batched_record(record, cell_idx=7, seeds=(10, 20, 30))
    assert bundle is not None
    assert bundle.cell_idx == 7
    assert bundle.seeds == (10, 20, 30)
    assert bundle.n_bursts == n_bursts
    assert bundle.per_burst_online is not None
    np.testing.assert_array_equal(
        np.asarray(bundle.per_burst_online['w0']), pb,
    )
    assert bundle.final_online is not None
    np.testing.assert_array_equal(
        np.asarray(bundle.final_online['w0']), fn,
    )


# ============ __post_init__ invariants ============

def test_post_init_rejects_duplicate_seeds() -> None:
    """Bundle construction rejects duplicate seeds — protects
    extract_qcheckpoint's `seeds.index(s)` from silently returning
    the same slice twice on a corrupt bundle."""
    with pytest.raises(ValueError, match='seeds must be unique'):
        _ = QCheckpointBundle(
            cell_idx=0,
            seeds=(0, 1, 1),
            n_bursts=0,
            per_burst_online=None, per_burst_target=None,
            final_online={'w0': jnp.zeros((3, 2))},
            final_target={'w0': jnp.zeros((3, 2))},
        )


def test_post_init_rejects_inconsistent_n_bursts_with_payload() -> None:
    """n_bursts > 0 with per_burst_*=None (or vice versa) is a
    contradictory state — surface at construction."""
    with pytest.raises(ValueError, match='per_burst_. is None'):
        _ = QCheckpointBundle(
            cell_idx=0, seeds=(0,), n_bursts=4,
            per_burst_online=None, per_burst_target=None,
            final_online=None, final_target=None,
        )
    with pytest.raises(ValueError, match='n_bursts=0'):
        _ = QCheckpointBundle(
            cell_idx=0, seeds=(0,), n_bursts=0,
            per_burst_online={'w0': jnp.zeros((1, 1, 2))},
            per_burst_target={'w0': jnp.zeros((1, 1, 2))},
            final_online=None, final_target=None,
        )


def test_post_init_rejects_seed_axis_shape_mismatch() -> None:
    """Per-leaf axis-0 must equal len(seeds) — catches a producer
    that mistakenly emits arrays for the wrong batch size."""
    with pytest.raises(ValueError, match='axis-0='):
        _ = QCheckpointBundle(
            cell_idx=0, seeds=(0, 1, 2), n_bursts=0,
            per_burst_online=None, per_burst_target=None,
            # Only 2 seeds in the array but seeds tuple has 3.
            final_online={'w0': jnp.zeros((2, 4))},
            final_target={'w0': jnp.zeros((2, 4))},
        )


def test_post_init_rejects_cross_leaf_burst_inconsistency() -> None:
    """When per_burst arrays disagree on axis-1 (the burst dim),
    construction must refuse — catches the multi-leaf bug where
    one param key has 50 bursts and another has 49."""
    with pytest.raises(ValueError, match='axis-1='):
        _ = QCheckpointBundle(
            cell_idx=0, seeds=(0, 1), n_bursts=4,
            # w0 has 4 bursts (matches n_bursts), b0 has 3 (doesn't).
            per_burst_online={
                'w0': jnp.zeros((2, 4, 8)),
                'b0': jnp.zeros((2, 3, 8)),
            },
            per_burst_target={
                'w0': jnp.zeros((2, 4, 8)),
                'b0': jnp.zeros((2, 4, 8)),
            },
            final_online=None, final_target=None,
        )


# ============ bool-as-int rejection in load_bundle ============

def test_load_bundle_rejects_bool_cell_idx(tmp_path: Path) -> None:
    """`isinstance(True, int) is True` quirk — load_bundle must
    reject bool to refuse a hand-crafted or corrupt msgpack whose
    cell_idx decodes as True/False."""
    from flax import serialization as _fs
    path = tmp_path / 'cell000.msgpack'
    _ = path.write_bytes(_fs.msgpack_serialize({
        'cell_idx': True, 'seeds': [0], 'n_bursts': 0,
        'per_burst_online': None, 'per_burst_target': None,
        'final_online': None, 'final_target': None,
    }))
    with pytest.raises(ValueError, match='cell_idx'):
        _ = load_bundle(path)


def test_load_bundle_rejects_bool_n_bursts(tmp_path: Path) -> None:
    from flax import serialization as _fs
    path = tmp_path / 'cell000.msgpack'
    _ = path.write_bytes(_fs.msgpack_serialize({
        'cell_idx': 0, 'seeds': [0], 'n_bursts': False,
        'per_burst_online': None, 'per_burst_target': None,
        'final_online': None, 'final_target': None,
    }))
    with pytest.raises(ValueError, match='n_bursts'):
        _ = load_bundle(path)


# ============ Pre-flight seed-subset + uniqueness in resume ============

def test_extract_batched_init_override_rejects_duplicate_request(
    tmp_path: Path,
) -> None:
    """Requesting duplicate seeds would build identical vmap lanes
    (pseudo-replication leaks into n-seed-paired downstream
    parquet); fail fast at the batched-extract boundary."""
    bundle = _synthetic_bundle(n_seeds=4)
    with pytest.raises(ValueError, match='seeds must be unique'):
        _ = extract_batched_init_override(
            bundle, seeds=(0, 0, 1), role='final', load_target=False,
        )


def test_extract_batched_init_override_rejects_missing_seed(
    tmp_path: Path,
) -> None:
    """Pre-flight subset check surfaces the bundle-chunk mismatch
    BEFORE the partial-extract work — caller gets a clear
    diagnostic naming the missing seeds and the bundle's actual
    seed set."""
    bundle = _synthetic_bundle(n_seeds=3)
    with pytest.raises(ValueError, match='not present in bundle'):
        _ = extract_batched_init_override(
            bundle, seeds=(0, 99), role='final', load_target=False,
        )


# ============ save_bundle atomicity ============

def test_save_bundle_is_atomic_via_tmp_rename(tmp_path: Path) -> None:
    """save_bundle must use the tmp+rename pattern — the .tmp
    sidecar must NOT survive a successful write. A regression to
    write-direct (no tmp) would leave a torn bundle on a crashed
    write, which load_bundle's downstream readers would silently
    accept as a non-existent path AND then truncate."""
    bundle = _synthetic_bundle()
    path = bundle_path(tmp_path, cell_idx=bundle.cell_idx)
    save_bundle(path, bundle)
    assert path.exists()
    assert not path.with_suffix(path.suffix + '.tmp').exists()


# ============ bundle_path parameter no longer shadows ============

def test_load_batched_init_override_from_bundle_no_shadow(
    tmp_path: Path,
) -> None:
    """The deprecated-API entry takes a `path: Path` (was
    `bundle_path: Path` — shadowed the module-level
    bundle_path() function). Smoke that the rename didn't break
    the deprecated wrapper."""
    bundle = _synthetic_bundle(n_seeds=2, n_bursts=2)
    p = bundle_path(tmp_path, cell_idx=bundle.cell_idx)
    save_bundle(p, bundle)
    override = load_batched_init_override_from_bundle(
        p, seeds=(0, 1), role='final', load_target=False,
    )
    assert override.online_params is not None
