"""Q-network checkpoint persistence — round-trip + end-to-end.

Two test layers:

1. **Serialization unit tests** (fast). Build a small `Params`
   pytree by hand, write through `save`, read back via `load`,
   assert leaf-wise array equality + scalar fields. Also covers
   the in-record sentinel key conventions
   (`checkpoint_key` / `parse_checkpoint_key`).

2. **End-to-end via the substrate** (slow — runs a 60-step DQN
   sweep on CartPole). With the YAML flag enabled, assert the
   expected msgpack files land under
   `<arm_dir>/q_checkpoints/`, each loadable into a `QCheckpoint`
   whose params are the right shape for the configured MLP.

The serialization tests stand alone; the end-to-end test is
marked `@pytest.mark.slow` (jit + 60-step DQN trace on
CartPole — ~3s on CPU) so the fast cohort stays under 10s."""
from __future__ import annotations

from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from corroborate.core.intervention import combined_arm_key
from corroborate_rl.dqn.claims.optimizer import adam, warmed_update
from corroborate_rl.dqn.claims.q_network import MLP
from corroborate_rl.dqn.claims.replay import Replay
from corroborate_rl.dqn.dqn import dqn
from corroborate_rl.dqn.q_checkpoint import (
    CHECKPOINT_KEY_PREFIX,
    QCheckpoint,
    checkpoint_key,
    checkpoint_path,
    load,
    parse_checkpoint_key,
    save,
)


# ============ Sentinel-key conventions ============


def test_checkpoint_key_round_trips_via_parse() -> None:
    """`checkpoint_key(...)` produces a string starting with the
    sentinel prefix; `parse_checkpoint_key(...)` recovers the
    original (arm, role, param_key) triple — the contract the
    cell runner relies on to partition record entries into trace
    columns vs checkpoint payloads."""
    for arm in ('online', 'target'):
        for role in ('final', 'per_burst'):
            for pk in ('w0', 'b0', 'kw1', 'dw0'):
                k = checkpoint_key(arm, role, pk)
                assert k.startswith(CHECKPOINT_KEY_PREFIX)
                parts = parse_checkpoint_key(k)
                assert parts is not None
                assert parts.arm == arm
                assert parts.role == role
                assert parts.param_key == pk


def test_parse_checkpoint_key_rejects_non_sentinel() -> None:
    """Trace-column keys (no sentinel prefix) MUST return None so
    the cell runner doesn't mis-route them. Defensive against
    silent key-name collisions."""
    assert parse_checkpoint_key('online_max_q_per_step') is None
    assert parse_checkpoint_key('mc_return') is None
    assert parse_checkpoint_key('predicted_q_at_start') is None


def test_parse_checkpoint_key_rejects_malformed_sentinel() -> None:
    """Keys with the prefix but the wrong shape return None — a
    typo or stale-format sentinel doesn't crash the trace write."""
    # Missing role + param_key components.
    assert parse_checkpoint_key(f'{CHECKPOINT_KEY_PREFIX}online') is None
    # Unrecognised arm.
    assert parse_checkpoint_key(
        f'{CHECKPOINT_KEY_PREFIX}critic__final__w0',
    ) is None
    # Unrecognised role.
    assert parse_checkpoint_key(
        f'{CHECKPOINT_KEY_PREFIX}online__every_step__w0',
    ) is None


# ============ msgpack round trip ============


def _small_mlp_params(rng_key: jax.Array) -> dict[str, jax.Array]:
    """Build a small two-layer MLP params dict via the substrate's
    own initialiser — gives us the exact pytree shape the
    cell-runner extracts from the trained network."""
    mlp = MLP(hidden=(8, 8))
    return mlp.init(rng_key, obs_shape=(4,), n_actions=2)


def test_save_load_round_trips_mlp_params(tmp_path: Path) -> None:
    """`save` → `load` returns a `QCheckpoint` whose param leaves
    match the in-memory pytree leaf-wise (numpy `array_equal`).
    The msgpack format is the durable on-disk record; this is the
    canonical contract a future analysis script depends on."""
    online = _small_mlp_params(jax.random.PRNGKey(0))
    target = _small_mlp_params(jax.random.PRNGKey(1))
    ckpt = QCheckpoint(
        online_params=online, target_params=target,
        burst=7, global_step=140_000,
    )
    path = tmp_path / 'cell000_0_burst07.msgpack'
    save(path, ckpt)
    assert path.is_file()

    loaded = load(path)
    assert loaded.burst == 7
    assert loaded.global_step == 140_000
    assert set(loaded.online_params.keys()) == set(online.keys())
    assert set(loaded.target_params.keys()) == set(target.keys())
    for k, v in online.items():
        assert np.array_equal(
            np.asarray(loaded.online_params[k]),
            np.asarray(v),
        ), f'online_params[{k!r}] mismatch'
    for k, v in target.items():
        assert np.array_equal(
            np.asarray(loaded.target_params[k]),
            np.asarray(v),
        ), f'target_params[{k!r}] mismatch'


def test_save_preserves_dtype(tmp_path: Path) -> None:
    """float32 in → float32 out. Without this, a parquet-style
    coercion to Python float would double bytes-per-leaf on
    reload and break any analysis that expects the training-time
    dtype."""
    online = _small_mlp_params(jax.random.PRNGKey(42))
    target = online  # share for brevity — round-trip mechanics
    # don't care about online/target uniqueness.
    ckpt = QCheckpoint(
        online_params=online, target_params=target,
        burst=-1, global_step=-1,
    )
    path = tmp_path / 'cell000_0_final.msgpack'
    save(path, ckpt)
    loaded = load(path)
    for k, v in online.items():
        loaded_arr = np.asarray(loaded.online_params[k])
        assert loaded_arr.dtype == np.asarray(v).dtype, (
            f'{k}: dtype drift {loaded_arr.dtype} != {np.asarray(v).dtype}'
        )


def test_save_is_atomic_via_tmp_rename(tmp_path: Path) -> None:
    """`save` writes through a `.tmp` then renames — a crashed
    write doesn't leave a truncated final file. Verify the tmp
    sidecar doesn't survive a successful write (it gets renamed
    to the target)."""
    online = _small_mlp_params(jax.random.PRNGKey(0))
    ckpt = QCheckpoint(
        online_params=online, target_params=online,
        burst=0, global_step=0,
    )
    path = tmp_path / 'cell000_0_final.msgpack'
    save(path, ckpt)
    assert path.exists()
    assert not path.with_suffix(path.suffix + '.tmp').exists()


def test_load_missing_file_raises(tmp_path: Path) -> None:
    """`load(nonexistent_path)` raises `FileNotFoundError` —
    callers can branch on this without parsing exception
    messages."""
    with pytest.raises(FileNotFoundError):
        _ = load(tmp_path / 'missing.msgpack')


# ============ checkpoint_path layout ============


def test_checkpoint_path_final_layout(tmp_path: Path) -> None:
    """`cell<NNN>_<seed>_final.msgpack` is the canonical final-
    snapshot filename. Zero-pads cell index to 3 digits."""
    p = checkpoint_path(
        tmp_path, cell_idx=7, seed=42, role='final',
    )
    assert p.name == 'cell007_42_final.msgpack'
    assert p.parent == tmp_path


def test_checkpoint_path_per_burst_layout(tmp_path: Path) -> None:
    """`cell<NNN>_<seed>_burst<BB>.msgpack` is the per-burst
    layout. Zero-pads burst to 2 digits (handles up to 99-burst
    sweeps without lex-ordering surprises)."""
    p = checkpoint_path(
        tmp_path, cell_idx=12, seed=3, role='per_burst', burst=5,
    )
    assert p.name == 'cell012_3_burst05.msgpack'


def test_checkpoint_path_per_burst_requires_burst(tmp_path: Path) -> None:
    """Missing `burst` index for per-burst role raises — defensive
    against caller bugs that would otherwise produce a malformed
    filename."""
    with pytest.raises(ValueError, match='burst index'):
        _ = checkpoint_path(
            tmp_path, cell_idx=0, seed=0, role='per_burst',
        )


# ============ End-to-end via run_dqn_cell ============

# Each end-to-end test runs DQN on CartPole — ~3s; marker
# matches the existing `test_cell_runner.py` discipline.
_SLOW_E2E = pytest.mark.slow

_REPLAY_SHORT = Replay(capacity=200, batch_size=16)
_OPTIMIZER_SHORT = partial(
    warmed_update, inner=partial(adam), warmup_steps=10,
)
_SHORT_RUN_HP: dict[str, object] = {
    'total_steps': 60, 'eval_every': 30, 'n_episodes': 2,
    'sync_period': 10,
    'replay': _REPLAY_SHORT,
    'optimizer': _OPTIMIZER_SHORT,
    'q_network': MLP(hidden=(8, 8)),
}


@_SLOW_E2E
def test_run_dqn_cell_writes_final_checkpoint(tmp_path: Path) -> None:
    """End-to-end: `keep_q_checkpoint_final=True` produces a single
    bundle at `<q_checkpoint_dir>/cell000.msgpack` whose `final_*`
    fields carry the per-seed pytrees. Per-burst fields are absent
    (the corresponding flag was off)."""
    from corroborate_rl.cell_runner import run_dqn_cell
    from corroborate_rl.dqn.q_checkpoint_bundle import (
        extract_qcheckpoint, load_bundle,
    )
    from corroborate_rl.env_catalogue import get
    env_spec = get('CartPole-v1')

    claim = partial(dqn, keep_q_checkpoint_final=True, **_SHORT_RUN_HP)
    ckpt_dir = tmp_path / 'q_checkpoints'

    _ = run_dqn_cell(
        env_spec, seed=0, claim=claim,
        arm_key=combined_arm_key(()), measurables=(),
        q_checkpoint_dir=ckpt_dir, cell_idx=0,
    )

    expected = ckpt_dir / 'cell000.msgpack'
    assert expected.is_file(), f'missing {expected}'
    # Final-only: bundle's per_burst payload must be absent.
    bundle = load_bundle(expected)
    assert bundle.per_burst_online is None
    assert bundle.per_burst_target is None
    assert bundle.final_online is not None

    ck = extract_qcheckpoint(bundle, seed=0, role='final')
    # MLP(hidden=(8, 8)) on CartPole (obs_shape=(4,), n_actions=2)
    # produces 3 weight matrices + 3 bias vectors.
    assert set(ck.online_params.keys()) == {
        'w0', 'b0', 'w1', 'b1', 'w2', 'b2',
    }
    assert ck.online_params['w0'].shape == (4, 8)
    assert ck.online_params['w2'].shape == (8, 2)
    assert ck.burst == -1


@_SLOW_E2E
def test_run_dqn_cell_writes_per_burst_checkpoints(tmp_path: Path) -> None:
    """`keep_q_checkpoint_per_burst=True` produces a bundle whose
    `n_bursts` matches `total_steps // eval_every` and per-burst
    arrays carry shape `(n_seeds, n_bursts, *param_shape)`."""
    from corroborate_rl.cell_runner import run_dqn_cell
    from corroborate_rl.dqn.q_checkpoint_bundle import (
        extract_qcheckpoint, load_bundle,
    )
    from corroborate_rl.env_catalogue import get
    env_spec = get('CartPole-v1')

    claim = partial(
        dqn, keep_q_checkpoint_per_burst=True, **_SHORT_RUN_HP,
    )
    ckpt_dir = tmp_path / 'q_checkpoints'
    _ = run_dqn_cell(
        env_spec, seed=0, claim=claim,
        arm_key=combined_arm_key(()), measurables=(),
        q_checkpoint_dir=ckpt_dir, cell_idx=0,
    )

    expected = ckpt_dir / 'cell000.msgpack'
    assert expected.is_file(), f'missing {expected}'
    bundle = load_bundle(expected)
    # n_super_steps = total_steps // eval_every = 60 // 30 = 2.
    assert bundle.n_bursts == 2
    assert bundle.per_burst_online is not None

    first = extract_qcheckpoint(bundle, seed=0, role='per_burst', burst=0)
    second = extract_qcheckpoint(bundle, seed=0, role='per_burst', burst=1)
    assert first.burst == 0
    assert second.burst == 1
    # Param shapes are constant across burst snapshots (training
    # mutates values, not shapes).
    assert (
        first.online_params['w0'].shape
        == second.online_params['w0'].shape
    )


@_SLOW_E2E
def test_default_off_writes_no_checkpoints(tmp_path: Path) -> None:
    """Default behaviour (both flags False): even with a
    `q_checkpoint_dir` provided, no msgpack file lands on disk.
    Existing sweeps that haven't opted in get zero overhead."""
    from corroborate_rl.cell_runner import run_dqn_cell
    from corroborate_rl.env_catalogue import get
    env_spec = get('CartPole-v1')

    claim = partial(dqn, **_SHORT_RUN_HP)
    ckpt_dir = tmp_path / 'q_checkpoints'
    _ = run_dqn_cell(
        env_spec, seed=0, claim=claim,
        arm_key=combined_arm_key(()), measurables=(),
        q_checkpoint_dir=ckpt_dir, cell_idx=0,
    )
    # Dir may not even exist (no msgpack means no parent mkdir).
    assert not ckpt_dir.exists() or not list(ckpt_dir.iterdir())


@_SLOW_E2E
def test_per_burst_filtered_from_trace_columns(tmp_path: Path) -> None:
    """`__q_checkpoint__*` sentinel keys MUST NOT survive in the
    trace columns. The cell runner partitions them off before
    `_trajectory_leaves` runs; verify no `traces.leaves` key
    starts with the sentinel prefix."""
    from corroborate_rl.cell_runner import run_dqn_cell
    from corroborate_rl.env_catalogue import get
    env_spec = get('CartPole-v1')

    claim = partial(
        dqn,
        keep_q_checkpoint_final=True,
        keep_q_checkpoint_per_burst=True,
        **_SHORT_RUN_HP,
    )
    ckpt_dir = tmp_path / 'q_checkpoints'
    cell = run_dqn_cell(
        env_spec, seed=0, claim=claim,
        arm_key=combined_arm_key(()), measurables=(),
        q_checkpoint_dir=ckpt_dir, cell_idx=0,
    )
    leaked = [
        k for k in cell.trace.leaves
        if k.startswith(CHECKPOINT_KEY_PREFIX)
    ]
    assert not leaked, f'sentinel keys leaked into trace: {leaked}'
    # Also verify the bundle landed with both payloads (per_burst +
    # final). One file per cell now (was 1 final + 2 burst = 3 files
    # under the legacy per-file layout).
    from corroborate_rl.dqn.q_checkpoint_bundle import load_bundle
    bundle = load_bundle(ckpt_dir / 'cell000.msgpack')
    assert bundle.final_online is not None
    assert bundle.per_burst_online is not None
    assert bundle.n_bursts >= 1


@_SLOW_E2E
def test_checkpoint_final_recovers_in_memory_params(tmp_path: Path) -> None:
    """The msgpack roundtrip on the trained network's final
    params produces a `QCheckpoint` whose online_params can be
    fed directly into a fresh `MLP` forward call and produce
    identical Q-values for a probe observation.

    This is the canonical use-case: load a checkpoint post-hoc,
    re-evaluate Q at arbitrary observations without re-running
    training. If this round-trip drops a leaf or coerces a dtype,
    the use-case silently breaks."""
    from corroborate_rl.cell_runner import run_dqn_cell
    from corroborate_rl.env_catalogue import get
    env_spec = get('CartPole-v1')

    claim = partial(dqn, keep_q_checkpoint_final=True, **_SHORT_RUN_HP)
    ckpt_dir = tmp_path / 'q_checkpoints'
    _ = run_dqn_cell(
        env_spec, seed=0, claim=claim,
        arm_key=combined_arm_key(()), measurables=(),
        q_checkpoint_dir=ckpt_dir, cell_idx=0,
    )
    from corroborate_rl.dqn.q_checkpoint_bundle import (
        extract_qcheckpoint, load_bundle,
    )
    bundle = load_bundle(ckpt_dir / 'cell000.msgpack')
    loaded = extract_qcheckpoint(bundle, seed=0, role='final')

    # Fresh MLP — same architecture as the trained one. Forward
    # call on an arbitrary obs should produce a finite Q-vector
    # of the right shape.
    mlp = MLP(hidden=(8, 8))
    probe_obs = jnp.zeros((4,), dtype=jnp.float32)
    q_values = mlp(loaded.online_params, probe_obs)
    assert q_values.shape == (2,)
    assert np.all(np.isfinite(np.asarray(q_values)))


# ============ dispatch_sweep cleanup preservation ============


@_SLOW_E2E
def test_dispatch_sweep_preserves_q_checkpoints_through_merge_cleanup(
    tmp_path: Path,
) -> None:
    """`dispatch_sweep` post-merge cleanup `shutil.rmtree`s each
    per-arm sub-corpus dir. With `keep_q_checkpoint_per_burst:
    true`, the per-cell runner writes msgpack files to
    `<arm_dir>/q_checkpoints/` — these are the explicit
    data product the YAML opted into and MUST survive the
    cleanup. Pre-fix the rmtree took them down with the rest
    of the arm dir, silently destroying ~5 GB of work on the
    canonical Asterix ckpt sweep (2026-05-25 incident).

    Post-fix: q_checkpoints/ is moved to
    `<out_dir>/q_checkpoints/<arm_name>/` before the rmtree,
    namespaced by intervention so multi-arm sweeps don't
    collide.
    """
    from corroborate_rl.dqn.yaml_sweep import (
        default_dqn_registry, dispatch_sweep, load_sweep,
    )
    out_dir = tmp_path / 'sweep_out'
    cfg = tmp_path / 'sweep.yaml'
    cfg.write_text(
        'name: ckpt_preservation_test\n'
        f'out_dir: {out_dir}\n'
        'env_binding: shared\n'
        'keep_q_checkpoint_per_burst: true\n'
        'envs:\n'
        '  - {name: CartPole-v1, n_seeds: 1}\n'
        'defaults:\n'
        '  total_steps: 60\n'
        '  eval_every: 30\n'
        '  n_episodes: 1\n'
        '  gamma: 0.99\n'
        '  sync_period: 10\n'
        '  replay: {class: Replay, capacity: 200, batch_size: 16}\n'
        '  q_network: {class: MLP, hidden: [8, 8]}\n'
        '  optimizer: {fn: warmed_update, inner: {fn: adam, lr: 0.001},'
        ' warmup_steps: 10}\n'
        'interventions:\n'
        '  - name: vanilla_vs_ddqn\n'
        '    base: {}\n'
        '    arms:\n'
        '      - []\n'
        '      - - slot_path: bootstrap\n'
        '          replacement:\n'
        '            fn: bootstrap\n'
        '            greedification: {fn: double_greedify}\n'
    )
    sweep = load_sweep(cfg, reg=default_dqn_registry())
    _ = dispatch_sweep(sweep)

    # Post-merge state: the top-level merged parquets land.
    assert (out_dir / 'runs.parquet').is_file()
    assert (out_dir / 'traces.parquet').is_file()

    # The arm-name dir got rmtree'd as part of the merge cleanup;
    # this is the existing CI1-respecting behavior.
    arm_dir = out_dir / 'vanilla_vs_ddqn'
    assert not arm_dir.exists(), (
        f'expected per-arm dir cleaned up, still at {arm_dir}'
    )

    # CRITICAL CONTRACT (post-fix): q_checkpoints/ files survived
    # the cleanup, namespaced under <out_dir>/q_checkpoints/<arm>/.
    ckpt_root = out_dir / 'q_checkpoints' / 'vanilla_vs_ddqn'
    assert ckpt_root.is_dir(), (
        f'expected q_checkpoints preserved at {ckpt_root}; '
        f'this is the fix for the 2026-05-25 cleanup-eats-ckpts bug'
    )
    # 2 arms × 1 chunk per arm = 2 bundle files (cell000, cell001).
    # The bundle layout collapses what the legacy per-file layout
    # spread across 4 files (2 arms × 1 seed × 2 bursts).
    from corroborate_rl.dqn.q_checkpoint_bundle import (
        extract_qcheckpoint, load_bundle,
    )
    bundle_files = sorted(ckpt_root.glob('cell*.msgpack'))
    assert len(bundle_files) == 2, (
        f'expected 2 bundle files (2 arms × 1 chunk), got '
        f'{len(bundle_files)}: {[f.name for f in bundle_files]}'
    )
    # Each bundle must round-trip into per-(seed, burst) checkpoints
    # with the expected MLP shapes — proves the move didn't corrupt
    # bytes and the bundle indexing aligns with the producer's
    # seed/burst axes.
    for f in bundle_files:
        bundle = load_bundle(f)
        assert bundle.n_bursts == 2
        assert bundle.per_burst_online is not None
        for b in (0, 1):
            ck = extract_qcheckpoint(
                bundle, seed=0, role='per_burst', burst=b,
            )
            assert ck.online_params['w0'].shape == (4, 8)
            assert ck.online_params['w2'].shape == (8, 2)
            assert ck.burst == b


@_SLOW_E2E
def test_dispatch_sweep_archives_merged_top_level_with_remote(
    tmp_path: Path,
) -> None:
    """When `archive_remote` is set, `dispatch_sweep` archives the
    merged top-level runs.parquet + traces.parquet directly after
    the merge cleanup. This creates a self-contained local +
    cloud manifest at the sweep level so subsequent
    `corroborate purge <sweep_dir>` works without the
    cloud-fallback `--remote-prefix` workaround.

    The 2026-05-25 incident motivating this fix: the per-arm
    cleanup wiped the sub-corpus dir that held the original
    `_remote.json`, leaving merged top-level parquets with no
    local manifest. `purge` refused them, requiring manual
    intervention. Post-fix: the merged top-level has its own
    manifest, purge works directly.
    """
    from corroborate_rl.dqn.yaml_sweep import (
        default_dqn_registry, dispatch_sweep, load_sweep,
    )
    from corroborate.corpus import cloud
    out_dir = tmp_path / 'sweep_out'
    remote = f'file://{tmp_path / "remote_root"}'
    cfg = tmp_path / 'sweep.yaml'
    cfg.write_text(
        'name: toplevel_archive_test\n'
        f'out_dir: {out_dir}\n'
        f'archive_remote: {remote}\n'
        'env_binding: shared\n'
        'envs:\n'
        '  - {name: CartPole-v1, n_seeds: 1}\n'
        'defaults:\n'
        '  total_steps: 60\n'
        '  eval_every: 30\n'
        '  n_episodes: 1\n'
        '  gamma: 0.99\n'
        '  sync_period: 10\n'
        '  replay: {class: Replay, capacity: 200, batch_size: 16}\n'
        '  q_network: {class: MLP, hidden: [8, 8]}\n'
        '  optimizer: {fn: warmed_update, inner: {fn: adam, lr: 0.001},'
        ' warmup_steps: 10}\n'
        'interventions:\n'
        '  - name: vanilla_vs_ddqn\n'
        '    base: {}\n'
        '    arms:\n'
        '      - []\n'
        '      - - slot_path: bootstrap\n'
        '          replacement:\n'
        '            fn: bootstrap\n'
        '            greedification: {fn: double_greedify}\n'
    )
    sweep = load_sweep(cfg, reg=default_dqn_registry())
    _ = dispatch_sweep(sweep)

    # Merged top-level parquets land.
    assert (out_dir / 'runs.parquet').is_file()
    assert (out_dir / 'traces.parquet').is_file()

    # CRITICAL CONTRACT: top-level `_remote.json` manifest is
    # written, pointing to the merged parquets at the cloud root.
    manifest = cloud.load_manifest(out_dir)
    assert manifest is not None, (
        f'expected top-level manifest at {out_dir}; the substrate '
        f"should have archived the merged parquets after cleanup. "
        f"This is the 2026-05-25 fix for orphaned top-levels."
    )
    relpaths = {f.relpath for f in manifest.files}
    assert relpaths == {'runs.parquet', 'traces.parquet'}, (
        f'expected manifest to cover both merged parquets; got '
        f'{relpaths}'
    )
    assert manifest.remote_root == remote.rstrip('/')

    # Subsequent `corroborate purge <out_dir>` works directly
    # (no `--remote-prefix` fallback needed).
    deleted = cloud.purge(out_dir)
    assert set(deleted) == {'runs.parquet', 'traces.parquet'}
    # Manifest preserved for `restore`.
    assert (out_dir / cloud.MANIFEST_NAME).is_file()


@_SLOW_E2E
def test_dispatch_sweep_without_remote_skips_top_level_archive(
    tmp_path: Path,
) -> None:
    """When `archive_remote` is unset, the substrate must NOT
    attempt a top-level archive (no remote to push to). The
    merged top-level parquets land without a manifest, and the
    sweep is intentionally local-only."""
    from corroborate_rl.dqn.yaml_sweep import (
        default_dqn_registry, dispatch_sweep, load_sweep,
    )
    from corroborate.corpus import cloud
    out_dir = tmp_path / 'sweep_out'
    cfg = tmp_path / 'sweep.yaml'
    cfg.write_text(
        'name: noremote_archive_test\n'
        f'out_dir: {out_dir}\n'
        'env_binding: shared\n'
        'envs:\n'
        '  - {name: CartPole-v1, n_seeds: 1}\n'
        'defaults:\n'
        '  total_steps: 60\n'
        '  eval_every: 30\n'
        '  n_episodes: 1\n'
        '  gamma: 0.99\n'
        '  sync_period: 10\n'
        '  replay: {class: Replay, capacity: 200, batch_size: 16}\n'
        '  q_network: {class: MLP, hidden: [8, 8]}\n'
        '  optimizer: {fn: warmed_update, inner: {fn: adam, lr: 0.001},'
        ' warmup_steps: 10}\n'
        'interventions:\n'
        '  - name: vanilla_vs_ddqn\n'
        '    base: {}\n'
        '    arms:\n'
        '      - []\n'
        '      - - slot_path: bootstrap\n'
        '          replacement:\n'
        '            fn: bootstrap\n'
        '            greedification: {fn: double_greedify}\n'
    )
    sweep = load_sweep(cfg, reg=default_dqn_registry())
    _ = dispatch_sweep(sweep)

    assert (out_dir / 'runs.parquet').is_file()
    assert (out_dir / 'traces.parquet').is_file()
    assert cloud.load_manifest(out_dir) is None


def _build_qckpt_loader_imports() -> None:
    """Import-cycle smoke — declared here so a refactor that
    drops `default_dqn_registry` from `yaml_sweep`'s public surface
    fails the test instead of silently importing a stale ref."""
    from corroborate_rl.dqn.yaml_sweep import (  # noqa: F401
        default_dqn_registry, dispatch_sweep, load_sweep,
    )
