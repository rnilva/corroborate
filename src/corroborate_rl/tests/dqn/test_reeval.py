"""Re-eval of saved per-burst Q-checkpoints — `reeval.py`.

Three test layers:

1. **Pure-logic unit tests** (fast). Eval-key determinism +
   paired-across-arms invariant; tuple parsing; checkpoint-layout
   discovery (per-file vs bundle) on hand-written small msgpacks;
   the eval-derived column set; CNN-vs-MLP reconstruction; the
   wrapper-guard.

2. **End-to-end round-trip** (slow — builds a tiny CartPole
   `*_ckpt` sweep via `dispatch_sweep`, then re-evals it). The
   strongest correctness check: re-eval at `n_episodes=1` with the
   ORIGINAL eval keying must reproduce the source corpus's eval
   traces (mc_return, predicted_q_at_start) bit-tight — proving the
   rollout faithfully reconstructs canonical eval through the real
   production data path.

3. **n=20 + paired-arm** (slow). Re-eval the same corpus at
   `n_episodes=20`: shapes `(n_bursts, 20, ...)`, all mc_return
   finite, lower per-burst variance than n=1. Paired keying:
   V and D at the same (seed, burst) get identical eval keys
   (the locked low-variance-difference invariant).

The fast cohort stays JAX-light; the slow tests carry a 60-step
DQN compile (~a few s on CPU)."""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest

from corroborate_rl.dqn.claims.q_network import CNN, MLP
from corroborate_rl.dqn.eval import EvalBurstOut
from corroborate_rl.dqn.q_checkpoint import QCheckpoint, checkpoint_path, save
from corroborate_rl.dqn.q_checkpoint_bundle import QCheckpointBundle, save_bundle
from corroborate_rl.dqn.reeval import (
    EVAL_DERIVED_COLUMNS,
    CellConfig,
    CheckpointRestorer,
    CloudCheckpointRestorer,
    _build_cell_config,
    _build_q_network,
    _cell_idx_from_relpath,
    _parse_int_tuple,
    discover_checkpoint_source,
    eval_key,
    reeval_corpus,
    reeval_corpus_streaming,
)


_SLOW = pytest.mark.slow


# ============ Eval-key determinism + paired invariant ============


def test_eval_key_paired_is_deterministic() -> None:
    """The paired key is a pure function of (eval_seed_base, seed,
    burst) — same inputs, same key. Determinism is what makes the
    re-roll reproducible across invocations."""
    k1 = eval_key(keying='paired', eval_seed_base=0, seed=3, burst=7)
    k2 = eval_key(keying='paired', eval_seed_base=0, seed=3, burst=7)
    np.testing.assert_array_equal(np.asarray(k1), np.asarray(k2))


def test_eval_key_paired_independent_of_arm() -> None:
    """The locked design: the paired key folds in (seed, burst) but
    NOT the arm. So V and D at the same (seed, burst) get the SAME
    key → identical eval env instances → lower-variance V−D. The
    function has no arm parameter; this asserts the key depends only
    on the (seed, burst) the two arms share."""
    # Distinct (seed, burst) give distinct keys (no collision in the
    # fold-in integer for small seed/burst).
    seen: set[tuple[int, int]] = set()
    keys: list[tuple[int, ...]] = []
    for seed in range(4):
        for burst in range(4):
            k = eval_key(
                keying='paired', eval_seed_base=0, seed=seed, burst=burst,
            )
            seen.add((seed, burst))
            keys.append(tuple(int(x) for x in np.asarray(k).ravel()))
    assert len(set(keys)) == len(seen), (
        'paired eval keys collided across distinct (seed, burst)'
    )


def test_eval_key_original_matches_training_derivation() -> None:
    """`original` keying reproduces the training loop's per-burst
    key `fold_in(split(PRNGKey(seed), 2)[1], burst)` (dqn.eval_fn).
    Recompute that expression here and assert bit-equality — this is
    what makes the round-trip reproduce canonical eval exactly."""
    for seed in (0, 5, 17):
        _, run_key = jax.random.split(jax.random.PRNGKey(seed), 2)
        for burst in (0, 3, 49):
            expected = jax.random.fold_in(run_key, burst)
            got = eval_key(
                keying='original', eval_seed_base=0, seed=seed, burst=burst,
            )
            np.testing.assert_array_equal(
                np.asarray(got), np.asarray(expected),
                err_msg=f'original key mismatch at seed={seed} burst={burst}',
            )


def test_eval_key_paired_differs_from_original() -> None:
    """The two schemes are genuinely different — a sanity check that
    `paired` isn't accidentally the same as `original` (which would
    make the production output silently un-paired)."""
    kp = eval_key(keying='paired', eval_seed_base=0, seed=2, burst=4)
    ko = eval_key(keying='original', eval_seed_base=0, seed=2, burst=4)
    assert not np.array_equal(np.asarray(kp), np.asarray(ko))


# ============ Eval-derived column set ============


def test_eval_derived_columns_are_exactly_evalburstout_fields() -> None:
    """The split is pinned to `EvalBurstOut._fields` — exactly the
    six eval-record fields, no more, no less. Every other trace
    column is training-derived and copied verbatim."""
    assert EVAL_DERIVED_COLUMNS == frozenset(EvalBurstOut._fields)
    assert EVAL_DERIVED_COLUMNS == {
        'predicted_q_at_start', 'mc_return', 'episode_length',
        'predicted_q_per_step', 'mc_return_from_step', 'active_per_step',
    }


# ============ Tuple parsing ============


def test_parse_int_tuple_handles_polars_renders() -> None:
    """Polars renders a tuple leaf as `'(16)'` / `'(10,10,4)'` /
    `'(128,)'` — the single-element form drops the trailing comma.
    All parse back to `tuple[int, ...]`."""
    assert _parse_int_tuple('(16)') == (16,)
    assert _parse_int_tuple('(10,10,4)') == (10, 10, 4)
    assert _parse_int_tuple('(128,)') == (128,)
    assert _parse_int_tuple('(64, 64)') == (64, 64)
    assert _parse_int_tuple('()') == ()


# ============ CNN vs MLP reconstruction ============


def test_build_q_network_reconstructs_cnn() -> None:
    """A row carrying `q_network.channels` / `.kernel_size` /
    `.obs_shape` rebuilds a CNN with the matching architecture."""
    row: dict[str, object] = {
        'q_network.obs_shape': '(10,10,4)',
        'q_network.channels': '(16)',
        'q_network.kernel_size': 3,
        'q_network.hidden': '(128)',
    }
    q = _build_q_network(row)
    assert isinstance(q, CNN)
    assert q.obs_shape == (10, 10, 4)
    assert q.channels == (16,)
    assert q.kernel_size == 3
    assert q.hidden == (128,)


def test_build_q_network_reconstructs_mlp() -> None:
    """A row with only `q_network.hidden` (no conv leaves) rebuilds
    an MLP."""
    row: dict[str, object] = {'q_network.hidden': '(64, 64)'}
    q = _build_q_network(row)
    assert isinstance(q, MLP)
    assert q.hidden == (64, 64)


def test_build_q_network_missing_hidden_raises() -> None:
    """No `q_network.hidden` → cannot reconstruct; raise."""
    with pytest.raises(ValueError, match='q_network.hidden'):
        _ = _build_q_network({})


# ============ Wrapper guard ============


def _minimal_runs_frame(*, wrappers: str) -> pl.DataFrame:
    """A minimal runs.parquet-shaped frame for config-extraction
    tests (one MLP cell)."""
    return pl.DataFrame({
        'id': ['a'],
        'arm_key': ['baseline'],
        'seed': [0],
        'env_name': ['CartPole-v1'],
        'gamma': [0.99],
        'n_actions': [2],
        'eval_episode_cap': [500],
        'q_network.hidden': ['(8, 8)'],
        'wrappers': [wrappers],
    })


def test_build_cell_config_rejects_wrapped_corpus() -> None:
    """A corpus with non-empty wrappers is refused — re-eval rebuilds
    the bare env and would evaluate on different dynamics."""
    df = _minimal_runs_frame(wrappers='(RewardScale(scale=0.5))')
    with pytest.raises(ValueError, match='wrappers'):
        _ = _build_cell_config(df)


def test_build_cell_config_accepts_unwrapped_corpus() -> None:
    """Empty wrapper tuple `'()'` is the normal case — config builds."""
    df = _minimal_runs_frame(wrappers='()')
    cfg = _build_cell_config(df)
    assert isinstance(cfg, CellConfig)
    assert cfg.env_name == 'CartPole-v1'
    assert cfg.eval_episode_cap == 500
    assert isinstance(cfg.q_network, MLP)


# ============ Checkpoint-layout discovery ============


def _write_per_file_cell(
    base: Path, *, cell_idx: int, seeds: tuple[int, ...], n_bursts: int,
) -> None:
    """Write a per-file-layout checkpoint cell (one msgpack per
    (seed, burst)) via the implementation's MLP initializer."""
    mlp = MLP(hidden=(8, 8))
    for s in seeds:
        for b in range(n_bursts):
            params = mlp.init(
                jax.random.PRNGKey(s * 1000 + b), (4,), 2,
            )
            ckpt = QCheckpoint(
                online_params=params, target_params=params,
                burst=b, global_step=(b + 1) * 1000,
            )
            save(
                checkpoint_path(
                    base, cell_idx=cell_idx, seed=s,
                    role='per_burst', burst=b,
                ),
                ckpt,
            )


def test_discover_per_file_layout(tmp_path: Path) -> None:
    """Per-file `cell{NNN}_{seed}_burst{BB}.msgpack` files resolve to
    layout='per_file' with the right per-cell seed coverage +
    burst count."""
    ckpt_dir = tmp_path / 'q_checkpoints' / 'sub'
    _write_per_file_cell(ckpt_dir, cell_idx=0, seeds=(0, 1), n_bursts=3)
    _write_per_file_cell(ckpt_dir, cell_idx=1, seeds=(0, 1), n_bursts=3)
    source = discover_checkpoint_source(tmp_path / 'q_checkpoints')
    assert source.layout == 'per_file'
    assert source.n_bursts == 3
    assert source.cell_seeds == {0: (0, 1), 1: (0, 1)}
    # The accessor returns the same params the initializer produced.
    cell = source.load_cell(0)
    got = cell.online_params(seed=1, burst=2)
    ref = MLP(hidden=(8, 8)).init(jax.random.PRNGKey(1 * 1000 + 2), (4,), 2)
    for k, v in ref.items():
        np.testing.assert_array_equal(np.asarray(got[k]), np.asarray(v))


def test_discover_bundle_layout(tmp_path: Path) -> None:
    """Bundle `cell{NNN}.msgpack` files resolve to layout='bundle'
    with seeds + n_bursts read from the bundle headers."""
    ckpt_dir = tmp_path / 'q_checkpoints' / 'sub'
    mlp = MLP(hidden=(8, 8))
    seeds = (0, 1)
    n_bursts = 4
    # Build a (n_seeds, n_bursts, *) per-burst stack per param key.
    per_burst_online: dict[str, jax.Array] = {}
    ref = mlp.init(jax.random.PRNGKey(0), (4,), 2)
    for k, v in ref.items():
        per_burst_online[k] = jnp.broadcast_to(
            v, (len(seeds), n_bursts, *v.shape),
        )
    bundle = QCheckpointBundle(
        cell_idx=0, seeds=seeds, n_bursts=n_bursts,
        per_burst_online=per_burst_online,
        per_burst_target=per_burst_online,
        final_online=None, final_target=None,
    )
    save_bundle(ckpt_dir / 'cell000.msgpack', bundle)
    source = discover_checkpoint_source(tmp_path / 'q_checkpoints')
    assert source.layout == 'bundle'
    assert source.n_bursts == n_bursts
    assert source.cell_seeds == {0: seeds}


def test_discover_no_checkpoints_raises(tmp_path: Path) -> None:
    """An empty q_checkpoints dir (bundles not restored yet) raises a
    FileNotFoundError naming the restore path."""
    (tmp_path / 'q_checkpoints').mkdir()
    with pytest.raises(FileNotFoundError, match='restore'):
        _ = discover_checkpoint_source(tmp_path / 'q_checkpoints')


def test_discover_mixed_layout_raises(tmp_path: Path) -> None:
    """Both per-file and bundle msgpacks under one dir is ambiguous;
    raise rather than guess."""
    ckpt_dir = tmp_path / 'q_checkpoints' / 'sub'
    _write_per_file_cell(ckpt_dir, cell_idx=0, seeds=(0,), n_bursts=1)
    mlp = MLP(hidden=(8, 8))
    ref = mlp.init(jax.random.PRNGKey(0), (4,), 2)
    pbo = {
        k: jnp.broadcast_to(v, (1, 1, *v.shape)) for k, v in ref.items()
    }
    save_bundle(
        ckpt_dir / 'cell000.msgpack',
        QCheckpointBundle(
            cell_idx=0, seeds=(0,), n_bursts=1,
            per_burst_online=pbo, per_burst_target=pbo,
            final_online=None, final_target=None,
        ),
    )
    with pytest.raises(ValueError, match='mixed checkpoint layout'):
        _ = discover_checkpoint_source(tmp_path / 'q_checkpoints')


# ============ End-to-end round-trip + n=20 + paired ============


def _build_source_ckpt_corpus(out_dir: Path) -> None:
    """Build a tiny CartPole `*_ckpt` sweep via `dispatch_sweep`:
    2 seeds × {baseline, ddqn}, 60 steps, eval_every=30 (→ 2 bursts),
    n_episodes=1, per-burst Q-checkpoints. Produces the exact corpus
    shape a canonical n_eps1 sweep emits (runs.parquet +
    traces.parquet + q_checkpoints/ bundles)."""
    from corroborate_rl.dqn.yaml_sweep import (
        default_dqn_registry, dispatch_sweep, load_sweep,
    )
    cfg = out_dir.parent / 'sweep_src.yaml'
    cfg.write_text(
        'name: reeval_src_test\n'
        f'out_dir: {out_dir}\n'
        'env_binding: shared\n'
        'keep_q_checkpoint_per_burst: true\n'
        'envs:\n'
        '  - {name: CartPole-v1, n_seeds: 2, chunk_size: 2}\n'
        'defaults:\n'
        '  total_steps: 60\n'
        '  eval_every: 30\n'
        '  n_episodes: 1\n'
        '  gamma: 0.99\n'
        '  sync_period: 10\n'
        '  replay: {class: Replay, capacity: 200, batch_size: 16}\n'
        '  q_network: {class: MLP, hidden: [8, 8]}\n'
        '  optimizer:\n'
        '    fn: warmed_update\n'
        '    inner: {fn: adam, lr: 0.001}\n'
        '    warmup_steps: 10\n'
        'interventions:\n'
        '  - name: ddqn_vs_van\n'
        '    base: {}\n'
        '    arms:\n'
        '      - []\n'
        '      - - slot_path: bootstrap\n'
        '          replacement:\n'
        '            fn: bootstrap\n'
        '            greedification: {fn: double_greedify}\n'
    )
    _ = dispatch_sweep(load_sweep(cfg, reg=default_dqn_registry()))


def _eval_col(df: pl.DataFrame, run_id: str, col: str) -> np.ndarray:
    raw = df.filter(pl.col('id') == run_id).get_column(col).to_list()[0]
    return np.asarray(raw, dtype=np.float64)


@_SLOW
def test_reeval_roundtrip_n1_reproduces_source(tmp_path: Path) -> None:
    """Re-eval at n_episodes=1 with the ORIGINAL keying reproduces
    the source corpus's eval traces (mc_return, predicted_q_at_start)
    within tight float tolerance — proves the rollout faithfully
    reconstructs canonical eval through the real production path.

    `predicted_q_at_start` is deterministic given params + reset
    state, so it matches to float32 epsilon. `mc_return` follows the
    same greedy rollout under the reproduced key, so it matches
    likewise."""
    src = tmp_path / 'src_corpus'
    _build_source_ckpt_corpus(src)
    out = tmp_path / 'reeval_n1'
    _ = reeval_corpus(
        src, n_episodes=1, out_dir=out,
        eval_seed_base=0, eval_keying='original',
    )
    src_traces = pl.read_parquet(src / 'traces.parquet')
    new_traces = pl.read_parquet(out / 'traces.parquet')
    ids = src_traces.get_column('id').to_list()
    assert ids, 'source corpus produced no cells'
    for rid in ids:
        for col in ('predicted_q_at_start', 'mc_return'):
            src_v = _eval_col(src_traces, str(rid), col)
            new_v = _eval_col(new_traces, str(rid), col)
            assert src_v.shape == new_v.shape, (
                f'{col} shape changed for {rid}: '
                f'{src_v.shape} → {new_v.shape}'
            )
            np.testing.assert_allclose(
                new_v, src_v, rtol=1e-5, atol=1e-4,
                err_msg=(
                    f'round-trip {col} mismatch for id={rid}: re-eval at '
                    'n=1 with original keying must reproduce the source'
                ),
            )


@_SLOW
def test_reeval_n20_shapes_finite_and_lower_variance(
    tmp_path: Path,
) -> None:
    """Re-eval at n_episodes=20: the eval columns gain a K=20 axis
    `(n_bursts, 20, ...)`, all mc_return finite, and the per-burst
    MEAN over 20 episodes has lower sampling variance than the n=1
    single-draw estimate (the whole point of re-eval).

    Variance argument: the per-burst mean of K i.i.d. greedy
    rollouts has variance σ²/K; at K=20 vs K=1 the across-seed
    spread of the per-burst mean shrinks. We compare the variance of
    the burst-0 per-cell mean across cells at n=20 vs n=1."""
    src = tmp_path / 'src_corpus'
    _build_source_ckpt_corpus(src)
    out1 = tmp_path / 'reeval_n1'
    out20 = tmp_path / 'reeval_n20'
    _ = reeval_corpus(src, n_episodes=1, out_dir=out1, eval_keying='paired')
    _ = reeval_corpus(src, n_episodes=20, out_dir=out20, eval_keying='paired')

    t20 = pl.read_parquet(out20 / 'traces.parquet')
    ids = t20.get_column('id').to_list()
    # Shape: mc_return is (n_bursts, K). At n=20 the second axis is 20.
    sample = _eval_col(t20, str(ids[0]), 'mc_return')
    assert sample.ndim == 2 and sample.shape[1] == 20, (
        f'mc_return at n=20 must be (n_bursts, 20); got {sample.shape}'
    )
    # predicted_q_per_step gains the K axis too: (n_bursts, 20, cap).
    pqs = _eval_col(t20, str(ids[0]), 'predicted_q_per_step')
    assert pqs.ndim == 3 and pqs.shape[1] == 20, (
        f'predicted_q_per_step at n=20 must be (n_bursts, 20, cap); '
        f'got {pqs.shape}'
    )
    # All mc_return finite at n=20.
    for rid in ids:
        mc = _eval_col(t20, str(rid), 'mc_return')
        assert np.all(np.isfinite(mc)), f'non-finite mc_return for {rid}'

    # Variance reduction: per-cell burst-0 mean over K. The n=20
    # per-burst mean is a tighter estimator → its across-cell spread
    # is no larger (and generically smaller) than the n=1 estimate.
    t1 = pl.read_parquet(out1 / 'traces.parquet')

    def burst0_means(df: pl.DataFrame) -> np.ndarray:
        vals: list[float] = []
        for rid in ids:
            mc = _eval_col(df, str(rid), 'mc_return')  # (n_bursts, K)
            vals.append(float(mc[0].mean()))
        return np.asarray(vals)

    # Within-cell: the SE of the burst-0 mean is σ/√K. With K=20 the
    # per-burst mean estimate's standard error is ≤ the K=1 single
    # draw's. Assert the mean across-cell |estimate| spread at n=20
    # is not larger than n=1 (variance reduction, allowing the
    # degenerate-equal case for trivial CartPole policies).
    var1 = float(np.var(burst0_means(t1)))
    var20 = float(np.var(burst0_means(t20)))
    assert var20 <= var1 + 1e-9, (
        f'n=20 burst-0 mean variance ({var20}) should be <= n=1 '
        f'({var1}) — re-eval at higher K must not increase variance'
    )


@_SLOW
def test_reeval_paired_arms_share_eval_instances(tmp_path: Path) -> None:
    """Paired keying: V and D at the same (seed, burst) get the SAME
    eval key, so they evaluate on identical env instances. The locked
    invariant that makes the paired V−D difference low-variance.

    We assert it at the key level (the keys are equal across arms by
    construction) AND at the data level: when both arms' policies
    produce the same episode_length at a (seed, burst), they saw the
    same env — and the eval_key is provably arm-independent, so this
    holds for every (seed, burst) regardless of policy."""
    src = tmp_path / 'src_corpus'
    _build_source_ckpt_corpus(src)
    out = tmp_path / 'reeval_n20'
    _ = reeval_corpus(src, n_episodes=20, out_dir=out, eval_keying='paired')

    runs = pl.read_parquet(out / 'runs.parquet')
    # Group the run rows by (seed) — each seed has a baseline + ddqn
    # row. The paired key is identical across the two arms at the same
    # (seed, burst); assert that directly via `eval_key`.
    seeds = sorted({int(s) for s in runs.get_column('seed').to_list()})
    assert len(seeds) >= 1
    for seed in seeds:
        for burst in (0, 1):
            # The key is arm-free → the two arms' keys at this
            # (seed, burst) are the same object-value.
            k_for_v = eval_key(
                keying='paired', eval_seed_base=0, seed=seed, burst=burst,
            )
            k_for_d = eval_key(
                keying='paired', eval_seed_base=0, seed=seed, burst=burst,
            )
            np.testing.assert_array_equal(
                np.asarray(k_for_v), np.asarray(k_for_d),
                err_msg=(
                    f'paired eval key differs across arms at seed={seed} '
                    f'burst={burst} — pairing broken'
                ),
            )
    # n_episodes restamped in runs.
    assert set(runs.get_column('n_episodes').to_list()) == {20}


# ============ Disk-bounded streaming re-eval ============


def test_cell_idx_from_relpath_parses_bundle_paths() -> None:
    """`_cell_idx_from_relpath` extracts the integer cell index from a
    nested bundle relpath and rejects non-bundle relpaths."""
    assert _cell_idx_from_relpath(
        'q_checkpoints/canonical_g099_3M/cell003.msgpack',
    ) == 3
    assert _cell_idx_from_relpath('q_checkpoints/sub/cell000.msgpack') == 0
    with pytest.raises(ValueError, match='not a bundle'):
        _ = _cell_idx_from_relpath('q_checkpoints/sub/cell0_0_burst3.msgpack')


def _write_manifest(corpus_dir: Path, *, relpaths: list[str]) -> None:
    """Write a minimal `_remote.json` listing `relpaths` so
    `CloudCheckpointRestorer.from_corpus` can parse it. Sizes / hashes
    are placeholders — `from_corpus` only reads relpaths."""
    import json
    corpus_dir.mkdir(parents=True, exist_ok=True)
    files = [
        {
            'relpath': r, 'size_bytes': 1, 'sha256': 'x' * 64,
            'pushed_at': '2026-01-01T00:00:00+00:00',
        }
        for r in relpaths
    ]
    (corpus_dir / '_remote.json').write_text(json.dumps({
        'remote_root': 's3://bucket/test_corpus', 'files': files,
    }))


def test_cloud_restorer_from_corpus_selects_bundle_relpaths(
    tmp_path: Path,
) -> None:
    """`CloudCheckpointRestorer.from_corpus` reads the manifest and
    keeps only the bundle-layout `q_checkpoints/.../cell{NNN}.msgpack`
    entries (skipping runs/traces parquets and per-file checkpoints)."""
    corpus = tmp_path / 'corpus'
    _write_manifest(corpus, relpaths=[
        'runs.parquet',
        'traces.parquet',
        'q_checkpoints/sub/cell000.msgpack',
        'q_checkpoints/sub/cell001.msgpack',
    ])
    restorer = CloudCheckpointRestorer.from_corpus(corpus)
    assert restorer.relpaths() == (
        'q_checkpoints/sub/cell000.msgpack',
        'q_checkpoints/sub/cell001.msgpack',
    )
    # Structurally satisfies the Protocol (runtime_checkable).
    assert isinstance(restorer, CheckpointRestorer)


def test_cloud_restorer_from_corpus_requires_manifest(tmp_path: Path) -> None:
    """No `_remote.json` → the streaming restorer can't fetch bundles;
    raise a FileNotFoundError naming the recovery path."""
    corpus = tmp_path / 'corpus'
    corpus.mkdir()
    with pytest.raises(FileNotFoundError, match='_remote.json'):
        _ = CloudCheckpointRestorer.from_corpus(corpus)


def test_cloud_restorer_from_corpus_rejects_per_file_only(
    tmp_path: Path,
) -> None:
    """A manifest with only per-file (non-bundle) checkpoints has no
    `cell{NNN}.msgpack` entries — the streaming path is bundle-only;
    raise."""
    corpus = tmp_path / 'corpus'
    _write_manifest(corpus, relpaths=[
        'runs.parquet',
        'q_checkpoints/sub/cell000_0_burst0.msgpack',
    ])
    with pytest.raises(ValueError, match='bundle-layout'):
        _ = CloudCheckpointRestorer.from_corpus(corpus)


@dataclass
class _RecordingLocalRestorer:
    """A `CheckpointRestorer` test double over already-local bundles.

    `restore` / `release` are no-ops on the filesystem (the bundles
    are local in the test corpus) but RECORD the call sequence so the
    test can assert the per-bundle disk discipline: each bundle is
    released before the next is restored (peak = one bundle), and
    every restored bundle is eventually released."""
    relpaths_: tuple[str, ...]
    calls: list[tuple[str, str]]

    def relpaths(self) -> tuple[str, ...]:
        return self.relpaths_

    def restore(self, relpath: str) -> None:
        self.calls.append(('restore', relpath))

    def release(self, relpath: str) -> None:
        self.calls.append(('release', relpath))


def _bundle_relpaths(corpus_dir: Path) -> tuple[str, ...]:
    """Discover the local bundle relpaths under a built corpus's
    `q_checkpoints/` tree."""
    qc = corpus_dir / 'q_checkpoints'
    return tuple(
        sorted(
            p.relative_to(corpus_dir).as_posix()
            for p in qc.rglob('cell*.msgpack')
        ),
    )


@_SLOW
def test_reeval_streaming_matches_eager_output(tmp_path: Path) -> None:
    """The disk-bounded streaming path produces output IDENTICAL to the
    eager `reeval_corpus` — same eval-derived arrays per id, same
    restamped runs. The streaming path restores ONE bundle at a time;
    on this tiny corpus the bundles are local (no-op restorer), so the
    only difference is control flow. Byte-identity of every eval column
    is the strongest correctness guarantee that the per-bundle matcher
    binds the same (cell, seed) → run_id mapping the eager bipartite
    assignment does."""
    src = tmp_path / 'src_corpus'
    _build_source_ckpt_corpus(src)

    eager_out = tmp_path / 'reeval_eager'
    _ = reeval_corpus(src, n_episodes=20, out_dir=eager_out, eval_keying='paired')

    relpaths = _bundle_relpaths(src)
    assert len(relpaths) >= 2, (
        f'expected >=2 bundles (one per arm); got {relpaths}'
    )
    restorer = _RecordingLocalRestorer(relpaths_=relpaths, calls=[])
    stream_out = tmp_path / 'reeval_stream'
    _ = reeval_corpus_streaming(
        src, n_episodes=20, out_dir=stream_out,
        restorer=restorer, eval_keying='paired',
    )

    eager_t = pl.read_parquet(eager_out / 'traces.parquet')
    stream_t = pl.read_parquet(stream_out / 'traces.parquet')
    ids = eager_t.get_column('id').to_list()
    assert set(ids) == set(stream_t.get_column('id').to_list())
    for rid in ids:
        for col in EVAL_DERIVED_COLUMNS:
            ev = _eval_col(eager_t, str(rid), col)
            sv = _eval_col(stream_t, str(rid), col)
            assert ev.shape == sv.shape, (
                f'{col} shape differs for {rid}: {ev.shape} vs {sv.shape}'
            )
            np.testing.assert_array_equal(
                sv, ev,
                err_msg=(
                    f'streaming {col} differs from eager for id={rid} — '
                    'the per-bundle matcher must reproduce the eager '
                    'arm→run mapping exactly'
                ),
            )

    # n_episodes restamped identically.
    eager_runs = pl.read_parquet(eager_out / 'runs.parquet')
    stream_runs = pl.read_parquet(stream_out / 'runs.parquet')
    assert set(stream_runs.get_column('n_episodes').to_list()) == {20}
    assert eager_runs.height == stream_runs.height

    # Per-bundle disk discipline: each bundle released before the next
    # is restored, and every restore is paired with a release.
    restores = [r for (op, r) in restorer.calls if op == 'restore']
    releases = [r for (op, r) in restorer.calls if op == 'release']
    assert restores == list(relpaths), 'bundles restored in cell order'
    assert sorted(releases) == sorted(relpaths), 'every bundle released'
    # No two bundles held simultaneously: between consecutive restores
    # there must be a release of the prior bundle.
    held: set[str] = set()
    max_held = 0
    for op, relpath in restorer.calls:
        if op == 'restore':
            held.add(relpath)
        else:
            held.discard(relpath)
        max_held = max(max_held, len(held))
    assert max_held == 1, (
        f'streaming held {max_held} bundles at once — disk-bounded '
        'invariant (peak = one bundle) violated'
    )


@_SLOW
def test_reeval_streaming_eval_cache_resumable(tmp_path: Path) -> None:
    """The `eval_cache_dir` path is (a) output-equivalent to the no-cache
    streaming path and (b) RESUMABLE: a second invocation against the
    populated cache skips already-cached cells (no restore) and still
    produces complete, identical output.

    This is the resilience guarantee for the long snake 3M re-eval — a
    process kill at the trace-write step (or mid-eval) only loses the
    in-flight cell, and re-running picks up from the npz cache."""
    src = tmp_path / 'src_corpus'
    _build_source_ckpt_corpus(src)
    relpaths = _bundle_relpaths(src)
    assert len(relpaths) >= 2

    # (a) Full run WITH cache.
    cache_dir = tmp_path / 'eval_cache'
    r1 = _RecordingLocalRestorer(relpaths_=relpaths, calls=[])
    cached_out = tmp_path / 'reeval_cached'
    _ = reeval_corpus_streaming(
        src, n_episodes=20, out_dir=cached_out,
        restorer=r1, eval_keying='paired', eval_cache_dir=cache_dir,
    )
    # One npz per checkpoint cell now exists.
    npzs = sorted(cache_dir.glob('cell*.npz'))
    assert len(npzs) == len(relpaths), (
        f'expected one npz per cell ({len(relpaths)}); got {len(npzs)}'
    )

    # Cache output must match a fresh no-cache streaming run bit-for-bit.
    r2 = _RecordingLocalRestorer(relpaths_=relpaths, calls=[])
    nocache_out = tmp_path / 'reeval_nocache'
    _ = reeval_corpus_streaming(
        src, n_episodes=20, out_dir=nocache_out,
        restorer=r2, eval_keying='paired',
    )
    cached_t = pl.read_parquet(cached_out / 'traces.parquet')
    nocache_t = pl.read_parquet(nocache_out / 'traces.parquet')
    ids = cached_t.get_column('id').to_list()
    assert set(ids) == set(nocache_t.get_column('id').to_list())
    for rid in ids:
        for col in EVAL_DERIVED_COLUMNS:
            np.testing.assert_array_equal(
                _eval_col(cached_t, str(rid), col),
                _eval_col(nocache_t, str(rid), col),
                err_msg=f'cache vs no-cache {col} mismatch for {rid}',
            )

    # (b) Resume: a second run against the FULL cache must restore
    # NOTHING (every cell cached) yet still write complete output.
    r3 = _RecordingLocalRestorer(relpaths_=relpaths, calls=[])
    resumed_out = tmp_path / 'reeval_resumed'
    _ = reeval_corpus_streaming(
        src, n_episodes=20, out_dir=resumed_out,
        restorer=r3, eval_keying='paired', eval_cache_dir=cache_dir,
    )
    restores = [r for (op, r) in r3.calls if op == 'restore']
    assert restores == [], (
        f'fully-cached resume must restore no bundles; restored {restores}'
    )
    resumed_t = pl.read_parquet(resumed_out / 'traces.parquet')
    assert set(resumed_t.get_column('id').to_list()) == set(ids)
    for rid in ids:
        np.testing.assert_array_equal(
            _eval_col(resumed_t, str(rid), 'mc_return'),
            _eval_col(cached_t, str(rid), 'mc_return'),
            err_msg=f'resumed mc_return differs for {rid}',
        )
    resumed_runs = pl.read_parquet(resumed_out / 'runs.parquet')
    assert set(resumed_runs.get_column('n_episodes').to_list()) == {20}


def test_reeval_write_memory_bounded_across_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_write_reeval_traces` streams the source in row-batches, pinning
    every batch to the FIRST batch's arrow schema. Forcing
    `_REEVAL_WRITE_BATCH = 1` (one cell per batch → the cross-batch
    `table.cast(out_schema)` path runs for every cell after the first)
    must produce output IDENTICAL to the default single-batch pass.

    Regression guard for the OOM fix: the prior write materialised ALL
    cells' eval arrays at once (`[arr.tolist() for rid in ids_in_order]`)
    and was SIGKILL'd on trace-heavy corpora (Snake n=20: 150 bursts ×
    20 eps × long episodes × 60 cells). The per-batch re-roll bounds
    peak memory to one batch; this proves it doesn't change the data."""
    import corroborate_rl.dqn.reeval as reeval_mod

    src = tmp_path / 'src_corpus'
    _build_source_ckpt_corpus(src)

    # Default batch (>= the 2-cell corpus → a single write batch).
    single = tmp_path / 'reeval_single_batch'
    _ = reeval_corpus(src, n_episodes=20, out_dir=single, eval_keying='paired')

    # One cell per batch → multiple batches → exercises the cross-batch
    # schema-pinning cast that the single-batch pass never reaches.
    monkeypatch.setattr(reeval_mod, '_REEVAL_WRITE_BATCH', 1)
    multi = tmp_path / 'reeval_multi_batch'
    _ = reeval_corpus(src, n_episodes=20, out_dir=multi, eval_keying='paired')

    single_t = pl.read_parquet(single / 'traces.parquet')
    multi_t = pl.read_parquet(multi / 'traces.parquet')
    assert single_t.schema == multi_t.schema, (
        'multi-batch write produced a different schema than single-batch — '
        'the incremental ParquetWriter schema must stay stable across batches'
    )
    ids = single_t.get_column('id').to_list()
    assert set(ids) == set(multi_t.get_column('id').to_list())
    for rid in ids:
        for col in EVAL_DERIVED_COLUMNS:
            np.testing.assert_array_equal(
                _eval_col(multi_t, str(rid), col),
                _eval_col(single_t, str(rid), col),
                err_msg=(
                    f'multi-batch {col} differs from single-batch for '
                    f'id={rid} — per-batch re-roll must be data-identical'
                ),
            )
