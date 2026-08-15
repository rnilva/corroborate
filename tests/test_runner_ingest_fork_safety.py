"""Parallel corpus-ingest must not deadlock under fork-after-threading.

`runner._load_directory` parallelises multi-corpus ingest across a
`ProcessPoolExecutor`. The measurable registry is populated by
importing implementation modules that transitively import fork-UNSAFE
libraries (numpy/OpenBLAS thread pools, JAX's import-time thread
pool, boto3/fsspec sessions). The CLI imports the hypothesis
module — and thus those libraries — in the PARENT. Forking workers
from that threaded parent (the historical `fork` start method)
clones internal mutexes locked by parent threads that don't exist
in the child; the child's first numpy/JAX/boto call blocks forever
in `futex_wait`.

The fix switches the parallel path to `forkserver` (workers forked
from a clean server process that never imported the heavy libs)
with an `initializer` that re-imports the implementation module(s) so a
fresh worker's registry matches the parent's.

Three guards here:

1. `test_parallel_ingest_uses_fork_safe_start_method` — the
   parallel path requests `forkserver`/`spawn`, NEVER `fork`.
2. `test_parallel_ingest_does_not_deadlock_after_fork_unsafe_import`
   — the hard-hang reproduction. A child process imports a
   fork-unsafe lib (JAX if installed, else a threaded-BLAS warmup)
   THEN runs the parallel ingest; the test fails if it doesn't
   complete within a wall-clock timeout. This is the red test
   against the old `fork` code (where it hangs indefinitely). The
   hang's hardness is environment-dependent (JAX present → certain
   deadlock; BLAS-only sandboxes may not hang on every libc), but
   the fork-after-threading hazard the fix addresses is real
   regardless.
3. `test_parallel_matches_sequential_ingest` — results-equivalence.
   The parallel path must produce the SAME cells + measurable
   values as the sequential path; this catches the registry NOT
   being re-established in workers (which would silently null-out
   the implementation measurable).
"""
from __future__ import annotations

import multiprocessing as mp
import os
from collections.abc import Mapping
from pathlib import Path

import polars as pl
import pytest

from tests import _ingest_fork_fixture as fix

#: Wall-clock budget for a 2-corpus parallel ingest. The real
#: compute is microseconds (sum of a 4-element list); anything
#: approaching this means a forked worker is wedged in futex_wait.
# A genuine deadlock hangs forever, so any finite bound detects it;
# 60s is comfortably above the ~5s a healthy run takes (with headroom
# for CPU contention from concurrent work) yet half the original 120s,
# so a slow-but-not-hung regression still trips it.
_DEADLOCK_TIMEOUT_S = 60.0

#: Per-cell signal arrays. Distinct values per cell so the
#: measurable output discriminates cells (a null-out bug can't
#: accidentally match).
_CELL_SIGNALS: dict[str, list[float]] = {
    'cell-a': [1.0, 2.0, 3.0, 4.0],
    'cell-b': [5.0, 6.0, 7.0, 8.0],
    'cell-c': [10.0, 20.0, 30.0],
    'cell-d': [0.5, 1.5, 2.5, 3.5],
}


def _build_two_corpora(root: Path) -> None:
    """Write two distinct corpora (env_a / env_b) under `root`,
    each with its own runs + traces store."""
    fix.write_corpus(
        root / 'corpus_a', env_name='Env_A',
        cell_signals={
            'cell-a': _CELL_SIGNALS['cell-a'],
            'cell-b': _CELL_SIGNALS['cell-b'],
        },
    )
    fix.write_corpus(
        root / 'corpus_b', env_name='Env_B',
        cell_signals={
            'cell-c': _CELL_SIGNALS['cell-c'],
            'cell-d': _CELL_SIGNALS['cell-d'],
        },
    )


def _ingest(root: Path) -> pl.DataFrame:
    """Run the directory-walk ingest over `root`'s two corpora,
    requiring the fixture measurable. Whether this goes sequential
    or parallel is governed by `CORROBORATE_CACHE_WORKERS` in the
    caller's environment."""
    from corroborate.runner.runner import (
        _load_directory,
    )
    return _load_directory(
        root,
        restore_from_cloud=False,
        required=(fix.MEASURABLE_NAME,),
        bridges=(),
        # Substrate-agnostic default would also work here (the
        # fixture module is registered in this process); pass it
        # explicitly to exercise the threaded `initializer_modules`
        # parameter end-to-end.
        initializer_modules=(fix.__name__,),
    )


# ----------------------------------------------------------------------
# Guard 1: start method is fork-safe.
# ----------------------------------------------------------------------
def test_parallel_ingest_uses_fork_safe_start_method(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The multi-corpus parallel path must request a fork-safe
    start method. Forking from the threaded parent is the
    deadlock; `forkserver` (or `spawn` where unavailable) is the
    fix. We intercept `multiprocessing.get_context` to record the
    requested method and assert it's never `fork`."""
    monkeypatch.setenv('CORROBORATE_CACHE_WORKERS', '2')
    _build_two_corpora(tmp_path)

    import multiprocessing as _mp
    requested: list[str] = []
    real_get_context = _mp.get_context

    def _record_get_context(method: str | None = None) -> object:
        if method is not None:
            requested.append(method)
        return real_get_context(method)

    monkeypatch.setattr(_mp, 'get_context', _record_get_context)

    df = _ingest(tmp_path)

    # The pool path must have been taken (workers forced to 2) and
    # requested a non-fork method.
    pool_methods = [m for m in requested if m in {'fork', 'forkserver', 'spawn'}]
    assert pool_methods, (
        'parallel path did not request a pool start method; '
        f'recorded contexts: {requested}'
    )
    assert 'fork' not in pool_methods, (
        f'parallel ingest still uses bare fork() — deadlock hazard. '
        f'Requested start methods: {pool_methods}'
    )
    assert pool_methods[-1] in {'forkserver', 'spawn'}
    # And it actually ingested both corpora through the pool.
    assert df.height == 4


# ----------------------------------------------------------------------
# Guard 2: no deadlock after a fork-unsafe import in the parent.
# ----------------------------------------------------------------------
def _has_jax() -> bool:
    import importlib.util
    return importlib.util.find_spec('jax') is not None


def _warmup_fork_unsafe_libs() -> None:
    """Spin up fork-unsafe internal thread pools in THIS process so
    a subsequent fork would inherit their locked mutexes. Prefer
    JAX (its import starts a thread pool and it warns that fork
    deadlocks); fall back to a threaded-BLAS matmul via numpy +
    a boto3 session (both hold internal locks)."""
    if _has_jax():
        import jax.numpy as jnp
        a = jnp.ones((128, 128))
        _ = (a @ a).block_until_ready()
        return
    import numpy as np
    # Threaded OpenBLAS GEMM — leaves worker threads holding the
    # BLAS thread-pool mutex.
    m = np.random.default_rng(0).standard_normal((512, 512))
    _ = m @ m
    # boto3 is an optional cloud dep; resolve it dynamically so this
    # file type-checks without it installed. A live session holds
    # internal locks that compound the fork hazard when present.
    import importlib
    import importlib.util
    if importlib.util.find_spec('boto3') is not None:
        boto3 = importlib.import_module('boto3')
        boto3.session.Session()


#: Queue payload: ('ok', cell_count) on success, ('err', message)
#: on failure. A closed-set discriminated tuple keeps the parent's
#: unpack typed without an `object`-narrowing dance.
_IngestResult = tuple[str, int | str]


def _ingest_child(
    root_str: str, result_q: "mp.Queue[_IngestResult]",
) -> None:
    """Top-level (picklable) child body: warm up fork-unsafe libs in
    the child's parent context, then run the parallel ingest. Pushes
    the resulting cell count (or an error string) onto `result_q`.

    Runs as its own process so that (a) a hang doesn't wedge the
    pytest worker and (b) the JAX/BLAS warmup is isolated from the
    rest of the suite."""
    try:
        os.environ['CORROBORATE_CACHE_WORKERS'] = '2'
        _warmup_fork_unsafe_libs()
        df = _ingest(Path(root_str))
        result_q.put(('ok', int(df.height)))
    except BaseException as exc:  # noqa: BLE001 — report, don't crash silently
        result_q.put(('err', f'{type(exc).__name__}: {exc}'))


def test_parallel_ingest_does_not_deadlock_after_fork_unsafe_import(
    tmp_path: Path,
) -> None:
    """Red test against the old `fork` code: import a fork-unsafe
    lib in the parent, then run the multi-corpus parallel ingest,
    bounded by a wall-clock timeout. The OLD code (bare `fork`)
    hangs in `futex_wait` and trips the timeout; the fixed code
    (`forkserver` + registry re-import) completes.

    The child runs under `spawn` from the test so its JAX/BLAS
    warmup can't contaminate the pytest process, and a hang stays
    contained (we `terminate()` on timeout)."""
    _build_two_corpora(tmp_path)

    # `spawn` for the OUTER child so the warmup is isolated from the
    # test process. (The INNER ingest pool picks its own start
    # method — `forkserver` post-fix — independent of this.)
    ctx = mp.get_context('spawn')
    result_q: "mp.Queue[_IngestResult]" = ctx.Queue()
    proc = ctx.Process(target=_ingest_child, args=(str(tmp_path), result_q))
    proc.start()
    proc.join(timeout=_DEADLOCK_TIMEOUT_S)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=10)
        pytest.fail(
            f'parallel ingest did not complete within '
            f'{_DEADLOCK_TIMEOUT_S:.0f}s after a fork-unsafe import in '
            f'the parent — fork-after-threading deadlock reproduced. '
            f'(jax_installed={_has_jax()})'
        )

    assert not result_q.empty(), (
        'ingest child exited without reporting a result '
        f'(exitcode={proc.exitcode})'
    )
    status, payload = result_q.get()
    assert status == 'ok', f'ingest child errored: {payload}'
    assert payload == 4, f'expected 4 cells ingested, got {payload}'


# ----------------------------------------------------------------------
# Guard 3: parallel ≡ sequential (registry actually re-established).
# ----------------------------------------------------------------------
def _ingest_in_child(
    root_str: str, workers: str, result_pickle: str,
) -> None:
    """Run `_ingest` with a forced worker count in an isolated
    process and write the resulting (id, measurable) rows to a
    parquet at `result_pickle`. Isolated so the sequential and
    parallel runs don't share the per-corpus `measurements.parquet`
    sidecar (each gets its own corpus tree)."""
    os.environ['CORROBORATE_CACHE_WORKERS'] = workers
    df = _ingest(Path(root_str))
    df.select(['id', fix.MEASURABLE_NAME]).sort('id').write_parquet(
        result_pickle,
    )


def _run_ingest_isolated(root: Path, workers: str, out: Path) -> None:
    ctx = mp.get_context('spawn')
    proc = ctx.Process(
        target=_ingest_in_child, args=(str(root), workers, str(out)),
    )
    proc.start()
    proc.join(timeout=_DEADLOCK_TIMEOUT_S)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=10)
        pytest.fail(
            f'ingest (workers={workers}) did not complete within '
            f'{_DEADLOCK_TIMEOUT_S:.0f}s'
        )
    assert proc.exitcode == 0, (
        f'ingest child (workers={workers}) exited {proc.exitcode}'
    )


def test_parallel_matches_sequential_ingest(tmp_path: Path) -> None:
    """The parallel (workers=2, forkserver) and sequential
    (workers=1, in-process) paths must agree on the ingested cells
    AND the trace-derived measurable values — the latter is the
    real guard that the worker registry was re-established (a
    missing registry silently null-pads the measurable column).

    Each path gets its OWN corpus tree so neither sees the other's
    per-corpus `measurements.parquet` sidecar. The expected values
    come from the fixture's closed form (`2 * sum(signal)`)."""
    seq_root = tmp_path / 'seq'
    par_root = tmp_path / 'par'
    _build_two_corpora(seq_root)
    _build_two_corpora(par_root)

    seq_out = tmp_path / 'seq.parquet'
    par_out = tmp_path / 'par.parquet'
    # Sequential could run in-process, but run both isolated for a
    # clean apples-to-apples comparison (same import state).
    _run_ingest_isolated(seq_root, '1', seq_out)
    _run_ingest_isolated(par_root, '2', par_out)

    seq_df = pl.read_parquet(seq_out)
    par_df = pl.read_parquet(par_out)

    # Same cells in both.
    assert seq_df['id'].to_list() == ['cell-a', 'cell-b', 'cell-c', 'cell-d']
    assert par_df['id'].to_list() == seq_df['id'].to_list()

    # The measurable column is present + finite in BOTH paths (a
    # dropped registry would null it out in the parallel path only).
    par_vals = par_df[fix.MEASURABLE_NAME].to_list()
    assert all(v is not None for v in par_vals), (
        f'parallel path null-padded {fix.MEASURABLE_NAME} — worker '
        f'registry was NOT re-established: {par_vals}'
    )

    # Parallel ≡ sequential, and both equal the closed form.
    _assert_measurable_matches(seq_df)
    _assert_measurable_matches(par_df)
    assert par_vals == seq_df[fix.MEASURABLE_NAME].to_list()


def _assert_measurable_matches(df: pl.DataFrame) -> None:
    by_id: Mapping[str, object] = dict(
        zip(df['id'].to_list(), df[fix.MEASURABLE_NAME].to_list()),
    )
    for cell_id, signal in _CELL_SIGNALS.items():
        got = by_id[cell_id]
        assert isinstance(got, float)
        expected = fix.expected_value(signal)
        assert got == pytest.approx(expected), (
            f'{cell_id}: framework computed {got}, closed form '
            f'{expected} (= 2 * sum({signal}))'
        )


# ----------------------------------------------------------------------
# Guard 4: the worker registry matches the parent after the initializer.
# ----------------------------------------------------------------------
def _registry_probe_child(
    init_modules: tuple[str, ...], result_q: "mp.Queue[tuple[bool, bool]]",
) -> None:
    """Fresh-interpreter probe isolating the initializer's effect.

    `spawn`/`forkserver` re-import the target's module, which
    transitively imports the fixture — so by the time this runs the
    measurable may already be registered. To test
    `_reestablish_registry` specifically, we first CLEAR the global
    registry (reaching into `_REGISTRY._entries` — acceptable test
    introspection), assert the measurable is then gone, run the
    initializer, and report whether it came back.

    Reports `(absent_after_clear, present_after_init)` — the parent
    asserts both, proving the initializer (not the spawn-reimport)
    is what re-established the registry.

    To make the initializer's `importlib.import_module` actually
    re-execute the fixture (rather than hit the `sys.modules`
    cache the spawn-reimport populated), we evict the fixture from
    `sys.modules` first. This reproduces a genuine `forkserver`
    worker, whose clean server-process parent never imported the
    implementation, so the worker's first import runs the module body
    and re-fires the `@measurable` decorators."""
    import sys
    from corroborate.measurables import registered_names
    from corroborate.measurables.measurable import (
        _REGISTRY,
    )
    from corroborate.runner.runner import (
        _reestablish_registry,
    )
    # Simulate a never-imported implementation: drop the cached module +
    # wipe the registry this process inherited via spawn-reimport,
    # so the only path back to the measurable is the initializer's
    # fresh import.
    for mod in init_modules:
        sys.modules.pop(mod, None)
    _REGISTRY._entries.clear()
    absent_after_clear = fix.MEASURABLE_NAME not in registered_names()
    _reestablish_registry(init_modules)
    present_after_init = fix.MEASURABLE_NAME in registered_names()
    result_q.put((absent_after_clear, present_after_init))


def test_worker_initializer_reestablishes_registry(tmp_path: Path) -> None:
    """Directly verify the worker-side contract: after the
    `_reestablish_registry` initializer runs in a fresh
    interpreter, the implementation measurable is back in the registry
    (the `forkserver`/`spawn` worker would otherwise start empty).

    Uses the registry's own `registry_source_modules()` as the
    default re-import set — the implementation-agnostic recovery surface
    the parallel path defaults to — to prove that default actually
    re-registers the fixture measurable in a clean process."""
    from corroborate.measurables import (
        registered_names,
        registry_source_modules,
    )

    # Parent registry includes the fixture (imported at module top).
    assert fix.MEASURABLE_NAME in registered_names()
    init_modules = registry_source_modules()
    assert fix.__name__ in init_modules, (
        'registry_source_modules() omitted the fixture module that '
        f'defines {fix.MEASURABLE_NAME}; default re-import set is '
        f'incomplete: {init_modules}'
    )

    ctx = mp.get_context('spawn')
    result_q: "mp.Queue[tuple[bool, bool]]" = ctx.Queue()
    proc = ctx.Process(
        target=_registry_probe_child, args=(init_modules, result_q),
    )
    proc.start()
    proc.join(timeout=_DEADLOCK_TIMEOUT_S)
    if proc.is_alive():  # pragma: no cover — defensive timeout
        proc.terminate()
        proc.join(timeout=10)
        pytest.fail('registry-probe child did not complete')
    assert proc.exitcode == 0, f'probe child exited {proc.exitcode}'
    assert not result_q.empty(), 'probe child reported no result'
    absent_after_clear, present_after_init = result_q.get()
    # Sanity: the wipe actually removed the measurable (otherwise the
    # "present after init" assertion below would be vacuous).
    assert absent_after_clear, (
        'probe failed to clear the registry — parity assertion would '
        'be vacuous'
    )
    assert present_after_init, (
        f'{fix.MEASURABLE_NAME} NOT registered after '
        f'_reestablish_registry in a fresh interpreter — the '
        f'initializer failed to re-establish the registry'
    )


# ----------------------------------------------------------------------
# Guard 5: `register_as`-only modules are in the re-import set.
# ----------------------------------------------------------------------
def test_register_as_only_module_in_source_modules() -> None:
    """A module registering measurables ONLY via `register_as` (no
    plain `@measurable`) must appear in `registry_source_modules()` —
    the implementation-agnostic re-import set a `forkserver` / `spawn`
    worker runs to rebuild its (initially empty) registry.

    A `register_as` alias carries the *factory's* `fn.__module__`
    (`corroborate.measurables.reductions`), not the aliasing module,
    so a `fn.__module__`-only scan omits it; a fresh worker would then
    silently fail to re-register the alias and its column null-pads —
    the exact silent-corruption class the fork-safe path exists to
    prevent. The plain-`@measurable` fixture used by guards 1-4 is
    captured via `fn.__module__` and does NOT exercise this path; this
    one (mirroring `trace_reductions.py`) does. Red against the pre-fix
    code, which only scanned `fn.__module__`.

    Substrate-grounded: real `register_as` → real registry → real
    `registry_source_modules()`; the assertion is on the framework's
    module-capture transform, not a stamped read-back."""
    import tests._register_as_only_fixture as ra_fix
    from corroborate.measurables import (
        registered_names,
        registry_source_modules,
    )

    # Importing the fixture registered its alias as a side effect.
    assert ra_fix.ALIAS_NAME in registered_names(), (
        'register_as fixture did not register its alias'
    )
    mods = registry_source_modules()
    assert ra_fix.__name__ in mods, (
        f'register_as-only module {ra_fix.__name__!r} absent from the '
        f'forkserver re-import set — its alias would null-pad in a '
        f'fresh worker. Set: {mods}'
    )
