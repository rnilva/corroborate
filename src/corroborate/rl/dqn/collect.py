"""High-level collect primitive — author hypotheses + envs + seeds,
get a parquet pair out.

Replaces the ~300-line `collect_*.py` boilerplate (TRACE_POST_*
exprs, `_run_one_arm` chunked-vmap loop, `_stream_concat` schema
union, `_grid_points` Cartesian iteration) with a single
function call. Each experiment script now authors only:

- list of `Hypothesis` instances (one per intervention × HP grid
  point — the substrate's HP-name → dqn-kwarg authoring lives
  with the hypothesis, not the framework's grid)
- list of env names + per-env seed/chunk-size config
- output directory

`collect_sweep_to_parquet(hypotheses, env_configs, *, out_dir,
trace_reductions, trace_drops)` then:

1. Iterates `hypotheses × env_configs`. Each (hypothesis, env)
   pair is one *arm*.
2. Runs the arm via `DQNRunner`, chunking seeds when the env
   config sets `chunk_size < n_seeds`.
3. Applies trace post-reductions (default: the canonical
   `Q_TRACE_REDUCTIONS` for DQN — per-step max/min/mean/std/
   argmax of Q vectors).
4. Writes per-arm parquets to `tmp/`, then concatenates them
   into `runs.parquet` and `traces.parquet` via
   `stream_concat_parquets` (with type-widening for schema
   conflicts that occur when one arm has int columns where
   another has float).

Returns the final paths (`runs_path`, `traces_path`)."""
from __future__ import annotations

import gc
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import jax

from corroborate.hypothesis import Hypothesis
from corroborate.persistence import (
    apply_trace_reductions,
    stream_concat_parquets,
    write_runrows,
    write_tracerows,
)
from corroborate.rl.dqn.invariants import DQNTrajectoryRecord
from corroborate.rl.dqn.trace_reductions import (
    Q_TRACE_DROPS, Q_TRACE_REDUCTIONS,
)
from corroborate.rl.env_catalogue import EnvSpec
from corroborate.rl.sweep import DQNRunner
from corroborate.sweep import SweepCellResult


@dataclass(frozen=True, slots=True)
class EnvConfig:
    """Per-env sweep config: env name, total seed count, and the
    vmap chunk size for memory management.

    `chunk_size = n_seeds` runs the whole arm in one vmap; smaller
    chunks split the arm into sequential calls (used when the
    env's obs-shape × capacity blows up the f32[cap, n_seeds, obs]
    replay tensor on the GPU)."""
    env_name: str
    n_seeds: int = 30
    chunk_size: int = 30


def _run_one_arm(
    hypothesis: Hypothesis[DQNTrajectoryRecord],
    env_config: EnvConfig, runner: DQNRunner,
    tmp_dir: Path, arm_idx: int, arm_tag: str,
    *,
    trace_reductions: Sequence,
    trace_drops: Sequence[str],
) -> tuple[Path, Path]:
    """Run one (hypothesis, env) arm via DQNRunner. Chunks seeds
    when needed; concatenates per-chunk cells; persists per-arm
    parquets at native shape with the supplied trace reductions
    applied."""
    runs_path = tmp_dir / f'arm{arm_idx:03d}__{arm_tag}__runs.parquet'
    traces_path = tmp_dir / f'arm{arm_idx:03d}__{arm_tag}__traces.parquet'

    seeds = tuple(range(env_config.n_seeds))
    chunks = [
        seeds[i:i + env_config.chunk_size]
        for i in range(0, len(seeds), env_config.chunk_size)
    ]
    all_runs: list = []
    all_traces: list = []
    for chunk in chunks:
        cell_result: SweepCellResult = runner(
            hypothesis,
            {'env_name': env_config.env_name, 'seeds': chunk},
        )
        all_runs.extend(cell_result.runs)
        all_traces.extend(cell_result.traces)
        del cell_result
        jax.clear_caches()
        gc.collect()

    write_runrows(tuple(all_runs), runs_path)
    reduced = apply_trace_reductions(
        list(all_traces),
        add=trace_reductions, drop=trace_drops,
    )
    write_tracerows(reduced, traces_path)

    del all_runs, all_traces, reduced
    jax.clear_caches()
    gc.collect()
    return runs_path, traces_path


def collect_sweep_to_parquet(
    hypotheses: Sequence[Hypothesis[DQNTrajectoryRecord]],
    env_configs: Sequence[EnvConfig],
    *,
    out_dir: Path,
    env_specs: Mapping[str, EnvSpec] | None = None,
    trace_reductions: Sequence = Q_TRACE_REDUCTIONS,
    trace_drops: Sequence[str] = Q_TRACE_DROPS,
    arm_tag: 'callable[[Hypothesis[DQNTrajectoryRecord], EnvConfig], str] | None' = None,
    archive_remote: str | None = None,
) -> tuple[Path, Path]:
    """Run all (hypothesis × env_config) arms, persist per-arm,
    concatenate to `out_dir/{runs,traces}.parquet`. Returns the
    two final paths.

    `env_specs` defaults to looking up each env_config.env_name via
    `corroborate.rl.env_catalogue.get`; pass explicitly to share a
    pre-resolved catalogue across multiple `collect_*` calls.

    `trace_reductions` / `trace_drops` default to the canonical
    `Q_TRACE_REDUCTIONS` / `Q_TRACE_DROPS` (per-step Q reductions
    + drop the 2-D `online_q_per_action` / `target_q_per_action`).
    Override for substrates with different traces.

    `arm_tag` produces the filename suffix for each arm's parquets.
    Default: `{env_name}__{hypothesis.name}` — uniquely identifies
    arms within a sweep but not across HP grid points; callers
    that vary HPs through hypothesis_count should override to
    encode the grid point in the tag.

    `archive_remote`: optional fsspec URI prefix
    (e.g. `s3://corroborate-archive/<sweep>`). When set, each
    arm's tmp parquet pair is uploaded to remote storage right
    after the arm completes, and the local copies are purged —
    bounding peak local-disk usage to ~one arm's worth across
    the whole sweep. The final merge then reads back from the
    remote URIs in the manifest, writes the merged
    `{runs,traces}.parquet` locally, and archives those merged
    outputs (without purging — they stay local for downstream
    analysis)."""
    from corroborate.rl.env_catalogue import get as _get_spec

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / 'tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)
    final_runs = out_dir / 'runs.parquet'
    final_traces = out_dir / 'traces.parquet'

    if env_specs is None:
        env_specs = {
            ec.env_name: _get_spec(ec.env_name)
            for ec in env_configs
        }
    runner = DQNRunner(env_specs)

    if arm_tag is None:
        arm_tag = lambda h, ec: f'{ec.env_name}__{h.name}'  # noqa: E731

    arms: list[tuple[Hypothesis[DQNTrajectoryRecord], EnvConfig]] = [
        (h, ec) for h in hypotheses for ec in env_configs
    ]
    n_total_cells = sum(ec.n_seeds for ec in env_configs) * len(hypotheses)
    print(
        f'sweep: {len(arms)} arms ({len(hypotheses)} hypotheses × '
        f'{len(env_configs)} envs), {n_total_cells} cells total',
        flush=True,
    )

    runs_paths: list[Path] = []
    traces_paths: list[Path] = []
    archived_runs_uris: list[str] = []
    archived_traces_uris: list[str] = []
    t_start = time.time()
    for idx, (h, ec) in enumerate(arms):
        t_arm = time.time()
        tag = arm_tag(h, ec)
        print(
            f'  [{idx+1}/{len(arms)}] {tag} '
            f'(seeds={ec.n_seeds}, chunk={ec.chunk_size}) ...',
            flush=True,
        )
        rp, tp = _run_one_arm(
            h, ec, runner, tmp_dir, idx, tag,
            trace_reductions=trace_reductions,
            trace_drops=trace_drops,
        )
        runs_paths.append(rp)
        traces_paths.append(tp)
        if archive_remote is not None:
            from corroborate.cloud import archive
            rp_rel = rp.relative_to(out_dir).as_posix()
            tp_rel = tp.relative_to(out_dir).as_posix()
            archive(
                out_dir, archive_remote,
                files=[rp_rel, tp_rel], purge_local=True,
            )
            archived_runs_uris.append(
                f'{archive_remote.rstrip("/")}/{rp_rel}'
            )
            archived_traces_uris.append(
                f'{archive_remote.rstrip("/")}/{tp_rel}'
            )
        elapsed = time.time() - t_arm
        total = time.time() - t_start
        print(
            f'    done in {elapsed:.1f}s '
            f'(cumulative {total/60:.1f} min)',
            flush=True,
        )

    print()
    print('merging per-arm parquets ...', flush=True)
    if archive_remote is not None:
        # Local arm parquets were purged; merge directly from remote
        # URIs so peak disk stays bounded by one arm's worth.
        stream_concat_parquets(
            [str(u) for u in archived_runs_uris], final_runs,
        )
        stream_concat_parquets(
            [str(u) for u in archived_traces_uris], final_traces,
        )
        # Archive the merged outputs too — without purging, since
        # the caller usually wants them local for analysis.
        from corroborate.cloud import archive as _archive_merged
        _archive_merged(
            out_dir, archive_remote,
            files=[
                final_runs.relative_to(out_dir).as_posix(),
                final_traces.relative_to(out_dir).as_posix(),
            ],
            purge_local=False,
        )
    else:
        stream_concat_parquets(runs_paths, final_runs)
        stream_concat_parquets(traces_paths, final_traces)
    print(f'  → {final_runs}')
    print(f'  → {final_traces}')
    return final_runs, final_traces


__all__ = ['EnvConfig', 'collect_sweep_to_parquet']
