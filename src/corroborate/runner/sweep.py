"""Sweep — paired-intervention exogenous-grid runner.

Substrate-agnostic primitive: takes a `DoEffect` (the typed
contrast), a substrate-provided `base` Callable (the substrate's
theory pre-bound with HPs), an optional `measurables` tuple (for
eager per-cell scalar persistence), a discrete `grid_points`
sequence, and a `Runner` that knows how to execute one cell.
The framework composes treatment + baseline claims via
`apply_interventions(base, intervention_tuple)`, iterates the
grid, and dispatches both arms to the runner per grid point.

**Paired-sweep-per-intervention.** Each call to `run_intervention`
runs both arms of one `DoEffect` in lockstep. Treatment and
baseline cells share the same `grid_point` — pairing is
intrinsic, not reconstructed post-hoc via arm_key match.

Multi-arm sweeps (3-way, factorial) are expressed as multiple
calls to `run_intervention`, each with its own DoEffect. Vanilla
cells appear in each baseline-shared sweep — that's the cost of
the self-contained-contrast discipline (see
`project_multi_arm_intervention_primitive` memory).

The framework knows nothing about RL concepts (`env`, `seed`,
`total_steps`). Those are exogenous *names the substrate chose*.
A non-RL substrate sweeping over (`patient_id`, `dose`,
`measurement_day`) uses the same primitive — different
`exogenous_grid` and `Runner`.

**Arm identity** flows from the Intervention tuples, NOT from
`canonical_str(claim)`. The framework derives
`arm_key = combined_arm_key(intervention_tuple)` and passes it
through to the runner; the runner sets `RunRow.arm_key` to
this value. HPs baked into `base` thus do NOT distinguish arms
— they're cell covariates per CLAUDE.md's leaves-as-covariates
discipline."""
from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import polars as pl

from corroborate.core.intervention import DoEffect, apply_interventions
from corroborate.corpus.persistence import (
    apply_trace_reductions,
    stream_concat_parquets,
    read_graphs_sidecar,
    write_graphs_sidecar,
    write_runrows,
    write_tracerows,
)
from corroborate.corpus.schema import RunRow, TraceRow
from corroborate.graph import Graph
from corroborate.graph.computation import ComputationGraph
from corroborate.measurables import Measurable


@dataclass(frozen=True, slots=True)
class CellFailure:
    """One cell that raised during execution. Captures the arm key,
    grid-point values, and the exception's string representation.
    Sweeps return failures alongside successful results so callers
    see the gap explicitly (no silent drops)."""
    arm_key: str
    grid_point: Mapping[str, object]
    error: str
    duration_s: float


@dataclass(frozen=True, slots=True)
class SweepCellResult:
    """One runner-call's output: per-seed records + the
    arm-level computation graph captured during the call.

    Substrates that don't capture a graph (non-RL, or substrates
    without `@claim` records) emit an empty `Graph()`; the
    optionality is 'graph has nodes', not 'graph is None'."""
    runs: tuple[RunRow, ...]
    traces: tuple[TraceRow, ...]
    graph: ComputationGraph


@dataclass(frozen=True, slots=True)
class SweepResult:
    """Aggregated sweep output across all grid points × both arms.
    Failures captured alongside successes — the framework never
    silently drops a failed cell."""
    cell_results: tuple[SweepCellResult, ...]
    failures: tuple[CellFailure, ...] = field(default_factory=tuple)

    @property
    def all_runs(self) -> tuple[RunRow, ...]:
        return tuple(r for cr in self.cell_results for r in cr.runs)

    @property
    def all_traces(self) -> tuple[TraceRow, ...]:
        return tuple(t for cr in self.cell_results for t in cr.traces)


class Runner[R: Mapping[str, object]](Protocol):
    """Substrate's bridge into corroborate.sweep. Receives one
    composed Claim + its arm_key + the (optional) measurables to
    pre-register + one exogenous-grid point; returns a
    `SweepCellResult` with per-cell records + captured graph.

    Protocol (not a bare Callable alias) so substrates can hold
    init state — e.g. the RL runner caches the env catalogue and
    JIT-compiles once per arm, not once per grid point.

    `measurables` is OPTIONAL: substrates that want certain
    scalars baked into RunRow.measurements at sweep time pass them
    here (the runner computes each per-record and persists them
    under the measurable's `.name`). Substrates that compute
    mediators post-sweep from raw traces leave it empty.

    The cell runner's contract:
    - Invoke `claim(...)` parameterised by `grid_point` to produce
      one record per seed/replicate.
    - For each `Measurable` in `measurables`, compute and persist
      the scalar at `RunRow.measurements[<measurable>.name]`.
    - Set `RunRow.arm_key = arm_key` on every emitted RunRow —
      arm identity is framework-derived (canonical_str of the
      Intervention tuple), NOT substrate-chosen."""
    def __call__(
        self,
        claim: Callable[..., R],
        arm_key: str,
        measurables: tuple[Measurable[R, object], ...],
        grid_point: Mapping[str, object],
    ) -> SweepCellResult: ...


def empty_graph() -> ComputationGraph:
    """Convenience for substrates that don't capture a graph.
    Returns a fresh empty `Graph[str, ComputationEdge]`."""
    return Graph()


# ============ run_intervention — the framework's `do()` operator ============

def run_intervention[R: Mapping[str, object]](
    intervention: DoEffect,
    *,
    base: Callable[..., R],
    measurables: tuple[Measurable[R, object], ...] = (),
    grid_points: Sequence[Mapping[str, object]],
    runner: Runner[R],
    out_dir: Path,
    archive_remote: str | None = None,
    arm_tag: Callable[[str, Mapping[str, object]], str] | None = None,
    trace_reductions: Sequence[pl.Expr] = (),
    trace_drops: Sequence[str] = (),
) -> tuple[Path, Path]:
    """Execute the typed contrast `intervention` against `base`
    over the discrete sequence `grid_points`; persist per-cell
    parquets; merge to a corpus.

    The framework's rung-2 `do()` operator at the corpus level.
    A sweep-time primitive: its job is to compose treatment +
    baseline claims via `apply_interventions(base,
    intervention.treatment / .baseline)` and dispatch each
    (claim, arm_key, measurables, grid_point) to `runner`.
    Pairing is intrinsic — treatment and baseline cells at the
    same `grid_point` ARE matched by construction.

    `base` is the substrate's theory pre-bound with HPs (e.g.
    `partial(dqn, gamma=0.99, lr=1e-3, total_steps=200_000, ...)`).
    The framework does not introspect or modify it; it just
    threads it through `apply_interventions`. HPs are substrate-
    side; they live on `base` — never on the `intervention`
    tuple, per the leaves-as-covariates discipline.

    `measurables` is OPTIONAL: pre-registered Measurable instances
    the substrate's runner persists per cell at sweep time
    (typically used to bake outcome reductions or cheap-to-compute
    scalars into RunRow.measurements alongside the leaf
    fingerprint). Substrates that compute mediators post-sweep
    from raw traces leave this empty; substrates that want
    eagerly-computed scalars baked into the corpus pass them.

    `grid_points` is a discrete sequence of grid_point dicts —
    NOT a Cartesian-product mapping. Substrates that want
    Cartesian product compose `itertools.product` themselves;
    substrates with heterogeneous-shape grids (e.g. different
    chunk_sizes per env) emit the flat list directly. Empty
    sequence runs zero cells; `[{}]` runs one cell per arm with
    an empty grid_point.

    `arm_tag` produces the filename suffix for each cell's
    parquets. Default: `f'{arm_key}'` — caller usually overrides
    to encode grid_point keys (e.g. `env_name`).

    `archive_remote`: optional fsspec URI prefix; uploads each
    cell's parquet pair to remote storage right after the cell
    completes, purges local; merges from remote at the end.

    `trace_reductions` / `trace_drops` forwarded to
    `apply_trace_reductions` per cell."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / 'tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)
    final_runs = out_dir / 'runs.parquet'
    final_traces = out_dir / 'traces.parquet'

    treatment_claim = apply_interventions(base, intervention.treatment)
    baseline_claim = apply_interventions(base, intervention.baseline)
    treatment_arm_key = intervention.treatment_arm_key()
    baseline_arm_key = intervention.baseline_arm_key()
    arms: tuple[
        tuple[Callable[..., R], str], tuple[Callable[..., R], str],
    ] = (
        (treatment_claim, treatment_arm_key),
        (baseline_claim, baseline_arm_key),
    )

    if arm_tag is None:
        def arm_tag_default(
            arm_key: str, grid_point: Mapping[str, object],
        ) -> str:
            del grid_point
            return arm_key
        effective_arm_tag: Callable[
            [str, Mapping[str, object]], str,
        ] = arm_tag_default
    else:
        effective_arm_tag = arm_tag

    grid_point_list = list(grid_points)
    n_cells = len(grid_point_list) * len(arms)
    print(f'sweep: {n_cells} cells '
          f'({len(grid_point_list)} grid points × {len(arms)} arms)',
          flush=True)

    runs_paths: list[Path] = []
    traces_paths: list[Path] = []
    graph_paths: list[Path] = []
    arm_graphs: dict[str, ComputationGraph] = {}
    archived_runs_uris: list[str] = []
    archived_traces_uris: list[str] = []
    failures: list[CellFailure] = []
    t_start = time.monotonic()

    cell_idx = 0
    for grid_point in grid_point_list:
        for claim, arm_key in arms:
            t_cell = time.monotonic()
            tag = effective_arm_tag(arm_key, grid_point)
            runs_path = (
                tmp_dir / f'cell{cell_idx:03d}__{tag}__runs.parquet'
            )
            traces_path = (
                tmp_dir / f'cell{cell_idx:03d}__{tag}__traces.parquet'
            )
            graph_path = (
                tmp_dir / f'cell{cell_idx:03d}__{tag}__graph.json'
            )
            rp_rel = runs_path.relative_to(out_dir).as_posix()
            tp_rel = traces_path.relative_to(out_dir).as_posix()

            # Resume support: skip cells whose outputs already
            # exist (locally or remotely).
            if archive_remote is not None:
                from corroborate.corpus.cloud import (
                    archived_uri, is_archived,
                )
                if (
                    is_archived(out_dir, rp_rel)
                    and is_archived(out_dir, tp_rel)
                ):
                    archived_runs_uris.append(
                        archived_uri(archive_remote, rp_rel),
                    )
                    archived_traces_uris.append(
                        archived_uri(archive_remote, tp_rel),
                    )
                    if graph_path.exists():
                        graph_paths.append(graph_path)
                    print(
                        f'  [{cell_idx+1}/{n_cells}] {tag} '
                        f'✓ already archived, skipping',
                        flush=True,
                    )
                    cell_idx += 1
                    continue
            elif runs_path.exists() and traces_path.exists():
                runs_paths.append(runs_path)
                traces_paths.append(traces_path)
                if graph_path.exists():
                    graph_paths.append(graph_path)
                print(
                    f'  [{cell_idx+1}/{n_cells}] {tag} '
                    f'✓ local parquets exist, skipping',
                    flush=True,
                )
                cell_idx += 1
                continue

            print(
                f'  [{cell_idx+1}/{n_cells}] {tag} '
                f'(grid_point keys: {sorted(grid_point.keys())}) ...',
                flush=True,
            )
            try:
                cell_result = runner(
                    claim, arm_key, measurables, grid_point,
                )
            except Exception as exc:  # noqa: BLE001
                failures.append(CellFailure(
                    arm_key=arm_key,
                    grid_point=dict(grid_point),
                    error=f'{type(exc).__name__}: {exc}',
                    duration_s=time.monotonic() - t_cell,
                ))
                cell_idx += 1
                continue
            write_runrows(cell_result.runs, runs_path)
            reduced = apply_trace_reductions(
                list(cell_result.traces),
                add=trace_reductions, drop=trace_drops,
            )
            write_tracerows(reduced, traces_path)
            write_graphs_sidecar(
                {arm_key: cell_result.graph}, graph_path,
            )
            arm_graphs[arm_key] = cell_result.graph
            runs_paths.append(runs_path)
            traces_paths.append(traces_path)
            graph_paths.append(graph_path)
            if archive_remote is not None:
                from corroborate.corpus.cloud import archive
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
            elapsed = time.monotonic() - t_cell
            total = time.monotonic() - t_start
            print(
                f'    done in {elapsed:.1f}s '
                f'(cumulative {total/60:.1f} min)',
                flush=True,
            )
            cell_idx += 1

    print()
    print('merging per-cell parquets ...', flush=True)
    if archive_remote is not None:
        stream_concat_parquets(
            [str(u) for u in archived_runs_uris], final_runs,
        )
        stream_concat_parquets(
            [str(u) for u in archived_traces_uris], final_traces,
        )
        from corroborate.corpus.cloud import archive as _archive_merged
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

    merged_graphs: dict[str, ComputationGraph] = dict(arm_graphs)
    for gp_path in graph_paths:
        if gp_path.exists():
            merged_graphs.update(read_graphs_sidecar(gp_path))
    if merged_graphs:
        final_graphs = out_dir / 'graphs.json'
        write_graphs_sidecar(merged_graphs, final_graphs)
        print(f'  → {final_graphs}')

    if failures:
        print(f'  WARN: {len(failures)} cell failures')
        for f in failures:
            print(f'    [{f.arm_key}] {f.grid_point}: {f.error}')
    return final_runs, final_traces


__all__ = [
    'CellFailure',
    'Runner',
    'SweepCellResult',
    'SweepResult',
    'empty_graph',
    'run_intervention',
]
