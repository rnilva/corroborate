"""Sweep — exogenous-grid runner.

Substrate-agnostic primitive: takes a `Hypothesis[R]`, an
exogenous-variable grid (substrate-named keys × value lists), and
a `Runner[R]` that knows how to execute one cell. The framework
iterates the Cartesian product of the grid, collects per-cell
results from each runner call, and returns them with any failures.

The framework knows nothing about RL concepts (`env`, `seed`,
`total_steps`). Those are exogenous *names the substrate chose*.
A non-RL substrate sweeping over (`patient_id`, `dose`,
`measurement_day`) uses the same primitive — it just authors a
different `exogenous_grid` and a different `Runner`.

The contract:

- `exogenous_grid: Mapping[str, Sequence[object]]` — each key is
  a name the substrate chose; values are the levels to vary across
  cells. Cartesian product produces the cell list.
- `Runner[R]` — Protocol with `__call__(h, grid_point) ->
  SweepCellResult`. The substrate may implement Runner as a class
  with init state (e.g. RL substrate caches an env catalogue) or
  as a bare function. Both satisfy the Protocol.

`Runner.__call__` returns a `SweepCellResult` carrying the
per-seed RunRows + TraceRows + the captured ComputationGraph.
The graph is structurally constant across seeds in a vmap-batched
substrate and is the substrate's contribution to the
mechanism_key / redundancy primitives downstream.

Subprocess isolation (one process per cell) is deferred. v0 runs
in-process; large sweeps that need isolation can wrap this
primitive without changing the contract."""
from __future__ import annotations

import itertools
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import polars as pl

from corroborate.graph.computation import ComputationGraph
from corroborate.graph import Graph
from corroborate.core.hypothesis import Hypothesis
from corroborate.persistence import (
    apply_trace_reductions,
    stream_concat_parquets,
    read_graphs_sidecar,
    write_graphs_sidecar,
    write_runrows,
    write_tracerows,
)
from corroborate.schema import RunRow, TraceRow


@dataclass(frozen=True, slots=True)
class CellFailure:
    """One cell that raised during execution. Captures the
    grid-point values + the exception's str representation. Sweeps
    return failures alongside successful results so callers see
    the gap explicitly (no silent drops)."""
    intervention_name: str
    grid_point: Mapping[str, object]
    error: str
    duration_s: float


@dataclass(frozen=True, slots=True)
class SweepCellResult:
    """One runner-call's output: per-seed records + the
    Hypothesis-level graph captured during the call.

    Substrates that don't capture a graph (non-RL, or substrates
    without `@claim` records) emit an empty `Graph()`; the
    optionality is 'graph has nodes', not 'graph is None'."""
    runs: tuple[RunRow, ...]
    traces: tuple[TraceRow, ...]
    graph: ComputationGraph


@dataclass(frozen=True, slots=True)
class SweepResult:
    """Aggregated sweep output across all grid points. Failures
    captured alongside successes — the framework never silently
    drops a failed cell.

    `cell_results` preserves per-grid-point grouping; the
    `all_runs` / `all_traces` properties flatten when the consumer
    just wants the row collection."""
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
    Hypothesis + one exogenous-grid point; returns a
    `SweepCellResult` with per-cell records + captured graph.

    Protocol (not a bare Callable alias) so substrates can hold
    init state — e.g. the RL runner caches the env catalogue and
    JIT-compiles once per arm, not once per grid point. Bare
    functions still satisfy via their implicit `__call__`.

    The grid_point is `Mapping[str, object]` rather than a typed
    dict — substrate runners type-narrow each key at the boundary.
    Parameterizing Runner over a typed grid shape would ripple
    through `sweep` and force users to declare a dataclass per
    substrate; not worth the cost."""
    def __call__(
        self,
        h: Hypothesis[R],
        grid_point: Mapping[str, object],
    ) -> SweepCellResult: ...


def sweep[R: Mapping[str, object]](
    h: Hypothesis[R],
    *,
    exogenous_grid: Mapping[str, Sequence[object]],
    runner: Runner[R],
) -> SweepResult:
    """Run `h` on each Cartesian point of `exogenous_grid`.
    Returns a `SweepResult` with `cell_results` (per-grid-point
    SweepCellResults) and `failures` (per-grid-point CellFailures
    for cells that raised).

    `exogenous_grid` keys are substrate-chosen. Iteration order
    follows `dict.items()` order; values are zipped via
    `itertools.product`. An empty grid (`{}`) runs the runner
    exactly once with an empty grid_point.

    Each `runner(h, grid_point)` call is wrapped in try/except —
    exceptions become `CellFailure` entries with the offending
    grid_point and the exception's string. The runner is
    responsible for the structure of the returned SweepCellResult
    (e.g. how many seeds it batches over)."""
    keys = list(exogenous_grid.keys())
    value_lists = [list(exogenous_grid[k]) for k in keys]
    if not keys:
        # Empty grid → one cell with empty grid_point.
        grid_points: list[dict[str, object]] = [{}]
    else:
        grid_points = [
            {k: v for k, v in zip(keys, point, strict=True)}
            for point in itertools.product(*value_lists)
        ]

    cell_results: list[SweepCellResult] = []
    failures: list[CellFailure] = []
    for grid_point in grid_points:
        t0 = time.monotonic()
        try:
            result = runner(h, grid_point)
        except Exception as exc:  # noqa: BLE001
            failures.append(CellFailure(
                intervention_name=h.name,
                grid_point=dict(grid_point),
                error=f'{type(exc).__name__}: {exc}',
                duration_s=time.monotonic() - t0,
            ))
            continue
        cell_results.append(result)
    return SweepResult(
        cell_results=tuple(cell_results),
        failures=tuple(failures),
    )


def empty_graph() -> ComputationGraph:
    """Convenience for substrates that don't capture a graph.
    Returns a fresh empty `Graph[str, ComputationEdge]`."""
    return Graph()


# ============ run_hypotheses — the framework's `do()` operator ============

def run_hypotheses[R: Mapping[str, object]](
    arms: Sequence[tuple[Hypothesis[R], Mapping[str, object]]],
    *,
    runner: Runner[R],
    out_dir: Path,
    archive_remote: str | None = None,
    arm_tag: Callable[[Hypothesis[R], Mapping[str, object]], str] | None = None,
    trace_reductions: Sequence[pl.Expr] = (),
    trace_drops: Sequence[str] = (),
) -> tuple[Path, Path]:
    """Execute each `(hypothesis, grid_point)` arm via runner;
    persist per-arm parquets; merge to a corpus.

    This is the framework's rung-2 `do()` operator at the corpus
    level: authored Hypothesis (claim graph + intervention spec) +
    exogenous grid_point in, materialized RunRow / TraceRow corpus
    out. The corpus IS the evidence produced by the intervention.

    `arms` are EXPLICIT pairs. Substrates that want a Cartesian
    product author it inline:
      `arms = [(h, gp) for h in hypotheses for gp in grid_per_arm]`
    Substrates that want hypothesis-paired-with-env (e.g. each
    hypothesis has a CNN configured for one env) author it
    directly. The framework doesn't impose a structure.

    Each arm is one runner call → one `SweepCellResult` → one
    (runs, traces) parquet pair. Final step concatenates via
    `stream_concat_parquets` (`diagonal_relaxed`).

    Substrate-agnostic: works for any `Runner[R]`. Substrates
    handle chunking by authoring multiple grid_points sharing the
    same outer key (e.g. `env_name`) but different inner ranges
    (e.g. `seeds`). Each chunk becomes its own arm.

    `archive_remote`: optional fsspec URI prefix
    (e.g. `s3://corroborate-archive/<sweep>`). When set, each
    arm's tmp parquet pair is uploaded to remote storage right
    after the arm completes, and the local copies are purged —
    bounding peak local-disk usage to ~one arm's worth across
    the whole sweep. The final merge then reads back from the
    remote URIs in the manifest, writes the merged
    `{runs,traces}.parquet` locally, and archives those merged
    outputs (without purging — they stay local for downstream
    analysis).

    `arm_tag` produces the filename suffix for each arm's
    parquets. Default: `{hypothesis.name}` — caller usually
    overrides to encode grid_point keys (e.g. `env_name`).

    `trace_reductions` / `trace_drops` are forwarded to
    `apply_trace_reductions` per arm; substrate authors who
    need to shrink trajectory columns before persistence pass
    polars expressions here."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / 'tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)
    final_runs = out_dir / 'runs.parquet'
    final_traces = out_dir / 'traces.parquet'

    if arm_tag is None:
        def arm_tag_default(
            h: Hypothesis[R], grid_point: Mapping[str, object],
        ) -> str:
            del grid_point
            return f'{h.name}'
        effective_arm_tag: Callable[
            [Hypothesis[R], Mapping[str, object]], str,
        ] = arm_tag_default
    else:
        effective_arm_tag = arm_tag

    print(f'sweep: {len(arms)} arms', flush=True)

    runs_paths: list[Path] = []
    traces_paths: list[Path] = []
    graph_paths: list[Path] = []
    # Per-arm graphs keyed by `Hypothesis.arm_key()` — the static
    # call topology each arm captured during its first vmap pass.
    # Merged into a single `graphs.json` sidecar after the loop.
    arm_graphs: dict[str, ComputationGraph] = {}
    archived_runs_uris: list[str] = []
    archived_traces_uris: list[str] = []
    t_start = time.monotonic()
    for idx, (h, grid_point) in enumerate(arms):
        t_arm = time.monotonic()
        tag = effective_arm_tag(h, grid_point)
        runs_path = tmp_dir / f'arm{idx:03d}__{tag}__runs.parquet'
        traces_path = tmp_dir / f'arm{idx:03d}__{tag}__traces.parquet'
        graph_path = tmp_dir / f'arm{idx:03d}__{tag}__graph.json'
        rp_rel = runs_path.relative_to(out_dir).as_posix()
        tp_rel = traces_path.relative_to(out_dir).as_posix()

        # Resume support: skip arms whose outputs already exist —
        # either remotely (archive_remote with manifest entry) or
        # locally (no archive_remote, parquets on disk). Lets a
        # relaunch after a partial crash pick up where it left
        # off without redoing finished arms.
        if archive_remote is not None:
            from corroborate.cloud import archived_uri, is_archived
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
                # Recover graph from local sidecar if present (the
                # graph isn't archived remotely yet — same-pass
                # only). Resume continues without it for archived-
                # only arms; the merged sidecar carries whatever
                # graphs the local pass produced.
                if graph_path.exists():
                    graph_paths.append(graph_path)
                print(
                    f'  [{idx+1}/{len(arms)}] {tag} '
                    f'✓ already archived, skipping',
                    flush=True,
                )
                continue
        elif runs_path.exists() and traces_path.exists():
            runs_paths.append(runs_path)
            traces_paths.append(traces_path)
            if graph_path.exists():
                graph_paths.append(graph_path)
            print(
                f'  [{idx+1}/{len(arms)}] {tag} '
                f'✓ local parquets exist, skipping',
                flush=True,
            )
            continue

        print(
            f'  [{idx+1}/{len(arms)}] {tag} '
            f'(grid_point keys: {sorted(grid_point.keys())}) ...',
            flush=True,
        )
        cell_result = runner(h, grid_point)
        write_runrows(cell_result.runs, runs_path)
        reduced = apply_trace_reductions(
            list(cell_result.traces),
            add=trace_reductions, drop=trace_drops,
        )
        write_tracerows(reduced, traces_path)
        # Per-arm graph sidecar; same arm_key may repeat across
        # grid points (HP variation doesn't perturb arm_key) — last
        # writer wins, which is fine since the topology is
        # constant for a given arm_key by construction.
        write_graphs_sidecar(
            {h.arm_key(): cell_result.graph}, graph_path,
        )
        arm_graphs[h.arm_key()] = cell_result.graph
        runs_paths.append(runs_path)
        traces_paths.append(traces_path)
        graph_paths.append(graph_path)
        if archive_remote is not None:
            from corroborate.cloud import archive
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
        elapsed = time.monotonic() - t_arm
        total = time.monotonic() - t_start
        print(
            f'    done in {elapsed:.1f}s '
            f'(cumulative {total/60:.1f} min)',
            flush=True,
        )

    print()
    print('merging per-arm parquets ...', flush=True)
    if archive_remote is not None:
        stream_concat_parquets(
            [str(u) for u in archived_runs_uris], final_runs,
        )
        stream_concat_parquets(
            [str(u) for u in archived_traces_uris], final_traces,
        )
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

    # Merge per-arm graph sidecars into one corpus-level
    # `graphs.json`. Resume mode: read from existing per-arm
    # sidecars; same-pass mode: `arm_graphs` already populated.
    merged_graphs: dict[str, ComputationGraph] = dict(arm_graphs)
    for gp in graph_paths:
        if gp.exists():
            merged_graphs.update(read_graphs_sidecar(gp))
    if merged_graphs:
        final_graphs = out_dir / 'graphs.json'
        write_graphs_sidecar(merged_graphs, final_graphs)
        print(f'  → {final_graphs}')
    return final_runs, final_traces
