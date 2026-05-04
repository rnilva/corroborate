"""Per-env convergence classification on an RL corpus.

Layered on top of `env_solve_thresholds.SOLVE_THRESHOLDS`. Given
a corpus of `RunRow`s + an outcome path, classifies each env into
`solved` / `partial` / `unsolved` / `absent` based on the fraction
of cells (within an arm) that meet the env's solve threshold.

Convergence-aware verdicts are critical because RL training at
a finite horizon often doesn't reach steady-state for all envs.
A naïve treatment-vs-baseline ATE pooled across envs conflates:
- envs where the agent has converged (mechanism effect on a
  trained policy),
- envs where the agent is mid-trajectory (mechanism effect on a
  transient training curve),
- envs where the agent isn't learning at all (effect ≈ noise).

Restricting verdicts to converged-on-baseline envs answers the
sharper question: *given that vanilla DQN has reached a learned
policy on this env, what is the mechanism's effect?* The framework
treats this as a scope condition — the corpus's training horizon
is itself part of the scope.

This module operates on the BASELINE arm by default — the
convergence question is "does the *unintervened* agent solve
this env?" — and the classification rate gates which envs feed
the §3 verdict pattern in convergence-conditioned mode."""
from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from corroborate_rl.env_solve_thresholds import (
    SOLVE_THRESHOLDS, SolveThreshold,
)
from corroborate.corpus.schema import MeasurementLeaf, RunRow


type ConvergenceClass = Literal[
    'solved', 'partial', 'unsolved', 'absent',
]
"""How the env classifies under the convergence audit:

- `solved` — final-mean solve rate ≥ `solved_threshold` (default
  0.5). The baseline arm reliably reaches the env's solve criterion.
- `partial` — final-mean solve rate in (0, solved_threshold). The
  agent solves the env *sometimes* but not robustly; mechanism
  effects from this env are interpretable but heterogeneous.
- `unsolved` — final-mean solve rate is exactly 0. The agent does
  not solve the env at this horizon; mechanism effects from this
  env reflect transient training dynamics, not policy quality.
- `absent` — the env has no defensible solve threshold. Excluded
  from convergence-conditioned verdicts."""


@dataclass(frozen=True, slots=True)
class EnvConvergence:
    """Per-env convergence summary on the baseline arm.

    `best_mean` / `final_mean` — average over baseline cells of the
      best-eval-burst-mean / final-eval-mean outcome paths.
    `best_solve_rate` / `final_solve_rate` — fraction of baseline
      cells where outcome ≥ threshold. None when the env has no
      threshold (`'absent'` confidence).
    `n_cells` — number of baseline cells contributing.
    `classification` — derived from `final_solve_rate` (the strict
      criterion: did training maintain the threshold at the end,
      not just reach it transiently)."""
    env_name: str
    threshold: SolveThreshold
    best_mean: float
    final_mean: float
    best_solve_rate: float | None
    final_solve_rate: float | None
    n_cells: int
    classification: ConvergenceClass


# ============ Classification ============

def _classify(
    rate: float | None, *, solved_threshold: float,
) -> ConvergenceClass:
    if rate is None:
        return 'absent'
    if rate >= solved_threshold:
        return 'solved'
    if rate > 0.0:
        return 'partial'
    return 'unsolved'


def _scalar_outcome(
    run: RunRow, path: str,
) -> float | None:
    v: MeasurementLeaf | None = run.measurements.get(path)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    fv = float(v)
    if math.isnan(fv):
        return None
    return fv


def classify_envs(
    baseline_runs: Sequence[RunRow],
    *,
    best_burst_path: str = 'eval_best_burst_mean',
    final_mean_path: str = 'eval_final_mean',
    table: Mapping[str, SolveThreshold] = SOLVE_THRESHOLDS,
    solved_threshold: float = 0.5,
    group_by: str = 'env_name',
) -> Mapping[str, EnvConvergence]:
    """Classify each env's convergence on the baseline arm.

    `solved_threshold` — the cutoff on `final_solve_rate` separating
      `solved` from `partial`. 0.5 by default: at least half of
      baseline cells must reach the env's solve criterion at the
      end of training. Strict; consumers can lower it if their
      research question tolerates partial convergence.

    Returns a mapping `env_name → EnvConvergence`. Envs with NO
    cells in `baseline_runs` are omitted entirely (caller can
    detect via membership)."""
    by_env: dict[str, list[RunRow]] = {}
    for r in baseline_runs:
        env_v = r.measurements.get(group_by)
        if not isinstance(env_v, str):
            continue
        by_env.setdefault(env_v, []).append(r)

    out: dict[str, EnvConvergence] = {}
    for env_name, env_runs in by_env.items():
        if env_name not in table:
            # Env not in threshold table — emit an `absent` row
            # using a synthetic SolveThreshold so the consumer
            # sees the env was considered.
            spec = SolveThreshold(
                env_name=env_name, threshold=None,
                source='not-in-table', confidence='absent',
            )
        else:
            spec = table[env_name]

        best_vals = [
            v for v in (
                _scalar_outcome(r, best_burst_path) for r in env_runs
            ) if v is not None
        ]
        final_vals = [
            v for v in (
                _scalar_outcome(r, final_mean_path) for r in env_runs
            ) if v is not None
        ]

        best_mean = (
            sum(best_vals) / len(best_vals) if best_vals
            else float('nan')
        )
        final_mean = (
            sum(final_vals) / len(final_vals) if final_vals
            else float('nan')
        )

        best_solve_rate: float | None = None
        final_solve_rate: float | None = None
        if spec.threshold is not None:
            t = spec.threshold
            if best_vals:
                best_solve_rate = sum(
                    1 for v in best_vals if v >= t
                ) / len(best_vals)
            else:
                best_solve_rate = 0.0
            if final_vals:
                final_solve_rate = sum(
                    1 for v in final_vals if v >= t
                ) / len(final_vals)
            else:
                final_solve_rate = 0.0

        classification = _classify(
            final_solve_rate, solved_threshold=solved_threshold,
        )

        out[env_name] = EnvConvergence(
            env_name=env_name, threshold=spec,
            best_mean=best_mean, final_mean=final_mean,
            best_solve_rate=best_solve_rate,
            final_solve_rate=final_solve_rate,
            n_cells=len(env_runs),
            classification=classification,
        )
    return out


# ============ Filtering helpers ============

def envs_in_class(
    classifications: Mapping[str, EnvConvergence],
    target: ConvergenceClass,
) -> tuple[str, ...]:
    """Names of envs in the given convergence class, sorted."""
    return tuple(sorted(
        name for name, c in classifications.items()
        if c.classification == target
    ))


def filter_to_classes(
    runs: Sequence[RunRow],
    classifications: Mapping[str, EnvConvergence],
    targets: tuple[ConvergenceClass, ...],
    *,
    group_by: str = 'env_name',
) -> list[RunRow]:
    """Return runs whose `group_by` value falls in any of the
    `target` classifications. Use `targets=('solved',)` for the
    converged-only restriction; `targets=('solved', 'partial')`
    for the looser version."""
    keep_envs = frozenset(
        name for name, c in classifications.items()
        if c.classification in targets
    )
    return [
        r for r in runs
        if r.measurements.get(group_by) in keep_envs
    ]


# ============ Mediator differential between scope classes ============

@dataclass(frozen=True, slots=True)
class PathDifferential:
    """Per-path Hedges' g between two scope classes' cells.

    `path` — the measurement path being compared (e.g.,
      `'mediator.q_max_growth'`).
    `g` — Hedges' g of the path's value, treating each cell as
      one observation. Sign convention: `mean(class_a) −
      mean(class_b)` divided by pooled std. Positive → path
      tends higher in `class_a`. Convention: `class_a` is the
      "of-interest" class (typically *unsolved* when looking for
      failure-mode signatures).
    `n_a`, `n_b` — number of finite-valued cells contributing to
      each class.
    `mean_a`, `mean_b` — class means (informative for interpreting
      the sign of g)."""
    path: str
    g: float
    n_a: int
    n_b: int
    mean_a: float
    mean_b: float


def _scalar_path(
    run: RunRow, path: str,
) -> float | None:
    v: MeasurementLeaf | None = run.measurements.get(path)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    fv = float(v)
    if not (fv == fv) or fv in (float('inf'), float('-inf')):
        return None
    return fv


def _within_env_zscore(
    cells: Sequence[RunRow], path: str, group_by: str,
) -> list[tuple[RunRow, float]]:
    """Replace each cell's `path` value with its z-score *within
    that cell's `group_by` group*. Cells with non-finite path
    values are dropped. Groups with <2 cells or zero std are
    dropped (no valid z-score)."""
    by_group: dict[str, list[tuple[RunRow, float]]] = {}
    for r in cells:
        v = _scalar_path(r, path)
        if v is None:
            continue
        gv = r.measurements.get(group_by)
        if not isinstance(gv, str):
            continue
        by_group.setdefault(gv, []).append((r, v))
    out: list[tuple[RunRow, float]] = []
    for grp_cells in by_group.values():
        if len(grp_cells) < 2:
            continue
        vals = [v for _, v in grp_cells]
        mean = statistics.fmean(vals)
        try:
            std = statistics.stdev(vals)
        except statistics.StatisticsError:
            continue
        if std <= 0.0:
            continue
        for r, v in grp_cells:
            out.append((r, (v - mean) / std))
    return out


type Aggregation = Literal['cell', 'env_mean']


def mediator_differential(
    runs: Sequence[RunRow],
    classifications: Mapping[str, EnvConvergence],
    *,
    paths: Sequence[str],
    class_a: ConvergenceClass = 'unsolved',
    class_b: ConvergenceClass = 'solved',
    group_by: str = 'env_name',
    aggregation: Aggregation = 'env_mean',
    z_score_within_group: bool = False,
) -> tuple[PathDifferential, ...]:
    """For each path in `paths`, compute Hedges' g of the path's
    value between `class_a`'s envs and `class_b`'s envs.

    `aggregation` — controls the unit of analysis:
      - `'env_mean'` (default, *honest*) — each env contributes one
        observation: the mean of its cells' path values. Hedges'
        g is computed across env-level means. Low-power when there
        are few envs per class but UNCONFOUNDED by env-level scale
        differences. The right unit when class membership is
        env-determined (every cell within an env has the same
        class).
      - `'cell'` — pool individual cell values across envs.
        High-power but *confounded* when envs differ in scale and
        class membership is env-defined: the differential
        rediscovers env identity, not failure mode. Use only with
        `z_score_within_group=True` (which strips env-scale at the
        cost of stripping all cross-env signal).

    `z_score_within_group` — when True (cell mode only), each cell's
      path value is replaced by its within-`group_by` z-score
      before pooling. Mathematically forces the cross-class
      differential to ≈0 when class membership is env-determined
      (since z-scoring per env makes within-env mean=0 by
      construction). Useful only when classes have *cells from
      both classes within the same env* — not the case for the
      convergence-class application.

    Use case: surface trace-features that distinguish unsolved
    envs from solved envs. The top-|g| paths under env_mean mode
    name candidate failure-mode signatures.

    *Caveats baked in by design*:
    - When either class has < 2 envs (env_mean) or < 2 cells
      (cell) with finite values, g is NaN.
    - Cells / envs with non-finite values are dropped silently."""
    a_envs = frozenset(
        name for name, c in classifications.items()
        if c.classification == class_a
    )
    b_envs = frozenset(
        name for name, c in classifications.items()
        if c.classification == class_b
    )

    a_cells = [
        r for r in runs
        if r.measurements.get(group_by) in a_envs
    ]
    b_cells = [
        r for r in runs
        if r.measurements.get(group_by) in b_envs
    ]

    def _env_means(cells: list[RunRow], path: str) -> list[float]:
        """One mean per env. Envs with no finite cells contribute
        nothing."""
        by_env: dict[str, list[float]] = {}
        for r in cells:
            v = _scalar_path(r, path)
            if v is None:
                continue
            ev = r.measurements.get(group_by)
            if not isinstance(ev, str):
                continue
            by_env.setdefault(ev, []).append(v)
        return [
            statistics.fmean(vals) for vals in by_env.values() if vals
        ]

    out: list[PathDifferential] = []
    for path in paths:
        if aggregation == 'env_mean':
            a_vals = _env_means(a_cells, path)
            b_vals = _env_means(b_cells, path)
        elif z_score_within_group:
            a_pairs = _within_env_zscore(a_cells, path, group_by)
            b_pairs = _within_env_zscore(b_cells, path, group_by)
            a_vals = [v for _, v in a_pairs]
            b_vals = [v for _, v in b_pairs]
        else:
            a_vals = [
                v for v in (_scalar_path(r, path) for r in a_cells)
                if v is not None
            ]
            b_vals = [
                v for v in (_scalar_path(r, path) for r in b_cells)
                if v is not None
            ]
        n_a, n_b = len(a_vals), len(b_vals)
        if n_a < 2 or n_b < 2:
            out.append(PathDifferential(
                path=path, g=float('nan'), n_a=n_a, n_b=n_b,
                mean_a=(
                    statistics.fmean(a_vals) if a_vals else float('nan')
                ),
                mean_b=(
                    statistics.fmean(b_vals) if b_vals else float('nan')
                ),
            ))
            continue
        mean_a = statistics.fmean(a_vals)
        mean_b = statistics.fmean(b_vals)
        var_a = statistics.variance(a_vals)
        var_b = statistics.variance(b_vals)
        pooled = (
            ((n_a - 1) * var_a + (n_b - 1) * var_b)
            / (n_a + n_b - 2)
        )
        if pooled <= 0.0:
            g = float('nan')
        else:
            d = (mean_a - mean_b) / pooled ** 0.5
            c4 = 1.0 - 3.0 / (4 * (n_a + n_b) - 9)
            g = d * c4
        out.append(PathDifferential(
            path=path, g=g, n_a=n_a, n_b=n_b,
            mean_a=mean_a, mean_b=mean_b,
        ))
    out.sort(
        key=lambda d: (
            float('-inf') if d.g != d.g else -abs(d.g)
        ),
    )
    return tuple(out)
