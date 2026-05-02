"""`paired_g_among_solvers` — paired Hedges' g restricted to
(env, pair) pairs where BOTH treatment and baseline cells crossed
the env-specific gate threshold.

The shape FINDINGS revision 6 consumes: among cells that solved
at all, does the treatment reach threshold faster (smaller
best-burst-step) than baseline? Sample-efficiency probe at the
link edge — the conventional `eval_best_burst_mean`
saturates at the discounted-return ceiling for jointly-solved
envs, hiding any first-crossing-step difference.

Distinct from `paired_g_pooled`: the gate filter excludes pairs
where at least one arm failed to solve, so the per-env pair
count is `min(n_seeds, n_solved_pairs)`. The pool is over envs
where ≥2 surviving pairs produce finite g/SE.

Wraps `per_env_paired_g_panel` with a per-env `cell_predicate`
that gates by `gate_thresholds[env]` — the underlying pairing
loop lives in `paired_g.fn`."""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

from corroborate.analyses.paired_g import per_env_paired_g_panel
from corroborate.analyses.paired_g_pooled import (
    PerEnvG, PooledPairedGResult,
)
from corroborate.analysis import analysis
from corroborate.statistics import random_effects_summary


@analysis
def paired_g_among_solvers(
    cells: Iterable[Mapping[str, object]],
    *,
    treatment_arm: str,
    baseline_arm: str,
    source: str,
    gate_column: str,
    gate_thresholds: Mapping[str, float],
    pair_by: tuple[str, ...] = ('seed',),
    env_filter: tuple[str, ...] = (),
    arm_field: str = 'intervention_name',
    total_steps_filter: int | None = None,
    total_steps_field: str = 'total_steps',
) -> PooledPairedGResult:
    """Per-env paired g on `source`, restricted to (env, pair_by)
    pairs where BOTH cells satisfy `cell[gate_column] >=
    gate_thresholds[env]`. Pooled across `env_filter` via random
    effects.

    `gate_thresholds` is the env-keyed solve-threshold mapping
    (typically derived from
    `corroborate.rl.env_solve_thresholds.SOLVE_THRESHOLDS`).
    Envs without a threshold entry are skipped — `absent`-class
    envs can't participate in the solve filter."""
    cells_list = list(cells)
    if total_steps_filter is not None:
        cells_list = [
            c for c in cells_list
            if c.get(total_steps_field) == total_steps_filter
        ]

    # `cell_predicate` reads the gate threshold per cell's env;
    # cells in envs without a threshold drop out (so the panel
    # ends up restricted to gated envs even when env_filter is
    # empty).
    def gate(cell: Mapping[str, object]) -> bool:
        env_v = cell.get('env_name')
        if not isinstance(env_v, str):
            return False
        threshold = gate_thresholds.get(env_v)
        if threshold is None:
            return False
        v = cell.get(gate_column)
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return False
        fv = float(v)
        if math.isnan(fv):
            return False
        return fv >= threshold

    # Effective env set: explicit `env_filter` ∩ envs with a
    # gate, else all gated envs in `cells_list`.
    if env_filter:
        target = tuple(e for e in env_filter if e in gate_thresholds)
    else:
        envs_seen = {
            c.get('env_name') for c in cells_list
            if isinstance(c.get('env_name'), str)
        }
        target = tuple(sorted(
            e for e in envs_seen
            if isinstance(e, str) and e in gate_thresholds
        ))

    panel = per_env_paired_g_panel(
        cells_list,
        treatment_arm=treatment_arm,
        baseline_arm=baseline_arm,
        source=source,
        env_filter=target,
        pair_by=pair_by,
        arm_field=arm_field,
        cell_predicate=gate,
    )

    pool_obs: list[tuple[float, float]] = [
        (s.g, s.se) for s in panel
        if s.n_pairs >= 2
        and not math.isnan(s.g) and not math.isnan(s.se)
        and s.se > 0.0
    ]
    pooled = random_effects_summary(pool_obs)
    return PooledPairedGResult(
        pooled=pooled,
        per_env=panel,
        n_envs=len(pool_obs),
        total_steps_filter=total_steps_filter,
        measurable=source,
        treatment_arm=treatment_arm,
        baseline_arm=baseline_arm,
    )


# Re-export PerEnvG for substrate-side imports.
__all__ = ['PerEnvG', 'paired_g_among_solvers']
