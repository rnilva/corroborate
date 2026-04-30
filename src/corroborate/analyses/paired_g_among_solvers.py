"""`paired_g_among_solvers` — paired Hedges' g restricted to
(env, pair) pairs where BOTH treatment and baseline cells crossed
the env-specific gate threshold.

The shape FINDINGS revision 6 consumes: among cells that solved
at all, does the treatment reach threshold faster (smaller
best-burst-step) than baseline? Sample-efficiency probe at the
link edge — the conventional `outcome.eval_best_burst_mean`
saturates at the discounted-return ceiling for jointly-solved
envs, hiding any first-crossing-step difference.

Distinct from `paired_g_pooled`: the gate filter excludes pairs
where at least one arm failed to solve, so the per-env pair
count is `min(n_seeds, n_solved_pairs)`. The pool is over envs
where ≥2 surviving pairs produce finite g/SE."""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

from corroborate.analyses.paired_g_pooled import (
    PerEnvG, PooledPairedGResult,
)
from corroborate.analysis import analysis
from corroborate.statistics import (
    hedges_g_paired, random_effects_summary,
)


def _passes_gate(
    cell: Mapping[str, object],
    gate_column: str,
    threshold: float,
) -> bool:
    v = cell.get(gate_column)
    if not isinstance(v, (int, float)):
        return False
    if math.isnan(float(v)):
        return False
    return float(v) >= threshold


def _scalar(cell: Mapping[str, object], path: str) -> float | None:
    v = cell.get(path)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    fv = float(v)
    return fv if not math.isnan(fv) else None


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

    by_env_arm_key: dict[
        tuple[str, str, tuple[object, ...]], dict[str, float],
    ] = {}
    envs_seen: set[str] = set()
    for cell in cells_list:
        env_v = cell.get('env_name')
        arm_v = cell.get(arm_field)
        if not isinstance(env_v, str):
            continue
        envs_seen.add(env_v)
        if arm_v not in (treatment_arm, baseline_arm):
            continue
        if env_v not in gate_thresholds:
            continue
        threshold = gate_thresholds[env_v]
        if not _passes_gate(cell, gate_column, threshold):
            continue
        s = _scalar(cell, source)
        if s is None:
            continue
        key = tuple(cell[k] for k in pair_by)
        slot = by_env_arm_key.setdefault(
            (env_v, str(arm_v), key), {},
        )
        slot[source] = s

    target_envs: tuple[str, ...]
    if env_filter:
        target_envs = tuple(e for e in env_filter if e in envs_seen)
    else:
        target_envs = tuple(sorted(envs_seen))

    per_env: list[PerEnvG] = []
    pool_obs: list[tuple[float, float]] = []
    for env in target_envs:
        if env not in gate_thresholds:
            per_env.append(PerEnvG(
                env_name=env, g=float('nan'),
                se=float('nan'), n_pairs=0,
            ))
            continue
        treatment_keys: dict[tuple[object, ...], float] = {}
        baseline_keys: dict[tuple[object, ...], float] = {}
        for (e, arm, key), data in by_env_arm_key.items():
            if e != env:
                continue
            v = data.get(source)
            if v is None:
                continue
            if arm == treatment_arm:
                treatment_keys[key] = v
            elif arm == baseline_arm:
                baseline_keys[key] = v
        paired_keys = sorted(set(treatment_keys) & set(baseline_keys))
        deltas = [
            treatment_keys[k] - baseline_keys[k] for k in paired_keys
        ]
        n_pairs = len(deltas)
        g, se = (
            hedges_g_paired(deltas) if n_pairs >= 2
            else (float('nan'), float('nan'))
        )
        per_env.append(PerEnvG(
            env_name=env, g=g, se=se, n_pairs=n_pairs,
        ))
        if (
            n_pairs >= 2 and not math.isnan(g)
            and not math.isnan(se) and se > 0.0
        ):
            pool_obs.append((g, se))

    pooled = random_effects_summary(pool_obs)
    return PooledPairedGResult(
        pooled=pooled,
        per_env=tuple(per_env),
        n_envs=len(pool_obs),
        total_steps_filter=total_steps_filter,
        measurable=source,
        treatment_arm=treatment_arm,
        baseline_arm=baseline_arm,
    )


__all__ = ['paired_g_among_solvers']
