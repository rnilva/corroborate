"""`paired_g_pooled` — per-env paired Hedges' g pooled by random
effects across an explicit env subset.

The shape FINDINGS revision 1 consumes: stratify cells by env,
compute paired g per env, pool with DerSimonian-Laird across the
panel of envs. The bridge supplies `env_filter` to commit the
analysis to a specific scope-class (e.g. converged envs); empty
tuple = all envs in the corpus.

Distinct from `paired_g` (single-env) and from
`meta_regression_paired_g` (per-env g regressed on covariates):
this analysis answers "what's the pooled effect across THIS list
of envs?" — the corpus-level convergence-conditioned claim shape.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from corroborate.analyses.paired_g import per_env_paired_g_panel
from corroborate.analysis import analysis
from corroborate.stats import (
    PooledStats, random_effects_summary,
)
from corroborate.stratum import StratumG


# Backward-compatible alias for substrate code that imports the
# old class-name shape.
type PerEnvG = StratumG[str]


@dataclass(frozen=True, slots=True)
class PooledPairedGResult:
    """Output of `paired_g_pooled`. `pooled` is the random-effects
    summary across `per_env`; `n_envs` is the count of envs that
    contributed (i.e. had ≥2 paired observations and finite g/SE).
    `total_steps_filter` is the filter that was applied (None if
    all total_steps values were included)."""
    pooled: PooledStats
    per_env: tuple[StratumG[str], ...]
    n_envs: int
    total_steps_filter: int | None
    measurable: str
    treatment_arm: str
    baseline_arm: str


@analysis
def paired_g_pooled(
    cells: Iterable[Mapping[str, object]],
    *,
    treatment_arm: str,
    baseline_arm: str,
    source: str,
    pair_by: tuple[str, ...] = ('seed',),
    env_filter: tuple[str, ...] = (),
    arm_field: str = 'intervention_name',
    total_steps_filter: int | None = None,
    total_steps_field: str = 'total_steps',
) -> PooledPairedGResult:
    """Per-env paired g pooled across `env_filter`.

    `env_filter` empty → use every env present in `cells`.
    `total_steps_filter` filters cells to a single training
    horizon (the rev 1 200k corpus has both 50k and 200k cells —
    bridges should commit to one horizon explicitly)."""
    cells_list = list(cells)
    if total_steps_filter is not None:
        cells_list = [
            c for c in cells_list
            if c.get(total_steps_field) == total_steps_filter
        ]

    panel = per_env_paired_g_panel(
        cells_list,
        treatment_arm=treatment_arm,
        baseline_arm=baseline_arm,
        source=source,
        env_filter=env_filter,
        pair_by=pair_by,
        arm_field=arm_field,
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


__all__ = [
    'PerEnvG', 'PooledPairedGResult', 'paired_g_pooled',
]
