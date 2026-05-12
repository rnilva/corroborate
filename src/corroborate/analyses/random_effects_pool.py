"""`random_effects_pool` — per-stratum paired g pooled with the
heterogeneity-flagged verdict.

The framework's "find the scope" mission, expressed at the verdict
layer: when a corpus pools positive but I² ≥ 0.5, the right
verdict is `HELD_WITH_SCOPE_FLAG` rather than plain `HELD` —
the population-level claim corroborates but effects vary
substantially across strata. The flag is the trigger for a
meta-regression follow-up bridge that asks "which covariate
predicts the per-stratum effect?".

The principle is documented in
- `corroborate/bridge/verdict.py` (HELD_WITH_SCOPE_FLAG enum
  semantics)
- `corroborate/stats/effect_size.py::random_effects_verdict`
  (the dispatch rule)
- `corroborate/stats/meta_regression.py` (the cleavage-axis
  sibling analysis)
- `ANALYSIS_RECIPE.md` §1.5 (the loop authoring discipline)

This primitive is the *discoverable* surface for the pool side
of the loop — bridges fixture it like any other analysis, and
the bridge body returns the result's verdict directly. The
sibling meta-regression bridge runs on the same scope (same
extent_hash → automatic cluster on the post-evaluated graph)."""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from corroborate.analyses.paired_g import per_env_paired_g_panel
from corroborate.bridge.analysis import analysis
from corroborate.bridge.verdict import RefutationClass, Verdict
from corroborate.corpus.schema import StratumG
from corroborate.stats import (
    PooledStats, random_effects_summary, random_effects_verdict,
)
from corroborate.stats.effect_size import PredictedDirection


@dataclass(frozen=True, slots=True)
class RandomEffectsPoolResult:
    """Pooled per-stratum effect with heterogeneity-flagged verdict.

    `pooled` carries the DerSimonian-Laird random-effects summary
    (pooled_g, se, I², τ², prediction interval, n_cells); inspect
    it to read the heterogeneity diagnostics.

    `verdict` is `random_effects_verdict`'s output: `HELD`,
    `HELD_WITH_SCOPE_FLAG`, `NO_EFFECT`, or `POWER_INSUFFICIENT`.
    Bridges that fixture this analysis typically return
    `(verdict, refutation)` directly — the heterogeneity
    discipline is enforced inside the analysis, not by the
    bridge body.

    `assumption_violations` propagates the DL small-g / τ²-clip
    artifact flags from the pool (`stats/effect_size.py`'s
    `_dl_assumption_violations`)."""
    pooled: PooledStats
    verdict: Verdict
    refutation: RefutationClass | None
    per_env: tuple[StratumG[str], ...]
    n_strata: int
    measurable: str
    treatment_arm: str
    baseline_arm: str
    assumption_violations: tuple[str, ...] = ()


@analysis
def random_effects_pool(
    cells: Iterable[Mapping[str, object]],
    *,
    treatment_arm: str,
    baseline_arm: str,
    source: str,
    pair_by: tuple[str, ...] = ('seed',),
    env_filter: tuple[str, ...] = (),
    arm_field: str = 'arm_key',
    total_steps_filter: int | None = None,
    total_steps_field: str = 'total_steps',
    predicted_direction: PredictedDirection | None = None,
) -> RandomEffectsPoolResult:
    """Per-stratum paired g pooled by DL random-effects, with
    `(verdict, refutation)` dispatched per the I²-threshold +
    PI rules in `random_effects_verdict`.

    `predicted_direction` flows from the bridge decorator
    automatically (the framework injects it into analyses that
    name it as a kwarg). When `None`, the verdict path falls
    through to `random_effects_verdict`'s default behavior
    (treats PI excluding zero on either side as HELD)."""
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
    verdict, refutation = random_effects_verdict(
        pooled, predicted_direction=predicted_direction,
    )
    return RandomEffectsPoolResult(
        pooled=pooled,
        verdict=verdict,
        refutation=refutation,
        per_env=panel,
        n_strata=len(pool_obs),
        measurable=source,
        treatment_arm=treatment_arm,
        baseline_arm=baseline_arm,
        assumption_violations=pooled.assumption_violations,
    )


__all__ = ['RandomEffectsPoolResult', 'random_effects_pool']
