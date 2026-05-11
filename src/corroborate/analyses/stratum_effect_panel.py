"""`stratum_effect_panel` — per-stratum treatment-effect Δs on
N measurables, as a shared substrate for downstream stratum-
level analyses (regression, partial correlation, mediation,
DoWhy backdoor, DerSimonian-Laird pool).

The two-layer architecture this primitive enables:

  Layer 1 (this primitive):
    cells → StratumEffectPanel
    For each stratum: Δ_X = mean(X|T, stratum) − mean(X|B, stratum)
    No per-pair Δs.

  Layer 2 (downstream consumers, each ~30 lines):
    panel_regress(panel, x, y)              — OLS slope of Δ_Y on Δ_X
    panel_partial_correlation(panel, ...)   — partial Spearman / Pearson
    panel_proportion_mediated(panel, ...)   — linear mediation share
    panel_pool(panel, target, method='DL')  — DerSimonian-Laird pool
    panel_dowhy_backdoor(panel, ...)        — DoWhy on stratum-Δ panel

**Why this design.** Per-pair Δs (`paired_g`, `paired_link_*`,
`proportion_mediated` in its current form) assume vanilla and
DDQN cells share a unit through pairing. In RL, the two arms
diverge from step 1: same seed only matches RNG state at
initialization. The "paired" Δ measures init-distribution-
induced correlation, not a treatment-effect coupling.

Stratum-level Δ = mean(T) − mean(B) within each stratum (env,
burst, sync, …). It's a treatment-effect *estimate* at the
stratum level, not a within-pair contrast on divergent
trajectories. Downstream consumers operate on the panel of
stratum-level effect estimates — the inferentially-honest
form for cross-stratum claims about treatment effects.

The framework stays substrate-neutral here: this primitive
knows nothing about RL, envs, or any substrate-specific
measurable. The substrate's bridge declares which measurables
to populate and what stratify_by to use. Same primitive
serves any substrate whose cells admit (treatment, baseline,
strata) decomposition.
"""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.stats as _stats

from corroborate.analyses.paired_g import resolve_value
from corroborate.bridge.analysis import analysis


type StratumAggregator = Literal['mean', 'median']


@dataclass(frozen=True, slots=True)
class StratumEffectPanel:
    """Per-stratum treatment-effect Δ panel.

    `strata` — tuple of stratum-id tuples (each = `stratify_by`
    values for that stratum).
    `measurables` — tuple of column names whose Δs were computed.
    `deltas` — `{measurable_name: per-stratum Δ tuple}`. Each
    Δ_X[s] = `mean(X | T, stratum s) − mean(X | B, stratum s)`.
    `n_treatment` / `n_baseline` — per-stratum cell counts for
    each arm (downstream consumers use them for inverse-variance
    weighting or SE estimation).
    """
    stratify_by: tuple[str, ...]
    strata: tuple[tuple[object, ...], ...]
    measurables: tuple[str, ...]
    deltas: Mapping[str, tuple[float, ...]]
    n_treatment: tuple[int, ...]
    n_baseline: tuple[int, ...]
    treatment_arm: str
    baseline_arm: str
    aggregator: StratumAggregator = 'mean'

    @property
    def n_strata(self) -> int:
        return len(self.strata)


@analysis
def stratum_effect_panel(
    cells: Iterable[Mapping[str, object]],
    *,
    treatment_arm: str,
    baseline_arm: str,
    measurables: tuple[str, ...],
    stratify_by: tuple[str, ...] = ('env_name',),
    arm_field: str = 'arm_key',
    min_seeds_per_arm: int = 3,
    aggregator: StratumAggregator = 'mean',
) -> StratumEffectPanel:
    """Compute per-stratum treatment-effect Δs on each named
    measurable. Returns the panel for downstream consumption.

    A stratum is included iff BOTH arms have ≥ `min_seeds_per_arm`
    cells in it. Strata where either arm is sparse get dropped
    (their Δ would be a noisy point estimate; downstream analyses
    should see a clean panel).

    `measurables` resolves via `resolve_value` — measurable
    registry first, field-path read second. The framework's
    measurable-resolution discipline.

    `aggregator` controls the per-stratum reduction:
    - `'mean'` (default): `Δ_X = mean(X|T) − mean(X|B)`. The
      population-of-inits expected-value summary; matches the
      paired-Δ form's `mean_diff` field. Sensitive to bimodal
      seed-level distributions (catastrophic outliers can flip
      the sign vs. the median reading).
    - `'median'`: `Δ_X = median(X|T) − median(X|B)`. The
      typical-init summary; robust to seed-level outliers. Best
      when training dynamics produce sign-mixed seed populations
      (some inits catastrophic, others rescued — the
      MetaMaze γ=0.999 case).
    Use both as sibling bridges when the seed-level distribution
    is plausibly bimodal — together they characterize the
    population mean and the typical case.
    """
    # Group by (arm, stratum).
    per_arm_stratum: dict[
        tuple[str, tuple[object, ...]], list[Mapping[str, object]],
    ] = defaultdict(list)
    for c in cells:
        arm = c.get(arm_field)
        if not isinstance(arm, str) or arm not in (treatment_arm, baseline_arm):
            continue
        sk = tuple(c.get(k) for k in stratify_by)
        per_arm_stratum[(arm, sk)].append(c)

    all_strata: set[tuple[object, ...]] = set()
    for arm, sk in per_arm_stratum.keys():
        all_strata.add(sk)

    # Find strata with adequate seeds in both arms.
    valid_strata: list[tuple[object, ...]] = []
    for sk in sorted(all_strata, key=lambda s: tuple(repr(v) for v in s)):
        t_cells = per_arm_stratum.get((treatment_arm, sk), [])
        b_cells = per_arm_stratum.get((baseline_arm, sk), [])
        if len(t_cells) >= min_seeds_per_arm and len(b_cells) >= min_seeds_per_arm:
            valid_strata.append(sk)

    # Per stratum, per measurable: compute mean(T) - mean(B).
    deltas: dict[str, list[float]] = {m: [] for m in measurables}
    n_t_list: list[int] = []
    n_b_list: list[int] = []
    for sk in valid_strata:
        t_cells = per_arm_stratum[(treatment_arm, sk)]
        b_cells = per_arm_stratum[(baseline_arm, sk)]
        n_t_list.append(len(t_cells))
        n_b_list.append(len(b_cells))
        for m in measurables:
            t_vals = [resolve_value(c, m) for c in t_cells]
            b_vals = [resolve_value(c, m) for c in b_cells]
            t_vals_finite = [v for v in t_vals if not math.isnan(v)]
            b_vals_finite = [v for v in b_vals if not math.isnan(v)]
            if len(t_vals_finite) >= 1 and len(b_vals_finite) >= 1:
                match aggregator:
                    case 'mean':
                        agg_t = float(np.mean(t_vals_finite))
                        agg_b = float(np.mean(b_vals_finite))
                    case 'median':
                        agg_t = float(np.median(t_vals_finite))
                        agg_b = float(np.median(b_vals_finite))
                delta = agg_t - agg_b
            else:
                delta = float('nan')
            deltas[m].append(delta)

    return StratumEffectPanel(
        stratify_by=tuple(stratify_by),
        strata=tuple(valid_strata),
        measurables=tuple(measurables),
        deltas={m: tuple(deltas[m]) for m in measurables},
        n_treatment=tuple(n_t_list),
        n_baseline=tuple(n_b_list),
        treatment_arm=treatment_arm,
        baseline_arm=baseline_arm,
        aggregator=aggregator,
    )


# ============ Downstream consumer: panel_regress ============


@dataclass(frozen=True, slots=True)
class PanelRegressionResult:
    """OLS regression of `panel.deltas[y]` on `panel.deltas[x]`
    across strata. Treats each stratum as one observation.

    `slope` — coefficient of x on y; the chain-amplifier-style
    claim's load-bearing quantity.
    `slope_se` — Wald SE on the slope.
    `p_value` — two-sided p for `slope ≠ 0` (scipy linregress's
    Wald-test against H0: slope == 0).
    `r_squared` — fraction of variance in y explained by x.
    `n_strata` — strata contributing (after NaN drops).
    `x_values` / `y_values` — the actual stratum-Δ panel
    columns used (for diagnostic inspection).

    NaN throughout when n_strata < 3 (regression requires at
    least 3 points for a meaningful slope).
    """
    x: str
    y: str
    slope: float
    intercept: float
    slope_se: float
    p_value: float
    r_squared: float
    n_strata: int
    x_values: tuple[float, ...]
    y_values: tuple[float, ...]


def panel_regress(
    panel: StratumEffectPanel,
    *,
    x: str,
    y: str,
) -> PanelRegressionResult:
    """OLS regression of `panel.deltas[y]` on `panel.deltas[x]`
    across strata.

    Pure stratum-level inference: each stratum's Δ is one
    observation. No per-pair Δ structure. Strata where either
    Δ is NaN are dropped.

    Free function (not an @analysis fixture) because it composes
    on an already-resolved panel rather than from cells. Bridges
    consume `stratum_effect_panel` as the fixture and call
    `panel_regress` from the body to test a specific (x, y)
    relationship.
    """
    if x not in panel.deltas or y not in panel.deltas:
        raise KeyError(
            f'panel_regress: panel must include {x!r} and {y!r} '
            f'in its `measurables` tuple; got {panel.measurables}'
        )
    x_vals_all = panel.deltas[x]
    y_vals_all = panel.deltas[y]
    paired = [
        (xv, yv) for xv, yv in zip(x_vals_all, y_vals_all, strict=True)
        if not (math.isnan(xv) or math.isnan(yv))
    ]
    n = len(paired)
    if n < 3:
        return PanelRegressionResult(
            x=x, y=y,
            slope=float('nan'), intercept=float('nan'),
            slope_se=float('nan'), p_value=float('nan'),
            r_squared=float('nan'),
            n_strata=n,
            x_values=tuple(xv for xv, _ in paired),
            y_values=tuple(yv for _, yv in paired),
        )
    x_arr = np.asarray([xv for xv, _ in paired], dtype=np.float64)
    y_arr = np.asarray([yv for _, yv in paired], dtype=np.float64)
    lr = _stats.linregress(x_arr, y_arr)
    return PanelRegressionResult(
        x=x, y=y,
        slope=float(lr.slope),
        intercept=float(lr.intercept),
        slope_se=float(lr.stderr),
        p_value=float(lr.pvalue),
        r_squared=float(lr.rvalue ** 2),
        n_strata=n,
        x_values=tuple(x_arr.tolist()),
        y_values=tuple(y_arr.tolist()),
    )
