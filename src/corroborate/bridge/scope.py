"""Scope — gap-grounded scope of a Hypothesis claim.

The framework's two-phase output bundled into one slim record:

- Phase 1 (find scope) — `cleavage: MetaRegressionResult`
  carries the inverse-variance-weighted regression of the
  role's per-stratum effect sizes on the *invariance gap*
  aggregated per stratum. The regression's significant
  coefficient (CI excludes zero) corroborates the framework's
  central claim: "the mechanism activates where the invariant
  has a gap."
- Phase 2 (verify chain) — `chain: CausalGraph` is the typed
  causal graph from the Hypothesis subgraph verdict, with
  Tier-typed BridgeEdges. The chain's edges record whether
  each link of `env feature → invariance gap → mechanism
  activation → outcome` was corroborated.

Gap-grounded by design — the cleavage covariate isn't arbitrary;
it's a per-stratum reduction of the invariance gap declared by
the hypothesis's invariant bridge. `gap_name` records the
Measurable's identity; `threshold` evolves through the discovery
→ commit cycle (`None` during discovery, a `float` once the
author commits scope based on Phase-1's findings).

Slim by design — every field is either a primitive (`str`,
`float`, `float | None`) or a reference to an existing typed
artifact (`MetaRegressionResult`, `CausalGraph`). Adding new
types here would re-introduce v9's primitive proliferation."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from corroborate.graph.causal import CausalGraph
from corroborate.bridge.hypothesis_verdict import HypothesisVerdict
from corroborate.stats import (
    MetaRegressionResult,
    StratumObservation,
    meta_regression,
)
from corroborate.corpus.schema import MeasurementLeaf, RunRow


@dataclass(frozen=True, slots=True)
class Scope:
    """The scope of a Hypothesis claim — where the mechanism
    activates and via what causal chain.

    `hypothesis_name` — the Hypothesis whose scope this is.
    `gap_name` — the invariant Measurable's name (e.g.
      `'jensen_overestimation_gap'`). Identifies which gap was
      used as the cleavage axis; the Scope is gap-grounded by
      construction.
    `cleavage` — meta-regression of per-stratum g on the per-
      stratum gap (single covariate, named after `gap_name`).
      `cleavage.coefficients[0]` carries (coef, CI, p);
      `cleavage.cleavage_axes` is non-empty iff the gap predicts
      the effect.
    `chain` — the typed CausalGraph from Phase 2's verdict-walk,
      with Tier-typed BridgeEdges and verdicts per edge.
    `alpha` — significance threshold used in the meta-regression.
    `threshold` — `None` during discovery (no scope commitment
      yet); a `float` once the author commits scope based on
      Phase-1's findings. Discovery → commit cycle: run with
      `None`, observe the cleavage, choose a threshold separating
      in-scope from out-of-scope envs, re-run with that
      threshold to validate."""
    hypothesis_name: str
    gap_name: str
    cleavage: MetaRegressionResult
    chain: CausalGraph
    alpha: float
    threshold: float | None = None

    def is_in_scope(self, gap_value: float) -> bool:
        """Predicate: True iff `gap_value` is within the committed
        scope. Always True in discovery mode (`threshold=None`) —
        the predicate has no content until commitment.

        NaN gap is OUT of scope: data was missing, can't claim
        the env was in scope without evidence."""
        if math.isnan(gap_value):
            return False
        if self.threshold is None:
            return True
        return gap_value <= self.threshold


def _per_group_gap_mean(
    baseline_runs: Sequence[RunRow],
    *,
    group_by: str,
    gap_path: str,
) -> Mapping[str, float]:
    """Mean of `gap_path` over baseline runs, grouped by
    `group_by`. Cells with NaN or non-numeric gap are skipped.
    A group with all-NaN cells produces NaN."""
    grouped: dict[str, list[float]] = {}
    for r in baseline_runs:
        group_v = r.measurements.get(group_by)
        if not isinstance(group_v, (str, int, float, bool)):
            continue
        gap_v: MeasurementLeaf | None = r.measurements.get(gap_path)
        if not isinstance(gap_v, (int, float)) or isinstance(gap_v, bool):
            continue
        if math.isnan(float(gap_v)):
            continue
        grouped.setdefault(str(group_v), []).append(float(gap_v))
    return {
        k: sum(vs) / len(vs) if vs else float('nan')
        for k, vs in grouped.items()
    }


def build_scope(
    verdict: HypothesisVerdict[Mapping[str, object]],
    baseline_runs: Sequence[RunRow],
    *,
    gap_path: str,
    gap_name: str,
    target: str,
    alpha: float = 0.05,
    threshold: float | None = None,
    log_scale: bool = False,
) -> Scope:
    """Construct a `Scope` from a HypothesisVerdict + baseline
    runs + an invariance-gap path.

    Phase 1: aggregate per-stratum baseline gap mean from
    `baseline_runs.measurements[gap_path]`, build a single-
    covariate observation per stratum, and run `meta_regression`.
    Phase 2: take the typed CausalGraph already in the verdict.

    `gap_path` — where the per-cell gap_value is recorded on
    each RunRow's measurements. With proper invariant wiring this
    is `f'invariant.{at_most_bridge.name}.stats.gap_value'`; for
    corpora that only carry the cell-runner's flat projection it
    can also be `'jensen_gap'`.

    `gap_name` — the Measurable's name (e.g.
    `'jensen_overestimation_gap'`). Stored on `Scope.gap_name`;
    used as the regression covariate's name (with `'log_'` prefix
    when `log_scale=True`).

    `target` — the comparison's target path (the outcome path the
    hypothesis edge claims). Names which row in
    `verdict.comparison_rows` carries the per-stratum effect-sizes
    that the meta-regression cleaves on.

    `log_scale` — when True, regress on `log10(gap_mean)` instead
    of the raw mean. Necessary when gap magnitudes span many
    orders of magnitude across envs (e.g. Atari-MinAtar ~10⁶ vs
    bsuite ~10⁻³); raw scale would let one env dominate the
    weighted residual.

    `threshold` — committed scope threshold. None during
    discovery; passed through to `Scope.threshold`. Doesn't
    affect the regression; this is metadata recording the
    author's commitment.

    Strata where the per-env gap is zero (or negative) are dropped
    when `log_scale=True`. Strata with NaN g/se or non-positive
    se are dropped before regression — meta_regression's design
    matrix would carry NaN through to a numerical crash otherwise.

    Raises `ValueError` when the target doesn't have a comparison
    row in the verdict, when the comparison row's group_by is
    None, or when after filtering there are too few strata."""
    row = verdict.comparison_rows.get(target)
    if row is None:
        raise ValueError(
            f'build_scope: no comparison row for target={target!r}',
        )
    if row.group_by is None:
        raise ValueError(
            f'build_scope: comparison row for {target!r} '
            f'has group_by=None — meta-regression needs strata',
        )

    gap_by_group = _per_group_gap_mean(
        baseline_runs, group_by=row.group_by, gap_path=gap_path,
    )
    cov_name = f'log_{gap_name}' if log_scale else gap_name

    observations: list[StratumObservation] = []
    for gs in row.per_group:
        if gs.effect_size_g is None or gs.se is None:
            continue
        if math.isnan(gs.effect_size_g) or math.isnan(gs.se):
            continue
        if gs.se <= 0.0:
            continue
        gap_v = gap_by_group.get(str(gs.group_value))
        if gap_v is None or math.isnan(gap_v):
            continue
        if log_scale:
            if gap_v <= 0.0:
                continue
            cov_val = math.log10(gap_v)
        else:
            cov_val = gap_v
        observations.append(StratumObservation(
            stratum_id=gs.group_value,
            g=gs.effect_size_g,
            se=gs.se,
            covariates={cov_name: cov_val},
        ))

    cleavage = meta_regression(observations, alpha=alpha)
    return Scope(
        hypothesis_name=verdict.hypothesis.name,
        gap_name=gap_name,
        cleavage=cleavage,
        chain=verdict.graph,
        alpha=alpha,
        threshold=threshold,
    )
