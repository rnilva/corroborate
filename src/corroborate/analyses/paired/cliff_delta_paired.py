"""Cliff's δ paired — rank-based effect size complementing paired_g.

Cliff's δ is the **non-parametric** analog of Hedges' g:

    δ = P(Δ > 0) - P(Δ < 0)

where Δ is the per-pair (treatment - baseline) difference. Unlike
Hedges' g (which is `mean(Δ) / sd(Δ) · c_4(n)`), Cliff's δ depends
ONLY on the SIGNS of Δ — it ignores magnitude and is therefore
**skew-robust by construction**.

Range: δ ∈ [-1, 1].
  -1 → treatment ALWAYS worse than baseline (every pair).
   0 → treatment as often better as worse.
  +1 → treatment ALWAYS better.

Rough magnitude conventions (Romano et al. 2006):
  |δ| < 0.147  →  negligible
  |δ| < 0.330  →  small
  |δ| < 0.474  →  medium
  |δ| ≥ 0.474  →  large

When to use:
  - Skewed Δ (paired_g is biased upward; see ROBUSTNESS.md).
  - Bounded outcomes near saturation (mean and sd are biased).
  - When the bridge's claim is "treatment helps in MOST pairs"
    rather than "the average effect size is large." Magnitude-
    independent semantics — useful when reward scales differ
    across cells.

When NOT to use:
  - Bridges that assert standardized magnitude ("g > 0.5"). Cliff's
    δ doesn't measure magnitude; use paired_g.
  - Tiny n (< 10): SE on δ is wide, MDE comparable to ±0.5.

This primitive is structurally distinct from paired_g. It does not
replace paired_g — it complements it. Substrate-author guidance is
in ROBUSTNESS.md.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import polars as pl

from corroborate._internals.polars import as_rows
from corroborate.analyses._cell_value import key_tuple, resolve_value
from corroborate.bridge.analysis import analysis


@dataclass(frozen=True, slots=True)
class CliffDeltaResult:
    """Output of Cliff's δ paired across pair-keys.

    `delta` ∈ [-1, 1]: P(Δ>0) - P(Δ<0). Skew-robust by construction.
    `se` is the analytical SE under Cliff's (1996) variance formula:
        Var(δ) = (1 - δ²) / (n - 1)         (large-n approximation)
    `n_pairs` is the count of (T, B) pairs after `pair_by` matching.
    `n_positive`, `n_negative`, `n_tied` are the per-pair sign tallies
    (sum to n_pairs). Ties (Δ == 0 exactly) contribute neither to
    n_positive nor n_negative — the framework reports n_tied rather
    than choosing a tie-handling convention.

    NaN-filled when n_pairs < 2."""
    delta: float
    se: float
    n_pairs: int
    n_positive: int
    n_negative: int
    n_tied: int
    pair_by: tuple[str, ...]
    measurable: str
    treatment_arm: str
    baseline_arm: str

    @property
    def p_value(self) -> float:
        """Two-sided p-value for `δ != 0` from |δ/se| under
        normal approximation. NaN under the same degenerate
        conditions as `delta`/`se`."""
        if math.isnan(self.delta) or math.isnan(self.se) or self.se == 0.0:
            return float('nan')
        z = abs(self.delta / self.se)
        return math.erfc(z / math.sqrt(2))


@analysis
def cliff_delta_paired(
    cells: pl.DataFrame | Iterable[Mapping[str, object]],
    *,
    source: str,
    treatment_arm: str,
    baseline_arm: str,
    pair_by: tuple[str, ...] = ('seed',),
    arm_field: str = 'arm_key',
    dedupe_strategy: str = 'mean',
) -> CliffDeltaResult:
    """Compute Cliff's δ paired across matched (T, B) pairs.

    The pairing surface mirrors `paired_g`: same `pair_by`,
    `arm_field`, `dedupe_strategy` semantics (defaults to `'mean'`
    aggregation; pass `'raise'` to error on duplicate
    `(arm, pair_by)` cells). Bridges authored to consume both
    primitives can swap kwargs between them without further
    translation.

    See module docstring for when to use Cliff's δ vs paired_g."""
    cells = as_rows(cells)
    if dedupe_strategy not in ('raise', 'mean'):
        raise ValueError(
            f'cliff_delta_paired: unknown dedupe_strategy '
            f'{dedupe_strategy!r}; expected "raise" or "mean"',
        )
    treatment_buckets: dict[tuple[object, ...], list[float]] = {}
    baseline_buckets: dict[tuple[object, ...], list[float]] = {}
    for cell in cells:
        arm = cell.get(arm_field)
        if arm == treatment_arm:
            key = key_tuple(cell, pair_by)
            bucket = treatment_buckets.setdefault(key, [])
            if bucket and dedupe_strategy == 'raise':
                raise ValueError(
                    f'cliff_delta_paired: duplicate cell for '
                    f'{treatment_arm!r} at pair_by={pair_by} key={key}.',
                )
            bucket.append(resolve_value(cell, source))
        elif arm == baseline_arm:
            key = key_tuple(cell, pair_by)
            bucket = baseline_buckets.setdefault(key, [])
            if bucket and dedupe_strategy == 'raise':
                raise ValueError(
                    f'cliff_delta_paired: duplicate cell for '
                    f'{baseline_arm!r} at pair_by={pair_by} key={key}.',
                )
            bucket.append(resolve_value(cell, source))

    treatment: dict[tuple[object, ...], float] = {
        k: (
            sum(v for v in vs if not math.isnan(v))
            / max(1, sum(1 for v in vs if not math.isnan(v)))
        ) if any(not math.isnan(v) for v in vs) else float('nan')
        for k, vs in treatment_buckets.items()
    }
    baseline: dict[tuple[object, ...], float] = {
        k: (
            sum(v for v in vs if not math.isnan(v))
            / max(1, sum(1 for v in vs if not math.isnan(v)))
        ) if any(not math.isnan(v) for v in vs) else float('nan')
        for k, vs in baseline_buckets.items()
    }

    paired_keys = sorted(set(treatment) & set(baseline))
    deltas = [
        treatment[k] - baseline[k]
        for k in paired_keys
        if not (math.isnan(treatment[k]) or math.isnan(baseline[k]))
    ]
    n_pairs = len(deltas)
    n_positive = sum(1 for d in deltas if d > 0.0)
    n_negative = sum(1 for d in deltas if d < 0.0)
    n_tied = n_pairs - n_positive - n_negative

    if n_pairs >= 2:
        delta = (n_positive - n_negative) / n_pairs
        # Cliff (1996) large-n SE; tighter formulas exist (Feng &
        # Cliff 2004) but require pairwise dominance counts that
        # add complexity without changing the verdict layer's
        # routing. The (1 - δ²)/(n-1) form is the standard
        # textbook approximation.
        var = (1.0 - delta * delta) / max(1, n_pairs - 1)
        se = math.sqrt(var)
    else:
        delta = se = float('nan')

    return CliffDeltaResult(
        delta=delta,
        se=se,
        n_pairs=n_pairs,
        n_positive=n_positive,
        n_negative=n_negative,
        n_tied=n_tied,
        pair_by=pair_by,
        measurable=source,
        treatment_arm=treatment_arm,
        baseline_arm=baseline_arm,
    )


__all__ = ['CliffDeltaResult', 'cliff_delta_paired']
