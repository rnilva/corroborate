"""`paired_g` — paired Hedges' g + raw-diff across pair keys.

The headline analysis shape for paired intervention claims.
Pairs treatment cells with baseline cells on a key tuple
(typically `('seed',)`), computes per-pair Δ on a `source`
quantity, returns BOTH standardized Hedges' g AND raw mean-diff
in `PairedGResult`. Bridges pick whichever they need:
standardized-g when comparing across heterogeneous-scale envs,
raw mean-diff when reward magnitude is itself the intervention
axis (Hedges' g standardizes that away — the under-learning
rescue case).

`source` resolves through the measurable registry first
(fixture-style: declare a `@measurable outcome_native` and any
analysis can request it by name), falling back to a direct
field-path read on the cell record. This is the convergence
point for the framework's "(1) claim outputs + (2) post-run
measurables" architecture: ALL per-cell quantities (raw or
derived) become resolvable by name through one resolver,
and analyses are generic over what they consume.

`extra_filters` lets bridges scope to any (env, reward_scale,
total_steps, …) sub-corpus without bespoke per-bridge analyses.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from corroborate.analysis import analysis


@dataclass(frozen=True, slots=True)
class PairedGResult:
    """Output of paired Hedges' g across pair-keys.

    `g` and `se` are the standardized effect size + its SE.
    `mean_diff` and `mean_diff_se` are the raw paired mean
    difference + its standard error — the same input cells, but
    NOT pooled-SD-scaled. Bridges that test interventions on the
    reward magnitude itself must consume `mean_diff` (Hedges' g
    cancels reward-scale variance via the pooled SD).

    `helped_fraction` is the fraction of pairs with positive Δ
    (treatment > baseline) — the count-style report a number of
    bridges want alongside the standardized magnitude. NaN when
    `n_pairs == 0`.

    All other quantities are NaN if `n_pairs < 2` or per-pair Δ
    has zero spread."""
    g: float
    se: float
    mean_diff: float
    mean_diff_se: float
    n_pairs: int
    n_treatment: int
    n_baseline: int
    helped_fraction: float
    pair_by: tuple[str, ...]
    measurable: str
    treatment_arm: str
    baseline_arm: str

    @property
    def p_value(self) -> float:
        """Two-sided p-value for `g != 0` from |g/se| → z under
        normal approximation. NaN when `g`/`se` are NaN or `se`
        is zero."""
        if math.isnan(self.g) or math.isnan(self.se) or self.se == 0.0:
            return float('nan')
        z = abs(self.g / self.se)
        return math.erfc(z / math.sqrt(2))

    @property
    def mean_diff_p_value(self) -> float:
        """Two-sided p-value for `mean_diff != 0` from a paired
        t-test on the per-pair Δ — uses |t/se| with df = n_pairs−1.
        Distinct from the standardized-g p_value: same paired
        Δ's, but the test stat doesn't divide by pooled SD.
        NaN under the same degenerate conditions."""
        if math.isnan(self.mean_diff) or math.isnan(self.mean_diff_se) \
                or self.mean_diff_se == 0.0 or self.n_pairs < 2:
            return float('nan')
        from scipy.stats import t as _t
        t_stat = abs(self.mean_diff / self.mean_diff_se)
        return float(2.0 * (1.0 - _t.cdf(t_stat, df=self.n_pairs - 1)))


def _resolve_value(record: Mapping[str, object], source: str) -> float:
    """Resolve `source` from a cell record. Tries the measurable
    registry first; falls back to a direct field-path read.

    The measurable route lets analyses reference any registered
    `@measurable` by name (`outcome_native`, `effective_horizon`,
    …); per-cell values are computed via the Measurable's
    `__call__`. The field-path fallback covers raw claim outputs
    that aren't wrapped as measurables.

    Raises `KeyError` if neither route resolves to a scalar."""
    from corroborate.measurable import get_registered as _get_m
    m = _get_m(source)
    if m is not None:
        v: object = m(record)
        if isinstance(v, (int, float)):
            return float(v)
        raise TypeError(
            f'measurable {source!r} returned non-scalar '
            f'{type(v).__name__}; paired-g source must be scalar',
        )
    v = record.get(source)
    if isinstance(v, (int, float)):
        return float(v)
    raise KeyError(
        f'no measurable named {source!r} and no scalar at path '
        f'{source!r} in record',
    )


def _key_tuple(
    record: Mapping[str, object], pair_by: tuple[str, ...],
) -> tuple[object, ...]:
    return tuple(record[k] for k in pair_by)


def _matches_filters(
    cell: Mapping[str, object],
    extra_filters: Mapping[str, object],
) -> bool:
    """Check `cell` matches every (key, value) in `extra_filters`.
    Numeric comparisons use a small absolute tolerance so float
    keys like `reward_scale=0.1` from YAML parse equal regardless
    of representation drift."""
    for k, expected in extra_filters.items():
        actual = cell.get(k)
        if isinstance(expected, float) and isinstance(actual, (int, float)):
            if abs(float(actual) - float(expected)) > 1e-9:
                return False
        elif actual != expected:
            return False
    return True


@analysis
def paired_g(
    cells: Iterable[Mapping[str, object]],
    *,
    treatment_arm: str,
    baseline_arm: str,
    pair_by: tuple[str, ...] = ('seed',),
    source: str,
    env_name: str | None = None,
    arm_field: str = 'intervention_name',
    extra_filters: Mapping[str, object] = MappingProxyType({}),
    extra_min_pairs: tuple[tuple[str, float], ...] = (),
    extra_max_pairs: tuple[tuple[str, float], ...] = (),
) -> PairedGResult:
    """Pair `treatment_arm` cells with `baseline_arm` cells on
    `pair_by`, compute per-pair Δ at `source`, return Hedges' g
    + raw mean-diff (both with their SEs) + helped-fraction.

    `source` resolves through the measurable registry (preferred)
    or as a field-path read on the cell record. Bridges declare
    `source='outcome_native'` to consume the registered
    measurable, or any field-path string for a raw column.

    `env_name`, `extra_filters`, `extra_min_pairs`, and
    `extra_max_pairs` scope the corpus pre-pairing. Equality
    via `extra_filters={'reward_scale': 0.1}`; numeric thresholds
    via `extra_min_pairs=(('effective_horizon', 50.0),)` (column
    ≥ value) or `extra_max_pairs=(('total_steps', 200000.0),)`
    (column ≤ value). All filters AND together; cells not
    satisfying any predicate drop out before pairing."""
    from corroborate.statistics import hedges_g_paired

    treatment: dict[tuple[object, ...], float] = {}
    baseline: dict[tuple[object, ...], float] = {}
    for cell in cells:
        if env_name is not None and cell.get('env_name') != env_name:
            continue
        if extra_filters and not _matches_filters(cell, extra_filters):
            continue
        if not _matches_thresholds(cell, extra_min_pairs, extra_max_pairs):
            continue
        arm = cell.get(arm_field)
        if arm == treatment_arm:
            treatment[_key_tuple(cell, pair_by)] = _resolve_value(
                cell, source,
            )
        elif arm == baseline_arm:
            baseline[_key_tuple(cell, pair_by)] = _resolve_value(
                cell, source,
            )

    paired_keys = sorted(set(treatment) & set(baseline))
    deltas = [
        treatment[k] - baseline[k] for k in paired_keys
    ]
    n_pairs = len(deltas)

    if n_pairs >= 2:
        g, se = hedges_g_paired(deltas)
        n = float(n_pairs)
        mean_diff = sum(deltas) / n
        sd = math.sqrt(
            sum((d - mean_diff) ** 2 for d in deltas) / (n - 1.0),
        )
        mean_diff_se = sd / math.sqrt(n)
    else:
        g = se = mean_diff = mean_diff_se = float('nan')
    helped_fraction = (
        sum(1 for d in deltas if d > 0.0) / n_pairs
        if n_pairs > 0 else float('nan')
    )

    return PairedGResult(
        g=g, se=se,
        mean_diff=mean_diff,
        mean_diff_se=mean_diff_se,
        n_pairs=n_pairs,
        n_treatment=len(treatment),
        n_baseline=len(baseline),
        helped_fraction=helped_fraction,
        pair_by=pair_by,
        measurable=source,
        treatment_arm=treatment_arm,
        baseline_arm=baseline_arm,
    )


def _matches_thresholds(
    cell: Mapping[str, object],
    min_pairs: tuple[tuple[str, float], ...],
    max_pairs: tuple[tuple[str, float], ...],
) -> bool:
    """All `(col, val)` in `min_pairs` require `cell[col] >= val`;
    all in `max_pairs` require `cell[col] <= val`. NaN or missing
    values fail the predicate (defensive — don't include cells
    with ambiguous threshold positioning)."""
    for col, thr in min_pairs:
        v = cell.get(col)
        if not isinstance(v, (int, float)):
            return False
        f = float(v)
        if math.isnan(f) or f < thr:
            return False
    for col, thr in max_pairs:
        v = cell.get(col)
        if not isinstance(v, (int, float)):
            return False
        f = float(v)
        if math.isnan(f) or f > thr:
            return False
    return True


__all__ = ['PairedGResult', 'paired_g']
