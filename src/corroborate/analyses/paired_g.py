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
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from corroborate.analysis import analysis
from corroborate.stratum import StratumG


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
    """Resolve `source` from a cell record. Tries the persisted
    field-path read first; falls back to the measurable registry.

    Record-first matches the "persisted columns are authoritative"
    discipline: when cell_runner has already persisted a scalar at
    `source`, downstream analyses use that value rather than
    recomputing via the registered measurable (which might fail
    on a corpus row that doesn't carry the source-side raw arrays).
    The measurable fallback covers analyses that request a derived
    quantity by name on a record where only the raw inputs were
    persisted (e.g. ad-hoc reductions over the trace store).

    Raises `KeyError` if neither route resolves to a scalar."""
    v = record.get(source)
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    from corroborate.measurable import get_registered as _get_m
    m = _get_m(source)
    if m is not None:
        computed: object = m(record)
        if isinstance(computed, (int, float)):
            return float(computed)
        raise TypeError(
            f'measurable {source!r} returned non-scalar '
            f'{type(computed).__name__}; paired-g source must be scalar',
        )
    raise KeyError(
        f'no scalar at path {source!r} in record and no measurable '
        f'named {source!r}',
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
    cell_predicate: Callable[[Mapping[str, object]], bool] | None = None,
) -> PairedGResult:
    """Pair `treatment_arm` cells with `baseline_arm` cells on
    `pair_by`, compute per-pair Δ at `source`, return Hedges' g
    + raw mean-diff (both with their SEs) + helped-fraction.

    `source` resolves through the measurable registry (preferred)
    or as a field-path read on the cell record. Bridges declare
    `source='outcome_native'` to consume the registered
    measurable, or any field-path string for a raw column.

    `env_name`, `extra_filters`, `extra_min_pairs`, `extra_max_pairs`,
    and `cell_predicate` scope the corpus pre-pairing. Equality
    via `extra_filters={'reward_scale': 0.1}`; numeric thresholds
    via `extra_min_pairs=(('effective_horizon', 50.0),)` (column
    ≥ value) or `extra_max_pairs=(('total_steps', 200000.0),)`
    (column ≤ value). `cell_predicate` is a callable for richer
    per-cell scope (e.g. "both arms cleared an env-specific solve
    threshold"). All filters AND together; cells not satisfying
    every predicate drop out before pairing."""
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
        if cell_predicate is not None and not cell_predicate(cell):
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


# ============ per-env panel helper ============

def per_env_paired_g_panel(
    cells: Sequence[Mapping[str, object]],
    *,
    treatment_arm: str,
    baseline_arm: str,
    source: str,
    env_filter: tuple[str, ...] = (),
    pair_by: tuple[str, ...] = ('seed',),
    arm_field: str = 'intervention_name',
    extra_filters: Mapping[str, object] = MappingProxyType({}),
    extra_min_pairs: tuple[tuple[str, float], ...] = (),
    extra_max_pairs: tuple[tuple[str, float], ...] = (),
    cell_predicate: Callable[[Mapping[str, object]], bool] | None = None,
) -> tuple[StratumG[str], ...]:
    """Per-env paired-g panel — one `StratumG[str]` per env in
    `env_filter` (or every env present in `cells` when empty).

    Calls `paired_g.fn` per env (with `env_name=env` and the
    optional `cell_predicate` / threshold filters), packs the
    per-env result as `StratumG[str]`. NO panel-level filtering:
    every env in the target set produces an entry, including
    degenerate ones (n_pairs<2 → g/se=NaN). Consumers that need
    to drop underpowered strata filter at their own boundary so
    they can decide what to report (e.g. an explicit `n_pairs=0`
    entry tells `paired_g_among_solvers` "this env was in
    `gate_thresholds` but no surviving pair").

    The framework's shared per-env loop primitive — consumers
    (`paired_g_pooled`, `paired_g_among_solvers`,
    `meta_regression_paired_g`) iterate the panel rather than
    re-pairing."""
    envs_seen: set[str] = set()
    for c in cells:
        env_v = c.get('env_name')
        if isinstance(env_v, str):
            envs_seen.add(env_v)
    target_envs: tuple[str, ...]
    if env_filter:
        target_envs = tuple(e for e in env_filter if e in envs_seen)
    else:
        target_envs = tuple(sorted(envs_seen))

    panel: list[StratumG[str]] = []
    for env in target_envs:
        result = paired_g.fn(
            cells,
            treatment_arm=treatment_arm,
            baseline_arm=baseline_arm,
            pair_by=pair_by,
            source=source,
            env_name=env,
            arm_field=arm_field,
            extra_filters=extra_filters,
            extra_min_pairs=extra_min_pairs,
            extra_max_pairs=extra_max_pairs,
            cell_predicate=cell_predicate,
        )
        panel.append(StratumG[str](
            stratum_id=env,
            g=result.g,
            se=result.se,
            n_pairs=result.n_pairs,
        ))
    return tuple(panel)


__all__ = ['PairedGResult', 'paired_g', 'per_env_paired_g_panel']
