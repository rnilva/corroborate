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

Cell-set scoping (env, HP equality, threshold gates, arbitrary
predicates) lives upstream on `Bridge.scope` as a polars
`pl.Expr`; `claim_bridge.evaluate()` filters before this analysis
sees the cells. Analyses pair, never scope.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

from corroborate.analyses._panel import per_stratum_panel
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

    A *present* key with a None / NaN / non-numeric value is
    treated as a cached miss (returns NaN) — DO NOT fall through
    to the measurable. The bridge cache writes None for cells
    where the measurable couldn't resolve at build time (e.g.
    corpus without traces); recomputing here would re-trigger the
    same failure with no new information AND mask the universal-
    merge schema heterogeneity. Only an *absent* key falls through
    to the registry.

    Raises `KeyError` if `source` isn't in the record AND no
    measurable is registered under that name."""
    if source in record:
        v = record[source]
        if isinstance(v, bool):
            return float(v)
        if isinstance(v, (int, float)):
            return float(v)
        return float('nan')
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


@analysis
def paired_g(
    cells: Iterable[Mapping[str, object]],
    *,
    source: str,
    treatment_arm: str,
    baseline_arm: str,
    pair_by: tuple[str, ...] = ('seed',),
    arm_field: str = 'intervention_name',
    dedupe_strategy: str = 'raise',
) -> PairedGResult:
    """Compute paired Hedges' g + raw mean-diff at `source` across
    matched (T, B) pairs in `cells`.

    Pairing is by string match on `arm_field` (typically
    `intervention_name`). `treatment_arm` and `baseline_arm` come
    from the bridge's `DoEffect` — either via per-bridge
    `source = DoEffect(...)` decorator override or via file-level
    `INTERVENTION = DoEffect(...)` resolved at decoration time.
    `claim_bridge.evaluate()` extracts the contrast and forwards
    the arm strings into this analysis's kwargs.

    `source` resolves through the measurable registry (preferred)
    or as a field-path read on the cell record.

    Cell-level scoping (env, HP equality, threshold gates,
    arbitrary predicates) lives on `Bridge.scope` as a polars
    `pl.Expr`; `claim_bridge.evaluate()` filters before this
    analysis sees the cells. The analysis itself does not scope.

    `dedupe_strategy` controls the policy when multiple cells share
    the same `(arm, pair_by)` tuple:
    - `'raise'` (default): error loudly. The dict-overwrite would
      silently drop data; force the bridge author to either tighten
      `pair_by` or opt into aggregation.
    - `'mean'`: average the per-cell `source` values within each
      `(arm, pair_by)` bucket, then run paired-g on the aggregated
      values. The intended use is M2M scenarios where the user
      genuinely wants to pool across e.g. multiple corpora at the
      same `(seed, env)`."""
    from corroborate.statistics import hedges_g_paired

    if dedupe_strategy not in ('raise', 'mean'):
        raise ValueError(
            f'paired_g: unknown dedupe_strategy {dedupe_strategy!r}; '
            f'expected "raise" or "mean"',
        )
    treatment_buckets: dict[tuple[object, ...], list[float]] = {}
    baseline_buckets: dict[tuple[object, ...], list[float]] = {}
    for cell in cells:
        arm = cell.get(arm_field)
        if arm == treatment_arm:
            key = _key_tuple(cell, pair_by)
            bucket = treatment_buckets.setdefault(key, [])
            if bucket and dedupe_strategy == 'raise':
                raise ValueError(
                    f'paired_g: duplicate cell for {treatment_arm!r} at '
                    f'pair_by={pair_by} key={key}. The dict-overwrite '
                    f'silently kept the last-written value, dropping '
                    f'data. Tighten `pair_by` to a discriminating tuple, '
                    f'set dedupe_strategy="mean" to aggregate the '
                    f'cells, or use an M2M-friendly analysis '
                    f'(e.g. stratified Spearman) instead.',
                )
            bucket.append(_resolve_value(cell, source))
        elif arm == baseline_arm:
            key = _key_tuple(cell, pair_by)
            bucket = baseline_buckets.setdefault(key, [])
            if bucket and dedupe_strategy == 'raise':
                raise ValueError(
                    f'paired_g: duplicate cell for {baseline_arm!r} at '
                    f'pair_by={pair_by} key={key}. The dict-overwrite '
                    f'silently kept the last-written value, dropping '
                    f'data. Tighten `pair_by` to a discriminating tuple, '
                    f'set dedupe_strategy="mean" to aggregate the '
                    f'cells, or use an M2M-friendly analysis '
                    f'(e.g. stratified Spearman) instead.',
                )
            bucket.append(_resolve_value(cell, source))

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
    # NaN-skip pairs where either side is missing (e.g. cells from
    # corpora that didn't carry the source column, surfacing as
    # NaN through `_resolve_value`'s present-but-None path). The
    # statistics primitives reject NaN-containing inputs; better
    # to filter at the analysis boundary than to crash.
    deltas = [
        treatment[k] - baseline[k]
        for k in paired_keys
        if not (math.isnan(treatment[k]) or math.isnan(baseline[k]))
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
) -> tuple[StratumG[str], ...]:
    """Per-env paired-g panel — one `StratumG[str]` per env in
    `env_filter` (or every env present in `cells` when empty).

    Wraps the generic `per_stratum_panel`: stratifies cells by
    `env_name`, calls `paired_g.fn` on each env's subset, packs
    the result as `StratumG[str]`. Consumers (`paired_g_pooled`,
    `paired_g_among_solvers`, `meta_regression_paired_g`) iterate
    the panel rather than re-pairing.

    NO panel-level filtering: every env in the target set produces
    an entry, including degenerate ones (n_pairs<2 → g/se=NaN).
    Consumers that need to drop underpowered strata filter at
    their own boundary so they can decide what to report (e.g. an
    explicit `n_pairs=0` entry tells `paired_g_among_solvers`
    "this env was in `gate_thresholds` but no surviving pair")."""
    def _stratify(cell: Mapping[str, object]) -> str | None:
        env = cell.get('env_name')
        return env if isinstance(env, str) else None

    key_filter: Callable[[str], bool] | None = (
        (lambda env: env in env_filter) if env_filter else None
    )

    def _analyze(subset: Sequence[Mapping[str, object]]) -> PairedGResult:
        return paired_g.fn(
            subset,
            treatment_arm=treatment_arm,
            baseline_arm=baseline_arm,
            pair_by=pair_by,
            source=source,
            arm_field=arm_field,
        )

    panel = per_stratum_panel(
        cells, stratify_by=_stratify, analysis=_analyze,
        key_filter=key_filter,
    )
    return tuple(
        StratumG[str](
            stratum_id=env, g=r.g, se=r.se, n_pairs=r.n_pairs,
        )
        for env, r in panel
    )


__all__ = ['PairedGResult', 'paired_g', 'per_env_paired_g_panel']
