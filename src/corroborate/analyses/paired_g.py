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
    # Auto-discover path populates the configurable column whose
    # value defines the contrast (e.g. `'bootstrap'`); legacy
    # arm-name path leaves it `None`.
    intervention_column: str | None = None

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


# ============ Auto-discover pair-coherence helper ============

# Columns the auto-discover path strips before computing
# pair-coherence and intervention-column candidacy. Mirrors the
# provenance-strip done by `RunRow.from_row_dict` so the auto-
# discover surface is robust to flat cell-dict input from parquet.
_PROVENANCE_KEYS_AUTO: frozenset[str] = frozenset({
    'id', 'parent_id', 'cycle_id', 'timestamp', 'verdict',
    'arm_key', 'claim_graph_signature', 'intervention_name',
})


def _configurable_columns(
    cells: Sequence[Mapping[str, object]],
    *,
    exogenous_keys: frozenset[str],
    source: str,
) -> tuple[str, ...]:
    """Names of columns that participate in pair-coherence:
    every key in the cells EXCLUDING provenance, registered
    measurables, substrate-exogenous keys, and the bridge's
    `source` (we don't want the source measurable itself to
    define the contrast). Returns sorted; includes only keys
    whose values are scalar (string / int / float / bool) — list
    or struct columns can't be compared for equality across
    cells without ambiguity."""
    from corroborate.measurable import registered_names
    excluded = (
        _PROVENANCE_KEYS_AUTO
        | exogenous_keys
        | frozenset(registered_names())
        | {source}
    )
    seen: set[str] = set()
    for c in cells:
        for k, v in c.items():
            if k in excluded or k in seen:
                continue
            # Framework-emitted bracketed namespaces (e.g.
            # `at_most[jensen_dormancy_gap<=0].reason`,
            # `.stats.*`, `.targets`) are invariant-evaluation
            # results — runtime outputs, not configurable keys.
            # Only the `.verdict` is registered; the siblings
            # surface as columns alongside but should be filtered
            # from pair-coherence checks.
            if '[' in k and ']' in k:
                continue
            if isinstance(v, (str, int, float, bool)):
                seen.add(k)
    return tuple(sorted(seen))


def _paired_g_auto_discover(
    cells: Iterable[Mapping[str, object]],
    *,
    source: str,
    pair_by: tuple[str, ...],
    env_name: str | None,
    extra_filters: Mapping[str, object],
    extra_min_pairs: tuple[tuple[str, float], ...],
    extra_max_pairs: tuple[tuple[str, float], ...],
    cell_predicate: Callable[[Mapping[str, object]], bool] | None,
    exogenous_keys: frozenset[str],
    intervention_slot: str | None,
) -> PairedGResult:
    """Pair-coherent auto-discover: T and B agree on every
    configurable column except exactly one — that column IS the
    intervention slot, identified from the data.

    Algorithm:
    1. Filter cells by all scope predicates.
    2. Among configurable columns, find the unique one with
       exactly two distinct values across the filtered set
       (skipping `pair_by` columns since those define the pairing
       axis, not the contrast). That's the contrast column; its
       two values are treatment / baseline (deterministic by
       sort order).
    3. Group cells by `pair_by` tuple. Within each group, take
       cells whose contrast-column value is the treatment value
       and cells whose value is the baseline value. Both sides
       must additionally agree on every OTHER configurable
       column for the pair to be coherent.
    4. Compute Δ at source for each coherent pair.

    `n_pairs == 0` (NaN g/se) under any of:
    - 0 configurable columns vary (no contrast available).
    - 2+ configurable columns vary (ambiguous; bridge author
      must scope tighter via `extra_filters`).
    - Contrast column has more than 2 distinct values (panel
      shape; use a different analysis tool).
    - No coherent pairs survive the configurable-key match
      (e.g. cells from different sweeps differ on >1 column)."""
    from corroborate.statistics import hedges_g_paired

    filtered: list[Mapping[str, object]] = []
    for cell in cells:
        if env_name is not None and cell.get('env_name') != env_name:
            continue
        if extra_filters and not _matches_filters(cell, extra_filters):
            continue
        if not _matches_thresholds(cell, extra_min_pairs, extra_max_pairs):
            continue
        if cell_predicate is not None and not cell_predicate(cell):
            continue
        filtered.append(cell)

    if not filtered:
        return _empty_result(pair_by, source, intervention_column=None)

    # Identify configurable columns (excluding pair_by axes —
    # those are the pairing dimensions, not contrast candidates).
    pair_by_set = frozenset(pair_by)
    config_cols = tuple(
        c for c in _configurable_columns(
            filtered, exogenous_keys=exogenous_keys, source=source,
        )
        if c not in pair_by_set
    )

    # Find columns with multiple distinct values in scope.
    varying: dict[str, set[object]] = {}
    for c in filtered:
        for col in config_cols:
            if col not in c:
                continue
            v = c[col]
            varying.setdefault(col, set()).add(v)
    multi_value_cols = tuple(
        col for col, vs in varying.items() if len(vs) >= 2
    )

    # Bridge-supplied intervention_slot wins over heuristic
    # auto-discover. Useful when scope has orthogonal HP variation
    # alongside the structural intervention; scope-tightening
    # could remove it but the bridge author often prefers to
    # name the slot directly.
    if intervention_slot is not None:
        if intervention_slot not in multi_value_cols:
            return _empty_result(
                pair_by, source, intervention_column=intervention_slot,
            )
        contrast_col = intervention_slot
    elif len(multi_value_cols) != 1:
        # 0 → no contrast; 2+ → ambiguous. Either way: empty.
        return _empty_result(pair_by, source, intervention_column=None)
    else:
        contrast_col = multi_value_cols[0]
    contrast_values = tuple(sorted(  # deterministic ordering
        str(v) for v in varying[contrast_col]
    ))
    if len(contrast_values) != 2:
        return _empty_result(
            pair_by, source, intervention_column=contrast_col,
        )
    baseline_val, treatment_val = contrast_values

    # Group cells by (pair_by_tuple, contrast_value).
    by_pair_key: dict[
        tuple[object, ...], dict[str, list[Mapping[str, object]]],
    ] = {}
    for c in filtered:
        pkey = _key_tuple(c, pair_by)
        cval = str(c.get(contrast_col))
        by_pair_key.setdefault(pkey, {}).setdefault(cval, []).append(c)

    # For each pair_by key, find pair-coherent (T, B) — they
    # must agree on every configurable column except `contrast_col`
    # AND its sub-paths. Sub-leaves at-or-under the contrast slot
    # (e.g. `bootstrap.greedification` when contrast_col is
    # `bootstrap`) co-vary with the slot itself; treating them
    # as independent coherence constraints would reject every
    # legitimate pair.
    contrast_prefix = contrast_col + '.'
    other_cols = tuple(
        c for c in config_cols
        if c != contrast_col and not c.startswith(contrast_prefix)
    )
    deltas: list[float] = []
    n_treatment = 0
    n_baseline = 0
    for pkey, by_val in by_pair_key.items():
        ts = by_val.get(treatment_val, [])
        bs = by_val.get(baseline_val, [])
        n_treatment += len(ts)
        n_baseline += len(bs)
        for t_cell in ts:
            t_other = tuple(t_cell.get(c) for c in other_cols)
            for b_cell in bs:
                b_other = tuple(b_cell.get(c) for c in other_cols)
                if t_other != b_other:
                    continue  # confounded — differs on >1 column
                t_v = _resolve_value(t_cell, source)
                b_v = _resolve_value(b_cell, source)
                if math.isnan(t_v) or math.isnan(b_v):
                    continue
                deltas.append(t_v - b_v)
                break  # one match per (pkey, t_cell)

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
        n_treatment=n_treatment,
        n_baseline=n_baseline,
        helped_fraction=helped_fraction,
        pair_by=pair_by,
        measurable=source,
        treatment_arm=treatment_val,
        baseline_arm=baseline_val,
        intervention_column=contrast_col,
    )


def _empty_result(
    pair_by: tuple[str, ...], source: str,
    intervention_column: str | None,
) -> PairedGResult:
    """Degenerate `PairedGResult` for the auto-discover path's
    no-contrast / ambiguous-contrast / no-coherent-pairs cases.
    `intervention_column` is the discovered slot if discovery
    got that far before falling through, else None."""
    return PairedGResult(
        g=float('nan'), se=float('nan'),
        mean_diff=float('nan'), mean_diff_se=float('nan'),
        n_pairs=0,
        n_treatment=0, n_baseline=0,
        helped_fraction=float('nan'),
        pair_by=pair_by,
        measurable=source,
        treatment_arm='',
        baseline_arm='',
        intervention_column=intervention_column,
    )


@analysis
def paired_g(
    cells: Iterable[Mapping[str, object]],
    *,
    source: str,
    pair_by: tuple[str, ...] = ('seed',),
    treatment_arm: str | None = None,
    baseline_arm: str | None = None,
    arm_field: str = 'intervention_name',
    env_name: str | None = None,
    extra_filters: Mapping[str, object] = MappingProxyType({}),
    extra_min_pairs: tuple[tuple[str, float], ...] = (),
    extra_max_pairs: tuple[tuple[str, float], ...] = (),
    cell_predicate: Callable[[Mapping[str, object]], bool] | None = None,
    exogenous_keys: frozenset[str] = frozenset(),
    intervention_slot: str | None = None,
) -> PairedGResult:
    """Compute paired Hedges' g + raw mean-diff at `source` across
    matched (T, B) pairs in `cells`.

    Two pairing modes share this entrypoint:

    - **Auto-discover (default)**: when both `treatment_arm` and
      `baseline_arm` are `None`, the analysis identifies the
      contrast structurally — within the filtered scope, find the
      single configurable-key column with exactly two distinct
      values; that column IS the intervention slot. Cells are
      pair-coherent iff they agree on every other configurable
      key + match on `pair_by`. This is the portable shape
      bridges should prefer; the bridge declares only its scope
      (env/HP filters) and the framework discovers the contrast
      from the data.

    - **Legacy arm-name (when `treatment_arm` and `baseline_arm`
      are set)**: pair by string match on `arm_field` (typically
      `intervention_name`). Used by per-corpus bridges authored
      against a single sweep where the substrate stamped the arm
      identity into a string column.

    `source` resolves through the measurable registry (preferred)
    or as a field-path read on the cell record.

    `env_name`, `extra_filters`, `extra_min_pairs`,
    `extra_max_pairs`, and `cell_predicate` scope the corpus
    pre-pairing. `exogenous_keys` (auto-discover only) names
    columns the substrate declared as exogenous metadata
    (`{'env_name', 'seed', 'total_steps', ...}`); these are
    excluded from pair-coherence checks alongside provenance and
    registered measurables."""
    if treatment_arm is None and baseline_arm is None:
        return _paired_g_auto_discover(
            cells, source=source, pair_by=pair_by, env_name=env_name,
            extra_filters=extra_filters,
            extra_min_pairs=extra_min_pairs,
            extra_max_pairs=extra_max_pairs,
            cell_predicate=cell_predicate,
            exogenous_keys=exogenous_keys,
            intervention_slot=intervention_slot,
        )
    if treatment_arm is None or baseline_arm is None:
        raise ValueError(
            'paired_g: pass BOTH treatment_arm and baseline_arm '
            'for the legacy arm-name path, or NEITHER for the '
            'auto-discover path. Half-specified is ambiguous.',
        )

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
