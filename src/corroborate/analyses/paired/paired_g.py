"""`paired_g` — paired Hedges' g + raw-diff across pair keys.

The headline analysis shape for paired intervention claims.
Pairs treatment rows with baseline rows on a key tuple
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
field-path read on the row record. This is the convergence
point for the framework's "(1) claim outputs + (2) post-run
measurables" architecture: ALL per-row quantities (raw or
derived) become resolvable by name through one resolver,
and analyses are generic over what they consume.

Row-set scoping (env, HP equality, threshold gates, arbitrary
predicates) lives upstream on `Bridge.scope` as a polars
`pl.Expr`; `claim_bridge.evaluate()` filters before this analysis
sees the df. Analyses pair, never scope.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import polars as pl

from corroborate._internals.polars import to_dicts
from corroborate.analyses._row_value import key_tuple, resolve_value
from corroborate.analyses.panel import per_stratum_panel
from corroborate.bridge.analysis import analysis
from corroborate.corpus.schema import StratumG


@dataclass(frozen=True, slots=True)
class PairedGResult:
    """Output of paired Hedges' g across pair-keys.

    `g` and `se` are the standardized effect size + its SE.
    `mean_diff` and `mean_diff_se` are the raw paired mean
    difference + its standard error — the same input rows, but
    NOT pooled-SD-scaled. Bridges that test interventions on the
    reward magnitude itself must consume `mean_diff` (Hedges' g
    cancels reward-scale variance via the pooled SD).

    `helped_fraction` is the fraction of pairs with positive Δ
    (treatment > baseline) — the count-style report a number of
    bridges want alongside the standardized magnitude. NaN when
    `n_pairs == 0`.

    All other quantities are NaN if `n_pairs < 2` or per-pair Δ
    has zero spread.

    `assumption_violations` carries one entry per detected
    distributional concern (skew, heavy tails, etc.). Empty when
    the input passes all calibrated checks; populated from
    heuristics whose thresholds are derived from the empirical
    bias map at `tests/analytic/robustness/test_paired_g_skew_robustness.py`.
    Each string carries the diagnostic + the predicted bias
    magnitude so a reader can decide whether it matters at their
    effect-size scale. Bridges should propagate these into the
    BridgeReportEntry audit trail."""
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
    assumption_violations: tuple[str, ...] = ()

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




@analysis
def paired_g(
    df: pl.DataFrame,
    *,
    source: str,
    treatment_arm: str,
    baseline_arm: str,
    pair_by: tuple[str, ...] = ('seed',),
    arm_field: str = 'arm_key',
    dedupe_strategy: str = 'mean',
) -> PairedGResult:
    """Compute paired Hedges' g + raw mean-diff at `source` across
    matched (T, B) pairs in `rows`.

    Pairing is by string match on `arm_field` (defaults to
    `'arm_key'`; the `canonical_str` fingerprint the substrate's
    row runner stamps on each row). `treatment_arm` and
    `baseline_arm` come from the bridge's `DoEffect` — either via
    per-bridge `source = DoEffect(...)` decorator override or via
    file-level `INTERVENTION = DoEffect(...)` resolved at
    decoration time. `claim_bridge.evaluate()` derives the
    canonical arm strings via `DoEffect.arm_keys()` (treatment =
    `arm_keys()[1]`, baseline = `arm_keys()[0]` in the binary
    case) and forwards them into this analysis's kwargs.

    `source` resolves through the measurable registry (preferred)
    or as a field-path read on the row record.

    Row-level scoping (env, HP equality, threshold gates,
    arbitrary predicates) lives on `Bridge.scope` as a polars
    `pl.Expr`; `claim_bridge.evaluate()` filters before this
    analysis sees the df. The analysis itself does not scope.

    `dedupe_strategy` controls the policy when multiple rows share
    the same `(arm, pair_by)` tuple:
    - `'mean'` (default): average the per-row `source` values
      within each `(arm, pair_by)` bucket, then run paired-g on
      the aggregated values. Suits M2M scopes that legitimately
      pool across multiple corpora at the same `(seed, env)`. The
      aggregated value is the per-row mean, NOT the underlying
      raw rows; downstream SE / df reflect the post-aggregation
      n_pairs.

      **WARNING:** when the duplicate rows differ on regime-
      defining fields (γ, total_steps, sync_period, …) NOT in
      `pair_by`, `'mean'` silently averages causally distinct
      experiments. Pass `'raise'` to detect this — it reports
      which columns differ between the duplicates so the author
      can tighten `pair_by`.

    - `'raise'`: error loudly on any `(arm, pair_by)` bucket with
      more than one row. Use when the corpus shouldn't have
      duplicates and silent averaging would hide a regime split —
      e.g. a sweep that meant to produce one row per
      `(env, seed, sync_period)` but `pair_by=('seed',)` collapses
      sync regimes into one bucket. The error message names the
      duplicate key so the author can either tighten `pair_by` or
      explicitly opt into the `'mean'` aggregation.

    **Robustness.** Hedges' c_4 small-sample correction is exact
    only under NORMAL Δ. Under skewed/heavy-tailed Δ:
      - `g` is OVERESTIMATED (skew/heavy-tail inflates mean more
        than sd in the small-n regime).
      - `se` is ANTI-CONSERVATIVE at large n (Pearson formula
        misses heavy-tail contribution to g's sampling SD).
    Empirical bias map (`tests/analytic/robustness/test_paired_g_skew_robustness.py`):
      - log-normal Δ (skew ≈ 1.86): bias = +0.33 at n=10,
        +0.13 at n=30, +0.07 at n=100.
      - t(df=5) Δ: bias = +0.18 at n=10, +0.06 at n=30.
    Substrate-author guidance: pair with `bootstrap_paired_g` or
    `cliff_delta_paired` (when available) on small-n skewed-Δ
    corpora. `result.assumption_violations` flags inputs that
    cross the empirically-derived skew/kurtosis thresholds."""
    return _paired_g_rows(
        to_dicts(df), source=source, treatment_arm=treatment_arm,
        baseline_arm=baseline_arm, pair_by=pair_by,
        arm_field=arm_field, dedupe_strategy=dedupe_strategy,
    )


def _require_complete_pair_key(
    key: tuple[object, ...],
    pair_by: tuple[str, ...],
    row: Mapping[str, object],
) -> None:
    """A null pairing-unit key silently merges unrelated rows into
    one `(None, ...)` bucket — after frame materialisation, rows
    that never carried the key are indistinguishable from rows
    carrying an explicit null, so the loud check lives here."""
    if any(v is None for v in key):
        raise ValueError(
            f'paired_g: pairing key {pair_by} is null on row '
            f'id={row.get("id")!r} — a null unit key would merge '
            f'unrelated rows into one bucket; scope the claim to '
            f'rows carrying the key, or repair the record',
        )


def _paired_g_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    source: str,
    treatment_arm: str,
    baseline_arm: str,
    pair_by: tuple[str, ...],
    arm_field: str,
    dedupe_strategy: str,
) -> PairedGResult:
    """Rows-level core of `paired_g` — in-tree per-stratum
    delegates (`per_env_paired_g_panel`) call this directly so a
    per-env subset never round-trips through a frame; the public
    `@analysis` wraps it behind the plain-DataFrame boundary."""
    from corroborate.stats import hedges_g_paired

    if dedupe_strategy not in ('raise', 'mean'):
        raise ValueError(
            f'paired_g: unknown dedupe_strategy {dedupe_strategy!r}; '
            f'expected "raise" or "mean"',
        )

    # Collect rows alongside values so duplicate-bucket inspection
    # can call `distinguishing_columns` on the actual rows (not
    # just values). Under `dedupe_strategy='raise'`, a duplicate
    # bucket whose rows differ ONLY on framework-provenance tags
    # / None-vs-explicit-default columns is a TRUE replicate — the
    # raise would be a false positive. Defer the check to the
    # post-collection pass below; pre-fix this fired on the bucket-
    # count alone, falsely flagging cross-sub-sweep replicate
    # aliases (cf. `findings_dqn_bridges_regime_mixing.md`).
    from corroborate.analyses._dedup_diagnostics import (
        distinguishing_columns, format_diff,
    )
    treatment_buckets: dict[
        tuple[object, ...], list[tuple[Mapping[str, object], float]],
    ] = {}
    baseline_buckets: dict[
        tuple[object, ...], list[tuple[Mapping[str, object], float]],
    ] = {}
    for row in rows:
        arm = row.get(arm_field)
        if arm == treatment_arm:
            key = key_tuple(row, pair_by)
            _require_complete_pair_key(key, pair_by, row)
            treatment_buckets.setdefault(key, []).append(
                (row, resolve_value(row, source)),
            )
        elif arm == baseline_arm:
            key = key_tuple(row, pair_by)
            _require_complete_pair_key(key, pair_by, row)
            baseline_buckets.setdefault(key, []).append(
                (row, resolve_value(row, source)),
            )

    def _check_or_raise(
        arm: str,
        buckets: dict[
            tuple[object, ...], list[tuple[Mapping[str, object], float]],
        ],
    ) -> None:
        if dedupe_strategy != 'raise':
            return
        for key, items in buckets.items():
            if len(items) <= 1:
                continue
            rows_in_bucket = [c for c, _ in items]
            diff = distinguishing_columns(
                rows_in_bucket,
                skip=frozenset(pair_by) | {arm_field},
            )
            if not diff:
                # True replicates (only provenance / None-default
                # drift) — silently fall through to mean below.
                continue
            raise ValueError(
                f'paired_g: duplicate rows for {arm!r} at '
                f'pair_by={pair_by} key={key} are not replicates — '
                f'they differ on: {format_diff(diff)}. Tighten '
                f'`pair_by` to a discriminating tuple, scope the '
                f'bridge to a single regime, or pass '
                f'`dedupe_strategy="mean"` to mean-aggregate.',
            )

    _check_or_raise(treatment_arm, treatment_buckets)
    _check_or_raise(baseline_arm, baseline_buckets)

    # Mean-aggregate (which is a no-op for size-1 buckets and the
    # right thing for true-replicate buckets that survived the
    # `_check_or_raise` pass).
    treatment: dict[tuple[object, ...], float] = {
        k: (
            sum(v for _, v in items if not math.isnan(v))
            / max(1, sum(1 for _, v in items if not math.isnan(v)))
        ) if any(not math.isnan(v) for _, v in items) else float('nan')
        for k, items in treatment_buckets.items()
    }
    baseline: dict[tuple[object, ...], float] = {
        k: (
            sum(v for _, v in items if not math.isnan(v))
            / max(1, sum(1 for _, v in items if not math.isnan(v)))
        ) if any(not math.isnan(v) for _, v in items) else float('nan')
        for k, items in baseline_buckets.items()
    }

    paired_keys = sorted(set(treatment) & set(baseline))
    # NaN-skip pairs where either side is missing (e.g. rows from
    # corpora that didn't carry the source column, surfacing as
    # NaN through `resolve_value`'s present-but-None path). The
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
        assumption_violations=_paired_g_assumption_violations(deltas),
    )


def _paired_g_assumption_violations(
    deltas: list[float],
) -> tuple[str, ...]:
    """Heuristic distributional checks on the per-pair Δ.

    Thresholds are calibrated from the empirical probe at
    `tests/analytic/robustness/test_paired_g_skew_robustness.py`:
      - At |skew(Δ)| ≈ 1.86 (log-normal σ_log=0.7), n=30 bias
        is +12% relative. `|skew| > 1.0` flags the regime where
        bias exceeds ~5%.
      - At kurtosis(Δ) > 5.0 (heavy tails), the framework's
        Pearson-based SE is anti-conservative; CIs under-cover.
      - At n < 10, even normal Δ has wide MC sampling SD
        (CV ≈ 70%); estimate is unreliable for inference.

    Each violation string includes the measured value AND the
    predicted bias magnitude so the reader can decide whether
    it matters at their effect-size scale.
    """
    violations: list[str] = []
    n = len(deltas)
    if n < 2:
        return ()
    if n < 10:
        violations.append(
            f'small_n_unreliable (n_pairs={n}, MC_CV~70% even on normal Δ)'
        )
    if n < 4:
        # Sample skew/kurtosis are undefined / unstable below n=4.
        return tuple(violations)
    import numpy as _np
    arr = _np.asarray(deltas, dtype=_np.float64)
    mean = float(arr.mean())
    centered = arr - mean
    m2 = float((centered ** 2).mean())
    if m2 == 0.0:
        return tuple(violations)
    m3 = float((centered ** 3).mean())
    m4 = float((centered ** 4).mean())
    # Adjusted Fisher-Pearson sample skewness with bias correction (G_1).
    g1 = m3 / (m2 ** 1.5)
    skew_d = math.sqrt(n * (n - 1)) / (n - 2) * g1
    # Bias-corrected sample excess kurtosis (G_2).
    g2 = m4 / (m2 ** 2) - 3.0
    kurt_d = ((n - 1) / ((n - 2) * (n - 3))) * ((n + 1) * g2 + 6.0)
    if abs(skew_d) > 1.0:
        # Predicted relative inflation at this skew + n; rough
        # interpolation of the probe's empirical map.
        approx_pct = round(min(30.0, abs(skew_d) * 12.0 * 30.0 / max(n, 30)))
        violations.append(
            f'skew_bias_likely (skew={skew_d:+.2f}, predicted '
            f'paired_g inflation ~{approx_pct}% at n={n})'
        )
    if kurt_d > 5.0:
        violations.append(
            f'heavy_tail_se_anti_conservative '
            f'(excess_kurtosis={kurt_d:.2f}, framework_se will '
            f'under-cover by ~10-25% at this kurtosis)'
        )
    return tuple(violations)


# ============ per-env panel helper ============

def per_env_paired_g_panel(
    rows: Sequence[Mapping[str, object]],
    *,
    treatment_arm: str,
    baseline_arm: str,
    source: str,
    env_filter: tuple[str, ...] = (),
    pair_by: tuple[str, ...] = ('seed',),
    arm_field: str = 'arm_key',
    dedupe_strategy: str = 'mean',
) -> tuple[StratumG[str], ...]:
    """Per-env paired-g panel — one `StratumG[str]` per env in
    `env_filter` (or every env present in `rows` when empty).

    Wraps the generic `per_stratum_panel`: stratifies rows by
    `env_name`, calls `paired_g.fn` on each env's subset, packs
    the result as `StratumG[str]`. Consumers (e.g.
    `meta_regression_paired_g`) iterate the panel rather than
    re-pairing.

    NO panel-level filtering: every env in the target set produces
    an entry, including degenerate ones (n_pairs<2 → g/se=NaN).
    Consumers that need to drop underpowered strata filter at
    their own boundary so they can decide what to report."""
    def _stratify(row: Mapping[str, object]) -> str | None:
        env = row.get('env_name')
        return env if isinstance(env, str) else None

    key_filter: Callable[[str], bool] | None = (
        (lambda env: env in env_filter) if env_filter else None
    )

    def _analyze(subset: Sequence[Mapping[str, object]]) -> PairedGResult:
        # Re-entry at a stratum subset: straight into the rows
        # core — a per-env subset never round-trips through a
        # frame just to be decoded again.
        return _paired_g_rows(
            subset,
            treatment_arm=treatment_arm,
            baseline_arm=baseline_arm,
            pair_by=pair_by,
            source=source,
            arm_field=arm_field,
            dedupe_strategy=dedupe_strategy,
        )

    panel = per_stratum_panel(
        rows, stratify_by=_stratify, analysis=_analyze,
        key_filter=key_filter,
    )
    return tuple(
        StratumG[str](
            stratum_id=env, g=r.g, se=r.se, n_pairs=r.n_pairs,
        )
        for env, r in panel
    )


__all__ = ['PairedGResult', 'paired_g', 'per_env_paired_g_panel']
