"""`paired_link_per_burst` — per-(env, burst) link strength panel.

Where `paired_g_per_burst` computes per-burst Δ on a single
measurable (the per-burst effect size), `paired_link_per_burst`
computes per-burst correlation r(Δ_target, Δ_predictor) across
paired seeds. This IS the empirical link from mediator to
outcome, evaluated burst-by-burst.

Why a separate primitive: scalar mech-link analyses (pair the
trajectory-averaged Δ_predictor with trajectory-averaged Δ_target)
silently combine causally opposite phases when training has a
phase transition. The trajectory-averaged scalar slope ≈ 0
because the two phases cancel. Per-burst link unmasks this.

Per-burst is the canonical form for any substrate where the
mediator's dynamics aren't monotone across training (phase-
transition-prone runs). The strata returned (env, burst → r,
p, n_pairs) feed bridges that assert phase-structured claims
("link is active early and reversed late") — the panel makes
the temporal structure typed and corroborable.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from corroborate.analyses.paired_g_per_burst import evaluate_per_burst_source
from corroborate.runner.analysis import analysis
from corroborate.measurables import Measurable


@dataclass(frozen=True, slots=True)
class PerBurstLinkStratum:
    """One (env, burst) stratum: paired link statistics across
    paired seeds. `r` is correlation (link structure); `slope`
    is OLS β of Δ_target on Δ_predictor (conversion efficiency:
    outcome units per unit predictor change); `mean_d_predictor`
    and `mean_d_target` give the per-seed mean Δ on each axis;
    `sd_d_target` is the per-seed Δ_target dispersion (outcome
    noise floor at this burst)."""
    env_name: str
    burst_index: int
    r: float
    p: float
    slope: float
    mean_d_predictor: float
    mean_d_target: float
    sd_d_target: float
    n_pairs: int


@dataclass(frozen=True, slots=True)
class PerBurstLinkResult:
    """Output of `paired_link_per_burst`: panel of per-(env, burst)
    paired link r-values plus the input shape parameters for the
    bridge to introspect."""
    strata: tuple[PerBurstLinkStratum, ...]
    target: str
    predictor: str
    treatment_arm: str
    baseline_arm: str
    pair_by: tuple[str, ...]

    @property
    def n_strata(self) -> int:
        return len(self.strata)


def _pearson_r_p_slope(
    x: np.ndarray, y: np.ndarray,
) -> tuple[float, float, float]:
    """Pearson r + two-sided p-value (Fisher z) + OLS slope β of
    y on x. Returns (nan, nan, nan) if insufficient variance or
    n<3."""
    n = len(x)
    if n < 3 or x.std() == 0 or y.std() == 0:
        return float('nan'), float('nan'), float('nan')
    r = float(np.corrcoef(x, y)[0, 1])
    if not np.isfinite(r):
        return float('nan'), float('nan'), float('nan')
    slope = float(r * y.std(ddof=1) / x.std(ddof=1))
    if abs(r) >= 1.0:
        return r, 0.0, slope
    z = 0.5 * np.log((1 + r) / (1 - r))
    se = 1.0 / np.sqrt(n - 3)
    from scipy.stats import norm
    p = float(2 * (1 - norm.cdf(abs(z) / se)))
    return r, p, slope


@analysis
def paired_link_per_burst(
    cells: Iterable[Mapping[str, object]],
    *,
    treatment_arm: str,
    baseline_arm: str,
    target: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ],
    predictor: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ],
    pair_by: tuple[str, ...] = ('seed',),
    env_name: str | None = None,
    arm_field: str = 'intervention_name',
    dedupe_strategy: str = 'raise',
) -> PerBurstLinkResult:
    """Per-(env, burst) paired link r(Δ_target, Δ_predictor) panel.

    For each cell, evaluate `target` and `predictor` to per-burst
    vectors. For each (env, burst), pair treatment ↔ baseline on
    `pair_by`, compute Δ_target and Δ_predictor across seeds, then
    Pearson r between them.

    Both `target` and `predictor` are typed Measurables returning
    per-burst NDArrays. The canonical mediator → outcome link
    composes a per-burst-mean of the outcome key against a per-
    burst-mean of a substrate-named mediator-residual measurable:

        target = reduce_axis(from_key('<outcome_key>'), axis=-1, op='mean')
        predictor = reduce_axis(<mediator_per_eps>, axis=-1, op='mean')

    `r` is computed against the *negated* predictor so the value
    reads "active link = positive r" — the convention is "more
    mediator reduction → more outcome gain." Positive r at burst
    b means: at this training stage, the predicted causal arrow
    holds. Negative r means the relationship has flipped
    (substrate-specific phase-transition).

    `env_name`, when supplied, restricts the analysis to one env.

    `dedupe_strategy` mirrors `paired_g`: when multiple cells
    share the same `(env, arm, pair_by)` tuple,
    - `'raise'` (default) errors loudly so the bridge author
      tightens scope or opts into aggregation;
    - `'mean'` averages the per-burst (target, predictor) vectors
      element-wise within each duplicate bucket."""
    if dedupe_strategy not in ('raise', 'mean'):
        raise ValueError(
            f'paired_link_per_burst: unknown dedupe_strategy '
            f'{dedupe_strategy!r}; expected "raise" or "mean"',
        )

    from corroborate.analyses._dedup_diagnostics import (
        _distinguishing_columns, format_diff,
    )

    # Carry the cell mapping with each (target, predictor) pair so
    # duplicate detection can introspect distinguishing columns.
    by_env_arm: dict[tuple[str, str], dict[
        tuple[object, ...],
        list[tuple[Mapping[str, object], np.ndarray, np.ndarray]],
    ]] = {}
    for cell in cells:
        env = cell.get('env_name')
        arm = cell.get(arm_field)
        if not isinstance(env, str) or not isinstance(arm, str):
            continue
        if env_name is not None and env != env_name:
            continue
        if arm not in (treatment_arm, baseline_arm):
            continue
        target_v = evaluate_per_burst_source(target, cell)
        predictor_v = evaluate_per_burst_source(predictor, cell)
        if target_v.size == 0 or predictor_v.size == 0:
            continue
        if target_v.shape[0] != predictor_v.shape[0]:
            continue
        bucket = by_env_arm.setdefault((env, arm), {})
        key = tuple(cell[k] for k in pair_by)
        existing = bucket.setdefault(key, [])
        if existing and dedupe_strategy == 'raise':
            prior_cells = [c for c, _, _ in existing]
            diff = _distinguishing_columns(
                [*prior_cells, cell],
                skip=frozenset(pair_by) | {'env_name', arm_field},
            )
            if not diff:
                raise ValueError(
                    f'paired_link_per_burst: replicate cells at '
                    f'(env={env!r}, arm={arm!r}, '
                    f'{tuple(pair_by)}={key}) differ only on '
                    f'provenance tags. Pass '
                    f'dedupe_strategy="mean" to aggregate them.',
                )
            raise ValueError(
                f'paired_link_per_burst: cells at (env={env!r}, '
                f'arm={arm!r}, {tuple(pair_by)}={key}) are not '
                f'replicates — they differ on: {format_diff(diff)}. '
                f'Add the regime-defining column(s) to pair_by so '
                f'each regime is its own stratum, or scope the '
                f'bridge to a single regime.',
            )
        existing.append((cell, target_v, predictor_v))

    # Collapse via element-wise mean. Shape mismatch → not
    # replicates; raise with the regime-mismatch report.
    collapsed: dict[tuple[str, str], dict[
        tuple[object, ...], tuple[np.ndarray, np.ndarray],
    ]] = {}
    for env_arm, kvs in by_env_arm.items():
        out: dict[tuple[object, ...], tuple[np.ndarray, np.ndarray]] = {}
        for k, items in kvs.items():
            if len(items) == 1:
                _, ts0, ps0 = items[0]
                out[k] = (ts0, ps0)
                continue
            target_arrays = [t for _, t, _ in items]
            predictor_arrays = [p for _, _, p in items]
            shapes = {t.shape for t in target_arrays} | {
                p.shape for p in predictor_arrays
            }
            if len({t.shape for t in target_arrays}) > 1 or len(
                {p.shape for p in predictor_arrays},
            ) > 1:
                cells_with_dup = [c for c, _, _ in items]
                diff = _distinguishing_columns(
                    cells_with_dup,
                    skip=frozenset(pair_by) | {'env_name', arm_field},
                )
                raise ValueError(
                    f'paired_link_per_burst: cannot mean-aggregate '
                    f'cells at (env={env_arm[0]!r}, '
                    f'arm={env_arm[1]!r}, {tuple(pair_by)}={k}) — '
                    f'per-burst array shapes differ ({sorted(shapes)}). '
                    f'The cells are not replicates; they differ on: '
                    f'{format_diff(diff)}. Add these to pair_by.',
                )
            ts = np.mean(np.stack(target_arrays, axis=0), axis=0)
            ps = np.mean(np.stack(predictor_arrays, axis=0), axis=0)
            out[k] = (ts, ps)
        collapsed[env_arm] = out
    by_env_arm_final = collapsed

    strata: list[PerBurstLinkStratum] = []
    envs = {env for (env, _) in by_env_arm_final.keys()}
    for env in sorted(envs):
        treat = by_env_arm_final.get((env, treatment_arm), {})
        base = by_env_arm_final.get((env, baseline_arm), {})
        paired_keys = sorted(set(treat) & set(base))
        if not paired_keys:
            continue
        # Per-key arm-shape match — the real invariant. Cross-key
        # uniformity is NOT required: multi-regime corpora can
        # legitimately have different burst counts at different
        # `pair_by` keys (e.g. cells from total_steps=200k vs 1M).
        for k in paired_keys:
            if (
                treat[k][0].shape[0] != base[k][0].shape[0]
                or treat[k][1].shape[0] != base[k][1].shape[0]
                or treat[k][0].shape[0] != treat[k][1].shape[0]
            ):
                raise ValueError(
                    f'{env}: arm/operand shape mismatch at pair_by '
                    f'key {k}',
                )

        # Walk the union of burst indices; each burst pools only
        # the keys whose arrays extend that far.
        max_bursts = max(treat[k][0].shape[0] for k in paired_keys)
        for b in range(max_bursts):
            contributors = [
                k for k in paired_keys if treat[k][0].shape[0] > b
            ]
            d_target = np.array([
                treat[k][0][b] - base[k][0][b] for k in contributors
            ], dtype=np.float64)
            d_predictor = np.array([
                treat[k][1][b] - base[k][1][b] for k in contributors
            ], dtype=np.float64)
            finite = np.isfinite(d_target) & np.isfinite(d_predictor)
            d_t = d_target[finite]
            d_p = d_predictor[finite]
            # Convention: link strength is positive when active.
            # The natural primitive is "more mediator reduction →
            # more outcome gain": more reduction (more *negative*
            # Δ_predictor) correlates with more positive Δ_target.
            # We negate Δ_predictor so the reported r and slope
            # read positive-when-active.
            r, p, slope = _pearson_r_p_slope(-d_p, d_t)
            mean_p = float(d_p.mean()) if d_p.size else float('nan')
            mean_t = float(d_t.mean()) if d_t.size else float('nan')
            sd_t = float(d_t.std(ddof=1)) if d_t.size > 1 else float('nan')
            strata.append(PerBurstLinkStratum(
                env_name=env, burst_index=b,
                r=r, p=p, slope=slope,
                mean_d_predictor=mean_p,
                mean_d_target=mean_t,
                sd_d_target=sd_t,
                n_pairs=int(finite.sum()),
            ))

    return PerBurstLinkResult(
        strata=tuple(strata),
        target=target.name,
        predictor=predictor.name,
        treatment_arm=treatment_arm,
        baseline_arm=baseline_arm,
        pair_by=pair_by,
    )


def phase_link_consistency(
    result: PerBurstLinkResult,
    *,
    env_name: str | None = None,
    significance: float = 0.05,
    expected_sign: int = +1,
) -> float:
    """Scalar derived from a `PerBurstLinkResult`: proportion of
    bursts where the link r matches `expected_sign` AND p <
    `significance`. High = link is uniformly active across
    training; low = phase-dependent (Phase 1 holds, Phase 2 doesn't,
    etc.).

    Default `expected_sign=+1` matches the bias-correction story
    under the panel's positive-when-active convention: r > 0 means
    more bias reduction → more outcome gain.

    Returns nan if the panel has no strata for the env."""
    panel = (
        result.strata if env_name is None
        else tuple(s for s in result.strata if s.env_name == env_name)
    )
    if not panel:
        return float('nan')
    sign_match = sum(
        1 for s in panel
        if (
            np.isfinite(s.r) and np.isfinite(s.p)
            and s.p < significance
            and (s.r * expected_sign) > 0
        )
    )
    return sign_match / len(panel)


__all__ = [
    'PerBurstLinkResult',
    'PerBurstLinkStratum',
    'paired_link_per_burst',
    'phase_link_consistency',
]
