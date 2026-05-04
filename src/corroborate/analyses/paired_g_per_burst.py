"""`paired_g_per_burst` — per-(env, burst) paired Hedges' g panel.

The shape FINDINGS revisions 9, 11, 12 consume: per-cell trace
data has burst-level arrays (`predicted_q_at_start`, `mc_return`,
shape `(n_bursts, n_episodes)`); for each (env, burst_index) we
compute paired-g across seeds on a typed per-burst measurable
(`source: Measurable[Mapping, NDArray]`).

The analysis returns a panel keyed by `(env_name, burst_index)`
with per-(env, burst) Hedges' g + SE + n_pairs. Bridges consume
this panel and assert claims like "DDQN's mechanism operates
early; r(Δbias, Δret) is negative at every burst on FourRooms"
(revision 9).

`source` is a typed Measurable returning a per-burst NDArray
(shape `(n_bursts,)`). Compose via `reduce_axis(from_key('X'),
axis=-1, op='mean')` for "per-burst-mean of column X", or
`reduce_axis(jensen_bias_per_eps, axis=-1, op='mean')` for the
per-burst Jensen-bias mean. The substrate's named per-eps
measurables (e.g. `jensen_bias_per_eps`) compose with the
generic reduction primitives (`reduce_axis`, `slice_axis`) to
parameterise this shape end-to-end.

Bursts are 1-step apart in `eval_step_index` (typically
`eval_every` steps); the analysis doesn't assume any particular
spacing.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from corroborate.analysis import analysis
from corroborate.measurable import Measurable
from corroborate.reductions import from_key, reduce_axis


DEFAULT_PER_BURST_SOURCE: Measurable[
    Mapping[str, object], npt.NDArray[np.floating],
] = reduce_axis(from_key('mc_return'), axis=-1, op='mean')


@dataclass(frozen=True, slots=True)
class PerBurstStratum:
    """One (env, burst) stratum: paired Hedges' g + SE + count.
    `helped_fraction` is the fraction of paired (treatment, baseline)
    seeds where Δ_source > 0 — the rank-free signed-direction
    aggregate that complements the standardised `g`."""
    env_name: str
    burst_index: int
    g: float
    se: float
    n_pairs: int
    helped_fraction: float


@dataclass(frozen=True, slots=True)
class PerBurstResult:
    """Output of `paired_g_per_burst`: panel of per-(env, burst)
    paired-g values plus the input shape parameters for the
    bridge to introspect."""
    strata: tuple[PerBurstStratum, ...]
    measurable: str
    treatment_arm: str
    baseline_arm: str
    pair_by: tuple[str, ...]

    @property
    def n_strata(self) -> int:
        return len(self.strata)


def evaluate_per_burst_source(
    source: Measurable[Mapping[str, object], npt.NDArray[np.floating]],
    cell: Mapping[str, object],
) -> np.ndarray:
    """Per-burst value extraction with cache-first dispatch.

    1. **Cache hit**: if `source.name` is a column on `cell` (the
       per-module cache materialised the composed Measurable as a
       list column at build time), read the pre-computed array.
       Cells flagged None for that column (universal-merge corpora
       lacking traces for some arms) are skipped to an empty array.
    2. **Fallback**: evaluate `source(cell)` from the raw record.
       Used when no cache (synthetic test cells, ad-hoc analyses,
       on-the-fly bridge invocation against runs.parquet).

    The cache-first path is what lets the runner DROP the raw
    trace columns after `_compute_measurables` — once the
    per-burst array is materialised under `source.name`, no
    consumer needs the 2D `(n_bursts, n_episodes)` source again.
    Returns shape `(n_bursts,)` on success or `()` on missing /
    malformed input so downstream filtering naturally excludes
    the cell."""
    cached = cell.get(source.name)
    if cached is not None:
        try:
            arr = np.asarray(cached, dtype=np.float64)
        except (TypeError, ValueError):
            arr = None
        else:
            if arr.ndim == 1:
                return arr
    try:
        arr = np.asarray(source(cell), dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return np.array([], dtype=np.float64)
    if arr.ndim != 1:
        return np.array([], dtype=np.float64)
    return arr


def _key_tuple(
    cell: Mapping[str, object], pair_by: tuple[str, ...],
) -> tuple[object, ...]:
    return tuple(cell[k] for k in pair_by)


@analysis
def paired_g_per_burst(
    cells: Iterable[Mapping[str, object]],
    *,
    treatment_arm: str,
    baseline_arm: str,
    pair_by: tuple[str, ...] = ('seed',),
    source: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ] = DEFAULT_PER_BURST_SOURCE,
    env_name: str | None = None,
    arm_field: str = 'intervention_name',
    dedupe_strategy: str = 'raise',
) -> PerBurstResult:
    """Per-(env, burst) paired Hedges' g panel.

    For each cell, evaluate `source` to get a per-burst vector
    (length `n_bursts`). Group cells by env_name; for each
    (env, burst) pair treatment ↔ baseline cells on `pair_by`
    and compute Hedges' g + SE on the burst-Δs.

    `source` is a typed Measurable returning a per-burst NDArray.
    Default: `reduce_axis(from_key('mc_return'), axis=-1, op='mean')`
    — the per-burst-mean of `mc_return`. Bridges that want a
    different per-burst quantity (e.g. Jensen-bias gap) compose:
    `reduce_axis(jensen_bias_per_eps, axis=-1, op='mean')`. The
    substrate-named per-eps measurable + a generic reduction
    primitive expresses any per-burst reduction without enumerated
    `xxx_mean` strings.

    `env_name`, when supplied, restricts the analysis to one env
    (skips cells with `record['env_name'] != env_name`). When
    None, all envs participate.

    `dedupe_strategy` mirrors `paired_g`: when multiple cells
    share the same `(env, arm, pair_by)` tuple,
    - `'raise'` (default) errors loudly so the bridge author
      tightens scope or opts into aggregation;
    - `'mean'` averages the per-burst vectors element-wise within
      each duplicate bucket. Intended for M2M scopes that
      legitimately combine repeated experiments (multiple
      corpora supplying the same `(env, arm, seed)`)."""
    from corroborate.statistics import hedges_g_paired

    if dedupe_strategy not in ('raise', 'mean'):
        raise ValueError(
            f'paired_g_per_burst: unknown dedupe_strategy '
            f'{dedupe_strategy!r}; expected "raise" or "mean"',
        )

    from corroborate.analyses._dedup_diagnostics import (
        _distinguishing_columns, format_diff,
    )

    # Group cells by (env_name, arm), key on pair_by. Carry the
    # source cell mapping alongside the per-burst array so duplicate
    # detection can introspect which cell-level columns distinguish
    # the duplicates (regime mismatch report — see
    # `_dedup_diagnostics`).
    by_env_arm: dict[tuple[str, str], dict[
        tuple[object, ...], list[tuple[Mapping[str, object], np.ndarray]],
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
        per_burst = evaluate_per_burst_source(source, cell)
        if per_burst.size == 0:
            continue
        bucket = by_env_arm.setdefault((env, arm), {})
        key = _key_tuple(cell, pair_by)
        existing = bucket.setdefault(key, [])
        if existing and dedupe_strategy == 'raise':
            prior_cells = [c for c, _ in existing]
            diff = _distinguishing_columns(
                [*prior_cells, cell],
                skip=frozenset(pair_by) | {'env_name', arm_field},
            )
            if not diff:
                raise ValueError(
                    f'paired_g_per_burst: replicate cells at '
                    f'(env={env!r}, arm={arm!r}, '
                    f'{tuple(pair_by)}={key}) differ only on '
                    f'provenance tags. Pass '
                    f'dedupe_strategy="mean" to aggregate them, or '
                    f'tighten the cache so only one replicate '
                    f'survives.',
                )
            raise ValueError(
                f'paired_g_per_burst: cells at (env={env!r}, '
                f'arm={arm!r}, {tuple(pair_by)}={key}) are not '
                f'replicates — they differ on: {format_diff(diff)}. '
                f'Add the regime-defining column(s) to pair_by so '
                f'each regime is its own stratum, or scope the '
                f'bridge to a single regime. '
                f'`dedupe_strategy="mean"` only fits when the '
                f'duplicates are true replicates.',
            )
        existing.append((cell, per_burst))

    # Collapse list[(cell, ndarray)] → ndarray via element-wise mean.
    # Single-element buckets are unchanged. Multi-element buckets
    # only land here under dedupe_strategy='mean'; if their per-burst
    # shapes don't line up, the cells aren't replicates and we raise
    # with a regime-mismatch report rather than a numpy stack error.
    collapsed: dict[tuple[str, str], dict[
        tuple[object, ...], np.ndarray,
    ]] = {}
    for env_arm, kvs in by_env_arm.items():
        out: dict[tuple[object, ...], np.ndarray] = {}
        for k, items in kvs.items():
            if len(items) == 1:
                out[k] = items[0][1]
                continue
            arrays = [arr for _, arr in items]
            shapes = {arr.shape for arr in arrays}
            if len(shapes) > 1:
                cells_with_dup = [c for c, _ in items]
                diff = _distinguishing_columns(
                    cells_with_dup,
                    skip=frozenset(pair_by) | {'env_name', arm_field},
                )
                raise ValueError(
                    f'paired_g_per_burst: cannot mean-aggregate '
                    f'cells at (env={env_arm[0]!r}, arm={env_arm[1]!r}, '
                    f'{tuple(pair_by)}={k}) — per-burst array shapes '
                    f'differ ({sorted(shapes)}). The cells are not '
                    f'replicates; they differ on: {format_diff(diff)}. '
                    f'Add these to pair_by so each regime is its '
                    f'own stratum.',
                )
            out[k] = np.mean(np.stack(arrays, axis=0), axis=0)
        collapsed[env_arm] = out
    by_env_arm_final = collapsed

    strata: list[PerBurstStratum] = []
    envs = {env for (env, _) in by_env_arm_final.keys()}
    for env in sorted(envs):
        treat = by_env_arm_final.get((env, treatment_arm), {})
        base = by_env_arm_final.get((env, baseline_arm), {})
        paired_keys = sorted(set(treat) & set(base))
        if not paired_keys:
            continue
        # Verify burst-vector lengths match across pairs.
        n_bursts = treat[paired_keys[0]].shape[0]
        for k in paired_keys:
            if (
                treat[k].shape[0] != n_bursts
                or base[k].shape[0] != n_bursts
            ):
                raise ValueError(
                    f'{env}: per-burst vector length mismatch '
                    f'across pairs',
                )
        # For each burst, compute paired g.
        for b in range(n_bursts):
            deltas = [
                float(treat[k][b] - base[k][b]) for k in paired_keys
            ]
            n_pairs = len(deltas)
            g, se = (
                hedges_g_paired(deltas) if n_pairs >= 2
                else (float('nan'), float('nan'))
            )
            helped_fraction = (
                sum(1 for d in deltas if d > 0.0) / n_pairs
                if n_pairs > 0 else float('nan')
            )
            strata.append(PerBurstStratum(
                env_name=env, burst_index=b,
                g=g, se=se, n_pairs=n_pairs,
                helped_fraction=helped_fraction,
            ))

    return PerBurstResult(
        strata=tuple(strata),
        measurable=source.name,
        treatment_arm=treatment_arm,
        baseline_arm=baseline_arm,
        pair_by=pair_by,
    )


def panel_for_env(
    result: PerBurstResult, env_name: str,
) -> tuple[PerBurstStratum, ...]:
    """Convenience: filter strata to one env in burst order."""
    return tuple(
        s for s in result.strata
        if s.env_name == env_name
    )


__all__ = [
    'PerBurstResult', 'PerBurstStratum', 'paired_g_per_burst',
    'evaluate_per_burst_source', 'DEFAULT_PER_BURST_SOURCE',
]
