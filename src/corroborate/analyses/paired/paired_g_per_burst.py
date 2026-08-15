"""`paired_g_per_burst` — per-(env, burst) paired Hedges' g panel.

Per-cell trace data has burst-level arrays of shape
`(n_bursts, n_episodes)`; for each (env, burst_index) we compute
paired-g across seeds on a typed per-burst measurable
(`source: Measurable[Mapping, NDArray]`). The analysis returns
a panel keyed by `(env_name, burst_index)` with per-(env, burst)
Hedges' g + SE + n_pairs. Bridges consume this panel and assert
phase-structured claims about when in training the treatment
effect activates.

`source` is a typed Measurable returning a per-burst NDArray
(shape `(n_bursts,)`). Compose via `reduce_axis(from_key('<key>'),
axis=-1, op='mean')` for "per-burst-mean of column <key>", or
e.g. `reduce_axis(<substrate-named-measurable>, axis=-1,
op='mean')` for a per-burst-mean of a substrate-defined per-
episode measurable. Substrate-side measurables compose with the
generic reduction primitives (`reduce_axis`, `slice_axis`) to
parameterise this shape end-to-end.

Bursts are 1-step apart along the substrate's burst axis
(typically every `eval_every` training steps); the analysis
doesn't assume any particular spacing.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from corroborate.analyses._cell_value import (
    evaluate_per_burst_source, key_tuple,
)
from corroborate.bridge.analysis import analysis
from corroborate.measurables import Measurable
from corroborate.measurables.reductions import from_key, reduce_axis


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
    arm_field: str = 'arm_key',
    dedupe_strategy: str = 'mean',
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
    - `'mean'` (default) averages the per-burst vectors
      element-wise within each duplicate bucket. Suits M2M scopes
      that combine repeated experiments. **WARNING**: when the
      duplicate cells differ on regime-defining fields not in
      `pair_by`, `'mean'` silently averages causally distinct
      experiments — pass `'raise'` to detect this.
    - `'raise'` errors loudly on duplicates AND reports which
      columns differ between them so the bridge author tightens
      scope (extend `pair_by`) or explicitly opts into `'mean'`."""
    from corroborate.stats import hedges_g_paired

    if dedupe_strategy not in ('raise', 'mean'):
        raise ValueError(
            f'paired_g_per_burst: unknown dedupe_strategy '
            f'{dedupe_strategy!r}; expected "raise" or "mean"',
        )

    from corroborate.analyses._dedup_diagnostics import (
        distinguishing_columns, format_diff,
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
        key = key_tuple(cell, pair_by)
        existing = bucket.setdefault(key, [])
        if existing and dedupe_strategy == 'raise':
            prior_cells = [c for c, _ in existing]
            diff = distinguishing_columns(
                [*prior_cells, cell],
                skip=frozenset(pair_by) | {'env_name', arm_field},
            )
            if diff:
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
            # Empty diff → true replicates (only provenance / None-
            # default / derived-measurable drift). Fall through to
            # mean-aggregation below; matches `paired_g.fn`'s
            # smarter `'raise'` semantics.
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
                diff = distinguishing_columns(
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
        # Per-key arm-shape match — the real invariant. The same
        # configuration (same `pair_by` tuple) must produce the
        # same per-burst length across both arms; otherwise some
        # data integrity is broken upstream.
        for k in paired_keys:
            if treat[k].shape[0] != base[k].shape[0]:
                raise ValueError(
                    f'{env}: arm shape mismatch at pair_by key '
                    f'{k} (treat={treat[k].shape}, '
                    f'base={base[k].shape})',
                )
        # Walk the union of burst indices. At each burst, only
        # keys whose arrays extend that far contribute. Multi-
        # regime corpora (e.g. cells from total_steps=200k AND
        # 1M sharing one (env, arm) bucket) are accommodated:
        # the 200k keys contribute to bursts 0..9, the 1M keys
        # contribute to bursts 0..49. `n_pairs` per stratum
        # naturally reflects the regime overlap; downstream
        # bridges that need uniform power can post-filter.
        max_bursts = max(treat[k].shape[0] for k in paired_keys)
        for b in range(max_bursts):
            contributors = [
                k for k in paired_keys if treat[k].shape[0] > b
            ]
            deltas = [
                float(treat[k][b] - base[k][b]) for k in contributors
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
