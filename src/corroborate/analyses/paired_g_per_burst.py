"""`paired_g_per_burst` — per-(env, burst) paired Hedges' g panel.

The shape FINDINGS revisions 9, 11, 12 consume: per-cell trace
data has burst-level arrays (`predicted_q_at_start`, `mc_return`,
shape `(n_bursts, n_episodes)`); for each (env, burst_index) we
compute paired-g across seeds on a burst-mean reduction.

The analysis returns a panel keyed by `(env_name, burst_index)`
with per-(env, burst) Hedges' g + SE + n_pairs. Bridges consume
this panel and assert claims like "DDQN's mechanism operates
early; r(Δbias, Δret) is negative at every burst on FourRooms"
(revision 9).

Bursts are 1-step apart in `eval_step_index` (typically
`eval_every` steps); the analysis doesn't assume any particular
spacing.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from corroborate.analysis import analysis


@dataclass(frozen=True, slots=True)
class PerBurstStratum:
    """One (env, burst) stratum: paired Hedges' g + SE + count."""
    env_name: str
    burst_index: int
    g: float
    se: float
    n_pairs: int


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
    reduction: str  # 'mean' | 'mc_minus_q' (the burst-mean reduction)

    @property
    def n_strata(self) -> int:
        return len(self.strata)


def cell_burst_values(
    cell: Mapping[str, object],
    measurable: str,
    reduction: str,
) -> np.ndarray:
    """Collapse a cell's `(n_bursts, n_episodes)` array to a
    per-burst vector of length `n_bursts`. `reduction='mean'`
    averages over episodes. `reduction='mc_minus_q'` is the
    Jensen-bias proxy: mean(predicted_q_at_start) -
    mean(mc_return) per burst."""
    if reduction == 'mean':
        v = cell.get(measurable)
        if v is None:
            return np.array([], dtype=np.float64)
        arr = np.asarray(v, dtype=np.float64)
        if arr.ndim != 2:
            return np.array([], dtype=np.float64)
        return arr.mean(axis=1)
    if reduction == 'mc_minus_q':
        q = cell.get('predicted_q_at_start')
        m = cell.get('mc_return')
        if q is None or m is None:
            return np.array([], dtype=np.float64)
        q_arr = np.asarray(q, dtype=np.float64)
        m_arr = np.asarray(m, dtype=np.float64)
        if q_arr.ndim != 2 or m_arr.ndim != 2:
            return np.array([], dtype=np.float64)
        return q_arr.mean(axis=1) - m_arr.mean(axis=1)
    raise ValueError(f'unknown reduction {reduction!r}')


def _key_tuple(
    cell: Mapping[str, object], pair_by: tuple[str, ...],
) -> tuple[object, ...]:
    return tuple(cell[k] for k in pair_by)


@analysis(reads=('mc_return', 'predicted_q_at_start'))
def paired_g_per_burst(
    cells: Iterable[Mapping[str, object]],
    *,
    treatment_arm: str,
    baseline_arm: str,
    pair_by: tuple[str, ...] = ('seed',),
    source: str = 'mc_return',
    reduction: str = 'mean',
    env_name: str | None = None,
    arm_field: str = 'intervention_name',
) -> PerBurstResult:
    """Per-(env, burst) paired Hedges' g panel.

    For each cell, project `source` to a burst-mean vector
    (length `n_bursts`). Group cells by env_name; for each
    (env, burst) pair treatment ↔ baseline cells on `pair_by`
    and compute Hedges' g + SE on the burst-Δs.

    `reduction='mean'`: burst-mean of the named measurable.
    `reduction='mc_minus_q'`: Jensen-bias proxy (predicted_q -
    mc_return), per-burst-mean. Custom reductions go in the
    fn — keep the kwarg-driven branching shallow.

    `env_name`, when supplied, restricts the analysis to one env
    (skips cells with `record['env_name'] != env_name`). When
    None, all envs participate."""
    from corroborate.statistics import hedges_g_paired

    # Group cells by (env_name, arm), key on pair_by.
    by_env_arm: dict[tuple[str, str], dict[
        tuple[object, ...], np.ndarray,
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
        per_burst = cell_burst_values(cell, source, reduction)
        if per_burst.size == 0:
            continue
        bucket = by_env_arm.setdefault((env, arm), {})
        bucket[_key_tuple(cell, pair_by)] = per_burst

    strata: list[PerBurstStratum] = []
    envs = {env for (env, _) in by_env_arm.keys()}
    for env in sorted(envs):
        treat = by_env_arm.get((env, treatment_arm), {})
        base = by_env_arm.get((env, baseline_arm), {})
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
            strata.append(PerBurstStratum(
                env_name=env, burst_index=b,
                g=g, se=se, n_pairs=n_pairs,
            ))

    return PerBurstResult(
        strata=tuple(strata),
        measurable=source,
        treatment_arm=treatment_arm,
        baseline_arm=baseline_arm,
        pair_by=pair_by,
        reduction=reduction,
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
    'cell_burst_values',
]
