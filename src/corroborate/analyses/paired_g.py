"""`paired_g` — paired Hedges' g across seeds (or any pair-key).

The headline analysis shape for the DDQN study and most paired
intervention claims. Pairs treatment cells with baseline cells
on a key tuple (typically `('seed',)`), computes per-pair Δ on a
named outcome path, returns Hedges' g + its SE + the pair count.

Bridges consume `PairedGResult` to assert claims like "DDQN
moves outcome.eval_best_burst_mean by g > 0.3 across seeds on
Acrobot." The threshold and verdict logic lives in the bridge's
`holds_when` body, not in the analysis.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from corroborate.analysis import analysis


@dataclass(frozen=True, slots=True)
class PairedGResult:
    """Output of paired Hedges' g across pair-keys.

    `g` and `se` are NaN if `n_pairs < 2` or per-pair Δ has zero
    spread. `n_treatment` / `n_baseline` are the pre-pair cell
    counts; `n_pairs` is the number of matched pairs (≤ both)."""
    g: float
    se: float
    n_pairs: int
    n_treatment: int
    n_baseline: int
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
        # Survival function of |Z| under standard normal —
        # 2 * Φ(-|z|).
        return math.erfc(z / math.sqrt(2))


def _resolve_path(
    record: Mapping[str, object], path: str,
) -> float:
    """Pull a scalar from a record at a dotted path. Records are
    flat path-keyed dicts (per the framework's persistence
    convention), so this is just a `record[path]` cast to float."""
    v = record.get(path)
    if isinstance(v, (int, float)):
        return float(v)
    raise KeyError(
        f'record is missing scalar at path {path!r}; '
        f'got {type(v).__name__}',
    )


def _key_tuple(
    record: Mapping[str, object], pair_by: tuple[str, ...],
) -> tuple[object, ...]:
    return tuple(record[k] for k in pair_by)


@analysis(name='paired_g')
def paired_g(
    cells: Iterable[Mapping[str, object]],
    *,
    treatment_arm: str,
    baseline_arm: str,
    pair_by: tuple[str, ...] = ('seed',),
    source: str,
    env_name: str | None = None,
    arm_field: str = 'intervention_name',
) -> PairedGResult:
    """Pair `treatment_arm` cells with `baseline_arm` cells on
    `pair_by`, compute per-pair Δ at `source`, return Hedges' g
    + SE.

    `arm_field` is the column that names which arm a cell came
    from (defaults to the framework's `intervention_name`).
    `env_name`, when supplied, restricts both arms to one env
    (cells matching `env_name == record['env_name']`).

    `source` is the outcome path; the bridge's structural
    `source` field maps directly. The function signature
    declares `source` as a kwarg so the framework's resolver
    forwards `bridge.source` to it."""
    from corroborate.statistics import hedges_g_paired

    treatment: dict[tuple[object, ...], float] = {}
    baseline: dict[tuple[object, ...], float] = {}
    for cell in cells:
        if env_name is not None and cell.get('env_name') != env_name:
            continue
        arm = cell.get(arm_field)
        if arm == treatment_arm:
            treatment[_key_tuple(cell, pair_by)] = _resolve_path(
                cell, source,
            )
        elif arm == baseline_arm:
            baseline[_key_tuple(cell, pair_by)] = _resolve_path(
                cell, source,
            )

    paired_keys = sorted(set(treatment) & set(baseline))
    deltas = [
        treatment[k] - baseline[k] for k in paired_keys
    ]
    n_pairs = len(deltas)
    g, se = (
        hedges_g_paired(deltas) if n_pairs >= 2
        else (float('nan'), float('nan'))
    )

    return PairedGResult(
        g=g, se=se,
        n_pairs=n_pairs,
        n_treatment=len(treatment),
        n_baseline=len(baseline),
        pair_by=pair_by,
        measurable=source,
        treatment_arm=treatment_arm,
        baseline_arm=baseline_arm,
    )


__all__ = ['PairedGResult', 'paired_g']
