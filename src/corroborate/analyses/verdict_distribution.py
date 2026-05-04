"""`verdict_distribution_per_env` — per-(env, arm) verdict tally.

Reads a verdict-string column from each cell, groups by env and
arm, returns counts of `HELD` / `INVARIANT_VIOLATION` /
`POWER_INSUFFICIENT` / other. Bridges then assert "this env's
dominant verdict on this column should be X" — the dormancy-
classifier check at the heart of FINDINGS revision 7.

Distinct from `paired_g` family: that family asks "does the
treatment shift a continuous outcome relative to baseline"; this
asks "is the categorical verdict on a (per-cell) invariant
column what we predicted from theory". Both are evidence
generators for typed Bridge verdicts."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from corroborate.bridge.analysis import analysis


@dataclass(frozen=True, slots=True)
class VerdictCounts:
    """Counts of each verdict value for one (env, arm) cell.
    `total` is the total cells observed; `dominant` is whichever
    verdict has strictly the highest count (or `''` on tie /
    empty)."""
    held: int
    invariant_violation: int
    power_insufficient: int
    other: int
    total: int
    dominant: str

    @property
    def held_fraction(self) -> float:
        return self.held / self.total if self.total else float('nan')

    @property
    def violation_fraction(self) -> float:
        if self.total == 0:
            return float('nan')
        return self.invariant_violation / self.total


@dataclass(frozen=True, slots=True)
class VerdictDistributionResult:
    """Output of `verdict_distribution_per_env`. Bridges look up
    by env_name; absent envs return `None` (consumer should map
    that to POWER_INSUFFICIENT)."""
    per_env: Mapping[str, VerdictCounts]

    def for_env(self, env_name: str) -> VerdictCounts | None:
        return self.per_env.get(env_name)


def _dominant(
    held: int, violation: int, power: int, other: int,
) -> str:
    pairs = (
        ('held', held),
        ('invariant_violation', violation),
        ('power_insufficient', power),
        ('other', other),
    )
    sorted_pairs = sorted(pairs, key=lambda p: p[1], reverse=True)
    if (
        len(sorted_pairs) >= 2
        and sorted_pairs[0][1] == sorted_pairs[1][1]
    ):
        return ''
    if sorted_pairs[0][1] == 0:
        return ''
    return sorted_pairs[0][0]


@analysis
def verdict_distribution_per_env(
    cells: Iterable[Mapping[str, object]],
    *,
    arm_filter: str,
    arm_field: str = 'intervention_name',
    verdict_column: str,
) -> VerdictDistributionResult:
    """Per-env tally of `verdict_column` values for cells whose
    `arm_field` equals `arm_filter`. Verdict strings are
    case-folded to lowercase before matching the four canonical
    buckets — the persisted column uses the lowercased
    `Verdict.value` enum strings."""
    per_env_counts: dict[str, list[int]] = {}
    for cell in cells:
        env_v = cell.get('env_name')
        arm_v = cell.get(arm_field)
        if not isinstance(env_v, str) or arm_v != arm_filter:
            continue
        verdict_v = cell.get(verdict_column)
        if not isinstance(verdict_v, str):
            continue
        bucket = per_env_counts.setdefault(env_v, [0, 0, 0, 0])
        v = verdict_v.lower()
        if v == 'held':
            bucket[0] += 1
        elif v == 'invariant_violation':
            bucket[1] += 1
        elif v == 'power_insufficient':
            bucket[2] += 1
        else:
            bucket[3] += 1

    out: dict[str, VerdictCounts] = {}
    for env_name, (h, v, p, o) in per_env_counts.items():
        total = h + v + p + o
        out[env_name] = VerdictCounts(
            held=h, invariant_violation=v,
            power_insufficient=p, other=o,
            total=total,
            dominant=_dominant(h, v, p, o),
        )
    return VerdictDistributionResult(per_env=out)


__all__ = [
    'VerdictCounts', 'VerdictDistributionResult',
    'verdict_distribution_per_env',
]
