"""`factorial_2x2_interaction` — 2×2 factorial design analysis
producing the four within-pair Hedges' g's plus the interaction
contrast.

A 2×2 factorial across two binary axes on a panel of envs. For
each env, four arms paired by `pair_by` produce four Hedges' g's:

  - (B−A): treatment-2 effect at baseline-1
  - (D−C): treatment-2 effect at treatment-1
  - (C−A): treatment-1 effect at baseline-2
  - (D−B): treatment-1 effect at treatment-2
  - INT  : (D−B) − (C−A) — interaction contrast on the
           treatment-1 axis. Negative INT discriminates two
           readings: (a) over-correction (treatment-1 +
           treatment-2 compounds harm) OR (b) attenuation
           (treatment-1's marginal benefit shrinks where
           treatment-2 already covers part of the same axis).

The discriminator between (a) and (b) requires inspecting the
(B−A) and (D−C) signs/magnitudes alongside the interaction —
that's the bridge body's job, not the analysis."""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from corroborate.analysis import analysis
from corroborate.stats import hedges_g_paired


@dataclass(frozen=True, slots=True)
class FactorialPerEnv:
    """One env's 2×2 factorial decomposition. Paired Hedges' g's
    are computed across the (env, pair_by) keys present in ALL
    four arms — paired drops if any arm is missing the seed.
    `n_pairs` is the count of fully-paired keys."""
    env_name: str
    g_b_minus_a: float
    se_b_minus_a: float
    g_d_minus_c: float
    se_d_minus_c: float
    g_c_minus_a: float
    se_c_minus_a: float
    g_d_minus_b: float
    se_d_minus_b: float
    g_interaction: float
    se_interaction: float
    n_pairs: int


@dataclass(frozen=True, slots=True)
class Factorial2x2Result:
    """Output of `factorial_2x2_interaction`. `per_env` is the
    per-env decomposition table; consumers look up by env_name.
    `arm_a/b/c/d` and `source` are echoed for audit clarity."""
    per_env: tuple[FactorialPerEnv, ...]
    arm_a: str
    arm_b: str
    arm_c: str
    arm_d: str
    source: str

    def for_env(self, env_name: str) -> FactorialPerEnv | None:
        for p in self.per_env:
            if p.env_name == env_name:
                return p
        return None


def _g_paired_from_two_arms(
    arm_x: Mapping[tuple[object, ...], float],
    arm_y: Mapping[tuple[object, ...], float],
    keys: Iterable[tuple[object, ...]],
) -> tuple[float, float]:
    """g(Y − X) on the keys' intersection (assumes `keys` are
    already in both arms). NaN/NaN when n_pairs < 2."""
    deltas = [arm_y[k] - arm_x[k] for k in keys]
    if len(deltas) < 2:
        return float('nan'), float('nan')
    return hedges_g_paired(deltas)


@analysis
def factorial_2x2_interaction(
    cells: Iterable[Mapping[str, object]],
    *,
    arm_a: str,
    arm_b: str,
    arm_c: str,
    arm_d: str,
    source: str,
    pair_by: tuple[str, ...] = ('seed',),
    arm_field: str = 'intervention_name',
    env_filter: tuple[str, ...] = (),
    total_steps_filter: int | None = None,
    total_steps_field: str = 'total_steps',
) -> Factorial2x2Result:
    """For each env, compute the (B−A, D−C, C−A, D−B, INT)
    paired Hedges' g table on `source` across the 4 arms.

    Pairs are formed on the intersection of `pair_by` keys
    present in all 4 arms — a seed missing from any one arm
    drops the pair entirely (the factorial design needs all 4
    cells for that seed)."""
    cells_list = list(cells)
    if total_steps_filter is not None:
        cells_list = [
            c for c in cells_list
            if c.get(total_steps_field) == total_steps_filter
        ]

    by_env_arm: dict[str, dict[str, dict[tuple[object, ...], float]]] = {}
    envs_seen: set[str] = set()
    for cell in cells_list:
        env_v = cell.get('env_name')
        arm_v = cell.get(arm_field)
        if not isinstance(env_v, str):
            continue
        envs_seen.add(env_v)
        if arm_v not in (arm_a, arm_b, arm_c, arm_d):
            continue
        v = cell.get(source)
        if not isinstance(v, (int, float)):
            continue
        fv = float(v)
        if math.isnan(fv):
            continue
        key = tuple(cell[k] for k in pair_by)
        env_arms = by_env_arm.setdefault(env_v, {})
        env_arm = env_arms.setdefault(str(arm_v), {})
        env_arm[key] = fv

    target_envs: tuple[str, ...]
    if env_filter:
        target_envs = tuple(e for e in env_filter if e in envs_seen)
    else:
        target_envs = tuple(sorted(envs_seen))

    out: list[FactorialPerEnv] = []
    for env in target_envs:
        arms = by_env_arm.get(env, {})
        a, b = arms.get(arm_a, {}), arms.get(arm_b, {})
        c_, d = arms.get(arm_c, {}), arms.get(arm_d, {})
        common = sorted(set(a) & set(b) & set(c_) & set(d))
        n_pairs = len(common)
        g_ba, se_ba = _g_paired_from_two_arms(a, b, common)
        g_dc, se_dc = _g_paired_from_two_arms(c_, d, common)
        g_ca, se_ca = _g_paired_from_two_arms(a, c_, common)
        g_db, se_db = _g_paired_from_two_arms(b, d, common)
        # Interaction contrast (D−B) − (C−A) computed directly on
        # the per-pair deltas so its SE is honest under the same
        # paired structure.
        if n_pairs >= 2:
            int_deltas = [
                (d[k] - b[k]) - (c_[k] - a[k]) for k in common
            ]
            g_int, se_int = hedges_g_paired(int_deltas)
        else:
            g_int, se_int = float('nan'), float('nan')
        out.append(FactorialPerEnv(
            env_name=env,
            g_b_minus_a=g_ba, se_b_minus_a=se_ba,
            g_d_minus_c=g_dc, se_d_minus_c=se_dc,
            g_c_minus_a=g_ca, se_c_minus_a=se_ca,
            g_d_minus_b=g_db, se_d_minus_b=se_db,
            g_interaction=g_int, se_interaction=se_int,
            n_pairs=n_pairs,
        ))

    return Factorial2x2Result(
        per_env=tuple(out),
        arm_a=arm_a, arm_b=arm_b, arm_c=arm_c, arm_d=arm_d,
        source=source,
    )


__all__ = [
    'FactorialPerEnv', 'Factorial2x2Result',
    'factorial_2x2_interaction',
]
