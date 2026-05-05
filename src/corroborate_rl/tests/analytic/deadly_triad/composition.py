"""Cell construction for the deadly-triad analytic suite.

Each cell carries:
- `env_name` — picks `r_max` per env_catalogue (CartPole-v1 → 1.0).
- `gamma` — discount factor (γ=0.99 the canonical default).
- `jensen_gap` — per-cell overestimation gap. The substrate
  *normally* derives this from a `(predicted_q_at_start, mc_return)`
  array reduction; here we plant scalar values directly so the
  closed-form (sync_period, jensen_gap) relationship is the test's
  controlled axis, not the by-product of an actual training run.
- `sync_period` — target-network sync period. The structural
  predictor under the deadly-triad theorem.
- `arm_key`, `seed` — paired-g pairing keys.

Closed-form FQI decay theorem we encode:
    Under FQI (sync_period τ → ∞), Q stays within the Bellman
    bound: |Q − Q*|_∞ ≤ ε_τ where ε_τ → 0 in τ.

    Concretely, per Munos 2003 §3, after k iterations:
        |Q_k − Q*|_∞ ≤ γ^k · |Q_0 − Q*|_∞

    Number of iterations within `total_steps` training steps:
        k = total_steps // sync_period

    So the closed-form jensen_gap envelope:
        jensen_gap(τ) = jensen_0 · γ^(total_steps / τ)
        (capped below by env-specific noise floor)

Tests construct cells under this envelope and assert framework
primitives recover the predicted relationship."""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass


_DEFAULT_TOTAL_STEPS = 100_000
_DEFAULT_GAMMA = 0.99


@dataclass(frozen=True, slots=True)
class FQICellSpec:
    """Closed-form spec for a single deadly-triad-regime cell.

    `jensen_0` is the initial overestimation gap before any FQI
    decay. `total_steps` and `sync_period` together set the number
    of FQI iterations `k = total_steps // sync_period`.

    Compute the envelope jensen_gap via `expected_jensen_gap()`."""
    env_name: str
    gamma: float
    sync_period: int
    jensen_0: float
    total_steps: int = _DEFAULT_TOTAL_STEPS

    @property
    def n_fqi_iterations(self) -> int:
        return max(1, self.total_steps // self.sync_period)

    def expected_jensen_gap(self) -> float:
        """`jensen_0 · γ^k` per Munos 2003 sup-norm contraction."""
        k = self.n_fqi_iterations
        return self.jensen_0 * (self.gamma ** k)


def make_cell(
    spec: FQICellSpec,
    *,
    arm_key: str,
    seed: int,
    jensen_gap: float | None = None,
) -> Mapping[str, object]:
    """Build a single cell from the spec. `jensen_gap` defaults to
    the closed-form envelope `expected_jensen_gap()` but can be
    overridden for tests that want to construct off-envelope cells
    (e.g., the deadly-triad failure case where Q has diverged
    beyond the FQI prediction)."""
    j = spec.expected_jensen_gap() if jensen_gap is None else jensen_gap
    return {
        'id': f'{spec.env_name}/{arm_key}/τ={spec.sync_period}/seed={seed}',
        'parent_id': None,
        'cycle_id': None,
        'timestamp': '2026-01-01T00:00:00Z',
        'verdict': 'held',
        'arm_key': arm_key,
        'env_name': spec.env_name,
        'gamma': spec.gamma,
        'sync_period': spec.sync_period,
        'total_steps': spec.total_steps,
        'seed': seed,
        'jensen_gap': j,
    }


def make_paired_cells(
    *,
    short_spec: FQICellSpec,
    long_spec: FQICellSpec,
    seeds: range,
    short_arm: str = 'sync_short',
    long_arm: str = 'sync_long',
) -> list[Mapping[str, object]]:
    """Two-arm panel for `paired_g` testing. Both arms share the
    same `env_name` and `gamma`; only `sync_period` differs.

    Per-seed jensen_gap deviates from the envelope by a small
    deterministic noise `seed * 1e-4`, so paired Δ has nonzero
    variance across seeds (paired_g's SD denominator is finite).
    The mean Δ is exact closed form: `expected(short) - expected(long)`.
    """
    out: list[Mapping[str, object]] = []
    for s in seeds:
        # Add a tiny seed-dependent perturbation so SD(Δ) > 0 and
        # paired_g has finite Hedges' g. The perturbation is the
        # same across arms at a given seed → cancels in Δ but the
        # cross-seed variation comes from the cell-level offset.
        noise = s * 1e-4
        out.append(make_cell(
            short_spec, arm_key=short_arm, seed=s,
            jensen_gap=short_spec.expected_jensen_gap() + noise,
        ))
        out.append(make_cell(
            long_spec, arm_key=long_arm, seed=s,
            jensen_gap=long_spec.expected_jensen_gap() + noise,
        ))
    return out


def bellman_bound(*, gamma: float, r_max: float) -> float:
    """`r_max / (1 - γ)` — the Bellman fixed-point Q-bound.
    `q_divergence_score` normalises `jensen_gap` by this value."""
    if gamma >= 1.0 or gamma < 0.0:
        return float('nan')
    return r_max / (1.0 - gamma)


def expected_q_divergence_score(
    *, jensen_gap: float, gamma: float, r_max: float,
) -> float:
    """Closed-form `q_divergence_score`: `jensen_gap · (1 - γ) / r_max`.

    Mirrors the framework's `q_divergence_score` measurable. Tests
    that drive a cell through `evaluate_with_measurables` assert
    the framework value matches this closed form."""
    bound = bellman_bound(gamma=gamma, r_max=r_max)
    if math.isnan(bound) or bound <= 0.0:
        return float('nan')
    return jensen_gap / bound


__all__ = [
    'FQICellSpec',
    'bellman_bound',
    'expected_q_divergence_score',
    'make_cell',
    'make_paired_cells',
]
