"""Tests for `analyses.random_effects_pool` — the discoverable
analysis surface for the heterogeneity-flagged pool verdict.

Each test constructs a synthetic multi-env corpus with controlled
per-env effect sizes, runs the analysis, and asserts on the
resulting `(Verdict, RefutationClass)` against the framework's
`random_effects_verdict` dispatch rules:

- PI excludes zero in predicted direction + I² < 0.5 → HELD.
- PI excludes zero in predicted direction + I² ≥ 0.5 →
  HELD_WITH_SCOPE_FLAG.
- PI brackets zero → NO_EFFECT/NULL_EFFECT.
- < 3 strata → POWER_INSUFFICIENT/UNDERPOWERED.
- PI strictly opposite to predicted → NO_EFFECT/SIGN_FLIP."""
from __future__ import annotations

import random

from corroborate.analyses.random_effects_pool import (
    RandomEffectsPoolResult, random_effects_pool,
)
from corroborate.bridge.verdict import RefutationClass, Verdict


_TREATMENT = 'treatment_arm'
_BASELINE = 'baseline_arm'


def _make_cells(
    env_effects: list[tuple[str, float, float]],
    *,
    n_seeds: int = 30,
    noise: float = 0.1,
    seed: int = 0,
) -> list[dict[str, object]]:
    """Build paired cells per env with controlled means.

    `env_effects = [(env_name, treatment_mean, baseline_mean), ...]`.
    Each env contributes `n_seeds` treatment + `n_seeds` baseline
    cells with Gaussian noise of `noise`."""
    rng = random.Random(seed)
    out: list[dict[str, object]] = []
    for env, t_mean, b_mean in env_effects:
        for s in range(n_seeds):
            out.append({
                'arm_key': _TREATMENT, 'seed': s, 'env_name': env,
                'outcome': t_mean + rng.gauss(0, noise),
            })
            out.append({
                'arm_key': _BASELINE, 'seed': s, 'env_name': env,
                'outcome': b_mean + rng.gauss(0, noise),
            })
    return out


def _run(
    cells: list[dict[str, object]],
    *,
    predicted_direction: str | None = 'a_gt_b',
) -> RandomEffectsPoolResult:
    """Direct call to the analysis (bypasses bridge fixture
    layer; tests the primitive in isolation)."""
    result = random_effects_pool.fn(
        cells,
        treatment_arm=_TREATMENT,
        baseline_arm=_BASELINE,
        source='outcome',
        pair_by=('seed',),
        predicted_direction=predicted_direction,
    )
    assert isinstance(result, RandomEffectsPoolResult)
    return result


def test_pool_held_uniform() -> None:
    """Five envs with the same large effect (uniform population) →
    PI excludes zero, I² near 0 → HELD."""
    envs = [(f'env_{i}', 1.0, 0.0) for i in range(5)]
    cells = _make_cells(envs, n_seeds=30, noise=0.1)
    result = _run(cells)
    assert result.n_strata == 5
    assert result.verdict == Verdict.HELD
    assert result.refutation is None


def test_pool_held_with_scope_flag_high_heterogeneity() -> None:
    """Per-env effects spread tight enough to keep PI positive
    while wide enough relative to within-env noise to push I²
    above 0.5. Pool excludes zero in predicted direction (all
    positive) → HELD_WITH_SCOPE_FLAG, not NO_EFFECT.

    Construction: 6 envs with t-means linearly spaced 0.5 → 1.5
    against b=0; noise=0.5 keeps per-env SE moderate so I² climbs
    without τ² blowing the PI past zero."""
    envs = [
        ('env_a', 0.5, 0.0),
        ('env_b', 0.6, 0.0),
        ('env_c', 0.7, 0.0),
        ('env_d', 0.8, 0.0),
        ('env_e', 0.9, 0.0),
        ('env_f', 1.0, 0.0),
        ('env_g', 1.1, 0.0),
        ('env_h', 1.2, 0.0),
    ]
    cells = _make_cells(envs, n_seeds=100, noise=0.5)
    result = _run(cells)
    assert result.n_strata == 8
    assert result.pooled.I2 >= 0.5
    assert result.verdict == Verdict.HELD_WITH_SCOPE_FLAG


def test_pool_no_effect_pi_brackets_zero() -> None:
    """Per-env effects centered on zero with high heterogeneity:
    {-2.0, -1.0, 0.0, +1.0, +2.0}. PI brackets zero → NO_EFFECT/
    NULL_EFFECT regardless of predicted direction."""
    envs = [
        ('env_a', -2.0, 0.0),
        ('env_b', -1.0, 0.0),
        ('env_c', 0.0, 0.0),
        ('env_d', 1.0, 0.0),
        ('env_e', 2.0, 0.0),
    ]
    cells = _make_cells(envs, n_seeds=30, noise=0.1)
    result = _run(cells)
    assert result.verdict == Verdict.NO_EFFECT
    assert result.refutation == RefutationClass.NULL_EFFECT


def test_pool_power_insufficient_n_under_three() -> None:
    """Fewer than 3 strata with finite g/SE → POWER_INSUFFICIENT/
    UNDERPOWERED per `random_effects_verdict`'s n_cells < 3 gate."""
    envs = [('only_env', 1.0, 0.0), ('one_more', 1.0, 0.0)]
    cells = _make_cells(envs, n_seeds=30, noise=0.1)
    result = _run(cells)
    assert result.n_strata == 2
    assert result.verdict == Verdict.POWER_INSUFFICIENT
    assert result.refutation == RefutationClass.UNDERPOWERED


def test_pool_refutation_class_sign_flip() -> None:
    """Predicted positive, observed PI strictly negative → NO_EFFECT
    with SIGN_FLIP refinement. Tests the verdict-enum's refutation-
    classification path through the analysis."""
    envs = [(f'env_{i}', -1.0, 0.0) for i in range(5)]
    cells = _make_cells(envs, n_seeds=30, noise=0.1)
    result = _run(cells, predicted_direction='a_gt_b')
    assert result.verdict == Verdict.NO_EFFECT
    assert result.refutation == RefutationClass.SIGN_FLIP
