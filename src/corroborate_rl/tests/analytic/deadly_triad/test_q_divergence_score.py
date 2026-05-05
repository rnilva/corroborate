"""Closed-form assertions on the `q_divergence_score` measurable.

The substrate's `q_divergence_score` normalizes `jensen_gap` by
the Bellman fixed-point bound:

    q_divergence_score = jensen_gap · (1 - γ) / r_max

`r_max` comes from the env_catalogue (env-driven), so the
measurable composes a cell-level value (`jensen_gap`, `gamma`)
with an env-driven dependency. Tests verify:

1. Bellman bound discriminator — score < 1 when jensen_gap stays
   within the bound (FQI regime), > 1 when it exceeds (deadly-
   triad regime).
2. Composition with env_catalogue — the same numerator
   `jensen_gap` produces different scores in different envs
   because `r_max` differs (CartPole r_max=1.0 vs MNISTBandit
   r_max=1.0 vs Catch-bsuite r_max=1.0 — all happen to share
   r_max=1, so test on at least one env where r_max differs).
3. NaN sentinel on degenerate inputs (γ ≥ 1, r_max ≤ 0,
   missing fields).

The framework's `evaluate_with_measurables` resolves the
`r_max` dep automatically from the env_catalogue side-effect
registration in `corroborate_rl.dqn`."""
from __future__ import annotations

import math

import corroborate_rl.dqn  # noqa: F401  # pyright: ignore[reportUnusedImport]  # side-effect: registers q_divergence_score + r_max

from corroborate.measurables import (
    evaluate_with_measurables,
    get_registered,
)

from tests.analytic.deadly_triad.composition import (
    bellman_bound,
    expected_q_divergence_score,
    make_cell,
    FQICellSpec,
)


def _q_divergence_score(record: dict[str, object]) -> float:
    """Resolve and call `q_divergence_score` via the framework's
    measurable resolver. `record` must carry `jensen_gap`, `gamma`,
    `env_name`; `r_max` is auto-injected from env_catalogue.

    The measurable's typed return is `float`, but the resolver's
    generic signature is `T = object` — `isinstance` narrows at
    the boundary where the value enters the test's typed surface."""
    m = get_registered('q_divergence_score')
    assert m is not None, 'q_divergence_score must be registered'
    result: object = evaluate_with_measurables(m.fn, record)
    assert isinstance(result, float)
    return result


# ============ Closed-form formula recovery ============

def test_q_divergence_score_recovers_closed_form_under_bound() -> None:
    """`jensen_gap = 50` at γ=0.99 on CartPole (r_max=1.0):
    Bellman bound = 1.0 / 0.01 = 100. Closed-form score = 50/100
    = 0.5. The measurable composition through env_catalogue +
    cell-level columns must recover this exactly."""
    spec = FQICellSpec(
        env_name='CartPole-v1', gamma=0.99,
        sync_period=10_000, jensen_0=50.0, total_steps=10_000,
    )
    # n_fqi_iterations = 1, jensen_gap = 50 · 0.99^1 = 49.5
    cell = make_cell(spec, arm_key='single', seed=0,
                     jensen_gap=50.0)  # explicit override
    score = _q_divergence_score(dict(cell))
    expected = expected_q_divergence_score(
        jensen_gap=50.0, gamma=0.99, r_max=1.0,
    )
    assert abs(score - expected) < 1e-9, (
        f'q_divergence_score = {score:.6f}, closed-form = '
        f'{expected:.6f} — measurable composition broke '
        f'jensen_gap·(1-γ)/r_max'
    )
    # Sanity: 50 / 100 = 0.5 (FQI regime).
    assert abs(score - 0.5) < 1e-9


def test_q_divergence_score_exceeds_bound_under_deadly_triad() -> None:
    """A cell with `jensen_gap = 1000` on CartPole at γ=0.99 has
    Q diverged 10× beyond the Bellman bound. Score = 1000 / 100
    = 10.0 — the canonical deadly-triad signal CLAIM 11 catches.

    A regression that swapped numerator/denominator, or that
    silently capped score at 1, would fail this catastrophically."""
    spec = FQICellSpec(
        env_name='CartPole-v1', gamma=0.99,
        sync_period=1, jensen_0=1000.0, total_steps=1000,
    )
    cell = make_cell(spec, arm_key='single', seed=0,
                     jensen_gap=1000.0)
    score = _q_divergence_score(dict(cell))
    assert abs(score - 10.0) < 1e-9, (
        f'q_divergence_score = {score:.6f}, expected 10.0 '
        f'(jensen_gap=1000, bound=100). Score must scale linearly '
        f'with jensen_gap; values >> 1 are the deadly-triad signal.'
    )


# ============ Bellman bound discriminator ============

def test_score_below_one_iff_jensen_gap_within_bellman_bound() -> None:
    """The discriminator: q_divergence_score < 1 ⟺ jensen_gap <
    r_max / (1 - γ). The Bellman bound is the FQI/deadly-triad
    discriminator the substrate's bridges use to classify regime."""
    bound = bellman_bound(gamma=0.99, r_max=1.0)
    assert abs(bound - 100.0) < 1e-9  # sanity, mod float precision

    # Just-below-bound cell → score just below 1.
    spec = FQICellSpec(
        env_name='CartPole-v1', gamma=0.99,
        sync_period=10_000, jensen_0=99.0, total_steps=10_000,
    )
    cell_under = dict(make_cell(spec, arm_key='single', seed=0,
                                jensen_gap=99.0))
    score_under = _q_divergence_score(cell_under)
    assert score_under < 1.0
    assert abs(score_under - 0.99) < 1e-6

    # Just-above-bound cell → score just above 1.
    cell_over = dict(make_cell(spec, arm_key='single', seed=0,
                               jensen_gap=101.0))
    score_over = _q_divergence_score(cell_over)
    assert score_over > 1.0
    assert abs(score_over - 1.01) < 1e-6


# ============ NaN sentinel on degenerate inputs ============

def test_score_is_nan_when_gamma_at_or_above_one() -> None:
    """γ ≥ 1 makes the Bellman bound infinite (or undefined); the
    measurable returns NaN rather than ∞ or a meaningless number."""
    cell: dict[str, object] = {
        'env_name': 'CartPole-v1',
        'gamma': 1.0,  # degenerate
        'jensen_gap': 50.0,
    }
    score = _q_divergence_score(cell)
    assert math.isnan(score), (
        f'score = {score} on γ=1.0; expected NaN sentinel'
    )


def test_score_is_nan_on_unknown_env() -> None:
    """Env not in the catalogue → r_max NaN → score NaN. Catches
    upstream regressions where env_catalogue.r_max returns 0 or
    silently substitutes 1 on unknown envs."""
    cell: dict[str, object] = {
        'env_name': 'UnknownEnv-v999',
        'gamma': 0.99,
        'jensen_gap': 50.0,
    }
    score = _q_divergence_score(cell)
    assert math.isnan(score)


def test_score_is_nan_on_missing_jensen_gap() -> None:
    """Cell without `jensen_gap` (e.g., a corpus row with no trace
    available) → NaN, NOT a silent zero."""
    cell: dict[str, object] = {
        'env_name': 'CartPole-v1',
        'gamma': 0.99,
        # jensen_gap omitted
    }
    score = _q_divergence_score(cell)
    assert math.isnan(score)
