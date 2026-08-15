"""Tests for `stratified_arm_diff_pooled`'s heterogeneity-flagged
verdict (added 2026-05-12 as the discoverable surface for the
scope-cluster pattern in docs/HYPOTHESIS_AS_GRAPH.md §3b).

Replaces the prior `test_random_effects_pool.py`. The earlier
primitive paired by seed via `per_env_paired_g_panel`, inheriting
the pseudo-replication failure mode flagged in
`stratified_arm_diff_pooled.py`'s docstring. The principled
primitive uses **independent-samples** Cohen's d per stratum +
DerSimonian-Laird pool + `random_effects_verdict` dispatch.

Each test constructs a synthetic multi-env corpus with controlled
per-env (treatment_mean, baseline_mean) and asserts on the
emitted verdict against the rules in
`stats.effect_size.random_effects_verdict`."""
from __future__ import annotations

import random

from corroborate.analyses.panel.stratified_arm_diff_pooled import (
    StratifiedArmDiffPooledResult, stratified_arm_diff_pooled,
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
    """Build independent-samples cells per env with controlled means.

    Treatment + baseline are NOT paired — `seed` is just a row
    identifier within each (env, arm); the stratified analysis
    aggregates per (env, arm) and treats the (g, SE) per env as
    one observation."""
    rng = random.Random(seed)
    out: list[dict[str, object]] = []
    for env, t_mean, b_mean in env_effects:
        for s in range(n_seeds):
            out.append({
                'arm_key': _TREATMENT, 'seed': s, 'env_name': env,
                'outcome': t_mean + rng.gauss(0, noise),
                'jensen_gap': 1.0,
            })
            out.append({
                'arm_key': _BASELINE, 'seed': s, 'env_name': env,
                'outcome': b_mean + rng.gauss(0, noise),
                'jensen_gap': 1.0,
            })
    return out


def _run(
    cells: list[dict[str, object]],
    *,
    predicted_direction: str | None = 'a_gt_b',
) -> StratifiedArmDiffPooledResult:
    """Direct call to the analysis (bypasses bridge fixture
    layer; tests the primitive in isolation). `jensen_gap=1.0`
    in every cell so the stratum-level scope filter never
    excludes any strata."""
    result = stratified_arm_diff_pooled.fn(
        cells,
        source='outcome',
        treatment_arm=_TREATMENT,
        baseline_arm=_BASELINE,
        stratify_by=('env_name',),
        scope_predictor='jensen_gap',
        min_baseline_predictor=0.05,
        min_seeds_per_arm=5,
        predicted_direction=predicted_direction,
    )
    assert isinstance(result, StratifiedArmDiffPooledResult)
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
    positive) → HELD_WITH_SCOPE_FLAG.

    Construction tuned for independent-samples Cohen's d: 8 envs
    with t-means 0.5 → 1.2; noise=0.5 keeps per-env SE moderate
    so I² climbs without τ² blowing the PI past zero."""
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
    """Fewer than 3 strata with finite Cohen's d/SE →
    POWER_INSUFFICIENT/UNDERPOWERED per `random_effects_verdict`'s
    n_cells < 3 gate."""
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


def test_stratified_arm_diff_pooled_dataframe_input_identical_to_cells() -> None:
    """Canonical-input invariant: cells-input (Iterable[Mapping])
    and DataFrame-input produce the same
    `StratifiedArmDiffPooledResult`. Guards against the DataFrame
    branch diverging from cells branch."""
    import polars as pl

    envs = [(f'env_{i}', 1.0, 0.0) for i in range(5)]
    cells = _make_cells(envs, n_seeds=30, noise=0.1)
    result_cells = _run(cells, predicted_direction='a_gt_b')

    result_panel = stratified_arm_diff_pooled.fn(
        pl.DataFrame(cells),
        source='outcome',
        treatment_arm=_TREATMENT,
        baseline_arm=_BASELINE,
        stratify_by=('env_name',),
        scope_predictor='jensen_gap',
        min_baseline_predictor=0.05,
        min_seeds_per_arm=5,
        predicted_direction='a_gt_b',
    )
    # Verdict + refutation must match.
    assert result_panel.verdict == result_cells.verdict
    assert result_panel.refutation == result_cells.refutation
    # n_strata + pooled point estimate must match.
    assert result_panel.n_strata == result_cells.n_strata
    assert result_panel.pooled_d == result_cells.pooled_d
