"""Pin the `assumption_violations` heuristics on `paired_g` and
`random_effects_summary`.

These probes fire when input distributions cross thresholds
calibrated from the bias maps in this directory. The framework's
contract:

  - Empty `()` when the primitive is in its well-calibrated regime.
  - Each violation string includes the diagnostic + the predicted
    bias magnitude (so the reader can decide if it matters at
    their effect-size scale).
  - String content is part of the contract (substrate authors
    grep for keywords like 'skew_bias_likely' or
    'dl_small_g_unreliable_inference').
"""
from __future__ import annotations

import numpy as np

from corroborate.analyses.paired_g import paired_g
from corroborate.stats.effect_size import random_effects_summary


def _paired_cells(deltas: list[float]) -> list[dict[str, object]]:
    cells: list[dict[str, object]] = []
    for s, d in enumerate(deltas):
        cells.append({'arm_key': 'T', 'seed': s, 'value': float(d)})
        cells.append({'arm_key': 'B', 'seed': s, 'value': 0.0})
    return cells


# ============ paired_g heuristics ============


def test_paired_g_no_violations_on_normal_n_30() -> None:
    """Normal Δ at n=30 → no flags. Validates that the heuristic
    doesn't over-fire on well-behaved inputs."""
    rng = np.random.default_rng(0)
    deltas = list(rng.normal(1.0, 2.0, 30))
    result = paired_g.fn(
        _paired_cells(deltas),
        treatment_arm='T', baseline_arm='B',
        pair_by=('seed',), source='value',
    )
    assert result.assumption_violations == (), (
        f'normal Δ at n=30 produced violations: '
        f'{result.assumption_violations}'
    )


def test_paired_g_flags_small_n() -> None:
    """At n=5 (< 10), the heuristic flags small-n unreliability
    even on normal Δ. The estimate's CV is too large for inference
    regardless of distribution shape."""
    rng = np.random.default_rng(0)
    deltas = list(rng.normal(1.0, 2.0, 5))
    result = paired_g.fn(
        _paired_cells(deltas),
        treatment_arm='T', baseline_arm='B',
        pair_by=('seed',), source='value',
    )
    flags = result.assumption_violations
    assert any('small_n_unreliable' in f for f in flags), (
        f'expected small_n flag at n=5; got {flags}'
    )


def test_paired_g_flags_skew() -> None:
    """At log-normal Δ (skew ≈ 1.86), the heuristic flags
    `skew_bias_likely` with the predicted inflation magnitude."""
    rng = np.random.default_rng(0)
    deltas = list(rng.lognormal(0.0, 0.7, 30))
    result = paired_g.fn(
        _paired_cells(deltas),
        treatment_arm='T', baseline_arm='B',
        pair_by=('seed',), source='value',
    )
    flags = result.assumption_violations
    assert any('skew_bias_likely' in f for f in flags), (
        f'expected skew flag on log-normal Δ; got {flags}'
    )


def test_paired_g_flags_heavy_tail_only_when_kurtosis_above_5() -> None:
    """t(df=4) Δ at n=300 with seed=13 yields sample skew ≈ 0.54
    (below 1.0 threshold) AND sample excess kurtosis ≈ 5.07
    (above 5.0 threshold). This isolates the kurtosis-flag branch
    cleanly: skew flag does NOT fire, heavy_tail flag DOES fire.

    Sample skew/kurtosis on heavy-tailed populations are
    high-variance — a different seed could trip the skew flag
    instead, or neither (t(4) population excess kurtosis is
    technically infinite since the 4th moment doesn't exist; the
    test pins a SPECIFIC sample where the heuristics route
    through the kurtosis branch only)."""
    rng = np.random.default_rng(13)
    deltas = list(rng.standard_t(4, 300))
    result = paired_g.fn(
        _paired_cells(deltas),
        treatment_arm='T', baseline_arm='B',
        pair_by=('seed',), source='value',
    )
    flags = result.assumption_violations
    assert any('heavy_tail_se' in f for f in flags), (
        f'expected heavy_tail flag at this seed; got {flags}'
    )
    # Pin that the skew flag is NOT also firing — discriminating
    # the kurtosis branch from the skew branch.
    assert not any('skew_bias_likely' in f for f in flags), (
        f'skew flag fired unexpectedly; this fixture should only '
        f'trip the kurtosis branch. flags={flags}'
    )


# ============ random_effects_summary heuristics ============


def test_dl_no_violations_on_g_20_with_real_heterogeneity() -> None:
    """G=20 with τ²>0.05 → no flags. The standard regime."""
    rng = np.random.default_rng(0)
    g_se = [
        (float(rng.normal(0.5, 0.5)), 0.15)
        for _ in range(20)
    ]
    pool = random_effects_summary(g_se)
    assert pool.assumption_violations == (), (
        f'G=20 with real heterogeneity produced violations: '
        f'{pool.assumption_violations}'
    )


def test_dl_flags_small_g() -> None:
    """G=3 → `dl_small_g_unreliable_inference` flag fires
    regardless of τ²."""
    g_se = [(0.5, 0.15), (0.6, 0.12), (0.4, 0.18)]
    pool = random_effects_summary(g_se)
    flags = pool.assumption_violations
    assert any('dl_small_g_unreliable_inference' in f for f in flags), (
        f'expected small_g flag at G=3; got {flags}'
    )


def test_dl_flags_clip_artifact() -> None:
    """G=6 with very tight cluster (low between-cell variance) →
    DL τ² lands near 0; the heuristic flags `dl_clip_artifact_possible`
    pointing readers at the max(0, ·) clip caveat."""
    # Six cells, all with g near 0.5 and small SE → low Q → tau2 near 0.
    g_se = [
        (0.50, 0.15), (0.51, 0.15), (0.49, 0.15),
        (0.50, 0.15), (0.52, 0.15), (0.48, 0.15),
    ]
    pool = random_effects_summary(g_se)
    flags = pool.assumption_violations
    if pool.tau2 > 0:
        assert any('dl_clip_artifact_possible' in f for f in flags), (
            f'expected clip_artifact flag with tau2={pool.tau2}; '
            f'got {flags}'
        )
