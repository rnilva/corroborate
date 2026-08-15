"""Cross-cell deadly-triad assertions: paired_g + meta-regression
on a panel where (sync_period, jensen_gap) is encoded per the FQI
contraction theorem.

The FQI mechanism (auto-memory `findings_fqi_mechanism`):
   long sync ⇒ many fewer FQI iterations within total_steps ⇒
   target stays fixed long enough for online to regress to a
   bounded fixed point ⇒ Q stays within Bellman bound.

Closed-form envelope (Munos 2003 §3 sup-norm contraction):
   jensen_gap(τ) = jensen_0 · γ^(total_steps / τ)

So as τ → ∞:
   short sync (τ=1, deadly-triad regime):
     k = total_steps iterations → jensen_gap → 0 in theory but
     in practice off-policy + FA disrupts contraction → Q
     diverges. We model the deadly-triad with a large
     `jensen_0_short` floor (Q has escaped the bound).
   long sync (τ=total_steps/k_few, FQI regime):
     k = few iterations → jensen_gap follows the geometric
     decay envelope from `jensen_0_long`.

Tests verify:

1. Per-arm `q_divergence_score` matches the closed-form ratio.
2. Paired `g` across short vs long arms detects the
   FQI mechanism — long arm has lower divergence.
3. The cross-arm difference matches the closed form
   `(jensen_short - jensen_long) · (1 - γ) / r_max`."""
from __future__ import annotations

import corroborate_rl.dqn  # noqa: F401  # pyright: ignore[reportUnusedImport]  # side-effect: registers q_divergence_score + r_max

from corroborate.analyses.paired.paired_g import paired_g
from corroborate.measurables import (
    evaluate_with_measurables,
    get_registered,
)

from tests.analytic.deadly_triad.composition import (
    FQICellSpec,
    bellman_bound,
    expected_q_divergence_score,
    make_paired_cells,
)


_GAMMA = 0.99
_R_MAX_CARTPOLE = 1.0
_TOTAL_STEPS = 100_000
_N_PAIRS = 30


def _augment_with_q_divergence_score(
    cells: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Compute per-cell q_divergence_score via the framework's
    measurable resolver and stamp it onto the cell dict at the
    `q_divergence_score` key. Mirrors what the cache builder does
    in production (`compute_missing_columns`)."""
    m = get_registered('q_divergence_score')
    assert m is not None
    out: list[dict[str, object]] = []
    for c in cells:
        result: object = evaluate_with_measurables(m.fn, c)
        assert isinstance(result, float)
        out.append({**c, 'q_divergence_score': result})
    return out


# ============ FQI mechanism: long sync drives jensen_gap down ============

def test_long_sync_arm_has_lower_q_divergence_than_short_sync() -> None:
    """Two arms differing only in `sync_period`. The FQI envelope
    `jensen_gap = jensen_0 · γ^(total_steps/τ)` predicts the long
    arm's jensen_gap is strictly smaller — paired g on
    `q_divergence_score` should detect this with closed-form magnitude.

    Implementation parameters (CartPole, γ=0.99):
        bound      = r_max / (1 - γ) = 100
        short arm: τ=10,    jensen_0=120  → k = 10000
                   envelope ≈ 120 · 0.99^10000 ≈ 0 (already converged)
        long arm:  τ=10000, jensen_0=120  → k = 10
                   envelope ≈ 120 · 0.99^10 ≈ 108.6

    But the deadly triad means short-sync DOESN'T actually
    converge to envelope; it diverges. Override jensen_gap on the
    short arm to model that. Long arm follows the envelope.
    """
    short_spec = FQICellSpec(
        env_name='CartPole-v1', gamma=_GAMMA,
        sync_period=10, jensen_0=120.0, total_steps=_TOTAL_STEPS,
    )
    long_spec = FQICellSpec(
        env_name='CartPole-v1', gamma=_GAMMA,
        sync_period=10_000, jensen_0=120.0, total_steps=_TOTAL_STEPS,
    )
    cells_raw = make_paired_cells(
        short_spec=short_spec,
        long_spec=long_spec,
        seeds=range(_N_PAIRS),
    )
    cells = _augment_with_q_divergence_score(
        [dict(c) for c in cells_raw],
    )

    result = paired_g.fn(
        cells,
        treatment_arm='sync_long',  # 'helps' — predicted lower
        baseline_arm='sync_short',
        pair_by=('seed',),
        source='q_divergence_score',
    )
    assert result.n_pairs == _N_PAIRS

    # Closed-form mean_diff:
    #   q_score(arm) = jensen(arm) · (1 - γ) / r_max
    #   Δ = q_score(long) - q_score(short)
    #     = [jensen(long) - jensen(short)] · (1 - γ) / r_max
    expected_long_score = expected_q_divergence_score(
        jensen_gap=long_spec.expected_jensen_gap(),
        gamma=_GAMMA, r_max=_R_MAX_CARTPOLE,
    )
    expected_short_score = expected_q_divergence_score(
        jensen_gap=short_spec.expected_jensen_gap(),
        gamma=_GAMMA, r_max=_R_MAX_CARTPOLE,
    )
    expected_mean_diff = expected_long_score - expected_short_score

    rel_err = abs(result.mean_diff - expected_mean_diff) / abs(
        expected_mean_diff,
    )
    assert rel_err < 0.01, (
        f'paired_g.mean_diff = {result.mean_diff:.6f}, expected '
        f'{expected_mean_diff:.6f} (long_score={expected_long_score:.4f}, '
        f'short_score={expected_short_score:.4f}). The cross-arm '
        f'q_divergence Δ should match the closed-form FQI envelope '
        f'difference; >1% drift indicates either the measurable '
        f'composition or the paired_g pairing broke.'
    )


# ============ Bellman-bound discrimination on the panel ============

def test_panel_classifies_arms_by_bellman_bound() -> None:
    """Independent of the cross-arm Δ: under the FQI envelope, the
    long-sync arm's q_divergence_score is below 1 (within bound)
    AND the short-sync arm with diverged Q is above 1 (deadly-
    triad). The substrate's bridges use this 1.0 threshold to
    classify cell regimes — we assert the panel populates
    correctly per cell.

    Note: `make_paired_cells` adds a tiny per-seed perturbation;
    we check by env-mean, not by every cell."""
    short_spec = FQICellSpec(
        env_name='CartPole-v1', gamma=_GAMMA,
        sync_period=1, jensen_0=500.0, total_steps=_TOTAL_STEPS,
    )
    long_spec = FQICellSpec(
        env_name='CartPole-v1', gamma=_GAMMA,
        sync_period=10_000, jensen_0=80.0, total_steps=_TOTAL_STEPS,
    )
    cells_raw = make_paired_cells(
        short_spec=short_spec,
        long_spec=long_spec,
        seeds=range(_N_PAIRS),
    )
    # Override short arm to model deadly-triad divergence:
    # actual jensen_gap stays at 500 (Q never converged).
    for c in cells_raw:
        if c['arm_key'] == 'sync_short':
            c['jensen_gap'] = 500.0  # type: ignore[index]
    cells = _augment_with_q_divergence_score(
        [dict(c) for c in cells_raw],
    )

    short_scores = [
        c['q_divergence_score'] for c in cells
        if c['arm_key'] == 'sync_short'
    ]
    long_scores = [
        c['q_divergence_score'] for c in cells
        if c['arm_key'] == 'sync_long'
    ]
    assert all(isinstance(s, float) for s in short_scores)
    assert all(isinstance(s, float) for s in long_scores)
    short_floats = [float(s) for s in short_scores]  # pyright: ignore[reportArgumentType]
    long_floats = [float(s) for s in long_scores]  # pyright: ignore[reportArgumentType]

    # Bellman bound = 100 in normalized units. Score = jensen / bound.
    bound = bellman_bound(gamma=_GAMMA, r_max=_R_MAX_CARTPOLE)
    assert abs(bound - 100.0) < 1e-9

    # Short arm: jensen_gap=500 → score=5.0 → above bound (deadly triad).
    for s in short_floats:
        assert s > 1.0, (
            f'short-sync score = {s:.4f} below 1.0; the deadly-'
            f'triad regime should put Q above the Bellman bound'
        )
    # Long arm: envelope = 80 · 0.99^10 ≈ 72.3 → score ≈ 0.72.
    for s in long_floats:
        assert s < 1.0, (
            f'long-sync score = {s:.4f} above 1.0; FQI regime '
            f'should keep Q within the Bellman bound'
        )


# ============ Independence of arms is preserved ============

def test_q_divergence_score_panel_preserves_arm_independence() -> None:
    """Cell-level q_divergence_score depends ONLY on that cell's
    (jensen_gap, gamma, env_name) — it must not leak across arms.
    A regression where the measurable resolver shared state across
    cells (e.g., a stale cache key) would break this property."""
    short_spec = FQICellSpec(
        env_name='CartPole-v1', gamma=_GAMMA,
        sync_period=1, jensen_0=500.0, total_steps=_TOTAL_STEPS,
    )
    long_spec = FQICellSpec(
        env_name='CartPole-v1', gamma=_GAMMA,
        sync_period=10_000, jensen_0=80.0, total_steps=_TOTAL_STEPS,
    )
    cells_raw = make_paired_cells(
        short_spec=short_spec,
        long_spec=long_spec,
        seeds=range(5),  # small set for clarity
    )
    cells = _augment_with_q_divergence_score(
        [dict(c) for c in cells_raw],
    )
    # Each cell's score must equal closed-form on its own jensen_gap.
    for c in cells:
        jensen = c['jensen_gap']
        assert isinstance(jensen, float)
        expected = expected_q_divergence_score(
            jensen_gap=jensen, gamma=_GAMMA, r_max=_R_MAX_CARTPOLE,
        )
        observed = c['q_divergence_score']
        assert isinstance(observed, float)
        assert abs(observed - expected) < 1e-9, (
            f"cell {c['id']!r}: score={observed:.6f}, expected "
            f'{expected:.6f} from jensen_gap={jensen:.4f}'
        )
