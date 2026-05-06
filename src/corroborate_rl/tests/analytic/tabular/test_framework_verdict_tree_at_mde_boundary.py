"""Framework probe: `verdict_from_paired_stats` MDE-boundary
routing.

The framework's distinguishing methodological claim (CLAUDE.md
§3.4) is that POWER_INSUFFICIENT is a first-class verdict
DISTINCT from NO_EFFECT — treating an underpowered test as "no
effect" smuggles methodological problems past the reader.

The boundary that determines POWER_INSUFFICIENT vs HELD lies at
`g = MDE(n, α, power)`. Below MDE → POWER_INSUFFICIENT regardless
of sign. Above MDE → HELD (sign correct) or SIGN_FLIP (sign wrong).

This is NOT a closed-form-recovery test. The framework computes
MDE via `statsmodels.stats.power.TTestPower.solve_power` — we
read MDE FROM THE FRAMEWORK and assert the verdict tree routes
correctly across the threshold. This pins the routing logic, not
the MDE formula (the MDE primitive is its own thing).

What this catches:
- A regression that swapped the `<` and `>=` boundary on g vs MDE
- A regression that conflated POWER_INSUFFICIENT with NO_EFFECT
  (the load-bearing CLAUDE.md §3.4 claim)
- A regression that ignored the predicted_direction routing at
  the boundary (e.g., always returning HELD when |g| ≥ MDE)
"""
from __future__ import annotations

from corroborate.bridge.verdict import RefutationClass, Verdict
from corroborate.stats.effect_size import (
    mde_paired,
    verdict_from_paired_stats,
)


def test_verdict_held_at_g_just_above_mde() -> None:
    """At g = MDE + ε (just above threshold), `adequately_powered_paired`
    returns True → HELD when predicted_direction matches.

    Pin the routing layer above the boundary; the `==` exact case
    is asserted in `test_verdict_held_at_g_exactly_mde` below."""
    n = 20
    mde = mde_paired(n, alpha=0.05, power=0.8)
    g = mde + 1e-6     # just above MDE
    verdict, refutation, is_powered = verdict_from_paired_stats(
        g, se=0.1, n=n, predicted_direction='a_gt_b',
    )
    assert verdict is Verdict.HELD, (
        f'verdict at g = MDE+ε = {g:.6f}: {verdict.value!r}; '
        f'expected HELD (g just above MDE = {mde:.4f}, sign matches '
        f'predicted_direction).'
    )
    assert refutation is None
    assert is_powered is True


def test_verdict_held_at_g_exactly_mde() -> None:
    """Pin the inclusive boundary: at `g == MDE` exactly,
    `adequately_powered_paired` uses `abs(g) >= mde` → True →
    HELD when sign matches.

    A regression that flipped `>=` to `>` would breach here
    while passing the `g = MDE + 1e-6` test (which has positive
    margin). This is the load-bearing inclusivity assertion the
    reviewer flagged as missing in the post-Phase-2 audit.
    """
    n = 20
    mde = mde_paired(n, alpha=0.05, power=0.8)
    verdict, refutation, is_powered = verdict_from_paired_stats(
        mde, se=0.1, n=n, predicted_direction='a_gt_b',
    )
    assert verdict is Verdict.HELD, (
        f'verdict at g = MDE = {mde:.6f} exactly: {verdict.value!r}; '
        f'expected HELD. The framework defines '
        f'adequately_powered_paired via `abs(g) >= mde` (inclusive). '
        f'A regression to strict `>` would mis-route the boundary.'
    )
    assert is_powered is True
    assert refutation is None


def test_verdict_power_insufficient_at_g_just_below_mde() -> None:
    """At g = MDE − ε (just below threshold), `adequately_powered_paired`
    returns False → POWER_INSUFFICIENT regardless of sign or
    predicted_direction.

    Pin that the framework refuses the "no effect" smuggle: the
    structural claim of CLAUDE.md §3.4."""
    n = 20
    mde = mde_paired(n, alpha=0.05, power=0.8)
    g = mde - 1e-3     # just below MDE
    verdict, refutation, is_powered = verdict_from_paired_stats(
        g, se=0.1, n=n, predicted_direction='a_gt_b',
    )
    assert verdict is Verdict.POWER_INSUFFICIENT, (
        f'verdict at g = MDE−ε = {g:.4f}: {verdict.value!r}; '
        f'expected POWER_INSUFFICIENT (CLAUDE.md §3.4 — '
        f'underpowered ≠ no-effect).'
    )
    assert refutation is RefutationClass.UNDERPOWERED
    assert is_powered is False


def test_verdict_power_insufficient_distinct_from_null_effect() -> None:
    """The CLAUDE.md §3.4 distinguishing claim: a sub-MDE effect
    is NOT a null effect. Same `g, se, n` but with predicted_direction
    in `{'a_gt_b', 'a_lt_b', None}` should ALL return
    POWER_INSUFFICIENT (the upstream MDE check short-circuits
    before any direction-routing).

    A regression that conflated POWER_INSUFFICIENT with NO_EFFECT
    by some predicted_direction-conditional path would breach.
    """
    n = 20
    mde = mde_paired(n, alpha=0.05, power=0.8)
    g = mde - 0.05
    for direction in ('a_gt_b', 'a_lt_b', 'two_sided', None):
        verdict, _, _ = verdict_from_paired_stats(
            g, se=0.1, n=n, predicted_direction=direction,
        )
        assert verdict is Verdict.POWER_INSUFFICIENT, (
            f'verdict at predicted_direction={direction!r}: '
            f'{verdict.value!r}; expected POWER_INSUFFICIENT '
            f'across all directions (MDE check short-circuits).'
        )


def test_verdict_sign_flip_above_mde_with_wrong_direction() -> None:
    """Above MDE with the WRONG predicted direction → SIGN_FLIP.
    Combined with the previous tests, this pins the 3-way routing:

        |g| < MDE                → POWER_INSUFFICIENT
        |g| ≥ MDE, sign matches  → HELD
        |g| ≥ MDE, sign opposes  → NO_EFFECT/SIGN_FLIP

    Critically: predicted_direction='a_gt_b' (positive) with
    g < 0 (negative) should NOT collapse to POWER_INSUFFICIENT
    just because the magnitude is above MDE.
    """
    n = 20
    mde = mde_paired(n, alpha=0.05, power=0.8)
    g = -(mde + 0.1)   # negative g, magnitude above MDE
    verdict, refutation, is_powered = verdict_from_paired_stats(
        g, se=0.1, n=n, predicted_direction='a_gt_b',
    )
    assert verdict is Verdict.NO_EFFECT
    assert refutation is RefutationClass.SIGN_FLIP
    assert is_powered is True
