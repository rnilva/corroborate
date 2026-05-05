"""Framework's `paired_g` + `verdict_from_paired_stats` pipeline
correctly detects DDQN's Hasselt-bias correction on
tabular-MDP-generated cells.

The previous Hasselt-bias tests verified the math: max_greedify
overestimates by σ/√π, double_greedify with independent estimators
is unbiased. Those are self-tests of the substrate. This file
asks a different question: does the FRAMEWORK correctly detect
this structural difference when it sees paired vanilla / DDQN
cells?

Pipeline:
1. Generate per-seed paired cells via the tabular bias primitives:
   - vanilla:  jensen_gap = max_greedify_tabular(ε_v)
   - ddqn:     jensen_gap = double_greedify_tabular(ε_online, ε_target)
   where ε_v, ε_online, ε_target are independent N(0, σ²)^|A| draws.
2. Feed cells to `paired_g.fn(source='jensen_gap')`.
3. Assert paired_g recovers the closed-form mean_diff = -σ/√π
   within sampling SE.
4. Feed (g, se, n_pairs) to `verdict_from_paired_stats` with
   `predicted_direction='a_lt_b'` (DDQN PREDICTED to reduce gap).
5. Assert verdict matches closed-form: HELD when n_pairs large
   enough for adequate power; POWER_INSUFFICIENT when n_pairs
   is small relative to the (small-σ) effect.

This is the framework's *headline use case*: detect whether a
proposed mechanism (DDQN) reduces a structural quantity
(jensen_gap) on a paired panel. Hasselt's theorem gives the
closed-form ground truth; the framework's job is to recover it
with the right verdict.

Catches regressions in:
- paired_g.fn: pairing on seed, mean_diff aggregation, sign
- verdict_from_paired_stats: HELD / POWER_INSUFFICIENT / SIGN_FLIP
  decision tree
- The substrate-side bias primitives feeding cell construction
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from corroborate.analyses.paired_g import paired_g
from corroborate.bridge.verdict import RefutationClass, Verdict
from corroborate.stats.effect_size import verdict_from_paired_stats

from corroborate_rl.tabular import (
    double_greedify_tabular,
    hasselt_n2_max_bias,
    max_greedify_tabular,
)


def _generate_hasselt_paired_cells(
    *,
    n_pairs: int,
    sigma: float,
    n_actions: int = 2,
    seed_offset: int = 0,
) -> list[Mapping[str, object]]:
    """Generate paired vanilla / DDQN cells per Hasselt's setup.

    Per seed s:
      ε_v       ~ N(0, σ²)^|A|     (vanilla noise)
      ε_online  ~ N(0, σ²)^|A|     (DDQN online estimator noise)
      ε_target  ~ N(0, σ²)^|A|     (DDQN target estimator noise; ⫫ online)
      vanilla.jensen_gap = max_greedify_tabular(ε_v)         # E = σ/√π
      ddqn.jensen_gap    = double_greedify_tabular(           # E = 0
                              ε_online, ε_target)

    Returns 2 * n_pairs cells: each seed contributes one vanilla
    cell and one ddqn cell. Per-seed independent noise streams
    via numpy.random.default_rng + seed offsets."""
    cells: list[Mapping[str, object]] = []
    for s in range(n_pairs):
        rng = np.random.default_rng(seed_offset + s)
        eps_v = (rng.standard_normal(n_actions) * sigma).astype(np.float64)
        eps_online = (rng.standard_normal(n_actions) * sigma).astype(np.float64)
        eps_target = (rng.standard_normal(n_actions) * sigma).astype(np.float64)
        cells.append({
            'arm_key': 'vanilla', 'seed': s, 'env_name': 'hasselt_toy',
            'jensen_gap': max_greedify_tabular(eps_v),
        })
        cells.append({
            'arm_key': 'ddqn', 'seed': s, 'env_name': 'hasselt_toy',
            'jensen_gap': double_greedify_tabular(eps_online, eps_target),
        })
    return cells


def _closed_form_paired_delta_se(
    *, sigma: float, n_pairs: int,
) -> float:
    """Closed-form SE of paired Δ = ddqn_jensen - vanilla_jensen.

    Per pair, vanilla_jensen and ddqn_jensen are computed from
    independent noise streams, so:
        Var(Δ) = Var(vanilla_jensen) + Var(ddqn_jensen)
    For |A|=2 iid N(0, σ²) noise:
        Var(vanilla_jensen = max) = σ² · (1 - 1/π)
        Var(ddqn_jensen = ε_target[argmax ε_online]) = σ²
            (argmax_a ε_online is independent of ε_target, so the
             selected index is uniform → ε_target at that index
             is just N(0, σ²) → variance σ².)
    So Var(Δ) = σ² · (2 - 1/π).
    SE(mean over n_pairs) = sqrt(σ² · (2 - 1/π) / n_pairs)
                          = σ · sqrt((2 - 1/π) / n_pairs)
    """
    var_per_pair = sigma * sigma * (2.0 - 1.0 / math.pi)
    return math.sqrt(var_per_pair / n_pairs)


# ============ Headline: paired_g recovers Hasselt mean_diff ============

def test_paired_g_recovers_closed_form_hasselt_bias_correction() -> None:
    """The framework's paired_g.fn on Hasselt-toy cells must
    report mean_diff matching the closed form -σ/√π within 4·SE.

    Closed form: ddqn cells have E[jensen_gap] = 0 (unbiased),
    vanilla cells have E[jensen_gap] = σ/√π. Paired Δ has
    E[Δ] = -σ/√π = -0.5642 (for σ=1).

    A regression in paired_g's pairing-on-seed, mean computation,
    or sign would breach the bound. The substrate primitives
    `max_greedify_tabular` / `double_greedify_tabular` are also
    in the call path — a bug there propagates here."""
    sigma = 1.0
    n_pairs = 200  # tight SE: 4·SE ~ 0.367 around closed-form -0.564

    cells = _generate_hasselt_paired_cells(
        n_pairs=n_pairs, sigma=sigma, n_actions=2,
    )
    result = paired_g.fn(
        cells,
        treatment_arm='ddqn',
        baseline_arm='vanilla',
        pair_by=('seed',),
        source='jensen_gap',
    )
    assert result.n_pairs == n_pairs

    expected_mean_diff = -hasselt_n2_max_bias(sigma)  # = -σ/√π
    assert abs(expected_mean_diff + sigma / math.sqrt(math.pi)) < 1e-12

    se = _closed_form_paired_delta_se(sigma=sigma, n_pairs=n_pairs)
    bound = 4.0 * se
    assert abs(result.mean_diff - expected_mean_diff) < bound, (
        f'paired_g.mean_diff = {result.mean_diff:.4f}, closed-form '
        f'-σ/√π = {expected_mean_diff:.4f}, 4·SE = {bound:.4f}. '
        f'The framework should detect DDQN reducing jensen_gap by '
        f'σ/√π exactly.'
    )
    # And the framework's reported g should be strongly negative
    # — DDQN reduces the gap.
    assert result.g < 0, (
        f'paired_g.g = {result.g:.4f}; DDQN reduces jensen_gap, '
        f'so g should be negative under treatment=ddqn ordering'
    )


# ============ Verdict: HELD with predicted_direction='a_lt_b' ============

def test_verdict_held_when_ddqn_predicted_to_reduce_gap_at_adequate_power() -> None:
    """End-to-end pipeline: cells → paired_g → verdict_from_paired_stats.
    With predicted_direction='a_lt_b' (DDQN predicted < vanilla on
    jensen_gap) and adequate power (n_pairs=200, σ=1 → strong
    effect), the verdict must be HELD with no refutation.

    Catches a regression where the verdict layer mis-routes
    predicted_direction: 'a_lt_b' should accept negative g."""
    sigma = 1.0
    n_pairs = 200
    cells = _generate_hasselt_paired_cells(
        n_pairs=n_pairs, sigma=sigma, n_actions=2,
    )
    result = paired_g.fn(
        cells,
        treatment_arm='ddqn',
        baseline_arm='vanilla',
        pair_by=('seed',),
        source='jensen_gap',
    )
    verdict, refutation, is_powered = verdict_from_paired_stats(
        result.g, result.se, n=result.n_pairs,
        predicted_direction='a_lt_b',  # DDQN predicted < vanilla
    )
    assert verdict is Verdict.HELD, (
        f'verdict = {verdict.value!r}; DDQN reduces jensen_gap '
        f'with closed-form mean_diff = -σ/√π and 200 paired seeds '
        f'→ adequately-powered HELD with predicted_direction=a_lt_b. '
        f'(g = {result.g:.4f}, se = {result.se:.4f})'
    )
    assert refutation is None
    assert is_powered


# ============ POWER_INSUFFICIENT: small σ + small n_pairs ============

def test_verdict_power_insufficient_when_effect_is_below_mde() -> None:
    """At σ=0.05 and n_pairs=5, the closed-form effect mean_diff
    = -σ/√π ≈ -0.028 is tiny, n is small, → |g| ends up below
    MDE → POWER_INSUFFICIENT.

    The framework's distinguishing claim (CLAUDE.md acceptance
    §3.4) is that POWER_INSUFFICIENT is a first-class verdict
    DISTINCT from NO_EFFECT — the test exercises this on a
    structurally-real-but-tiny effect.

    A regression that conflated 'sub-MDE' with 'no effect' would
    return NO_EFFECT here, smuggling the methodological problem
    past the reader."""
    sigma = 0.05
    n_pairs = 5
    cells = _generate_hasselt_paired_cells(
        n_pairs=n_pairs, sigma=sigma, n_actions=2,
    )
    result = paired_g.fn(
        cells,
        treatment_arm='ddqn',
        baseline_arm='vanilla',
        pair_by=('seed',),
        source='jensen_gap',
    )
    verdict, refutation, is_powered = verdict_from_paired_stats(
        result.g, result.se, n=result.n_pairs,
        predicted_direction='a_lt_b',
    )
    assert verdict is Verdict.POWER_INSUFFICIENT, (
        f'verdict = {verdict.value!r}; tiny σ={sigma} + small '
        f'n_pairs={n_pairs} → effect well below MDE. The '
        f'framework MUST distinguish this from NO_EFFECT — there '
        f'IS a real (tiny) Hasselt bias being corrected, just no '
        f'power to detect it (g = {result.g:.4f})'
    )
    assert refutation is RefutationClass.UNDERPOWERED
    assert not is_powered


# ============ SIGN_FLIP: wrong-direction prediction ============

def test_verdict_sign_flip_on_wrong_direction_prediction() -> None:
    """Same data as the HELD test but with
    predicted_direction='a_gt_b' (DDQN PREDICTED to INCREASE gap).
    The closed-form data shows the opposite (g strongly negative)
    → NO_EFFECT/SIGN_FLIP. Catches a regression where the verdict
    layer accepts arbitrary signs under a directional prediction."""
    sigma = 1.0
    n_pairs = 200
    cells = _generate_hasselt_paired_cells(
        n_pairs=n_pairs, sigma=sigma, n_actions=2,
    )
    result = paired_g.fn(
        cells,
        treatment_arm='ddqn',
        baseline_arm='vanilla',
        pair_by=('seed',),
        source='jensen_gap',
    )
    verdict, refutation, _ = verdict_from_paired_stats(
        result.g, result.se, n=result.n_pairs,
        predicted_direction='a_gt_b',  # WRONG — DDQN actually decreases gap
    )
    assert verdict is Verdict.NO_EFFECT, (
        f'verdict = {verdict.value!r}; data shows g = '
        f'{result.g:.4f} (DDQN decreased gap), prediction was '
        f'a_gt_b — sign mismatch must be NO_EFFECT/SIGN_FLIP'
    )
    assert refutation is RefutationClass.SIGN_FLIP
