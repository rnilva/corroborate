"""DDQN's outcome benefit requires THREE jointly-necessary conditions.

The substrate-level claim: DDQN's bias-correction mechanism
translates to outcome improvement IFF all three conditions hold.
Removing any one renders the mechanism dormant (Conditions 1, 2)
or actively harmful (Condition 3 broken via reward shaping).

The conditions are factored into independent intervention tests
on three different sub-corpora, so the cluster verdict is
SUPPORTED iff all three bridges HELD on their respective
interventions:

  - **Condition 1**: σ × √(2 ln K) × 1/(1−γ) > 0 (Q-bias exists)
    Test: FR γ=0.999 × k=1-4 — Δ_jens scales with K (HELD)
  - **Condition 2**: FA has capacity room for Type 1 to manifest
    Test: MC γ=0.999 linear FA — Δ_jens ≈ 0 (HELD, NULL by design)
  - **Condition 3**: Policy lacks dense alternative signal
    Test: FR γ=0.999 SHAPED vs UNSHAPED — Δ_out flips (HELD)

Each bridge tests its condition by a clean intervention with
the others held fixed:
  - C1 varies K, holds (env, γ, FA, shaping) fixed
  - C2 varies FA, holds (env, γ, shaping) fixed
  - C3 varies shaping, holds (env, γ, FA, K) fixed

The substantive consequence: practitioner advice "use DDQN" is
only well-grounded in the scope where ALL THREE conditions hold.
In the canonical 12-env panel, this scope is small — likely
FourRooms γ→1 unshaped + a few MinAtar envs with sparse reward
+ deep FA. Everywhere else, DDQN's mech HELDs but outcome NULLs
(Type-2-dominated) or slightly hurts (clip wedge in negative-
reward Type-2 envs).

Empirical readings corroborated 2026-05-15 (see memory entries
`findings_two_types_of_bias` and
`findings_shaping_decouples_bias_from_outcome` for the
full diagnostic tables)."""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn_three_conditions.conditions import (
    condition_1__q_bias_exists_under_high_gamma_and_K,
    condition_2__fa_capacity_caps_type_1_in_linear_fa,
    condition_3__shaping_decouples_mech_from_outcome,
)


# Honest verdict state (review surfaced 2026-05-15):
#
# C1 HELD at current scope: per-k_eff Cohen's d on jensen_gap is
#   uniformly < -0.5 at FR γ=0.999 MLP[64,64] no-shaping, across
#   k_eff ∈ {4, 8, 12, 16}. Observational K-scaling, NOT a test
#   of the σ × √(2 ln K) Hasselt bound named in the module
#   docstring — σ factor is unmeasured.
# C2 POWER_INSUFFICIENT: scope admits only the MC γ=0.999 linear
#   FA stratum (n_strata=1). A single-cell null test cannot
#   distinguish "FA caps Type 1 universally" from "MC linear FA
#   happens to have small σ_action". Bridge body now honestly
#   delegates to the primitive's POWER_INSUFFICIENT verdict
#   rather than overriding.
# C3 POWER_INSUFFICIENT: same shape — scope admits only the FR
#   γ=0.999 MLP shaped stratum (n_strata=1). Alternative
#   explanations (ceiling, reward-scale unit) aren't ruled out.
#
# composed_verdict returns UNDERPOWERED. The substantive
# corroboration (in memory entries `findings_two_types_of_bias`
# and `findings_shaping_decouples_bias_from_outcome`) is real,
# but the formal Hypothesis-Protocol surface here doesn't yet
# carry the multi-stratum panels needed to upgrade to SUPPORTED.
EXPECTED: ClusterVerdict = ClusterVerdict.UNDERPOWERED


BLOCKED_ON: str | None = (
    'C2 needs ≥3 strata of linear-FA × {sparse-positive env, '
    'dense-negative env, γ-sweep} to test FA-caps necessity '
    "rather than the regime-mechanical null on MC. C3 needs "
    '≥3 shaping conditions OR shaping × {FR, MC, Acrobot} '
    'plus a ceiling-vs-decoupling control. C1 prose name-drops '
    'the σ × √(2 ln K) Hasselt bound but the bridge only tests '
    'K-scaling — either rename to an observational claim or '
    'measure σ via a derived per-cell measurable.'
)


BRIDGES: tuple[Bridge, ...] = (
    condition_1__q_bias_exists_under_high_gamma_and_K,
    condition_2__fa_capacity_caps_type_1_in_linear_fa,
    condition_3__shaping_decouples_mech_from_outcome,
)
