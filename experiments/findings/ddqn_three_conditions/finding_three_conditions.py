"""Three within-scope observations consistent with the
two-types-of-bias / shaping-decouples-outcome framework.

The bridges are SCOPED OBSERVATIONS, not universal-necessity
claims. Each tests what the corpus actually has cells for:

- **C1**: observational K-scaling within FR γ=0.999 MLP[64,64]
  no-shaping. DDQN reduces `jensen_gap` uniformly across k_eff
  ∈ {4, 8, 12, 16}. Multi-stratum HELD via
  `stratified_arm_diff_pooled`.
- **C2**: single-cell null observation at MountainCar γ=0.999 ×
  LINEAR FA. DDQN does NOT appreciably reduce jens on this
  cell. `arm_mean_diff` primitive (Welch's t).
- **C3**: single-cell null observation at FourRooms γ=0.999 ×
  MLP[64,64] × SHAPED. DDQN does NOT significantly improve raw
  outcome here. Same `arm_mean_diff` shape.

The cluster Finding asserts a within-scope consistency claim:
three independent observations, each in the direction the
two-types framework predicts for its scope. SUPPORTED when all
three observations hold.

**This Finding does NOT claim universal necessity.**
Generalization beyond the three observed scopes (e.g., "linear
FA caps Type 1 across all sparse-positive envs") would require
multi-env counter-tests that the current corpus doesn't yet
carry — see the upgrade-path comments below.

The substantive theoretical framework that motivated these
bridges lives in memory entries `findings_two_types_of_bias`,
`findings_shaping_decouples_bias_from_outcome`, and
`findings_regime_discriminator_polarity_x_gamma`. The bridges
here corroborate those memos at the scope they were tested on;
they do not extrapolate."""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn_three_conditions.conditions import (
    condition_1__q_bias_exists_under_high_gamma_and_K,
    condition_2__no_appreciable_jens_reduction_under_mc_linear_fa,
    condition_3__no_outcome_benefit_under_fr_shaped,
)


EXPECTED: ClusterVerdict = ClusterVerdict.SUPPORTED


BLOCKED_ON: str | None = None


BRIDGES: tuple[Bridge, ...] = (
    condition_1__q_bias_exists_under_high_gamma_and_K,
    condition_2__no_appreciable_jens_reduction_under_mc_linear_fa,
    condition_3__no_outcome_benefit_under_fr_shaped,
)


# Upgrade path from "scoped observation" to "universal-
# necessity claim" (deferred, NOT a current claim):
#
# - C2: add ≥3 strata of linear FA × env (sparse-positive +
#   dense-negative + γ-sweep) to test "FA caps Type 1" across
#   envs, especially sparse-positive where C1 fires.
# - C3: add ≥3 shaping conditions or shaping × multiple envs,
#   plus a ceiling-vs-decoupling control distinguishing
#   "ceiling saturation" from "policy-signal override".
# - C1: measure σ_action per cell (currently unmeasured) and
#   test the σ × √(2 ln K) Hasselt-bound's load-bearing σ
#   factor directly.
