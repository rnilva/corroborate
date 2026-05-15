"""DDQN reduces `jensen_gap` uniformly across action-multiplier
K at FourRooms γ=0.999 × MLP[64,64].

Single observational bridge HELD via multi-stratum (k_eff ∈
{4,8,12,16}) `stratified_arm_diff_pooled` with pooled Cohen's
d < -0.5 at every stratum. The substantive theoretical
framework that motivated this observation (two-types
decomposition, FA-capacity gate, policy-signal-strength) lives
in memory entries `findings_two_types_of_bias`,
`findings_shaping_decouples_bias_from_outcome`, and
`findings_regime_discriminator_polarity_x_gamma`.

**What this Finding does NOT claim**:
- The Hasselt σ × √(2 ln K) × 1/(1−γ) bound has been verified
  — σ_action is unmeasured.
- The framework generalizes across envs — only FR γ=0.999
  MLP[64,64] is tested.
- DDQN's *outcome* benefit requires Q-bias to exist — this
  bridge tests `jensen_gap` reduction, not outcome translation.

**Retracted bridges (round-3 audit, 2026-05-15)**: prior versions
of this module included two additional condition bridges:

- `condition_2__no_appreciable_jens_reduction_under_mc_linear_fa`
  — single-cell observation on MC γ=0.999 linear FA. Reviewer
  flagged that the null was *mechanically* null from regime
  (FA-capped σ_action for both arms) AND single-cell
  POWER_INSUFFICIENT at observed n=60 (MDE ≈ 0.51, observed
  d=-0.11 sits inside the band).
- `condition_3__no_outcome_benefit_under_fr_shaped` — single-
  cell observation on FR γ=0.999 MLP shaped. Same shape:
  POWER_INSUFFICIENT at observed n=30 (MDE ≈ 0.74, observed
  d=-0.23 sits inside the band). Earlier `|d| < threshold OR
  p > threshold` verdict logic was also one-sided-broken.

Both retracted rather than smuggled into HELD via verdict-rule
gymnastics. The substantive observations underlying them are
real (see memos above) but the formal Hypothesis-Protocol
surface here doesn't carry adequate power.

**Upgrade path to a multi-condition Finding** (deferred):
- For C2: build a 2-stratum panel of linear FA × env (FR γ=0.999
  linear unshaped + MC γ=0.999 linear) — requires generating FR
  linear-FA unshaped cells.
- For C3: build a 2×2 factorial on FR γ=0.999 ({shaped, unshaped}
  × {linear, MLP}) via `factorial_2x2` primitive — corpus has
  the cells; factorial bridge needs authoring."""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn_three_conditions.conditions import (
    ddqn_reduces_jens_uniformly_across_k_at_fr_high_gamma,
)


EXPECTED: ClusterVerdict = ClusterVerdict.SUPPORTED


BLOCKED_ON: str | None = None


BRIDGES: tuple[Bridge, ...] = (
    ddqn_reduces_jens_uniformly_across_k_at_fr_high_gamma,
)
