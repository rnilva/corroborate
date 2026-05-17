"""DDQN's effect on `jensen_gap` characterized along structural
factors that Hasselt 2010's bound suggests are worth looking at.

`bias ≤ σ_action × √(2 ln K) × 1/(1 − γ)` is a LENS, not the
testbed. The cluster's content is the EMPIRICAL characterization
of DDQN's behavior on each axis the bound names:

1. **k_eff** (`ddqn_reduces_jens_uniformly_across_k_at_fr_high_gamma`):
   at FR γ=0.999 × MLP × unshaped, DDQN reduces jens uniformly
   across k_eff ∈ {4, 8, 12, 16}. The reduction doesn't break
   down at high action counts (at this scope).

2. **γ** (`ddqn_reduction_amplified_by_gamma__fr_mlp_k4_unshaped`):
   at FR × MLP × k=4 × unshaped, DDQN's reduction magnitude
   grows with γ. ~47× amplification from γ=0.99 to 0.999.

3a. **FA capacity** (`fa_capacity_moderates_ddqn_jens_reduction`):
   across (env, γ, fa_kind), DDQN's effect on jens is more
   negative at MLP than at linear FA.

3b. **FA-capacity exception** (`linear_fa_cap_fails_at_metamaze_g999__exception`):
   at MetaMaze γ=0.999 × linear, DDQN substantially reduces
   jens — an empirical anomaly worth flagging.

**What this Finding claims**: DDQN's bias-reduction at this
corpus's scope varies systematically along k_eff, γ, and FA
capacity (with the MM × γ=0.999 × linear exception). The
characterization is empirical: under each axis, the
intervention's effect is what's reported.

**What this Finding does NOT claim**:
- That any specific factor (K, σ, γ, FA) is THE mechanism.
  The empirical correlations are real; the causal attribution
  is open. The sibling discriminator
  `sigma_action_predicts_ddqn_jens_reduction` (outside this
  cluster) tests whether σ_action is a useful predictor; it
  finds POW_INSUF / point estimate sign-flipped — σ alone
  doesn't explain the empirical pattern.
- Generalization beyond the envs in scope ({FR, Acrobot,
  MountainCar, MetaMaze}). The FA-capacity rule and exception
  are corroborated only at these envs; new-env behavior is
  unstudied at this writing.
- That Hasselt 2010's bound is a TIGHT bound on DDQN's
  empirical effect. Hasselt's formula is a lens for selecting
  axes; whether the bound is empirically tight is a separate
  question, not addressed by this cluster."""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn_three_conditions.jens_reduction_factors import (
    ddqn_reduces_jens_uniformly_across_k_at_fr_high_gamma,
    ddqn_reduction_amplified_by_gamma__fr_mlp_k4_unshaped,
    fa_capacity_moderates_ddqn_jens_reduction,
    linear_fa_cap_fails_at_metamaze_g999__exception,
)


EXPECTED: ClusterVerdict = ClusterVerdict.SUPPORTED


BLOCKED_ON: str | None = None


BRIDGES: tuple[Bridge, ...] = (
    ddqn_reduces_jens_uniformly_across_k_at_fr_high_gamma,
    ddqn_reduction_amplified_by_gamma__fr_mlp_k4_unshaped,
    fa_capacity_moderates_ddqn_jens_reduction,
    linear_fa_cap_fails_at_metamaze_g999__exception,
)
