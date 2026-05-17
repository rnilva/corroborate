"""Hasselt 2010's three-factor bound corroborated.

DDQN's bias-reduction at this envelope's scope is well-described
by `bias ≤ σ_action × √(2 ln K) × 1/(1 − γ)`. Four bridges, one
per factor (with σ_action as a rule + exception cluster):

1. `ddqn_reduces_jens_uniformly_across_k_at_fr_high_gamma`
   — the `√(2 ln K)` factor: jens reduction scales monotonically
   across k_eff ∈ {4, 8, 12, 16} at FR γ=0.999 MLP unshaped.

2. `ddqn_reduction_amplified_by_gamma__fr_mlp_k4_unshaped`
   — the `1/(1−γ)` factor: at controlled K (k_eff=4),
   |mean_diff(γ=0.999)| ≥ 3× |mean_diff(γ=0.99)| AND per-γ
   Cohen's d ≤ -0.8 at both γ.

3a. `fa_capacity_moderates_ddqn_jens_reduction`
   — the σ_action factor: random-effects meta-regression of
   per-(env, γ, fa_kind) Cohen's d on `fa_capacity` (0=linear,
   1=mlp_deep) across 12 strata ({FR, Acrobot, MountainCar} × 2
   γ × 2 fa). HELD iff slope ≤ −0.5 AND significant.

3b. `linear_fa_cap_fails_at_metamaze_g999__exception`
   — the named exception: at MetaMaze γ=0.999 × linear FA, the
   cap fails (d ≤ −0.3 across n_episodes strata) because the
   random-maze-per-episode structure forces FA-fit-error bias
   that DDQN clips via a non-σ path.

Bridges (3a, 3b) form a rule + exception cluster for the σ
factor — together they encode "σ-via-FA gates the Hasselt mech
EXCEPT where FA-fit error provides a parallel bias path".

**What this Finding does NOT claim**:
- That `σ_action × √(2 ln K) × 1/(1−γ)` is a TIGHT bound — only
  that the three factors' structural predictions hold
  empirically.
- Generalisation of the σ-via-FA rule beyond {FR, Acrobot, MC}.
- That MetaMaze γ=0.999's FA-fit-error mechanism (3b)
  generalises to other non-stationary envs."""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn_three_conditions.hasselt_bound import (
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
