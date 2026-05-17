"""Three bias-reduction scalings inspired by Hasselt 2010's
bound — corroborated as SCALINGS, not as factor identifications.

Hasselt 2010's `bias ≤ σ_action × √(2 ln K) × 1/(1 − γ)`
motivates three axes (action count, discount factor,
function-approximator capacity) on which to interrogate DDQN's
effect on `jensen_gap`. The four bridges below corroborate
that DDQN's bias-reduction is monotone on each axis at this
corpus's scope. They do NOT identify the bridge's manipulated
variable with Hasselt's specific factor — the bridge docstrings
disclaim each non-identification:

1. `ddqn_reduces_jens_uniformly_across_k_at_fr_high_gamma`
   — k_eff axis (FR γ=0.999 MLP unshaped × k_eff ∈ {4,8,12,16}).
   k_eff via action_duplicate creates correlated identical-
   effect actions, NOT iid K-armed-max draws. Hasselt's K
   factor is not identified.

2. `ddqn_reduction_amplified_by_gamma__fr_mlp_k4_unshaped`
   — γ axis (FR × MLP × unshaped × k_eff=4 × γ ∈ {0.99, 0.999}).
   ≥ 3× amplification threshold cannot discriminate Hasselt's
   1/(1−γ) factor from vanilla-degeneracy at γ→1 (see
   `findings_q_explosion_direct_evidence`).

3a. `fa_capacity_moderates_ddqn_jens_reduction`
   — FA axis (rule). Random-effects meta-regression on binary
   `fa_capacity` proxy across 12 strata ({FR, Acrobot, MC} × 2γ
   × 2fa). Cannot discriminate σ_action capping from Type-1
   FA-truncation; the continuous σ measurement
   (`q_action_std_late`) needed to discriminate is not
   consumed by this bridge body.

3b. `linear_fa_cap_fails_at_metamaze_g999__exception`
   — FA axis (exception). At MetaMaze γ=0.999 × linear, DDQN
   substantially reduces jens — an empirical anomaly relative
   to the FA-moderator rule. The "FA-fit-error from random-maze
   state-distribution shift" mechanism is asserted from env
   structure, not empirically discriminated.

**What this Finding claims**: DDQN's bias-reduction at this
corpus's scope exhibits three monotone scalings (K-axis,
γ-axis, FA-axis) consistent with — but not identified to —
Hasselt's bound. The empirically corroborated frame is the
Type 1 / Type 2 decomposition from
`findings_two_types_of_bias`, of which Hasselt's bound is one
theoretical instance.

**What this Finding does NOT claim**:
- That Hasselt's bound has been identified factor-by-factor.
- That `σ_action × √(2 ln K) × 1/(1−γ)` is a TIGHT bound.
- Generalisation of the FA-axis (rule + exception) beyond
  {FR, Acrobot, MountainCar} + the MetaMaze γ=0.999 exception
  cell.

**Sibling discriminator (NOT in this cluster)**:
`sigma_action_predicts_ddqn_jens_reduction` tests whether the
FA-axis bridge (3a) actually identifies Hasselt's σ_action.
Continuous meta-regression on the proper σ measure
(`q_action_std_late`) returns slope = +2.84, p=0.143 at
n_strata=12 — sign-flipped from Hasselt's prediction, NS.
Verdict: POWER_INSUFFICIENT, but the point estimate already
favors REFUTATION of the σ_action attribution. The FA-axis
HELD goes through a non-σ path. This bridge stays OUTSIDE
the cluster because the cluster's claim is about the
SCALINGS being monotone — which is corroborated. The σ
attribution is a separate claim, separately tested and not
identified."""
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
