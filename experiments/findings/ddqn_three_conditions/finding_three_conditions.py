"""DDQN's outcome benefit at FR γ=0.999 × MLP × unshaped has a
mechanism (C1: jens reduction across K) AND a moderator
(C2: linear FA caps the mech, C3b: shaping kills the outcome
translation). Four bridges, all multi-stratum panels with
adequate per-arm n.

The finding is the cluster:

- C1 (jens reduction × K) HELD — establishes the Hasselt mech
  fires.
- C2 (linear FA null × env) HELD — establishes FA capacity is
  load-bearing.
- C3a (outcome benefit × K, unshaped MLP) HELD — establishes
  the reference cell where translation DOES happen.
- C3b (no outcome benefit × γ, shaped MLP) HELD — establishes
  the moderation: same env, same FA, shaping kills outcome
  translation.

`SUPPORTED` requires ALL four bridges HELD. Mixed (some HELD,
some POWER_INSUFFICIENT or NO_EFFECT) → UNDERPOWERED. Any
INVARIANT_VIOLATION on C2/C3b → REFUTED.

**What this finding does NOT claim**:
- The Hasselt σ × √(2 ln K) × 1/(1−γ) bound has been verified
  — σ_action is unmeasured.
- Generalization beyond the 4 envs in C2's panel
  (FR + Acrobot + MM + MC).
- That shaping moderates the benefit at envs OTHER than FR (no
  shaped corpora for other envs).
- Cross-FA generalization of C3 (only MLP[64,64] shaped corpus
  was generated).

**History — round 4 recovery (2026-05-15)**: After three rounds
of audit shrunk this module to a single K-scaling bridge
(C1 only), the user requested recovery of C2 and C3 as
multi-stratum panels using existing corpora. The recovered
form uses:

- C2: `linear_fa_caps_type_1_across_envs__null_panel` (4-stratum
  env panel, `predicted_direction='null'`, ±0.5 null ceiling).
- C3: TWO sibling bridges — `ddqn_helps_outcome_at_fr_g999_mlp_unshaped__k_panel`
  (4-stratum positive arm) + `shaping_decouples_outcome_benefit__fr_shaped_fa_x_gamma_panel`
  (2-stratum null arm). The moderation pattern is read from the
  cluster.

Cells available across `fa_depth_fourrooms`, `fa_depth_xenv_gpu`,
`fa_degeneracy_shaped_only`, and the prior K-sweep corpora.
None of these required a new sweep — the recovery was an
authoring move on existing data."""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn_three_conditions.conditions import (
    ddqn_helps_outcome_at_fr_g999_mlp_unshaped__k_panel,
    ddqn_reduces_jens_uniformly_across_k_at_fr_high_gamma,
    linear_fa_caps_type_1_across_envs__null_panel,
    shaping_decouples_outcome_benefit__fr_shaped_fa_x_gamma_panel,
)


EXPECTED: ClusterVerdict = ClusterVerdict.UNDERPOWERED


BLOCKED_ON: str | None = (
    'C2 null_ceiling tightened from 0.5 to 0.3 (Cohen "small"). '
    'MetaMaze stratum at d=-0.24 CI=[-0.42, -0.06] straddles -0.3 '
    '→ POWER_INSUFFICIENT. Either (a) densify MM linear cells beyond '
    'n=180 per arm or (b) add new-env strata to dilute MM\'s pull on '
    'the panel\'s overall finding. Sweep pending; see '
    'experiments/configs/fa_linear_extra_envs.yaml.'
)


BRIDGES: tuple[Bridge, ...] = (
    ddqn_reduces_jens_uniformly_across_k_at_fr_high_gamma,
    linear_fa_caps_type_1_across_envs__null_panel,
    ddqn_helps_outcome_at_fr_g999_mlp_unshaped__k_panel,
    shaping_decouples_outcome_benefit__fr_shaped_fa_x_gamma_panel,
)
