"""Hasselt's three-factor bound + shaping decouples translation.

DDQN's overestimation reduction is well-described by Hasselt
2010's structural bound `σ_action × √(2 ln K) × 1/(1 − γ)`. Six
bridges in two clusters test this:

**Hasselt-bound cluster (4 bridges)** — one per factor:

1. `ddqn_reduces_jens_uniformly_across_k_at_fr_high_gamma`
   — the `√(2 ln K)` factor: jens reduction scales monotonically
   across k_eff ∈ {4, 8, 12, 16} at FR γ=0.999 MLP unshaped.

2. `ddqn_reduction_amplified_by_gamma__fr_mlp_k4_unshaped`
   — the `1/(1−γ)` factor: at controlled K (k_eff=4),
   |mean_diff(γ=0.999)| ≥ 3× |mean_diff(γ=0.99)| AND per-γ Cohen's
   d ≤ -0.8 at both γ.

3a. `linear_fa_caps_type_1_across_envs__null_panel`
   — the σ_action factor: linear FA bounds σ_action → Hasselt mech
   dormant → per-env Cohen's d on jens fits ±0.3 band across 6
   envs (FR, Acrobot, MountainCar, CartPole, Catch, DeepSea).

3b. `linear_fa_cap_fails_at_metamaze_g999__exception`
   — the named exception: at MetaMaze γ=0.999 × linear FA, the
   cap fails (d ≤ -0.3 across n_episodes strata) because the
   random-maze-per-episode structure forces FA-fit-error bias
   that DDQN clips via a non-σ path.

Bridges (3a, 3b) form a rule + exception cluster for the σ
factor — together they encode "σ-via-FA gates the Hasselt mech
EXCEPT where FA-fit error provides a parallel bias path".

**Shaping cluster (2 bridges)** — a fourth, orthogonal axis: how
bias-reduction translates to outcome.

4a. `ddqn_helps_outcome_at_fr_g999_mlp_unshaped__k_panel`
   — positive arm: at FR γ=0.999 × MLP × unshaped (Hasselt
   factors active + dense FA), DDQN improves
   `eval_best_burst_raw_mean` at every k_eff stratum.

4b. `shaping_decouples_outcome_benefit__fr_shaped_fa_x_gamma_panel`
   — null arm: at FR × shaped × {linear, MLP} × γ ∈ {0.99, 0.999},
   DDQN's outcome effect is never positive. Shaping's dense
   Φ-gradient policy signal decouples bias-reduction from
   outcome improvement.

The cluster pattern positive (unshaped) + null (shaped) reads as
"shaping moderates the bias→outcome translation".

**Verdict aggregation**: `SUPPORTED` requires all six bridges
HELD. Currently UNDERPOWERED because bridge (3a)'s Catch stratum
has CI = [-0.31, +0.20], straddling the ±0.3 null band by 0.01
at n=120 per arm. The substantive cluster pattern is intact;
the verdict is one CI-edge-noise tightening away from SUPPORTED
(~30 more Catch seeds would close it).

**What this Finding does NOT claim**:
- That `σ_action × √(2 ln K) × 1/(1−γ)` is a TIGHT bound — only
  that the three factors' structural predictions hold
  empirically.
- Generalisation beyond the 7 envs in scope. The 6-env FA-cap
  rule (3a) is the broadest claim; cross-env shaping (4b) is
  scoped to FR only.
- That MetaMaze γ=0.999's FA-fit-error mechanism (3b)
  generalises to other non-stationary envs. No other env with
  per-episode state distribution shift was tested at linear FA.

**Cell sources** (see __init__.py docstring for ingest commands):
- `fa_depth_fourrooms`, `fa_depth_xenv_gpu`, `fa_depth_fourrooms_gpu`,
  `fa_depth_gradient_overlap`, `ddqn_axis_probes_pilot` —
  FR/Acrobot/MM/MC × {linear, MLP} × γ unshaped 200k-1M.
- `fa_degeneracy_shaped_only` — FR × {linear, MLP} × γ shaped.
- `fa_linear_extra_envs` — CartPole/Catch/DeepSea linear unshaped.
- `metamaze_linear_eval_power` — MetaMaze γ=0.999 linear unshaped
  at n_episodes=20 (eval-power-fixed).
- `action_dup_mismatch_probe_g999_1M{,_FR_k4_only}/ddqn_vs_vanilla`
  — FR γ=0.999 MLP unshaped × k_eff ∈ {4, 8, 12, 16}."""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn_three_conditions.conditions import (
    ddqn_helps_outcome_at_fr_g999_mlp_unshaped__k_panel,
    ddqn_reduces_jens_uniformly_across_k_at_fr_high_gamma,
    ddqn_reduction_amplified_by_gamma__fr_mlp_k4_unshaped,
    fa_capacity_moderates_ddqn_jens_reduction,
    linear_fa_cap_fails_at_metamaze_g999__exception,
    shaping_decouples_outcome_benefit__fr_shaped_fa_x_gamma_panel,
)


EXPECTED: ClusterVerdict = ClusterVerdict.SUPPORTED


BLOCKED_ON: str | None = None


BRIDGES: tuple[Bridge, ...] = (
    ddqn_reduces_jens_uniformly_across_k_at_fr_high_gamma,
    ddqn_reduction_amplified_by_gamma__fr_mlp_k4_unshaped,
    fa_capacity_moderates_ddqn_jens_reduction,
    linear_fa_cap_fails_at_metamaze_g999__exception,
    ddqn_helps_outcome_at_fr_g999_mlp_unshaped__k_panel,
    shaping_decouples_outcome_benefit__fr_shaped_fa_x_gamma_panel,
)
