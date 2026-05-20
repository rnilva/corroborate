"""Pre-registered Finding: DDQN's outcome rescue at γ=0.999 sparse-reward
envs is mediated by a SHIFT in trajectory progress apportionment —
DDQN cells have a higher fraction of trajectory progress attributable
to policy growth (MC) vs bias-chain growth (Q) than vanilla cells.

Two bridges:
1. `ddqn_increases_policy_growth_fraction__fr_g999` — at FR γ=0.999 ×
   canonical (phase-transition rescue regime). Empirical preview:
   vanilla ~0.00, DDQN ~0.89. d very large.
2. `ddqn_increases_policy_growth_fraction__si_g999` — at SI γ=0.999 ×
   canonical (gradual rescue regime). Empirical preview: vanilla
   ~0.07, DDQN ~0.24. d moderate.

Together: scope-cluster test of "DDQN shifts trajectory apportionment
toward policy-side" across both phase-transition (FR) and gradual
(SI) rescue regimes.

Threshold-free reformulation. Earlier version used
`policy_anchors_before_bias` with mc_threshold=0.1 and q_threshold=9.2;
both thresholds were defensible but reviewer-fragile (the q_threshold
required knowing Lemma 2 asymptote × α=0.5 — methodological choice
that reviewers could object to). The growth-fraction
`mc_growth / (mc_growth + q_growth)` has no authored cutpoints — just
the relative magnitude of policy-side vs bias-side trajectory
progress.

Pre-registered: both bridges declare predicted_direction='a_gt_b' and
EXPECTED=SUPPORTED at this commit's source-hash. Framework's drift
detector catches if the materialized verdict diverges.

What this Finding does NOT claim:
- Cross-env meta-comparison via the same fraction (the ratio's scale
  is env-specific). Each bridge is within-env.
- That FR and SI rescue mechanisms are IDENTICAL — they differ in
  magnitude (FR phase-transition vs SI gradual). The shared structure
  is the SIGN: both shift fraction toward policy growth.
- That this mediator captures ALL of DDQN's effect. Per
  `findings_pc_discovery_trajectory_shape` SI's rescue also routes
  through Q-stability (q_traj_autocorr) which is a different axis.

Cross-refs:
- `findings_fr_g999_rescue_unified_narrative` — theoretical framing
- `finding_hasselt_chain_at_fr_g999_unshaped` — intervention legs
- `findings_pc_discovery_trajectory_shape` — PC discovery showing
  FR's direct arm→outcome and SI's mediation via Q-stability
- Measurable `policy_growth_fraction` at `corroborate_rl/dqn/measurables.py`
"""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn_three_conditions.temporal_ordering_fr_g999 import (
    ddqn_increases_policy_growth_fraction__fr_g999,
    ddqn_increases_policy_growth_fraction__si_g999,
)


EXPECTED: ClusterVerdict = ClusterVerdict.SUPPORTED


BLOCKED_ON: str | None = None


BRIDGES: tuple[Bridge, ...] = (
    ddqn_increases_policy_growth_fraction__fr_g999,
    ddqn_increases_policy_growth_fraction__si_g999,
)
