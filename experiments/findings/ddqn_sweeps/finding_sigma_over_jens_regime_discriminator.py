"""σ_VAN/jens_VAN at γ=0.999, conditional on Q-MC coupling, is a
regime discriminator for DDQN's outcome sign.

REFINEMENT (2026-05-17): the original σ/jens-only discriminator
fails on FR γ=0.999 — a "vanilla Q-explodes-and-decouples-from-MC"
regime where σ/jens looks low (uniform overestimation) but DDQN
RESCUES (not harms). The fix: scope-restrict every bridge in this
Finding to cells where `q_mc_burst_correlation_late >= 0.3` —
Q is at least modestly coupled to MC. Cells where Q has decoupled
from MC (regime C) are out of scope of the σ/jens theory.

This Finding aggregates nine bridges (3 hypothesis shapes × 3
outcome metrics) that operationalize the theory in
`findings_sigma_over_jens_regime_discriminator.md`:

Hypothesis shapes:
  (A) cross-env: Spearman ρ across envs between (σ_VAN/jens_VAN at
      γ=0.999) and per-env Cohen's d on outcome. ρ > 0 predicted.
  (B) single-env Type A learnable: Asterix γ=0.999 d_out CI fully
      below the harm floor (DDQN harms).
  (C) single-env Type B / FA-truncation: Breakout γ=0.999 d_out
      CI fully above the help floor (DDQN helps).

Outcome metrics:
  1. `eval_best_burst_raw_mean` — peak burst (best-of-training).
  2. `eval_late_burst_raw_mean` — last-25% burst average. Tests
     "Q-explosion bites where Q has grown most". Asterix d_out
     sharpens from −0.80 (best) to −1.07 (late).
  3. `eval_full_auc_raw_mean` — trajectory-averaged. Less
     sensitive to timing artifacts.

`composed_verdict` is AND-aggregate: SUPPORTED iff all 9 HELD;
REFUTED if any REFUTES. Multi-metric coverage hardens the claim
— if the theory is right, the regime classification should HELD
at multiple outcome shapes, not just best-burst.

Current empirical snapshot (k=1 sweep complete for Asterix +
Breakout γ=0.999; pending for Freeway + SI):
  best-burst: Asterix d=−0.80, Breakout d=+0.66, cross-env ρ trending +0.61
  late-burst: Asterix d=−1.07, Breakout d=+0.67 (Asterix sharpens)
  full-AUC:   Asterix d=−1.08, Breakout d=+0.42 (Breakout softens)

The per-burst metrics differ in their predicted ordering across
envs — late-burst should give the SHARPEST Type-A harm (Asterix),
while full-AUC averages and may soften it. The Finding tests
whether the theory holds at multiple operationalizations.

Setting EXPECTED to UNDERPOWERED + BLOCKED_ON to capture the
n_strata=6 + per-env n=30 power floors. Once the running γ × k
sweeps land (~3 days from now), each bridge gets more cells +
the cross-env bridges get more strata; this finding's verdict
should flip to SUPPORTED.
"""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn_sweeps.sigma_over_jens_regime import (
    ddqn_harms_asterix_gamma_999,
    ddqn_harms_asterix_gamma_999__full_auc,
    ddqn_harms_asterix_gamma_999__late_burst,
    ddqn_helps_breakout_gamma_999,
    ddqn_helps_breakout_gamma_999__full_auc,
    ddqn_helps_breakout_gamma_999__late_burst,
    ddqn_outcome_scales_with_sigma_over_jens__gamma_999_xenv,
    ddqn_outcome_scales_with_sigma_over_jens__gamma_999_xenv__full_auc,
    ddqn_outcome_scales_with_sigma_over_jens__gamma_999_xenv__late_burst,
)


EXPECTED: ClusterVerdict = ClusterVerdict.UNDERPOWERED


BLOCKED_ON: str | None = (
    'n_strata=6 envs with γ=0.999 data; cross-env Spearman ρ=+0.61 '
    'trending at p=0.15 with n=7. Per-env CIs (Asterix d_out=-0.80 '
    'z=-3.1 best / -1.07 z=-4.1 late / -1.08 z=-4.2 full_auc; '
    'Breakout d_out=+0.66 z=+2.6 best / +0.67 z=+2.6 late / +0.42 '
    'z=+1.6 full_auc) approach but do not yet uniformly exceed the '
    '|0.4–0.5| floors at n=30. Once minatar_gamma_sweep_k1 lands '
    'Freeway + SI γ=0.999 (n_strata → 8) and k=2/k=4 sweeps '
    'amplify per √(2 ln K), the cross-env bridges should HELD at '
    'ρ ≥ 0.5 p ≤ 0.10 and per-env bridges should flip HELD on '
    'stronger d-magnitudes.'
)


BRIDGES: tuple[Bridge, ...] = (
    ddqn_outcome_scales_with_sigma_over_jens__gamma_999_xenv,
    ddqn_outcome_scales_with_sigma_over_jens__gamma_999_xenv__late_burst,
    ddqn_outcome_scales_with_sigma_over_jens__gamma_999_xenv__full_auc,
    ddqn_harms_asterix_gamma_999,
    ddqn_harms_asterix_gamma_999__late_burst,
    ddqn_harms_asterix_gamma_999__full_auc,
    ddqn_helps_breakout_gamma_999,
    ddqn_helps_breakout_gamma_999__late_burst,
    ddqn_helps_breakout_gamma_999__full_auc,
)
