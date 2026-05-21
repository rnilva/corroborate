"""σ_VAN/jens_VAN at γ=0.999 is NOT a universal cross-env predictor
of DDQN's outcome sign.

This Finding records the REFUTATION of the universal-discriminator
claim. The Asterix-specific harm prediction (which the same
theory implies) is in a sibling Finding,
`finding_asterix_gamma_999_harm`, where it lands SUPPORTED.

History. After observing the Asterix γ=0.999 sign-flip
(`findings_minatar_gamma_sweep_first_results`), I proposed that
σ_VAN/jens_VAN at γ=0.999 should discriminate Type A (uniform
overestimation → DDQN harms) from Type B (asymmetric → DDQN
helps) across envs. The cross-env Spearman ρ between σ/jens and
per-env Cohen's d was predicted to be positive.

After refining the scope with `q_mc_burst_correlation_late >= 0.3`
(restricting to cells where vanilla's Q is meaningfully coupled
to MC — i.e., excluding regime-C "vanilla Q-explosion +
MC-decoupled" cases), the cross-env Spearman ρ on the 5
surviving envs is **+0.10, p=0.87** — null. The discriminator
is REFUTED at this panel.

Per-env Cohen's d after learnability scope (best-burst):

  env          | σ/jens  | d_out  | predicted | actual
  Acrobot-v1   | 0.0037  | −0.13  | Type A     | weak harm ≈
  FR-misc      | 0.0052  | +0.91  | Type A     | **helps strongly** ✗
  Asterix-MA   | 0.0155  | −1.35  | Type A     | strong harm ✓
  MetaMaze     | 0.0166  | −0.53? | Type A     | helps ✗
  Breakout-MA  | 0.0609  | +0.40  | Type B     | mild help ≈
  MountainCar  | 0.0014  | (drops)| —          | —

Two failures of the theory:
  - FourRooms γ=0.999 shaped cells (the ones that pass r ≥ 0.3)
    fall in regime B (FA-truncation-rescue) despite low σ/jens.
  - MetaMaze γ=0.999 helps DDQN at moderate σ/jens, contradicting
    the Type-A prediction.

The σ/jens predictor is 1-dimensional. A 2D classifier (σ/jens ×
env-class, where env-class captures whether reward-shaping /
exploration regime puts the env into FA-rescue territory) might
work — left as future work.

This Finding fires three cross-env bridges (best, late_burst,
full_auc outcomes); all predict ρ > 0 with sign=+1. The
empirical ρ is consistently near 0, so all three bridges
return NO_EFFECT (NULL_EFFECT). EXPECTED is REFUTED to make the
walk-back honest. The substrate-level claim (σ/jens-as-discriminator
in general) is REFUTED at canonical γ × k=1 data.

When the running k=2/k=4 sweeps land (next 3 days), the cross-env
n_strata grows (Freeway + SI both come online); the test may
gain power to detect a non-zero ρ if one exists. The empirical
state will be re-evaluated then.
"""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn_sweeps.sigma_over_jens_regime import (
    ddqn_outcome_scales_with_sigma_over_jens__gamma_999_xenv,
    ddqn_outcome_scales_with_sigma_over_jens__gamma_999_xenv__full_auc,
    ddqn_outcome_scales_with_sigma_over_jens__gamma_999_xenv__late_burst,
)


EXPECTED: ClusterVerdict = ClusterVerdict.REFUTED


BLOCKED_ON: str | None = None

# Resolution (2026-05-21): σ_VAN/jens_VAN converted from hardcoded
# `_SIGMA_OVER_JENS_PER_ENV` constant to `DerivedCovariateSpec`
# reading the new `sigma_over_jens_late` measurable. On canonical
# pool: cross-env ρ is null at all 3 outcome metrics (best-burst
# p=0.75, late_burst p=0.87, full_auc p=0.75) — none distinguishable
# from zero. The σ/jens-as-universal-discriminator claim is REFUTED
# substantively, not just due to canonical-cohort power loss. The
# prior hardcoded snapshot had Acrobot 15× under and MountainCar
# 13× under the canonical pool's σ/jens values; that mismatch had
# inflated the apparent cross-env signal. See memory
# `findings_sigma_lambda_a_hp_artifact_walkback` for the meta-
# pattern.


BRIDGES: tuple[Bridge, ...] = (
    ddqn_outcome_scales_with_sigma_over_jens__gamma_999_xenv,
    ddqn_outcome_scales_with_sigma_over_jens__gamma_999_xenv__late_burst,
    ddqn_outcome_scales_with_sigma_over_jens__gamma_999_xenv__full_auc,
)
