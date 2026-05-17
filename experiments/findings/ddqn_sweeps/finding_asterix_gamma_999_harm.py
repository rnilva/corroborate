"""Asterix γ=0.999 harm: DDQN's clip harms outcome at extreme γ on
the canonical Asterix env. Holds across best-burst, late-burst,
and full-AUC outcome metrics.

This is the SUPPORTED leg of the broader σ/jens regime theory
(`findings_sigma_over_jens_regime_discriminator.md`). The cross-
env discriminator part of that theory was REFUTED at canonical
data (see `finding_sigma_over_jens_xenv_predictor_refuted.py`);
the Asterix-specific prediction survived empirical test.

Substantive empirical content. Asterix γ=0.999, k=1, canonical
MinAtar HPs (sync=1000, cap=100k, CNN[16]/FC[128]), 30 seeds per
arm. Scoped to cells where `q_mc_burst_correlation_late >= 0.3`
(vanilla's Q-function is meaningfully coupled to MC, excluding
the regime-C "vanilla Q-collapse" cells).

Per-arm d_out (DDQN − vanilla):
  - best-burst: d = −0.80 (z = −3.1) → CI fully below −0.4 ✓ HELD
  - late-burst: d = −1.07 (z = −4.1) → CI fully below −0.5 ✓ HELD
  - full-AUC:   d = −1.08 (z = −4.2) → CI fully below −0.5 ✓ HELD

Mechanism (per
`findings_sigma_over_jens_regime_discriminator.md`): Asterix at
γ=0.999 has vanilla Q growing monotonically to 674 (Q/MC=74×)
but outcome preserved at 22 (identical to γ=0.95). The argmax
ordering is preserved despite huge |Q| because σ/|Q| ≈ 1%
(uniform overestimation across actions). DDQN's clip
`min(Q_target(s', argmax_online), max_a Q_target(s', a))`
introduces action-asymmetric noise — when online's argmax
differs from target's, the clipped bootstrap uses online's
choice, which the 1/(1−γ) = 1000× factor amplifies into argmax
corruption → outcome harm."""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn_sweeps.sigma_over_jens_regime import (
    ddqn_harms_asterix_gamma_999,
    ddqn_harms_asterix_gamma_999__full_auc,
    ddqn_harms_asterix_gamma_999__late_burst,
)


EXPECTED: ClusterVerdict = ClusterVerdict.SUPPORTED


BLOCKED_ON: str | None = None


BRIDGES: tuple[Bridge, ...] = (
    ddqn_harms_asterix_gamma_999,
    ddqn_harms_asterix_gamma_999__late_burst,
    ddqn_harms_asterix_gamma_999__full_auc,
)
