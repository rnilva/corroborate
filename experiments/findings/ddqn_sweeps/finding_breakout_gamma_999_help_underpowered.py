"""Breakout γ=0.999 help prediction: UNDERPOWERED at n=30 seeds.

The σ/jens theory predicts Type B (FA-truncation rescue) on
Breakout γ=0.999 → DDQN helps with d_out CI fully above the help
floor. Empirical state (k=1, n=30 seeds per arm, learnability-
scoped):

  best-burst: d = +0.66 z = +2.6 → CI ≈ [+0.16, +1.16], help_floor=+0.4 → CI_low < 0.4 → PI
  late-burst: d = +0.67 z = +2.6 → similar shape → PI
  full-AUC:   d = +0.42 z = +1.6 → weaker, help_floor=+0.3 → PI

All three bridges land POWER_INSUFFICIENT. The direction is
correct (positive d_out everywhere), but the CIs at n=30 don't
fully exceed the help floor.

Path to HELD: k=2 / k=4 sweeps amplify the bias-floor by √(2 ln K),
so per Hasselt the d magnitudes should grow ~1.4× at k=4. That
should push CI_low ≥ help_floor and flip these bridges HELD.

Until then, the Finding's EXPECTED is UNDERPOWERED + BLOCKED_ON
documents the path forward.

PRE-REGISTERED DRIFT (2026-05-18): The k=2 sweep is RUNNING at
the time of this commit (background task #80, ~16h elapsed). The
prediction committed at this commit hash: k=2 amplifies d by
√(2 ln 8)/√(2 ln 4) ≈ 1.22× → CI_low approaches but may not
exceed help_floor; k=4 amplifies by 1.41× → CI_low ≥ +0.4 → at
least the best-burst bridge DRIFTs to HELD. If both k=2 and k=4
data land WITHOUT any of the three bridges DRIFTing, the
√(2 ln K) Hasselt-bound amplification claim walks back. If k=4
data lands and Breakout's d magnitudes do NOT grow ~1.4×, the
amplification claim walks back. Verification fold-in:
`findings_drift_verification_k4` memory entry post-ingest."""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn_sweeps.sigma_over_jens_regime import (
    ddqn_helps_breakout_gamma_999,
    ddqn_helps_breakout_gamma_999__full_auc,
    ddqn_helps_breakout_gamma_999__late_burst,
)


EXPECTED: ClusterVerdict = ClusterVerdict.UNDERPOWERED


BLOCKED_ON: str | None = (
    'Breakout γ=0.999 d_out is positive across all 3 outcome '
    'metrics (best +0.66 z+2.6 / late +0.67 z+2.6 / full_auc '
    '+0.42 z+1.6) but CI_low does not fully exceed the help_floor '
    '(+0.4 / +0.4 / +0.3) at n=30. k=2/k=4 sweeps should amplify '
    'the bias-floor via √(2 ln K) → d magnitude grows ~1.4× at '
    'k=4 → CI_low ≥ help_floor → HELD. ETA ~3 days.'
)


BRIDGES: tuple[Bridge, ...] = (
    ddqn_helps_breakout_gamma_999,
    ddqn_helps_breakout_gamma_999__late_burst,
    ddqn_helps_breakout_gamma_999__full_auc,
)
