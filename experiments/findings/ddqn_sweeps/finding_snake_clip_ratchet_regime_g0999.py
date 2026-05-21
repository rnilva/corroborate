"""Snake γ=0.999 CLIP-RATCHET confirmatory pre-registration.

Pre-registered 2026-05-18 BEFORE the T3a panel-extension sweep
(`g0999_panel_extension_jumanji.yaml`) lands Snake γ=0.999 cells.
The git commit that lands this Finding IS the pre-registration
timestamp.

Predicted verdict: SUPPORTED post-T3a (Snake γ=0.999 cells in
cache, all 3 bridges HELD).

Function. T2a's `finding_snake_clip_ratchet_regime` SUPPORTED the
CLIP-RATCHET regime at γ=0.99, but Bridge 2 was substituted
post-hoc (`q_late_mean` d=+0.20 NS → `q_action_std_late` d=+0.65
sig). THIS Finding lifts the same 3-bridge cluster to γ=0.999 as
a genuinely-pre-registered confirmatory test. The σ_Q-inflation
predicate is now committed to git BEFORE the data exists; the
T3a sweep's verdict on it is a clean prospective test.

Walk-back conditions:
- All 3 HELD at γ=0.999 → SUPPORTED. CLIP-RATCHET regime is
  γ-portable; T2a's post-hoc substitution has confirmatory
  evidence; 4-bin classifier validated at the env level.
- 1 bridge DRIFTs → cluster UNDERPOWERED; partial regime
  γ-portability. Specific drift pattern informs the walk-back
  shape.
- 2+ bridges DRIFT → REFUTED. CLIP-RATCHET regime is γ=0.99-
  specific; Snake reclassifies into one of the existing 3 bins
  at γ=0.999 (likely Q-EXPLODED).
"""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn_sweeps.snake_clip_ratchet_regime_g0999 import (
    snake_g0999_arm_drives_temporal_cv,
    snake_g0999_arm_inflates_action_std,
    snake_g0999_arm_outcome_marginal_independent,
)


EXPECTED: ClusterVerdict = ClusterVerdict.REFUTED


BLOCKED_ON: str | None = None

# Resolution (2026-05-21): T3a Snake-jumanji γ=0.999 cells
# ingested. Pre-registered prediction was SUPPORTED (replicating
# the γ=0.99 CLIP-RATCHET regime at γ=0.999). Empirical: REFUTED.
#
#   * arm_inflates_action_std: NO_EFFECT (SIGN_FLIP). DDQN does
#     NOT inflate σ_Q at Snake γ=0.999 — opposite direction.
#   * arm_drives_temporal_cv: POWER_INSUFFICIENT.
#   * arm_outcome_marginal_independent: POWER_INSUFFICIENT
#     (DDQN HELPS at Snake γ=0.999; arm ⫫ outcome breaks).
#
# Substantive walk-back (memory
# `findings_snake_g0999_ddqn_helps_via_bias_clip`): Snake γ=0.999
# routes through canonical Hasselt bias-clip (jensen_gap d=−1.25,
# DDQN cuts bias 40% → eval_best d=+0.63), NOT through the
# γ=0.99 CLIP-RATCHET regime. The CLIP-RATCHET regime is
# γ=0.99-specific at Snake; the γ→0.999 transition flips Snake
# out of clip-ratchet and into bias-dominated, where DDQN's clip
# becomes helpful rather than destabilising. This is the
# regime-transition predicted by Theorem 1 (Λ_m gate crossing
# as γ→1) — Snake's Λ_m at γ=0.999 enters the bias-dominated
# regime.


BRIDGES: tuple[Bridge, ...] = (
    snake_g0999_arm_drives_temporal_cv,
    snake_g0999_arm_inflates_action_std,
    snake_g0999_arm_outcome_marginal_independent,
)
