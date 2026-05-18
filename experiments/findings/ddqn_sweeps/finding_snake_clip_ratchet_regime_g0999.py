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


EXPECTED: ClusterVerdict = ClusterVerdict.EMPTY_EXTENT


BLOCKED_ON: str | None = (
    'PRE-REGISTERED DRIFT 2026-05-18 — predicted post-ingest '
    'EXPECTED=SUPPORTED. Awaiting T3a panel-extension sweep '
    '(`experiments/configs/g0999_panel_extension_jumanji.yaml`) to '
    'land Snake-jumanji γ=0.999 canonical-HP cells (n=30, 1M '
    'steps, CNN HPs). All 3 bridges fire POWER_INSUFFICIENT until '
    'the sweep ingests. Predicted post-ingest: SUPPORTED — same '
    'CLIP-RATCHET signatures (σ_Q d ≥ +0.4, PC `arm — '
    'q_max_temporal_cv` edge, marginal `arm ⫫ outcome`) replicate '
    'at γ=0.999. Walk-back path: see module docstring. When the '
    'sweep lands, this BLOCKED_ON gets cleared and EXPECTED is '
    'lifted to whatever the empirical verdict is; the renderer '
    'surfaces `← DRIFT` if it differs from this pre-registered '
    'prediction.'
)


BRIDGES: tuple[Bridge, ...] = (
    snake_g0999_arm_drives_temporal_cv,
    snake_g0999_arm_inflates_action_std,
    snake_g0999_arm_outcome_marginal_independent,
)
