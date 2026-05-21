"""Q-smoothness-channel sign-alignment Finding at canonical γ=0.999.

**Pre-registered claim REFUTED on 8-env panel** (2026-05-21).
The 4-env eyeball panel (FR/SI/Asterix/Breakout γ=0.999) used
to motivate the claim showed 3/4 same-aligned, but the full
canonical pool gives 4/8 — exactly random (binomial p=0.637).
The smoothness sign-alignment does NOT generalize cluster-wide;
in contrast, the loop-channel sign-alignment HELDs 7/8 at
p=0.035 (`finding_loop_channel_consistency_g0999`).

Per-env breakdown (DDQN - vanilla, late window):
  Asterix-MinAtar:        d_grad=−2.13 d_eval=−0.80 ✓ same
  Breakout-MinAtar:       d_grad=−0.46 d_eval=+0.66 ✗ opposite
  FourRooms-misc:         d_grad=+2.05 d_eval=+3.76 ✓ same
  Freeway-MinAtar:        d_grad=+0.54 d_eval=+0.10 ✓ same
  LunarLander-v2-jax:     d_grad=−0.08 d_eval=+0.22 ✗ opposite
  MountainCar-v0:         d_grad=+0.11 d_eval=−0.32 ✗ opposite
  Snake-jumanji:          d_grad=−0.92 d_eval=+0.63 ✗ opposite
  SpaceInvaders-MinAtar:  d_grad=+0.67 d_eval=+2.16 ✓ same

The structural misalignments at Breakout/Snake (both DDQN
CUTS smoothness AND HELPS outcome) directly contradict the
"smoothness preservation tracks outcome" hypothesis. Snake +
Breakout share: DDQN's clip reduces smoothness modestly (d~−0.5
to −0.9) while still helping outcome via other channels (loop
reduction at these envs is the dominant mediator per the loop
bridge's HELD verdict at the same scope).

What this Finding contributes:
- **Negative result corroborating that smoothness is NOT a
  universal mediator** at canonical γ=0.999. Sibling bridges in
  `pc_cross_env_smoothness` already established that
  smoothness's structural role is env-specific (independent
  channel at Asterix; jens-shadow at Breakout; structurally
  inactive at Freeway). This Finding closes the SUFFICIENT-
  condition question at the cluster level.
- The Asterix γ=0.999 smoothness-harm chain remains real (see
  `finding_asterix_g999_smoothness_harm_chain`); it's just
  not the cross-env regularity I'd hoped.

Pre-registration discipline note: the framework's verdict
machinery flagged the refutation immediately on the n=60-per-env
panel — the 4-env eyeball that motivated the claim was
overweighted on FR/SI (both same-aligned) and underweighted
on the 4 envs that misalign. Authoring this Finding committed
the claim BEFORE inspecting the 8-env adjudicating data; the
refutation is honest.

Companion to:
  - `finding_loop_channel_consistency_g0999` (loop channel,
    alignment='opposite' — DDQN's ↓loops aligns with ↑outcome,
    HELD 7/8 at p=0.035)
  - `finding_asterix_g999_smoothness_harm_chain` (single-env
    smoothness mechanism at Asterix γ=0.999, HELD)
  - `finding_pc_cross_env_smoothness` (per-env PC structural
    role of smoothness across Asterix/Breakout/Freeway)
"""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn_sweeps.smoothness_alignment_consistency import (
    ddqn_outcome_aligns_with_q_smoothness__canonical_g0999,
)


EXPECTED: ClusterVerdict = ClusterVerdict.REFUTED


BLOCKED_ON: str | None = None


BRIDGES: tuple[Bridge, ...] = (
    ddqn_outcome_aligns_with_q_smoothness__canonical_g0999,
)
