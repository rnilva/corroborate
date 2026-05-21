"""Loop-channel sign-alignment Finding at canonical γ=0.999.

Pre-registers the loop-reduction-channel sign-alignment claim
from `REPORT_loop_hypothesis_synthesis.md` §2.1 as a framework-
typed bridge. The 5-env eyeball panel in the report showed 5/5
envs sign-aligned (DDQN's outcome direction opposes its
revisit-rate direction). The framework-typed bridge fires HELD
at 7/8 envs opposite-aligned (p=0.035), within the
state-hash-meaningful scope (MetaMaze + Snake excluded for
state-hash degeneracy).

Post-data verdict (2026-05-21): 7/8 envs at γ=0.999 align in
the predicted opposite direction. Per-env breakdown:
  Asterix:  d_x=−0.80 d_y=+1.92 ✓ opposite
  Breakout: d_x=+0.66 d_y=−0.29 ✓ opposite
  FR:       d_x=+3.76 d_y=−4.92 ✓ opposite
  Freeway:  d_x=+0.10 d_y=−0.61 ✓ opposite
  LL:       d_x=+0.21 d_y=−0.11 ✓ opposite
  MC:       d_x=−0.32 d_y=+0.19 ✓ opposite
  SI:       d_x=+2.16 d_y=−3.45 ✓ opposite
  Acrobot:  d_x=−0.01 d_y=−0.01   misaligned (Hasselt-mech-
            dormant 18%; intervention effect near-zero because
            the mechanism is dormant, NOT because state-hash
            is degenerate — see `findings_acrobot_dormancy_
            mech_walkback`)

Companion to:
  - `finding_jens_reduction_consistency_g0999` (mech-channel
    consistency: DDQN reduces jens at every env)
  - `finding_dormancy_diagnostic_acrobot_g0999` (per-env
    dormancy diagnostic)

These three Findings together form the two-channel-
decomposition cluster the report §4 describes — each channel
pre-registered as a typed bridge at the cross-env-consistency
claim shape, all three HELD on the canonical γ=0.999 pool.
"""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn_sweeps.loop_channel_consistency import (
    ddqn_outcome_opposes_loop_rate__canonical_g0999,
)


EXPECTED: ClusterVerdict = ClusterVerdict.SUPPORTED


BLOCKED_ON: str | None = None


BRIDGES: tuple[Bridge, ...] = (
    ddqn_outcome_opposes_loop_rate__canonical_g0999,
)
