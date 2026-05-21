"""Loop-channel sign-alignment Finding at canonical γ=0.999.

Pre-registers the loop-reduction-channel sign-alignment claim
from `REPORT_loop_hypothesis_synthesis.md` §2.1 as a framework-
typed bridge. The 5-env eyeball panel in the report shows 5/5
envs sign-aligned (DDQN's outcome direction opposes its
revisit-rate direction). At the n=10 canonical pool, the
binomial sign-test gates HELD at ≥9/10 alignment (p ≤ 0.011).

Pre-registration: BLOCKED_ON the
`state_repeat_rate_window64_late` measurable backfill. The
bridge fires POWER_INSUFFICIENT today (no canonical-pool cells
have the measurable populated). When the cross-corpus trace
restore + recompute lands (similar discipline to the dormancy
backfill in `findings_cross_env_jensen_dormancy_map_g0999`),
the bridge auto-fires and the verdict is the post-hoc verdict.

Companion to:
  - `finding_jens_reduction_consistency_g0999` (mech-channel
    consistency: DDQN reduces jens at every env)
  - `finding_dormancy_diagnostic_acrobot_g0999` (per-env
    dormancy diagnostic)

These three Findings together would form the
two-channel-decomposition cluster the report §4 describes —
each channel pre-registered as a typed bridge at the
cross-env-consistency claim shape.
"""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn_sweeps.loop_channel_consistency import (
    ddqn_outcome_opposes_loop_rate__canonical_g0999,
)


EXPECTED: ClusterVerdict = ClusterVerdict.UNDERPOWERED


BLOCKED_ON: str | None = (
    'POST-DATA 2026-05-21. Backfill landed: 8 of 10 canonical '
    'corpora have `state_repeat_rate_window64_late` populated '
    '(Snake skipped — per-cell trace files only, no merged '
    'traces.parquet; LL not yet in cache — separate ingest). '
    'Bridge fires POWER_INSUFFICIENT p=0.145 (6/8 envs '
    'opposite-aligned). Per-env breakdown:\n'
    '  Asterix:  d_x=−0.80 d_y=+1.92 ✓ opposite\n'
    '  Breakout: d_x=+0.66 d_y=−0.29 ✓ opposite\n'
    '  FR:       d_x=+3.76 d_y=−4.92 ✓ opposite\n'
    '  Freeway:  d_x=+0.10 d_y=−0.61 ✓ opposite\n'
    '  MC:       d_x=−0.32 d_y=+0.19 ✓ opposite\n'
    '  SI:       d_x=+2.16 d_y=−3.45 ✓ opposite\n'
    '  Acrobot:  d_x=−0.01 d_y=−0.01   degenerate (Hasselt-'
    'dormant — 18% of cells dormant per `findings_acrobot_'
    'dormancy_mech_walkback`; both arms ~identical)\n'
    '  MetaMaze: d_x=−0.08 d_y=±0.00   degenerate (rep_ea=1.0 '
    'saturated for both arms — agent revisits every state '
    'inside 64-step window)\n'
    'The 6 non-degenerate envs at γ=0.999 are 6/6 opposite-'
    'aligned; sign-test power is killed by 2 strata where the '
    'intervention has no measurable effect on revisit rate. '
    'Walk-back from pre-reg HELD-prediction: cross-env power '
    'gate sensitive to degenerate strata, even when the '
    'substantive claim survives at non-degenerate scope. '
    'Resolution paths: (a) author a sibling bridge with '
    '`null_floor_y` set to a principled threshold so the '
    'degenerate strata drop out — this asks the gated '
    'question "where DDQN materially shifts revisit-rate, '
    'does outcome go opposite?"; (b) interpret PI honestly '
    'and let the per-env evidence stand. Resolution (a) '
    'requires a fresh pre-registered bridge to avoid post-hoc '
    'parameter tuning.'
)


BRIDGES: tuple[Bridge, ...] = (
    ddqn_outcome_opposes_loop_rate__canonical_g0999,
)
