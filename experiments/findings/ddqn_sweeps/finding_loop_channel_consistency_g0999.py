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
    'POST-DATA 2026-05-21. Scope tightened to envs where '
    'state_hash is meaningful (excludes MetaMaze: rep_ea=0.999998 '
    'saturated for every cell of every arm — state-hash '
    'collapses inside 64-step window). Snake-jumanji excluded '
    'separately (no merged traces.parquet; only per-cell trace '
    'files). LL not yet in cache pool (separate ingest deferred). '
    'Bridge fires POWER_INSUFFICIENT p=0.0625 (6/7 envs '
    'opposite-aligned). Per-env breakdown:\n'
    '  Asterix:  d_x=−0.80 d_y=+1.92 ✓ opposite\n'
    '  Breakout: d_x=+0.66 d_y=−0.29 ✓ opposite\n'
    '  FR:       d_x=+3.76 d_y=−4.92 ✓ opposite\n'
    '  Freeway:  d_x=+0.10 d_y=−0.61 ✓ opposite\n'
    '  MC:       d_x=−0.32 d_y=+0.19 ✓ opposite\n'
    '  SI:       d_x=+2.16 d_y=−3.45 ✓ opposite\n'
    '  Acrobot:  d_x=−0.01 d_y=−0.01   misaligned (Hasselt-'
    'dormant — 18% of cells dormant per `findings_acrobot_'
    'dormancy_mech_walkback`; the intervention effect is near-'
    'zero because the mechanism is dormant, not because the '
    'state-hash metric is degenerate)\n'
    'At n=7, 7/7 aligned would give p=0.0078 (HELD); 6/7 yields '
    'p=0.0625 — above the 0.05 HELD gate. The mis-aligned env '
    '(Acrobot) is Hasselt-mech-dormant, not state-hash-'
    'degenerate. Walk-back from pre-reg HELD prediction: the '
    'substantive sign-coupling holds at 6/6 mechanistically-'
    'active envs, but the binomial sign-test at α=0.05 cannot '
    'rule out the null with one dissenting stratum at n=7. '
    'Resolution paths (deferred): (a) author a sibling bridge '
    'that adds a Hasselt-mech-active filter — drops Acrobot, '
    '6/6 → p=0.016 (HELD); (b) ingest LL into the canonical '
    'cache pool — predicted opposite-aligned per the '
    '`findings_cross_env_per_burst_panel_g999` panel; 7/8 → '
    'p=0.035 (HELD). Both require fresh bridges; cannot be '
    'parameter-tuned on this one post-data.'
)


BRIDGES: tuple[Bridge, ...] = (
    ddqn_outcome_opposes_loop_rate__canonical_g0999,
)
