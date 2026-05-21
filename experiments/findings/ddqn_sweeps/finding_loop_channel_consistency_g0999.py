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


EXPECTED: ClusterVerdict = ClusterVerdict.EMPTY_EXTENT


BLOCKED_ON: str | None = (
    'PRE-REGISTRATION 2026-05-21. The measurable '
    '`state_repeat_rate_window64_late` is not yet backfilled '
    'across the canonical γ=0.999 pool (reads '
    '`state_hash_per_step` trace col; 2 of 10 canonical corpora '
    'have traces locally — Freeway, SI). The bridge fires PI '
    'until the cross-corpus trace restore + recompute lands '
    '(estimated ~3.5h GPU + analyst time, similar discipline '
    'to the dormancy backfill in `findings_cross_env_jensen_'
    'dormancy_map_g0999`). REPORT_loop_hypothesis_synthesis.md '
    "§2.1's 5-env eyeball table reports 5/5 sign-aligned at "
    'opposite direction (FR +0.96/−0.07; SI +14.5/−0.06; '
    'Breakout +2.88/−0.01; LL +3.99/−0.004; Asterix −2.45/+0.029). '
    'At the n=10 canonical pool the predicted verdict is HELD '
    'iff ≥9/10 envs align (p ≤ 0.011). The report\'s "loop-'
    'reduction channel" claim is fundamentally a cross-env '
    'consistency claim about the sign-coupling of arm-effect-'
    'on-outcome and arm-effect-on-revisit-rate — this bridge '
    'is the typed pre-registration.'
)


BRIDGES: tuple[Bridge, ...] = (
    ddqn_outcome_opposes_loop_rate__canonical_g0999,
)
