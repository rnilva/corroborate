"""Outcome is NOT consistently improved by paired_dqn — UNDERPOWERED at
n=4 (the scope-dependence the per-env spectrum shows can't yet be
certified).

The per-env greedy late-eval effect spans free lunch (SpaceInvaders
d=+5.7, Asterix +1.7) through mild help (Freeway +0.6) to HARM
(Breakout -2.9). Only 3/4 envs improve (Breakout flips), so the
"consistent improvement" sign-test is p=0.31 -> POWER_INSUFFICIENT, and
the DL magnitude pool's PI brackets zero (I²≈0.98). Contrast with the
mechanism's 4/4 consistency: identical mechanism (jensen_gap->0 at all
four) but inconsistent outcome — the mechanism ↛ outcome dissociation,
visible per-env but not terminal at n=4. EXPECTED pinned to the honest
underpowered state; BLOCKED_ON names the gap.
"""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn2010_xenv.bridges import (
    paired_improves_outcome_consistently__minatar4,
)

EXPECTED: ClusterVerdict = ClusterVerdict.UNDERPOWERED
BLOCKED_ON: str | None = (
    'Outcome improves at 3/4 envs (Breakout harms, d=-2.9); consistency '
    'p=0.31 at n=4 and the pooled-magnitude PI brackets zero (I2=0.98), '
    'so scope-dependence is visible per-env (+5.7 SI .. -2.9 Breakout) '
    'but underpowered to certify. Needs more envs.'
)
BRIDGES: tuple[Bridge, ...] = (paired_improves_outcome_consistently__minatar4,)
