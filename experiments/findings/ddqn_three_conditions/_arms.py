"""Intervention shape — DDQN vs vanilla.

Reuses the same INTERVENTION shape as the canonical
`experiments.findings.ddqn._arms`. The three-conditions
hypothesis tests the SAME bootstrap intervention; what differs
between bridges is the SCOPE (the env / γ / FA / shaping cells
admitted)."""
from __future__ import annotations

from functools import partial

from corroborate.core.intervention import DoEffect, Intervention
from corroborate_rl.dqn.claims.bootstrap import bootstrap, double_greedify


DDQN_SWAP = Intervention(
    slot_path='bootstrap',
    replacement=partial(bootstrap, greedification=double_greedify),
)

INTERVENTION = DoEffect(arms=((), (DDQN_SWAP,)))
