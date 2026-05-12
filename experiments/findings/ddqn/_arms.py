"""Arm constants: DDQN ↔ vanilla and adaptive-controller swaps.

Defines the typed `Intervention` deltas the framework's `DoEffect`
contrasts compose. Bridges whose source is a measurable (link /
mediator) but whose analyses still need arm pairing import
`DDQN_ARM` / `VANILLA_ARM` as holds_when defaults; the bridge
runner injects them via `bridge.params` → analysis kwargs."""
from __future__ import annotations

from functools import partial

from corroborate.core.intervention import DoEffect, Intervention
from corroborate_rl.dqn.claims.bootstrap import (
    adaptive_dormancy_greedify, bootstrap, double_greedify,
)


DDQN_SWAP = Intervention(
    slot_path='bootstrap',
    replacement=partial(bootstrap, greedification=double_greedify),
)
ADAPTIVE_DQN_FACTOR_0P5_SWAP = Intervention(
    slot_path='bootstrap',
    replacement=partial(
        bootstrap,
        greedification=partial(
            adaptive_dormancy_greedify, sigma_floor_factor=0.5,
        ),
    ),
)

# File-level intervention: do(bootstrap = ddqn) → effect.
INTERVENTION = DoEffect(treatment=(DDQN_SWAP,), baseline=())

# Arm-key strings derived from INTERVENTION. Used by bridges whose
# source is a measurable (the runner only auto-injects arm kwargs
# when source is a DoEffect; measurable-sourced bridges supply
# defaults that flow through bridge.params).
DDQN_ARM = INTERVENTION.treatment_arm_key()
VANILLA_ARM = INTERVENTION.baseline_arm_key()
