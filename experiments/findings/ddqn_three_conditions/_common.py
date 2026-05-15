"""Shared constants for the three-conditions hypothesis.

CLAIM is the outermost claim — the substrate's `dqn` Free Claim,
shared with the canonical `experiments.findings.ddqn` module.
The hypothesis tests THREE necessary conditions for DDQN's
outcome benefit to translate from mech (Δ_jensen_gap < 0) to
outcome (Δ_eval > 0). The conditions are jointly required —
absence of any one suffices to make DDQN's mechanism inactive
or harmful."""
from __future__ import annotations

from corroborate_rl.dqn.dqn import dqn


CLAIM = dqn


# Arm key for DDQN vs vanilla — matches the canonical convention
# from `experiments.findings.ddqn._arms`. The arm-tagging is set
# by `dispatch_sweep`'s `combined_arm_key` on the intervention
# tuple. DDQN appends a single `bootstrap=partial(...)` slot.
DDQN_ARM = (
    'bootstrap=partial(Claim:bootstrap;greedification=Claim:double_greedify)'
)
VANILLA_ARM = 'baseline'
