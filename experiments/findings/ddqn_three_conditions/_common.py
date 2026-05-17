"""Shared constants for the ddqn_three_conditions hypothesis.

CLAIM is the outermost claim — the substrate's `dqn` Free Claim,
shared with the canonical `experiments.findings.ddqn` module.
The hypothesis tests Hasselt 2010's three-factor bound
`bias ≤ σ_action × √(2 ln K) × 1/(1−γ)` factor-by-factor (the
`finding_hasselt_bound` cluster) and how potential-based shaping
decouples bias-reduction from outcome (the
`finding_shaping_decouples` cluster)."""
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
