"""γ-sweep bridges from within_env — exposed under ddqn_sweeps so
they can fire on cells with γ ∈ {0.995, 0.999} (excluded by ddqn
canonical scope which pins γ=0.99).

The bridge functions still live in
`experiments.findings.ddqn.within_env`; only their exposure under
`BRIDGES` moves here. The metamaze γ amplification finding moves
to `experiments.findings.ddqn_sweeps.finding_metamaze_gamma_amplification`
(new file)."""
from __future__ import annotations

from experiments.findings.ddqn.within_env import (
    ddqn_benefit_scales_with_effective_horizon__fourrooms,
    metamaze_link_steeper_at_high_gamma,
)


BRIDGES = (
    ddqn_benefit_scales_with_effective_horizon__fourrooms,
    metamaze_link_steeper_at_high_gamma,
)
