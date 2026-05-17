"""σ_VAN/jens_VAN at γ=0.999 is a regime discriminator for DDQN's
outcome sign.

This Finding aggregates three bridges that operationalize the
theory in `findings_sigma_over_jens_regime_discriminator.md`:

  Bridge 1 (cross-env):
    `ddqn_outcome_scales_with_sigma_over_jens__gamma_999_xenv`
    Spearman ρ across envs between (σ_VAN/jens_VAN at γ=0.999) and
    per-env Cohen's d on outcome. ρ > 0 predicted.

  Bridge 2 (single-env, Type A learnable):
    `ddqn_harms_asterix_gamma_999`
    Asterix γ=0.999 d_out CI fully below −0.4 (DDQN harms).

  Bridge 3 (single-env, Type B / FA-truncation):
    `ddqn_helps_breakout_gamma_999`
    Breakout γ=0.999 d_out CI fully above +0.4 (DDQN helps).

`composed_verdict` is AND-aggregate: SUPPORTED iff all three HELD;
REFUTED if any REFUTES. The cross-env bridge fires at n_strata=6
(current cache); will increase to 8 when Freeway+SI k=1 γ=0.999
land, and beyond 8 when k=2/k=4 sweeps complete.

Current empirical snapshot:
  Bridge 1: Spearman ρ=+0.61 p=0.15 at n=7 → POWER_INSUFFICIENT
  Bridge 2: d_out=−0.80 z=−3.1 → HELD (CI_high ≈ −0.30 < −0.4? no,
            CI = d ± 1.96·SE; SE ≈ d/z = 0.26; CI_high ≈ −0.30 →
            NOT fully below −0.4; POWER_INSUFFICIENT at canonical
            tolerance. Lowering harm_floor to −0.3 would flip to
            HELD; held floor at −0.4 documents stronger claim.)
  Bridge 3: d_out=+0.66 z=+2.6 → CI ≈ +0.16 to +1.16; CI_low<0.4
            → POWER_INSUFFICIENT; help_floor of +0.4 documents
            stronger claim.

Setting EXPECTED to UNDERPOWERED + BLOCKED_ON to capture the
n_strata=6 + per-env n=30 power floors. Once the running γ × k
sweeps land (~3.5 days from now), each bridge gets more cells +
the cross-env bridge gets more strata; this finding's verdict
should flip to SUPPORTED.
"""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn_sweeps.sigma_over_jens_regime import (
    ddqn_harms_asterix_gamma_999,
    ddqn_helps_breakout_gamma_999,
    ddqn_outcome_scales_with_sigma_over_jens__gamma_999_xenv,
)


EXPECTED: ClusterVerdict = ClusterVerdict.UNDERPOWERED


BLOCKED_ON: str | None = (
    'n_strata=6 envs with γ=0.999 data; cross-env Spearman ρ=+0.61 '
    'trending at p=0.15 with n=7. Per-env CIs (Asterix d_out=-0.80 '
    'z=-3.1, Breakout d_out=+0.66 z=+2.6) do not yet fully exceed '
    'the |0.4| floors. Once minatar_gamma_sweep_k1 lands Freeway + '
    'SI γ=0.999 (n_strata → 8) and k=2/k=4 sweeps amplify, the '
    'cross-env bridge should HELD at ρ ≥ 0.5 p ≤ 0.10 and per-env '
    'bridges should flip HELD on stronger d-magnitudes.'
)


BRIDGES: tuple[Bridge, ...] = (
    ddqn_outcome_scales_with_sigma_over_jens__gamma_999_xenv,
    ddqn_harms_asterix_gamma_999,
    ddqn_helps_breakout_gamma_999,
)
