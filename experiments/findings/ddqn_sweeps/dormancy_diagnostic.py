"""Diagnostic bridge: within-arm Jensen-dormancy gates the
observed bias at Acrobot γ=0.999.

Acts as a "compensating" counterpart to the Mech bridge
(`ddqn_reduces_jens_consistently__canonical_g0999`). The Mech
bridge tests "DDQN reduces jens at every env"; this diagnostic
bridge tests "the dormancy measurable correctly indexes per-cell
bias presence" — validating the reason for tolerating the
Acrobot γ=0.999 outlier (d_jens = +0.10) as "mech partially
dormant" rather than "DDQN fails here."

At Acrobot γ=0.999, the consistency bridge's lone outlier comes
from 13% (vanilla) / 23% (DDQN) of cells with
`jensen_dormancy_gap > 0` — observed bias below the σ_Q-based
structural floor. Per CLAUDE.md the verdict on dormant-mech cells
is UNTESTABLE, not NULL. The d_jens=+0.10 result then mixes
mech-active cells (DDQN cuts bias normally) with mech-dormant
cells (jens clamped at zero via `max(0, Q − MC)`).

For the dormancy-noise explanation to hold, the dormancy
measurable must do its diagnostic job: high dormancy ↔ low
observed bias WITHIN each arm. If not, the dormancy reading
might itself be the artifact.

The bridge tests `ρ(jensen_dormancy_gap, jensen_gap | arm) < 0`
(strong negative). At Acrobot γ=0.999 (n=60, both arms):
empirical ρ_partial = −0.664 p = 2.1e-9. Within vanilla
(n=30): ρ = −0.59 p=6e-4. Within DDQN (n=30): ρ = −0.73 p=5e-6.

Both arms confirm dormancy → jens link in the predicted direction.
The dormancy diagnostic is well-behaved at this env; the
consistency bridge's accommodation of the d_jens=+0.10 outlier is
substantively justified.

Future work: when other canonical-pool corpora have
`jensen_dormancy_gap` backfilled (currently only Acrobot +
MetaMaze + Snake have it computed locally), this bridge extends
to a cross-env consistency claim using
`cross_env_consistency_binomial` over per-env partial-r values.
For now: Acrobot γ=0.999 single-env diagnostic.
"""
from __future__ import annotations

import math

import polars as pl

from corroborate.analyses.spearman.partial_spearman import (
    PartialSpearmanResult,
)
from corroborate.bridge.bridge import Direction, Tier, claim_bridge
from corroborate.bridge.verdict import RefutationClass, Verdict


@claim_bridge(
    source='jensen_dormancy_gap',
    target='jensen_gap',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=(
        (pl.col('env_name') == 'Acrobot-v1')
        & (pl.col('gamma') == 0.999)
        & (pl.col('corpus') == 'gamma_sweep_acrobot')
        & pl.col('jensen_dormancy_gap').is_finite()
        & pl.col('jensen_gap').is_finite()
        & pl.col('arm_is_baseline').is_finite()
    ),
    predicted_direction='a_lt_b',
)
def dormancy_gates_jens_at_acrobot_g0999(
    partial_spearman: PartialSpearmanResult,
    *,
    x: str = 'jensen_dormancy_gap',
    y: str = 'jensen_gap',
    conditioning: tuple[str, ...] = ('arm_is_baseline',),
    stratify_by: str = 'env_name',
    min_stratum_size: int = 30,
    rho_threshold_held: float = -0.5,
    p_threshold_held: float = 0.05,
    null_threshold: float = 0.1,
    sign_flip_threshold: float = 0.5,
) -> tuple[Verdict, RefutationClass | None]:
    """Within-arm partial-Spearman ρ(dormancy, jens | arm) at
    Acrobot γ=0.999.

    `predicted_direction='a_lt_b'` here means ρ < 0 — high
    dormancy ↔ low observed bias. The compensating diagnostic
    for the Mech consistency bridge: validates that the
    dormancy measurable's reading at this env is well-behaved
    (high-dormancy cells genuinely have low jens), which in turn
    justifies attributing the consistency bridge's +0.10 outlier
    to mech-dormancy noise rather than "DDQN fails."

    Verdict matrix:
      HELD                : ρ ≤ −0.5 AND p ≤ 0.05
      NO_EFFECT (NULL)    : |ρ| ≤ 0.1
      NO_EFFECT (SIGN_FLIP): ρ ≥ +0.5 (decisive wrong direction)
      POWER_INSUFFICIENT  : in-between

    Empirical at the gamma_sweep_acrobot corpus, n=60:
      ρ_pooled = −0.664, p = 2.1e-9 → HELD."""
    del x, y, conditioning, stratify_by, min_stratum_size
    rho = partial_spearman.rho_pooled
    p = partial_spearman.p_value
    if math.isnan(rho) or math.isnan(p):
        return Verdict.POWER_INSUFFICIENT, None
    if rho <= rho_threshold_held and p <= p_threshold_held:
        return Verdict.HELD, None
    if rho >= sign_flip_threshold:
        return Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP
    if abs(rho) <= null_threshold:
        return Verdict.NO_EFFECT, RefutationClass.NULL_EFFECT
    return Verdict.POWER_INSUFFICIENT, None


__all__ = ['dormancy_gates_jens_at_acrobot_g0999']
