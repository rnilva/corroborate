"""Transfer test: does the FR γ-WHY chain (`finding_sigma_action_completes_chain`,
γ → {self_ref, σ_action} → jens) replicate at MinAtar envs?

The FR γ-WHY finding established that at FR × MLP × unshaped ×
baseline (γ ∈ {0.99, 0.999}), γ amplifies vanilla jens through
two paths jointly: bootstrap_self_reference_fraction (anchor
failure, vanilla can't find reward) and q_action_std_late
(per-state Q variance growth). Joint partial ρ(γ, jens | self_ref
+ σ_action) ≈ +0.06 NS — 92% reduction from marginal +0.78.

At MinAtar, the 4-env γ=0.999 panel reveals a regime split:
- **SI γ=0.999 (Q-STRUCTURED)**: vanilla outcome drops 101→74
  going γ=0.99 → γ=0.999; jens 3.5 → 56.7 (~16×); DDQN rescues
  outcome to 105 with d_out=+2.18 (biggest help in panel).
  Closest analogue to FR γ=0.999 mechanism.
- **Asterix γ=0.999 (Q-EXPLODED)**: vanilla outcome STAYS at 22
  across γ ∈ {0.95, 0.99, 0.999} — no anchor failure. Q grows
  monotonically 0.9→436 but argmax preserved; DDQN's clip
  CORRUPTS the working argmax. d_out=-0.76 (harm). FR mechanism
  doesn't apply.
- **Breakout γ=0.999**: FA-truncation regime (Type-2 bias per
  `findings_two_types_of_bias`). DDQN helps (+0.70).
- **Freeway γ=0.999**: saturated, mech inactive.

This module tests transfer per env. Expected:
- SI: chain HELDs (FR mechanism class-portable to Q-STRUCTURED).
- Asterix: chain REFUTES (anchor-failure mechanism doesn't apply
  — γ doesn't worsen vanilla outcome here).
- Breakout: chain HELDs partially (FA-truncation overlaps with
  Type-1 bias self-reference) — exploratory.
- Freeway: chain UNDERPOWERED (mech inactive).

Scope philosophy. The chain bridges here mirror the FR-scoped
bridges in `jens_reduction_factors.py` but per MinAtar env at
canonical-shape HPs. Strata = within-env across γ cells; the
test is the within-env Spearman over baseline cells at γ ∈
{0.95, 0.99, 0.999}.

Required measurables (REQUIRED_MEASURABLES in `__init__.py`):
- `bootstrap_self_reference_fraction` — anchor failure indicator
- `q_action_std_late` — per-state Q variance
- `jensen_gap` — Q-MC bias (already universal)

The MinAtar γ-sweep corpora `minatar_gamma_sweep_k1/g0{95,99,999}_*`
need traces restored + measurables backfilled on the two SI
sub-corpora (the others already carry the measurables)."""
from __future__ import annotations

import polars as pl

from corroborate.analyses.spearman.partial_spearman import (
    PartialSpearmanResult,
)
from corroborate.bridge.bridge import Direction, Tier, claim_bridge
from corroborate.bridge.predicates import finite
from corroborate.bridge.verdict import Verdict

from experiments.findings.ddqn_three_conditions._verdicts import (
    spearman_rho_verdict,
)


_MINATAR_ENVS = pl.col('env_name').is_in([
    'SpaceInvaders-MinAtar', 'Asterix-MinAtar',
    'Breakout-MinAtar', 'Freeway-MinAtar',
])
_MINATAR_GAMMAS = pl.col('gamma').is_in([0.95, 0.99, 0.999])


# --- SpaceInvaders γ-WHY chain (FR analogue) ---

@claim_bridge(
    source='gamma',
    target='bootstrap_self_reference_fraction',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=(
        (pl.col('env_name') == 'SpaceInvaders-MinAtar')
        & _MINATAR_GAMMAS
        & (pl.col('arm_key') == 'baseline')
        & finite(pl.col('bootstrap_self_reference_fraction'))
    ),
    predicted_direction='a_gt_b',
)
def gamma_predicts_q_self_reference_at_si(
    partial_spearman: PartialSpearmanResult,
    *,
    x: str = 'gamma',
    y: str = 'bootstrap_self_reference_fraction',
    stratify_by: str = 'env_name',
    min_stratum_size: int = 30,
    rho_threshold: float = 0.5,
) -> Verdict:
    """Stage 1 — γ predicts self_ref at SI baseline.

    Within SI baseline cells (γ ∈ {0.95, 0.99, 0.999}), ρ(γ,
    self_ref) ≥ +0.5. If γ doesn't grow self_ref at SI, the FR
    anchor-failure mechanism is not operational here.

    HELDs iff ρ ≥ +0.5 AND p < 0.05."""
    del x, y, stratify_by, min_stratum_size
    return spearman_rho_verdict(
        partial_spearman,
        sign=+1,
        threshold=rho_threshold,
    )


@claim_bridge(
    source='bootstrap_self_reference_fraction',
    target='jensen_gap',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=(
        (pl.col('env_name') == 'SpaceInvaders-MinAtar')
        & _MINATAR_GAMMAS
        & (pl.col('arm_key') == 'baseline')
        & finite(pl.col('bootstrap_self_reference_fraction'))
        & finite(pl.col('jensen_gap'))
    ),
    predicted_direction='a_gt_b',
)
def q_self_reference_predicts_jens_at_si(
    partial_spearman: PartialSpearmanResult,
    *,
    x: str = 'bootstrap_self_reference_fraction',
    y: str = 'jensen_gap',
    stratify_by: str = 'env_name',
    min_stratum_size: int = 30,
    rho_threshold: float = 0.5,
) -> Verdict:
    """Stage 2 — self_ref predicts jens at SI baseline.

    Within SI baseline (γ ∈ {0.95, 0.99, 0.999}), ρ(self_ref,
    jens) ≥ +0.5. The self_ref → jens link from FR transfers to
    SI iff this HELDs.

    HELDs iff ρ ≥ +0.5 AND p < 0.05."""
    del x, y, stratify_by, min_stratum_size
    return spearman_rho_verdict(
        partial_spearman,
        sign=+1,
        threshold=rho_threshold,
    )


@claim_bridge(
    source='gamma',
    target='jensen_gap',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=(
        (pl.col('env_name') == 'SpaceInvaders-MinAtar')
        & _MINATAR_GAMMAS
        & (pl.col('arm_key') == 'baseline')
        & finite(pl.col('bootstrap_self_reference_fraction'))
        & finite(pl.col('q_action_std_late'))
        & finite(pl.col('jensen_gap'))
    ),
    predicted_direction='null',
)
def gamma_jens_jointly_mediated_by_self_ref_and_sigma_action_at_si(
    partial_spearman: PartialSpearmanResult,
    *,
    x: str = 'gamma',
    y: str = 'jensen_gap',
    conditioning: tuple[str, ...] = (
        'bootstrap_self_reference_fraction', 'q_action_std_late',
    ),
    stratify_by: str = 'env_name',
    min_stratum_size: int = 30,
    rho_threshold: float = 0.3,
) -> Verdict:
    """Stage 3 — γ → jens fully mediated by {self_ref, σ_action} at SI.

    Within SI baseline (γ ∈ {0.95, 0.99, 0.999}), partial ρ(γ,
    jens | self_ref + σ_action) ≤ 0.3 (null after jointly
    partialling).

    The FR γ-WHY finding established |partial| = +0.06 NS at FR
    (92% reduction from marginal +0.78). If the same chain
    operates at SI, the joint partial here should also be null.

    HELDs iff |ρ_partial| ≤ 0.3 AND p ≥ 0.05 (null prediction).
    Refutations:
    - NO_EFFECT (sig non-null): γ has residual predictive power
      on jens beyond {self_ref, σ_action} at SI — additional
      mediator at SI not present at FR.
    - NO_EFFECT (sig negative): suppression structure.
    """
    del x, y, conditioning, stratify_by, min_stratum_size
    return spearman_rho_verdict(
        partial_spearman,
        sign=0,
        threshold=rho_threshold,
    )


# --- Asterix γ-WHY chain (predicted to REFUTE — different regime) ---

@claim_bridge(
    source='gamma',
    target='bootstrap_self_reference_fraction',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=(
        (pl.col('env_name') == 'Asterix-MinAtar')
        & _MINATAR_GAMMAS
        & (pl.col('arm_key') == 'baseline')
        & finite(pl.col('bootstrap_self_reference_fraction'))
    ),
    predicted_direction='null',
)
def gamma_self_ref_null_at_asterix(
    partial_spearman: PartialSpearmanResult,
    *,
    x: str = 'gamma',
    y: str = 'bootstrap_self_reference_fraction',
    stratify_by: str = 'env_name',
    min_stratum_size: int = 30,
    rho_threshold: float = 0.3,
) -> Verdict:
    """Asterix doesn't show anchor failure — predicted ρ(γ,
    self_ref) ≈ 0 at Asterix baseline.

    Vanilla outcome at Asterix is constant across γ ∈ {0.95,
    0.999} (=22 at both). Reward signal stays accessible; the
    bootstrap chain has MC anchor at all γ. So self_ref should
    NOT grow with γ — distinguishing Q-EXPLODED regime (high
    jens via uniform-across-actions overestimation) from
    FR/SI-style anchor-failure regime.

    HELDs iff |ρ| ≤ 0.3 AND p ≥ 0.05. Refutation by significant
    positive ρ would mean Asterix IS partially anchor-failure
    driven (would weaken the regime separation).
    """
    del x, y, stratify_by, min_stratum_size
    return spearman_rho_verdict(
        partial_spearman,
        sign=0,
        threshold=rho_threshold,
    )


@claim_bridge(
    source='gamma',
    target='jensen_gap',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=(
        (pl.col('env_name') == 'Asterix-MinAtar')
        & _MINATAR_GAMMAS
        & (pl.col('arm_key') == 'baseline')
        & finite(pl.col('bootstrap_self_reference_fraction'))
        & finite(pl.col('q_action_std_late'))
        & finite(pl.col('jensen_gap'))
    ),
    predicted_direction='a_gt_b',
)
def gamma_jens_residual_at_asterix_after_fr_mediators(
    partial_spearman: PartialSpearmanResult,
    *,
    x: str = 'gamma',
    y: str = 'jensen_gap',
    conditioning: tuple[str, ...] = (
        'bootstrap_self_reference_fraction', 'q_action_std_late',
    ),
    stratify_by: str = 'env_name',
    min_stratum_size: int = 30,
    rho_threshold: float = 0.5,
) -> Verdict:
    """At Asterix, γ predicts jens via mechanism OTHER than the
    FR chain. Partial ρ(γ, jens | self_ref + σ_action) at
    Asterix baseline ≥ +0.5 (substantial residual).

    Asterix's jens γ-scaling is super-linear (jens γ=0.95→γ=0.999
    grows 622× per `findings_si_corroborates_regime_classification`).
    If the FR mediators don't absorb this, the chain refutes for
    Asterix and the mechanism is genuinely different (Q-EXPLODED,
    not anchor-failure).

    HELDs iff residual ρ ≥ +0.5 AND p < 0.05 (FR chain doesn't
    transfer; Asterix needs different mediators).

    A null partial (HELD with sign=0) would mean Asterix IS
    driven by the FR mechanism after all — overruling the
    regime split. Either outcome is publishable; the bridge
    pins the prediction that Asterix is genuinely different.
    """
    del x, y, conditioning, stratify_by, min_stratum_size
    return spearman_rho_verdict(
        partial_spearman,
        sign=+1,
        threshold=rho_threshold,
    )


__all__ = (
    'gamma_predicts_q_self_reference_at_si',
    'q_self_reference_predicts_jens_at_si',
    'gamma_jens_jointly_mediated_by_self_ref_and_sigma_action_at_si',
    'gamma_self_ref_null_at_asterix',
    'gamma_jens_residual_at_asterix_after_fr_mediators',
)
