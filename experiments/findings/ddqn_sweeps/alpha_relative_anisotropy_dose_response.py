"""Dose-response on the convex DDQN-vanilla interpolation —
relative-anisotropy chain at Asterix γ=0.999.

Pre-registered 2026-05-26 BEFORE α ∈ {0.25, 0.5, 0.75} cells land.
The α=0 (vanilla) and α=1 (DDQN) endpoints come from
`canonical_n_eps20_ckpt` (15 V + 15 D); the intermediate α arms are
in `asterix_g0999_dampened_alpha_ckpt` (α=0.25, 0.5) and
`asterix_g0999_dampened_alpha075_ckpt` (α=0.75). Together a
5-point × 15-seed = 75-cell dose-response panel.

Hypothesis (causal chain — 5 bridges, scope-cluster per
HYPOTHESIS_AS_GRAPH §3b):

  α (treatment) ─→ |Q| deflation ─→ relative anisotropy GROWS
                                  ─→ state-coverage SHRINKS
                                  ─→ eval_best DROPS

Each leg is its own bridge. All HELD → mechanism CORROBORATED.
Failure-mode reading by leg specified in the bridges' docstrings.

The novel piece (Bridge 2) is **relative** anisotropy
(q_argmax_margin / |q_late_mean|): both numerator and denominator
deflate under DDQN, but the magnitude deflates MORE → ratio
GROWS. The within-arm V↔D contrast at canonical n_eps20_ckpt
shows V's relative gap ≈ 0.0037 vs D's ≈ 0.0045 (~22% larger).
The dose-response predicts linear monotonic growth in this ratio
with α."""
from __future__ import annotations

import polars as pl

from corroborate.analyses.spearman.partial_spearman import (
    PartialSpearmanResult,
)
from corroborate.bridge.bridge import Direction, Tier, claim_bridge
from corroborate.bridge.predicates import finite
from corroborate.bridge.verdict import Verdict

from experiments.findings.ddqn._verdicts import (
    partial_spearman_signed_verdict,
)


# Scope: Asterix γ=0.999, canonical-shape HPs, all five α arms.
# Selecting by finite `effective_alpha` (parses arm_key into α ∈
# {0, 0.25, 0.5, 0.75, 1.0}); excludes other algorithmic arms
# (action-duplicate, n-step, expectile, polyak) that yield NaN
# effective_alpha by design.
_ASTERIX_G0999_ALPHA_SCOPE: pl.Expr = (
    (pl.col('env_name') == 'Asterix-MinAtar')
    & (pl.col('gamma') == 0.999)
    & finite('effective_alpha')
    & finite('eval_best_burst_raw_mean')
    & finite('q_late_mean')
    & ((pl.col('n_step') == 1) | pl.col('n_step').is_null())
    & pl.col('action_duplicate_k').is_null()
    & (pl.col('reward_scale').is_null() | (pl.col('reward_scale') == 1.0))
    & pl.col('target_sync.tau').is_null()
)


# ---------------------------------------------------------------
# Bridge 1 — α scales Q magnitude deflation (mechanical check).
# ---------------------------------------------------------------
@claim_bridge(
    source='effective_alpha',
    target='q_late_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=_ASTERIX_G0999_ALPHA_SCOPE,
    predicted_direction='a_lt_b',
)
def alpha_scales_q_magnitude_deflation__asterix_g0999(
    partial_spearman: PartialSpearmanResult,
    *,
    x: str = 'effective_alpha',
    y: str = 'q_late_mean',
    conditioning: tuple[str, ...] = (),
    stratify_by: str = 'corpus',
    min_stratum_size: int = 5,
    rho_threshold: float = 0.4,
    min_strata: int = 1,
) -> Verdict:
    """Mechanical: α scales the convex-mix's deflation of late-window
    mean Q. Predicted NEGATIVE Spearman ρ(α, q_late_mean) — higher α
    → more DDQN-mixed target → more Q deflation.

    If this NULL-s, the dampened_double_greedify operator isn't
    doing what theory says — sweep is broken upstream and Bridges
    2-5's verdicts are uninterpretable. Acts as a power gate."""
    del x, y, conditioning, stratify_by, min_stratum_size
    return partial_spearman_signed_verdict(
        partial_spearman,
        threshold=rho_threshold, sign=-1, min_strata=min_strata,
    )


# ---------------------------------------------------------------
# Bridge 2 — α grows RELATIVE anisotropy (the novel prediction).
# ---------------------------------------------------------------
@claim_bridge(
    source='effective_alpha',
    target='q_action_gap_relative_late',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=_ASTERIX_G0999_ALPHA_SCOPE & finite('q_action_gap_relative_late'),
    predicted_direction='a_gt_b',
)
def alpha_grows_relative_anisotropy__asterix_g0999(
    partial_spearman: PartialSpearmanResult,
    *,
    x: str = 'effective_alpha',
    y: str = 'q_action_gap_relative_late',
    conditioning: tuple[str, ...] = (),
    stratify_by: str = 'corpus',
    min_stratum_size: int = 5,
    rho_threshold: float = 0.3,
    min_strata: int = 1,
) -> Verdict:
    """Novel prediction: DDQN's wedge deflates absolute Q magnitudes
    MORE than it deflates absolute action gaps, so the RATIO
    `q_argmax_margin_late / |q_late_mean|` GROWS with α. Predicted
    POSITIVE Spearman ρ ≥ 0.30.

    Within-arm endpoint contrast (canonical n_eps20_ckpt n=30):
      V (α=0) relative gap ≈ 0.0037
      D (α=1) relative gap ≈ 0.0045 (~22% larger)

    Dose-response predicts linear interpolation in between. NULL
    here would refute the "DDQN induces relative anisotropy" frame
    — the V→D difference would have to be a coincidence, not a
    dose-responsive effect."""
    del x, y, conditioning, stratify_by, min_stratum_size
    return partial_spearman_signed_verdict(
        partial_spearman,
        threshold=rho_threshold, sign=+1, min_strata=min_strata,
    )


# ---------------------------------------------------------------
# Bridge 3 — α shrinks state coverage (lock-in proxy).
# ---------------------------------------------------------------
@claim_bridge(
    source='effective_alpha',
    target='unique_states_visited_late',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=_ASTERIX_G0999_ALPHA_SCOPE & finite('unique_states_visited_late'),
    predicted_direction='a_lt_b',
)
def alpha_shrinks_state_coverage__asterix_g0999(
    partial_spearman: PartialSpearmanResult,
    *,
    x: str = 'effective_alpha',
    y: str = 'unique_states_visited_late',
    conditioning: tuple[str, ...] = (),
    stratify_by: str = 'corpus',
    min_stratum_size: int = 5,
    rho_threshold: float = 0.3,
    min_strata: int = 1,
) -> Verdict:
    """Lock-in chain proxy: higher α → sharper Q → policy commits
    to specific actions earlier → less state exploration → fewer
    unique states visited in late window.

    Within-arm endpoint contrast (canonical n_eps20_ckpt):
      V (α=0) visits ~1351 unique states
      D (α=1) visits ~1216 unique states (~10% fewer)

    Dose-response predicts NEGATIVE Spearman ρ(α, unique_states)."""
    del x, y, conditioning, stratify_by, min_stratum_size
    return partial_spearman_signed_verdict(
        partial_spearman,
        threshold=rho_threshold, sign=-1, min_strata=min_strata,
    )


# ---------------------------------------------------------------
# Bridge 4 — α scales eval harm (outcome leg — corroborates V/D).
# ---------------------------------------------------------------
@claim_bridge(
    source='effective_alpha',
    target='eval_best_burst_raw_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=_ASTERIX_G0999_ALPHA_SCOPE,
    predicted_direction='a_lt_b',
)
def alpha_scales_eval_harm__asterix_g0999(
    partial_spearman: PartialSpearmanResult,
    *,
    x: str = 'effective_alpha',
    y: str = 'eval_best_burst_raw_mean',
    conditioning: tuple[str, ...] = (),
    stratify_by: str = 'corpus',
    min_stratum_size: int = 5,
    rho_threshold: float = 0.4,
    min_strata: int = 1,
) -> Verdict:
    """Outcome leg: dose-response confirms the binary V/D contrast as
    a continuous monotonic effect. Predicted NEGATIVE Spearman
    ρ(α, eval_best) — higher α → lower outcome.

    Binary endpoints (canonical_n_eps20_ckpt): d_eval (V−D) = +1.45
    → marginal ρ(arm, eval_best) ≈ −0.60.

    Dose-response linear interpolation predicts |ρ| ≈ 0.4 at the
    n=75 panel level. Fragility: if α=0.5 already saturates the
    harm (eval ≈ D's), this would still HELD (negative slope
    survives saturation); if α=0.5 has no harm but α=0.75 does,
    the dose-response is step-function and HELD with non-linear
    shape (a Finding-level annotation, not a Bridge verdict)."""
    del x, y, conditioning, stratify_by, min_stratum_size
    return partial_spearman_signed_verdict(
        partial_spearman,
        threshold=rho_threshold, sign=-1, min_strata=min_strata,
    )


# ---------------------------------------------------------------
# Bridge 5 — Relative anisotropy MEDIATES α → eval harm.
# ---------------------------------------------------------------
@claim_bridge(
    source='effective_alpha',
    target='eval_best_burst_raw_mean',
    direction=Direction.AT_MOST,
    tier=Tier.ASSOCIATIONAL,
    scope=_ASTERIX_G0999_ALPHA_SCOPE & finite('q_action_gap_relative_late'),
    predicted_direction='a_gt_b',
)
def relative_anisotropy_mediates_alpha_harm__asterix_g0999(
    partial_spearman: PartialSpearmanResult,
    *,
    x: str = 'effective_alpha',
    y: str = 'eval_best_burst_raw_mean',
    conditioning: tuple[str, ...] = ('q_action_gap_relative_late',),
    stratify_by: str = 'corpus',
    min_stratum_size: int = 5,
    rho_threshold: float = 0.2,
    min_strata: int = 1,
) -> Verdict:
    """Mediation: conditioning on relative anisotropy ATTENUATES
    the α → eval marginal correlation. Predicted: marginal ρ(α,
    eval_best) ≈ −0.6 → partial ρ | rel_anisotropy → |ρ_partial|
    < 0.20 (substantial attenuation).

    The bridge's `predicted_direction='a_gt_b'` encodes
    "ρ_partial > -threshold" (i.e., closer to zero than marginal).
    The verdict helper checks |ρ_partial| ≤ rho_threshold.

    HELD with attenuation closes the chain — α's harm channel
    runs THROUGH relative anisotropy. Partial ρ still strongly
    negative would refute this leg: anisotropy growth happens
    (B2) but doesn't mediate the harm (B5 NULL) → chain is
    BROKEN at the mediation step."""
    del x, y, conditioning, stratify_by, min_stratum_size
    return partial_spearman_signed_verdict(
        partial_spearman,
        threshold=rho_threshold, sign=+1, min_strata=min_strata,
    )


BRIDGES = (
    alpha_scales_q_magnitude_deflation__asterix_g0999,
    alpha_grows_relative_anisotropy__asterix_g0999,
    alpha_shrinks_state_coverage__asterix_g0999,
    alpha_scales_eval_harm__asterix_g0999,
    relative_anisotropy_mediates_alpha_harm__asterix_g0999,
)


__all__ = (
    'BRIDGES',
    'alpha_scales_q_magnitude_deflation__asterix_g0999',
    'alpha_grows_relative_anisotropy__asterix_g0999',
    'alpha_shrinks_state_coverage__asterix_g0999',
    'alpha_scales_eval_harm__asterix_g0999',
    'relative_anisotropy_mediates_alpha_harm__asterix_g0999',
)
