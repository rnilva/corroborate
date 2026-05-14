"""Cross-env mediation bridges: does DDQN's MECHANISM effect
predict its OUTCOME effect across envs?

The substantive question that the per-cell ρ(jens, outcome)
tests can't answer (Q-MC tautology: jens ≡ Q − MC, out_disc ≡
MC, so per-cell ρ(jens, out_disc) is structurally negative
by algebraic identity — see memory
`findings_bg_not_causally_manipulated_at_canonical`).

The clean test is at the ARM-DIFF level cross-env:
  x = Δ_predictor = mean(DDQN predictor) − mean(vanilla predictor)
  y = Δ_target    = mean(DDQN target)    − mean(vanilla target)

Each env contributes ONE (x, y) point. Cross-env Spearman ρ
asks: are the envs where DDQN reduces mech MORE also the envs
where DDQN improves outcome MORE?

This is **not** subject to the Q-MC algebraic identity because
the arm-diff is computed at the env-mean level, and the
intervention assignment (vanilla vs DDQN) is the causal cut —
not the within-cell jens-vs-outcome covariation that algebra
pins.

Cluster shape: three sibling bridges using different mech
predictors:

  - `ddqn_outcome_scales_with_jens_reduction__xenv`:
        predictor = `jensen_gap` (Q − MC bias; mech-canonical)
  - `ddqn_outcome_scales_with_bg_frac_active__xenv`:
        predictor = `bootstrap_gap_frac_active` (MC-free,
        avoids Q-MC tautology entirely)
  - `ddqn_outcome_scales_with_bg_q99__xenv`:
        predictor = `bootstrap_gap_q99` (tail magnitude of bg)

All three predict NEGATIVE ρ (more mech reduction → more
outcome improvement; arm-diff signs make Δ_predictor negative
when DDQN reduces, Δ_target positive when DDQN improves).

Outcome target is `eval_best_burst_raw_mean` (γ-invariant,
across-env-comparable per memory `findings_units_bug`).

Scope: cross-env (n=12 canonical envs); excludes saturating-
outcome envs (FourRooms via 100k slice, CartPole) so they don't
inject rank-tie noise. Sibling LOO-robust bridge for each
ensures HELD readings aren't driven by a single env."""
from __future__ import annotations

import math

import numpy as np
import polars as pl
import scipy.stats as stats

from corroborate.analyses.cross_stratum_arm_diff_slope import (
    CrossStratumArmDiffSlopeResult,
)
from corroborate.bridge.bridge import Direction, Tier, claim_bridge
from corroborate.bridge.verdict import RefutationClass, Verdict

from experiments.findings.ddqn._arms import DDQN_ARM, VANILLA_ARM
from experiments.findings.ddqn._scope import DDQN_RELEVANT_SCOPE


_TARGET = 'eval_best_burst_raw_mean'

# CartPole still saturated on its 1M canonical (both arms hit 500
# step cap consistently). FourRooms now sliced to 100k where
# vanilla has burst-0 variance — not strictly saturated, so we
# include it here. Re-evaluate this list after Maze+Sokoban land.
_SATURATING_OUTCOME_ENVS: tuple[str, ...] = (
    'CartPole-v1',
)
_NONSATURATED_SCOPE: pl.Expr = (
    ~pl.col('env_name').is_in(_SATURATING_OUTCOME_ENVS)
)


# **Q-MC algebraic tautology caveat for the `jensen_gap`
# predictor**. The substrate's `jensen_gap` is `Q − MC_disc` by
# definition. Cross-env arm-diff: `Δ_jens = ΔQ − ΔMC_disc`. When
# the env's disc-MC and raw outcome co-vary tightly
# (`env_disc_raw_alignment` high), `ΔMC_disc ≈ Δ_outcome_raw`, so
# `Δ_jens ≈ ΔQ − Δ_outcome_raw`, and the cross-env
# `ρ(Δ_jens, Δ_outcome_raw)` is algebraically negative *by
# construction* whenever `cov(ΔQ, Δ_outcome_raw) ≪
# var(Δ_outcome_raw)` — not from a substantive
# mech→outcome causal relationship.
#
# An earlier version of this file scoped to `align > 0.7`
# thinking this filtered to envs where the disc/raw translation
# was clean. That scope MAXIMIZED the tautology rather than
# mitigating it; it was retracted. The current file runs the
# `jensen_gap`-predictor bridges unscoped (full canonical n=12)
# with the explicit caveat that any negative ρ contains an
# algebraic component the cross-env Δ-Δ form cannot
# disentangle. The MC-free predictor
# (`bootstrap_gap_frac_active`) is algebraically clean: defined
# from Q-network outputs only, no `MC` term, so its cross-env
# ρ(Δ_bg_frac, Δ_outcome_raw) is a substantive causal test.
#
# Memory: `findings_bg_not_causally_manipulated_at_canonical`
# captured the bg-aggregation diagnosis; the user critique
# 2026-05-14 ("scoping the bridge where the tautology is more
# prominent") flagged the alignment-scope inversion. See also
# `CHAINED_BRIDGES_DESIGN.md` — chained-edge preconditions would
# make the algebraic interaction between `outcome_translation_
# consistent` (the would-be precondition edge) and the dependent
# `jens → raw` edge VISIBLE in the graph topology rather than
# buried in a scope predicate.
_XENV_MECH_TO_OUTCOME_SCOPE: pl.Expr = (
    DDQN_RELEVANT_SCOPE & _NONSATURATED_SCOPE
)


def _spearman_loo_min(
    xs: tuple[float, ...], ys: tuple[float, ...],
) -> float:
    n = len(xs)
    if n < 5:
        return float('nan')
    rhos: list[float] = []
    for i in range(n):
        xs_loo = np.asarray(xs[:i] + xs[i + 1:], dtype=np.float64)
        ys_loo = np.asarray(ys[:i] + ys[i + 1:], dtype=np.float64)
        rho_raw, _ = stats.spearmanr(xs_loo, ys_loo)
        rho_i = float(rho_raw)
        if math.isnan(rho_i):
            return float('nan')
        rhos.append(rho_i)
    return min(rhos)


def _xenv_mech_verdict(
    result: CrossStratumArmDiffSlopeResult,
    *,
    min_strata: int,
    rho_threshold_held: float,
    p_threshold: float,
    sign_flip_threshold: float,
) -> tuple[Verdict, RefutationClass | None]:
    """Verdict matrix for negative-direction cross-env tests.

    Predicted: ρ ≤ −threshold (mech reduces, outcome improves).
    The verdict's sign convention is **opposite** to
    `bias_correction_dose_response__xenv_arm_diff` (which
    predicts positive ρ for bg_magnitude → outcome co-direction).
    Here the arm-diff sign convention makes more-reduction more-
    negative, so negative ρ is the predicted HELD direction.

      HELD              : ρ ≤ −threshold_held AND p ≤ p_threshold
      NO_EFFECT (NULL)  : |ρ| < sign_flip_threshold
      NO_EFFECT (SIGN_FLIP) : ρ ≥ +sign_flip_threshold (wrong way)
      POWER_INSUFFICIENT : otherwise / n_strata < min
    """
    if result.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT, None
    rho = result.rho
    p = result.p_value
    if math.isnan(rho) or math.isnan(p):
        return Verdict.POWER_INSUFFICIENT, None
    if rho <= -rho_threshold_held and p <= p_threshold:
        return Verdict.HELD, None
    if rho >= sign_flip_threshold:
        return Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP
    if abs(rho) < sign_flip_threshold:
        return Verdict.NO_EFFECT, RefutationClass.NULL_EFFECT
    return Verdict.POWER_INSUFFICIENT, None


# ============ predictor: jensen_gap (mech-canonical) ============


@claim_bridge(
    source='jensen_gap',
    target=_TARGET,
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=_XENV_MECH_TO_OUTCOME_SCOPE,
    predicted_direction='a_lt_b',
)
def ddqn_outcome_scales_with_jens_reduction__xenv(
    cross_stratum_arm_diff_slope: CrossStratumArmDiffSlopeResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    predictor: str = 'jensen_gap',
    target: str = _TARGET,
    stratify_by: tuple[str, ...] = ('env_name',),
    min_seeds_per_arm: int = 5,
    min_strata: int = 5,
    rho_threshold_held: float = 0.5,
    p_threshold: float = 0.05,
    sign_flip_threshold: float = 0.3,
) -> tuple[Verdict, RefutationClass | None]:
    """Per-env Δ_jens vs Δ_outcome_raw cross-env Spearman.

    Predicts: more-negative Δ_jens (DDQN reduces jens more) →
    more-positive Δ_outcome → cross-env ρ NEGATIVE.

    NOTE: Q-MC algebraic identity applies to PER-CELL
    ρ(jens, out_disc) but NOT to this cross-env arm-diff form.
    Here each env contributes ONE (Δ_jens, Δ_out_raw) point;
    Δ_out_raw is undiscounted; the algebraic pinning of disc
    forms doesn't apply.

    Empirical pre-author (n=12): ρ=-0.35 (modest). With saturating
    envs excluded and at the n=10-11 scope, the bridge will likely
    land NO_EFFECT (NULL) or POWER_INSUFFICIENT — the cross-env
    magnitude conversion varies a lot by env (PacMan d_jens=-34
    d_out=+166 dominates; CartPole d_jens=-4 d_out=0 saturated)."""
    del (
        treatment_arm, baseline_arm, predictor, target,
        stratify_by, min_seeds_per_arm,
    )
    return _xenv_mech_verdict(
        cross_stratum_arm_diff_slope,
        min_strata=min_strata,
        rho_threshold_held=rho_threshold_held,
        p_threshold=p_threshold,
        sign_flip_threshold=sign_flip_threshold,
    )


@claim_bridge(
    source='jensen_gap',
    target=_TARGET,
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=_XENV_MECH_TO_OUTCOME_SCOPE,
    predicted_direction='a_lt_b',
)
def ddqn_outcome_scales_with_jens_reduction__xenv_loo_robust(
    cross_stratum_arm_diff_slope: CrossStratumArmDiffSlopeResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    predictor: str = 'jensen_gap',
    target: str = _TARGET,
    stratify_by: tuple[str, ...] = ('env_name',),
    min_seeds_per_arm: int = 5,
    min_strata: int = 5,
    loo_max_rho_threshold: float = -0.3,
    sign_flip_threshold: float = 0.3,
) -> tuple[Verdict, RefutationClass | None]:
    """LOO-robust sibling: HELD requires the cross-env Spearman
    to stay ≤ −0.3 under every LOO env removal. Refutes the
    anchor's HELD if it's driven by a single high-leverage env
    (e.g. PacMan with extreme Δ_jens/Δ_out ratio)."""
    del (
        treatment_arm, baseline_arm, predictor, target,
        stratify_by, min_seeds_per_arm,
    )
    if cross_stratum_arm_diff_slope.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT, None
    rho = cross_stratum_arm_diff_slope.rho
    if math.isnan(rho):
        return Verdict.POWER_INSUFFICIENT, None
    rho_loo_max = -_spearman_loo_min(
        tuple(-x for x in cross_stratum_arm_diff_slope.arm_diff_predictor),
        cross_stratum_arm_diff_slope.arm_diff_target,
    )  # min over LOO of -ρ is -max ρ; we want max ρ under LOO ≤ −0.3
    # Simpler: compute LOO ρs directly on the predictor (Δ_jens) sequence.
    xs = cross_stratum_arm_diff_slope.arm_diff_predictor
    ys = cross_stratum_arm_diff_slope.arm_diff_target
    n = len(xs)
    if n < 5:
        return Verdict.POWER_INSUFFICIENT, None
    loo_rhos: list[float] = []
    for i in range(n):
        xs_loo = np.asarray(xs[:i] + xs[i+1:], dtype=np.float64)
        ys_loo = np.asarray(ys[:i] + ys[i+1:], dtype=np.float64)
        rho_loo_raw, _ = stats.spearmanr(xs_loo, ys_loo)
        rho_loo = float(rho_loo_raw)
        if math.isnan(rho_loo):
            return Verdict.POWER_INSUFFICIENT, None
        loo_rhos.append(rho_loo)
    if max(loo_rhos) <= loo_max_rho_threshold:
        return Verdict.HELD, None
    if min(loo_rhos) >= sign_flip_threshold:
        return Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP
    return Verdict.POWER_INSUFFICIENT, None


# ============ predictor: bootstrap_gap_frac_active ============


@claim_bridge(
    source='bootstrap_gap_frac_active',
    target=_TARGET,
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=_XENV_MECH_TO_OUTCOME_SCOPE,
    predicted_direction='a_lt_b',
)
def ddqn_outcome_scales_with_bg_frac_active__xenv(
    cross_stratum_arm_diff_slope: CrossStratumArmDiffSlopeResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    predictor: str = 'bootstrap_gap_frac_active',
    target: str = _TARGET,
    stratify_by: tuple[str, ...] = ('env_name',),
    min_seeds_per_arm: int = 5,
    min_strata: int = 5,
    rho_threshold_held: float = 0.5,
    p_threshold: float = 0.05,
    sign_flip_threshold: float = 0.3,
) -> tuple[Verdict, RefutationClass | None]:
    """Per-env Δ_bg_frac_active vs Δ_outcome_raw cross-env Spearman.

    `bg_frac_active` is MC-FREE (algebraically independent of MC
    in its definition), so this bridge tests the substantive
    mech→outcome claim without the Q-MC tautology that haunts
    `jensen_gap`-based tests.

    Predicts: more-negative Δ_bg_frac_active (DDQN reduces argmax
    disagreement frequency more) → more-positive Δ_outcome →
    cross-env ρ NEGATIVE.

    Empirical pre-author: bg_frac_active is a new measurable
    (this commit); cross-env Spearman to be computed at ingest."""
    del (
        treatment_arm, baseline_arm, predictor, target,
        stratify_by, min_seeds_per_arm,
    )
    return _xenv_mech_verdict(
        cross_stratum_arm_diff_slope,
        min_strata=min_strata,
        rho_threshold_held=rho_threshold_held,
        p_threshold=p_threshold,
        sign_flip_threshold=sign_flip_threshold,
    )
