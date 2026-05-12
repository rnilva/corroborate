"""`within_arm_link` — within-arm correlation between predictor + outcome.

Replaces the conceptually-confused `paired_link r(Δ_predictor,
Δ_outcome)` with the correct test of "does predictor correlate
with outcome within each arm?". Reports two within-arm
correlations (treatment, baseline) plus a MODERATION test for
whether the slopes differ between arms.

**Honest framing — moderation, not mediation.** This primitive
asks "does the predictor-outcome correlation differ between
arms?" — that is a moderation question. It does NOT identify
"does the intervention CAUSE the outcome benefit through the
predictor?" — that is a mediation question requiring do-calculus
or a partial-Spearman | mediator analysis. For the latter, see
`corroborate.graph.discovery.partial_spearman_rho` and
`corroborate.analyses.proportion_mediated`.

Why this primitive exists:
  - `paired_link` correlates per-seed Δ_predictor with per-seed
    Δ_outcome. Empirically this equals approximately the average
    of the two within-arm correlations — NOT a "differential"
    signal between treatment and baseline. The framing was
    misleading.
  - The actual claims this primitive tests:
    1. Within the treatment arm, does the predictor correlate
       with outcome? (`r_treatment`, `r_treatment_p_value`)
    2. Within the baseline arm, does the predictor correlate
       with outcome? (`r_baseline`, `r_baseline_p_value`)
    3. Do the two within-arm correlations DIFFER? (`r_moderation`,
       `r_moderation_p_value`) — moderation, NOT mediation.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from corroborate.bridge.analysis import analysis

from corroborate.analyses.paired_g import resolve_value


@dataclass(frozen=True, slots=True)
class WithinArmLinkResult:
    """Within-arm Pearson correlation results.

    `r_treatment` and `r_baseline` are within-arm Pearson r
    between `predictor` and `target`. Each is the structural
    "does X correlate with Y within this arm" quantity — testable
    independently per arm.

    `r_moderation` and `r_moderation_se` test whether the two
    within-arm rs are significantly different (Fisher-z form).
    A non-zero `r_moderation` means the intervention CHANGES the
    correlation structure between predictor and outcome — i.e.
    the relationship is moderated by arm. **This is moderation,
    not mediation.** It does not establish that bias-reduction
    CAUSES outcome benefit; for mediation use partial-Spearman.

    `n_treatment` / `n_baseline` are per-arm sample sizes.

    `slope_treatment` / `slope_baseline` are the within-arm
    regression slopes Δoutcome / Δpredictor — same direction info
    as r but on the natural-scale of the predictor.
    """
    r_treatment: float
    r_baseline: float
    n_treatment: int
    n_baseline: int
    r_moderation: float
    r_moderation_se: float
    slope_treatment: float
    slope_baseline: float
    predictor: str
    target: str
    treatment_arm: str
    baseline_arm: str
    arm_field: str

    @property
    def r_treatment_p_value(self) -> float:
        """Two-sided p-value for `r_treatment != 0` (Fisher-z form).
        NaN if `n_treatment < 4` or `|r| ≥ 1`."""
        return _r_p_value(self.r_treatment, self.n_treatment)

    @property
    def r_baseline_p_value(self) -> float:
        """Two-sided p-value for `r_baseline != 0` (Fisher-z form).
        NaN if `n_baseline < 4` or `|r| ≥ 1`."""
        return _r_p_value(self.r_baseline, self.n_baseline)

    @property
    def r_moderation_p_value(self) -> float:
        """Two-sided p-value for `r_moderation != 0` (Fisher-z difference).
        NaN if either arm has `n < 4`. Tests whether the within-arm
        bias-outcome correlation differs between arms — a MODERATION
        signal, NOT mediation."""
        if (
            math.isnan(self.r_moderation)
            or math.isnan(self.r_moderation_se)
            or self.r_moderation_se == 0.0
        ):
            return float('nan')
        z = abs(self.r_moderation / self.r_moderation_se)
        return float(math.erfc(z / math.sqrt(2)))


def _r_p_value(r: float, n: int) -> float:
    if math.isnan(r) or n < 4 or abs(r) >= 1.0:
        return float('nan')
    z = 0.5 * math.log((1.0 + r) / (1.0 - r))
    se = 1.0 / math.sqrt(n - 3)
    return float(math.erfc(abs(z / se) / math.sqrt(2)))


def _pearson(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Return (r, slope) between xs and ys; (NaN, NaN) on degeneracy."""
    n = len(xs)
    if n < 2 or n != len(ys):
        return float('nan'), float('nan')
    mx = sum(xs) / n; my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / max(n - 1, 1)
    var_x = sum((x - mx) ** 2 for x in xs) / max(n - 1, 1)
    var_y = sum((y - my) ** 2 for y in ys) / max(n - 1, 1)
    denom = math.sqrt(var_x * var_y)
    if denom <= 0.0:
        return float('nan'), float('nan')
    r = cov / denom
    slope = cov / var_x if var_x > 0 else float('nan')
    return r, slope


@analysis
def within_arm_link(
    cells: Iterable[Mapping[str, object]],
    *,
    predictor: str,
    target: str,
    treatment_arm: str,
    baseline_arm: str,
    arm_field: str = 'arm_key',
) -> WithinArmLinkResult:
    """Within-arm Pearson r between `predictor` and `target` for
    each arm; difference test for whether the two rs differ.

    Three independent claims are testable from the result:
      1. `r_treatment` (with `r_treatment_p_value`): within the
         treatment arm, does predictor correlate with outcome?
      2. `r_baseline` (with `r_baseline_p_value`): within baseline,
         does predictor correlate with outcome?
      3. `r_moderation` (with `r_moderation_p_value`): does the
         intervention CHANGE the predictor-outcome correlation?
         This is a MODERATION test (does the within-arm slope
         differ?), NOT a mediation identification — mediation
         requires partial-Spearman / do-calculus.

    Replaces the misleading `paired_link r(Δ_pred, Δ_out)` which
    is approximately the average of (1) and (2) and doesn't
    cleanly distinguish them. Each within-arm r is the actual
    quantity of substantive interest; `r_moderation` is the
    cleanest "does the intervention CHANGE the link" signal —
    moderation, not mediation (mediation needs partial-Spearman).

    `predictor` and `target` resolve through the same path as
    `paired_g`'s `source` — persisted-column first, measurable
    registry second. Both must be scalar per cell.

    `arm_field` defaults to `'arm_key'`; must match the cell's
    treatment / baseline canonical-string key (same convention as
    `paired_g`).
    """
    treatment_xs: list[float] = []; treatment_ys: list[float] = []
    baseline_xs: list[float] = []; baseline_ys: list[float] = []
    for cell in cells:
        arm = cell.get(arm_field)
        x = resolve_value(cell, predictor)
        y = resolve_value(cell, target)
        if math.isnan(x) or math.isnan(y):
            continue
        if arm == treatment_arm:
            treatment_xs.append(x); treatment_ys.append(y)
        elif arm == baseline_arm:
            baseline_xs.append(x); baseline_ys.append(y)

    n_t = len(treatment_xs); n_b = len(baseline_xs)
    r_t, slope_t = _pearson(treatment_xs, treatment_ys)
    r_b, slope_b = _pearson(baseline_xs, baseline_ys)

    # Fisher-z difference test for r_t - r_b (moderation signal —
    # does the link slope DIFFER between arms? Not mediation.)
    if n_t < 4 or n_b < 4 or abs(r_t) >= 1 or abs(r_b) >= 1:
        r_mod = float('nan'); r_mod_se = float('nan')
    else:
        # SE of the Fisher-z difference assumes independent samples
        z_se = math.sqrt(1.0 / (n_t - 3) + 1.0 / (n_b - 3))
        r_mod = r_t - r_b  # raw r-scale difference for readability
        # Approximate r-scale SE via delta-method on Fisher-z
        # (dr/dz = 1 - r²; average across the two arms)
        dr_dz_t = 1.0 - r_t ** 2; dr_dz_b = 1.0 - r_b ** 2
        r_mod_se = z_se * 0.5 * (dr_dz_t + dr_dz_b)

    return WithinArmLinkResult(
        r_treatment=r_t,
        r_baseline=r_b,
        n_treatment=n_t,
        n_baseline=n_b,
        r_moderation=r_mod,
        r_moderation_se=r_mod_se,
        slope_treatment=slope_t,
        slope_baseline=slope_b,
        predictor=predictor,
        target=target,
        treatment_arm=treatment_arm,
        baseline_arm=baseline_arm,
        arm_field=arm_field,
    )


__all__ = ['WithinArmLinkResult', 'within_arm_link']
