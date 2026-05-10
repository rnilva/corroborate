"""`arm_mean_diff` — independent-samples mean comparison between two arms.

Replaces `paired_g`'s paired-difference t-test for the common
case where seed-pairing doesn't actually couple the two arms (any
env where ρ(v_outcome, d_outcome | seed) ≈ 0 — most stochastic
envs). Uses Welch's t-test on the two arms' marginal samples.
Reports ρ(treatment_value, baseline_value) as a diagnostic — when
ρ is significantly > 0, paired_g would be more powerful and
should be preferred; otherwise this independent-samples form is
the conceptually-correct test.

Why this primitive exists:
  - `paired_g` mathematically pairs by seed via per-pair Δ. On
    deterministic envs (FourRooms) that's powerful; on stochastic
    envs (Pacman, MetaMaze γ=0.999) ρ ≈ 0 and the paired form is
    just an independent test minus 1 df.
  - The framework's intent for `pair_by=('seed',)` is "use seed as
    aggregation unit per arm, compare distributions head-to-head".
    `arm_mean_diff` realizes that intent directly.
  - Per-pair Δ is still computed for the ρ diagnostic; not used
    in the test stat.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from corroborate.bridge.analysis import analysis

from corroborate.analyses.paired_g import key_tuple, resolve_value


@dataclass(frozen=True, slots=True)
class ArmMeanDiffResult:
    """Output of independent-samples mean comparison between two arms.

    `mean_diff` is `mean(treatment) - mean(baseline)` (sign matches
    `paired_g`'s direction). `mean_diff_se` is the Welch SE
    (independent-samples), NOT the paired-Δ SE.

    `n_treatment` / `n_baseline` are the per-arm sample sizes
    (independent — these are not "pairs"). When the bridge ALSO
    wants per-pair coupling info, `n_paired` reports how many
    cells actually share a pair_by key, and `pairing_rho` is the
    Pearson correlation between paired (treatment, baseline)
    values across those keys.

    `pairing_rho` is a DIAGNOSTIC, not used in the test stat:
      - `pairing_rho > 0` substantially: paired_g would be more
        powerful; consider switching analyses.
      - `pairing_rho ≈ 0` (CI spans 0): independent-samples test
        is the conceptually-correct choice; `arm_mean_diff` ≈
        `paired_g` in power but cleaner conceptually.
      - `pairing_rho < 0`: paired_g would be ANTI-paired (worse
        power than independent). Use `arm_mean_diff`.
    `pairing_rho_se` is the Fisher-z SE for the diagnostic CI.
    Both NaN when `n_paired < 5`.
    """
    mean_treatment: float
    mean_baseline: float
    sd_treatment: float
    sd_baseline: float
    mean_diff: float
    mean_diff_se: float
    n_treatment: int
    n_baseline: int
    welch_df: float
    n_paired: int
    pairing_rho: float
    pairing_rho_se: float
    measurable: str
    treatment_arm: str
    baseline_arm: str
    arm_field: str

    @property
    def mean_diff_p_value(self) -> float:
        """Two-sided Welch's t-test p-value for `mean_diff != 0`.
        NaN when SE is zero or `n_treatment < 2` / `n_baseline < 2`."""
        if (
            math.isnan(self.mean_diff)
            or math.isnan(self.mean_diff_se)
            or self.mean_diff_se == 0.0
            or self.n_treatment < 2
            or self.n_baseline < 2
        ):
            return float('nan')
        from scipy.stats import t as _t
        t_stat = abs(self.mean_diff / self.mean_diff_se)
        return float(2.0 * (1.0 - _t.cdf(t_stat, df=self.welch_df)))

    @property
    def standardized_effect(self) -> float:
        """Cohen's d (independent-samples form): mean_diff /
        pooled_sd. Distinct from `paired_g`'s Hedges' g (which is
        scaled by the per-pair Δ SD). NaN when both arm SDs are 0."""
        if math.isnan(self.sd_treatment) or math.isnan(self.sd_baseline):
            return float('nan')
        n_t = self.n_treatment; n_b = self.n_baseline
        if n_t < 2 or n_b < 2:
            return float('nan')
        var_pooled = (
            (n_t - 1) * self.sd_treatment**2
            + (n_b - 1) * self.sd_baseline**2
        ) / (n_t + n_b - 2)
        if var_pooled <= 0.0:
            return float('nan')
        return self.mean_diff / math.sqrt(var_pooled)


@analysis
def arm_mean_diff(
    cells: Iterable[Mapping[str, object]],
    *,
    source: str,
    treatment_arm: str,
    baseline_arm: str,
    pair_by: tuple[str, ...] = ('seed',),
    arm_field: str = 'arm_key',
) -> ArmMeanDiffResult:
    """Independent-samples mean comparison: `mean(treatment) -
    mean(baseline)` with Welch's t-test SE; ρ(treatment, baseline)
    as a paired-coupling diagnostic.

    Unlike `paired_g`:
      - Test stat does NOT difference per-pair, so on stochastic
        envs (where `ρ(v, d) ≈ 0`) it gives the same power as the
        paired form (modulo 1 df). On deterministic envs (`ρ > 0`)
        paired_g would be more powerful — `pairing_rho` flags
        when to switch.
      - No `dedupe_strategy` parameter — the analysis is on
        marginal arm samples, so duplicate cells per (arm,
        pair_by) bucket are flat-aggregated by the same arm-mean.
        Mean-aggregating per bucket (paired_g's `'mean'` strategy)
        and flat-mean across all cells in an arm give the same
        per-arm mean when bucket sizes are uniform.

    `source` resolves through the persisted-column path or the
    measurable registry, same as `paired_g`. Pairing keys are only
    used to compute the `pairing_rho` diagnostic; NOT used in the
    test statistic.

    Honest indep-samples form: every cell that passes scope
    contributes to its arm's marginal sample, independently. We
    do NOT intersect on `pair_by` to ensure "both arms have a
    counterpart" — that would smuggle paired-sample dependence
    back into a Welch SE that assumes independence, biasing SE
    low. If asymmetric scope predicates (e.g. `jens > 0.05`
    filtering more DDQN cells than vanilla) bias the comparison,
    that's a SCOPE issue to handle at the bridge level, not by
    silently filtering arm samples here."""
    treatment_vals: list[float] = []
    baseline_vals: list[float] = []
    treatment_paired: dict[tuple[object, ...], list[float]] = {}
    baseline_paired: dict[tuple[object, ...], list[float]] = {}

    for cell in cells:
        arm = cell.get(arm_field)
        v = resolve_value(cell, source)
        if math.isnan(v):
            continue
        if arm == treatment_arm:
            treatment_vals.append(v)
            treatment_paired.setdefault(
                key_tuple(cell, pair_by), [],
            ).append(v)
        elif arm == baseline_arm:
            baseline_vals.append(v)
            baseline_paired.setdefault(
                key_tuple(cell, pair_by), [],
            ).append(v)

    n_t = len(treatment_vals); n_b = len(baseline_vals)
    if n_t < 2 or n_b < 2:
        return ArmMeanDiffResult(
            mean_treatment=float('nan'),
            mean_baseline=float('nan'),
            sd_treatment=float('nan'),
            sd_baseline=float('nan'),
            mean_diff=float('nan'),
            mean_diff_se=float('nan'),
            n_treatment=n_t, n_baseline=n_b,
            welch_df=float('nan'),
            n_paired=0,
            pairing_rho=float('nan'),
            pairing_rho_se=float('nan'),
            measurable=source,
            treatment_arm=treatment_arm,
            baseline_arm=baseline_arm,
            arm_field=arm_field,
        )

    mean_t = sum(treatment_vals) / n_t
    mean_b = sum(baseline_vals) / n_b
    var_t = sum((v - mean_t) ** 2 for v in treatment_vals) / (n_t - 1)
    var_b = sum((v - mean_b) ** 2 for v in baseline_vals) / (n_b - 1)
    sd_t = math.sqrt(var_t); sd_b = math.sqrt(var_b)

    # Welch's t-test
    se_diff = math.sqrt(var_t / n_t + var_b / n_b)
    if (var_t / n_t + var_b / n_b) > 0:
        welch_df = (var_t / n_t + var_b / n_b) ** 2 / (
            (var_t / n_t) ** 2 / (n_t - 1)
            + (var_b / n_b) ** 2 / (n_b - 1)
        )
    else:
        welch_df = float('nan')

    # Pairing diagnostic (NOT used in test stat): align cells that
    # share a pair_by key, take per-key arm means, compute ρ. This
    # tells the bridge author whether seed-pairing would gain power
    # over the indep-samples Welch's used here. CI excluding 0
    # toward + → paired_g would be more powerful.
    paired_keys = sorted(set(treatment_paired) & set(baseline_paired))
    n_paired = len(paired_keys)
    if n_paired >= 5:
        paired_t = [
            sum(treatment_paired[k]) / len(treatment_paired[k])
            for k in paired_keys
        ]
        paired_b = [
            sum(baseline_paired[k]) / len(baseline_paired[k])
            for k in paired_keys
        ]
        m_pt = sum(paired_t) / n_paired; m_pb = sum(paired_b) / n_paired
        cov = sum(
            (t - m_pt) * (b - m_pb)
            for t, b in zip(paired_t, paired_b)
        ) / max(n_paired - 1, 1)
        var_pt = sum((t - m_pt) ** 2 for t in paired_t) / max(n_paired - 1, 1)
        var_pb = sum((b - m_pb) ** 2 for b in paired_b) / max(n_paired - 1, 1)
        denom = math.sqrt(var_pt * var_pb)
        rho = cov / denom if denom > 0 else float('nan')
        # Fisher z SE (one-sigma in ρ space): se ≈ 1/√(n-3)
        rho_se = 1.0 / math.sqrt(max(n_paired - 3, 1)) if n_paired > 3 else float('nan')
    else:
        rho = float('nan'); rho_se = float('nan')

    return ArmMeanDiffResult(
        mean_treatment=mean_t,
        mean_baseline=mean_b,
        sd_treatment=sd_t,
        sd_baseline=sd_b,
        mean_diff=mean_t - mean_b,
        mean_diff_se=se_diff,
        n_treatment=n_t,
        n_baseline=n_b,
        welch_df=welch_df,
        n_paired=n_paired,
        pairing_rho=rho,
        pairing_rho_se=rho_se,
        measurable=source,
        treatment_arm=treatment_arm,
        baseline_arm=baseline_arm,
        arm_field=arm_field,
    )


__all__ = ['ArmMeanDiffResult', 'arm_mean_diff']
