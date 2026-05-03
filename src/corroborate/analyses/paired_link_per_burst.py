"""`paired_link_per_burst` — per-(env, burst) link strength panel.

Where `paired_g_per_burst` computes per-burst Δ on a single
measurable (the per-burst effect size), `paired_link_per_burst`
computes per-burst correlation r(Δ_target, Δ_predictor) across
paired seeds. This IS the empirical link from mediator to
outcome, evaluated burst-by-burst.

Why a separate primitive: scalar mech-link analyses (pair the
trajectory-averaged Δ_jens with trajectory-averaged Δ_out) silently
combine causally opposite phases. SpaceInvaders has Phase 1
(early bursts: bias correction → outcome, link active) and
Phase 2 (late bursts: Q-explosion → outcome reversal, link
flipped). The trajectory-averaged scalar slope ≈ 0 because the
two phases cancel. Per-burst link unmasks this.

The empirical lesson generalizes — see `findings_fourrooms_time_series.md`
and `findings_per_burst_canonical.md`. Per-burst is the canonical
form for any env where Q dynamics aren't monotone (Q-explosion-
prone / phase-transition envs).

The strata returned (env, burst → r, p, n_pairs) feed bridges
that assert claims like "link is active in Phase 1 (bursts ≤
n_phase) and reversed in Phase 2" — the per-burst panel makes
the temporal structure typed and corroborable.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from corroborate.analysis import analysis
from corroborate.analyses.paired_g_per_burst import cell_burst_values


@dataclass(frozen=True, slots=True)
class PerBurstLinkStratum:
    """One (env, burst) stratum: paired link statistics across paired
    seeds. `r` is correlation (link structure); `slope` is OLS β
    of Δ_target on Δ_predictor (conversion efficiency: outcome
    units per unit bias reduction); `mean_d_predictor` and
    `mean_d_target` give the headroom DDQN exploited and the
    outcome change observed; `sd_d_target` is the per-seed
    Δ_target dispersion (outcome noise floor at this burst)."""
    env_name: str
    burst_index: int
    r: float
    p: float
    slope: float
    mean_d_predictor: float
    mean_d_target: float
    sd_d_target: float
    n_pairs: int


@dataclass(frozen=True, slots=True)
class PerBurstLinkResult:
    """Output of `paired_link_per_burst`: panel of per-(env, burst)
    paired link r-values plus the input shape parameters for the
    bridge to introspect."""
    strata: tuple[PerBurstLinkStratum, ...]
    target: str
    target_reduction: str
    predictor: str
    predictor_reduction: str
    treatment_arm: str
    baseline_arm: str
    pair_by: tuple[str, ...]

    @property
    def n_strata(self) -> int:
        return len(self.strata)


def _pearson_r_p_slope(
    x: np.ndarray, y: np.ndarray,
) -> tuple[float, float, float]:
    """Pearson r + two-sided p-value (Fisher z) + OLS slope β of
    y on x. Returns (nan, nan, nan) if insufficient variance or
    n<3."""
    n = len(x)
    if n < 3 or x.std() == 0 or y.std() == 0:
        return float('nan'), float('nan'), float('nan')
    r = float(np.corrcoef(x, y)[0, 1])
    if not np.isfinite(r):
        return float('nan'), float('nan'), float('nan')
    slope = float(r * y.std(ddof=1) / x.std(ddof=1))
    if abs(r) >= 1.0:
        return r, 0.0, slope
    z = 0.5 * np.log((1 + r) / (1 - r))
    se = 1.0 / np.sqrt(n - 3)
    from scipy.stats import norm
    p = float(2 * (1 - norm.cdf(abs(z) / se)))
    return r, p, slope


@analysis(reads=('mc_return', 'predicted_q_at_start'))
def paired_link_per_burst(
    cells: Iterable[Mapping[str, object]],
    *,
    treatment_arm: str,
    baseline_arm: str,
    pair_by: tuple[str, ...] = ('seed',),
    target: str = 'mc_return',
    target_reduction: str = 'mean',
    predictor: str = 'mc_return',
    predictor_reduction: str = 'mc_minus_q',
    env_name: str | None = None,
    arm_field: str = 'intervention_name',
) -> PerBurstLinkResult:
    """Per-(env, burst) paired link r(Δ_target, Δ_predictor) panel.

    For each cell, project both `target` and `predictor` to per-
    burst vectors. For each (env, burst), pair treatment ↔
    baseline on `pair_by`, compute Δ_target and Δ_predictor across
    seeds, then Pearson r between them.

    Default config tests the **mech → outcome link**:
      target = mc_return with reduction='mean' (per-burst outcome)
      predictor = mc_return with reduction='mc_minus_q'
                  (per-burst Jensen bias = E[Q] - E[MC])

    `r` is computed against the *negated* predictor so the value
    reads "active link = positive r" (matches the bias-correction
    framing in the DDQN literature: more reduction → more outcome
    gain). Positive r at burst b means: at this point in training,
    more bias reduction by DDQN translates to bigger outcome
    benefit (the textbook story). Negative r means the
    relationship has flipped (Q-explosion-induced anti-link).

    `env_name`, when supplied, restricts the analysis to one env."""
    by_env_arm: dict[tuple[str, str], dict[
        tuple[object, ...], tuple[np.ndarray, np.ndarray],
    ]] = {}
    for cell in cells:
        env = cell.get('env_name')
        arm = cell.get(arm_field)
        if not isinstance(env, str) or not isinstance(arm, str):
            continue
        if env_name is not None and env != env_name:
            continue
        if arm not in (treatment_arm, baseline_arm):
            continue
        target_v = cell_burst_values(cell, target, target_reduction)
        predictor_v = cell_burst_values(cell, predictor, predictor_reduction)
        if target_v.size == 0 or predictor_v.size == 0:
            continue
        if target_v.shape[0] != predictor_v.shape[0]:
            continue
        bucket = by_env_arm.setdefault((env, arm), {})
        key = tuple(cell[k] for k in pair_by)
        bucket[key] = (target_v, predictor_v)

    strata: list[PerBurstLinkStratum] = []
    envs = {env for (env, _) in by_env_arm.keys()}
    for env in sorted(envs):
        treat = by_env_arm.get((env, treatment_arm), {})
        base = by_env_arm.get((env, baseline_arm), {})
        paired_keys = sorted(set(treat) & set(base))
        if not paired_keys:
            continue
        n_bursts = treat[paired_keys[0]][0].shape[0]
        for k in paired_keys:
            if (
                treat[k][0].shape[0] != n_bursts
                or base[k][0].shape[0] != n_bursts
            ):
                raise ValueError(
                    f'{env}: per-burst vector length mismatch across pairs',
                )

        for b in range(n_bursts):
            d_target = np.array([
                treat[k][0][b] - base[k][0][b] for k in paired_keys
            ], dtype=np.float64)
            d_predictor = np.array([
                treat[k][1][b] - base[k][1][b] for k in paired_keys
            ], dtype=np.float64)
            finite = np.isfinite(d_target) & np.isfinite(d_predictor)
            d_t = d_target[finite]
            d_p = d_predictor[finite]
            # Convention: link strength is positive when active.
            # The natural primitive is "bias reduction → outcome
            # gain": more reduction (i.e., more *negative* Δ_jens
            # under DDQN's mc_minus_q) correlates with more positive
            # Δ_outcome. We negate Δ_predictor so the reported r and
            # slope read positive-when-active, matching the
            # bias-correction framing in the DDQN literature
            # (Hasselt 2010 et seq.) and the docstring conventions in
            # the bridge zoo (`g_link = +0.34, link works`).
            r, p, slope = _pearson_r_p_slope(-d_p, d_t)
            mean_p = float(d_p.mean()) if d_p.size else float('nan')
            mean_t = float(d_t.mean()) if d_t.size else float('nan')
            sd_t = float(d_t.std(ddof=1)) if d_t.size > 1 else float('nan')
            strata.append(PerBurstLinkStratum(
                env_name=env, burst_index=b,
                r=r, p=p, slope=slope,
                mean_d_predictor=mean_p,
                mean_d_target=mean_t,
                sd_d_target=sd_t,
                n_pairs=int(finite.sum()),
            ))

    return PerBurstLinkResult(
        strata=tuple(strata),
        target=target,
        target_reduction=target_reduction,
        predictor=predictor,
        predictor_reduction=predictor_reduction,
        treatment_arm=treatment_arm,
        baseline_arm=baseline_arm,
        pair_by=pair_by,
    )


def phase_link_consistency(
    result: PerBurstLinkResult,
    *,
    env_name: str | None = None,
    significance: float = 0.05,
    expected_sign: int = +1,
) -> float:
    """Scalar derived from a `PerBurstLinkResult`: proportion of
    bursts where the link r matches `expected_sign` AND p <
    `significance`. High = link is uniformly active across
    training; low = phase-dependent (Phase 1 holds, Phase 2 doesn't,
    etc.).

    Default `expected_sign=+1` matches the bias-correction story
    under the panel's positive-when-active convention: r > 0 means
    more bias reduction → more outcome gain.

    Returns nan if the panel has no strata for the env."""
    panel = (
        result.strata if env_name is None
        else tuple(s for s in result.strata if s.env_name == env_name)
    )
    if not panel:
        return float('nan')
    sign_match = sum(
        1 for s in panel
        if (
            np.isfinite(s.r) and np.isfinite(s.p)
            and s.p < significance
            and (s.r * expected_sign) > 0
        )
    )
    return sign_match / len(panel)


__all__ = [
    'PerBurstLinkResult',
    'PerBurstLinkStratum',
    'paired_link_per_burst',
    'phase_link_consistency',
]
