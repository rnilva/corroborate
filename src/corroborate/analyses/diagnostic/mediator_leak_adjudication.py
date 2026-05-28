"""`mediator_leak_adjudication` v3 — env-conditional adjudication for
soft-tautological mediators, using paired McNemar test + power-aware
disposition + multiplicity hook.

The strict `tautology_audit` primitive flags any mediator whose
`reads` overlap with the outcome's `reads`. That verdict is binary
(`OUTCOME` or `clean`) and conservative — it cannot distinguish:

  * **Hard tautology**: mediator IS a restatement of outcome inputs.
  * **Soft tautology**: mediator combines outcome inputs with
    INDEPENDENT inputs (e.g. `bias = Q − MC` combines outcome's
    `mc_return` with the independent `predicted_q_per_step`).

This primitive runs the empirical adjudication test. v3 supersedes
v2 (which had structural issues a critic review surfaced: noise-pad
"df-matching" was theatre, independent-binomial Wald SE under-stated
SE for paired data, no UNDERPOWERED disposition, no multiplicity
correction).

**v3 design**:

1. Two PC runs (no noise-pad): `{sibling}` at depth-1 vs
   `{mediator, sibling}` at depth-2. The runs use the SAME
   marg-edge bursts (depth-0 marg test is mediator-independent),
   so per-burst d-sep booleans are paired.

2. **McNemar paired test** on the discordant pairs. Among bursts
   with marg_edge=True, let `n_01` = #bursts where joint
   d-separates but sibling does not (joint > sibling evidence),
   `n_10` = #bursts where sibling d-separates but joint does not
   (anomaly). Continuity-corrected z:

   ```
   z = (n_01 − n_10) / sqrt(n_01 + n_10)   if n_01 + n_10 ≥ 5
   ```

   This is the correct paired test — the asymmetric-depth concern
   is rolled into the fact that joint can only add information at
   the population level, not subtract it; finite-sample violations
   are pure noise in `n_10`.

3. **Power-aware disposition**:

   - `z ≥ z_genuine` (default 1.65, optionally Bonferroni-adjusted
     via `n_strata_for_multiplicity`): **GENUINE** — mediator
     carries information beyond sibling's rank-monotone span at
     this stratum.
   - `|z| < z_genuine` AND `n_01 + n_10 < min_discordant`: **UNDERPOWERED_FOR_GENUINE**
     — insufficient discordant pairs to detect a significant
     effect. Distinguishes "no evidence" from "evidence of no
     effect."
   - `|z| < z_genuine` AND `n_01 + n_10 ≥ min_discordant`: **LEAK**
     — adequately powered to detect Δ, none found. Mediator is
     consistent with outcome-leak alone.
   - `n_marg_edge < min_marginal_edges`: **UNDERPOWERED** (the
     entire stratum has too few marg-edge bursts).

4. **Multiplicity correction (Bonferroni)**: caller can pass
   `n_strata_for_multiplicity` to switch the GENUINE threshold to
   `z(1 − α / n_strata)` (one-sided). Conservative; doesn't
   pre-screen UNDERPOWERED strata from the multiplicity count.

**Interpretation caveat** (preserved from v2): GENUINE means
"mediator's signal extends beyond the **rank-monotone** span of
sibling" (partial-Spearman conditions on rank residuals, not
linear span — v2's "linear span" wording was technically wrong).
It is necessary-but-not-sufficient for the mediator's claimed
causal mechanism. At γ=0.99 with `bias = Q − MC`, GENUINE detects
Q-information beyond mean-MC, which could equally well be (a) the
bias-clip mechanism OR (b) Q's independent causal effects via
state-visitation / exploration. The test cannot distinguish.

Reference: see `TAUTOLOGY_AUDIT_DISCIPLINE.md`."""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

import polars as pl
from scipy import stats

from corroborate.analyses.dynamic_mediation.pc_adjacency import (
    DynamicPCResult, dynamic_pc_adjacency,
)
from corroborate.bridge.analysis import analysis
from corroborate.measurables.measurable import Measurable


class LeakAdjudication(Enum):
    """Per-stratum disposition for a soft-tautological mediator.

    GENUINE: McNemar z ≥ z_genuine. Mediator's discordant pairs
      favor joint over sibling significantly. Necessary-but-not-
      sufficient for the mediator's mechanism claim.
    LEAK: z < z_genuine AND discordant n ≥ min_discordant. The
      test was adequately powered to detect a real effect and
      found none.
    UNDERPOWERED_FOR_GENUINE: z < z_genuine but discordant n <
      min_discordant. Cannot adjudicate at this n.
    UNDERPOWERED: n_marg_edge < min_marginal_edges — the entire
      stratum lacks marg-edge bursts for the test."""
    GENUINE = 'GENUINE'
    LEAK = 'LEAK'
    UNDERPOWERED_FOR_GENUINE = 'UNDERPOWERED_FOR_GENUINE'
    UNDERPOWERED = 'UNDERPOWERED'


@dataclass(frozen=True, slots=True)
class StratumLeakResult:
    """Per-stratum adjudication record. The discordant counts
    `n_discordant_joint_only` / `n_discordant_sibling_only` are the
    McNemar (b, c) cells; `z_mcnemar` is the paired-test statistic
    (continuity-corrected for small n)."""
    stratum_id: tuple[object, ...]
    n_marginal_edges: int
    dsep_sibling_only: float
    dsep_joint: float
    n_discordant_joint_only: int    # joint d-sep, sib does not
    n_discordant_sibling_only: int  # sib d-sep, joint does not
    z_mcnemar: float
    disposition: LeakAdjudication


@dataclass(frozen=True, slots=True)
class MediatorLeakAdjudicationResult:
    per_stratum: tuple[StratumLeakResult, ...]
    mediator_name: str
    sibling_name: str
    outcome_name: str
    z_genuine_threshold: float
    n_strata_for_multiplicity: int | None

    def by_disposition(
        self, disposition: LeakAdjudication,
    ) -> tuple[tuple[object, ...], ...]:
        return tuple(
            s.stratum_id for s in self.per_stratum
            if s.disposition is disposition
        )

    def summary(self) -> dict[LeakAdjudication, int]:
        counts: dict[LeakAdjudication, int] = {d: 0 for d in LeakAdjudication}
        for s in self.per_stratum:
            counts[s.disposition] += 1
        return counts


def _per_burst_booleans(
    result_dict: Mapping[tuple[object, ...], DynamicPCResult],
    stratum_id: tuple[object, ...],
    alpha: float,
) -> tuple[list[bool], list[bool]]:
    """For a stratum, return (marg_edge_per_burst, dsep_per_burst)
    booleans. Derived from p_marginal/p_conditional + alpha."""
    res = result_dict.get(stratum_id)
    if res is None:
        return [], []
    marg = []
    dsep = []
    for p_m, p_c in zip(res.p_marginal, res.p_conditional, strict=False):
        marg_edge = not math.isnan(p_m) and p_m < alpha
        if marg_edge:
            # cond CI: edge present iff p_c < alpha. dsep iff edge absent.
            cond_edge_absent = math.isnan(p_c) or p_c >= alpha
            dsep_b = cond_edge_absent
        else:
            dsep_b = False
        marg.append(marg_edge)
        dsep.append(dsep_b)
    return marg, dsep


def _mcnemar_z(
    n_01: int, n_10: int, *,
    min_discordant: int = 5,
    exact_threshold: int = 25,
) -> float:
    """One-sided McNemar z for "joint d-separates more often than sibling."

    n_01 = bursts where joint d-sep, sib not. n_10 = the opposite.
    Returns NaN when discordant pairs < min_discordant.

    At small discordant counts (n_disc < exact_threshold, default 25)
    the normal approximation breaks down; uses the EXACT one-sided
    binomial test instead and converts to z-equivalent via probit:
    `z = Φ⁻¹(1 − p_exact)`. This gives the appropriate small-sample
    rigor that the standard McNemar literature recommends.

    At n_disc ≥ exact_threshold, uses Edwards continuity-corrected
    normal approximation:
      `z = (n_01 − n_10 − sign(Δ)) / sqrt(n_disc)`."""
    n_disc = n_01 + n_10
    if n_disc < min_discordant:
        return float('nan')
    if n_disc < exact_threshold:
        # Exact one-sided binomial: P(X ≥ n_01 | n_disc, p=0.5). Under
        # H₀ (population monotonicity → equal discordant proportions),
        # the larger of (n_01, n_10) is binomial(n_disc, 0.5).
        from scipy.stats import binom, norm  # pyright: ignore[reportAttributeAccessIssue]
        p_exact = 1.0 - binom.cdf(n_01 - 1, n_disc, 0.5)
        # Convert to z via probit. Clip to avoid +/-inf at boundary.
        p_exact = max(min(p_exact, 1 - 1e-15), 1e-15)
        return float(norm.ppf(1 - p_exact))
    return (n_01 - n_10 - (1 if n_01 > n_10 else -1)) / math.sqrt(n_disc)


@analysis
def mediator_leak_adjudication(
    cells: pl.DataFrame,
    *,
    mediator_per_burst: str,
    sibling_per_burst: str,
    outcome_per_burst: str | Measurable[Mapping[str, object], object],
    arm_field: str = 'arm_key',
    stratify_by: tuple[str, ...] = ('env_name',),
    min_n_per_burst: int = 8,
    min_marginal_edges: int = 3,
    min_discordant: int = 5,
    z_genuine: float = 1.65,
    alpha: float = 0.05,
    n_strata_for_multiplicity: int | None = None,
) -> MediatorLeakAdjudicationResult:
    """Adjudicate a soft-tautological mediator against its outcome-input
    sibling via McNemar paired test on per-burst d-separation booleans.

    `mediator_per_burst` and `sibling_per_burst` must be string column
    names; the primitive uses them to query `dynamic_pc_adjacency`
    at depth-1 (sibling alone) and depth-2 (mediator + sibling).

    `n_strata_for_multiplicity`: if given, switches `z_genuine` to the
    Bonferroni-adjusted one-sided z corresponding to α / n_strata. Use
    the total number of strata tested across the audit (i.e. NOT just
    the n_strata that came back GENUINE)."""
    # Resolve the effective z_genuine.
    if n_strata_for_multiplicity is not None and n_strata_for_multiplicity > 1:
        alpha_corrected = (1 - stats.norm.cdf(z_genuine)) / n_strata_for_multiplicity
        z_genuine_eff = stats.norm.ppf(1 - alpha_corrected)
    else:
        z_genuine_eff = z_genuine

    if cells.height == 0:
        return MediatorLeakAdjudicationResult(
            per_stratum=(),
            mediator_name=str(mediator_per_burst),
            sibling_name=str(sibling_per_burst),
            outcome_name=str(outcome_per_burst),
            z_genuine_threshold=z_genuine_eff,
            n_strata_for_multiplicity=n_strata_for_multiplicity,
        )

    # **NaN-coupling pre-filter** (third-critic-review fix). The
    # depth-1 sibling-only run drops cells with NaN in `sibling`; the
    # depth-2 joint run drops cells with NaN in EITHER `sibling` OR
    # `mediator`. Without pre-filtering, the two runs operate on
    # different sample sets → different marg-edge bursts → McNemar
    # pairing assumption violated. Pre-filter to cells where BOTH
    # columns are entirely finite across the burst axis so the two
    # runs share a common cell-level domain.
    import numpy as _np
    def _all_finite(arr: object) -> bool:
        if arr is None:
            return False
        a = _np.asarray(arr, dtype=_np.float64)
        if a.size == 0:
            return False
        return bool(_np.all(_np.isfinite(a)))
    med_finite = [_all_finite(x) for x in cells[mediator_per_burst].to_list()]
    sib_finite = [_all_finite(x) for x in cells[sibling_per_burst].to_list()]
    both_finite = [m and s for m, s in zip(med_finite, sib_finite, strict=True)]
    cells_aligned = cells.filter(pl.Series(both_finite))
    if cells_aligned.height == 0:
        return MediatorLeakAdjudicationResult(
            per_stratum=(),
            mediator_name=str(mediator_per_burst),
            sibling_name=str(sibling_per_burst),
            outcome_name=str(outcome_per_burst),
            z_genuine_threshold=z_genuine_eff,
            n_strata_for_multiplicity=n_strata_for_multiplicity,
        )

    common_kwargs = dict(
        arm_field=arm_field,
        outcome_per_burst=outcome_per_burst,
        stratify_by=stratify_by,
        min_n_per_burst=min_n_per_burst,
        alpha=alpha,
    )
    sibling_res = dynamic_pc_adjacency.fn(
        cells_aligned,
        mediator_per_burst=sibling_per_burst,
        **common_kwargs,
    )
    joint_res = dynamic_pc_adjacency.fn(
        cells_aligned,
        mediator_per_burst=(mediator_per_burst, sibling_per_burst),
        **common_kwargs,
    )

    strata = sorted(set(sibling_res.keys()) & set(joint_res.keys()))
    per_stratum: list[StratumLeakResult] = []
    for s_id in strata:
        sib_marg, sib_dsep = _per_burst_booleans(sibling_res, s_id, alpha)
        joint_marg, joint_dsep = _per_burst_booleans(joint_res, s_id, alpha)
        # marg-edge set must be identical across the two runs because
        # the depth-0 marg test is mediator-independent. Take the
        # intersection defensively (handles edge cases where one run
        # had NaN p-value at a burst).
        n_bursts = min(len(sib_marg), len(joint_marg))
        marg_intersect = [
            sib_marg[b] and joint_marg[b] for b in range(n_bursts)
        ]
        n_marg = sum(marg_intersect)
        if n_marg < min_marginal_edges:
            per_stratum.append(StratumLeakResult(
                stratum_id=s_id,
                n_marginal_edges=n_marg,
                dsep_sibling_only=float('nan'),
                dsep_joint=float('nan'),
                n_discordant_joint_only=0,
                n_discordant_sibling_only=0,
                z_mcnemar=float('nan'),
                disposition=LeakAdjudication.UNDERPOWERED,
            ))
            continue

        # Restrict to marg-edge bursts; compute paired d-sep counts.
        n_sib_dsep = sum(
            sib_dsep[b] for b in range(n_bursts) if marg_intersect[b]
        )
        n_joint_dsep = sum(
            joint_dsep[b] for b in range(n_bursts) if marg_intersect[b]
        )
        n_01 = sum(  # joint d-sep, sib does not
            (joint_dsep[b] and not sib_dsep[b])
            for b in range(n_bursts) if marg_intersect[b]
        )
        n_10 = sum(  # sib d-sep, joint does not
            (sib_dsep[b] and not joint_dsep[b])
            for b in range(n_bursts) if marg_intersect[b]
        )

        z = _mcnemar_z(n_01, n_10, min_discordant=min_discordant)
        n_disc = n_01 + n_10

        if math.isnan(z):
            disposition = LeakAdjudication.UNDERPOWERED_FOR_GENUINE
        elif z >= z_genuine_eff:
            disposition = LeakAdjudication.GENUINE
        elif n_disc < min_discordant:
            disposition = LeakAdjudication.UNDERPOWERED_FOR_GENUINE
        else:
            disposition = LeakAdjudication.LEAK

        per_stratum.append(StratumLeakResult(
            stratum_id=s_id,
            n_marginal_edges=n_marg,
            dsep_sibling_only=n_sib_dsep / n_marg * 100,
            dsep_joint=n_joint_dsep / n_marg * 100,
            n_discordant_joint_only=n_01,
            n_discordant_sibling_only=n_10,
            z_mcnemar=z,
            disposition=disposition,
        ))

    return MediatorLeakAdjudicationResult(
        per_stratum=tuple(per_stratum),
        mediator_name=str(mediator_per_burst),
        sibling_name=str(sibling_per_burst),
        outcome_name=str(outcome_per_burst),
        z_genuine_threshold=z_genuine_eff,
        n_strata_for_multiplicity=n_strata_for_multiplicity,
    )


__all__ = [
    'LeakAdjudication',
    'StratumLeakResult',
    'MediatorLeakAdjudicationResult',
    'mediator_leak_adjudication',
]
