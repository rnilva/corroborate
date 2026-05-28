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
    # Underlying per-stratum PC results. Exposed so callers can
    # access cluster-bootstrap CIs (`bootstrap_marginal`,
    # `bootstrap_partial`, `bootstrap_edge_counts`) when this
    # primitive is called with `n_bootstrap > 0`. McNemar's z
    # itself is NOT bootstrapped here; consumers needing a
    # cluster-robust CI on the d-sep rate read these directly.
    sibling_pc_per_stratum: Mapping[tuple[object, ...], DynamicPCResult]
    joint_pc_per_stratum: Mapping[tuple[object, ...], DynamicPCResult]

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


def _mcnemar_exact_p_one_sided(n_01: int, n_10: int) -> float:
    """Exact one-sided McNemar p-value: P(X ≥ n_01 | n_disc, p=0.5)
    under H₀ that discordant proportions are equal. The one-sided
    direction tests H₁: joint d-separates strictly MORE than
    sibling alone."""
    n_disc = n_01 + n_10
    if n_disc == 0:
        return float('nan')
    from scipy.stats import binom  # pyright: ignore[reportAttributeAccessIssue]
    return float(1.0 - binom.cdf(n_01 - 1, n_disc, 0.5))


def _mcnemar_z_normal(n_01: int, n_10: int) -> float:
    """Edwards continuity-corrected normal-approximation McNemar z.

    Reported as the SUMMARY statistic regardless of n_disc; sign
    matches (n_01 − n_10). Stays well-behaved near 0 when data is
    consistent with H₀ (in contrast to probit-of-exact-p which
    blows up to ±∞ at boundary). The disposition decision is made
    via `_mcnemar_exact_p_one_sided`, not via this z."""
    n_disc = n_01 + n_10
    if n_disc == 0:
        return float('nan')
    delta = n_01 - n_10
    if delta == 0:
        return 0.0
    sign = 1 if delta > 0 else -1
    # Subtract the continuity correction toward zero (so the
    # adjusted |Δ| is one unit smaller; if |Δ|=1 the corrected
    # delta is 0).
    corrected = delta - sign
    return float(corrected) / math.sqrt(n_disc)


@analysis
def mediator_leak_adjudication(
    cells: pl.DataFrame,
    *,
    mediator_per_burst: str,
    sibling_per_burst: str | tuple[str, ...],
    outcome_per_burst: str | Measurable[Mapping[str, object], object],
    arm_field: str = 'arm_key',
    stratify_by: tuple[str, ...] = ('env_name',),
    min_n_per_burst: int = 8,
    min_marginal_edges: int = 3,
    min_discordant: int = 5,
    z_genuine: float = 1.65,
    alpha: float = 0.05,
    n_strata_for_multiplicity: int | None = None,
    n_bootstrap: int = 0,
) -> MediatorLeakAdjudicationResult:
    """Adjudicate a soft-tautological mediator against its outcome-input
    sibling via McNemar paired test on per-burst d-separation booleans.

    `mediator_per_burst` is a single string column name. `sibling_per_burst`
    is either a single string OR a tuple of strings — the multi-input
    form lets the test ask "does mediator add info beyond the JOINT
    sibling set?" (e.g., does bias add info beyond all other Q-summaries
    jointly?). The sibling-only PC runs at depth-k (k = #siblings);
    the joint PC runs at depth-(k+1) with (mediator,) + sibling_tuple.

    `n_strata_for_multiplicity`: if given, switches `z_genuine` to the
    Bonferroni-adjusted one-sided z corresponding to α / n_strata. Use
    the total number of strata tested across the audit (i.e. NOT just
    the n_strata that came back GENUINE).

    Note on df: dynamic_pc_adjacency uses Fisher-z df = n − 3 − k
    where k is the conditioning-set size. Multi-input sibling tests
    consume df fast; ensure `min_n_per_burst` is large enough relative
    to the sibling set size (rule-of-thumb: min_n_per_burst ≥ 8 + k)."""
    # Resolve the effective z_genuine.
    if n_strata_for_multiplicity is not None and n_strata_for_multiplicity > 1:
        alpha_corrected = (1 - stats.norm.cdf(z_genuine)) / n_strata_for_multiplicity
        z_genuine_eff = stats.norm.ppf(1 - alpha_corrected)
    else:
        z_genuine_eff = z_genuine

    # Normalise sibling to a tuple for downstream construction.
    if isinstance(sibling_per_burst, str):
        sibling_tuple: tuple[str, ...] = (sibling_per_burst,)
    else:
        sibling_tuple = tuple(sibling_per_burst)
        if len(sibling_tuple) == 0:
            raise ValueError(
                'sibling_per_burst must be a non-empty tuple of column names'
            )
    # The joint conditioning set: mediator first, then all siblings.
    joint_set: tuple[str, ...] = (mediator_per_burst,) + sibling_tuple
    # Display names: single string stays unwrapped; tuple becomes
    # parenthesised comma-list, matching how authors will write it.
    sibling_display: str = (
        sibling_per_burst if isinstance(sibling_per_burst, str)
        else '(' + ', '.join(sibling_tuple) + ')'
    )

    if cells.height == 0:
        return MediatorLeakAdjudicationResult(
            per_stratum=(),
            mediator_name=str(mediator_per_burst),
            sibling_name=sibling_display,
            outcome_name=str(outcome_per_burst),
            z_genuine_threshold=z_genuine_eff,
            n_strata_for_multiplicity=n_strata_for_multiplicity,
            sibling_pc_per_stratum={},
            joint_pc_per_stratum={},
        )

    common_kwargs = dict(
        arm_field=arm_field,
        outcome_per_burst=outcome_per_burst,
        stratify_by=stratify_by,
        min_n_per_burst=min_n_per_burst,
        alpha=alpha,
        n_bootstrap=n_bootstrap,
    )
    # Sibling-only run: pass the sibling set as-is to dynamic_pc, which
    # already accepts tuple-or-string. Single sibling → depth-1; tuple
    # → depth-k where k = len(tuple).
    sibling_res = dynamic_pc_adjacency.fn(
        cells,
        mediator_per_burst=sibling_tuple if len(sibling_tuple) > 1 else sibling_tuple[0],
        **common_kwargs,
    )
    # Joint run: (mediator, *siblings) at depth-(k+1).
    joint_res = dynamic_pc_adjacency.fn(
        cells,
        mediator_per_burst=joint_set,
        **common_kwargs,
    )

    strata = sorted(set(sibling_res.keys()) & set(joint_res.keys()))
    per_stratum: list[StratumLeakResult] = []
    for s_id in strata:
        sib_marg, sib_dsep = _per_burst_booleans(sibling_res, s_id, alpha)
        joint_marg, joint_dsep = _per_burst_booleans(joint_res, s_id, alpha)
        # **Per-burst NaN-coupling pre-filter** (v5 fix). The depth-1
        # sibling-only run drops cells with NaN in `sibling`; the
        # depth-2 joint run drops cells with NaN in EITHER `sibling`
        # OR `mediator`, so the two runs may use different cell
        # subsets at each burst. Intersect via `n_per_burst >=
        # min_n_per_burst` from both runs to score only the bursts
        # where both had adequate sample size. Earlier "cell-level"
        # pre-filter (v4) was overly strict — discarded usable bursts
        # for a single NaN entry.
        sib_n = sibling_res[s_id].n_per_burst
        joint_n = joint_res[s_id].n_per_burst
        n_bursts = min(len(sib_marg), len(joint_marg))
        valid_burst = [
            (sib_n[b] >= min_n_per_burst and joint_n[b] >= min_n_per_burst)
            for b in range(n_bursts)
        ]
        # marg-edge intersection — only count a burst as a marg-edge
        # pair if (a) both runs sampled enough cells, AND (b) the
        # marg-edge boolean agrees. Since marg is depth-0 and
        # mediator-independent over the SAME cells, the agreement is
        # by construction on `valid_burst` bursts; the AND below is
        # defensive.
        marg_intersect = [
            valid_burst[b] and sib_marg[b] and joint_marg[b]
            for b in range(n_bursts)
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

        n_disc = n_01 + n_10
        # **Decoupled test vs summary** (v5 fix). Disposition uses the
        # appropriate test per sample size: exact binomial at small
        # n_disc, normal-approximation at large. Reported `z_mcnemar`
        # is ALWAYS Edwards continuity-corrected normal (well-behaved
        # signed value bounded near zero under H₀), regardless of
        # which test was used to decide.
        z_reported = _mcnemar_z_normal(n_01, n_10)
        if n_disc < min_discordant:
            disposition = LeakAdjudication.UNDERPOWERED_FOR_GENUINE
        else:
            # Convert z_genuine_eff (a one-sided z) into a target
            # p-value, then compare exact-binomial p to it.
            p_target = float(1 - stats.norm.cdf(z_genuine_eff))
            p_exact = _mcnemar_exact_p_one_sided(n_01, n_10)
            if math.isnan(p_exact):
                disposition = LeakAdjudication.UNDERPOWERED_FOR_GENUINE
            elif p_exact <= p_target:
                disposition = LeakAdjudication.GENUINE
            else:
                disposition = LeakAdjudication.LEAK

        per_stratum.append(StratumLeakResult(
            stratum_id=s_id,
            n_marginal_edges=n_marg,
            dsep_sibling_only=n_sib_dsep / n_marg * 100,
            dsep_joint=n_joint_dsep / n_marg * 100,
            n_discordant_joint_only=n_01,
            n_discordant_sibling_only=n_10,
            z_mcnemar=z_reported,
            disposition=disposition,
        ))

    return MediatorLeakAdjudicationResult(
        per_stratum=tuple(per_stratum),
        mediator_name=str(mediator_per_burst),
        sibling_name=sibling_display,
        outcome_name=str(outcome_per_burst),
        z_genuine_threshold=z_genuine_eff,
        n_strata_for_multiplicity=n_strata_for_multiplicity,
        sibling_pc_per_stratum=dict(sibling_res),
        joint_pc_per_stratum=dict(joint_res),
    )


__all__ = [
    'LeakAdjudication',
    'StratumLeakResult',
    'MediatorLeakAdjudicationResult',
    'mediator_leak_adjudication',
]
