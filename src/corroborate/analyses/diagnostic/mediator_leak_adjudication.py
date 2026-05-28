"""`mediator_leak_adjudication` — env-conditional adjudication for
soft-tautological mediators.

The strict `tautology_audit` primitive flags any mediator whose
`reads` overlap with the outcome's `reads`. That verdict is binary
(`OUTCOME` or `clean`) and conservative — it cannot distinguish:

  * **Hard tautology**: mediator IS a restatement of outcome inputs.
  * **Soft tautology**: mediator combines outcome inputs with
    INDEPENDENT inputs (e.g. `bias = Q − MC` combines outcome's
    `mc_return` with the independent `predicted_q_per_step`).

This primitive runs the empirical adjudication test laid out in
`TAUTOLOGY_AUDIT_DISCIPLINE.md` with **rigor upgrades** flagged by
the 2026-05-28 critic review:

  Compare per-stratum d-separation under THREE depth-2 conditioning
  sets via `dynamic_pc_adjacency`. df-asymmetry (the killer issue in
  the v1 implementation: depth-1 sibling vs depth-2 joint gave a df
  cost rather than an info gain) is fixed by padding the single-
  mediator runs with a noise variate so all three runs are
  depth-2:

    1. `{sibling, ε}`         — what the outcome-input leak alone
                                explains (ε ~ N(0,1) per burst)
    2. `{mediator, ε}`        — what the full mediator alone explains
    3. `{mediator, sibling}`  — does conditioning on BOTH improve
                                over sibling alone?

  Per-stratum `Δ = dsep({mediator, sibling}) − dsep({sibling, ε})`.
  Wald SE on the proportion difference under binomial:
  `SE_Δ = sqrt(p_j(1−p_j)/n + p_s(1−p_s)/n)` where n = n_marg-edge
  bursts. Disposition is `z = Δ / SE_Δ`:

    z ≥ +z_genuine (default 1.65 → 95% one-sided) → GENUINE
    z ≤ −z_hurts (default −1.65)                  → HURTS
    |z| < z_genuine                               → LEAK
    n_marg < min_marginal_edges                   → UNDERPOWERED

**Interpretation caveat** (per critic review): GENUINE means
"the mediator's signal extends beyond the LINEAR span of mean-MC
(the sibling)." It is **necessary-but-not-sufficient** for the
mediator's claimed causal mechanism. At γ=0.99 canonical with
`bias = Q − MC`, GENUINE detects Q-information beyond mean-MC,
which could equally well be (a) the bias-clip mechanism (Hasselt's
claim) OR (b) Q's independent causal effects via state-visitation
/ exploration. The test cannot distinguish.

Higher moments of the mediator (variance, sign concentration) and
nonlinear couplings are not in the sibling-span; the test is
sensitive to the linear part of the MC-leak. Bridges using this
primitive should report disposition plus the substantive question
the test does NOT adjudicate.

Reference implementation: `papers/g099_mediation/scripts/
gen_mc_leak_adjudication.py`."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

import numpy as np
import polars as pl

from corroborate.analyses.dynamic_mediation.pc_adjacency import (
    dynamic_pc_adjacency,
)
from corroborate.bridge.analysis import analysis
from corroborate.measurables.measurable import Measurable


_NOISE_COL = '_leak_adj_noise_per_burst'


class LeakAdjudication(Enum):
    """Per-stratum disposition for a soft-tautological mediator.

    GENUINE: joint conditioning adds info beyond the (df-matched)
      sibling alone at z ≥ z_genuine (default 95% one-sided). The
      mediator carries independent-input signal not in the linear
      span of mean(sibling-reads).
    LEAK: |z| < z_genuine. The mediator is largely outcome-leak at
      this stratum (within the sibling's linear span).
    HURTS: joint d-sep DECREASES significantly (z ≤ −z_hurts).
      Mediator is contaminating the conditioning set; refuse.
    UNDERPOWERED: fewer than `min_marginal_edges` marg-edge bursts
      to support the adjudication."""
    GENUINE = 'GENUINE'
    LEAK = 'LEAK'
    HURTS = 'HURTS'
    UNDERPOWERED = 'UNDERPOWERED'


@dataclass(frozen=True, slots=True)
class StratumLeakResult:
    """Per-stratum adjudication record for one (mediator, sibling) pair.

    All three `dsep_*` fields are computed at **depth-2** conditioning
    via noise-padding (so df is matched across runs and Δ is an
    apples-to-apples comparison)."""
    stratum_id: tuple[object, ...]
    n_marginal_edges: int
    dsep_mediator_alone: float    # d-sep% under {mediator, ε}
    dsep_sibling_alone: float     # d-sep% under {sibling, ε}
    dsep_joint: float             # d-sep% under {mediator, sibling}
    delta_pp: float               # joint − sibling, in percentage points
    se_delta_pp: float            # binomial Wald SE on Δ
    z_score: float                # Δ / SE_Δ
    disposition: LeakAdjudication


@dataclass(frozen=True, slots=True)
class MediatorLeakAdjudicationResult:
    """Output of `mediator_leak_adjudication`. Per-stratum dispositions
    plus the underlying d-sep numbers for transparent inspection."""
    per_stratum: tuple[StratumLeakResult, ...]
    mediator_name: str
    sibling_name: str
    outcome_name: str

    def by_disposition(
        self, disposition: LeakAdjudication,
    ) -> tuple[tuple[object, ...], ...]:
        """Stratum IDs with the given disposition."""
        return tuple(
            s.stratum_id for s in self.per_stratum
            if s.disposition is disposition
        )

    def summary(self) -> dict[LeakAdjudication, int]:
        """Count of strata per disposition."""
        counts: dict[LeakAdjudication, int] = {d: 0 for d in LeakAdjudication}
        for s in self.per_stratum:
            counts[s.disposition] += 1
        return counts


def _inject_noise_column(
    cells: pl.DataFrame, mediator_col: str, *, seed: int,
) -> pl.DataFrame:
    """Add a per-burst N(0,1) noise list-column matched in length
    to `mediator_col`'s per-cell arrays. Used to pad single-mediator
    runs to depth-2 so df is matched against the joint run."""
    rng = np.random.default_rng(seed)
    med_lists = cells[mediator_col].to_list()
    noise: list[list[float] | None] = []
    for arr in med_lists:
        if arr is None:
            noise.append(None)
        else:
            n = len(arr)
            noise.append(rng.standard_normal(n).tolist())
    return cells.with_columns(pl.Series(_NOISE_COL, noise))


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
    z_genuine: float = 1.65,    # 95% one-sided (5% Type I)
    z_hurts: float = 1.65,      # symmetric
    alpha: float = 0.05,
    noise_seed: int = 0,
) -> MediatorLeakAdjudicationResult:
    """Adjudicate a soft-tautological mediator against its outcome-
    input sibling, per stratum, with df-matched conditioning sets +
    binomial-Wald SE for the Δ.

    `mediator_per_burst` is the candidate mediator (typically a
    soft-tautology like `mean_per_state_cumulative_bias_per_burst`).
    `sibling_per_burst` is the matched outcome-input-only column
    (e.g. `mean_mc_per_state_per_burst` for the bias mediator). The
    sibling is what would remain if the mediator were "merely
    tautology" — its reads cover the outcome-overlap but none of
    the mediator's independent inputs.

    For each stratum we run `dynamic_pc_adjacency` three times, all
    at depth-2 (df-matched):

      - {mediator, ε} — single-mediator + ε noise pad
      - {sibling, ε}  — single-sibling + ε noise pad
      - {mediator, sibling} — joint

    Compares joint d-sep% against sibling d-sep% with binomial-Wald
    SE on the proportion difference. The z-score classifies per
    stratum: GENUINE / LEAK / HURTS / UNDERPOWERED.

    Currently requires `mediator_per_burst` and `sibling_per_burst`
    to be string column names (not bare Measurable instances) so the
    noise-padding can use the column's per-cell array lengths.
    Lifting this constraint requires walking the Measurable's
    transitive reads to recover lengths — left for a follow-up."""
    if cells.height == 0:
        return MediatorLeakAdjudicationResult(
            per_stratum=(),
            mediator_name=str(mediator_per_burst),
            sibling_name=str(sibling_per_burst),
            outcome_name=str(outcome_per_burst),
        )

    cells_aug = _inject_noise_column(
        cells, mediator_per_burst, seed=noise_seed,
    )

    common_kwargs = dict(
        arm_field=arm_field,
        outcome_per_burst=outcome_per_burst,
        stratify_by=stratify_by,
        min_n_per_burst=min_n_per_burst,
        alpha=alpha,
    )
    mediator_res = dynamic_pc_adjacency.fn(
        cells_aug,
        mediator_per_burst=(mediator_per_burst, _NOISE_COL),
        **common_kwargs,
    )
    sibling_res = dynamic_pc_adjacency.fn(
        cells_aug,
        mediator_per_burst=(sibling_per_burst, _NOISE_COL),
        **common_kwargs,
    )
    joint_res = dynamic_pc_adjacency.fn(
        cells_aug,
        mediator_per_burst=(mediator_per_burst, sibling_per_burst),
        **common_kwargs,
    )

    strata = sorted(
        set(mediator_res.keys())
        & set(sibling_res.keys())
        & set(joint_res.keys())
    )
    per_stratum: list[StratumLeakResult] = []
    for s_id in strata:
        m = mediator_res[s_id]
        sib = sibling_res[s_id]
        joint = joint_res[s_id]
        marg = m.n_bursts_marginal_edge
        if marg < min_marginal_edges:
            per_stratum.append(StratumLeakResult(
                stratum_id=s_id,
                n_marginal_edges=marg,
                dsep_mediator_alone=float('nan'),
                dsep_sibling_alone=float('nan'),
                dsep_joint=float('nan'),
                delta_pp=float('nan'),
                se_delta_pp=float('nan'),
                z_score=float('nan'),
                disposition=LeakAdjudication.UNDERPOWERED,
            ))
            continue

        m_dsep = m.n_bursts_mediator_dseparates / marg * 100
        sib_marg = sib.n_bursts_marginal_edge
        joint_marg = joint.n_bursts_marginal_edge
        sib_dsep = (
            sib.n_bursts_mediator_dseparates / sib_marg * 100
            if sib_marg > 0 else 0.0
        )
        joint_dsep = (
            joint.n_bursts_mediator_dseparates / joint_marg * 100
            if joint_marg > 0 else 0.0
        )

        # Binomial Wald SE on the proportion difference.
        # SE_Δ = sqrt(p_j(1−p_j)/n_j + p_s(1−p_s)/n_s) — approximate
        # (paired structure would McNemar-style be tighter; treat as
        # conservative upper bound on Type-I error).
        p_j = joint_dsep / 100
        p_s = sib_dsep / 100
        n_j = max(joint_marg, 1)
        n_s = max(sib_marg, 1)
        se_j = (p_j * (1 - p_j) / n_j) ** 0.5
        se_s = (p_s * (1 - p_s) / n_s) ** 0.5
        se_delta_pp = ((se_j ** 2 + se_s ** 2) ** 0.5) * 100
        delta = joint_dsep - sib_dsep
        z = delta / se_delta_pp if se_delta_pp > 1e-9 else 0.0

        if z >= z_genuine:
            disposition = LeakAdjudication.GENUINE
        elif z <= -z_hurts:
            disposition = LeakAdjudication.HURTS
        else:
            disposition = LeakAdjudication.LEAK

        per_stratum.append(StratumLeakResult(
            stratum_id=s_id,
            n_marginal_edges=marg,
            dsep_mediator_alone=m_dsep,
            dsep_sibling_alone=sib_dsep,
            dsep_joint=joint_dsep,
            delta_pp=delta,
            se_delta_pp=se_delta_pp,
            z_score=z,
            disposition=disposition,
        ))

    return MediatorLeakAdjudicationResult(
        per_stratum=tuple(per_stratum),
        mediator_name=str(mediator_per_burst),
        sibling_name=str(sibling_per_burst),
        outcome_name=str(outcome_per_burst),
    )


__all__ = [
    'LeakAdjudication',
    'StratumLeakResult',
    'MediatorLeakAdjudicationResult',
    'mediator_leak_adjudication',
]
