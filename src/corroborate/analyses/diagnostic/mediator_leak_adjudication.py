"""`mediator_leak_adjudication` — env-conditional adjudication for
soft-tautological mediators.

The strict `tautology_audit` primitive flags any mediator whose
`reads` overlap with the outcome's `reads`. That verdict is binary
(`OUTCOME` or `clean`) and conservative — it cannot distinguish:

  * **Hard tautology**: mediator IS a restatement of outcome inputs.
  * **Soft tautology**: mediator combines outcome inputs with
    INDEPENDENT inputs and may still carry genuine causal information.

This primitive runs the empirical adjudication test laid out in
`TAUTOLOGY_AUDIT_DISCIPLINE.md`:

  Compare per-stratum d-separation under three multi-mediator
  conditioning sets via `dynamic_pc_adjacency`:

    1. `{sibling}`        — what the outcome-input leak alone explains
    2. `{mediator}`        — the full mediator's apparent power
    3. `{mediator, sibling}` — does conditioning on BOTH improve over
                                sibling alone?

  Per-stratum `Δ = dsep({mediator, sibling}) − dsep({sibling})`
  classifies the mediator at that stratum:

    Δ ≥ +genuine_threshold (default +10pp) → GENUINE
    Δ <  +genuine_threshold and ≥ -hurts_threshold (default -5pp) → LEAK
    Δ < -hurts_threshold → HURTS (mediator contaminates conditioning)
    n_marg < min_marginal_edges (default 3) → UNDERPOWERED

The result is per-stratum dispositions + the underlying d-sep
numbers; bridges consume this to make scope-conditional claims
("mediator M is GENUINE at envs {…}; LEAK at envs {…}").

Reference implementation: `papers/g099_mediation/scripts/
gen_mc_leak_adjudication.py`."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

import polars as pl

from corroborate.analyses.dynamic_mediation.pc_adjacency import (
    dynamic_pc_adjacency,
)
from corroborate.bridge.analysis import analysis
from corroborate.measurables.measurable import Measurable


class LeakAdjudication(Enum):
    """Per-stratum disposition for a soft-tautological mediator.

    GENUINE: joint conditioning adds ≥ genuine_threshold d-sep beyond
      the sibling alone. The mediator carries independent-input info.
    LEAK: joint adds < genuine_threshold (default +10pp). The
      mediator is mostly outcome-leak at this stratum.
    HURTS: joint d-sep DECREASES by hurts_threshold (default -5pp).
      Mediator is contaminating the conditioning set; refuse.
    UNDERPOWERED: fewer than min_marginal_edges marginal-edge bursts
      to support the adjudication."""
    GENUINE = 'GENUINE'
    LEAK = 'LEAK'
    HURTS = 'HURTS'
    UNDERPOWERED = 'UNDERPOWERED'


@dataclass(frozen=True, slots=True)
class StratumLeakResult:
    """Per-stratum adjudication record for one (mediator, sibling) pair.

    `dsep_*` fields are the d-separation percentage (0..100) under
    each conditioning set, taken among bursts where the depth-0
    marginal arm↔outcome edge is detected. `delta_pp` is the gain in
    percentage points of `{mediator, sibling}` over `{sibling}`."""
    stratum_id: tuple[object, ...]
    n_marginal_edges: int
    dsep_mediator_alone: float
    dsep_sibling_alone: float
    dsep_joint: float
    delta_pp: float
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


@analysis
def mediator_leak_adjudication(
    cells: pl.DataFrame,
    *,
    mediator_per_burst: str | Measurable[Mapping[str, object], object],
    sibling_per_burst: str | Measurable[Mapping[str, object], object],
    outcome_per_burst: str | Measurable[Mapping[str, object], object],
    arm_field: str = 'arm_key',
    stratify_by: tuple[str, ...] = ('env_name',),
    min_n_per_burst: int = 8,
    min_marginal_edges: int = 3,
    genuine_threshold_pp: float = 10.0,
    hurts_threshold_pp: float = -5.0,
    alpha: float = 0.05,
) -> MediatorLeakAdjudicationResult:
    """Adjudicate a soft-tautological mediator against its outcome-input
    sibling, per stratum.

    `mediator_per_burst` is the candidate mediator (typically a
    soft-tautology like `mean_per_state_cumulative_bias_per_burst`).
    `sibling_per_burst` is the matched outcome-input-only column
    (e.g. `mean_mc_per_state_per_burst` for the bias mediator). The
    sibling is what would remain if the mediator were "merely tautology."

    For each stratum we run `dynamic_pc_adjacency` three times:
      - mediator alone
      - sibling alone
      - joint {mediator, sibling}
    and compare d-separation rates among the bursts where the
    depth-0 marginal arm↔outcome edge is detected.

    Returns a per-stratum disposition (`GENUINE` / `LEAK` / `HURTS`
    / `UNDERPOWERED`) plus the underlying numbers."""
    if cells.height == 0:
        return MediatorLeakAdjudicationResult(
            per_stratum=(),
            mediator_name=str(mediator_per_burst),
            sibling_name=str(sibling_per_burst),
            outcome_name=str(outcome_per_burst),
        )

    mediator_res = dynamic_pc_adjacency.fn(
        cells, arm_field=arm_field,
        mediator_per_burst=mediator_per_burst,
        outcome_per_burst=outcome_per_burst,
        stratify_by=stratify_by, min_n_per_burst=min_n_per_burst,
        alpha=alpha,
    )
    sibling_res = dynamic_pc_adjacency.fn(
        cells, arm_field=arm_field,
        mediator_per_burst=sibling_per_burst,
        outcome_per_burst=outcome_per_burst,
        stratify_by=stratify_by, min_n_per_burst=min_n_per_burst,
        alpha=alpha,
    )
    joint_res = dynamic_pc_adjacency.fn(
        cells, arm_field=arm_field,
        mediator_per_burst=(mediator_per_burst, sibling_per_burst),
        outcome_per_burst=outcome_per_burst,
        stratify_by=stratify_by, min_n_per_burst=min_n_per_burst,
        alpha=alpha,
    )

    # `dynamic_pc_adjacency` returns Mapping[Stratum, DynamicPCResult].
    # Cross stratum keys across the three runs.
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
                disposition=LeakAdjudication.UNDERPOWERED,
            ))
            continue
        m_dsep = m.n_bursts_mediator_dseparates / marg * 100
        sib_dsep = (
            sib.n_bursts_mediator_dseparates / sib.n_bursts_marginal_edge * 100
            if sib.n_bursts_marginal_edge > 0 else 0.0
        )
        joint_dsep = (
            joint.n_bursts_mediator_dseparates / joint.n_bursts_marginal_edge * 100
            if joint.n_bursts_marginal_edge > 0 else 0.0
        )
        delta = joint_dsep - sib_dsep
        if delta >= genuine_threshold_pp:
            disposition = LeakAdjudication.GENUINE
        elif delta < hurts_threshold_pp:
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
