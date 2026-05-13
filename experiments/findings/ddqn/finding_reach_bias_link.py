"""DDQN's bootstrap clip does NOT translate to outcome benefit
across envs at canonical scope — REFUTED.

The cross-env dose-response cluster (anchor Spearman + LOO
robustness sibling) fires NO_EFFECT on the canonical 1M panel:

  anchor   : ρ = +0.105, p = 0.75 (n=10 envs after saturation
             guard drops CartPole / FourRooms)
  LOO      : min(ρ_LOO) far below the +0.3 robustness gate

The substantive reading: at canonical scope (1 config per env,
sync/replay/network at env-class-canonical defaults), envs where
DDQN induces a bigger Δ in `bootstrap_gap_magnitude` do NOT
systematically see a bigger Δ in `eval_best_burst_raw_mean`. The
per-step clip inequality (`findings_clip_to_trained_q_propagation`)
remains structurally true; its translation to env-level
outcome-benefit is empirically null at the population-of-envs
scope where the practitioner-facing claim lives.

Renamed from `finding_reach_bias_link` (REACH-cohort jens→outcome
test, retired 2026-05-13 because tautological). The
previous cross-stratum DoWhy cluster
(`bias_correction_clip_predicts_outcome_*`) was deleted — it
asked a pooled-cross-stratum slope question that at canonical
n_strata=12 was structurally underpowered; the question itself
needed more strata than canonical provides. The cleanly-scoped
cross-env dose-response replaces it.

Anchor + robustness cluster shape per HYPOTHESIS_AS_GRAPH.md
§3b — composed_verdict AND-aggregates the two NO_EFFECT members
into REFUTED at the Finding level."""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn.bias_correction_xenv import (
    bias_correction_dose_response__xenv_arm_diff,
    bias_correction_dose_response__xenv_arm_diff_loo_robust,
)


EXPECTED: ClusterVerdict = ClusterVerdict.REFUTED


BLOCKED_ON: str | None = None


BRIDGES: tuple[Bridge, ...] = (
    bias_correction_dose_response__xenv_arm_diff,
    bias_correction_dose_response__xenv_arm_diff_loo_robust,
)
