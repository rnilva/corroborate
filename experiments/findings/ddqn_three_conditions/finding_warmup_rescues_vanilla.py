"""Causal-intervention test: deferring Q-updates rescues vanilla
at FR γ=0.999.

The first **interventional-tier** Finding in this module. Prior
WHY-chain Findings (`finding_sigma_action_completes_chain`,
`finding_gamma_amplification_anchor_gated`) are observational —
they characterize correlations across existing cells. This Finding
manipulates `optimizer.warmup_steps` to test whether the
anchor-failure mechanism is CAUSALLY responsible for vanilla's
Q-explosion at FR γ=0.999.

The intervention: `warmed_update.warmup_steps` zeros Q-parameter
updates for the first N training steps (per
`claims/optimizer.py:warmed_update`'s step-function schedule —
`where(count < warmup_steps, 0.0, 1.0)`). ε-greedy still rolls
out during warmup; replay fills. After warmup, training resumes
on the prefilled buffer.

Test corpus: `experiments/probes/fr_warmup_intervention` (30 seeds
× 2 arms × 3 warmup levels = 180 cells). Warmup levels: 100
(current default — Q-explosion regime), 10k (some seeds find
goal during warmup), 100k (most seeds find goal during warmup).

Two bridges in `jens_reduction_factors.py`:

1. `vanilla_outcome_recovers_with_warmup_at_fr_g999_mlp`:
   per-cell Spearman ρ(warmup_steps, eval_best_burst_raw_mean)
   over baseline cells. Predicted ρ ≥ +0.5: longer warmup →
   higher vanilla outcome.

2. `vanilla_jens_shrinks_with_warmup_at_fr_g999_mlp`:
   per-cell Spearman ρ(warmup_steps, jensen_gap) over baseline
   cells. Predicted ρ ≤ -0.5: longer warmup → smaller jens.

If both HELD, the causal claim is supported: **vanilla's Q-explosion
at FR γ=0.999 is triggered by training before reward observation;
deferring Q-updates allows ε-greedy to find the goal, MC then
anchors Q, and the explosion is prevented.**

If `vanilla_outcome_recovers_*` REFUTES (NO_EFFECT or
POWER_INSUFFICIENT), the anchor-failure mechanism is at most
PARTIAL — some other mechanism (exploration inefficiency
independent of Q-drift speed; FA-bias-amplification independent
of reward observation; ε-greedy structural failure at long horizon)
contributes to the FR γ=0.999 failure mode.

Related Findings (read together):
- `finding_gamma_amplification_anchor_gated`: cross-env evidence
  that γ-amplification co-occurs with anchor collapse — the
  observational version of this same claim.
- `finding_sigma_action_completes_chain`: the FR-baseline
  mediation structure (γ → jens via {self_ref, σ_action}) — what
  ELSE moves when the anchor mechanism fires.

**Pre-ingest state**: bridges return empty extent until the
`fr_warmup_intervention` corpus lands and is ingested. Set
EXPECTED to the empirical state once data arrives; for now,
EXPECTED is UNDERPOWERED with BLOCKED_ON describing the gap."""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn_three_conditions.jens_reduction_factors import (
    vanilla_jens_shrinks_with_warmup_at_fr_g999_mlp,
    vanilla_outcome_recovers_with_warmup_at_fr_g999_mlp,
)


EXPECTED: ClusterVerdict = ClusterVerdict.UNDERPOWERED


BLOCKED_ON: str | None = (
    'awaiting fr_warmup_intervention corpus ingest — '
    '180 cells (3 warmup × 2 arms × 30 seeds) on FR × γ=0.999 × '
    'MLP × unshaped pending. Sweep config at '
    'experiments/configs/fr_warmup_intervention.yaml.'
)


BRIDGES: tuple[Bridge, ...] = (
    vanilla_outcome_recovers_with_warmup_at_fr_g999_mlp,
    vanilla_jens_shrinks_with_warmup_at_fr_g999_mlp,
)
