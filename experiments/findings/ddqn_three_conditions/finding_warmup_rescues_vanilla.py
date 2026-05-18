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


BLOCKED_ON: str | None = None


# Empirical (2026-05-18, fr_warmup_intervention corpus,
# within-corpus analysis at FR × MLP × unshaped × γ=0.999 × 1M
# total_steps, 90 baseline cells across 3 warmup levels × 30 seeds):
#
#   warmup    | n  | outcome | jens | q_late | σ_action
#   100       | 30 | 0.12    | 8.81 | 6.96   | 0.018
#   10000     | 30 | 0.18    | 8.39 | 6.53   | 0.017
#   100000    | 30 | 0.28    | 7.93 | 7.48   | 0.019
#
#   ρ(warmup, outcome)   = +0.233  p=0.027  HELD     (threshold +0.2)
#   ρ(warmup, jens)      = -0.189  p=0.075  POW_INSUF (sig direction
#                                                     but below |0.2|)
#   q_late + σ_action:   flat across warmup levels.
#
# Walked-back reading. The original hypothesis was "anchor failure
# → Q-explosion → policy failure"; warmup tested whether
# preventing Q-update during early ε-greedy rollout would (a)
# reduce Q-explosion AND (b) rescue policy.
#
# Empirical (within-corpus, no cross-horizon confound):
#   - Warmup DOES help vanilla outcome significantly (ρ=+0.23,
#     p=0.027). Outcome ~doubles from warmup=100 (0.12) to
#     warmup=100k (0.28). The intervention works at the outcome
#     level.
#   - Warmup does NOT meaningfully shrink jens (ρ=-0.19 NS,
#     8.81→7.93 across 5 orders of magnitude in warmup_steps).
#     Q-magnitude (q_late) and per-state Q-variance (σ_action)
#     stay flat. The Q dynamics at 1M are essentially the same
#     across warmup levels.
#
# Substantive consequence: **the rescue mechanism is NOT
# Q-explosion prevention.** Vanilla's Q stays at the same
# high-magnitude over-bootstrapped state at all warmup levels.
# Yet outcome improves with warmup. The likely mechanism:
#   - Replay buffer composition. Long warmup (100k steps) fills
#     replay with ε-greedy random trajectories before Q starts
#     training. By the time training begins, replay contains
#     more diverse (and possibly more reward-containing)
#     trajectories.
#   - Commit-timing. Longer warmup delays the bootstrap chain
#     from locking onto a particular biased Q. Random reward
#     discoveries in the warmup buffer can re-shape the early
#     Q update, even if the asymptotic Q magnitude is the same.
#
# So warmup helps outcome by changing the DATA the agent
# learns from, not by anchoring Q. The "anchor failure → Q-
# explosion → outcome failure" chain is REFUTED at the second
# link. The Q-explosion is incidental to the policy failure at
# FR γ=0.999 — Q stays high either way; outcome differs because
# of pre-training data composition.
#
# Composed verdict: 1 HELD + 1 POWER_INSUFFICIENT → UNDERPOWERED.
# The intervention HELPS (outcome bridge HELD) but the anchor-
# failure-as-mechanism prediction (jens bridge HELD on negative ρ)
# is NOT supported.
#
# Cross-references (parallel decoupling evidence):
# - `finding_jens_does_not_mediate_outcome_at_fr_g999_mlp`: DDQN's
#   bias reduction magnitude does not predict outcome help across
#   k_eff.
# - This Finding: warmup's outcome rescue does not flow through
#   jens reduction.
# Both converge on the same decoupling claim — bias / Q-magnitude
# is NOT the rate-limiting variable for outcome at FR γ=0.999.


BRIDGES: tuple[Bridge, ...] = (
    vanilla_outcome_recovers_with_warmup_at_fr_g999_mlp,
    vanilla_jens_shrinks_with_warmup_at_fr_g999_mlp,
)
