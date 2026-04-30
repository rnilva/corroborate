"""DDQN measurement graph — curated edge set.

The subset of `dqn_bridges` whose verdict matches the bridge's
predicted direction against the corpus it was authored for. This
is the DDQN study's measurement graph after the evidence is in:
each entry here is one *corroborated* edge.

Curation rule:

  - HELD bridge → predicted positive direction confirmed.
  - NO_EFFECT bridge → predicted null-link claim confirmed
    (the bridge's body returns NO_EFFECT when the prediction
    holds; e.g. `ddqn_link_to_outcome_null__converged_subset`).
  - POWER_INSUFFICIENT / sign-flipped bridges are excluded — the
    measurement graph commits only to edges where the data
    speaks.

Excluded from the zoo:

  - `ddqn_reduces_jensen_gap__cartpole` — sign-flipped on |A|=2
    (POWER_INSUFFICIENT). The rev 8 reading "DDQN reduces gap"
    only holds on |A|≥3; CartPole is the structural exception
    captured by the dormancy-active scope bridge.
  - `log_action_dim_drives_jensen_gap_reduction` — homogeneous-HP
    corpus, n_strata=4, POWER_INSUFFICIENT.
  - `jensen_gap_outcome_borderline` — only fires on the 200k DDQN
    corpus (rev 5); on action_dim_sweep the within-env signal is
    insignificant.
  - `log_action_dim_drives_g_mech` — POWER_INSUFFICIENT on the
    expectile_3way joined panel; rev 10's β=−0.39 was on the
    200k corpus where this corpus's per-burst panel doesn't
    reproduce.

The graph is organized by edge role, not by FINDINGS revision —
revisions tell *when* a claim was made; the role tells *what* the
edge IS in the DDQN claim graph (`arm → mechanism → link →
outcome`, plus moderators + identification primitives).
"""
from __future__ import annotations

from experiments.findings.dqn_bridges import (
    # Mechanism activation: arm → mechanism.jensen_gap
    ddqn_reduces_jensen_gap__acrobot,
    ddqn_reduces_jensen_gap__catch,
    ddqn_reduces_jensen_gap__converged_subset,
    ddqn_reduces_jensen_gap__discounting_chain,
    expectile_reduces_jensen_gap_more_than_ddqn__fourrooms,
    nstep_3step_reduces_bias_on_top_of_ddqn,
    # Scope at mechanism: env → invariant.jensen_dormancy_gap
    jensen_premise_active__acrobot,
    jensen_premise_active__cartpole,
    jensen_premise_active__discounting_chain,
    jensen_premise_dormant__catch,
    # Link broken (NO_EFFECT corroborates the null-link claim):
    # mechanism.jensen_gap → outcome
    ddqn_link_to_outcome_null__converged_subset,
    ddqn_outcome_zero_across_bursts__catch,
    nstep_3step_does_not_help_outcome__pool,
    time_to_solve_link_null__pooled,
    # Per-env outcome edges: arm → outcome.* (env-specific)
    ddqn_outcome_stable_across_bursts__fourrooms,
    ddqn_outperforms_expectile_on_outcome__fourrooms,
    ddqn_solves_faster__spaceinvaders,
    nstep_3step_helps_outcome__discounting_chain,
    nstep_3step_hurts_outcome__catch,
    # Factorial decomposition: (greedification × n_step) → outcome
    expectile_reproduces_mechanism_link_disconnect__fourrooms,
    factorial_ddqn_attenuation__fourrooms,
    factorial_variance_amplification__catch,
    # Chain moderators: env-feature → g_link
    log_obs_dim_drives_g_link,
    # Mediator audit (corpus-level filter on candidate mediators)
    greedy_match_late_hp_shadow__cartpole_hp,
    learning_curve_auc_outcome_tautological__cartpole_hp,
    state_coverage_kl_clean_mediator__cartpole_hp,
    # Causal identification: mediator → outcome | HPs (Pearl rung 2)
    state_coverage_kl_causes_outcome,
)


# === Mechanism activation: arm → mechanism.jensen_gap ===
#
# DDQN reduces the empirical Jensen-overestimation gap. Per-env
# (action_dim_sweep) on |A|≥3 envs; pooled (200k corpus) on the
# convergence-conditioned subset. n-step on top of DDQN
# additionally reduces it; expectile reduces it MORE than DDQN
# on FourRooms. Mechanism is robust across these intervention
# axes — the bias-reduction operator works as theory predicts.
MECHANISM_EDGES = (
    ddqn_reduces_jensen_gap__acrobot,
    ddqn_reduces_jensen_gap__catch,
    ddqn_reduces_jensen_gap__discounting_chain,
    ddqn_reduces_jensen_gap__converged_subset,
    nstep_3step_reduces_bias_on_top_of_ddqn,
    expectile_reduces_jensen_gap_more_than_ddqn__fourrooms,
)


# === Scope at mechanism: env → invariant.jensen_dormancy_gap ===
#
# The framework's-own dormancy invariant (gap = max(0, σ·√(2 log
# |A|) − observed_bias)) fires correctly per env. Premise active
# on Acrobot / CartPole / DiscountingChain (DDQN's mechanism has
# room to operate); premise dormant on Catch (the floor exceeds
# observed bias — DDQN's correction has nothing to bite on).
SCOPE_AT_MECHANISM = (
    jensen_premise_active__acrobot,
    jensen_premise_active__cartpole,
    jensen_premise_active__discounting_chain,
    jensen_premise_dormant__catch,
)


# === Link broken: mechanism → outcome ===
#
# The headline DDQN finding. Verdicts here are NO_EFFECT-as-
# corroboration: the bridge's predicted null is confirmed.
# - converged_subset: pooled g(eval_best_burst_mean) ≈ -0.03
#   despite mechanism g ≈ -0.93 on the same envs.
# - time_to_solve: even sample-efficiency proxy is null.
# - n-step refutation: 3-step doesn't help outcome on average
#   despite DOES reduce bias additionally (mechanism HELD above).
# - Catch per-burst: every burst's g < 0.1 — saturated, no signal
#   to translate.
LINK_BROKEN = (
    ddqn_link_to_outcome_null__converged_subset,
    ddqn_outcome_zero_across_bursts__catch,
    nstep_3step_does_not_help_outcome__pool,
    time_to_solve_link_null__pooled,
)


# === Per-env outcome: arm → outcome (where the link does fire) ===
#
# The link is broken on average, but per-env exceptions exist.
# These are the envs where DDQN's bias-reduction *does* convert
# into an outcome benefit — narrow scope, env-specific reasons.
PER_ENV_OUTCOME = (
    ddqn_outcome_stable_across_bursts__fourrooms,
    ddqn_outperforms_expectile_on_outcome__fourrooms,
    ddqn_solves_faster__spaceinvaders,
    nstep_3step_helps_outcome__discounting_chain,
    nstep_3step_hurts_outcome__catch,
)


# === Factorial decomposition: (greedification × n_step) → outcome ===
#
# Rev 12's 2×2 factorial discriminates over-correction vs
# attenuation vs variance-amplification readings. FourRooms
# decomposes as DDQN-attenuation (interaction g=-0.71); Catch
# as variance-amplification (n-step alone hurts both arms;
# DDQN orthogonal). Plus the cross-mechanism reproduction:
# expectile shows the same mechanism-HELD ↛ link-HELD pattern.
FACTORIAL_DECOMPOSITION = (
    expectile_reproduces_mechanism_link_disconnect__fourrooms,
    factorial_ddqn_attenuation__fourrooms,
    factorial_variance_amplification__catch,
)


# === Chain moderators: env-feature → g_link / g_mech ===
#
# Rev 10's chain decomposition. log_obs_dim moderates g_link
# (smaller-obs envs convert bias-reduction to outcome benefit
# more reliably). The g_mech moderator (log_action_dim) doesn't
# reproduce on the per-burst panel of expectile_3way and is
# excluded from this graph; the rev 10 200k-corpus result lives
# in narrative form, not as a corroborated bridge.
CHAIN_MODERATORS = (
    log_obs_dim_drives_g_link,
)


# === Mediator audit: corpus-level filtering ===
#
# Three-check audit on the CartPole HP 180-cell corpus. Most
# "solve predictors" are mechanical (outcome-tautological reads-
# overlap) or HP-shadow (no within-stratum signal). One survives
# all three checks: state_coverage_kl_uniform_late.
MEDIATOR_AUDIT = (
    greedy_match_late_hp_shadow__cartpole_hp,
    learning_curve_auc_outcome_tautological__cartpole_hp,
    state_coverage_kl_clean_mediator__cartpole_hp,
)


# === Causal identification: mediator → outcome | HPs ===
#
# DoWhy backdoor on CartPole HP v2. state_coverage_kl_uniform_late
# is the first mediator that survives every framework check:
# audit-clean + backdoor ATE > 0 + placebo refutation + random-
# common-cause refutation. Rung-2-conditional-on-DAG; reverse
# direction (outcome → SCV) remains observationally
# indistinguishable.
CAUSAL_IDENTIFICATION = (
    state_coverage_kl_causes_outcome,
)


# === The measurement graph ===
#
# Concatenation of every corroborated edge. Run all of these
# against the same set of corpora to refresh the verdicts.
DDQN_MEASUREMENT_GRAPH = (
    *MECHANISM_EDGES,
    *SCOPE_AT_MECHANISM,
    *LINK_BROKEN,
    *PER_ENV_OUTCOME,
    *FACTORIAL_DECOMPOSITION,
    *CHAIN_MODERATORS,
    *MEDIATOR_AUDIT,
    *CAUSAL_IDENTIFICATION,
)


__all__ = [
    'CAUSAL_IDENTIFICATION',
    'CHAIN_MODERATORS',
    'DDQN_MEASUREMENT_GRAPH',
    'FACTORIAL_DECOMPOSITION',
    'LINK_BROKEN',
    'MECHANISM_EDGES',
    'MEDIATOR_AUDIT',
    'PER_ENV_OUTCOME',
    'SCOPE_AT_MECHANISM',
]
