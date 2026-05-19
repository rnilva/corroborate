"""Synthetic Type-A/B controlled-substrate pre-registered Finding (v2).

v1 → v2 transition: the previous v1 design was scrapped after a
hostile review (`/tmp/synthetic_env_roast.md`) revealed five
load-bearing flaws: (1) action-independent transitions (a bandit
in a tuxedo), (2) `reward_variance_scale` conflating Var_a[Q*],
|Q|, and Δ_v in lockstep, (3) no FA-capacity axis (tabular env
with overparameterised MLP), (4) γ pinned at 0.99 (the
load-bearing axis), (5) σ/Δ in the 30-50% range vs Asterix-like
1%. v2 addresses #1, #2, #3, #4 and partially #5.

The v2 panel has TWO structural axes:
- L = n_states ∈ {16, 64} (FA-capacity axis)
- α = anisotropy_alpha ∈ {-0.5, 0, +0.5} (Type-A/B axis)

Plus γ ∈ {0.95, 0.99, 0.999} (substrate axis). 6 envs × 3 γ ×
2 arms × 12 seeds = 432 cells, ≤ 2h CPU budget.

The v2 bridges:
- P1: ρ(α, d_out) ≤ −0.4 pooled across γ — α drives DDQN's sign.
- P2: ρ(α, d_out) ≤ −0.5 at γ=0.999 — γ amplifies the split.
- P3: ρ(α × log(L), d_out) ≤ −0.4 — L modulates Type-B signal.
- N1: |ρ(L, d_out)| ≤ 0.3 at α=0 — capacity alone is null.

EXPECTED = SUPPORTED at the predicted bridges' joint admission.
BLOCKED_ON until the sweep ingests. Pre-registered: v2 bridges
+ this Finding committed BEFORE any cells run (cf.
`docs/PRE_REGISTRATION_synthetic_bias_typeb.md`).

Walk-back paths:
- P1 SIGN_FLIP → "anisotropy_alpha is not the causal driver of
  DDQN's outcome direction; either the natural-env Asterix
  pattern's policy-informative anisotropy isn't captured by
  per-action reward-noise asymmetry, or the construction
  parameters are wrong."
- P1 NULL_EFFECT → "α has no effect on DDQN d_out; the synthetic
  substrate fails to reproduce the natural-env pattern (rules
  out 'reward-noise anisotropy' as the env-intrinsic
  predictor). The natural-env mechanism is bound to specific
  dynamics not captured here."
- P2 NULL while P1 HELDs → "Type-A/B exists but γ doesn't
  amplify; the chain-amplification story is wrong in synthetic
  substrate. The 1/(1-γ) bias-magnification claim from
  Hasselt-style theorem doesn't transfer to this construction."
- P3 NULL while P1 HELDs → "L isn't a modulator; the
  (Λ_m, Λ_a, L) tuple's L axis is inactive in synthetic
  substrate. Either the MLP isn't actually capacity-bound at
  L=64, or the FA-capacity story is a natural-env-only
  phenomenon."
- N1 SIGN_FLIP (capacity DOES drive d_out at α=0) → "FA-capacity
  has an independent channel beyond the anisotropy axis;
  walks back the P3 interpretation (P3's signal may be L-only,
  not α × L)."

Critic recommendations addressed:
1. Action-dependent transitions: YES (s' = (s + a + 1) mod L).
2. Decouple Var_a[Q*] from Δ_v: YES (μ_best pinned;
   anisotropy_alpha varies SD asymmetry only).
3. L axis: YES (n_states ∈ {16, 64}; 256 deferred for cell
   budget — see PRE_REGISTRATION doc §Honest gaps).
4. γ sweep: YES (γ ∈ {0.95, 0.99, 0.999}).
5. Knife-edge σ/Δ regime: PARTIALLY (μ_best=0.05, σ_base=0.5
   gives σ/Δ ≈ 10; closer to natural-env than v1's 30-50%
   but still 10× larger than Asterix's 1%. A pilot at smaller
   σ_base requires re-tuning learning rate; deferred to a
   follow-up sweep if v2 reproduces neither regime cleanly).

Companion docs:
- `docs/PRE_REGISTRATION_synthetic_bias_typeb.md` — predictions
  + v1 walk-back rationale.
- `/tmp/synthetic_env_roast.md` — the hostile review of v1.
- `findings_theorem3_lg_scm_controlled_falsification` (memory)
  — the prior controlled-substrate test that pioneered this
  paradigm.
"""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn_sweeps.synthetic_bias_typeb import (
    ddqn_harm_amplified_at_g999__synthetic_typeb_v2,
    ddqn_harm_scales_with_type_b_score__synthetic_typeb_v2,
    ddqn_harms_under_positive_alpha__synthetic_typeb_v2,
    n_states_alone_does_not_predict_dout__synthetic_typeb_v2,
)


# Pre-registration: EXPECTED reflects the SUBSTANTIVE prediction
# that P1, P2, P3, N1 will land; the Finding fires UNDERPOWERED
# [blocked] until the sweep ingests, and the framework's DRIFT
# detection fires when verdicts deviate from this pin.
# EMPTY_EXTENT until the sweep ingests; all four bridges fire
# POWER_INSUFFICIENT (n_strata=0) without data.
EXPECTED: ClusterVerdict = ClusterVerdict.EMPTY_EXTENT


BLOCKED_ON: str | None = (
    'Pre-registered 2026-05-19 BEFORE v2 sweep ran (v1 scrapped — '
    'see docs/PRE_REGISTRATION_synthetic_bias_typeb.md). Awaiting '
    'ingest of `experiments/data/synthetic_bias_typeb_v2_pilot` '
    '(6 envs × 3 γ × 2 arms × 12 seeds = 432 cells). Predicted '
    'post-ingest EXPECTED=SUPPORTED: P1 (α → d_out ≤ −0.4 pooled), '
    'P2 (α → d_out ≤ −0.5 at γ=0.999), P3 (α × log(L) → d_out ≤ '
    '−0.4), N1 (L alone at α=0 doesn\'t predict d_out, |ρ| ≤ 0.3). '
    'Drift on any bridge fires DRIFT at hypothesis run-time. Walk-'
    'back paths documented in module docstring + PRE_REGISTRATION.'
)


BRIDGES: tuple[Bridge, ...] = (
    ddqn_harms_under_positive_alpha__synthetic_typeb_v2,
    ddqn_harm_amplified_at_g999__synthetic_typeb_v2,
    ddqn_harm_scales_with_type_b_score__synthetic_typeb_v2,
    n_states_alone_does_not_predict_dout__synthetic_typeb_v2,
)
