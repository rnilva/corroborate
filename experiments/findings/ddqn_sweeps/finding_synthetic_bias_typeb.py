"""Synthetic Type-A/B controlled-substrate pre-registered Finding.

Wraps the two bridges that test the pre-registered predictions
P1 (high `reward_variance_scale` → DDQN harms; ρ ≤ −0.5 cross-env)
and N1 (sparsity alone doesn't predict d_out; |ρ| ≤ 0.2). The
synthetic env panel turns the natural-env n=1 (Asterix only)
Type-B finding into a parametric sweep with cleanly-controlled
Var_a[Q*].

EXPECTED = SUPPORTED at the predicted bridges' joint admission:
- P1 HELD when ρ(rvs, d_out) ≤ −0.5 with p ≤ 0.05
- N1 HELD when |ρ(sparsity, d_out)| ≤ 0.2

BLOCKED_ON until the sweep ingests. Pre-registered: bridges +
this Finding committed BEFORE any cells run (cf.
`docs/PRE_REGISTRATION_synthetic_bias_typeb.md`).

Walk-back paths:
- P1 SIGN_FLIP → "rvs-as-Var_a[Q*] is not the causal driver of
  DDQN's outcome direction; either the natural-env Asterix
  pattern is qualitatively distinct from the synthetic
  reproduction, or the linear-mean-spread design doesn't
  capture what makes a Var_a[Q*] policy-informative."
- P1 NULL_EFFECT → "rvs has no effect at all on DDQN d_out;
  the synthetic substrate fails to reproduce the natural-env
  pattern (rules out the env-intrinsic predictor approach
  entirely; would need different env features)."
- N1 NO_EFFECT (sparsity DOES drive d_out) → "sparsity has an
  independent channel beyond Var_a[Q*]; needs separate
  characterisation; not a clean walk-back of P1."

Companion docs:
- `docs/PRE_REGISTRATION_synthetic_bias_typeb.md` — the
  predictions this Finding enforces.
- `findings_theorem3_lg_scm_controlled_falsification` (memory) —
  the prior controlled-substrate Cor 3.2 falsification (K=3
  one-step bootstrap MC). This Finding extends the
  controlled-substrate paradigm to DDQN training dynamics.
- `findings_lambda_a_within_arm_asymmetry` (Finding) — the
  natural-env within-cell evidence this synthetic sweep
  disambiguates from the per-env Asterix idiosyncrasy."""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn_sweeps.synthetic_bias_typeb import (
    ddqn_harms_under_high_rvs__synthetic_typeb,
    sparsity_alone_does_not_predict_dout__synthetic_typeb,
)


# Pre-registration: EXPECTED reflects the SUBSTANTIVE prediction
# that P1 (and N1) will land; the Finding fires UNDERPOWERED
# [blocked] until the sweep ingests, and the framework's DRIFT
# detection fires when verdicts deviate from this pin.
# EMPTY_EXTENT until the sweep ingests; both bridges fire
# POWER_INSUFFICIENT (n_strata=0) without data. Pre-registered
# DRIFT prediction: post-ingest the Finding should land at
# SUPPORTED if P1 (ρ(rvs, d_out) ≤ −0.5 p ≤ 0.05) AND N1
# (|ρ(sparsity, d_out)| ≤ 0.2) both admit. The framework's
# automatic DRIFT detection surfaces deviations from this pin.
EXPECTED: ClusterVerdict = ClusterVerdict.EMPTY_EXTENT


BLOCKED_ON: str | None = (
    'Pre-registered 2026-05-19 BEFORE sweep ran. Awaiting ingest '
    'of `experiments/data/synthetic_bias_typeb_pilot` (12 envs × '
    '2 arms × 15 seeds, ~360 cells). Predicted post-ingest '
    'EXPECTED=SUPPORTED: P1 (ρ(rvs, d_out) ≤ −0.5 p ≤ 0.05) AND '
    'N1 (|ρ(sparsity, d_out)| ≤ 0.2) both admit. Drift on either '
    'bridge fires DRIFT at hypothesis run-time; either walk-back '
    'path is publishable per docs/PRE_REGISTRATION_synthetic_bias_typeb.md.'
)


BRIDGES: tuple[Bridge, ...] = (
    ddqn_harms_under_high_rvs__synthetic_typeb,
    sparsity_alone_does_not_predict_dout__synthetic_typeb,
)
