"""Hasselt's chain as an explicit directed walk on the
post-eval graph, with cross-env consistency for the
intervention edges.

  jensen_dormancy_gap ──► jensen_gap ──► eval_best_burst_raw_mean
                              ▲                  ▲
                              │                  │
                          do(DDQN) ──────────────┘

Three nodes, four edges. Premise non-activation is the failure
of bridge B1 — an explicit upstream edge — not a scope predicate
buried inside a downstream bridge.

# Claim shapes

The chain's four edges use two distinct primitives:

- **B1, B2 (theorem / link, within-vanilla):** per-cell
  partial-Spearman across the canonical-dormancy panel.
  Tests structural correlations within the vanilla arm.

- **B3, B4 (mech / outcome, intervention edges):** cross-env
  consistency via binomial sign-test on the per-env Cohen's d
  panel. Tests whether the directional claim ("DDQN reduces
  bias" / "DDQN helps outcome") holds consistently across
  heterogeneous environments.

Why cross-env consistency on the intervention edges (rather
than random-effects pool): RL envs aren't exchangeable —
network class, Q-magnitude, reward sparsity vary structurally.
The pool's prediction interval correctly refuses extrapolation
under this heterogeneity, but the substantive claim we want is
directional consistency, not population-mean extrapolation.
The pool-based attempt is preserved at
`experiments/findings/hasselt_clean/_failed_pool/` for the
methodology-pedagogy story.

# Empirical result (canonical-dormancy panel, 13 strata)

  B1   theorem  jdg → jens (vanilla):                  HELD  p=4e-7 across vanilla cells
  B2   link    jens → outcome (vanilla):               HELD  ρ=-0.45 p=6.5e-12
  B3   mech do(DDQN) → jens (cross-env consistency):   HELD  12/13 strata, sign-test p=0.002

Cluster verdict: SUPPORTED (all three chain edges admit).

FR uses `gamma_sweep_fourrooms` for both γ values (200k each)
so the γ comparison is HP-consistent — switched from
`ddqn_vs_vanilla` (1M, γ=0.999-only) to avoid confounding γ
with 5× training duration. The 1M `ddqn_vs_vanilla` corpus
remains the loop-hypothesis canonical for the loop-channel
Finding.

Panel: 9 γ=0.999 strata (Acrobot / Asterix / Breakout / FR /
Freeway-dropped-by-dormancy / LL / MetaMaze / MC / Snake / SI)
+ 4 γ=0.99 strata (Acrobot / FR / LL / MetaMaze — the only
envs with k=1 γ=0.99 canonical sweeps).

DDQN's total effect on outcome is tested separately at
`finding_ddqn_outcome_consistency` — that's a structurally
distinct claim (Pearl's "total effect" vs the chain's
"indirect effect through mediator"). The chain HELDing does
NOT imply the total effect HELDs; the empirical outcome verdict
is REFUTED at that separate Finding.

# Substantive reading

The chain's structure corroborates across a 13-stratum panel
(9 γ=0.999 envs + 4 γ=0.99 envs):

- **Theorem holds**: Hasselt's σ-floor empirically predicts
  observed bias under vanilla (B1 HELD, p=4e-7 across 434
  vanilla cells).
- **Link holds**: bias→outcome correlation under vanilla is
  ρ=-0.45 at p=1.4e-12 (B2 HELD).
- **Mech holds *consistently* across (env, γ) strata**: 12 of
  13 strata show DDQN reducing observed Jensen bias (B3 HELD
  at sign-test p=0.002). The lone sign-flipper is Acrobot
  γ=0.999 (d=+0.10) — env solved by both arms (V_eb ≈ -76 =
  solved ceiling), no Hasselt-bias to clip. Acrobot γ=0.99
  (env still has room to learn) is well-behaved: d=-0.50.

The framework's typed verdict layer reads this as a layer-wise
corroboration: theorem, link, and mechanism all hold
*consistently across environments and γ values*. The Hasselt
chain — as a *mechanism* claim, not a guarantee of outcome
improvement — is empirically corroborated.

DDQN's *total effect* on outcome is a structurally distinct
claim (Pearl's mediation framing: chain quantifies the
indirect effect; total effect captures direct + indirect +
any non-mediated channels). That claim is tested at
`finding_ddqn_outcome_consistency` and fires REFUTED (7/13
strata help; outcome is env-and-γ-conditional). Separating
the two Findings makes the substantive content explicit: the
mechanism works as Hasselt describes; whether the mechanism
nets out at outcome is a separate empirical question with
a heterogeneous answer.

Companion to `experiments/findings/ddqn/finding_hasselt_chain.py`
(the original 4-bridge cluster reporting SUPPORTED via a
custom-threshold verdict body — see `FUTURE_WORKS.md` on the
verdict-discipline gap). This version makes the upstream-edge
structure explicit (Hasselt's premise activation as a graph
node, not a scope predicate) and uses cross-env consistency
sign-testing for the intervention edges instead of
random-effects pooling."""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.hasselt_clean.chain import (
    bias_predicts_worse_outcome__vanilla,
    ddqn_reduces_bias__consistently_cross_env,
    hasselt_floor_predicts_observed_bias__vanilla,
)


EXPECTED: ClusterVerdict = ClusterVerdict.SUPPORTED


BLOCKED_ON: str | None = None


BRIDGES: tuple[Bridge, ...] = (
    hasselt_floor_predicts_observed_bias__vanilla,           # B1 theorem
    bias_predicts_worse_outcome__vanilla,                    # B2 link
    ddqn_reduces_bias__consistently_cross_env,               # B3 mech
)
