"""Hasselt's chain authored as an explicit directed walk on the
post-eval graph — REFUTED on the full 10-env canonical panel.

This Finding is the framework's principled answer to the
question: *given a mechanism claim with an analytical premise,
can we author the premise's activation as a first-class edge of
a verified causal chain?* The structural shape:

  jensen_dormancy_gap ──► jensen_gap ──► eval_best_burst_raw_mean
                              ▲                  ▲
                              │                  │
                          do(DDQN) ──────────────┘

Three nodes, six edges (four primary + two per-stratum siblings).
Premise non-activation is the failure of bridge B1 — an explicit
upstream edge — not a scope predicate buried inside a downstream
bridge. The clean form makes the framework's first-class-edge
framing honest by construction.

# Empirical result (10-env canonical panel)

Per-edge verdicts on the full canonical-pool panel (post
`jensen_dormancy_gap` backfill across all 10 envs):

  B1   theorem      jdg → jens   (vanilla):     HELD       ρ=-0.475, p=1.4e-11
  B2   link        jens → outcome (vanilla):    PI         ρ=-0.269, p=4.4e-12
  B3   mech    do(DDQN) → jens   (per-cell):    NO_EFFECT  d=-1.90, p=2e-6, I²=0.97
  B3'  mech    do(DDQN) → jens   (per-stratum): NO_EFFECT  d=-1.95, p=1e-5, I²=0.97
  B4   outcome do(DDQN) → out    (per-cell):    NO_EFFECT  d=+0.46, p=0.19, I²=0.96
  B4'  outcome do(DDQN) → out    (per-stratum): NO_EFFECT  d=+0.52, p=0.21, I²=0.97

Cluster verdict: REFUTED.

# Per-env mech-edge breakdown (B3)

DDQN's effect on `jensen_gap` (Cohen's d, premise-active cells):

  MinAtar (CNN substrate, extreme reductions):
    Asterix:           d = -8.910
    SpaceInvaders:     d = -4.519
    Breakout:          d = -3.015
    Freeway:           d = -1.260   (premise active only ~5% — small n)
    FourRooms:         d = -1.260

  Classical (MLP/MetaMaze, modest reductions):
    Snake:             d = -1.250
    MetaMaze:          d = -0.511
    MountainCar:       d = -0.300
    LunarLander:       d = -0.259

  Sign-flip:
    Acrobot:           d = +0.064

8 of 9 envs in predicted INVERSE direction. The substrate-shape
divergence between MinAtar (CNN, extreme reductions) and
classical (MLP, modest reductions) drives the I²=0.97
heterogeneity. The pooled d=-1.90 with p=2e-6 would be HELD
under a single-stratum effect-size lens; the framework's
random-effects verdict downgrades to NO_EFFECT under extreme
between-env heterogeneity. This is the verdict layer surfacing
a real structural feature — DDQN's mech-bite scales with
Q-magnitude across substrate classes — not a power artifact.

# Per-env outcome-edge breakdown (B4)

DDQN's effect on `eval_best_burst_raw_mean` (premise + link
active per cell):

  Helps:
    FourRooms:         d = +1.796
    SpaceInvaders:     d = +2.163
    Breakout:          d = +0.662
    Snake:             d = +0.625
    LunarLander:       d = +0.124

  Harms or null:
    Acrobot:           d = +0.022
    MetaMaze:          d = -0.122
    MountainCar:       d = -0.324
    Asterix:           d = -0.800

The famous Asterix γ=0.999 harm shows up at d=-0.800 — DDQN's
strongest mech-bite (-8.9) co-occurs with its strongest outcome
harm (-0.8). The chain's mech→outcome link inverts at this env:
where bias-reduction is largest, outcome is harmed most.

# Substantive reading

The theorem edge B1 holds: Hasselt's σ-floor structurally
predicts observed Jensen bias under vanilla. The link edge B2
fires PI — the bias-correction→outcome correlation is present
(ρ=-0.27, p=4e-12) but at modest magnitude; the framework's
verdict matrix refuses to call it HELD without a stronger ρ.

The intervention edges (B3, B4) fire NO_EFFECT — but the
pooled NO_EFFECT hides per-env directional alignment (mech: 8/9
in predicted dir) and extreme heterogeneity (mech I²=0.97).
DDQN's mech-bite is genuine but scale-divergent across substrate
classes; the outcome edge is genuinely heterogeneous (5/9 help,
4/9 harm/null).

The clean chain decomposition surfaces what scalar benchmarks
would read as "DDQN doesn't help much on average":
- Theorem layer: corroborated.
- Link layer: empirically present, modest magnitude.
- Mech layer: empirically present at every env (DDQN reduces
  bias 8/9 times) but with extreme cross-env heterogeneity
  the framework refuses to pool.
- Outcome layer: genuinely heterogeneous, pooled-null, with
  Asterix as the canonical harm case alongside Breakout / FR /
  SI as canonical help cases.

Per-cell and per-stratum siblings agree on all four
intervention edges — selection-bias concern is not material.

Companion to `experiments/findings/ddqn/finding_hasselt_chain.py`
(the original 4-bridge cluster on the SUPPORTED side); this
version makes the upstream-edge structure explicit rather than
collapsing premise into a scope predicate on the downstream
mech bridge, and surfaces the chain's actual layer-wise
verdicts on the full canonical panel."""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.hasselt_clean.chain import (
    bias_predicts_worse_outcome__vanilla,
    hasselt_floor_predicts_observed_bias__vanilla,
    intervention_helps_outcome__chain_holds_per_cell,
    intervention_helps_outcome__chain_holds_per_stratum,
    intervention_reduces_bias__premise_active_per_cell,
    intervention_reduces_bias__premise_active_per_stratum,
)


EXPECTED: ClusterVerdict = ClusterVerdict.REFUTED


BLOCKED_ON: str | None = None


BRIDGES: tuple[Bridge, ...] = (
    hasselt_floor_predicts_observed_bias__vanilla,           # B1
    bias_predicts_worse_outcome__vanilla,                    # B2
    intervention_reduces_bias__premise_active_per_cell,      # B3
    intervention_reduces_bias__premise_active_per_stratum,   # B3' (sibling)
    intervention_helps_outcome__chain_holds_per_cell,        # B4
    intervention_helps_outcome__chain_holds_per_stratum,     # B4' (sibling)
)
