"""Hasselt's chain authored as an explicit directed walk on the
post-eval graph — REFUTED on the canonical-dormancy panel.

This Finding is the framework's principled answer to the
question: *given a mechanism claim with an analytical premise,
can we author the premise's activation as a first-class edge of
a verified causal chain?* The structural shape:

  jensen_dormancy_gap ──► jensen_gap ──► eval_best_burst_raw_mean
                              ▲                  ▲
                              │                  │
                          do(DDQN) ──────────────┘

Three nodes, four edges. Premise non-activation is the failure
of bridge B1 — an explicit upstream edge — not a scope predicate
buried inside a downstream bridge.

# Why per-stratum (not per-cell) scope on the intervention edges

The intervention edges B3 and B4 condition on premise activation
at the *env level* (`median jensen_dormancy_gap == 0` per env)
rather than the *cell level* (`jdg == 0` per cell). This is the
principled choice: DDQN's intervention itself affects which
cells satisfy `gap == 0` (DDQN reduces observed bias → cells
fall below σ-floor → become dormant). Per-cell premise scope
is post-treatment conditioning — a chain-internal collider —
introducing M-bias on the surviving cohort.

Acrobot γ=0.999 surfaces this directly: under per-cell scope,
DDQN's mean `jensen_gap` reads HIGHER than vanilla's (15.9 vs
12.6) — but this is the selection effect of comparing
"DDQN-active" (DDQN failed to push below floor) against
"vanilla-active" (typical high-bias cells). Per-stratum scope
sees DDQN's net effect ≈ 0 across all 60 Acrobot cells — the
honest answer: at Acrobot γ=0.999 (solved by both arms,
V_eb≈-76 = solved ceiling), Hasselt's mech has no failure mode
left to clip.

# Empirical result (canonical-dormancy panel)

  B1   theorem      jdg → jens   (vanilla):           HELD       ρ=-0.48 p=1.4e-11
  B2   link        jens → outcome (vanilla):          PI         ρ=-0.27 p=4.4e-12
  B3   mech    do(DDQN) → jens   (per-stratum):       NO_EFFECT  d=-1.90 p=1e-6 I²=0.97
  B4   outcome do(DDQN) → out    (per-stratum):       NO_EFFECT  d=+0.47 p=0.17 I²=0.97

Cluster verdict: REFUTED.

# Per-env mech-edge breakdown (B3)

DDQN's effect on `jensen_gap` (Cohen's d, per-stratum scope):

  MinAtar (CNN substrate):
    Asterix:           d = -8.910
    SpaceInvaders:     d = -4.519
    Breakout:          d = -3.015
    Freeway:           dropped — corpus median jdg > 0
                       (95% cells dormant; per-stratum filter excludes)

  Classical (MLP substrate):
    FourRooms:         d = -1.260
    Snake:             d = -1.250
    MetaMaze:          d = -0.520
    MountainCar:       d = -0.300
    LunarLander:       d = -0.259
    Acrobot:           d = -0.010

All 9 panel envs in the predicted INVERSE direction. The pooled
verdict reads NO_EFFECT because the random-effects prediction
interval (PI) brackets zero under I²=0.97 — the framework's
discipline says "with this much cross-env scatter, we can't
predict the direction of effect on a new env" even though the
CI for the pooled mean is clearly negative. This is structural:
DDQN's mech-bite scales with Q-magnitude across substrate
classes (CNN d≈-3 to -9; MLP d≈-0.3 to -1.3); pooling them
under random-effects refuses to declare a uniform population
effect.

# Per-env outcome-edge breakdown (B4)

DDQN's effect on `eval_best_burst_raw_mean` (per-stratum scope,
premise + link both active):

  Helps:
    FourRooms:         d = +1.796
    SpaceInvaders:     d = +2.163
    Breakout:          d = +0.662
    Snake:             d = +0.625
    LunarLander:       d = +0.124

  Harms or null:
    Acrobot:           d = +0.064
    MetaMaze:          d = -0.082
    MountainCar:       d = -0.324
    Asterix:           d = -0.800

5 of 9 envs help; 4 harm/null. Asterix's strongest mech-bite
(d=-8.9) co-occurs with its strongest outcome harm (d=-0.8) —
the chain's mech→outcome link inverts at this env.

# Substantive reading

The theorem edge B1 holds: Hasselt's σ-floor structurally
predicts observed Jensen bias under vanilla. The link edge B2
fires PI — the bias-correction→outcome correlation is present
(ρ=-0.27, p=4e-12) but at modest magnitude; the framework's
verdict matrix refuses to call it HELD without a stronger ρ.

The intervention edges (B3, B4) fire NO_EFFECT but the
underlying evidence is layered: B3's pooled d=-1.95 reflects
9/9 envs in the predicted direction, with the verdict
downgraded to NO_EFFECT under random-effects prediction-
interval discipline because cross-env scale heterogeneity is
extreme (I²=0.97). B4's pooled d=+0.52 is genuinely
heterogeneous (5/8 help, 3/8 harm/null).

The clean chain decomposition surfaces what scalar benchmarks
would read as "DDQN doesn't help much on average":
- Theorem layer: corroborated.
- Link layer: empirically present, modest magnitude.
- Mech layer: empirically present at every env (DDQN reduces
  bias 9/9 times) but with extreme cross-env heterogeneity the
  framework refuses to pool.
- Outcome layer: genuinely heterogeneous, pooled-null, with
  Asterix as the canonical harm case alongside Breakout / FR /
  SI as canonical help cases.

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
    intervention_helps_outcome__chain_holds,
    intervention_reduces_bias__premise_active,
)


EXPECTED: ClusterVerdict = ClusterVerdict.REFUTED


BLOCKED_ON: str | None = None


BRIDGES: tuple[Bridge, ...] = (
    hasselt_floor_predicts_observed_bias__vanilla,    # B1
    bias_predicts_worse_outcome__vanilla,             # B2
    intervention_reduces_bias__premise_active,        # B3
    intervention_helps_outcome__chain_holds,          # B4
)
