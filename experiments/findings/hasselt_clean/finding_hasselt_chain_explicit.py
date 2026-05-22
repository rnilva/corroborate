"""Hasselt's chain authored as an explicit directed walk on the
post-eval graph — first-run REFUTED on the 5-env subpanel.

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

# Empirical result (5-env subpanel, γ=0.999)

`jensen_dormancy_gap` is currently finite in the cache for only
5 of 10 canonical-pool envs (Acrobot / LL / MetaMaze / MC /
Snake). The 5 MinAtar+FR envs have the column null — engineering
debt from a 2026-05-22 ingest regression (see FUTURE_WORKS.md
§clean-chain-extension). The Finding's verdict is on the 5-env
subpanel.

Per-edge verdicts on the first run (2026-05-22):

  B1  theorem      jdg → jens   (vanilla):   HELD       ρ=-0.47, p=1.3e-9
  B2  link        jens → out    (vanilla):   HELD       ρ=-0.64, p≈0
  B3  mech    do(DDQN) → jens   (per-cell):  NO_EFFECT  d=-0.40, p=0.04, I²=0.78
  B3' mech    do(DDQN) → jens   (per-stratum): NO_EFFECT  d=-0.40, p=0.06, I²=0.81
  B4  outcome do(DDQN) → out    (per-cell):  NO_EFFECT  d=+0.04, p=0.77, I²=0.57
  B4' outcome do(DDQN) → out    (per-stratum): NO_EFFECT  d=+0.08, p=0.59, I²=0.67

Cluster verdict: REFUTED (any member REFUTED → cluster
REFUTED; here B3/B3'/B4/B4' all fire NO_EFFECT which stamps
'refuted' per `_stamp_level`).

# Substantive reading

The theorem edge B1 holds: Hasselt's σ-floor structurally
predicts observed Jensen bias under vanilla — ρ=-0.47 across
300 vanilla cells, near-zero p. The link edge B2 holds: higher
observed bias predicts lower outcome under vanilla. The
structural prerequisite for bias-reduction-as-mechanism is
empirically present.

But the intervention edges (B3, B4) do not deliver. DDQN's mech
bite on `jensen_gap` is heterogeneous (I²=0.78) and pooled at
d=-0.40 — meaningful but with high cross-env scatter. DDQN's
outcome bite is essentially null on the panel (pooled d=+0.04).
The per-cell and per-stratum siblings agree — the per-cell
selection-bias concern is not material at this scope.

The framework's typed verdict layer decomposes what scalar
benchmarks would read as "DDQN doesn't help much" into a
layer-wise diagnosis: *the theorem and link survive, the
intervention does not bite at canonical scope on the
dormancy-measured 5-env subpanel*. The clean-chain authoring is
what makes the layer-wise structure visible — the original
`finding_hasselt_chain.py` cluster collapsed dormancy into a
scope predicate and reported SUPPORTED at canonical, hiding
the heterogeneous intervention layer.

# Honest scope

REFUTED at 5-env subpanel. Extension to the full 10-env
canonical pool requires backfilling `jensen_dormancy_gap` for
Asterix / Breakout / Freeway / SI / FR (per
FUTURE_WORKS.md §substrate-jdg-backfill). The 5 missing envs
include high-DDQN-benefit MinAtar games; the cross-env pool
may shift toward HELD with their inclusion.

Companion to `experiments/findings/ddqn/finding_hasselt_chain.py`
(the original 4-bridge cluster on the SUPPORTED side); this
version makes the upstream-edge structure explicit rather than
collapsing premise into a scope predicate on the downstream
mech bridge."""
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
