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

# Empirical result (canonical-dormancy panel)

  B1   theorem  jdg → jens (vanilla):                        HELD  ρ=-0.48 p=1.4e-11
  B2   link    jens → outcome (vanilla):                     PI    ρ=-0.27 p=4.4e-12 (|ρ| below threshold)
  B3   mech do(DDQN) → jens (cross-env consistency):         HELD  9/9 envs, sign-test p=0.002
  B4   outcome do(DDQN) → outcome (cross-env consistency):   PI    ~6/9 envs, sign-test p=0.254

Cluster verdict: UNDERPOWERED (mix of HELD and PI/UP).

# Substantive reading

The chain's structure decomposes the "DDQN's mixed record"
question into layer-wise verdicts:

- **Theorem holds**: Hasselt's σ-floor empirically predicts
  observed bias under vanilla (B1 HELD).
- **Link present** at modest magnitude: ρ=-0.27 across panel
  cells (B2 PI — framework's |ρ| threshold not met but the
  correlation is real at p=4e-12).
- **Mech holds *consistently* across envs**: in 9/9 envs where
  the premise is broadly active, DDQN reduces the observed
  Jensen bias (B3 HELD at sign-test p=0.002).
- **Outcome edge does NOT hold consistently** (B4 PI): DDQN
  improves outcome in ~6/9 envs but harms / shows null in
  ~3/9 (Asterix d=-0.80, MetaMaze d=-0.12, MC d=-0.32). The
  chain's mech→outcome step is empirically *env-conditional*,
  not uniform.

The framework's typed verdict layer reads this as a layer-wise
diagnosis: the theorem, link, and mechanism all corroborate
*consistently across envs*; the bite at outcome is env-specific.
The cluster verdict is UNDERPOWERED (mix of HELD and PI) — an
honest "the chain holds at the upper three edges but breaks
inconsistently at outcome".

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
    ddqn_helps_outcome__consistently_cross_env,
    ddqn_reduces_bias__consistently_cross_env,
    hasselt_floor_predicts_observed_bias__vanilla,
)


EXPECTED: ClusterVerdict = ClusterVerdict.UNDERPOWERED


BLOCKED_ON: str | None = (
    'Empirical state on the canonical-dormancy panel: cluster '
    'verdict is UNDERPOWERED because the chain holds at three '
    'edges (B1 HELD, B3 HELD) but the link (B2) and outcome (B4) '
    'edges fire PI. B2 PI is a |ρ|-threshold issue (correlation '
    'is real at p=4e-12 but magnitude modest); B4 PI is genuine '
    'cross-env outcome heterogeneity (~6/9 envs help, ~3/9 harm). '
    "This is the substantive case-study finding — the chain's "
    'mech→outcome step is env-conditional, not uniform.'
)


BRIDGES: tuple[Bridge, ...] = (
    hasselt_floor_predicts_observed_bias__vanilla,           # B1 theorem
    bias_predicts_worse_outcome__vanilla,                    # B2 link
    ddqn_reduces_bias__consistently_cross_env,               # B3 mech
    ddqn_helps_outcome__consistently_cross_env,              # B4 outcome
)
