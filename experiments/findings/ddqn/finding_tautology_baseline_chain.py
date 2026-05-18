"""Substrate tautology baseline: Q → discounted → raw chain.

The substrate corpus targets RAW outcome (`eval_best_burst_raw_mean`,
`mc_return_raw_per_burst_mean`) for practitioner-facing performance
reporting. But two structural couplings link Q-magnitude to raw
outcome through the discounted intermediate:

  Edge (1): Q  →  γ-DISCOUNTED MC return
    - `q_to_mc_coupled__bellman_contraction_baseline`
    - Bellman contraction: Q is the value function, which
      estimates E[Σ γ^t r_t]. Training objective IS this edge.
    - Expected ρ > 0 within env; HELD when ρ_pool ≥ +0.2.

  Edge (2): γ-DISCOUNTED MC  →  RAW MC outcome
    - `mc_disc_raw_coupled__per_env_jci`
    - RL-practitioner convention: report raw return as performance
      even though Q targets discounted return. The two differ by
      γ^T weighting across episode-length variation.
    - Expected ρ > 0 within env; HELD when ρ_pool ≥ +0.5.

Together (1)→(2) is the substrate's Q → raw_outcome tautology
chain: downstream bridges that condition on `q_per_burst` to
partial out the Q-mediated effect (e.g.,
`q_action_std_per_burst_link_to_outcome__partial_q`) implicitly
ride this chain. Authoring the chain as an explicit Finding
makes the conditioning rationale load-bearing in the graph —
the chain HELDs at the Finding level iff both links HELD
individually, and downstream bridges can refer to this Finding
as the documented coupling they partial out.

Chain semantics via the walk primitives in `graph.causal`:

  composed_verdict(g, bridges=BRIDGES) → ClusterVerdict
    AND-aggregate; SUPPORTED iff both edges HELD; REFUTED if any
    refutes. The cluster-verdict rule IS the correct chain compose
    under monotone composition (transitivity of HELD = all-admit-
    non-empty). The verdict layer doesn't distinguish cluster from
    walk; the distinction is structural, captured by:

  walk_subgraph(g, nodes=('q_per_burst', 'mc_return__mean_axis_-1',
                          'eval_best_burst_raw_mean'))
    Induced subgraph along the directed walk. Renders the
    Q → discounted MC → raw MC topology explicitly.

  is_walk(g, bridges=BRIDGES) → True
    Validates the bridges form a connected directed walk
    (q_per_burst → mc_return__mean_axis_-1 via Bellman, then
    mc_return__mean_axis_-1 → eval_best_burst_raw_mean via the
    practitioner-coupling edge).

  walk_scope(BRIDGES) → pl.Expr
    Joint-scope predicate (both bridges' scope `&`-reduced).
    The cell-set on which the FULL chain is empirically
    corroborable — useful for downstream bridges that need to
    condition on the chain's validity (see PARTIAL bridge in
    `q_shape_mediation.py`).

  compose_direction([e for e in walk_subgraph(...).edges])
    = DIRECT × DIRECT = DIRECT. The chain's predicted direction.

The walk primitives don't change this Finding's verdict (still
`composed_verdict`), but make the chain's topology / direction /
joint-scope explicit and queryable for downstream consumers.
"""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn.bias_correction import (
    mc_disc_raw_coupled__per_env_jci,
)
from experiments.findings.ddqn.q_shape_mediation import (
    q_to_mc_coupled__bellman_contraction_baseline,
)


# REFUTED on the merged ddqn cache (post 2026-05-18 consolidation
# + origin/main cache growth): per-burst ρ(q, mc_discounted) pooled
# across envs landed at +0.156 (n=27600 obs, 10 strata) — below
# the +0.2 substantive threshold. The Bellman-contraction coupling
# is positive but weak at the merged cohort scale; the substrate
# tautology chain doesn't survive the stricter threshold on the
# enlarged corpus. The Finding documents the empirical REFUTATION;
# whether the threshold is the right cutoff is a separate
# methodology question.
EXPECTED: ClusterVerdict = ClusterVerdict.REFUTED


BLOCKED_ON: str | None = None


BRIDGES: tuple[Bridge, ...] = (
    q_to_mc_coupled__bellman_contraction_baseline,
    mc_disc_raw_coupled__per_env_jci,
)
