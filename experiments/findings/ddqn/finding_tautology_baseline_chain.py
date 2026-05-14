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

Chain semantics via `composed_verdict`: SUPPORTED iff both
edges HELD (transitivity of admissibility); REFUTED if any
link refuted (a broken link breaks the chain claim). The
Finding doesn't reify chain-specific reasoning (the framework
doesn't have a `path_verdict` primitive yet); the AND-aggregate
of `composed_verdict` IS the correct chain compose under
monotonicity assumptions (positive rank correlations compose
positively).
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


EXPECTED: ClusterVerdict = ClusterVerdict.SUPPORTED


BLOCKED_ON: str | None = None


BRIDGES: tuple[Bridge, ...] = (
    q_to_mc_coupled__bellman_contraction_baseline,
    mc_disc_raw_coupled__per_env_jci,
)
