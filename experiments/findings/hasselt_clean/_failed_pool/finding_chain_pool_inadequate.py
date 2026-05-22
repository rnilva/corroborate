"""Pedagogical Finding: random-effects pool on the chain's
intervention edges is the wrong claim shape — preserves the
NO_EFFECT verdicts the framework's PI-based discipline returns
when the strata aren't exchangeable.

This Finding fires REFUTED on the canonical-dormancy panel
because B3-pool and B4-pool both return NO_EFFECT under
`random_effects_verdict`'s prediction-interval test.

Substantive content: the random-effects model assumes envs are
exchangeable draws from a population with model `g_i ~ N(μ, τ²)`.
RL envs aren't — they differ in network class, Q-magnitude,
reward sparsity, etc. The pool's PI test correctly refuses to
extrapolate "the average effect" to an 11th env. The framework's
typed verdict layer makes this visible.

The chain's main Finding (`finding_hasselt_chain_explicit`)
uses `cross_env_consistency_binomial` (sign-test) instead —
the right tool when the question is "does the effect hold
consistently across heterogeneous envs?" rather than "what's
the extrapolable population effect?"."""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.hasselt_clean._failed_pool.chain_pool import (
    intervention_helps_outcome__pool_inadequate,
    intervention_reduces_bias__pool_inadequate,
)


EXPECTED: ClusterVerdict = ClusterVerdict.REFUTED


BLOCKED_ON: str | None = None


BRIDGES: tuple[Bridge, ...] = (
    intervention_reduces_bias__pool_inadequate,
    intervention_helps_outcome__pool_inadequate,
)
