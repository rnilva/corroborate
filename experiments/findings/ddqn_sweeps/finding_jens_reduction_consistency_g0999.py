"""DDQN reduces vanilla's jensen_gap consistently across the
canonical γ=0.999 envs (sign-test, not magnitude).

This Finding tests the canonical Hasselt-2010 bias-clip mechanism
as a CONSISTENCY claim across the 9-10 envs in
`_CANONICAL_G0999_CORPORA`. At n_strata ≈ 10, the population-
magnitude shape (Spearman ρ between env-feature and Δ_jens, or
DL-pooled effect size) is structurally underpowered — see the
walked-back σ_Λ_a / σ/jens findings — but the directional
consistency shape ("DDQN reduces jens at every env") survives
via the binomial sign-test.

Single bridge, single extent. The methodology lesson: claim
shape determines power. With 10 envs:
  - Spearman ρ needs |ρ| ≥ 0.71 at p=0.05.
  - Binomial sign-test needs 9/10 same direction for p ≤ 0.011.
  - Many real cross-env phenomena align directionally but
    don't have a population-scale-feature relationship strong
    enough to survive ρ ≥ 0.71. Consistency saves these claims.

Empirical preview on canonical-corpus γ=0.999 pool:
  9/10 envs show DDQN reduces jens (d_jens < 0). Acrobot is
  the lone exception (small magnitude). Binomial p = 0.011 →
  SUPPORTED.
"""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn_sweeps.jens_reduction_consistency import (
    ddqn_reduces_jens_consistently__canonical_g0999,
)


EXPECTED: ClusterVerdict = ClusterVerdict.SUPPORTED


BLOCKED_ON: str | None = None


BRIDGES: tuple[Bridge, ...] = (
    ddqn_reduces_jens_consistently__canonical_g0999,
)
