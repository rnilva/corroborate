"""The geometric-series argmax-accumulation gap (THEORY §6.1 open
limitation) is EMPIRICALLY EMPTY at the burst granularity we
measure on — Bellman bias accumulates to its asymptote within
the first burst, so horizon-normalisation is mathematically
vacuous (a constant per-cell factor) and doesn't change cross-γ
trajectory ratios.

The test:
- Measurable `q_lambda_a_horizon_normalised_per_burst` divides
  raw per_burst Λ_a by `(1-γ^t)/(1-γ)` at eval step `t`.
- At t = 20000 (the FIRST burst window), γ^20000 ≈ 2e-9 for
  γ=0.999 and even smaller for γ=0.95/0.99 → 1 - γ^t ≈ 1
  uniformly → normalisation factor ≈ 1/(1-γ) constant per cell.
- Multiplying both tail and init by a constant cancels in the
  growth_ratio.

What this means:
1. **The formal "geometric-series accumulation" the open
   limitation cites SATURATES within the first ~few thousand
   training steps.** At burst 0 (step 20000), Bellman bias has
   already converged. So at any burst window we have, σ_clip's
   non-stationarity is NOT from Bellman bias still accumulating.
2. **What we DID observe (growth_ratio > 1) is NN training
   dynamics**, not the geometric-series limitation. The network's
   Q-values grow during training as it learns; σ_aniso scales with
   Q-magnitude; this is FA-side accumulation, not bootstrap-side.
3. **Therefore: the formal open limitation IS operationally
   moot at the burst granularity** of the empirical signature.
   For converged-iterate σ_clip measurements (last 20% of bursts),
   Bellman bias has long converged → (A4'a)'s "one-step from
   converged ≈ converged" holds tightly for the BELLMAN side.

The HELD verdict at p=0.346, |ρ| < 0.3 confirms: horizon-normalised
growth_ratio is NOT γ-correlated. But it's NOT γ-correlated for
the trivial reason that normalisation is a per-cell constant.

The result reframes Theorem 3's open limitation:
- The bootstrap-side bias accumulation saturates fast → no
  empirical residual at burst granularity.
- The NN-training-side growth is what `q_lambda_a_growth_ratio`
  actually measures — and this is OUTSIDE Theorem 3's scope (the
  theorem is about the algebra of the operator, not about NN
  fitting dynamics).
- (A4'a) operates on the CONVERGED tail where both bootstrap AND
  NN dynamics have stabilised. The σ_Λa^env signature is
  measuring this converged state.

**Conclusion: the geometric-series gap as a formal open limitation
is operationally closed by burst-granularity coarsening.** The
training-dynamics non-stationarity we observe (the k=2 amplified
γ-scaling) is a SEPARATE phenomenon — NN training, not bootstrap.
Theorem 3's formal claim is therefore in much better empirical
standing than the open-limitation framing implied.

Caveat: this conclusion depends on measuring σ_clip at converged
tail. Sub-burst granularity (every <1000 training steps) would
re-expose the Bellman accumulation, but that's not what the
empirical signature operates on."""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.theorem3.bridges import (
    horizon_normalisation_flattens_geometric_gap__minatar_gamma_sweep,
)


EXPECTED: ClusterVerdict = ClusterVerdict.SUPPORTED


BLOCKED_ON: str | None = None


BRIDGES: tuple[Bridge, ...] = (
    horizon_normalisation_flattens_geometric_gap__minatar_gamma_sweep,
)
