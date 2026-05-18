"""Pre-registered Finding: Lemma 4's B-invariance prediction for
vanilla jensen_gap at FR γ=0.999 × 1M training.

Pre-registration committed at THEORY_bootstrap_dominance.md §12
(commit `b416432`). The empirical sweep `fr_batch_size_sweep`
(running 2026-05-18) materialises the test panel: B ∈ {128, 512,
2048} new sub-corpora + the existing B=32 canonical FR γ=0.999
cells. The bridge resolves once `batch_2048` completes and the
top-level merge produces the canonical corpus.

EXPECTED: SUPPORTED (Lemma 4 holds in expectation; the
refutation criterion |ρ| > 0.5 is conservative).

Honest expectation: HELD is the most likely outcome. Lemma 4
is textbook SGD theory; the only way it fails empirically at
1M-step training is if finite-T escape probabilities differ
substantially across B-levels (THEORY note §7 caveat). At FR
γ=0.999 the algorithm doesn't quite reach the bias-equilibrium
fixed point (q_late,V ≈ 8 vs Lemma 2 analytic 18.4), leaving
room for finite-T trajectory differences. Whether those
differences reach the |ρ| > 0.5 refutation threshold is the
empirical question.

Possible outcomes:
- HELD (|ρ| ≤ 0.5): Lemma 4 → Corollary 4.1 corroborated. The
  expected regime classification's B-independence carries
  through to 1M-step empirical jens. Theorem 1's regime
  predictions are robust to batch-size choices in practice.
- NO_EFFECT (significant trend in either direction): the §7
  caveat bites. Finite-T escape probability is significantly
  B-dependent. Theorem 1's expected-fixed-point claim still
  holds, but practitioners need to account for batch-size in
  per-trajectory training plans.
- UNDERPOWERED (insufficient signal at this n × B-levels):
  add longer training or more seeds before re-running.

This Finding is BLOCKED_ON the sweep completion. Once the cache
materialises (commit on sweep completion), the verdict resolves.

Methodology cross-refs:
- THEORY note §7 (Lemma 4) + Corollary 4.1.
- THEORY note §12 — the pre-registration text.
- `experiments/configs/fr_batch_size_sweep.yaml` — the sweep
  config that produces the test panel."""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.lemma4_batch_invariance.bridges import (
    lemma4_b_invariance__fr_g999_vanilla,
)


EXPECTED: ClusterVerdict = ClusterVerdict.SUPPORTED


BLOCKED_ON: str | None = (
    'fr_batch_size_sweep mid-flight (batch_2048 running 2026-05-18); '
    'verdict resolves on sweep completion + top-level merge + ingest.'
)


BRIDGES: tuple[Bridge, ...] = (
    lemma4_b_invariance__fr_g999_vanilla,
)
