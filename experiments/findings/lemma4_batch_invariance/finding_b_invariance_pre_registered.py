"""Pre-registered Finding: Lemma 4 → Corollary 4.1's B-invariance
prediction at FR γ=0.999 × 1M training — **REFUTED** at the
pre-registered threshold.

Pre-registration committed at THEORY_bootstrap_dominance.md §12
(commit `b416432`). The empirical sweep `fr_batch_size_sweep`
ran 2026-05-18 (14 hours wall time), B ∈ {128, 512, 2048} × FR
γ=0.999 × MLP[64,64] × unshaped × n=30 seeds per arm.

**Empirical result:**

    B       jens_mean   jens_SD   n
    128     4.56        1.51      30
    512     2.41        0.76      30
    2048    1.55        0.64      30

ρ(B, jens) Spearman = **−0.832, p=3e-24** — well past the
pre-registered |ρ| > 0.5 refutation criterion.

**Direction**: larger B → SMALLER jens (3× drop B=128 → B=2048).
Cross-seed SD also drops with B (1.51 → 0.64), consistent with
Lemma 4's Var[∇L] = O(1/B) prediction. The mean is what fails.

**Interpretation**: at FR γ=0.999 in 1M steps, larger B gives
smoother gradient updates → less variance-driven Q-growth → less
bias accumulation. Smaller B's high-variance updates compound
into more aggressive Q-divergence under the γ→1 chain amplifier.

This is the OPPOSITE direction the §7 caveat suggested. The
caveat anticipated small B might HELP escape bias-attraction;
instead, small B AMPLIFIES bias-attraction at FR γ=0.999.

**Implications:**

1. **Lemma 4's mathematical claim (E[∇L] B-invariant) is true.**
   Variance reduction with larger B is empirically confirmed
   (jens_SD drops 2.4× from B=128 to B=2048).

2. **Corollary 4.1 ("regime classification B-invariant") is
   empirically FALSE.** Theorem 1's Λ_m predicts the same regime
   at any B, but the 1M-step empirical jens magnitude moves 3×
   across batch sizes — the regime CLASSIFICATION may stay the
   same (bias-dominated at all B) but the magnitude is
   B-sensitive at this scope.

3. **Practitioner implication**: at γ→1 with finite training,
   batch size matters. Using B=32 or B=128 gives systematically
   different (larger) bias accumulation than B=2048. The choice
   isn't free even in expectation if "expectation" means
   1M-step finite training.

4. **Why empirical ≠ Lemma 4's expectation**: at 1M steps the
   algorithm hasn't reached the population fixed point. The
   trajectory's path matters, not just the limit. SGD variance
   compounds into bias-magnitude differences via the γ-chain
   amplifier.

EXPECTED: REFUTED (verdict matches the pre-registered direction).
The framework's typed bridge surfaces the refutation cleanly.

Methodology cross-refs:
- THEORY note §7 (Lemma 4) + Corollary 4.1.
- THEORY note §12 — the pre-registration text (committed
  before the sweep ran, so this is an honest pre-reg test).
- `experiments/configs/fr_batch_size_sweep.yaml` — the sweep
  config that produced the test panel."""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.lemma4_batch_invariance.bridges import (
    lemma4_b_invariance__fr_g999_vanilla,
)


EXPECTED: ClusterVerdict = ClusterVerdict.REFUTED


BLOCKED_ON: str | None = None


BRIDGES: tuple[Bridge, ...] = (
    lemma4_b_invariance__fr_g999_vanilla,
)
