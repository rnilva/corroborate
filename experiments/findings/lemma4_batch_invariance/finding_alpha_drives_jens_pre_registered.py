"""Pre-registered Finding (commit `e700e55` + `18a7597`):
Lemma 4 refutation's mechanism is α/√B effective-noise, not
pure-B. Spearman ρ(α, jens) at fixed B=128 HELD.

Pre-registration captured at sweep launch:
- `pre_registration.json` materialised at
  `experiments/probes/fr_lr_sweep_at_b128/` before any cell ran.
- Bridge `lr_drives_jens_up__fr_b128_g999_vanilla` source hash
  captured at sweep launch.
- Predicted direction: `a_gt_b` (positive ρ).
- Predicted verdict: `held`.
- Sweep config + bridge source pinned to git commit hash.

Empirical result (180 cells, 6 cells × 30 seeds, 4.5h CPU):

    α       α/√B    predicted jens (canonical α=1e-4 at)   empirical jens
    2.5e-5  2.21e-6 1.55 (canonical B=2048)                  1.80 (within 16%)
    1.0e-4  8.84e-6 4.56 (identity, canonical B=128)         4.56 (exact)
    2.0e-4  1.77e-5 11.4 (canonical B=32 default)            7.65 (67% of pred)

Bridge HELD: Spearman ρ = strong positive at p=0. Direction
prediction corroborated.

Magnitude prediction: PARTIALLY MATCHES. α=2.5e-5 and α=1e-4
match canonical references closely (16% and 0% deviation
respectively). α=2e-4 UNDERSHOOTS canonical B=32 by 33%.

**Honest empirical content:**

1. The α/√B effective-noise hypothesis explains the Lemma 4
   refutation direction. Larger effective noise → bias chain
   amplifier compounds → larger jens.

2. The magnitude is NOT purely α/√B — there's residual ~30%
   B-specific structure beyond effective noise. Candidate
   sources: replay buffer sample correlation at small B,
   Adam's variance normalisation interacting non-linearly with
   α, optimizer.warmup_steps×α interaction.

3. The original Lemma 4 → Cor 4.1 refutation
   (`findings_lemma4_b_invariance_refuted.md`) had its mechanism
   ATTRIBUTION WRONG. The empirical ρ(B, jens)=−0.83 was
   confounded with effective-noise variation under fixed α.

4. Refined Theorem 1 / Lemma 4 stance: the regime-classification
   prediction (Λ_m ≫ 1 → bias-dominated at all B) holds. The
   magnitude depends on α/√B PLUS some residual B-structure.
   Practitioner takeaway: at γ→1, both α and B matter for jens
   magnitude; α/√B is the leading-order knob, with ~30% residual.

EXPECTED: SUPPORTED. The pre-registered prediction held at the
direction-significance criterion.

**Why this is a CORROBORATION not a refutation:**

The pre-registration committed `a_gt_b` direction with `held`
verdict. Bridge fires with ρ > 0.3 sig positive (concretely,
ρ huge, p=0). The framework's typed verdict-shape matches the
committed shape exactly.

The magnitude undershoot at α=2e-4 is HONEST empirical content
that REFINES the mechanism interpretation but doesn't refute the
DIRECTIONAL pre-registration. Distinguishing direction-prediction
from magnitude-prediction is part of the framework's honest
verdict discipline."""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.lemma4_batch_invariance.bridges import (
    lr_drives_jens_up__fr_b128_g999_vanilla,
)


EXPECTED: ClusterVerdict = ClusterVerdict.SUPPORTED


BLOCKED_ON: str | None = None


BRIDGES: tuple[Bridge, ...] = (
    lr_drives_jens_up__fr_b128_g999_vanilla,
)
