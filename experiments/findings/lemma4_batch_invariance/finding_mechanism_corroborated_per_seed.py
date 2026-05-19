"""Theorem 1's bias-attraction chain (high jens → poor policy)
CORROBORATED per-seed at FR γ=0.999 × vanilla × B ∈ {512, 2048}.

This Finding refines the sibling pre-registered REFUTATION of
Cor 4.1 (B-invariance of jens magnitude). The MEAN jens shifts
with B because the escape fraction shifts; within-B per-seed
the mechanism chain works as Theorem 1 predicts.

Empirical content (commit `40b20f0` cache, 60 cells at B≥512):

    B       n_escape/30   ρ(jens, best_burst)   p
    128     3 (trivial)   +0.21                 NS
    512     11 (17%)      −0.48                 0.008
    2048    15 (20%)      −0.67                 <0.001

Bridge `mechanism_jens_predicts_outcome_within_high_B__fr_g999_
vanilla` pools B ∈ {512, 2048} via Fisher-z; predicts ρ ≤ −0.4
significantly negative.

Why this is a refinement, not a contradiction:
- Cor 4.1's REGIME-CLASSIFICATION B-invariance holds for the
  modal/median seed at every B (vanilla is bias-dominated, all
  3 B values).
- Cor 4.1's MAGNITUDE B-invariance fails via tail mixing: the
  escape fraction (rare seeds finding the policy) shifts with
  B, dragging the cross-seed jens mean down with it.
- Within-B, escaped vs stuck seeds DO follow Theorem 1's chain:
  lower jens → better policy.

So the original Lemma 4 refutation is correct (jens mean is
B-dependent), but the mechanism is "tail-mixing of stuck vs
escaped seeds", NOT a failure of Theorem 1's regime structure.
Theorem 1 holds at the per-seed level; the B-dependence is a
statistical-mixing artefact of how many seeds escape.

What this Finding DOES validate:
- Theorem 1's bias-attraction → poor-policy chain at the per-seed
  level (within-B).
- The bimodal escape mechanism: B affects ESCAPE FRACTION, not
  the trap depth.

What this Finding does NOT validate:
- That the escape mechanism is purely Lemma 4's variance-driven
  trajectory effect. Could also be seed-init effects, replay-
  buffer composition, or other finite-T phenomena. The per-seed
  ρ(jens, outcome) just shows the CHAIN works mechanistically
  when there's outcome variance to correlate with.
- That smaller B is the "worse" regime in some universal sense.
  At γ=0.999 with 1M steps, yes. At lower γ or longer training,
  the relationship may shift."""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.lemma4_batch_invariance.bridges import (
    mechanism_jens_predicts_outcome_within_high_B__fr_g999_vanilla,
)


EXPECTED: ClusterVerdict = ClusterVerdict.SUPPORTED


BLOCKED_ON: str | None = None


BRIDGES: tuple[Bridge, ...] = (
    mechanism_jens_predicts_outcome_within_high_B__fr_g999_vanilla,
)
