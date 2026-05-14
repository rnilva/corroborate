"""Cross-env mediation: DDQN's mech effect predicts its outcome effect — REFUTED.

This Finding tests the substantive bias-correction claim at the
cross-env arm-diff level: each env contributes one
(Δ_predictor, Δ_outcome_raw) point; cross-env Spearman asks "are
the envs where DDQN reduces mech MORE the envs where DDQN
improves outcome MORE?"

Empirical reading at canonical 1M (12 envs, ex-CartPole-saturated):

  - `ddqn_outcome_scales_with_jens_reduction__xenv`:
      POWER_INSUFFICIENT (ρ=−0.35, p=0.19). Direction-correct
      but below the substantive HELD threshold. With the Q-MC
      algebraic identity (see below), even this modest ρ is
      partly an algebraic shadow rather than substantive
      evidence — the cross-env Δ-Δ form does NOT escape the
      tautology, contra the docstring's earlier framing.
  - `ddqn_outcome_scales_with_jens_reduction__xenv_loo_robust`:
      POWER_INSUFFICIENT — same data, robustness gate fails on
      the underpowered anchor.
  - `ddqn_outcome_scales_with_bg_frac_active__xenv`:
      **NO_EFFECT (null_effect)** (p=0.63). This is the clean
      substantive test: `bootstrap_gap_frac_active` is MC-free
      (defined entirely from Q-network outputs, no MC term),
      so its cross-env scaling is NOT algebraically pinned. It
      finds NO substantive cross-env mech→outcome relationship.

Cluster verdict: REFUTED. The bg_frac NO_EFFECT(null) is the
substantively load-bearing reading; the jens bridges' modest
negative ρ is partly algebraic.

## The Q-MC algebraic tautology in the cross-env Δ-Δ form

The substrate's `jensen_gap` ≡ `Q − MC_disc` by definition. So:

    Δ_jens = ΔQ − ΔMC_disc

When the env's disc and raw outcomes co-vary tightly
(`env_disc_raw_alignment > 0.7`), `ΔMC_disc ≈ Δ_outcome_raw`,
so:

    Δ_jens ≈ ΔQ − Δ_outcome_raw

    ρ(Δ_jens, Δ_outcome_raw)
      = ρ(ΔQ − Δ_outcome_raw, Δ_outcome_raw)
      = [cov(ΔQ, Δ_out) − var(Δ_out)] / (σ_jens · σ_out)

The −`var(Δ_out)` term guarantees a negative ρ baseline whenever
cov(ΔQ, Δ_outcome_raw) is small. This is the cross-env analog of
the per-cell Q-MC tautology — the arm-diff DOES NOT escape it
when the env's outcome-translation is tight.

**The earlier alignment-scope (`align > 0.7`) was exactly
backwards**: it scoped to envs where the tautology is most
pronounced, then read the algebraically-guaranteed ρ=−0.83 as
substantive evidence. Retracted 2026-05-14 under user critique.

The genuine substantive cross-env test of mech→outcome requires
an MC-free predictor. `bootstrap_gap_frac_active` is that
predictor; it gives NO_EFFECT(null) at canonical scope. The
substrate's "DDQN's outcome benefit scales cross-env with its
bg-wedge frequency reduction" hypothesis is empirically refuted.

## What survives substantively

- DDQN reduces `jensen_gap` and Q on every canonical env
  (within-env mech HELD; see `finding_hasselt_chain`).
- DDQN improves `eval_best_burst_raw_mean` on 11/12 canonical
  envs (within-env outcome benefit).
- **Cross-env: the magnitude of mech reduction does NOT scale
  predictably with the magnitude of outcome improvement** when
  tested with a properly MC-free predictor. The env-to-env
  conversion ratio (mech reduction → outcome improvement) is
  env-structural noise from the cross-env test's perspective.

This is consistent with `findings_canonical_scope_reverification`:
substrate-level mech claims hold; cross-env outcome-translation
claims don't survive at canonical scope.

## What `CHAINED_BRIDGES_DESIGN.md` would fix

The algebraic entanglement between the alignment scope (an edge
on `MC_disc` and `raw`) and the dependent bridge (an edge on
`jens = Q − MC_disc` and `raw`) is invisible when scope is a
polars expression. Promoting alignment to a first-class
precondition edge would make the shared `MC_disc` node visible
in the graph topology — surfacing the tautology BEFORE empirical
evaluation. See the design doc."""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn.cross_env_mediation import (
    ddqn_outcome_scales_with_bg_frac_active__xenv,
    ddqn_outcome_scales_with_jens_reduction__xenv,
    ddqn_outcome_scales_with_jens_reduction__xenv_loo_robust,
)


EXPECTED: ClusterVerdict = ClusterVerdict.REFUTED


BLOCKED_ON: str | None = None


BRIDGES: tuple[Bridge, ...] = (
    ddqn_outcome_scales_with_jens_reduction__xenv,
    ddqn_outcome_scales_with_jens_reduction__xenv_loo_robust,
    ddqn_outcome_scales_with_bg_frac_active__xenv,
)
