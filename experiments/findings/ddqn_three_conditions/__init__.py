"""DDQN bias-reduction scalings (K-axis, γ-axis, FA-axis) +
outcome-translation scope. Six bridges across two cluster
Findings.

Hasselt 2010's `bias ≤ σ_action × √(2 ln K) × 1/(1 − γ)`
motivates three axes (action count, discount factor, FA
capacity) on which DDQN's bias-reduction might scale. The
bridges in `hasselt_bound.py` corroborate that the scalings are
empirically MONOTONE at this corpus's scope. They DO NOT
identify the manipulated variable with Hasselt's specific
factor — each bridge docstring disclaims the non-identification
(K-via-action_duplicate is a confounded K-proxy; γ-amplification
cannot discriminate Hasselt's 1/(1−γ) from vanilla-degeneracy at
γ→1; FA-capacity binary cannot discriminate σ_action from
Type-1 FA-truncation).

The empirically-corroborated frame is the Type 1 / Type 2
bias decomposition from `findings_two_types_of_bias`. Hasselt's
bound is the cleanest theoretical inspiration; this module's
bridges corroborate scalings consistent with — but not
identified to — that bound.

- `hasselt_bound.py` (4 bridges) — K-axis, γ-axis, FA-axis
  rule + FA-axis exception. Cluster Finding `finding_hasselt_bound`.
- `outcome_translation.py` (2 bridges) — positive arm at the
  bias-active reference cell + null arm at FR × shaped panel.
  Cluster Finding `finding_shaping_decouples`. NB: the two
  arms test non-matched scopes — the matched-scope claim
  (FR γ=0.999 × MLP × {shaped, unshaped} × k_eff sweep)
  requires a shaped × k_eff sweep that does not yet exist.

Substantive narrative cross-refs (memory):
- `findings_two_types_of_bias` — Type 1 (DDQN reduces) vs Type 2
  (FA/γ-truncation, DDQN cannot reduce) bias decomposition.
- `findings_q_explosion_direct_evidence` — alternative mechanism
  the γ-axis bridge cannot discriminate against.
- `findings_sigma_K_scaling_corroborated` — the previously
  retracted σ × √(2 ln K) claim; the FA-axis bridge avoids the
  same retracted shape only by disclaiming σ_action attribution.
- `findings_shaping_decouples_bias_from_outcome` — narrative for
  the null arm; partly tested, partly asserted.
- `findings_regime_discriminator_polarity_x_gamma` — env-feature
  taxonomy for when DDQN's reduction translates to outcome.

**Cache population** is canonical via `--ingest <corpus>` /
`--ingest-all <root>`. Three hypothesis-local derived
measurables (`shaping_kind`, `fa_kind`, `k_eff`) in
`_measurables.py` are auto-included via the framework's scope-
walk (`bridge.scope.meta.root_names()`). Required corpora at
the canonical scope:

- `fa_depth_fourrooms` — FR × {linear, MLP} × {0.99, 0.999}
  unshaped.
- `fa_depth_xenv_gpu` — Acrobot/MM/MC × {linear, MLP} ×
  {0.99, 0.999} unshaped.
- `fa_degeneracy_shaped_only` — FR × {linear, MLP} ×
  {0.99, 0.999} shaped (PotentialReward).
- `experiments/probes/action_dup_mismatch_probe_g999_1M/ddqn_vs_vanilla`
  (k_eff ∈ {4, 8, 12} from action_dup k=1,2,3) AND
- `experiments/probes/action_dup_mismatch_probe_g999_1M_FR_k4_only/ddqn_vs_vanilla`
  (k_eff = 16) — FR γ=0.999 × MLP × k_eff sweep traces, used
  by the K-scaling + outcome-positive bridges. **Ingest via
  absolute path** (the `--ingest <name>` shorthand resolves
  only against `experiments/data/`, missing the
  `experiments/probes/` root):

      python -m scripts.run_hypothesis experiments.findings.ddqn_three_conditions \\
          --ingest "$PWD/experiments/probes/action_dup_mismatch_probe_g999_1M/ddqn_vs_vanilla,\\
$PWD/experiments/probes/action_dup_mismatch_probe_g999_1M_FR_k4_only/ddqn_vs_vanilla,\\
fa_depth_fourrooms,fa_depth_xenv_gpu,fa_degeneracy_shaped_only"

  The `--leaves`/`catalogue` view via
  `python -m corroborate catalogue experiments/data experiments/probes
  --remote-prefix s3://corroborate-archive/` is the canonical
  discovery tool for finding which corpus directory carries the
  data a bridge needs."""
from __future__ import annotations

import polars as pl

import corroborate.analyses  # pyright: ignore[reportUnusedImport]  # populate registry
import corroborate_rl.dqn.measurables  # pyright: ignore[reportUnusedImport]  # populate measurable registry


MODULE_SCOPE = pl.col('gamma').is_in([0.99, 0.999])
# Bsuite envs (Catch, DeepSea, DiscountingChain, UmbrellaChain) are
# admitted — the FA-capacity moderator scope explicitly restricts
# to {FR, Acrobot, MountainCar} but other bridges may admit bsuite
# cells; the module-wide envelope stays env-agnostic.


# Register hypothesis-local derived measurables.
from experiments.findings.ddqn_three_conditions import _measurables  # pyright: ignore[reportUnusedImport]  # noqa: F401  -- registers @measurable side-effects
from experiments.findings.ddqn_three_conditions import (
    finding_hasselt_bound,
    finding_shaping_decouples,
)
from experiments.findings.ddqn_three_conditions._arms import (
    INTERVENTION as INTERVENTION,
)
from experiments.findings.ddqn_three_conditions._common import (
    CLAIM as CLAIM,
)
from experiments.findings.ddqn_three_conditions.hasselt_bound import (
    ddqn_reduces_jens_uniformly_across_k_at_fr_high_gamma,
    ddqn_reduction_amplified_by_gamma__fr_mlp_k4_unshaped,
    fa_capacity_moderates_ddqn_jens_reduction,
    linear_fa_cap_fails_at_metamaze_g999__exception,
    sigma_action_predicts_ddqn_jens_reduction,
)
from experiments.findings.ddqn_three_conditions.outcome_translation import (
    ddqn_helps_outcome_at_fr_g999_mlp_unshaped__k_panel,
    ddqn_no_positive_outcome_under_shaping__fr_shaped_fa_x_gamma_panel,
)


BRIDGES = (
    ddqn_reduces_jens_uniformly_across_k_at_fr_high_gamma,
    ddqn_reduction_amplified_by_gamma__fr_mlp_k4_unshaped,
    fa_capacity_moderates_ddqn_jens_reduction,
    linear_fa_cap_fails_at_metamaze_g999__exception,
    sigma_action_predicts_ddqn_jens_reduction,
    ddqn_helps_outcome_at_fr_g999_mlp_unshaped__k_panel,
    ddqn_no_positive_outcome_under_shaping__fr_shaped_fa_x_gamma_panel,
)


FINDINGS = (
    finding_hasselt_bound,
    finding_shaping_decouples,
)


# `shaping_kind` and `fa_kind` auto-enter via scope predicates;
# `k_eff` only appears in `stratify_by=(..., 'k_eff', ...)` of
# the K-scaling + outcome-positive bridges, which the runner's
# scope-walk does not see — so k_eff must be declared explicitly
# here (otherwise corpora that don't already carry it in their
# hashes/measurements parquet get NaN values, collapsing the
# k_eff panel to a single stratum at ingest time).
REQUIRED_MEASURABLES: tuple[str, ...] = (
    'k_eff',
    'q_action_std_late',
    'clip_wedge_polarity_aligned',  # signed wedge × polarity
    'bootstrap_gap_magnitude',      # |wedge| per cell
    'bootstrap_gap_frac_active',    # fraction of steps with wedge > 0
)
# `q_action_std_late` is the proper σ_action measure (within-state
# across-action Q SD, averaged over the late 50% of training,
# per `findings_sigma_K_scaling_corroborated`). The FA-moderator
# bridge predicts σ_action(vanilla, linear FA) bounds where the
# FA-cap claim holds: small σ_action → mech dormant → null
# Δ_jens; large σ_action → mech fires → Δ_jens substantially
# negative. MetaMaze γ=0.999 is the documented breaker
# (recurring-positive × γ→1 → value range too large for linear
# FA to bound σ_action).


__all__ = (
    'BRIDGES',
    'CLAIM',
    'FINDINGS',
    'INTERVENTION',
    'MODULE_SCOPE',
    'REQUIRED_MEASURABLES',
)
