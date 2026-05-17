"""Empirical characterization of DDQN's effect — on `jensen_gap`
(bias) and `eval_best_burst_raw_mean` (outcome) — viewed through
structural moderators that Hasselt 2010's bound suggests are
worth looking at.

The bridges here are NOT a test of Hasselt's theorem. Hasselt's
formula `bias ≤ σ_action × √(2 ln K) × 1/(1 − γ)` is a LENS —
it tells us where to look (action count, discount factor,
function-approximator capacity). What we find on each axis is
the empirical content.

- `jens_reduction_factors.py` (4 bridges) — how DDQN's effect
  on `jensen_gap` varies along k_eff, γ, FA-capacity, plus the
  MM γ=0.999 × linear exception. Cluster Finding
  `finding_jens_reduction_factors`.
- `outcome_translation.py` (2 bridges) — where DDQN's effect
  reaches `eval_best_burst_raw_mean`. Positive arm at the
  bias-active reference cell + null arm at the shaped panel.
  Cluster Finding `finding_shaping_decouples`. NB: the two
  arms test non-matched scopes; the matched-scope claim
  (FR γ=0.999 × MLP × {shaped, unshaped} × k_eff sweep)
  requires a shaped × k_eff sweep that does not yet exist.
- `sigma_action_predicts_ddqn_jens_reduction` (1 sibling
  bridge, NOT in either cluster Finding) — characterizes
  whether baseline σ_action is a useful predictor of DDQN's
  standardized effect on jens. Empirical answer: POW_INSUF /
  point estimate sign-flipped, so σ alone is NOT a clean
  predictor at this corpus's panel.

Substantive narrative cross-refs (memory):
- `findings_two_types_of_bias` — Type 1 (DDQN reduces) vs Type 2
  (FA/γ-truncation, DDQN cannot reduce) bias decomposition.
  The empirically-anchored interpretive frame.
- `findings_q_explosion_direct_evidence` — vanilla-degeneracy at
  γ→1; relevant context for the γ-axis bridge.
- `findings_sigma_K_scaling_corroborated` — earlier σ × √(2 ln K)
  framing was retracted; we now characterize σ empirically
  without theory-test framing.
- `findings_shaping_decouples_bias_from_outcome` — substantive
  narrative for the outcome null arm.
- `findings_regime_discriminator_polarity_x_gamma` — env-feature
  taxonomy for where DDQN's reduction translates to outcome.

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
    finding_gamma_amplification_anchor_gated,
    finding_jens_reduction_factors,
    finding_shaping_decouples,
)
from experiments.findings.ddqn_three_conditions._arms import (
    INTERVENTION as INTERVENTION,
)
from experiments.findings.ddqn_three_conditions._common import (
    CLAIM as CLAIM,
)
from experiments.findings.ddqn_three_conditions.jens_reduction_factors import (
    ddqn_reduces_jens_uniformly_across_k_at_fr_high_gamma,
    ddqn_reduction_amplified_by_gamma__fr_mlp_k4_unshaped,
    fa_capacity_moderates_ddqn_jens_reduction,
    linear_fa_cap_fails_at_metamaze_g999__exception,
    sigma_action_predicts_ddqn_jens_reduction,
    vanilla_anchor_collapses_with_gamma_at_fr_mlp,
    vanilla_anchor_preserved_with_gamma_at_acrobot_mlp,
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
    vanilla_anchor_collapses_with_gamma_at_fr_mlp,
    vanilla_anchor_preserved_with_gamma_at_acrobot_mlp,
)


FINDINGS = (
    finding_jens_reduction_factors,
    finding_shaping_decouples,
    finding_gamma_amplification_anchor_gated,
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
