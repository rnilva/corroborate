"""DDQN as Hasselt's three-factor bound, with shaping as a
separable outcome-translation axis.

Hasselt 2010's structural bound on Q-learning's overestimation:

    bias ≤ σ_action × √(2 ln K) × 1/(1 − γ)

Three multiplicative factors, three places to intervene. The
bridges in this module test each factor in isolation, then test
how bias-reduction translates to outcome and how reward shaping
decouples that translation.

**The Hasselt-bound cluster (3 bridges)** — one per factor:

- **K factor `√(2 ln K)`** — `ddqn_reduces_jens_uniformly_across_k_at_fr_high_gamma`:
  at FR γ=0.999 × MLP × unshaped, DDQN's `jensen_gap` reduction
  scales monotonically across k_eff ∈ {4, 8, 12, 16}. Stratified
  Cohen's d ≤ −0.5 at every stratum. HELD.

- **γ factor `1/(1 − γ)`** — `ddqn_reduction_amplified_by_gamma__fr_mlp_k4_unshaped`:
  at FR × MLP × unshaped × k_eff=4 (controlled K), per-γ Cohen's
  d ≤ −0.8 at both γ ∈ {0.99, 0.999} AND |mean_diff(γ=0.999)| ≥ 3×
  |mean_diff(γ=0.99)|. Empirical amplification ratio 46.6×
  exceeds the structural 10× via variance amplification. HELD.

- **σ_action factor (FA-capacity gate)** — rule + exception cluster:
  - rule `linear_fa_caps_type_1_across_envs__null_panel`:
    at linear FA, σ_action is bounded by FA capacity → Hasselt
    mech dormant → DDQN has nothing to reduce. Per-env Cohen's d
    on `jensen_gap` sits in ±0.3 band across 6 envs
    (FR, Acrobot, MountainCar, CartPole, Catch-bsuite,
    DeepSea-bsuite). Currently POWER_INSUFFICIENT only because
    Catch's CI = [−0.31, +0.20] straddles the band by 0.01.
  - exception `linear_fa_cap_fails_at_metamaze_g999__exception`:
    at MetaMaze γ=0.999 × linear, the cap FAILS — DDQN reduces
    jens by d ≤ −0.3 at every n_episodes stratum. The mechanism
    here is FA-fit-error × state-distribution-shift (MetaMaze
    re-draws the maze per evaluation episode → linear FA can't
    generalise → bias enters via FA-fit error, not σ × √(2 ln K)).
    Encoded as the opposite-direction prediction in the same
    (env, γ) scope. HELD.

**The shaping-decouples cluster (2 sibling bridges)** — the
fourth, orthogonal axis: how bias-reduction translates to outcome.

- positive arm `ddqn_helps_outcome_at_fr_g999_mlp_unshaped__k_panel`:
  at FR γ=0.999 × MLP × unshaped (the "all Hasselt factors
  active" reference cell), DDQN improves outcome
  (`eval_best_burst_raw_mean`) at every k_eff stratum. HELD.
- null arm `shaping_decouples_outcome_benefit__fr_shaped_fa_x_gamma_panel`:
  at FR × shaped × {linear, MLP} × γ ∈ {0.99, 0.999}, DDQN's
  outcome effect is never positive — shaping provides an
  alternative dense policy-gradient signal that decouples
  bias-reduction from outcome improvement. Predicted_direction
  `'a_lt_b'` (asymmetric: negative effects consistent).
  HELD.

The cluster pattern: positive (unshaped) + null (shaped) reads
as "shaping moderates the bias→outcome translation at FR ×
MLP × γ ∈ {0.99, 0.999}".

**Memory cross-references** for the substantive narrative
(theory + interpretation; framework-side claims live here):
- `findings_two_types_of_bias` — decomposing Type 1 (DDQN reduces)
  vs Type 2 (FA/γ-truncation, DDQN cannot reduce) bias.
- `findings_shaping_decouples_bias_from_outcome` — the shaping
  intervention's mechanism story.
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
  by C1 + C3a. **Ingest via absolute path** (the `--ingest <name>`
  shorthand resolves only against `experiments/data/`, missing
  the `experiments/probes/` root):

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
# admitted — C2's "linear FA caps Type 1 across envs" claim is
# env-agnostic and bsuite envs are valid scoping points. Earlier
# MODULE_SCOPE excluded `*-bsuite` for canonical-cohort discipline
# in other hypotheses; that exclusion blocked C2's new-env
# extension at ingest time so we drop it here.


# Register hypothesis-local derived measurables.
from experiments.findings.ddqn_three_conditions import _measurables  # pyright: ignore[reportUnusedImport]  # noqa: F401  -- registers @measurable side-effects
from experiments.findings.ddqn_three_conditions._arms import (
    INTERVENTION as INTERVENTION,
)
from experiments.findings.ddqn_three_conditions._common import (
    CLAIM as CLAIM,
)
from experiments.findings.ddqn_three_conditions import (
    finding_three_conditions,
)
from experiments.findings.ddqn_three_conditions.conditions import (
    ddqn_helps_outcome_at_fr_g999_mlp_unshaped__k_panel,
    ddqn_reduces_jens_uniformly_across_k_at_fr_high_gamma,
    ddqn_reduction_amplified_by_gamma__fr_mlp_k4_unshaped,
    fa_capacity_moderates_ddqn_jens_reduction,
    linear_fa_cap_fails_at_metamaze_g999__exception,
    shaping_decouples_outcome_benefit__fr_shaped_fa_x_gamma_panel,
)


BRIDGES = (
    ddqn_reduces_jens_uniformly_across_k_at_fr_high_gamma,
    ddqn_reduction_amplified_by_gamma__fr_mlp_k4_unshaped,
    fa_capacity_moderates_ddqn_jens_reduction,
    linear_fa_cap_fails_at_metamaze_g999__exception,
    ddqn_helps_outcome_at_fr_g999_mlp_unshaped__k_panel,
    shaping_decouples_outcome_benefit__fr_shaped_fa_x_gamma_panel,
)


FINDINGS = (
    finding_three_conditions,
)


# `shaping_kind` and `fa_kind` auto-enter via scope predicates;
# `k_eff` only appears in `stratify_by=(..., 'k_eff', ...)` of
# C1 and C3a, which the runner's scope-walk does not see — so
# k_eff must be declared explicitly here (otherwise corpora that
# don't already carry it in their hashes/measurements parquet
# get NaN values, collapsing the k_eff panel to a single stratum
# at ingest time).
REQUIRED_MEASURABLES: tuple[str, ...] = (
    'k_eff',
    'q_action_std_late',
    'clip_wedge_polarity_aligned',  # signed wedge × polarity
    'bootstrap_gap_magnitude',      # |wedge| per cell
    'bootstrap_gap_frac_active',    # fraction of steps with wedge > 0
)
# `q_action_std_late` is the proper σ_action measure (within-state
# across-action Q SD, averaged over the late 50% of training,
# per `findings_sigma_K_scaling_corroborated`). C2's mechanism-
# discriminator bridge predicts σ_action(vanilla, linear FA)
# bounds where the FA-cap claim holds: small σ_action → mech
# dormant → null Δ_jens; large σ_action → mech fires → Δ_jens
# substantially negative. MetaMaze γ=0.999 is the suspected
# breaker (recurring-positive × γ→1 → value range too large for
# linear FA to bound σ_action).


__all__ = (
    'BRIDGES',
    'CLAIM',
    'FINDINGS',
    'INTERVENTION',
    'MODULE_SCOPE',
    'REQUIRED_MEASURABLES',
)
