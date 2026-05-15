"""DDQN three-conditions Hypothesis — multi-stratum panels.

Three substantive conditions, each backed by an adequately-
powered multi-stratum panel:

C1 — DDQN reduces `jensen_gap` uniformly across action-multiplier
     k_eff at FR γ=0.999 × MLP[64,64] × no-shaping. 4 strata.
C2 — Linear FA caps Type 1 across envs at γ=0.999 × no-shaping.
     Per-env Cohen's d on `jensen_gap` sits in ±0.5 band.
     4 strata (FR + Acrobot + MM + MC).
C3a — DDQN improves outcome at FR γ=0.999 × MLP × no-shaping
      across k_eff strata. 4 strata.
C3b — Shaping decouples: no outcome benefit at FR × MLP × shaped
      across γ ∈ {0.99, 0.999} strata. 2 strata.

C3a + C3b form a sibling cluster — the moderation pattern is
read off the joint verdicts (positive + null) rather than
authored as a single bridge.

The substrate-corroborated framework prose (two-types
decomposition, FA-capacity gate, policy-signal-strength) lives
in memory entries:
- `findings_two_types_of_bias`
- `findings_shaping_decouples_bias_from_outcome`
- `findings_regime_discriminator_polarity_x_gamma`

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


MODULE_SCOPE = (
    ~pl.col('env_name').str.ends_with('-bsuite')
    & pl.col('gamma').is_in([0.99, 0.999])
)


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
    linear_fa_caps_type_1_across_envs__null_panel,
    shaping_decouples_outcome_benefit__fr_shaped_fa_x_gamma_panel,
)


BRIDGES = (
    ddqn_reduces_jens_uniformly_across_k_at_fr_high_gamma,
    linear_fa_caps_type_1_across_envs__null_panel,
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
REQUIRED_MEASURABLES: tuple[str, ...] = ('k_eff',)


__all__ = (
    'BRIDGES',
    'CLAIM',
    'FINDINGS',
    'INTERVENTION',
    'MODULE_SCOPE',
    'REQUIRED_MEASURABLES',
)
