"""DDQN K-scaling at FourRooms γ=0.999 — single observational
bridge from the within-FR action_duplicate sweep.

**Substantive claim**: at FR γ=0.999 × MLP[64,64] × no-shaping,
DDQN reduces `jensen_gap` uniformly across action-multiplier
k_eff ∈ {4, 8, 12, 16}. Multi-stratum HELD via
`stratified_arm_diff_pooled`.

**History**: the module was originally authored as a three-
conditions Hypothesis claiming that DDQN's outcome benefit
requires three jointly-necessary conditions (Q-bias exists,
FA capacity, no policy-signal-shaping). Three rounds of audit
surfaced:
- Round 1 — framework-mechanics issues (direction-consistency,
  stratify_by, verdict-override transparency).
- Round 2 — single-stratum HELDs for C2/C3 smuggled
  POWER_INSUFFICIENT → HELD; "jointly necessary" claim was
  universally-quantified but tested on one env each.
- Round 3 — even with `arm_mean_diff` single-cell Welch's t and
  scoped-observational reframing, the n=30/60 power was
  inadequate for null-prediction tests (MDE ≈ 0.51-0.74 vs
  observed |d| 0.11-0.23). The C3 verdict logic also carried
  an `OR` bug making HELD one-sided non-falsifiable.

C2 and C3 were RETRACTED rather than smuggled. The module now
carries the single bridge that has multi-stratum power on the
current corpus. The retained bridge name is descriptive
(`ddqn_reduces_jens_uniformly_across_k_at_fr_high_gamma`),
not framework-claiming.

The module directory name `ddqn_three_conditions` is a historical
artifact — the substantive content is now ONE within-FR
K-scaling observation, not three conditions. Renaming the
directory would be churn; the prose tells the reader what the
module actually carries.

**Substantive corroboration of the broader framework** (two-
types decomposition, FA-capacity gate, policy-signal-strength
decoupling) lives in memory entries
`findings_two_types_of_bias`,
`findings_shaping_decouples_bias_from_outcome`,
`findings_regime_discriminator_polarity_x_gamma`. The formal
Hypothesis-Protocol surface here corroborates only the K-scaling
piece because that's what the corpus has multi-stratum power
for.

**Cache population** is canonical via `--ingest <corpus>` /
`--ingest-all <root>`. Three hypothesis-local derived
measurables (`shaping_kind`, `fa_kind`, `k_eff`) in
`_measurables.py` are auto-included via the framework's scope-
walk (`bridge.scope.meta.root_names()`)."""
from __future__ import annotations

import polars as pl

import corroborate.analyses  # pyright: ignore[reportUnusedImport]  # populate registry
import corroborate_rl.dqn.measurables  # pyright: ignore[reportUnusedImport]  # populate measurable registry


MODULE_SCOPE = (
    ~pl.col('env_name').str.ends_with('-bsuite')
    & pl.col('gamma').is_in([0.99, 0.999])
)


# Register hypothesis-local derived measurables.
from experiments.findings.ddqn_three_conditions import (  # noqa: F401
    _measurables,
)
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
    ddqn_reduces_jens_uniformly_across_k_at_fr_high_gamma,
)


BRIDGES = (
    ddqn_reduces_jens_uniformly_across_k_at_fr_high_gamma,
)


FINDINGS = (
    finding_three_conditions,
)


# Empty — scope-referenced measurables (shaping_kind, fa_kind,
# k_eff) auto-enter via `b.scope.meta.root_names()` walk per
# `CACHE_ARCHITECTURE.md` §"Measurable resolution".
REQUIRED_MEASURABLES: tuple[str, ...] = ()


__all__ = (
    'BRIDGES',
    'CLAIM',
    'FINDINGS',
    'INTERVENTION',
    'MODULE_SCOPE',
    'REQUIRED_MEASURABLES',
)
