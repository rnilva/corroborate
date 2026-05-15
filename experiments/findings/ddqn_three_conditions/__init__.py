"""DDQN three-conditions hypothesis.

Substrate claim: DDQN's outcome improvement over vanilla DQN
requires THREE jointly-necessary conditions on the (env, HP,
training-config) tuple:

1. **σ × √(2 ln K) × 1/(1−γ) > 0** — vanilla's max-bias has
   magnitude that compounds through bootstrap chains. Driven by
   action-space size K and discount γ.
2. **FA has representational capacity for Type 1 to manifest** —
   if FA is bounded (linear), vanilla's Q is capped before max-
   bias compounds, leaving Type 2 (FA representational error)
   dominant. DDQN's mechanism has no room to operate.
3. **Policy lacks dense alternative observational signal** — if
   the env provides dense per-step reward (or potential-based
   shaping adds it), the policy's argmax follows the shaping
   gradient rather than Q-noise. DDQN's bias-reduction still
   fires on Q but no longer translates to outcome.

Each condition is tested by a clean intervention that varies
that condition while holding the others fixed:
- C1 — FR γ=0.999 × k=1-4 (varies K)
- C2 — MC γ=0.999 × {linear, MLP[64,64]} (varies FA capacity)
- C3 — FR γ=0.999 × {unshaped, potential-shaped} (varies policy signal)

The cluster Finding `finding_three_conditions` asserts the
three conditions are jointly necessary; verdict SUPPORTED when
all three bridges HELD on their respective intervention panels.

Distinct from the canonical `experiments.findings.ddqn`
hypothesis which scopes to a single config-per-env at 1M and
tests within-canonical-scope claims (mech / link / outcome
trichotomy). This hypothesis spans MULTIPLE HP regimes via
interventions on K, FA, and shaping.

**Cache population** is canonical — `--ingest <corpus>` /
`--ingest-all <root>` walks raw `runs.parquet` + `traces.parquet`
pairs and populates the per-hypothesis cache (default at
`experiments/data/cache/ddqn_three_conditions.parquet`).
Three derived measurables (`shaping_kind`, `fa_kind`, `k_eff`)
live in `_measurables.py` — they compute from existing
substrate fields at ingest time. Scope predicates reference
these endogenous env-properties, not corpus-source labels.

**Pending follow-up — reward polarity is implicit**: the
canonical `findings_ddqn_reward_sign_conditional` shows DDQN's
clip wedge slightly HURTS in Type-2-dominated negative-polarity
envs (Acrobot, MountainCar). This is implicit in the current
C2/C3 testbeds (MC is dense-negative, FR-shaped is positive-
via-potential) but not factored as a fourth condition. A future
extension would split C3 by reward-polarity to test the clip-
wedge sign-conditional. Deferred."""
from __future__ import annotations

import polars as pl

import corroborate.analyses  # pyright: ignore[reportUnusedImport]  # populate registry
import corroborate_rl.dqn.measurables  # pyright: ignore[reportUnusedImport]  # populate measurable registry


# Defensive scope: γ ∈ {0.99, 0.999} (the two regimes this
# hypothesis tests) AND non-bsuite envs (canonical-ddqn
# convention — bsuite envs are diagnostic probes, not chain
# MDPs). AND-combined into every bridge's scope by the runner
# (`getattr(h, 'MODULE_SCOPE', None)`).
MODULE_SCOPE = (
    ~pl.col('env_name').str.ends_with('-bsuite')
    & pl.col('gamma').is_in([0.99, 0.999])
)

# Register hypothesis-local derived measurables (shaping_kind,
# fa_kind, k_eff) so `--ingest` computes them at cache-build.
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
    condition_1__q_bias_exists_under_high_gamma_and_K,
    condition_2__fa_capacity_caps_type_1_in_linear_fa,
    condition_3__shaping_decouples_mech_from_outcome,
)


BRIDGES = (
    condition_1__q_bias_exists_under_high_gamma_and_K,
    condition_2__fa_capacity_caps_type_1_in_linear_fa,
    condition_3__shaping_decouples_mech_from_outcome,
)


FINDINGS = (
    finding_three_conditions,
)


__all__ = (
    'BRIDGES',
    'CLAIM',
    'FINDINGS',
    'INTERVENTION',
    'MODULE_SCOPE',
    'REQUIRED_MEASURABLES',
)


# Declared for the framework's `--ingest` machinery — these
# measurables must be computed and persisted per cell at ingest
# time. The trio is hypothesis-local (defined in `_measurables`)
# and reads existing per-cell scalar fields; no trace
# materialization needed.
REQUIRED_MEASURABLES: tuple[str, ...] = (
    'shaping_kind',
    'fa_kind',
    'k_eff',
)
