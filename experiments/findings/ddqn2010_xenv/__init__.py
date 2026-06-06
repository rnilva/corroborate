"""Cross-env DDQN-2010 (paired_dqn) study: mechanism ↛ outcome.

The independent-estimator van Hasselt 2010 program (`paired_dqn`)
eliminates overestimation (jensen_gap → 0) SCOPE-INVARIANTLY across 4
MinAtar envs at γ=0.999 — but its OUTCOME effect is scope-DEPENDENT,
spanning free lunch (SpaceInvaders d≈+5.7, Asterix +1.7) through mild
help (Freeway +0.6) to harm (Breakout −2.9) per env.

Contrast is `paired_dqn` vs `dqn` (vanilla) on the typed `program`
column (ddqn2016 scoped out). The cross-env claims use a DIRECTION
sign-test (`cross_env_consistency_binomial`) — the per-env magnitudes
are too heterogeneous (I²≈0.98) for a defensible pooled magnitude. At
only 4 envs BOTH cross-env claims are UNDERPOWERED (mechanism 4/4 but
binomial p=0.0625; outcome 3/4, Breakout flips) — the framework's
POWER_INSUFFICIENT-as-first-class verdict. The per-env effects are
real; the cross-env population claims await more envs (see each
Finding's BLOCKED_ON). Cache built by `_exploration.py`.
"""
from __future__ import annotations

from corroborate.core.intervention import DoEffect

from experiments.findings.ddqn2010_xenv._scope import ENVS, MODULE_SCOPE
from experiments.findings.ddqn2010_xenv.bridges import (
    paired_improves_outcome_consistently__minatar4,
    paired_reduces_jens_consistently__minatar4,
)
from experiments.findings.ddqn2010_xenv import (
    finding_mechanism_invariant,
    finding_outcome_scope_dependent,
)

# Placeholder: the treatment is the `paired_dqn` PROGRAM swap,
# contrasted via `arm_field='program'` inside the bridges — there is no
# slot-swap DoEffect for a program change. The bridges are measurable-
# sourced (source is a str), so verdict-time `evaluate` reads
# treatment/baseline from the bridge kwargs and never consults this
# DoEffect (verified). FRAGILE-BY-OMISSION: this module deliberately
# does NOT export `CLAIM` — adding one would activate the
# `exogenous_source` gate and flip the str-sourced INTERVENTIONAL
# bridges to INADMISSIBLE. Keep CLAIM unset (or relax the bridges).
INTERVENTION = DoEffect(arms=((), ()))

BRIDGES = (
    paired_reduces_jens_consistently__minatar4,
    paired_improves_outcome_consistently__minatar4,
)

FINDINGS = (
    finding_mechanism_invariant,
    finding_outcome_scope_dependent,
)

__all__ = ('INTERVENTION', 'MODULE_SCOPE', 'BRIDGES', 'FINDINGS', 'ENVS')
