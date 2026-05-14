"""Cross-γ scaling bridge — moved from `ddqn/outcome_scope.py` to
`ddqn_sweeps/` 2026-05-14 because canonical pins γ=0.99 (n_strata=1,
structurally unable to fire). γ-sweep lives in `ddqn_sweeps` per
the canonical-vs-HP-sweep separation in
`experiments/findings/ddqn/_scope.py`.

The bridge asks "does DDQN's outcome benefit scale with chain depth
(γ → 1)?" Operationalized as cross-γ Spearman ρ on per-γ Cohen's d.
At canonical γ ∈ {0.99}, n_strata=1 → structural POW_INSUF. Here
γ varies (sweep cohort), n_strata=3 max — still below the n=10
resolution band of the verdict helper, but at least the data
covers the predictor's range."""
from __future__ import annotations

from types import MappingProxyType

import polars as pl

from corroborate.analyses.cross_stratum_property_slope import (
    CrossStratumPropertySlopeResult,
)
from corroborate.bridge.bridge import Direction, Tier, claim_bridge
from corroborate.bridge.verdict import RefutationClass, Verdict

from experiments.findings.ddqn._arms import (
    DDQN_ARM, INTERVENTION, VANILLA_ARM,
)
from experiments.findings.ddqn._verdicts import (
    cross_stratum_signed_spearman_verdict,
)


# Per-γ log effective horizon = -log(1-γ). Independent of env.
_LOG_HORIZON_PER_GAMMA: MappingProxyType[object, MappingProxyType[str, float]] = (
    MappingProxyType({
        0.99:  MappingProxyType({'log_horizon': 4.605}),
        0.995: MappingProxyType({'log_horizon': 5.298}),
        0.999: MappingProxyType({'log_horizon': 6.908}),
    })
)


# CLAIM 36 — Link-layer: DDQN's OUTCOME benefit scales WITH chain
# depth (log effective horizon).
@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_raw_mean',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=(
        pl.col('eval_best_burst_raw_mean').is_finite()
        & pl.col('jensen_gap').is_finite()
        & pl.col('n_actions').is_finite() & (pl.col('n_actions') >= 3)
        & ((pl.col('n_step') == 1) | pl.col('n_step').is_null())
        & pl.col('action_duplicate_k').is_null()
        & ~((pl.col('env_name') == 'MetaMaze-misc')
            & (pl.col('gamma') == 0.999))
        & (pl.col('env_name') != 'CartPole-v1')
        & (pl.col('env_name') != 'SlidingTilePuzzle-jumanji')
        & (pl.col('wrappers') == '()')
        & pl.col('gamma').is_in(tuple(_LOG_HORIZON_PER_GAMMA.keys()))
    ),
    predicted_direction='a_gt_b',
)
def ddqn_link_outcome_scales_with_chain_depth__cross_env(
    cross_stratum_property_slope: CrossStratumPropertySlopeResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    source: str = 'eval_best_burst_raw_mean',
    stratify_by: tuple[str, ...] = ('gamma',),
    covariate_name: str = 'log_horizon',
    covariate_key_field: str = 'gamma',
    covariates_per_key: MappingProxyType[object, MappingProxyType[str, float]] = (
        _LOG_HORIZON_PER_GAMMA
    ),
    scope_predictor: str = 'jensen_gap',
    min_vanilla_predictor: float = 2.0,
    min_seeds_per_arm: int = 5,
    rho_threshold_held: float = 0.6,
    p_threshold: float = 0.05,
    null_threshold: float = 0.2,
    sign_flip_threshold: float = 0.5,
    min_strata: int = 10,
) -> tuple[Verdict, RefutationClass | None]:
    """Cross-γ Spearman ρ between `log_horizon` and per-γ Cohen's d
    on raw outcome.

    **Predicted direction**: ρ > 0 — longer chain (γ → 1) → larger
    DDQN outcome benefit. The theory's axis (ii) operates at the
    LINK layer: chain-depth amplifies the compounding-bias problem
    that DDQN corrects.

    **Calibrated for n_strata≥10**. The sweep cohort has γ ∈ {0.99,
    0.995, 0.999} so n_strata≤3 — fundamentally below the verdict
    helper's resolution band. Fires POWER_INSUFFICIENT honestly.
    This bridge documents the question's existence and the
    structural limit; cannot fire HELD without denser γ coverage."""
    del treatment_arm, baseline_arm, source, stratify_by
    del covariate_name, covariate_key_field, covariates_per_key
    del scope_predictor, min_vanilla_predictor, min_seeds_per_arm
    return cross_stratum_signed_spearman_verdict(
        cross_stratum_property_slope,
        sign=1,
        rho_threshold_held=rho_threshold_held,
        p_threshold=p_threshold,
        null_threshold=null_threshold,
        sign_flip_threshold=sign_flip_threshold,
        min_strata=min_strata,
    )


BRIDGES = (
    ddqn_link_outcome_scales_with_chain_depth__cross_env,
)
