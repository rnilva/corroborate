"""Cross-env consistency: DDQN reduces vanilla's jensen_gap at
every canonical γ=0.999 env.

Demonstration bridge for the value-level sign-test primitive
`cross_env_consistency_binomial`. At ~10 envs, the population
magnitude question (Spearman ρ between env-feature and Δ_jens)
is structurally underpowered, but the CONSISTENCY question
("at every env, Δ_jens < 0") survives at high alignment via
the binomial sign-test.

The claim shape (consistency, not magnitude) is what makes this
bridge survive at n_strata≈10 where σ_Λ_a and σ/jens moderation
bridges (population magnitude shape) honestly fired PI / REFUTED.

Empirical preview on the canonical-corpus γ=0.999 panel
(CANONICAL_G0999_CORPORA, n=9-10 envs, sign-test):
  9/10 envs negative direction (Acrobot is the lone exception,
  small magnitude). One-tailed binomial p = 0.011 → SUPPORTED.

This bridge exists primarily as a methodological demonstration:
the framework can express "consistency across envs" as a
typed-bridge separate from "magnitude scales with env-feature",
and at small n the two shapes give very different verdict
power.
"""
from __future__ import annotations


import polars as pl

from corroborate.analyses.panel.cross_env_consistency_binomial import (
    CrossEnvConsistencyBinomialResult,
    cross_env_consistency_binomial_verdict,
)
from corroborate.bridge.bridge import Direction, Tier, claim_bridge
from corroborate.bridge.verdict import RefutationClass, Verdict

from experiments.findings.ddqn._arms import (
    DDQN_ARM, INTERVENTION, VANILLA_ARM,
)
from experiments.findings.ddqn_sweeps.lambda_a_mediation import (
    CANONICAL_G0999_CORPORA,
)


@claim_bridge(
    source=INTERVENTION,
    target='jensen_gap',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=(
        (pl.col('gamma') == 0.999)
        & (
            pl.col('action_duplicate_k').is_null()
            | (pl.col('action_duplicate_k') == 1)
        )
        & pl.col('corpus').is_in(CANONICAL_G0999_CORPORA)
        & pl.col('jensen_gap').is_finite()
    ),
    predicted_direction='a_lt_b',
)
def ddqn_reduces_jens_consistently__canonical_g0999(
    cross_env_consistency_binomial: CrossEnvConsistencyBinomialResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    source: str = 'jensen_gap',
    stratify_by: tuple[str, ...] = ('env_name',),
    predicted_direction: str = 'a_lt_b',
    null_floor: float = 0.0,
    scope_predictor: str = 'jensen_gap',
    min_baseline_predictor: float = 0.0,
    min_seeds_per_arm: int = 5,
    p_threshold_held: float = 0.05,
    p_threshold_pi: float = 0.15,
    min_strata: int = 5,
) -> tuple[Verdict, RefutationClass | None]:
    """Cross-env sign-test on per-env Cohen's d for `jensen_gap`
    (DDQN − vanilla). Predicts d < 0 at every env (DDQN's
    bootstrap clip cuts bias consistently — canonical
    Hasselt-2010 mechanism on every learnable env).

    Verdict matrix:
      HELD                : binomial p ≤ 0.05 (n_strata-dependent;
                            at n=10, requires ≥9/10 same-direction)
      POWER_INSUFFICIENT  : 0.05 < p ≤ 0.15
      NO_EFFECT (NULL)    : p > 0.15, predicted-direction fraction
                            ≤ 0.6 of strata
      NO_EFFECT (SIGN_FLIP): wrong-direction fraction ≥ 0.7 of strata
                            (majority of envs show DDQN INCREASES jens)

    The verdict is robust at n=10 to single-env outliers — a
    9/10 alignment with one outlier still HELDs at p=0.011."""
    del (
        treatment_arm, baseline_arm, source, stratify_by,
        predicted_direction, null_floor, scope_predictor,
        min_baseline_predictor, min_seeds_per_arm,
    )
    # Defer to the framework-owned sign-test → verdict map (identical
    # bands; factored out so this logic lives in one tested place).
    return cross_env_consistency_binomial_verdict(
        cross_env_consistency_binomial,
        min_strata=min_strata,
        p_held=p_threshold_held,
        p_power_insufficient=p_threshold_pi,
    )


__all__ = [
    'ddqn_reduces_jens_consistently__canonical_g0999',
]
