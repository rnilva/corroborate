"""Stats — statistical primitives the framework uses to derive
verdicts from corpus measurements.

Hedges' g (paired and pooled), random-effects pooling, meta-
regression, MDE / power computation. All substrate-neutral —
operates on numeric arrays and stratum keys, doesn't know what
the substrate's measurables mean.

The decomposition mirrors the file boundary:
- `stats.core` — Hedges' g, PooledStats, random-effects, MDE,
  paired power tools.
- `stats.meta_regression` — single- and panel-meta-regression
  with covariate coefficients and cross-validation.

Consumers import from `corroborate.stats` (this module) — both
sub-modules' public surface re-exported here."""
from corroborate.stats.effect_size import (
    I2_THRESHOLD,
    PooledStats,
    adequately_powered_paired,
    derived_q_from_g_se,
    delta_i_from_q,
    hedges_g_paired,
    mde_paired,
    random_effects_summary,
    random_effects_verdict,
    recommended_n_paired,
    verdict_from_paired_stats,
)
from corroborate.stats.meta_regression import (
    CovariateCoefficient,
    CrossValResult,
    FoldResult,
    MetaRegressionResult,
    Pool,
    StratumGProtocol,
    StratumObservation,
    cross_validate_meta_regression,
    meta_regress_comparison,
    meta_regress_panel,
    meta_regression,
)

__all__ = [
    'CovariateCoefficient',
    'CrossValResult',
    'FoldResult',
    'I2_THRESHOLD',
    'MetaRegressionResult',
    'Pool',
    'PooledStats',
    'StratumGProtocol',
    'StratumObservation',
    'adequately_powered_paired',
    'cross_validate_meta_regression',
    'delta_i_from_q',
    'derived_q_from_g_se',
    'hedges_g_paired',
    'mde_paired',
    'meta_regress_comparison',
    'meta_regress_panel',
    'meta_regression',
    'random_effects_summary',
    'random_effects_verdict',
    'recommended_n_paired',
    'verdict_from_paired_stats',
]
