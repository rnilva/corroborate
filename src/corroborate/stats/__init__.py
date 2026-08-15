"""Stats — statistical primitives the framework uses to derive
verdicts from corpus measurements.

Hedges' g (paired and pooled), random-effects pooling, meta-
regression. All implementation-neutral — operates on numeric arrays
and stratum keys, doesn't know what the implementation's
measurables mean.

Two sub-modules:
- `stats.effect_size` — Hedges' g (paired), PooledStats,
  random-effects pooling, MDE / power.
- `stats.meta_regression` — single- and panel-meta-regression
  with covariate coefficients and cross-validation.

Public surface re-exported here: the canonical effect-size
primitives, the pooling result type, meta-regression entry
points + result types. Internal constants (`I2_THRESHOLD`),
power-checking helpers, and verdict-derivation utilities live
on the submodule path."""
from corroborate.stats.effect_size import (
    PooledStats,
    fisher_z_pool,
    hedges_g_paired,
    random_effects_summary,
    random_effects_verdict,
    recommended_n_paired,
)
from corroborate.stats.meta_regression import (
    CovariateCoefficient,
    MetaRegressionResult,
    StratumObservation,
    meta_regress_panel,
    meta_regression,
)

__all__ = [
    'CovariateCoefficient',
    'MetaRegressionResult',
    'PooledStats',
    'StratumObservation',
    'fisher_z_pool',
    'hedges_g_paired',
    'meta_regress_panel',
    'meta_regression',
    'random_effects_summary',
    'random_effects_verdict',
    'recommended_n_paired',
]
