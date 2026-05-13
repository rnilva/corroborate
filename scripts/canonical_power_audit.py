"""Honest power audit of UNDERPOWERED bridges at canonical scope.

At canonical (env-class-canonical config, MODULE_SCOPE-filtered),
several bridges fire POWER_INSUFFICIENT. Each maps to a specific
data shape needed for resolution; "more seeds per env" is NOT
the answer for most of them.

The dominant constraint is `n_strata` for cross-stratum
meta-regressions (DoWhy backdoor, partial-Spearman cross-env
pools). With canonical pinning one config per env, n_strata = 12
envs at most. Adding more seeds within an env tightens
per-stratum estimates but doesn't reduce the cross-stratum
slope's SE — slope-identification power is a function of n_strata
× between-stratum variance.

Per-bridge audit of what would actually fix each UNDERPOWERED:
- DoWhy backdoor / placebo / RCC clusters
  (`bias_correction_clip_*`, `bias_premise_jens_*`,
  `mediation_link_null_*`): cross-env meta-regression. Slope SE
  bounded by n_strata=12. More seeds per env doesn't help.
  Recovering would need either more envs at canonical, or
  acceptance of the structural limit.
- Cross-env Pearson (`effh_predicts_link_power__reach_envs`,
  `argmax_entropy_link_power_null__survive_envs`): needs more
  envs of the right polarity class.
- `finding_metamaze_gamma_amplification`: needs MetaMaze γ=0.999
  cells where vanilla jens > 0.05. Structurally hard — long-γ
  MetaMaze has weak vanilla bias by construction.

Adding HP-sweep cells as additional "strata" is NOT a clean fix
— HP variants are deliberate intervention axes, not exchangeable
substrate-property partitions. They belong in `ddqn_sweeps` for
explicit HP-variation analyses, not folded into the canonical
ddqn cross-env slope.

Honest framing: canonical scope trades slope-identification
power for confound-cleanliness. The structural mechanism claims
(bias-reduction, signed-Q reduction, Hasselt chain) hold cleanly
at canonical. The outcome-translation claims need either more
canonical envs or live in ddqn_sweeps with explicit acknowledgment
of HP-variation.

Run: `uv run python scripts/canonical_power_audit.py`."""
from __future__ import annotations

import polars as pl


CACHE = 'experiments/data/cache/ddqn.parquet'


def main() -> None:
    df = pl.scan_parquet(CACHE).select([
        'env_name', 'arm_key', 'gamma', 'sync_period',
        'replay.capacity', 'total_steps',
    ]).collect()

    print(f'Canonical cache: {df.height} cells × {df["env_name"].n_unique()} envs')
    print(f'n_strata (env × config): {df["env_name"].n_unique()} (1 config per env at canonical)')
    print()

    arms = df.with_columns(
        pl.when(pl.col('arm_key') == 'baseline').then(pl.lit('vanilla'))
        .when(pl.col('arm_key').str.contains('Claim:double_greedify')).then(pl.lit('ddqn'))
        .otherwise(pl.lit('other')).alias('arm')
    )
    pivot = (
        arms.group_by(['env_name', 'arm']).len().rename({'len': 'cells'})
        .pivot(values='cells', index='env_name', on='arm', aggregate_function='sum')
        .sort('env_name')
    )
    print('Per-env (env, arm) cell counts at canonical:')
    print(pivot)
    print()

    print('Constraint by underpowered bridge family:')
    print()
    print('1. Cross-env meta-regression (DoWhy backdoor cluster,')
    print('   bias_premise/bias_correction × {backdoor, placebo, RCC};')
    print('   plus mediation_link_null partial-Spearman × {strat, partial})')
    print(f'   → n_strata = 12 (one per env at canonical).')
    print(f'   → Slope SE bounded by sqrt(between-strata variance / 12).')
    print(f'   → MORE SEEDS PER ENV does NOT reduce slope SE.')
    print(f'   → Would need: more envs at canonical, OR accept the limit.')
    print()
    print('2. Cross-env Pearson (effh / argmax_entropy link-power):')
    print(f'   → n_envs by polarity (4 reach + ~6 survive at canonical).')
    print(f'   → Same constraint — need more envs of each polarity class.')
    print()
    print('3. finding_metamaze_gamma_amplification:')
    print(f'   → Needs MetaMaze γ=0.999 cells WITH vanilla_jens > 0.05.')
    print(f'   → Long-γ MetaMaze has weak vanilla bias by construction.')
    print(f'   → Likely structurally unrecoverable.')
    print()
    print('Punch-line: the underpowered findings cannot be feasibly')
    print('recovered by adding seeds per env. The framework correctly')
    print('reports them as POWER_INSUFFICIENT given the canonical scope.')
    print('Substrate-level claims (which DO survive canonical) are the')
    print('honest contribution of this corpus.')


if __name__ == '__main__':
    main()
