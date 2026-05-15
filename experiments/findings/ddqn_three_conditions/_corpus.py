"""Corpus joiner for the three-conditions hypothesis.

The hypothesis claims test conditions that span multiple sub-
corpora at different HP regimes. The joiner produces a unified
parquet at `experiments/data/cache/ddqn_three_conditions.parquet`
with three extra per-cell columns that the condition bridges
stratify by:

| column | values | source |
|---|---|---|
| `shaping_kind` | `none` / `potential_manhattan` | wrappers column |
| `fa_kind` | `mlp_deep` / `linear` | `q_network.hidden` |
| `k_eff` | int (= native_actions × action_duplicate_k) | wrappers × env catalogue |

Sub-corpora pulled (all γ=0.99 or 0.999, MLP[64,64] / linear FA,
1M total_steps with caveats):

1. **FR k-sweep at γ=0.999**:
   `experiments/probes/action_dup_mismatch_probe_g999_1M/...`
   + `experiments/probes/action_dup_mismatch_probe_g999_1M_FR_k4_only/...`
   → tests **Condition 1** (Q-bias exists and scales with K)
2. **FR with PotentialReward shaping**:
   `experiments/data/fa_degeneracy_shaped_only/...`
   → tests **Condition 3** (policy-signal-strength)
3. **MC linear vs deep FA at γ=0.999**:
   `experiments/probes/ddqn_axis_probes_mc_1m/fa_linear_g0999/...`
   `experiments/probes/ddqn_axis_probes_mc_1m/fa_deep_g0999/...`
   → tests **Condition 2** (FA-capacity gates Type 1)
4. **Acrobot baseline (control)**:
   `experiments/probes/action_dup_mismatch_probe_g999_1M/...` (Acrobot k=1)
   → control: T2-dominated env (DDQN dormant)

The joiner is idempotent and skips already-merged columns. Re-run
via `python -m experiments.findings.ddqn_three_conditions._corpus`.

Run as a module so the registry imports populate (CLAIM dqn,
measurables, analyses)."""
from __future__ import annotations

from pathlib import Path

import polars as pl


# Output cache. Hypothesis runner reads from here.
CACHE_PATH = Path('experiments/data/cache/ddqn_three_conditions.parquet')

# Source corpora (relative to repo root).
_SOURCES: tuple[tuple[str, Path], ...] = (
    (
        'fr_k_sweep_main',
        Path('experiments/probes/action_dup_mismatch_probe_g999_1M/'
             'ddqn_vs_vanilla/runs.parquet'),
    ),
    (
        'fr_k4_completion',
        Path('experiments/probes/action_dup_mismatch_probe_g999_1M_FR_k4_only/'
             'runs.parquet'),
    ),
    (
        'fr_shaped',
        Path('experiments/data/fa_degeneracy_shaped_only/runs.parquet'),
    ),
    (
        'mc_axis_linear_g0999',
        Path('experiments/probes/ddqn_axis_probes_mc_1m/'
             'fa_linear_g0999/runs.parquet'),
    ),
    (
        'mc_axis_deep_g0999',
        Path('experiments/probes/ddqn_axis_probes_mc_1m/'
             'fa_deep_g0999/runs.parquet'),
    ),
)


def build() -> pl.DataFrame:
    """Join all source corpora and add the per-cell stratification
    columns. Returns the merged dataframe."""
    parts: list[pl.DataFrame] = []
    for label, path in _SOURCES:
        if not path.exists():
            raise FileNotFoundError(
                f'three_conditions corpus joiner: missing source {label} '
                f'at {path}',
            )
        parts.append(pl.read_parquet(path).with_columns(
            source_label=pl.lit(label),
        ))

    df = pl.concat(parts, how='diagonal_relaxed')

    # === Add stratification columns ===
    # `shaping_kind`: identify potential-based shaping via wrappers.
    df = df.with_columns(
        shaping_kind=pl.when(
            pl.col('wrappers').str.contains('PotentialReward')
        ).then(pl.lit('potential_manhattan')).otherwise(pl.lit('none')),
    )
    # `fa_kind`: linear when hidden tuple is empty `()`; deep otherwise.
    df = df.with_columns(
        fa_kind=pl.when(pl.col('q_network.hidden') == '()')
            .then(pl.lit('linear'))
            .otherwise(pl.lit('mlp_deep')),
    )
    # `k_eff`: native_actions × action_duplicate_k. Native actions per env.
    native_actions: dict[str, int] = {
        'FourRooms-misc': 4,
        'Acrobot-v1': 3,
        'MountainCar-v0': 3,
        'MetaMaze-misc': 4,
    }
    df = df.with_columns(
        action_duplicate_k_eff=pl.col('action_duplicate_k').fill_null(1.0),
    )
    df = df.with_columns(
        k_eff=pl.col('env_name')
            .replace_strict(native_actions, default=None)
            .cast(pl.Int64) * pl.col('action_duplicate_k_eff').cast(pl.Int64),
    )

    return df


def write_cache() -> None:
    """Build + persist to `CACHE_PATH`. Idempotent."""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = build()
    df.write_parquet(CACHE_PATH)
    print(f'wrote {df.shape} → {CACHE_PATH}')


if __name__ == '__main__':
    write_cache()
