"""Shared scope predicates and config-aggregation keys.

`MODULE_SCOPE` is read by the runner via `getattr(h, 'MODULE_SCOPE',
None)` and AND-combined into every bridge's scope. Excludes bsuite
envs — diagnostic probes, not chain MDPs.

The config-mean G1 / Q-bounded predicates lift seed-asymmetric per-
cell filters onto config-mean (vanilla-only) via partition_aggregate
on a vanilla-masked column. See the audit story in
`UNCONSUMED_PRIMITIVES_AUDIT.md` and `BRIDGE_AUDIT_TABLE.md`."""
from __future__ import annotations

import polars as pl

from corroborate.bridge.predicates import (
    finite, partition_aggregate,
)


# Hypothesis-module-level scope (AND-combined into every bridge).
MODULE_SCOPE: pl.Expr = ~pl.col('env_name').str.ends_with('-bsuite')


# Config-discriminator keys used by partition_aggregate. Distinguish
# rs-shift / n-step / k-dup variants so their config-means don't
# contaminate the standard-config mean.
DDQN_CONFIG_KEYS: tuple[str, ...] = (
    'env_name', 'sync_period', 'gamma', 'total_steps',
    'n_step', 'reward_scale', 'action_duplicate_k',
)

# Vanilla-masked columns for partition_aggregate.
_VANILLA_JENS_GAP = pl.when(pl.col('arm_key') == 'baseline').then(
    pl.col('jensen_gap'),
).otherwise(None)
_VANILLA_DORMANCY_GAP = pl.when(pl.col('arm_key') == 'baseline').then(
    pl.col('jensen_dormancy_gap'),
).otherwise(None)
_VANILLA_Q_DIVERGENCE_SCORE = pl.when(pl.col('arm_key') == 'baseline').then(
    pl.col('q_divergence_score'),
).otherwise(None)


# G1 (premise active) at config-mean grain: vanilla mean(jens) > 0.05
# AND mean(dormancy) < 0.05. Both arms of an admitted config enter.
G1_VANILLA_CONFIG_PREMISE_ACTIVE = (
    (partition_aggregate(_VANILLA_JENS_GAP, by=DDQN_CONFIG_KEYS, op='mean') > 0.05)
    & (partition_aggregate(_VANILLA_DORMANCY_GAP, by=DDQN_CONFIG_KEYS, op='mean') < 0.05)
)

# Q-bounded regime: vanilla mean(dormancy) < 0.05 AND mean(q_div) < 1.
# Admits cells where Q is well-calibrated and not exploding,
# regardless of jens magnitude.
VANILLA_CONFIG_Q_BOUNDED = (
    (partition_aggregate(_VANILLA_DORMANCY_GAP, by=DDQN_CONFIG_KEYS, op='mean') < 0.05)
    & (partition_aggregate(_VANILLA_Q_DIVERGENCE_SCORE, by=DDQN_CONFIG_KEYS, op='mean') < 1.0)
)

# DDQN-relevant scope (G1 + G2 + standard-config gating).
DDQN_RELEVANT_SCOPE = (
    G1_VANILLA_CONFIG_PREMISE_ACTIVE
    # G2 — argmax bias-vulnerable. Heuristic `n_actions >= 3`.
    & finite('n_actions')
    & (pl.col('n_actions') >= 3)
    # Standard config (no n-step / action-duplicate / rs-shift /
    # polyak-τ).
    & ((pl.col('n_step') == 1) | pl.col('n_step').is_null())
    & pl.col('action_duplicate_k').is_null()
    & (pl.col('reward_scale').is_null() | (pl.col('reward_scale') == 1.0))
    & pl.col('target_sync.tau').is_null()
)


# REACH-polarity 4-env cohort used by the CLAIM 22 DoWhy trio.
REACH_ENVS_FOUR: tuple[str, ...] = (
    'FourRooms-misc',
    'Acrobot-v1',
    'MountainCar-v0',
    'MetaMaze-misc',
)
