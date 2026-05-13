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


# Canonical HP regime — the default training configuration that
# every "is DDQN substantively different from vanilla" bridge
# should operate under. Pins everything that's an HP-sweep axis
# elsewhere in the corpus, so cross-cell variation within this
# scope is dominated by SEED variance + within-config dynamics,
# not by HP/network/wrapper differences (the cross-config common-
# cause confound surfaced in `findings_two_channel_cross_corpus.md`).
#
# Per-env canonical config. 1M total_steps for ALL envs (state-based
# envs at 200k are undertrained — MountainCar stuck at -86.6 median).
# Tuples are (env_name, sync_period, replay_capacity, hidden,
# channels-or-None). total_steps is fixed at 1M.
_PER_ENV_CANONICAL: tuple[tuple[str, int, int, str, str | None], ...] = (
    # MLP-state at 1M
    ('Acrobot-v1',                100,  50000,  '(64,64)', None),
    ('CartPole-v1',               100,  50000,  '(64,64)', None),
    ('FourRooms-misc',            100,  50000,  '(64,64)', None),
    ('MountainCar-v0',            100,  50000,  '(64,64)', None),
    ('MetaMaze-misc',             100,  50000,  '(64,64)', None),
    # MinAtar paper-canonical
    ('Asterix-MinAtar',           1000, 100000, '(128)',  '(16)'),
    ('Breakout-MinAtar',          1000, 100000, '(128)',  '(16)'),
    ('Freeway-MinAtar',           1000, 100000, '(128)',  '(16)'),
    ('SpaceInvaders-MinAtar',     1000, 100000, '(128)',  '(16)'),
    # jumanji games
    ('PacMan-jumanji',            1000, 50000,  '(64)',   '(8,16)'),
    ('SlidingTilePuzzle-jumanji', 1000, 50000,  '(64)',   '(8,16)'),
    ('Snake-jumanji',             1000, 50000,  '(64)',   '(8,16)'),
)


def _build_per_env_canonical_filter() -> pl.Expr:
    """Disjunction of per-env canonical filters (1M total_steps for all)."""
    parts: list[pl.Expr] = []
    for env, sync, capacity, hidden, channels in _PER_ENV_CANONICAL:
        cond = (
            (pl.col('env_name') == env)
            & (pl.col('sync_period') == sync)
            & (pl.col('total_steps') == 1000000)
            & (pl.col('replay.capacity') == capacity)
            & (pl.col('q_network.hidden') == hidden)
        )
        if channels is None:
            cond = cond & pl.col('q_network.channels').is_null()
        else:
            cond = cond & (pl.col('q_network.channels') == channels)
        parts.append(cond)
    out = parts[0]
    for p in parts[1:]:
        out = out | p
    return out


DDQN_CANONICAL_REGIME: pl.Expr = (
    (pl.col('gamma') == 0.99)
    & (pl.col('optimizer.inner.lr') == 0.0001)
    & ((pl.col('n_step') == 1) | pl.col('n_step').is_null())
    & pl.col('action_duplicate_k').is_null()
    & (pl.col('reward_scale').is_null() | (pl.col('reward_scale') == 1.0))
    & pl.col('target_sync.tau').is_null()
    & (pl.col('wrappers') == '()')
    & _build_per_env_canonical_filter()
)


# Hypothesis-module-level scope (AND-combined into every bridge).
# `~bsuite`: bsuite diagnostic envs are excluded — they're not
# chain MDPs.
# `& DDQN_CANONICAL_REGIME`: every bridge in THIS hypothesis module
# operates under canonical HPs. Bridges that intentionally vary
# HPs (n-step, action-duplicate, γ-sweep, lr/capacity, Polyak-τ,
# reward-scale, wrappers) live in `experiments.findings.ddqn_sweeps`
# — a sibling hypothesis module with a relaxed scope.
MODULE_SCOPE: pl.Expr = (
    ~pl.col('env_name').str.ends_with('-bsuite')
    & DDQN_CANONICAL_REGIME
)


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


# Noise floor for "vanilla mean jens is meaningfully positive."
# Substrate convention (not physics-derived): per-cell vanilla
# jens has a small near-zero distribution from sampling noise +
# numerical drift; we require config-mean to exceed this floor
# before counting the stratum as G1-active. Threshold 0.05 is a
# conservative "1/20 unit reward" floor — well below all envs'
# typical mean jens (Acrobot ≈ 1.91, MC ≈ 16.6, FR-200k ≈ 0.28),
# trims the FR-1M saturated-vanilla case (mean jens ≈ 0.043).
# Used by:
#   - `G1_VANILLA_CONFIG_PREMISE_ACTIVE` (scope predicate)
#   - `stratified_arm_diff_pooled.min_vanilla_predictor` (analysis param)
#   - `stratum_*_dowhy.min_vanilla_predictor` (analysis param)
# All sites apply `vanilla_mean_predictor > VANILLA_JENS_NOISE_FLOOR`
# (admit) / `<= floor` (skip). Substrate-physical calibration to
# σ_Q × √(2 log K) per `findings_three_gate_empirical_taxonomy.md`
# would require per-env thresholds; deferred — the noise-floor
# framing is the current honest stand-in.
VANILLA_JENS_NOISE_FLOOR: float = 0.05


# G1 (premise active) at config-mean grain: vanilla mean(jens) >
# noise floor AND mean(dormancy) < noise floor. Both arms of an
# admitted config enter.
G1_VANILLA_CONFIG_PREMISE_ACTIVE = (
    (partition_aggregate(_VANILLA_JENS_GAP, by=DDQN_CONFIG_KEYS, op='mean') > VANILLA_JENS_NOISE_FLOOR)
    & (partition_aggregate(_VANILLA_DORMANCY_GAP, by=DDQN_CONFIG_KEYS, op='mean') < VANILLA_JENS_NOISE_FLOOR)
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


# Cross-env link-power cohorts (CLAIM 19/20). Per-cell
# `env_reward_polarity` admits the mixed-polarity envs Snake /
# MetaMaze / SI-MinAtar at their negative-polarity subset — bridge
# names like `__reach_envs` lie about what's admitted. These tuples
# pin env-level polarity (every cell of the env satisfies the
# bucket): pol_max < -0.3 for REACH, pol_min > +0.3 for SURVIVE.
LINK_POWER_REACH_ENVS: tuple[str, ...] = (
    'Acrobot-v1',
    'FourRooms-misc',
    'MountainCar-v0',
)


LINK_POWER_SURVIVE_ENVS: tuple[str, ...] = (
    'Asterix-MinAtar',
    'Breakout-MinAtar',
    'CartPole-v1',
)
