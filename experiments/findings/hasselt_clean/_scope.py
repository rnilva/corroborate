"""Scope predicates for the explicit Hasselt-chain hypothesis.

The clean-chain Finding tests an explicit directed walk
`jensen_dormancy_gap → jensen_gap → eval_best_burst_raw_mean`
plus do(DDQN) attacks on both downstream nodes. Scope predicates
named here are the cell-set conditioners each chain edge needs.

Honest scope limitation: `jensen_dormancy_gap` is currently
finite for only 5 of the 10 canonical-pool envs in the cache
(Acrobot, LL, MetaMaze, MC, Snake). The five MinAtar envs
(Asterix / Breakout / Freeway / SI) plus FR have the column
present but all-NaN — collateral damage from a 2026-05-22
ingest that truncated per-corpus `measurements.parquet` for
several corpora (engineering debt; see FUTURE_WORKS.md).

The chain Finding is scoped to envs where `jensen_dormancy_gap`
is finite (`JDG_AVAILABLE_ENVS`), giving an empirical 5-env panel.
Authoring discipline matches `HYPOTHESIS_AS_GRAPH.md`: scope
predicates as module-level constants → bridges share the same
extent → cluster identity is structural."""
from __future__ import annotations

import polars as pl


# Envs with `jensen_dormancy_gap` populated in the canonical
# cache. Updated 2026-05-22 after backfill via per-corpus
# sidecar-invalidation + cloud trace re-restore.
JDG_AVAILABLE_ENVS: tuple[str, ...] = (
    'Acrobot-v1',
    'Asterix-MinAtar',
    'Breakout-MinAtar',
    'FourRooms-misc',
    'Freeway-MinAtar',
    'LunarLander-v2-jax',
    'MetaMaze-misc',
    'MountainCar-v0',
    'Snake-jumanji',
    'SpaceInvaders-MinAtar',
)


# Canonical γ + k=1 + dormancy-availability scope. AND-combined
# into every chain bridge via the module-level `MODULE_SCOPE`.
CANONICAL_DORMANCY_SCOPE: pl.Expr = (
    (pl.col('gamma') == 0.999)
    & (
        pl.col('action_duplicate_k').is_null()
        | (pl.col('action_duplicate_k') == 1)
    )
    & pl.col('env_name').is_in(JDG_AVAILABLE_ENVS)
    & pl.col('jensen_dormancy_gap').is_finite()
    & pl.col('jensen_gap').is_finite()
    & pl.col('eval_best_burst_raw_mean').is_finite()
)


# Vanilla-only filter for the within-arm theorem / link edges
# (B1, B2). The theorem `σ_Q × √(2 log K) ≥ V_jens` is a property
# of the vanilla Q-update; the DDQN arm's clipped Q breaks the
# Jensen-floor by construction.
VANILLA_ONLY: pl.Expr = pl.col('arm_key') == 'baseline'


# Per-cell premise activation: the σ-floor bound is *saturated*
# at the observed bias when `jensen_dormancy_gap == 0`. Cells
# with positive gap have observed bias strictly below the
# Hasselt bound — the bound predicts a positive Jensen-bias that
# isn't empirically present, so the premise is dormant.
PREMISE_ACTIVE_PER_CELL: pl.Expr = pl.col('jensen_dormancy_gap') == 0.0


# Link-active scope: the bias→outcome link is env-conditional;
# `bootstrap_fraction > 0.5` is the load-bearing link-side
# scope feature per `experiments/findings/ddqn_summary.md` CLAIM 4
# (`bootstrap_fraction_drives_g_link__net_of_dormancy`).
LINK_ACTIVE_PER_CELL: pl.Expr = pl.col('bootstrap_fraction') > 0.5


# Per-stratum (corpus-level) premise-activation. Drops corpora
# where premise is *broadly* dormant — i.e., median per-cell
# `jensen_dormancy_gap` exceeds a structural threshold. Avoids
# the per-cell post-treatment selection bias where DDQN's
# intervention itself shifts which cells satisfy `gap == 0`.
#
# Partition is `corpus`, not `env_name`. The cache holds
# multiple corpora per env (e.g. Breakout appears in several
# HP sweeps); `over(['env_name'])` would pool the median across
# all of them, and any corpus with NaN jdg (e.g. older sweeps
# pre-dormancy-backfill) would NaN-propagate the median for
# the entire env. Partitioning by `corpus` gives per-corpus
# medians; the canonical-pool corpus's jdg distribution
# determines its own activation.
PREMISE_ACTIVE_PER_STRATUM: pl.Expr = (
    pl.col('jensen_dormancy_gap').median().over(['corpus']) == 0.0
)


# MODULE_SCOPE for the hasselt_clean hypothesis. Every bridge in
# this package AND-combines its own scope with this. The dormancy
# availability filter is load-bearing — without it, B1's
# partial-Spearman on `jensen_dormancy_gap` would fire NaN on the
# 5 envs missing the measurable.
MODULE_SCOPE: pl.Expr = CANONICAL_DORMANCY_SCOPE
