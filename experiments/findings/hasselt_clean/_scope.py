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


# Canonical k=1 corpora at γ=0.999 — one per env, picked to be
# HP-consistent with the γ=0.99 canonical at the same env so a
# γ-comparison reads cleanly.
#
# HP audit (2026-05-22):
# - Acrobot: gamma_sweep_acrobot (200k, both γ from same corpus)
# - MinAtar: minatar_gamma_sweep_k1/g0999_* (1M; γ=0.99 absent
#   at k=1 — these envs are γ=0.999-only in our panel)
# - FR: gamma_sweep_fourrooms (200k γ=0.999, matches its γ=0.99
#   slice from the same corpus). The 1M `ddqn_vs_vanilla`
#   corpus (loop-hypothesis test) is NOT used here — would
#   confound the γ comparison with 5× training difference.
# - LL: lunarlander_tuned_sync1000_gpu (1M)
# - MetaMaze: metamaze_g0999_1M_postfix (1M)
# - MC: fa_deep_g0999 (1M; γ=0.99 has no jdg)
# - Snake / Freeway / SI: g0999_*-MinAtar and g0999_Snake (1M)
CANONICAL_G0999_CORPORA: tuple[str, ...] = (
    'gamma_sweep_acrobot',
    'minatar_gamma_sweep_k1/g0999_Asterix-MinAtar',
    'minatar_gamma_sweep_k1/g0999_Breakout-MinAtar',
    'gamma_sweep_fourrooms',                 # FR γ=0.999 slice (200k, matches γ=0.99)
    'g0999_Freeway-MinAtar',
    'lunarlander_tuned_sync1000_gpu',        # LL canonical sync=1000
    'metamaze_g0999_1M_postfix',
    'fa_deep_g0999',                         # MC
    'g0999_Snake-jumanji',
    'g0999_SpaceInvaders-MinAtar',
)


# Canonical k=1 corpora at γ=0.99. For each env, the γ=0.99
# corpus is chosen to be HP-consistent with the γ=0.999 entry
# above (same corpus where possible, same total_steps otherwise).
#
# - Acrobot:  gamma_sweep_acrobot γ=0.99 slice (same corpus as
#             γ=0.999, both 200k)
# - FR:       gamma_sweep_fourrooms γ=0.99 slice (same corpus
#             as γ=0.999, both 200k)
# - LL:       g099_panel_extension_lunar_cpu (1M, matches LL γ=0.999 1M)
# - MetaMaze: metamaze_g099_1M_postfix (1M, matches γ=0.999 1M)
#
# The 5 MinAtar envs and MC at γ=0.99 are absent — no k=1 γ=0.99
# canonical sweep exists for them (MinAtar γ=0.99 only at k=2/k=4;
# fa_deep_g099 has no jdg).
CANONICAL_G099_CORPORA: tuple[str, ...] = (
    'gamma_sweep_acrobot',                   # Acrobot γ=0.99 slice
    'gamma_sweep_fourrooms',                 # FR γ=0.99 slice (HP-matched to γ=0.999)
    'g099_panel_extension_lunar_cpu',        # LL γ=0.99
    'metamaze_g099_1M_postfix',              # MetaMaze γ=0.99
)


# Canonical γ + k=1 + dormancy-availability scope. AND-combined
# into every chain bridge via the module-level `MODULE_SCOPE`.
CANONICAL_DORMANCY_SCOPE: pl.Expr = (
    (
        # γ=0.999 canonical corpora (full 10-env panel)
        ((pl.col('gamma') == 0.999) & pl.col('corpus').is_in(CANONICAL_G0999_CORPORA))
        # γ=0.99 canonical corpora (4-env subpanel — only envs with k=1 γ=0.99 sweeps)
        | ((pl.col('gamma') == 0.99) & pl.col('corpus').is_in(CANONICAL_G099_CORPORA))
    )
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


# Per-stratum (corpus + γ) premise-activation. Drops corpus-γ
# slices where premise is *broadly* dormant — i.e., median
# per-cell `jensen_dormancy_gap` exceeds a structural threshold.
# Avoids per-cell post-treatment selection bias.
#
# Partition is (corpus, gamma). Corpora like `gamma_sweep_acrobot`
# hold cells across multiple γ values; pooling the median over
# the whole corpus would mix the γ=0.99 and γ=0.999 regimes.
# Per-(corpus, γ) gives the right slice-level activation test.
PREMISE_ACTIVE_PER_STRATUM: pl.Expr = (
    pl.col('jensen_dormancy_gap').median().over(['corpus', 'gamma']) == 0.0
)


# MODULE_SCOPE for the hasselt_clean hypothesis. Every bridge in
# this package AND-combines its own scope with this. The dormancy
# availability filter is load-bearing — without it, B1's
# partial-Spearman on `jensen_dormancy_gap` would fire NaN on the
# 5 envs missing the measurable.
MODULE_SCOPE: pl.Expr = CANONICAL_DORMANCY_SCOPE
