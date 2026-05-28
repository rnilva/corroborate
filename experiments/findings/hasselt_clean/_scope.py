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
    'CartPole-v1',
    'FourRooms-misc',
    'Freeway-MinAtar',
    'LunarLander-v2-jax',
    'MetaMaze-misc',
    'MountainCar-v0',
    'PacMan-jumanji',
    'Snake-jumanji',
    'SpaceInvaders-MinAtar',
)


# Canonical k=1 corpora at γ=0.999 — one per env, picked to be
# HP-consistent with the γ=0.99 canonical at the same env so a
# γ-comparison reads cleanly.
#
# HP audit (2026-05-22):
# - Acrobot: acrobot_gamma_100k_short (100k, both γ from same
#   fresh sweep — supersedes gamma_sweep_acrobot to get current
#   trace schema with predicted_q_at_start + online_std_q_per_step,
#   so jensen_dormancy_gap computes rather than NaNing out).
# - MinAtar: minatar_gamma_sweep_k1/g0999_* (1M; γ=0.99 absent
#   at k=1 — these envs are γ=0.999-only in our panel)
# - FR: fr_gamma_100k_short (100k γ=0.999, matches its γ=0.99
#   slice from the same fresh sweep — supersedes
#   gamma_sweep_fourrooms for the same trace-schema reason).
# - LL: lunarlander_2M_30seeds_cpu/g0999 (2M)
# - MetaMaze: metamaze_g0999_1M_postfix (1M)
# - MC: fa_deep_g0999 (1M; γ=0.99 has no jdg)
# - Snake / Freeway / SI: g0999_*-MinAtar and g0999_Snake (1M)
CANONICAL_G0999_CORPORA: tuple[str, ...] = (
    'acrobot_gamma_50k_30seeds',             # Acrobot γ=0.999 slice (50k, 30 seeds — outlier-free regime)
    'minatar_gamma_sweep_k1/g0999_Asterix-MinAtar',
    'minatar_gamma_sweep_k1/g0999_Breakout-MinAtar',
    'fr_gamma_50k_30seeds',                  # FR γ=0.999 slice (50k, 30 seeds — pre-ceiling regime)
    'minatar_gamma_sweep_k1/g0999_Freeway-MinAtar',
    'lunarlander_2M_30seeds_cpu/g0999',      # LL γ=0.999 (2M, 30 seeds — supersedes lunarlander_tuned_sync1000_gpu 1M)
    'metamaze_g0999_1M_postfix',
    'g0999_Snake-jumanji',                   # Snake γ=0.999 (sub-corpus of g0999_panel_extension_snake_only, stamps bare)
    'g0999_PacMan-jumanji',                  # PacMan γ=0.999 (sub-corpus of pacman_g0999_n20, stamps bare; n=20 seeds, n_episodes=5)
    'minatar_gamma_sweep_k1/g0999_SpaceInvaders-MinAtar',
)


# Canonical k=1 corpora at γ=0.99. For each env, the γ=0.99
# corpus is chosen to be HP-consistent with the γ=0.999 entry
# above (same corpus where possible, same total_steps otherwise).
#
# - Acrobot:  acrobot_gamma_100k_short γ=0.99 slice (same corpus
#             as γ=0.999, both 100k, fresh sweep)
# - FR:       fr_gamma_100k_short γ=0.99 slice (same corpus
#             as γ=0.999, both 100k, fresh sweep)
# - LL:       lunarlander_2M_30seeds_cpu/g099 (2M, sync=1000 — matches LL γ=0.999 canonical at 2M)
# - MetaMaze: metamaze_g099_1M_postfix (1M, matches γ=0.999 1M)
#
# The 5 MinAtar envs and MC at γ=0.99 are absent — no k=1 γ=0.99
# canonical sweep exists for them (MinAtar γ=0.99 only at k=2/k=4;
# fa_deep_g099 has no jdg).
CANONICAL_G099_CORPORA: tuple[str, ...] = (
    # 6 MLP envs — all on the NEW full-Q canonical_n_eps20_ckpt surface,
    # paired with their seeds15to29 partner to reach n=30 per arm.
    # Replaces the legacy {gamma_50k, _2M, _1M_postfix} entries that
    # carried only the no-ckpt no-full-Q regime.
    'acrobot_g099_canonical_n_eps20_ckpt',
    'acrobot_g099_canonical_n_eps20_ckpt_seeds15to29',
    'cartpole_g099_canonical_n_eps20_ckpt',
    'cartpole_g099_canonical_n_eps20_ckpt_seeds15to29',
    'fr_g099_canonical_n_eps20_ckpt',
    'fr_g099_canonical_n_eps20_ckpt_seeds15to29',
    'lunarlander_g099_canonical_n_eps20_ckpt',
    'lunarlander_g099_canonical_n_eps20_ckpt_seeds15to29/canonical_g099_n_eps20',  # nested (top-level merge skipped on disk pressure)
    'metamaze_g099_canonical_n_eps20_ckpt',
    'metamaze_g099_canonical_n_eps20_ckpt_seeds15to29',
    'mountaincar_g099_canonical_n_eps20_ckpt',
    'mountaincar_g099_canonical_n_eps20_ckpt_seeds15to29',
    # MinAtar / Jumanji envs — partial canonical swap
    'g099_Asterix-MinAtar',                  # MinAtar γ=0.99 v2 (sub-corpus, parent has no runs.parquet)
    'g099_Breakout-MinAtar',
    'g099_Freeway-MinAtar',
    'g099_SpaceInvaders-MinAtar',            # MinAtar γ=0.99 v2 SI sibling — kept as a fallback alongside
                                             # `si_g099_canonical_n_eps1_ckpt` for cache-ingest robustness.
    'si_g099_canonical_n_eps1_ckpt',         # SpaceInvaders γ=0.99 (NEW: full-Q canonical_n_eps1_ckpt, n=30, 1M, sync=1000, supersedes g099_SpaceInvaders-MinAtar)
    'snake_1M',                              # Snake γ=0.99 (n_episodes=3 — noisier than rest)
    'pacman_1M_postfix',                     # PacMan γ=0.99 (n_episodes=3, 10 seeds/arm)
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
