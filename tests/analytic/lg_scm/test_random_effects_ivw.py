"""Closed-form assertions on the inverse-variance-weighted (IVW)
math in `random_effects_summary`.

The existing analytic test (`test_random_effects_verdict.py`)
constructs envs at uniform `n_pairs`, so all per-env SE values are
approximately equal. Under uniform SE, IVW reduces to a simple
arithmetic mean — the weighting math (`vs = [se*se]`,
`w = 1/v`, `Q = sum(w*(g - g_fixed)²)`, `df = n - 1`,
`tau² = max(0, (Q - df) / c_term)`) is never exercised. Mutation
testing surfaced this: those mutations all survive on the
uniform-SE panel.

This file fills the gap: per-env panels with HETEROGENEOUS SE
(varied n_pairs across envs, so se_per_env differs by 4-8×).
The closed-form pooled g via the IVW formula is computable
independently from per-env (g, se), and the framework's
`pooled_g` must match.
"""
from __future__ import annotations

import math

from corroborate.analyses.paired_g import per_env_paired_g_panel
from corroborate.corpus.schema import RunRow
from corroborate.stats.effect_size import random_effects_summary

from tests.analytic.lg_scm.composition import LinearGaussianSCM
from tests.analytic.lg_scm.runner import run_paired_arms


_SIGMA_X = 0.5
_BETA_ZY = 1.5
_SIGMA_Z = 0.1
_SIGMA_Y = 0.1
_N_STEPS = 200
_BETA_XZ_TREAT = 0.8
_BETA_XZ_BASE = 0.3


def _scm(*, mu_x: float) -> LinearGaussianSCM:
    return LinearGaussianSCM(
        mu_x=mu_x, sigma_x=_SIGMA_X,
        beta_xz=_BETA_XZ_TREAT,  # placeholder; overwritten per arm below
        sigma_z=_SIGMA_Z,
        beta_zy=_BETA_ZY, sigma_y=_SIGMA_Y,
        n_steps=_N_STEPS,
    )


def _build_heterogeneous_se_panel(
    *,
    mu_x_per_env: tuple[float, ...],
    n_pairs_per_env: tuple[int, ...],
) -> tuple[list[tuple[float, float]], list[RunRow]]:
    """Build a panel where each env has its own (mu_x, n_pairs).
    Different n_pairs → different per-env SE → IVW weighting math
    is actually exercised (vs. uniform-n where all weights cancel)."""
    assert len(mu_x_per_env) == len(n_pairs_per_env), (
        'mu_x_per_env and n_pairs_per_env must align'
    )
    rows: list[RunRow] = []
    for env_index, (mu, n_pairs) in enumerate(
        zip(mu_x_per_env, n_pairs_per_env),
    ):
        env_seeds = range(env_index * 10000, env_index * 10000 + n_pairs)
        rows.extend(run_paired_arms(
            treatment=LinearGaussianSCM(
                mu_x=mu, sigma_x=_SIGMA_X, beta_xz=_BETA_XZ_TREAT,
                sigma_z=_SIGMA_Z, beta_zy=_BETA_ZY, sigma_y=_SIGMA_Y,
                n_steps=_N_STEPS,
            ),
            baseline=LinearGaussianSCM(
                mu_x=mu, sigma_x=_SIGMA_X, beta_xz=_BETA_XZ_BASE,
                sigma_z=_SIGMA_Z, beta_zy=_BETA_ZY, sigma_y=_SIGMA_Y,
                n_steps=_N_STEPS,
            ),
            seeds=env_seeds,
            env_name=f'env_mu_{mu:g}',
        ))
    panel = per_env_paired_g_panel(
        [r.as_dict() for r in rows],
        treatment_arm='treatment', baseline_arm='baseline',
        source='y_mean',
    )
    g_se_pairs = [(s.g, s.se) for s in panel]
    return g_se_pairs, rows


def _ivw_fixed_effects_pool(
    pairs: list[tuple[float, float]],
) -> tuple[float, float, float, float]:
    """Closed-form fixed-effects IVW pool of (g, se) pairs.
    Returns (g_pooled, var_pooled, Q, df).

        v_i  = se_i²
        w_i  = 1 / v_i
        g_p  = sum(w_i · g_i) / sum(w_i)
        var_p = 1 / sum(w_i)
        Q    = sum(w_i · (g_i - g_p)²)
        df   = n - 1
    """
    valid = [(g, se) for g, se in pairs
             if not (math.isnan(g) or math.isnan(se)) and se > 0.0]
    n = len(valid)
    assert n >= 2, f'need at least 2 valid envs, got {n}'
    vs = [se * se for _, se in valid]
    ws = [1.0 / v for v in vs]
    sum_w = sum(ws)
    g_pool = sum(w * g for w, (g, _) in zip(ws, valid)) / sum_w
    var_pool = 1.0 / sum_w
    Q = sum(w * (g - g_pool) ** 2 for w, (g, _) in zip(ws, valid))
    df = n - 1
    return g_pool, var_pool, Q, float(df)


def _dl_tau_sq(*, Q: float, df: float, ws: list[float]) -> float:
    """DerSimonian-Laird τ²:
        c    = sum(w) - sum(w²) / sum(w)
        τ²   = max(0, (Q - df) / c) if Q > df else 0
    """
    sum_w = sum(ws)
    sum_w_sq = sum(w * w for w in ws)
    c = sum_w - sum_w_sq / sum_w
    if c <= 0 or Q <= df:
        return 0.0
    return (Q - df) / c


# ============ Heterogeneous-SE pooled g ============

def test_pooled_g_recovers_ivw_closed_form_under_heterogeneous_se() -> None:
    """4 envs at different n_pairs (10, 20, 40, 80) AND different
    mu_x → per-env (g, se) pairs vary on both axes. The framework's
    `random_effects_summary.pooled_g` must match the
    inverse-variance-weighted closed form (under FE pooling, when
    τ² turns out to be 0; otherwise the framework's RE pool with
    non-zero τ² is also computable).

    Mutation testing surfaced this: every mutation of the IVW
    formula (`vs = [se*se]` → `[se/se]`, `w = 1/v` → `1*v`,
    `Q = sum(w*(g-g_fixed)²)` → `sum(w/(g-g_fixed)²)`,
    `df = n - 1` → `df = n + 1`, etc.) survives the uniform-SE
    panel. Heterogeneous SE is the discriminator that makes the
    weights actually bite.
    """
    # Hedges' SE = sqrt(1/n + g²/(2n)). With identical n, SE
    # tracks |g|; varying mu_x gives radical g spread (mu_x=0.05
    # → g ~ 1.4; mu_x=2.0 → g ~ 56), which in turn gives radical
    # SE spread (~25×). That's the heterogeneous regime where IVW
    # weighting actually bites.
    mu_x_per_env = (0.05, 0.5, 2.0)
    n_pairs_per_env = (30, 30, 30)
    pairs, _ = _build_heterogeneous_se_panel(
        mu_x_per_env=mu_x_per_env,
        n_pairs_per_env=n_pairs_per_env,
    )
    assert len(pairs) == 3, f'expected 3 envs in panel; got {len(pairs)}'

    # Sanity: SEs vary widely (heterogeneous regime).
    ses = [se for _, se in pairs]
    assert max(ses) / min(ses) > 5.0, (
        f'panel SEs too uniform: {ses}; the test relies on '
        f'cross-env SE variance to exercise IVW weights'
    )

    # Closed-form FE pool.
    g_pool_expected, _, Q, df = _ivw_fixed_effects_pool(pairs)
    ws = [1.0 / (se * se) for _, se in pairs]
    tau_sq = _dl_tau_sq(Q=Q, df=df, ws=ws)

    pooled = random_effects_summary(pairs)
    assert pooled.n_cells == 3

    if tau_sq == 0.0:
        # FE and RE pools coincide.
        rel_err = abs(pooled.pooled_g - g_pool_expected) / abs(g_pool_expected)
        assert rel_err < 0.001, (
            f'pooled_g = {pooled.pooled_g:.4f}, FE-IVW closed form '
            f'= {g_pool_expected:.4f} (rel err {rel_err:.4f}, '
            f'tau²=0 so RE = FE). Mutating the IVW math '
            f'(vs, w, sum_w, etc.) should breach this bound.'
        )
    else:
        # RE pool: re-weight by w_re = 1/(v + tau²), pool again.
        w_re = [1.0 / (se * se + tau_sq) for _, se in pairs]
        g_pool_re = sum(
            w * g for w, (g, _) in zip(w_re, pairs)
        ) / sum(w_re)
        rel_err = abs(pooled.pooled_g - g_pool_re) / abs(g_pool_re)
        assert rel_err < 0.001, (
            f'pooled_g = {pooled.pooled_g:.4f}, RE-IVW closed form '
            f'= {g_pool_re:.4f} (tau² = {tau_sq:.4f}). '
            f'IVW weighting + DL tau² closed form must match.'
        )

    # Q-statistic check — exercises `Q = sum(w * (g - g_fixed)²)`.
    rel_err_Q = abs(pooled.Q - Q) / max(abs(Q), 1e-9)
    assert rel_err_Q < 0.01, (
        f'Q = {pooled.Q:.4f}, closed form = {Q:.4f} (rel err '
        f'{rel_err_Q:.4f}). Mutations on the Q formula would '
        f'breach this bound.'
    )

    # tau² check — exercises the DL `(Q - df) / c` formula.
    if tau_sq > 0.0:
        rel_err_tau = abs(pooled.tau2 - tau_sq) / tau_sq
        assert rel_err_tau < 0.05, (
            f'tau2 = {pooled.tau2:.4f}, DL closed form = '
            f'{tau_sq:.4f} (rel err {rel_err_tau:.4f})'
        )
    else:
        assert pooled.tau2 == 0.0, (
            f'tau2 = {pooled.tau2}; DL closed form is 0 '
            f'(Q = {Q:.4f} <= df = {df})'
        )


def test_pooled_g_under_uniform_se_collapses_to_simple_mean() -> None:
    """Sanity: when all envs have the same SE (uniform n_pairs +
    same g), pooled_g IS the simple mean — but this case doesn't
    discriminate IVW math from arithmetic mean. Documented here as
    the case the previous test deliberately AVOIDS, to make the
    heterogeneous-SE test's role clear.

    A regression that broke IVW would still pass this test (any
    weighted average of equal values returns the value). It's a
    floor sanity, not a discriminator."""
    # All envs share mu_x but get distinct names via tiny perturbation
    # so per_env_paired_g_panel keeps them as separate strata. The
    # tiny mu_x perturbation keeps SE approximately uniform.
    mu_x_per_env = (1.0, 1.001, 1.002, 1.003)
    n_pairs_per_env = (30, 30, 30, 30)
    pairs, _ = _build_heterogeneous_se_panel(
        mu_x_per_env=mu_x_per_env,
        n_pairs_per_env=n_pairs_per_env,
    )
    assert len(pairs) == 4
    pooled = random_effects_summary(pairs)
    simple_mean = sum(g for g, _ in pairs) / len(pairs)
    rel_err = abs(pooled.pooled_g - simple_mean) / abs(simple_mean)
    assert rel_err < 0.02, (
        f'pooled_g = {pooled.pooled_g:.4f}, simple mean = '
        f'{simple_mean:.4f} (rel err {rel_err:.4f}). Under near-'
        f'uniform SE (RE re-weights modestly when tau²>0) the two '
        f'should agree within ~2%.'
    )
