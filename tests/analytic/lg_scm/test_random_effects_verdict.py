"""End-to-end random-effects verdict over a multi-env LG-SCM panel.

The decision-tree branches of `random_effects_verdict` (HELD /
SIGN_FLIP / NULL_EFFECT / POWER_INSUFFICIENT / two-sided) are
already covered by synthetic-input tests in `tests/test_statistics.py`.
This file keeps ONE substrate-grounded test that exercises what
the synthetic-input tests can't: the full pipeline

    multi-env LG-SCM cells
      → per_env_paired_g_panel  (substrate-driven panel build)
      → random_effects_summary  (DerSimonian-Laird pool)
      → random_effects_verdict  (Popperian verdict)

A regression in the panel-builder (e.g., dropping envs with
NaN g, mis-keying strata, mishandling `n_pairs < 2`) could
silently pass the synthetic-input verdict test and only fail
when real cells flow through. This is the load-bearing
substrate-only contribution.

The test also asserts the pooled g recovers the closed-form
average per-env g — a tighter check than just verdict-class
membership, since a panel-builder bug that drops some envs
would shift the pooled average measurably."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from corroborate.analyses.paired.paired_g import per_env_paired_g_panel
from corroborate.bridge.verdict import Verdict
from corroborate.corpus.schema import RunRow
from corroborate.stats.effect_size import (
    random_effects_summary,
    random_effects_verdict,
)

from tests.analytic.lg_scm.composition import LinearGaussianSCM
from tests.analytic.lg_scm.runner import run_multi_env_paired_arms


_SIGMA_X = 0.5
_BETA_ZY = 1.5
_SIGMA_Z = 0.1
_SIGMA_Y = 0.1
_N_STEPS = 200
_N_PAIRS = 30

_BETA_XZ_TREAT = 0.8
_BETA_XZ_BASE = 0.3

# Tight grid: per-env g varies as `mu_x * sqrt(n_steps) / sigma_x * c_4`,
# so a narrow mu_x range keeps I² below the scope-flag threshold
# and PI excludes zero on the predicted direction. Wider grids
# inflate τ² to the point where PI brackets zero — correct
# framework behavior, but not what the corroboration branch tests.
_TIGHT_MU_X = (0.95, 1.0, 1.05, 1.1, 1.15)


def _scm(*, mu_x: float, beta_xz: float) -> LinearGaussianSCM:
    return LinearGaussianSCM(
        mu_x=mu_x, sigma_x=_SIGMA_X,
        beta_xz=beta_xz, sigma_z=_SIGMA_Z,
        beta_zy=_BETA_ZY, sigma_y=_SIGMA_Y,
        n_steps=_N_STEPS,
    )


def test_corroboration_pipeline_recovers_closed_form_pooled_g() -> None:
    """Real LG-SCM cells across 5 envs feed into
    `per_env_paired_g_panel`, then `random_effects_summary`, then
    `random_effects_verdict`. The full chain should:

    1. Build a panel of 5 valid `(g, se)` pairs (no envs dropped).
    2. Pool to a `pooled_g` near the closed-form average:
       `(mu_x_avg) * sqrt(n_steps) / sigma_x * c_4(n_pairs)` ≈ 28.9.
    3. Have PI excluding zero positively (low I² on tight grid).
    4. Resolve to HELD or HELD_WITH_SCOPE_FLAG with no refutation.

    A panel-builder regression that dropped envs, mis-keyed strata,
    or returned NaN g/se for valid cells would shift `pooled_g`
    measurably away from the closed form OR change `n_cells` —
    failing the test in a implementation-specific way the synthetic-
    input verdict tests can't catch."""
    envs: dict[str, tuple[LinearGaussianSCM, LinearGaussianSCM]] = {
        f'env_mu_{mu:g}': (
            _scm(mu_x=mu, beta_xz=_BETA_XZ_TREAT),
            _scm(mu_x=mu, beta_xz=_BETA_XZ_BASE),
        )
        for mu in _TIGHT_MU_X
    }
    rows: list[RunRow] = run_multi_env_paired_arms(
        envs=envs, seeds=range(_N_PAIRS),
    )
    cells: Sequence[Mapping[str, object]] = [r.as_dict() for r in rows]
    panel = per_env_paired_g_panel(
        cells, treatment_arm='treatment', baseline_arm='baseline',
        source='y_mean',
    )

    # Panel completeness — every env contributes; no silent drop.
    assert len(panel) == len(_TIGHT_MU_X), (
        f'panel has {len(panel)} envs, expected '
        f'{len(_TIGHT_MU_X)}; panel-builder dropped envs silently'
    )
    valid_g_se = [(s.g, s.se) for s in panel
                  if not math.isnan(s.g) and not math.isnan(s.se)]
    assert len(valid_g_se) == len(_TIGHT_MU_X), (
        f'panel produced NaN g/se on {len(_TIGHT_MU_X) - len(valid_g_se)} '
        f'envs — should not happen on closed-form-tractable cells'
    )

    pooled = random_effects_summary(valid_g_se)
    assert pooled.n_cells == len(_TIGHT_MU_X)

    # Closed-form pooled g: under shared-noise per-env g is
    # mu_x * sqrt(n_steps) / sigma_x * c_4. Pooled is the
    # inverse-variance-weighted average; with all envs at similar
    # SE, this is ≈ mean of per-env g.
    c4 = 1.0 - 3.0 / (4 * _N_PAIRS - 5)
    expected_per_env_g = [
        mu * math.sqrt(_N_STEPS) / _SIGMA_X * c4
        for mu in _TIGHT_MU_X
    ]
    expected_pooled = sum(expected_per_env_g) / len(expected_per_env_g)
    rel_err = abs(pooled.pooled_g - expected_pooled) / expected_pooled
    assert rel_err < 0.1, (
        f'pooled_g = {pooled.pooled_g:.4f}, closed-form average '
        f'= {expected_pooled:.4f} (rel err {rel_err:.4f}). The '
        f'pooled estimate should track the average of per-env g; '
        f'a >10% drift indicates panel-build or DL-pool bug.'
    )

    # PI must clear zero positively under the tight-grid construction.
    assert pooled.pi_lo > 0.0, (
        f'pi_lo = {pooled.pi_lo:.4f}; the tight mu_x grid keeps I² '
        f'low enough that PI excludes zero on a uniformly-positive '
        f'panel. PI bracketing zero indicates τ² inflation or a '
        f'panel-builder bug producing spurious heterogeneity'
    )

    # Final verdict: HELD or HELD_WITH_SCOPE_FLAG, no refutation.
    verdict, refutation = random_effects_verdict(
        pooled, predicted_direction='a_gt_b',
    )
    assert verdict in (Verdict.HELD, Verdict.HELD_WITH_SCOPE_FLAG), (
        f'verdict = {verdict.value!r}; expected a corroboration '
        f'verdict on a uniformly-positive multi-env panel'
    )
    assert refutation is None
