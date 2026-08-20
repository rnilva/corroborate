"""End-to-end verdict assignment over the LG-SCM substrate.

The framework's distinguishing claim from CLAUDE.md acceptance §3.4:

> POWER_INSUFFICIENT is a first-class verdict distinct from HELD
> and NO_EFFECT. Treating an underpowered test as 'no effect'
> smuggles methodological problems past the reader.

The synthetic-input variants (HELD, NO_EFFECT/SIGN_FLIP, two-sided
admission) of `verdict_from_paired_stats` are already covered in
`tests/test_statistics.py:122-172` with hand-picked (g, se, n)
tuples. This file keeps ONE substrate-grounded test that covers
the path the synthetic-input tests can't: the upstream pipeline
where real cells flow through `paired_g.fn`'s sample-statistic
computation, and the resulting (g, se) lands on the
POWER_INSUFFICIENT side of the MDE boundary.

A regression in the upstream Hedges' g computation that inflated
g (e.g., dropped Bessel) could silently push a structurally-
underpowered cell-set into the HELD bucket — the synthetic-input
tests can't catch that because they bypass the upstream stage."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from corroborate.analyses.paired.paired_g import paired_g
from corroborate.bridge.verdict import RefutationClass, Verdict
from corroborate.corpus.schema import RunRow
from corroborate.stats.effect_size import verdict_from_paired_stats
from corroborate.data import cells_to_dataframe

from tests.analytic.lg_scm.composition import LinearGaussianSCM
from tests.analytic.lg_scm.runner import run_paired_arms


def _scm(
    *, mu_x: float, sigma_x: float, beta_xz: float, n_steps: int,
) -> LinearGaussianSCM:
    return LinearGaussianSCM(
        mu_x=mu_x, sigma_x=sigma_x,
        beta_xz=beta_xz, sigma_z=0.1,
        beta_zy=1.5, sigma_y=0.1,
        n_steps=n_steps,
    )


def _as_dicts(rows: Sequence[RunRow]) -> list[Mapping[str, object]]:
    return [r.as_dict() for r in rows]


def test_underpowered_real_cells_resolve_to_power_insufficient_not_no_effect() -> None:
    """LG-SCM with `mu_x ≈ 0.01` produces a TRUE structural Δ of
    magnitude `Δβ_xz · β_zy · μ_x ≈ 0.0075` — small but nonzero.
    Combined with `n_pairs=5` and small `n_steps=10`, the
    closed-form Hedges' g is `mu_x · sqrt(n_steps) / sigma_x · c_4
    ≈ 0.025` — well below MDE at n=5 (~1.4).

    The framework MUST report POWER_INSUFFICIENT/UNDERPOWERED:
    NOT NO_EFFECT/NULL_EFFECT (the structural effect is real;
    the test lacked resolution), and NOT HELD (g << MDE).

    Why a substrate-grounded test instead of feeding (g=0.025,
    se=large, n=5) directly: the implementation verifies the upstream
    Hedges' g computation actually returns ~0.025 from real
    cells. A regression in Bessel correction or c_4 could shift
    the empirical g across the MDE boundary, which the
    synthetic-input verdict tests can't catch."""
    rows = run_paired_arms(
        treatment=_scm(mu_x=0.01, sigma_x=1.0, beta_xz=0.8, n_steps=10),
        baseline=_scm(mu_x=0.01, sigma_x=1.0, beta_xz=0.3, n_steps=10),
        seeds=range(5),
    )
    result = paired_g.fn(
        cells_to_dataframe(_as_dicts(rows)),
        treatment_arm='treatment',
        baseline_arm='baseline',
        pair_by=('seed',),
        source='y_mean',
    )
    # Closed-form: |g| ≈ mu_x * sqrt(n_steps) / sigma_x * c_4(5)
    #            ≈ 0.01 * sqrt(10) / 1.0 * 0.8 ≈ 0.025.
    # Bound at 0.2 absorbs the wide CV of d at n=5 while staying
    # tight enough to catch a regression that pushes |g| above MDE.
    assert abs(result.g) < 0.2, (
        f'paired_g.g = {result.g:.4f}; closed-form ≈ 0.025. A '
        f'value above 0.2 indicates the upstream Hedges computation '
        f'is inflating beyond the structural sample size — would '
        f'mis-classify this cell-set as HELD'
    )
    verdict, refutation, is_powered = verdict_from_paired_stats(
        result.g, result.se, n=result.n_pairs,
        predicted_direction='a_gt_b',
    )
    assert verdict is Verdict.POWER_INSUFFICIENT, (
        f'verdict = {verdict.value!r}; the structural effect is '
        f'real (mu_x · Δβ · β_zy = 0.0075) but at n_pairs=5 the '
        f'sample is below MDE → framework MUST distinguish '
        f'POWER_INSUFFICIENT from NO_EFFECT (CLAUDE.md acceptance '
        f'§3.4). HELD would be a regression toward calling 0.0075 '
        f'a real signal; NO_EFFECT would smuggle the methodological '
        f'problem of sub-MDE samples'
    )
    assert refutation is RefutationClass.UNDERPOWERED
    assert not is_powered
