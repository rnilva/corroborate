"""Phase C smoke — DDQN corpus through the four-way verdict +
meta-regression on env-metadata covariates.

Reads `runs_with_mediators.parquet` (post-migration), partitions
by typed `arm_key`, runs `HypothesisComparisonRow.from_cells`
stratified by `env_name`, and — when the top-level verdict is
HELD_WITH_SCOPE_FLAG — calls `meta_regress_comparison` with an
env-metadata covariate provider to identify cleavage axes.

The smoke replaces v10's `analysis_regime_predictor.py` shape:
where v10 asked "what env metadata predicts the per-env regime
classification?" (a category → category mapping), corroborate
asks "what env metadata predicts the per-stratum effect size?"
— the same scope question, but reframed as inverse-variance-
weighted OLS with explicit CIs, not a discrete classifier.

Safe to run pre-sweep-completion: prints a friendly status and
exits cleanly when the corpus isn't yet migrated."""
from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import cast

import polars as pl

from corroborate.aggregate import hypothesis_comparison_from_cells
from corroborate.hypothesis import Hypothesis
from corroborate.intervention import Intervention
from corroborate.meta_regression import meta_regress_comparison
from corroborate.persistence import read_runrows
from corroborate.rl.dqn.claims.bootstrap import bootstrap, double_greedify
from corroborate.rl.env_catalogue import (
    BenchmarkFamily,
    ENV_REGISTRY,
    EnvSpec,
    RewardRegime,
)
from corroborate.schema import RunRow
from corroborate.verdict import Verdict


# Drop-first reference levels to avoid intercept collinearity.
_FAMILY_REFERENCE: BenchmarkFamily = 'classic_control'
_FAMILIES: tuple[BenchmarkFamily, ...] = (
    'minatar', 'bsuite', 'bandit', 'misc',
)
_REGIME_REFERENCE: RewardRegime = 'per_step'
_REGIMES: tuple[RewardRegime, ...] = (
    'event_triggered', 'shaped', 'terminal_only',
)


def _env_covariates(env_name_obj: object) -> Mapping[str, float]:
    """Map an env's metadata to numeric covariates for meta-
    regression. Drop-first one-hot for `family` and
    `reward_regime` so the design matrix isn't rank-deficient
    against the intercept.

    Argument typed as `object` to match
    `meta_regress_comparison`'s `Callable[[object], ...]` shape
    (substrates use heterogeneous group identities); narrowed
    via `isinstance` to `str` for the env-name lookup."""
    if not isinstance(env_name_obj, str):
        raise TypeError(
            f'_env_covariates expected str env_name, got '
            f'{type(env_name_obj).__name__}',
        )
    spec: EnvSpec = ENV_REGISTRY[env_name_obj]
    out: dict[str, float] = {
        'reward_range': float(spec.r_max - spec.r_min),
        'horizon': float(spec.eval_episode_cap),
        'n_actions': float(spec.n_actions or 0),
    }
    for fam in _FAMILIES:
        out[f'family_{fam}'] = (
            1.0 if spec.benchmark_family == fam else 0.0
        )
    for regime in _REGIMES:
        out[f'regime_{regime}'] = (
            1.0 if spec.reward_regime == regime else 0.0
        )
    return out


def _ddqn_hypothesis() -> Hypothesis[Mapping[str, object]]:
    return Hypothesis(
        name='ddqn',
        intervention={},
        bridges=(),
        predicted_direction='a_gt_b',
        intervention_arms=(
            Intervention(
                slot_path='bootstrap',
                replacement=partial(
                    bootstrap, greedification=double_greedify,
                ),
            ),
        ),
    )


def _vanilla_hypothesis() -> Hypothesis[Mapping[str, object]]:
    return Hypothesis(
        name='vanilla_dqn',
        intervention={},
        bridges=(),
        predicted_direction=None,
        intervention_arms=(),
    )


def _check_corpus_ready(runs_path: Path) -> str | None:
    """Return None when the corpus is ready to analyse; an
    explanatory message string when it isn't."""
    if not runs_path.exists():
        return f'corpus not found: {runs_path} (sweep still running?)'
    df = pl.read_parquet(runs_path)
    if 'arm_key' not in df.columns:
        return (
            f'{runs_path} has no `arm_key` column — run '
            f'`uv run python experiments/migrate_runs_inject_arm_key.py` '
            f'after the sweep finishes'
        )
    keys = set(df['arm_key'].unique().to_list())
    if keys == {'baseline'}:
        return (
            f'{runs_path} has only `arm_key="baseline"` — looks like '
            f'a default-pad migration; re-run the migration script'
        )
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument(
        '--runs-path',
        type=Path,
        default=Path(
            '/workspace/corroborate/experiments/data/ddqn/'
            'runs_with_mediators.parquet'
        ),
        help='Path to the runs parquet to analyse.',
    )
    args = parser.parse_args()
    runs_path = cast(Path, args.runs_path)
    msg = _check_corpus_ready(runs_path)
    if msg is not None:
        print(f'[Phase C] {msg}')
        return

    treatment_h = _ddqn_hypothesis()
    baseline_h = _vanilla_hypothesis()
    treatment_arm_key = treatment_h.arm_key()
    baseline_arm_key = baseline_h.arm_key()

    runs: list[RunRow] = read_runrows(runs_path)
    treatment = [r for r in runs if r.arm_key == treatment_arm_key]
    baseline = [r for r in runs if r.arm_key == baseline_arm_key]

    print(f'[Phase C] read {len(runs)} runs from {runs_path}')
    print(f'  treatment ({treatment_arm_key!r}): {len(treatment)}')
    print(f'  baseline ({baseline_arm_key!r}): {len(baseline)}')

    if not treatment or not baseline:
        print('[Phase C] missing one of the arms — aborting')
        return

    row = hypothesis_comparison_from_cells(
        treatment_h, treatment, baseline,
        outcome_path='outcome.late_window_mean',
        pair_by=('seed',),
        group_by='env_name',
        baseline_h=baseline_h,
    )

    print()
    print(f'[Phase C] verdict: {row.verdict.value}')
    if row.pooled is not None:
        p = row.pooled
        print(
            f'  pooled_g={p.pooled_g:+.3f}  '
            f'I²={p.I2:.3f}  '
            f'PI=[{p.pi_lo:+.3f}, {p.pi_hi:+.3f}]  '
            f'n_strata={p.n_cells}'
        )

    print()
    print(f'[Phase C] per-stratum (n={len(row.per_group)}):')
    print(
        f'  {"env_name":<26} {"n":>3} {"g":>8} {"se":>8} '
        f'{"verdict":<22}'
    )
    for gs in sorted(row.per_group, key=lambda g: cast(str, g.group_value)):
        env = cast(str, gs.group_value)
        g = gs.effect_size_g if gs.effect_size_g is not None else float('nan')
        se = gs.se if gs.se is not None else float('nan')
        print(
            f'  {env:<26} {gs.n_pairs:>3} {g:>+8.3f} {se:>8.3f} '
            f'{gs.verdict.value:<22}'
        )

    if row.verdict is not Verdict.HELD_WITH_SCOPE_FLAG:
        print()
        print(
            f'[Phase C] verdict is {row.verdict.value!r}, not '
            f'HELD_WITH_SCOPE_FLAG — meta-regression skipped. '
            f'Cleavage analysis only fires on heterogeneous '
            f'corroboration.'
        )
        return

    result = meta_regress_comparison(row, _env_covariates, alpha=0.05)
    print()
    print(
        f'[Phase C] meta-regression on {result.n_strata} strata, '
        f'R²={result.r_squared:.3f}, intercept={result.intercept:+.3f}'
    )
    print(
        f'  {"covariate":<28} {"coef":>8} {"ci_lo":>8} '
        f'{"ci_hi":>8} {"p":>6} {"sig":>4}'
    )
    for c in result.coefficients:
        sig = '***' if c.is_significant else ''
        print(
            f'  {c.name:<28} {c.coefficient:>+8.3f} '
            f'{c.ci_lo:>+8.3f} {c.ci_hi:>+8.3f} '
            f'{c.p_value:>6.3f} {sig:>4}'
        )

    if result.cleavage_axes:
        print()
        print(
            f'[Phase C] cleavage axes (significant covariates): '
            f'{list(result.cleavage_axes)!r}'
        )
    else:
        print()
        print(
            f'[Phase C] no significant covariates at alpha=0.05; '
            f'heterogeneity is not explained by the candidate '
            f'covariate set'
        )


if __name__ == '__main__':
    main()
    sys.exit(0)
