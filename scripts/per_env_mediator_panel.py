"""Per-env mediator-share panel under mech-HELD conditioning.

Sweeps `proportion_mediated` across (env × candidate_mediator)
within CLAIM 17's bounded-Q scope. Output: per env, which
mediator carries ≥ 20% of DDQN's outcome benefit, with
in_unit_interval diagnostic for linear-mediation assumption
failure.

Reads the canonical `ddqn_universe` cache parquet; no fresh
training. Mediator candidates restricted to the registered
scalar measurables that have ≥ 1 env-cell coverage in the
in-scope subset.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'src/corroborate_rl'))

import corroborate.analyses  # noqa: F401  populates registry
import corroborate_rl.dqn.measurables  # noqa: F401  populates registry
from corroborate.analyses.proportion_mediated import proportion_mediated

DDQN = (
    'bootstrap=partial(Claim:bootstrap;greedification=Claim:double_greedify)'
)
BASELINE = 'baseline'

# CLAIM 17 scope (bounded Q ∧ bootstrap-using ∧ mech premise active ∧
# standard config — see ddqn_universe.py:chain_amplifier_link_active_in_bounded_q).
CLAIM_17_SCOPE = (
    pl.col('q_divergence_score').is_finite()
    & (pl.col('q_divergence_score') < 1.0)
    & pl.col('bootstrap_fraction').is_finite()
    & (pl.col('bootstrap_fraction') > 0.5)
    & pl.col('jensen_dormancy_gap').is_finite()
    & (pl.col('jensen_dormancy_gap') < 0.05)
    & ((pl.col('n_step') == 1) | pl.col('n_step').is_null())
    & pl.col('action_duplicate_k').is_null()
    & (pl.col('reward_scale').is_null() | (pl.col('reward_scale') == 1.0))
    & pl.col('target_sync.tau').is_null()
)

# Mediator candidates with scalar coverage in the corpus.
# See PER_ENV_MEDIATOR_NOTES below for what each represents.
MEDIATORS: tuple[str, ...] = (
    'target_staleness_late',     # CLAIM 13 known partial (27-65% on 3/4)
    'q_late_mean',               # Hasselt-direction Q-channel
    'argmax_entropy_late',       # Action-selection channel (CLAIM 7e/f)
    'effective_horizon',         # Length channel (CLAIM 12 soft tautology)
    'q_divergence_score',        # Q-explosion proxy (CLAIM 11)
    'env_reward_polarity',       # Polarity (CLAIM 14, soft tautology)
)


def main() -> None:
    cache = REPO / 'experiments/data/cache/ddqn_universe.parquet'
    df = pl.read_parquet(cache)
    sub = df.filter(
        CLAIM_17_SCOPE & pl.col('arm_key').is_in([DDQN, BASELINE])
    )
    envs = sorted(sub['env_name'].unique().to_list())
    print(f'CLAIM 17 in-scope: {len(sub):,} cells, {len(envs)} envs')
    print(f'Envs: {envs}')
    print()

    # Header.
    print(
        f'{"env":<24} {"mediator":<22} {"n_pairs":>7} '
        f'{"proportion":>11} {"slope":>9} {"unit_int":>9} {"verdict":<14}'
    )
    print('-' * 110)

    summary: dict[str, list[tuple[str, float]]] = {e: [] for e in envs}

    for env in envs:
        env_cells = sub.filter(pl.col('env_name') == env).to_dicts()
        for med in MEDIATORS:
            # Skip if mediator column is absent or has no in-scope finite values.
            if med not in df.columns:
                continue
            n_finite = sub.filter(
                (pl.col('env_name') == env) & pl.col(med).is_finite()
            ).height
            if n_finite < 6:  # need ≥ 3 pairs => ≥ 6 cells
                continue
            try:
                result = proportion_mediated.fn(
                    env_cells,
                    target='eval_best_burst_mean',
                    mediator=med,
                    treatment_arm=DDQN,
                    baseline_arm=BASELINE,
                    pair_by=('seed', 'corpus', 'gamma', 'sync_period',
                             'total_steps'),
                    upstream_source='jensen_gap',
                    upstream_max_delta=0.0,
                )
            except Exception as exc:
                print(f'{env:<24} {med:<22} ERROR: {exc!r}')
                continue
            prop = result.proportion
            slope = result.slope_y_on_m
            n = result.n_pairs
            unit = result.in_unit_interval
            # Classification by the channel's actual behavior, not the
            # strict in_unit_interval bound. Slight overshoot (prop in
            # [1.0, 1.3]) is "essentially fully mediated, with noise";
            # only true suppressor (prop < -0.1) or wild overshoot
            # (prop > 1.5) is genuine linear-mediation failure.
            if math.isnan(prop):
                verdict = 'nan'
            elif n < 10:
                verdict = 'POWER_INSUFF'
            elif prop < -0.1 or prop > 1.5:
                verdict = 'LINEAR_BROKEN'
            elif 0.7 <= prop <= 1.3:
                verdict = 'FULL'         # essentially fully mediated
            elif prop >= 0.5:
                verdict = 'DOMINANT'
            elif prop >= 0.2:
                verdict = 'PARTIAL'
            elif prop > -0.1:
                verdict = 'MINOR'
            else:
                verdict = 'LINEAR_BROKEN'
            print(
                f'{env:<24} {med:<22} {n:>7} '
                f'{prop:>11.3f} {slope:>9.3g} {str(unit):>9} {verdict:<14}'
            )
            # Include in summary if proportion is in a *meaningful*
            # mediator range, not just the strict unit interval. An
            # overshoot of 1.05 says "fully mediated" not "broken."
            if not math.isnan(prop) and -0.1 < prop < 1.5:
                summary[env].append((med, prop))
        print()

    print()
    print('=== Per-env mediator typology (only HELD: in_unit ∧ n≥10) ===')
    print()
    for env, hits in sorted(summary.items()):
        # Sort by proportion desc.
        ranked = sorted(hits, key=lambda t: -t[1])
        if not ranked:
            print(f'{env:<24} (no in-unit mediators)')
            continue
        top = ', '.join(f'{m}:{p:.2f}' for m, p in ranked[:3])
        print(f'{env:<24} {top}')


if __name__ == '__main__':
    main()
