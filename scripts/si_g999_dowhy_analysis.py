"""DoWhy backdoor + refutation + mediation at SI γ=0.999.

Companion to `si_g999_causal_analysis.py` (partial-Spearman form).
Per CLAUDE.md mediation recipe:

1. Power gate the TOTAL ATE via DoWhy backdoor + placebo + RCC.
2. Topology gate via PC discovery (done in the partial-Spearman script).
3. Mediation via partial-Spearman (done).
4. mediation_dowhy as DIAGNOSTIC (this script).
5. Refutations corroborate the foundation.

Output: corroborates the partial-Spearman result that bias is NOT
the load-bearing mediator, repeat-rate IS. Adds independent
identification + refutation evidence.
"""
from __future__ import annotations

import polars as pl
import experiments.findings.ddqn_three_conditions  # populate registries

from corroborate.analyses.dowhy import (
    backdoor_ate,
    placebo_refutation,
    random_common_cause_refutation,
)
from corroborate.analyses.dowhy.mediation_dowhy import mediation_dowhy


_SI_G999 = (
    (pl.col('env_name') == 'SpaceInvaders-MinAtar')
    & (pl.col('gamma') == 0.999)
    & pl.col('eval_best_burst_raw_mean').is_finite()
    & pl.col('jensen_gap').is_finite()
    & pl.col('state_repeat_rate_within_episode_window64_late').is_finite()
    & pl.col('state_hash_entropy_late').is_finite()
)


def main() -> None:
    df = pl.read_parquet('experiments/data/cache/ddqn_three_conditions.parquet')
    cells = df.filter(_SI_G999).with_columns(
        (pl.col('arm_key') != 'baseline').cast(pl.Float64).alias('arm'),
    )
    rows = list(cells.to_dicts())
    print(f'SI γ=0.999 cells: {len(rows)} (vanilla {sum(1 for r in rows if r["arm"]==0)}, DDQN {sum(1 for r in rows if r["arm"]==1)})')
    print()

    print('=== Stage 1 — Total ATE via DoWhy backdoor ===')
    dag_total = (('arm', 'eval_best_burst_raw_mean'),)
    result = backdoor_ate.fn(
        rows,
        treatment='arm',
        outcome='eval_best_burst_raw_mean',
        dag=dag_total,
    )
    print(f'  total ATE: {result.ate:+.3f}, identified={result.identified}, n={result.n_rows}')
    print()

    print('=== Stage 2 — Placebo refutation ===')
    placebo = placebo_refutation.fn(
        rows,
        treatment='arm',
        outcome='eval_best_burst_raw_mean',
        dag=dag_total,
    )
    print(f'  real ATE: {placebo.real_ate:+.3f}')
    print(f'  placebo ATE: {placebo.refuted_ate:+.3f}  (target: ≈ 0)')
    print(f'  drift: {placebo.drift:.3f}')
    print()

    print('=== Stage 3 — Random common cause refutation ===')
    rcc = random_common_cause_refutation.fn(
        rows,
        treatment='arm',
        outcome='eval_best_burst_raw_mean',
        dag=dag_total,
    )
    print(f'  real ATE: {rcc.real_ate:+.3f}')
    print(f'  RCC ATE: {rcc.refuted_ate:+.3f}  (target: ≈ real)')
    print(f'  drift: {rcc.drift:.3f}')
    print()

    print('=== Stage 4 — Mediation via DoWhy (LinearityStatus diagnostic) ===')
    print()
    for mediator_set in [
        ('jensen_gap',),
        ('state_repeat_rate_within_episode_window64_late',),
        ('state_hash_entropy_late',),
        ('state_repeat_rate_within_episode_window64_late', 'jensen_gap'),
        ('state_repeat_rate_within_episode_window64_late', 'state_hash_entropy_late'),
    ]:
        med_label = ' + '.join(m.replace('state_repeat_rate_within_episode_window64_late', 'repeat')
                                 .replace('state_hash_entropy_late', 'entropy')
                                 .replace('jensen_gap', 'jens')
                               for m in mediator_set)
        dag_med = [('arm', m) for m in mediator_set] + [(m, 'eval_best_burst_raw_mean') for m in mediator_set] + [('arm', 'eval_best_burst_raw_mean')]
        med = mediation_dowhy.fn(
            rows,
            treatment='arm',
            outcome='eval_best_burst_raw_mean',
            mediators=mediator_set,
            dag=dag_med,
        )
        ind_frac = med.indirect_proportion if med.indirect_proportion is not None else float('nan')
        print(f'  {med_label}:')
        print(f'    total: {med.total_ate:+.3f}, direct: {med.direct_ate:+.3f}, indirect: {med.indirect_ate:+.3f}')
        print(f'    indirect_proportion: {ind_frac:+.3f}')
        print(f'    linearity_status: {med.linearity_status}')
        print()


if __name__ == '__main__':
    main()
