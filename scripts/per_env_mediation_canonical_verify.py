"""Per-env DoWhy mediation on the canonical cache (post-canonical-
verify ingest). For Asterix-MinAtar and SpaceInvaders-MinAtar
separately, run `stratum_delta_link_dowhy` with `jensen_bias_per_
burst_mean` as the mediator and `mc_return_raw_per_burst_mean`
as the outcome.

This closes the "mystery cohort dissolved → does jens mediate
PER ENV at adequate power?" question that the cross-env bridges
(now HELD) answer in aggregate.

Memory pointers:
- `findings_mystery_cohort_dissolved`: drops cohort, but cross-env
  aggregate HELD doesn't pin per-env.
- `findings_ddqn_mediator_heterogeneity`: original per-env DoWhy
  with the noise-floor canonical cache; this updates that.
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'src/corroborate_rl'))

import corroborate.analyses  # noqa: F401  populate registry
import corroborate_rl.dqn.measurables  # noqa: F401  populate registry

from corroborate.analyses.stratum_delta_link_dowhy import (
    stratum_delta_link_dowhy,
)
from corroborate_rl.dqn.measurables import (
    jensen_bias_per_burst_mean,
    mc_return_per_burst_mean,
)

DDQN = 'bootstrap=partial(Claim:bootstrap;greedification=Claim:double_greedify)'
BASE = 'baseline'

# Per-env corpora — these have traces locally (or restorable from S3).
ENV_CORPORA: dict[str, tuple[str, ...]] = {
    'Asterix-MinAtar': (
        'experiments/data/asterix_si_canonical_verify/verify_Asterix-MinAtar',
        'experiments/data/asterix_si_canonical_verify_resume/verify_Asterix-MinAtar',
    ),
    'SpaceInvaders-MinAtar': (
        'experiments/data/si_canonical_verify',
    ),
    'MetaMaze-misc': (
        'experiments/data/metamaze_canonical_verify/g099',
    ),
}


def load_env_cells(env: str) -> list:
    """Load runs.parquet + project ONLY the trace columns the
    measurables need. Avoids pulling 8GB worth of unrelated trace
    arrays into RAM."""
    needed_trace_cols = {'id', 'mc_return', 'predicted_q_at_start'}
    corpora = ENV_CORPORA[env]
    dfs = []
    for corpus in corpora:
        print(f'  loading {corpus}...', flush=True)
        runs = pl.read_parquet(f'{corpus}/runs.parquet')
        traces_path = f'{corpus}/traces.parquet'
        # Column-projected read — only the columns the measurables need.
        trace_cols = list(needed_trace_cols)
        traces = pl.read_parquet(traces_path, columns=trace_cols)
        df = runs.join(traces.drop([c for c in traces.columns if c in runs.columns and c != 'id']), on='id')
        dfs.append(df)
    common = sorted(set.intersection(*(set(df.columns) for df in dfs)))
    df = pl.concat([d.select(common) for d in dfs])
    print(f'  joined: {df.shape}', flush=True)
    return list(df.iter_rows(named=True))


def main() -> None:
    for env in ENV_CORPORA:
        cells = load_env_cells(env)
        print(f'\n### {env}: {len(cells)} cells')

        result = stratum_delta_link_dowhy.fn(
            cells,
            treatment_arm=DDQN,
            baseline_arm=BASE,
            link_predictor=jensen_bias_per_burst_mean,
            link_target=mc_return_per_burst_mean,
            env_filter=(env,),
            method_name='backdoor.linear_regression',
            min_vanilla_predictor=0.05,
        )
        print(f'  n_strata (env, burst Δs): {result.n_strata}')
        print(f'  treatment_col: {result.treatment_col}')
        print(f'  outcome_col: {result.outcome_col}')
        print(f'  --- backdoor ---')
        for fname in result.backdoor.__dataclass_fields__:
            v = getattr(result.backdoor, fname)
            if isinstance(v, (int, float)):
                print(f'    {fname}: {v:+.4f}')
            else:
                print(f'    {fname}: {v}')
        print(f'  --- placebo ---')
        for fname in result.placebo.__dataclass_fields__:
            v = getattr(result.placebo, fname)
            if isinstance(v, (int, float)):
                print(f'    {fname}: {v:+.4f}')
        print(f'  --- random_common_cause ---')
        for fname in result.random_common_cause.__dataclass_fields__:
            v = getattr(result.random_common_cause, fname)
            if isinstance(v, (int, float)):
                print(f'    {fname}: {v:+.4f}')


if __name__ == '__main__':
    main()
