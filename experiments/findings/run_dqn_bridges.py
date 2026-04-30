"""Run the authored DDQN bridges against their corpora and print
every verdict.

This is the file-protocol artifact in action: the bridges in
`dqn_bridges.py` are authored once; this script loads each
relevant corpus, evaluates every bridge, and prints the typed
verdicts. Falsification = run against a different corpus, see
which verdicts change.

Usage:
  uv run python -m experiments.findings.run_dqn_bridges
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import polars as pl

from corroborate.claim_bridge import Bridge, evaluate
from experiments.findings.dqn_bridges import (
    ACTION_DIM_BRIDGES, CARTPOLE_HP_AUDIT_BRIDGES,
    CHAIN_DECOMPOSITION_BRIDGES,
    DDQN_200K_BRIDGES, EXPECTILE_PER_BURST_BRIDGES,
    EXPECTILE_STRATEGY_2_BRIDGES, NSTEP_FACTORIAL_BRIDGES,
    NSTEP_INTERVENTION_BRIDGES,
    jensen_gap_outcome_borderline,
    state_coverage_kl_causes_outcome,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ACTION_DIM_PARQUET = (
    REPO_ROOT / 'experiments' / 'data' / 'action_dim_sweep'
    / 'runs.parquet'
)
EXPECTILE_RUNS = (
    REPO_ROOT / 'experiments' / 'data' / 'expectile_3way'
    / 'runs.parquet'
)
EXPECTILE_TRACES = (
    REPO_ROOT / 'experiments' / 'data' / 'expectile_3way'
    / 'traces.parquet'
)
CARTPOLE_HP_MEDIATORS = (
    REPO_ROOT / 'experiments' / 'data' / 'cartpole_hp_v2'
    / 'runs_with_mediators.parquet'
)
CARTPOLE_HP_RUNS = (
    REPO_ROOT / 'experiments' / 'data' / 'cartpole_hp'
    / 'runs_with_mediators.parquet'
)
DDQN_200K_RUNS = (
    REPO_ROOT / 'experiments' / 'data' / 'ddqn' / 'runs.parquet'
)
NSTEP_INTERVENTION_RUNS = (
    REPO_ROOT / 'experiments' / 'data' / 'nstep_intervention'
    / 'runs.parquet'
)
NSTEP_INTERVENTION_FR_RUNS = (
    REPO_ROOT / 'experiments' / 'data' / 'nstep_intervention_fr'
    / 'runs.parquet'
)
NSTEP_VANILLA_ARMS_RUNS = (
    REPO_ROOT / 'experiments' / 'data' / 'nstep_vanilla_arms'
    / 'runs.parquet'
)


def _format_paired_g(result: object) -> str:
    from corroborate.analyses.paired_g import PairedGResult
    if not isinstance(result, PairedGResult):
        return str(result)
    return (
        f'g={result.g:+.3f}, p={result.p_value:.4f}, '
        f'n={result.n_pairs}'
    )


def _format_meta_regression(result: object) -> str:
    from corroborate.meta_regression import MetaRegressionResult
    if not isinstance(result, MetaRegressionResult):
        return str(result)
    return (
        f'n_strata={result.n_strata}, coefs='
        f'{[(c.name, round(c.coefficient, 3)) for c in result.coefficients]}'
    )


def _format_backdoor(result: object) -> str:
    from corroborate.analyses.dowhy import BackdoorResult
    if not isinstance(result, BackdoorResult):
        return str(result)
    return (
        f'ATE={result.ate:+.3f}, identified={result.identified}, '
        f'n={result.n_rows}'
    )


def _format_refutation(result: object) -> str:
    from corroborate.analyses.dowhy import RefutationResult
    if not isinstance(result, RefutationResult):
        return str(result)
    return (
        f'real={result.real_ate:+.3f}, refuted={result.refuted_ate:+.3f}, '
        f'drift={result.drift:.3f}'
    )


def _format_audit(result: object) -> str:
    from corroborate.analyses.tautology_audit import AuditResult
    if not isinstance(result, AuditResult):
        return str(result)
    parts: list[str] = []
    for r in result.reports:
        parts.append(
            f'{r.measurable_name}: jaccard={r.outcome_jaccard:.2f}, '
            f'clean={r.is_clean}',
        )
    return '; '.join(parts)


def _format_per_burst(result: object) -> str:
    from corroborate.analyses.paired_g_per_burst import PerBurstResult
    if not isinstance(result, PerBurstResult):
        return str(result)
    if not result.strata:
        return 'no strata'
    gs = [s.g for s in result.strata]
    return (
        f'n_strata={result.n_strata}, '
        f'g min={min(gs):+.2f} max={max(gs):+.2f} '
        f'mean={sum(gs)/len(gs):+.2f}'
    )


def _format_factorial(result: object) -> str:
    from corroborate.analyses.factorial_2x2 import Factorial2x2Result
    if not isinstance(result, Factorial2x2Result):
        return str(result)
    parts: list[str] = []
    for p in result.per_env:
        z = (
            p.g_interaction / p.se_interaction
            if p.se_interaction > 0 else float('nan')
        )
        parts.append(
            f'{p.env_name}: INT={p.g_interaction:+.2f} z={z:+.1f}',
        )
    return '; '.join(parts)


def _format_pooled(result: object) -> str:
    from corroborate.analyses.paired_g_pooled import PooledPairedGResult
    if not isinstance(result, PooledPairedGResult):
        return str(result)
    return (
        f'pooled_g={result.pooled.pooled_g:+.3f}, '
        f'I²={result.pooled.I2:.2f}, n_envs={result.n_envs}'
    )


def _format_verdict_distribution(result: object) -> str:
    from corroborate.analyses.verdict_distribution import (
        VerdictDistributionResult,
    )
    if not isinstance(result, VerdictDistributionResult):
        return str(result)
    parts: list[str] = []
    for env, c in sorted(result.per_env.items()):
        parts.append(
            f'{env}: held={c.held} violation={c.invariant_violation} '
            f'(n={c.total}, dom={c.dominant})',
        )
    return '; '.join(parts)


def _format_analysis_result(result: object) -> str:
    """Dispatch by registered analysis return type."""
    from corroborate.analyses.dowhy import (
        BackdoorResult, RefutationResult,
    )
    from corroborate.analyses.factorial_2x2 import Factorial2x2Result
    from corroborate.analyses.paired_g import PairedGResult
    from corroborate.analyses.paired_g_per_burst import PerBurstResult
    from corroborate.analyses.paired_g_pooled import PooledPairedGResult
    from corroborate.analyses.tautology_audit import AuditResult
    from corroborate.analyses.verdict_distribution import (
        VerdictDistributionResult,
    )
    from corroborate.meta_regression import MetaRegressionResult
    if isinstance(result, PairedGResult):
        return _format_paired_g(result)
    if isinstance(result, PerBurstResult):
        return _format_per_burst(result)
    if isinstance(result, PooledPairedGResult):
        return _format_pooled(result)
    if isinstance(result, Factorial2x2Result):
        return _format_factorial(result)
    if isinstance(result, MetaRegressionResult):
        return _format_meta_regression(result)
    if isinstance(result, AuditResult):
        return _format_audit(result)
    if isinstance(result, BackdoorResult):
        return _format_backdoor(result)
    if isinstance(result, RefutationResult):
        return _format_refutation(result)
    if isinstance(result, VerdictDistributionResult):
        return _format_verdict_distribution(result)
    return str(result)


def _print_verdicts(
    bridges: Sequence[Bridge],
    cells: Sequence[dict[str, object]],
) -> None:
    for bridge in bridges:
        out = evaluate(bridge, cells)
        for analysis_name, result in out.analysis_results.items():
            stats = _format_analysis_result(result)
            print(
                f'{bridge.name:<55} {out.verdict.value:<22} '
                f'[{analysis_name}] {stats}',
            )


def main() -> None:
    print(f'{"BRIDGE":<55} {"VERDICT":<22} STATS')
    print('=' * 110)

    if ACTION_DIM_PARQUET.exists():
        df = pl.read_parquet(ACTION_DIM_PARQUET)
        cells = list(df.iter_rows(named=True))
        print(
            f'\n# action_dim_sweep ({len(cells)} cells, '
            f'{df["env_name"].n_unique()} envs)',
        )
        print('-' * 110)
        _print_verdicts(ACTION_DIM_BRIDGES, cells)
        # Audit-style claim — homogeneous-HP corpus so within-env
        # ρ stratification is degenerate, but the structural
        # jaccard reproduces.
        _print_verdicts((jensen_gap_outcome_borderline,), cells)
    else:
        print(f'(skip action_dim_sweep — {ACTION_DIM_PARQUET} missing)')

    if EXPECTILE_RUNS.exists():
        runs_df = pl.read_parquet(EXPECTILE_RUNS)
        runs_cells = list(runs_df.iter_rows(named=True))
        print(
            f'\n# expectile_3way ({len(runs_cells)} cells, '
            f'{runs_df["env_name"].n_unique()} envs, '
            f'{runs_df["intervention_name"].n_unique()} arms)',
        )
        print('-' * 110)
        _print_verdicts(EXPECTILE_STRATEGY_2_BRIDGES, runs_cells)

        if EXPECTILE_TRACES.exists():
            runs = pl.read_parquet(
                EXPECTILE_RUNS,
                columns=[
                    'id', 'intervention_name', 'env_name', 'seed',
                ],
            )
            traces = pl.read_parquet(
                EXPECTILE_TRACES,
                columns=['id', 'mc_return', 'predicted_q_at_start'],
            )
            joined_cells = list(
                runs.join(traces, on='id', how='inner')
                .iter_rows(named=True),
            )
            print(
                f'\n# expectile_3way joined runs × traces '
                f'({len(joined_cells)} cells)',
            )
            print('-' * 110)
            _print_verdicts(EXPECTILE_PER_BURST_BRIDGES, joined_cells)
            _print_verdicts(CHAIN_DECOMPOSITION_BRIDGES, joined_cells)
    else:
        print(f'(skip expectile_3way — {EXPECTILE_RUNS} missing)')

    if CARTPOLE_HP_RUNS.exists():
        df = pl.read_parquet(CARTPOLE_HP_RUNS)
        cells = list(df.iter_rows(named=True))
        print(
            f'\n# cartpole_hp 180-cell audit ({len(cells)} cells)',
        )
        print('-' * 110)
        _print_verdicts(CARTPOLE_HP_AUDIT_BRIDGES, cells)
    else:
        print(
            f'(skip cartpole_hp audit — restore via `corroborate '
            f'restore experiments/data/cartpole_hp`)',
        )

    if CARTPOLE_HP_MEDIATORS.exists():
        df = pl.read_parquet(CARTPOLE_HP_MEDIATORS)
        cells = list(df.iter_rows(named=True))
        print(
            f'\n# cartpole_hp_v2 mediators ({len(cells)} cells)',
        )
        print('-' * 110)
        _print_verdicts((state_coverage_kl_causes_outcome,), cells)
    else:
        print(
            f'(skip DoWhy — restore cartpole_hp_v2 from R2 to '
            f'evaluate state_coverage_kl_causes_outcome)',
        )

    if DDQN_200K_RUNS.exists():
        df = pl.read_parquet(
            DDQN_200K_RUNS,
            columns=[
                'id', 'env_name', 'intervention_name', 'seed',
                'total_steps', 'mechanism.jensen_gap',
                'outcome.eval_best_burst_mean',
                'outcome.eval_best_burst_step',
            ],
        )
        cells = list(df.iter_rows(named=True))
        print(
            f'\n# ddqn 200k corpus ({len(cells)} cells, '
            f'{df["env_name"].n_unique()} envs, '
            f'{df["total_steps"].n_unique()} step-budgets)',
        )
        print('-' * 110)
        _print_verdicts(DDQN_200K_BRIDGES, cells)
    else:
        print(
            f'(skip ddqn 200k — restore via `corroborate restore '
            f'experiments/data/ddqn`)',
        )

    if NSTEP_INTERVENTION_RUNS.exists():
        df = pl.read_parquet(NSTEP_INTERVENTION_RUNS)
        cells = list(df.iter_rows(named=True))
        print(
            f'\n# nstep_intervention ({len(cells)} cells, '
            f'{df["env_name"].n_unique()} envs, '
            f'{df["intervention_name"].n_unique()} arms)',
        )
        print('-' * 110)
        _print_verdicts(NSTEP_INTERVENTION_BRIDGES, cells)

    factorial_paths = (
        NSTEP_INTERVENTION_RUNS, NSTEP_INTERVENTION_FR_RUNS,
        NSTEP_VANILLA_ARMS_RUNS,
    )
    if all(p.exists() for p in factorial_paths):
        cols = [
            'env_name', 'intervention_name', 'seed',
            'total_steps', 'outcome.eval_best_burst_mean',
        ]
        union_df = pl.concat(
            [pl.read_parquet(p, columns=cols) for p in factorial_paths],
            how='vertical_relaxed',
        )
        cells = list(union_df.iter_rows(named=True))
        print(
            f'\n# 2×2 factorial union ({len(cells)} cells, '
            f'{union_df["env_name"].n_unique()} envs, '
            f'{union_df["intervention_name"].n_unique()} arms)',
        )
        print('-' * 110)
        _print_verdicts(NSTEP_FACTORIAL_BRIDGES, cells)


if __name__ == '__main__':
    main()
