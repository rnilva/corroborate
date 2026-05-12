"""`tautology_audit` on the PC variable set — formalize which
mediator candidates are shadows.

The PC discovery on REACH (`run_pc_reach.py`) found `is_ddqn ⊥
jens | {q_div}` and `is_ddqn ⊥ q_div | {jens}`. The reviewer
flagged these as partly algebraic-tautology removals (qdiv =
jens / (R/(1−γ)) within fixed (env, γ)), distinct from genuine
empirical conditional-independence findings.

This script runs `tautology_audit` on the same variable set:
qdiv, argmax_H, stale, eff_h tested against jens-as-outcome
(structural / HP / stratified-ρ checks). The audit's typed
verdict lets us label each PC removal as "tautological shadow"
or "empirical refutation" without arguing from algebra.

Two audit configurations:

  1. `outcome_path='jensen_gap'` — "is X a shadow of jens?".
     The structural jaccard check tests reads-set overlap;
     stratified-ρ-by-γ tests within-γ qdiv/jens collinearity.

  2. `outcome_path='eval_best_burst_mean'` — "is X a shadow of
     outcome?" — sanity check; we don't expect any of these
     mediator candidates to read outcome's traces.
"""
from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import polars as pl

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

import corroborate.analyses  # pyright: ignore[reportUnusedImport]  # populate registry
import corroborate_rl.dqn.measurables  # pyright: ignore[reportUnusedImport]  # populate registry
from corroborate.analyses.tautology_audit import tautology_audit
from corroborate.runner.runner import (
    _load_directory,  # pyright: ignore[reportPrivateUsage]
    _compute_measurables,  # pyright: ignore[reportPrivateUsage]
)


REACH_ENVS = (
    'FourRooms-misc', 'Acrobot-v1', 'MountainCar-v0', 'MetaMaze-misc',
)
DDQN_ARM = 'bootstrap=partial(Claim:bootstrap;greedification=Claim:double_greedify)'
BASELINE_ARM = 'baseline'

# Mediator candidates — same as PC variable set minus is_ddqn and
# minus the "outcome" each audit treats as the protected target.
# Reads-sets MUST match the registered @measurable definitions in
# `corroborate_rl/dqn/measurables.py`; the structural check
# compares these directly.
MEDIATOR_SPECS: tuple[Mapping[str, object], ...] = (
    {
        'name': 'q_divergence_score',
        'reads': ('jensen_gap', 'gamma'),
    },
    {
        'name': 'argmax_entropy_late',
        'reads': ('online_argmax_per_step',),
    },
    {
        'name': 'target_staleness_late',
        'reads': ('online_max_q_per_step', 'target_max_q_per_step'),
    },
    {
        'name': 'effective_horizon',
        'reads': ('gamma',),
    },
)


def main() -> None:
    required = (
        'jensen_gap', 'q_divergence_score', 'argmax_entropy_late',
        'target_staleness_late', 'effective_horizon',
        'eval_best_burst_mean', 'jensen_dormancy_gap',
    )
    runs = _load_directory(
        _REPO_ROOT / 'experiments' / 'data',
        restore_from_cloud=False, required=required, bridges=(),
    )
    runs = _compute_measurables(runs, required)

    # Restrict to REACH cohort ∩ DDQN-relevant scope (G1∧G2 standard
    # config, matching the PC discovery's scope).
    cohort = runs.filter(
        pl.col('env_name').is_in(list(REACH_ENVS))
        & pl.col('arm_key').is_in([DDQN_ARM, BASELINE_ARM])
        & pl.col('jensen_gap').is_finite()
        & (pl.col('jensen_gap') > 0.05)
        & pl.col('jensen_dormancy_gap').is_finite()
        & (pl.col('jensen_dormancy_gap') < 0.05)
        & pl.col('n_actions').is_finite()
        & (pl.col('n_actions') >= 3)
        & ((pl.col('n_step') == 1) | pl.col('n_step').is_null())
        & pl.col('action_duplicate_k').is_null()
        & (pl.col('reward_scale').is_null() | (pl.col('reward_scale') == 1.0))
    )
    cells = cohort.to_dicts()
    print(f'REACH cohort in DDQN-relevant scope: {len(cells)} cells, '
          f'envs: {sorted(cohort["env_name"].unique().to_list())}')
    print()

    # Bare measurable paths — the audit's default convention is
    # `mediator.{name}`, but our cells expose them as bare flat
    # keys (`q_divergence_score` directly on the cell dict).
    mediator_paths: dict[str, str] = {}
    for m in MEDIATOR_SPECS:
        name_v = m['name']
        if not isinstance(name_v, str):
            raise TypeError(f'measurable spec name must be str: {m!r}')
        mediator_paths[name_v] = name_v

    # === Audit 1: jens as outcome ===
    print('=== Audit 1: "is X a shadow of jens?" ===\n')
    result_jens = tautology_audit.fn(
        cells,
        measurables=MEDIATOR_SPECS,
        outcome_path='jensen_gap',
        outcome_reads=('predicted_q_at_start', 'mc_return'),
        hp_axes=('gamma', 'sync_period', 'replay.capacity'),
        hp_stratum_axis='gamma',
        mediator_path_for=mediator_paths,
        outcome_jaccard_threshold=0.5,
        hp_r_squared_threshold=0.95,
        stratified_rho_threshold=0.1,
        stratified_alpha=0.05,
    )
    for r in result_jens.reports:
        flags: list[str] = []
        if r.flagged_outcome:
            flags.append('OUTCOME-shadow-of-jens')
        if r.flagged_hp:
            flags.append(f'HP-shadow={list(r.flagged_hp)}')
        if r.flagged_no_residual_signal:
            flags.append('no-residual-signal-vs-jens')
        if not flags:
            flags.append('CLEAN')
        print(f'  {r.measurable_name:<28s}: {", ".join(flags)}')
        print(f'    jaccard={r.outcome_jaccard:.3f}, '
              f'within-γ ρ(.,jens)={r.outcome_stratified_rho:+.3f} '
              f'(p={r.outcome_stratified_p:.3g}), '
              f'HP_R²={dict(r.hp_r_squared)}')
        print()

    # === Audit 2: outcome as outcome ===
    print('=== Audit 2: "is X a shadow of eval_best_burst_mean?" ===\n')
    result_out = tautology_audit.fn(
        cells,
        measurables=MEDIATOR_SPECS,
        outcome_path='eval_best_burst_mean',
        outcome_reads=('eval_return',),  # eval-time scalar; no shared trace cols
        hp_axes=('gamma', 'sync_period', 'replay.capacity'),
        hp_stratum_axis='gamma',
        mediator_path_for=mediator_paths,
        outcome_jaccard_threshold=0.5,
        hp_r_squared_threshold=0.95,
        stratified_rho_threshold=0.1,
        stratified_alpha=0.05,
    )
    for r in result_out.reports:
        flags: list[str] = []
        if r.flagged_outcome:
            flags.append('OUTCOME-shadow')
        if r.flagged_hp:
            flags.append(f'HP-shadow={list(r.flagged_hp)}')
        if r.flagged_no_residual_signal:
            flags.append('no-residual-signal-vs-outcome')
        if not flags:
            flags.append('CLEAN')
        print(f'  {r.measurable_name:<28s}: {", ".join(flags)}')
        print(f'    jaccard={r.outcome_jaccard:.3f}, '
              f'within-γ ρ(.,outcome)={r.outcome_stratified_rho:+.3f} '
              f'(p={r.outcome_stratified_p:.3g})')
        print()

    out_path = (
        _REPO_ROOT / 'experiments' / 'findings' /
        'ddqn_pc_reach' / 'tautology_audit.json'
    )
    out_dict = {
        'cohort_size': len(cells),
        'cohort_envs': sorted(cohort['env_name'].unique().to_list()),
        'audit_against_jens': [
            {
                'name': r.measurable_name,
                'outcome_jaccard': r.outcome_jaccard,
                'flagged_outcome': r.flagged_outcome,
                'hp_r_squared': dict(r.hp_r_squared),
                'flagged_hp': list(r.flagged_hp),
                'within_gamma_rho_vs_jens': r.outcome_stratified_rho,
                'within_gamma_p_vs_jens': r.outcome_stratified_p,
                'flagged_no_residual_signal': r.flagged_no_residual_signal,
                'is_clean': r.is_clean,
            }
            for r in result_jens.reports
        ],
        'audit_against_outcome': [
            {
                'name': r.measurable_name,
                'outcome_jaccard': r.outcome_jaccard,
                'flagged_outcome': r.flagged_outcome,
                'hp_r_squared': dict(r.hp_r_squared),
                'flagged_hp': list(r.flagged_hp),
                'within_gamma_rho_vs_outcome': r.outcome_stratified_rho,
                'within_gamma_p_vs_outcome': r.outcome_stratified_p,
                'flagged_no_residual_signal': r.flagged_no_residual_signal,
                'is_clean': r.is_clean,
            }
            for r in result_out.reports
        ],
    }
    out_path.write_text(json.dumps(out_dict, indent=2, default=str))
    print(f'wrote: {out_path}')


if __name__ == '__main__':
    main()
