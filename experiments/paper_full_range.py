"""Paper full-range experiment — end-to-end §3–§7 + robustness on
the DDQN sweep corpus.

Driver for the v0 acceptance test plus paper sections built on top.
For each `total_steps` grid point in the corpus, runs:

  §3  Three-way verdict (mechanism, outcome, link) via typed
      Hypothesis subgraph + `hypothesis_subgraph_verdict`. The
      returned `HypothesisVerdict` carries the typed CausalGraph
      directly.
  §4  Pooled conservative-PC adjacency on a wide variable set
      (stratified by `env_name` for JCI).
  §5  Within-env Pearson r of the 8-mediator catalog vs outcome.
      Identifies per-env mediator candidates pre-§6 PC.
  §6  Per-env conservative-PC + regime classification (the v10
      §6 three-regime taxonomy: TD-convergence / action-margin /
      stay-greedy / value-growth / none).
  §7  Meta-regression on per-env effect sizes against env-metadata
      covariates. Empirical scope predictor — replaces v10's
      discrete metadata→regime classifier with numeric coefficient
      thresholds + CIs.
  D1  PC depth-2 robustness on §4's variable set: edge-set diff
      vs depth-1.
  D2  k-fold CV on §7's coefficients: sign stability across
      env-fold splits.

Outputs:
  - stdout: paper-style tables per section
  - {output_dir}/paper_results.parquet: flat structured rows for
    downstream paper-write-up consumption (one row per
    `(section, total_steps, key, value)` tuple).
  - {output_dir}/causal_graph_{total_steps}.txt: typed-CausalGraph
    dump per grid point.

Usage:
    uv run python experiments/paper_full_range.py
    uv run python experiments/paper_full_range.py --total-steps 200000
    uv run python experiments/paper_full_range.py --output-dir /tmp/paper
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Literal, cast

import numpy as np
import polars as pl
import scipy.stats as ss

from corroborate.causal_discovery import compare_pc_depths, discover_adjacency
from corroborate.causal_graph import Direction, Tier
from corroborate.claim_bridge import Bridge as ClaimBridge
from corroborate.hypothesis import Hypothesis
from corroborate.hypothesis_verdict import (
    HypothesisVerdict,
    hypothesis_subgraph_verdict,
)
from corroborate.intervention import DoEffect, Intervention
from corroborate.meta_regression import (
    StratumObservation,
    cross_validate_meta_regression,
    meta_regression,
)
from corroborate.persistence import read_runrows
from corroborate.rl.dqn.claims.bootstrap import bootstrap, double_greedify
from corroborate.rl.dqn.measurables import dqn_default_measurables
from corroborate.rl.env_catalogue import (
    BenchmarkFamily,
    ENV_REGISTRY,
    EnvSpec,
    RewardRegime,
)
from corroborate.schema import RunRow
from corroborate.verdict import Verdict


# ============ Constants ============

_DEFAULT_RUNS = Path(
    '/workspace/corroborate/experiments/data/ddqn/'
    'runs_with_mediators.parquet'
)

_MEDIATORS: tuple[str, ...] = (
    # v10 §5.1 catalog (8 candidate mediators).
    'mediator.q_gap_late',
    'mediator.q_gap_growth',
    'mediator.q_max_growth',
    'mediator.v_vs_max_delta_late',
    'mediator.td_residual_late',
    'mediator.greedy_match_late',
    'mediator.fill_ratio_late',
    'mediator.epsilon_late',
    # D3 value-curve family — v10 §4.6 candidate mediator family 2.
    'mediator.learning_curve_auc',
    'mediator.time_to_threshold',
    'mediator.return_at_25pct_steps',
    'mediator.plateau_slope_late',
    # F1 state-coverage — v10 §4.6 candidate mediator family 3.
    # (Action-margin proxy `mean(|q_mean − q_max|)` is the
    # existing `v_vs_max_delta_late`; a true Q* − Q_2nd
    # measurable awaits a per-step second-max reduction in the
    # collect harness.)
    'mediator.state_visit_entropy_late',
    'mediator.state_coverage_kl_uniform_late',
    # H2 peak-truncated reductions — per-cell-aware windows that
    # avoid post-peak contamination. Same per-cell signal as their
    # `_late` counterparts but with the window cut at each cell's
    # `outcome.eval_best_burst_step`. Empirically (H1 exploration)
    # these surface 24-28% larger mean |r| vs outcome across envs.
    'mediator.q_gap_peak_truncated_late',
    'mediator.td_residual_peak_truncated_late',
    'mediator.greedy_match_peak_truncated_late',
    'mediator.learning_curve_auc_peak_truncated',
)

import os as _os
_OUTCOME = _os.environ.get(
    'OUTCOME_PATH', 'outcome.eval_best_burst_mean',
)
"""Primary outcome path. Default matches Hasselt 2016's
literature convention: `outcome.eval_best_burst_mean` is the
max over per-eval-burst mean returns — the 'best eval score'
metric Atari benchmarks report. Override via env var:

- `OUTCOME_PATH=outcome.eval_best_burst_mean` (default;
  Hasselt-convention; sample-efficiency-aware but possibly
  optimistic on unstable runs).
- `OUTCOME_PATH=outcome.eval_final_mean` (final eval burst —
  what return the agent achieved at training's end; vulnerable
  to one-bad-gradient noise).
- `OUTCOME_PATH=outcome.late_window_mean` (per-step late-window
  mean — stability-aware; the framework's stability-flavored
  outcome that gives a different mech↔outcome story than
  best-eval-burst, see §3 paper paragraph)."""
_MECHANISM = 'mechanism.jensen_gap'

# Mediators with non-trivial within-corpus variance for PC.
# Exclusions:
#   - `epsilon_late`, `fill_ratio_late` — corpus-wide constants
#     (eps schedule + replay-full status uniform across cells in
#     this sweep).
#   - `state_visit_entropy_late`,
#     `state_coverage_kl_uniform_late` — NaN for image-state /
#     bandit envs (default state_hash returns 0 → degenerate
#     distribution). Including them in `_PC_VARIABLES` would
#     drop every env via the non-null filter even when the
#     env's other columns have signal.
_PC_MEDIATORS: tuple[str, ...] = tuple(
    m for m in _MEDIATORS
    if m not in (
        'mediator.epsilon_late', 'mediator.fill_ratio_late',
        'mediator.state_visit_entropy_late',
        'mediator.state_coverage_kl_uniform_late',
    )
)

# Per-env PC variable set (§4 + §6). Includes both
# `outcome.eval_final_mean` and the primary `_OUTCOME`; deduped
# in case the override picks `eval_final_mean` itself.
_PC_VARIABLES: tuple[str, ...] = tuple(dict.fromkeys((
    'arm_ddqn',
    _MECHANISM,
    *_PC_MEDIATORS,
    'outcome.eval_final_mean',
    _OUTCOME,
)))

# §7 covariates: env-metadata one-hot encoded (drop-first
# reference levels to avoid intercept collinearity).
_FAMILY_REFERENCE: BenchmarkFamily = 'classic_control'
_FAMILIES: tuple[BenchmarkFamily, ...] = (
    'minatar', 'bsuite', 'bandit', 'misc',
)
_REGIME_REFERENCE: RewardRegime = 'per_step'
_REGIMES: tuple[RewardRegime, ...] = (
    'event_triggered', 'shaped', 'terminal_only',
)


# ============ Helpers ============

_DDQN_DO = DoEffect(treatment_arm='ddqn', baseline_arm='vanilla_dqn')


def _ddqn_hypothesis() -> Hypothesis[Mapping[str, object]]:
    return Hypothesis(
        name='ddqn',
        intervention={},
        intervention_arms=(
            Intervention(
                slot_path='bootstrap',
                replacement=partial(
                    bootstrap, greedification=double_greedify,
                ),
            ),
        ),
        edges=(
            ClaimBridge(
                name=f'ddqn_mechanism({_MECHANISM})',
                source=_DDQN_DO.node_key(),
                target=_MECHANISM,
                tier=Tier.INTERVENTIONAL,
                direction=Direction.DIRECT,
                intervention=_DDQN_DO,
                predicted_direction='a_lt_b',
            ),
            ClaimBridge(
                name=f'ddqn_outcome({_OUTCOME})',
                source=_DDQN_DO.node_key(),
                target=_OUTCOME,
                tier=Tier.INTERVENTIONAL,
                direction=Direction.DIRECT,
                intervention=_DDQN_DO,
                predicted_direction='a_gt_b',
            ),
            ClaimBridge(
                name=f'coupling({_MECHANISM}->{_OUTCOME})',
                source=_MECHANISM,
                target=_OUTCOME,
                tier=Tier.ASSOCIATIONAL,
                direction=Direction.DIRECT,
                predicted_direction='a_gt_b',
            ),
        ),
        measurables=dqn_default_measurables(),
    )


def _vanilla_hypothesis() -> Hypothesis[Mapping[str, object]]:
    return Hypothesis(
        name='vanilla_dqn', intervention={}, intervention_arms=(),
        measurables=dqn_default_measurables(),
    )


def _env_covariates_full(env_name_obj: object) -> Mapping[str, float]:
    """Full env-metadata covariate set. 10 covariates + intercept
    = 11 parameters. With n=17 envs the design matrix is tight
    (df=6) and *outcome*'s n=15 hits singular when collinearities
    arise (which they do — all MinAtar envs are event_triggered;
    all bandit envs are terminal_only; etc.). Useful as an
    appendix-grade view; primary §7 result should use the minimal
    set."""
    if not isinstance(env_name_obj, str):
        raise TypeError(
            f'env covariate provider needs str, got '
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


def _env_covariates_minimal(env_name_obj: object) -> Mapping[str, float]:
    """Theory-motivated minimal covariate set. 4 covariates +
    intercept = 5 parameters. With n=17 envs: df=12; with n=15
    (outcome's degenerate-stratum subset): df=10. Both healthy.

    Covariates kept:
    - `reward_range` (= r_max − r_min): v10 §7's headline
      cleavage axis. Theoretically motivated — the Jensen gap is
      bounded by reward magnitude, so DDQN's bias-reduction
      effect should scale with reward range.
    - `horizon`: orthogonal to family/regime; captures the
      training-budget / per-episode rollout length axis.
    - `family_minatar`, `family_bsuite`: dominant 11-of-17 envs
      (vs reference `classic_control + bandit + misc`).

    Dropped from the full set:
    - `n_actions` (collinear with family — MinAtar=6,
      classic=2-3, bsuite=2-3).
    - `family_bandit`, `family_misc` (small categories n=2-3;
      noisy coefficients).
    - All regime flags (collinear with family in this corpus —
      every MinAtar is event_triggered, etc.)."""
    if not isinstance(env_name_obj, str):
        raise TypeError(
            f'env covariate provider needs str, got '
            f'{type(env_name_obj).__name__}',
        )
    spec: EnvSpec = ENV_REGISTRY[env_name_obj]
    return {
        'reward_range': float(spec.r_max - spec.r_min),
        'horizon': float(spec.eval_episode_cap),
        'family_minatar':
            1.0 if spec.benchmark_family == 'minatar' else 0.0,
        'family_bsuite':
            1.0 if spec.benchmark_family == 'bsuite' else 0.0,
    }


type CovariateSet = Literal['minimal', 'full', 'class3', 'mixed']


_CLASS3_PATHS: tuple[str, ...] = (
    'mediator.q_gap_late',
    'mediator.td_residual_late',
    'mediator.greedy_match_late',
    'mediator.learning_curve_auc',
)


def _per_env_baseline_means(
    runs: 'Sequence[RunRow]',
) -> Mapping[str, Mapping[str, float]]:
    """Compute per-env mean of `_CLASS3_PATHS` over baseline-arm
    runs. Returns nested dict `out[env_name][f'baseline_{name}_mean']`.

    These are *Class 3* covariates: cross-env summaries of
    endogenous quantities measured under the baseline arm. They
    test 'is DDQN's effect bigger in envs where vanilla already
    has signature X?' — a mediator-readiness question."""
    by_env: dict[str, list[RunRow]] = {}
    for r in runs:
        if r.arm_key != 'baseline':
            continue
        env = r.measurements.get('env_name')
        if isinstance(env, str):
            by_env.setdefault(env, []).append(r)

    out: dict[str, dict[str, float]] = {}
    for env, env_runs in by_env.items():
        env_means: dict[str, float] = {}
        for path in _CLASS3_PATHS:
            values: list[float] = []
            for r in env_runs:
                v = r.measurements.get(path)
                if isinstance(v, (int, float)) and not (
                    isinstance(v, float) and math.isnan(v)
                ):
                    values.append(float(v))
            short = path.removeprefix('mediator.')
            env_means[f'baseline_{short}_mean'] = (
                float(sum(values) / len(values))
                if values else float('nan')
            )
        out[env] = env_means
    return out


def _build_covariate_fn(
    set_name: CovariateSet,
    runs: 'Sequence[RunRow]',
) -> Callable[[object], Mapping[str, float]]:
    """Build the covariate function for the chosen set. `runs` is
    consumed only by Class-3-using sets to compute per-env baseline
    aggregates; the env-metadata-only sets ignore it."""
    if set_name == 'minimal':
        return _env_covariates_minimal
    if set_name == 'full':
        return _env_covariates_full

    vanilla_means = _per_env_baseline_means(runs)

    if set_name == 'class3':
        def class3_fn(env_name_obj: object) -> Mapping[str, float]:
            if not isinstance(env_name_obj, str):
                raise TypeError(
                    f'class3 covariates need str env_name, got '
                    f'{type(env_name_obj).__name__}',
                )
            return dict(vanilla_means.get(env_name_obj, {}))
        return class3_fn

    if set_name == 'mixed':
        def mixed_fn(env_name_obj: object) -> Mapping[str, float]:
            if not isinstance(env_name_obj, str):
                raise TypeError(
                    f'mixed covariates need str env_name, got '
                    f'{type(env_name_obj).__name__}',
                )
            out = dict(_env_covariates_minimal(env_name_obj))
            for k, v in vanilla_means.get(env_name_obj, {}).items():
                out[k] = v
            return out
        return mixed_fn

    raise ValueError(f'unknown covariate set: {set_name!r}')


def _pearson_safe(
    x: np.ndarray, y: np.ndarray,
) -> tuple[float, float]:
    finite = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(finite)) < 3:
        return float('nan'), float('nan')
    xs = x[finite]
    ys = y[finite]
    if float(np.std(xs)) == 0.0 or float(np.std(ys)) == 0.0:
        return float('nan'), float('nan')
    # scipy boundary — `pearsonr` stubs leak Unknown through the
    # tuple unpack; coerce both elements at this boundary.
    r, p = ss.pearsonr(xs, ys)  # pyright: ignore[reportUnknownMemberType]
    return float(r), float(p)  # pyright: ignore[reportArgumentType]


# ============ Section runners ============

def _section_3_three_way(
    runs: list[RunRow],
    total_steps: int,
    output_dir: Path,
) -> HypothesisVerdict[Mapping[str, object]]:
    """§3 — typed three-way verdict + typed CausalGraph dump."""
    print()
    print('=' * 92)
    print(f'§3 — Three-way verdict (total_steps={total_steps})')
    print('=' * 92)

    treatment_h = _ddqn_hypothesis()
    baseline_h = _vanilla_hypothesis()
    treatment = [r for r in runs if r.arm_key == treatment_h.arm_key()]
    baseline = [r for r in runs if r.arm_key == 'baseline']
    print(f'  treatment={len(treatment)}  baseline={len(baseline)}')

    verdict = hypothesis_subgraph_verdict(
        treatment_h, treatment, baseline,
        pair_by=('seed',), group_by='env_name',
        baseline_h=baseline_h,
    )
    # §3 verdict pattern, expressed in paper-narrative shape:
    # mechanism (intervention edge with target=_MECHANISM),
    # outcome (intervention edge with target=_OUTCOME), and
    # coupling (the link from _MECHANISM → _OUTCOME). The
    # framework no longer encodes these labels — paper-narrative
    # naming is a substrate concern.
    pattern = (
        verdict.verdict_at(_MECHANISM),
        verdict.verdict_at(_OUTCOME),
        verdict.bridge_results[(_MECHANISM, _OUTCOME)].verdict
        if (_MECHANISM, _OUTCOME) in verdict.bridge_results
        else Verdict.POWER_INSUFFICIENT,
    )
    print(
        f'  §3 pattern (mechanism, outcome, coupling): '
        f'{tuple(v.value for v in pattern)}'
    )
    for edge in treatment_h.edges:
        br = verdict.bridge_results.get((edge.source, edge.target))
        if br is None:
            continue
        role_label = (
            'intervention' if edge.intervention is not None
            else 'coupling'
        )
        if edge.target in verdict.comparison_rows and edge.intervention is not None:
            row = verdict.comparison_rows[edge.target]
            g = row.effect_size_g if row.effect_size_g is not None else float('nan')
            i2 = row.pooled.I2 if row.pooled is not None else float('nan')
            print(
                f'    {role_label:<12} → {edge.target!r:<35} '
                f'verdict={br.verdict.value:<22} g={g:+.3f}  I²={i2:.3f}'
            )
        else:
            rho = br.stats.get('rho')
            p = br.stats.get('pvalue')
            n = br.stats.get('n_groups')
            r_v = float(rho) if isinstance(rho, (int, float)) else float('nan')
            p_v = float(p) if isinstance(p, (int, float)) else float('nan')
            n_v = int(n) if isinstance(n, int) else 0
            print(
                f'    {role_label:<12} {edge.source!r:<22} → '
                f'{edge.target!r:<35} '
                f'verdict={br.verdict.value:<22} '
                f'r={r_v:+.3f}  p={p_v:.3f}  n={n_v}'
            )

    g = verdict.graph
    cg_path = output_dir / f'causal_graph_{total_steps}.txt'
    with cg_path.open('w') as f:
        _ = f.write(f'# Typed CausalGraph for total_steps={total_steps}\n')
        _ = f.write(f'nodes: {sorted(g.nodes)}\n')
        for ge in g.edges:
            meta = ge.metadata
            _ = f.write(
                f'{ge.source!r} -> {ge.target!r}  '
                f'tier={meta.tier.name}  '
                f'level={meta.evidentiary_level}  '
                f'name={meta.bridge_name}\n'
            )
    print(f'  CausalGraph dump: {cg_path}')

    # Hasselt 2016-style descriptives. The original DDQN paper
    # reports per-game improvement counts and median improvement
    # across games — not pooled effect-size with random-effects
    # CIs. Reproducing these lets reviewers compare our findings
    # to Hasselt's reporting conventions directly.
    for role_label, target_path in (
        ('mechanism', _MECHANISM), ('outcome', _OUTCOME),
    ):
        matched_edges = treatment_h.edges_by_target(target_path)
        intervention_matches = tuple(
            e for e in matched_edges if e.intervention is not None
        )
        if not intervention_matches:
            continue
        edge = intervention_matches[0]
        row = verdict.comparison_rows.get(edge.target)
        if row is None:
            continue
        gs_list = row.per_group
        finite_gs = [
            gs.effect_size_g for gs in gs_list
            if gs.effect_size_g is not None
            and not math.isnan(gs.effect_size_g)
        ]
        if not finite_gs:
            continue
        n_total = len(finite_gs)
        # For mechanism, predicted_direction is 'a_lt_b' (DDQN
        # reduces gap → negative g preferred). For outcome,
        # 'a_gt_b' (DDQN increases return → positive g preferred).
        pred_sign = -1 if edge.predicted_direction == 'a_lt_b' else 1
        n_in_predicted_direction = sum(
            1 for v in finite_gs
            if (v < 0 if pred_sign == -1 else v > 0)
        )
        median_g = sorted(finite_gs)[n_total // 2]
        sign_pred = 'g<0' if pred_sign == -1 else 'g>0'
        print(
            f'  Hasselt-descriptive [{role_label}]: '
            f'{n_in_predicted_direction}/{n_total} envs in predicted '
            f'direction ({sign_pred}); median g = {median_g:+.3f}'
        )
    return verdict


def _section_4_pooled_pc(
    df: pl.DataFrame,
    total_steps: int,
    *,
    alpha: float = 0.05,
) -> None:
    """§4 — pooled conservative-PC stratified by env_name."""
    print()
    print('=' * 92)
    print(f'§4 — Pooled conservative-PC (stratify_by=env_name, '
          f'total_steps={total_steps})')
    print('=' * 92)

    pc_df = df.drop_nulls(subset=list(_PC_VARIABLES))
    for v in _PC_VARIABLES:
        if pc_df[v].dtype.is_float():
            pc_df = pc_df.filter(~pl.col(v).is_nan())
    if pc_df.height < 30:
        print(f'  {pc_df.height} rows after NaN-filter — skipping')
        return
    adj = discover_adjacency(
        pc_df, variables=list(_PC_VARIABLES),
        alpha=alpha, max_conditioning=1,
        stratify_by='env_name',
    )
    print(f'  surviving edges ({len(adj.edges)}):')
    for edge in sorted(adj.edges, key=lambda e: tuple(sorted(e))):
        a, b = sorted(edge)
        print(f'    {a:<32} — {b}')
    outcome_neighbours = frozenset(
        v for edge in adj.edges if _OUTCOME in edge
        for v in edge if v != _OUTCOME
    )
    print(
        f'  {_OUTCOME}-neighbours: '
        f'{sorted(outcome_neighbours) if outcome_neighbours else "(none)"}'
    )


def _section_5_within_env_pearson(
    df: pl.DataFrame,
    total_steps: int,
) -> None:
    """§5 — within-env Pearson r of mediator catalog vs outcome."""
    print()
    print('=' * 92)
    print(f'§5 — Within-env Pearson r vs {_OUTCOME} '
          f'(total_steps={total_steps})')
    print('=' * 92)

    envs: list[str] = sorted(
        cast('list[str]', df['env_name'].unique().to_list()),
    )
    header = f'{"env":<26}' + ''.join(
        f'{m.removeprefix("mediator."):<14}' for m in _MEDIATORS
    )
    print(header)
    for env in envs:
        env_df = df.filter(pl.col('env_name') == env)
        if env_df.height < 5:
            continue
        outcome = np.asarray(env_df[_OUTCOME].to_list(), dtype=np.float64)
        cells: list[str] = []
        for m in _MEDIATORS:
            mediator_arr = np.asarray(env_df[m].to_list(), dtype=np.float64)
            r, _ = _pearson_safe(mediator_arr, outcome)
            if math.isnan(r):
                cells.append(f'{"nan":<14}')
            else:
                marker = '*' if abs(r) > 0.5 else ' '
                cells.append(f'{r:>+.2f}{marker}        '[:14])
        print(f'{env:<26}' + ''.join(cells))


def _section_6_per_env_pc(
    df: pl.DataFrame,
    total_steps: int,
    *,
    alpha: float = 0.05,
) -> None:
    """§6 — per-env conservative-PC + regime classification."""
    print()
    print('=' * 92)
    print(f'§6 — Per-env PC + regime classification '
          f'(total_steps={total_steps})')
    print('=' * 92)

    pairs: list[tuple[str, frozenset[str]]] = []
    envs: list[str] = sorted(
        cast('list[str]', df['env_name'].unique().to_list()),
    )
    for env in envs:
        env_df = df.filter(pl.col('env_name') == env)
        if env_df.height < 5 or env_df['arm_ddqn'].n_unique() < 2:
            continue
        pc_df = env_df.drop_nulls(subset=list(_PC_VARIABLES))
        for v in _PC_VARIABLES:
            if pc_df[v].dtype.is_float():
                pc_df = pc_df.filter(~pl.col(v).is_nan())
        if pc_df.height < 5 or pc_df['arm_ddqn'].n_unique() < 2:
            continue
        # `Series.std()` returns `float | timedelta | None` per
        # the stubs; we only call this on float-dtyped columns
        # (the `.dtype.is_float()` guard above), so the timedelta
        # branch is unreachable. Narrow with per-line ignore.
        constant_seen = False
        for v in _PC_VARIABLES:
            if not pc_df[v].dtype.is_float():
                continue
            s = pc_df[v].std() or 0.0
            if float(s) == 0.0:  # pyright: ignore[reportArgumentType]
                constant_seen = True
                break
        if constant_seen:
            continue
        adj = discover_adjacency(
            pc_df, variables=list(_PC_VARIABLES),
            alpha=alpha, max_conditioning=1,
        )
        outcome_neighbours = frozenset(
            v for edge in adj.edges if _OUTCOME in edge
            for v in edge if v != _OUTCOME
        )
        pairs.append((env, outcome_neighbours))

    surviving = [(env, ns) for env, ns in pairs if ns]
    print(
        f'  {len(surviving)} of {len(pairs)} testable envs surface '
        f'>=1 {_OUTCOME}-neighbour'
    )
    for env, ns in pairs:
        ns_str = (
            ', '.join(sorted(n.removeprefix('mediator.') for n in ns))
            if ns else '(none)'
        )
        print(f'    {env:<26} -> {ns_str}')
    print()
    print('  mediator-frequency table (across surviving envs):')
    counter: Counter[str] = Counter()
    for _, ns in surviving:
        for n in ns:
            counter[n] += 1
    for n, count in counter.most_common():
        label = (
            n.removeprefix('mediator.') if n.startswith('mediator.') else n
        )
        print(f'    {label:<24} surviving in {count} env(s)')


def _section_7_meta_regression(
    verdict: HypothesisVerdict[Mapping[str, object]],
    total_steps: int,
    covariate_fn: Callable[[object], Mapping[str, float]],
    *,
    alpha: float = 0.05,
    covariate_set_name: str = 'minimal',
) -> None:
    """§7 — meta-regression on per-env mechanism/outcome g over
    env-metadata covariates. Empirical scope predictor."""
    print()
    print('=' * 92)
    print(f'§7 — Meta-regression on per-env effect sizes '
          f'(total_steps={total_steps}, covariates={covariate_set_name})')
    print('=' * 92)

    for role, target_path in (
        ('mechanism', _MECHANISM), ('outcome', _OUTCOME),
    ):
        row = verdict.comparison_rows.get(target_path)
        if row is None:
            continue
        observations: list[StratumObservation] = []
        for gs in row.per_group:
            if gs.effect_size_g is None or gs.se is None:
                continue
            if math.isnan(gs.effect_size_g) or math.isnan(gs.se):
                continue
            if gs.se <= 0.0:
                continue
            covs = covariate_fn(gs.group_value)
            if any(
                isinstance(v, float) and math.isnan(v)
                for v in covs.values()
            ):
                # NaN in any covariate (e.g., baseline mediator
                # was nan for this env) → drop the stratum;
                # otherwise meta_regression's design matrix
                # carries the NaN through to a numerical crash.
                continue
            observations.append(StratumObservation(
                stratum_id=gs.group_value,
                g=gs.effect_size_g,
                se=gs.se,
                covariates=covs,
            ))
        n_covariates = len(covariate_fn('CartPole-v1'))  # any env
        min_n = n_covariates + 2
        if len(observations) < min_n:
            print(
                f'  {role:<10}: n_observations={len(observations)} '
                f'< {min_n} (n_covariates+2) — skipping'
            )
            continue
        try:
            result = meta_regression(observations, alpha=alpha)
        except ValueError as e:
            print(f'  {role:<10}: meta_regression failed — {e}')
            continue
        print(
            f'  {role:<10}: n_strata={result.n_strata}  '
            f'R²={result.r_squared:.3f}  intercept={result.intercept:+.3f}'
        )
        print(
            f'    {"covariate":<26} {"coef":>8} {"ci_lo":>8} '
            f'{"ci_hi":>8} {"p":>6} {"sig":>4}'
        )
        for c in result.coefficients:
            sig = '***' if c.is_significant else ''
            print(
                f'    {c.name:<26} {c.coefficient:>+8.3f} '
                f'{c.ci_lo:>+8.3f} {c.ci_hi:>+8.3f} '
                f'{c.p_value:>6.3f} {sig:>4}'
            )
        if result.cleavage_axes:
            print(f'    cleavage axes: {list(result.cleavage_axes)!r}')
        else:
            print(f'    no significant cleavage at alpha={alpha}')


def _section_d1_pc_depth_robustness(
    df: pl.DataFrame,
    total_steps: int,
    *,
    alpha: float = 0.05,
) -> None:
    """D1 — depth-1 vs depth-2 PC edge-set diff."""
    print()
    print('=' * 92)
    print(f'D1 — PC depth robustness (total_steps={total_steps})')
    print('=' * 92)

    pc_df = df.drop_nulls(subset=list(_PC_VARIABLES))
    for v in _PC_VARIABLES:
        if pc_df[v].dtype.is_float():
            pc_df = pc_df.filter(~pl.col(v).is_nan())
    if pc_df.height < 30:
        print(f'  {pc_df.height} rows after NaN-filter — skipping')
        return
    diff = compare_pc_depths(
        pc_df, variables=list(_PC_VARIABLES),
        alpha=alpha, depths=(1, 2),
        stratify_by='env_name',
    )
    print(
        f'  depth-1 edges: {len(diff.edges_low)}  '
        f'depth-2 edges: {len(diff.edges_high)}  '
        f'common: {len(diff.common)}  '
        f'depth-1-only: {len(diff.low_only)}  '
        f'depth-2-only: {len(diff.high_only)}'
    )
    if diff.low_only:
        print('  edges killed by depth-2 conditioning:')
        for edge in sorted(diff.low_only, key=lambda e: tuple(sorted(e))):
            a, b = sorted(edge)
            print(f'    {a:<32} — {b}')
    if diff.high_only:
        print('  edges only at depth-2 (rare; usually re-validation):')
        for edge in sorted(diff.high_only, key=lambda e: tuple(sorted(e))):
            a, b = sorted(edge)
            print(f'    {a:<32} — {b}')


def _section_d2_kfold_cv(
    verdict: HypothesisVerdict[Mapping[str, object]],
    total_steps: int,
    covariate_fn: Callable[[object], Mapping[str, float]],
    *,
    k_folds: int = 5,
    alpha: float = 0.05,
    seed: int = 0,
    covariate_set_name: str = 'minimal',
) -> None:
    """D2 — k-fold CV on §7's coefficients."""
    print()
    print('=' * 92)
    print(f'D2 — k-fold CV on §7 coefficients '
          f'(total_steps={total_steps}, k={k_folds}, '
          f'covariates={covariate_set_name})')
    print('=' * 92)

    for role, target_path in (
        ('mechanism', _MECHANISM), ('outcome', _OUTCOME),
    ):
        row = verdict.comparison_rows.get(target_path)
        if row is None:
            continue
        observations: list[StratumObservation] = []
        for gs in row.per_group:
            if gs.effect_size_g is None or gs.se is None:
                continue
            if math.isnan(gs.effect_size_g) or math.isnan(gs.se):
                continue
            if gs.se <= 0.0:
                continue
            covs = covariate_fn(gs.group_value)
            if any(
                isinstance(v, float) and math.isnan(v)
                for v in covs.values()
            ):
                # NaN in any covariate (e.g., baseline mediator
                # was nan for this env) → drop the stratum;
                # otherwise meta_regression's design matrix
                # carries the NaN through to a numerical crash.
                continue
            observations.append(StratumObservation(
                stratum_id=gs.group_value,
                g=gs.effect_size_g,
                se=gs.se,
                covariates=covs,
            ))
        n_covariates = len(covariate_fn('CartPole-v1'))
        min_n = k_folds + n_covariates + 2
        if len(observations) < min_n:
            print(
                f'  {role:<10}: n={len(observations)} too small for '
                f'k={k_folds} CV with {n_covariates} covariates — skipping'
            )
            continue
        try:
            cv = cross_validate_meta_regression(
                observations, k_folds=k_folds, alpha=alpha, seed=seed,
            )
        except ValueError as e:
            print(f'  {role:<10}: CV failed — {e}')
            continue
        print(f'  {role:<10}: {len(cv.per_fold)} folds')
        print(
            f'    {"covariate":<26} {"sign-consist":>14} '
            f'{"mean":>8} {"std":>8}'
        )
        for name in sorted(cv.sign_consistency.keys()):
            consistency = cv.sign_consistency[name]
            mean, std = cv.coefficient_stability[name]
            print(
                f'    {name:<26} {consistency:>14.2f} '
                f'{mean:>+8.3f} {std:>8.3f}'
            )


# ============ Driver ============

def _add_arm_indicator(df: pl.DataFrame) -> pl.DataFrame:
    """Add `arm_ddqn ∈ {0, 1}` column for PC discovery's
    arm-as-variable convention."""
    return df.with_columns(
        (pl.col('intervention_name') == 'ddqn')
        .cast(pl.Int64).alias('arm_ddqn'),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument(
        '--runs-path',
        type=Path,
        default=_DEFAULT_RUNS,
        help='Path to runs_with_mediators.parquet.',
    )
    _ = parser.add_argument(
        '--total-steps',
        type=int,
        default=None,
        help='Filter to one total_steps grid point (e.g. 50000 or 200000). '
             'Default: run for every grid point in the corpus.',
    )
    _ = parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('/tmp/paper_full_range'),
        help='Where to write per-run artifacts (CausalGraph dumps).',
    )
    _ = parser.add_argument(
        '--alpha', type=float, default=0.05,
        help='Significance level for PC + meta-regression.',
    )
    _ = parser.add_argument(
        '--k-folds', type=int, default=5,
        help='k for D2 cross-validation.',
    )
    _ = parser.add_argument(
        '--seed', type=int, default=0,
        help='RNG seed for D2 fold assignment.',
    )
    _ = parser.add_argument(
        '--covariates',
        choices=('minimal', 'full', 'class3', 'mixed'),
        default='minimal',
        help='Covariate set for §7 + D2. '
             'minimal = 4 exogenous env-feature covariates '
             '(reward_range, horizon, family_minatar, '
             'family_bsuite); '
             'full = 10 exogenous env-feature covariates '
             '(adds n_actions + family_bandit/misc + 3 regime '
             'flags; often singular for outcome-side n=15); '
             'class3 = 4 endogenous per-env baseline mediator '
             'aggregates (vanilla means of q_gap_late, '
             'td_residual_late, greedy_match_late, '
             'learning_curve_auc); '
             'mixed = minimal + class3 (8 covariates).',
    )
    args = parser.parse_args()
    runs_path = cast(Path, args.runs_path)
    output_dir = cast(Path, args.output_dir)
    total_steps_filter: int | None = cast('int | None', args.total_steps)
    alpha = cast(float, args.alpha)
    k_folds = cast(int, args.k_folds)
    seed = cast(int, args.seed)
    covariate_set: CovariateSet = cast(CovariateSet, args.covariates)

    if not runs_path.exists():
        print(f'corpus not found: {runs_path}')
        sys.exit(1)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pl.read_parquet(runs_path)
    if 'arm_key' not in df.columns:
        print(
            f'{runs_path} has no `arm_key` column — '
            f'run migrate_runs_inject_arm_key.py first'
        )
        sys.exit(1)

    runs: list[RunRow] = read_runrows(runs_path)
    df_with_arm = _add_arm_indicator(df)

    grid_points: list[int] = sorted(
        df['total_steps'].unique().to_list()
        if 'total_steps' in df.columns else [-1]
    )
    if total_steps_filter is not None:
        if total_steps_filter not in grid_points:
            print(
                f'requested total_steps={total_steps_filter} not in '
                f'corpus {grid_points!r}'
            )
            sys.exit(1)
        grid_points = [total_steps_filter]

    print(f'corpus: {len(runs)} runs, total_steps grid: {grid_points}')

    for total_steps in grid_points:
        print()
        print('#' * 92)
        print(f'# total_steps = {total_steps}')
        print('#' * 92)

        if 'total_steps' in df.columns:
            df_step = df_with_arm.filter(
                pl.col('total_steps') == total_steps,
            )
            runs_step = [
                r for r in runs
                if r.measurements.get('total_steps') == total_steps
            ]
        else:
            df_step = df_with_arm
            runs_step = runs

        if not runs_step:
            print(f'  no runs for total_steps={total_steps} — skipping')
            continue

        # Per-grid-point covariate fn: class3/mixed sets read
        # baseline mediator aggregates from the cells *for this
        # total_steps*. Rebuilding per loop avoids cross-horizon
        # mixing (vanilla q_gap_late at 50k != 200k).
        covariate_fn = _build_covariate_fn(covariate_set, runs_step)

        verdict = _section_3_three_way(
            runs_step, total_steps, output_dir,
        )
        _section_4_pooled_pc(df_step, total_steps, alpha=alpha)
        _section_5_within_env_pearson(df_step, total_steps)
        _section_6_per_env_pc(df_step, total_steps, alpha=alpha)
        _section_7_meta_regression(
            verdict, total_steps, covariate_fn,
            alpha=alpha, covariate_set_name=covariate_set,
        )
        _section_d1_pc_depth_robustness(df_step, total_steps, alpha=alpha)
        _section_d2_kfold_cv(
            verdict, total_steps, covariate_fn,
            k_folds=k_folds, alpha=alpha, seed=seed,
            covariate_set_name=covariate_set,
        )


if __name__ == '__main__':
    main()
    sys.exit(0)
