"""Layer 5 companion — per-env, PER-BURST best mediator.

Hypothesis (user, 2026-05-28): each burst may have a DIFFERENT load-
bearing mediator. The L3b cell-level best-mediator pipeline picks
ONE mediator per env (highest absorption on a single scalar). At
per-burst granularity the answer may shift over training — early
bursts route through a Q-shape channel; mid-training through a
Bellman-residual channel; late bursts through state-coverage.

For each env, iterate through the canonical per-burst mediator
candidates (the framework's registered `_per_burst` measurables) and
run `dynamic_pc_adjacency` per candidate. At each burst, record:

  - which candidates d-separate arm→outcome at α=0.05
  - which has the highest absorption (1 − |ρ_partial| / |ρ_marginal|)
  - whether the winner is stable across the trajectory or shifts

Output:
  - Per-env figure: per-burst winner trajectory color-coded by
    mediator family (Bellman / Q-shape / policy / state-coverage)
  - Stability summary: for each env, how many bursts have a marginal
    arm→outcome edge, and across those bursts, how many DIFFERENT
    candidates win the burst.

Per-burst candidates auto-detected per env (handles MinAtar/Jumanji
corpora that lack a subset of the broad candidate space).
"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
from _common import (
    BASELINE_ARM, COLOR_HELPS, COLOR_HARMS, COLOR_NULL,
    ENV_ORDER, TREATMENT_ARM,
    env_label,
)

# Per-burst outcome. Switched back to the γ-discounted variant
# (`mc_return__mean_axis_-1`) on the rebuilt canonical cache —
# it's populated 440/440. The raw variant (`mc_return_raw__mean_axis_-1`)
# is absent in the rebuilt cache. At γ=0.99 the discounting effect
# on per-burst correlations is modest (γ^t fades over ~100 steps
# of evaluation), and the cross-env comparison is internally
# consistent since every env shares γ.
OUTCOME_PER_BURST_COL: str = 'mc_return__mean_axis_-1'

from corroborate.analyses.dynamic_mediation.pc_adjacency import (
    dynamic_pc_adjacency,
)


# Single source of truth: the shared canonical-panel loader (same
# cache + scope as L1-L5), rather than a script-local reader.
def _load_panel() -> pl.DataFrame:
    from _common import load_g099_canonical_panel
    return load_g099_canonical_panel()


OUT_PNG = SCRIPT_DIR.parent / 'figures' / '05b_dynamic_best_mediator.png'
OUT_CSV = SCRIPT_DIR.parent / 'figures' / '05b_dynamic_best_mediator.csv'

# Pass via --include-bias-soft-taut to put soft-tautological bias mediators
# (`mean_per_state_cumulative_bias_per_burst`, `normalized_bias_redq_per_burst`)
# back into the candidate pool. Default EXCLUDES them — they win by
# MC-input overlap, not by capturing a distinct mediation channel
# (see `feedback_tautology_audit_is_conservative`).
INCLUDE_BIAS_SOFT_TAUT: bool = '--include-bias-soft-taut' in sys.argv

# Candidate per-burst mediators, grouped by family for color coding.
CANDIDATES_BY_FAMILY: dict[str, tuple[str, ...]] = {
    'Bellman': (
        'bootstrap_gap_magnitude_per_burst',
        'bootstrap_disagree_rate_per_burst',
        'bootstrap_disagree_gap_conditional_per_burst',
        'greedy_match_per_burst',
    ),
    'Q-shape': (
        'q_argmax_margin_per_burst',
        'q_action_std_per_burst',
        'q_autocorr_per_burst',
        'q_lambda_a_per_burst',
    ),
    'Policy': (
        'argmax_entropy_per_burst',
        'state_conditional_argmax_entropy_per_burst',
    ),
    'State-coverage': (
        'state_hash_n_unique_per_burst',
        'state_hash_entropy_per_burst',
        'state_repeat_rate_window64_per_burst',
    ),
    'Bias (soft-taut)': (
        'mean_per_state_cumulative_bias_per_burst',
        'normalized_bias_redq_per_burst',
    ),
}
FAMILY_COLOR: dict[str, str] = {
    'Bellman': '#1f77b4',
    'Q-shape': '#d62728',
    'Policy': '#2ca02c',
    'State-coverage': '#9467bd',
    'Bias (soft-taut)': '#ff7f0e',
    '(no winner)': '#cccccc',
}


def _candidate_family(name: str) -> str:
    for fam, members in CANDIDATES_BY_FAMILY.items():
        if name in members:
            return fam
    return '(no winner)'


def _populated_candidates(sub: pl.DataFrame) -> tuple[str, ...]:
    out: list[str] = []
    for fam, fam_members in CANDIDATES_BY_FAMILY.items():
        if fam == 'Bias (soft-taut)' and not INCLUDE_BIAS_SOFT_TAUT:
            continue
        for c in fam_members:
            if c not in sub.columns:
                continue
            # Per-burst columns are list-typed; check that at least
            # one row has a non-empty list with finite values
            try:
                n_with = sub.select(pl.col(c).list.len().sum()).item()
                if n_with > 0:
                    out.append(c)
            except Exception:
                pass
    return tuple(out)


def main() -> None:
    df = _load_panel()
    print(f'panel: {df.height} cells')

    # Per-env: run dynamic_pc_adjacency per candidate, collect per-burst
    # results, then determine winner per burst.
    per_env_records: dict[str, dict[str, object]] = {}
    rows_csv: list[str] = []

    for env in ENV_ORDER:
        sub = df.filter(pl.col('env_name') == env)
        print(f'  {env_label(env):15s}  n_cells={sub.height}')
        if sub.height < 10:
            continue
        cands = _populated_candidates(sub)
        if not cands:
            print(f'  {env}: no populated per-burst candidates; skipping')
            continue

        # Run PC for each candidate; collect (rho_marginal, rho_partial,
        # p_marginal, p_conditional) per burst per candidate.
        cand_results: dict[str, object] = {}
        for c in cands:
            try:
                r = dynamic_pc_adjacency.fn(
                    sub, arm_field='arm_key',
                    mediator_per_burst=c,
                    outcome_per_burst=OUTCOME_PER_BURST_COL,
                    stratify_by=('env_name',),
                    min_n_per_burst=10,
                    alpha=0.05,
                )
                if not r:
                    if env == 'Asterix-MinAtar':
                        print(f'    [{env_label(env)}/{c}] empty result')
                    continue
                stratum = next(iter(r.keys()))
                cand_results[c] = r[stratum]
            except Exception as e:
                print(f'    [{env_label(env)}/{c}] {type(e).__name__}: {e}')

        if not cand_results:
            print(f'  {env}: PC failed for all candidates; skipping')
            continue
        # Burst axis from any candidate (they share the n_bursts axis)
        ref = next(iter(cand_results.values()))
        n_bursts = len(ref.p_marginal)

        # Per burst: identify the candidate that d-separates (if any),
        # picking the one with highest absorption when multiple qualify.
        per_burst_winner: list[str | None] = []
        per_burst_absorption: list[float] = []
        n_marg_bursts = 0
        for b in range(n_bursts):
            # Use ref's marginal-edge status; same across candidates since
            # depth-0 test is mediator-independent.
            p_m = ref.p_marginal[b]
            if (
                p_m is None
                or (isinstance(p_m, float) and np.isnan(p_m))
                or p_m >= 0.05
            ):
                per_burst_winner.append(None)
                per_burst_absorption.append(float('nan'))
                continue
            n_marg_bursts += 1
            best_cand: str | None = None
            best_absorb = -1.0
            for c, res in cand_results.items():
                p_c = res.p_conditional[b]
                rho_m = res.rho_marginal[b]
                rho_p = res.rho_partial[b]
                if (
                    p_c is None
                    or (isinstance(p_c, float) and np.isnan(p_c))
                    or p_c < 0.05  # mediator does NOT d-separate at this burst
                ):
                    continue
                if abs(rho_m) < 1e-9 or np.isnan(rho_m) or np.isnan(rho_p):
                    continue
                # No sign-flip: partial sign matches marginal OR partial near 0
                if (rho_m > 0) != (rho_p > 0) and abs(rho_p) > 0.1:
                    continue
                absorb = 1 - abs(rho_p) / abs(rho_m)
                if absorb > best_absorb:
                    best_absorb = absorb
                    best_cand = c
            per_burst_winner.append(best_cand)
            per_burst_absorption.append(best_absorb)

        # Count winner families across the burst trajectory
        from collections import Counter
        winner_count = Counter(
            (_candidate_family(w) if w else '(no winner)')
            for w in per_burst_winner
        )
        n_winning = sum(1 for w in per_burst_winner if w is not None)
        n_distinct = len({w for w in per_burst_winner if w is not None})

        per_env_records[env] = {
            'n_bursts': n_bursts,
            'n_marg_bursts': n_marg_bursts,
            'n_winning_bursts': n_winning,
            'n_distinct_winners': n_distinct,
            'winner_per_burst': per_burst_winner,
            'absorption_per_burst': per_burst_absorption,
            'family_counts': dict(winner_count),
            'candidates_used': len(cand_results),
        }
        print(f'  {env_label(env):15s}  n_marg={n_marg_bursts:3d}  '
              f'n_winning={n_winning:3d}  distinct={n_distinct}')

    # ─── figure: per-env strip showing winner trajectory ───
    envs = [e for e in ENV_ORDER if e in per_env_records]
    n = len(envs)
    fig, ax = plt.subplots(figsize=(14, max(3.5, 0.45 * n)))
    y = np.arange(n)
    # X-axis is NORMALISED to fraction-of-training (burst b of N bursts
    # → [b/N, (b+1)/N]) so envs with different burst counts (Snake 150,
    # most 50, some 20) are directly comparable — each env's strip spans
    # the full [0, 1] width regardless of its absolute burst count.
    for i, env in enumerate(envs):
        rec = per_env_records[env]
        winners = rec['winner_per_burst']
        nb = rec['n_bursts']
        w_frac = 1.0 / nb
        for b, w in enumerate(winners):
            fam = _candidate_family(w) if w else '(no winner)'
            color = FAMILY_COLOR[fam]
            ax.add_patch(plt.Rectangle(
                (b * w_frac, i - 0.35), w_frac * 0.95, 0.7,
                facecolor=color, edgecolor='white', linewidth=0.3,
                alpha=0.85 if w else 0.2,
            ))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_yticks(y)
    ax.set_yticklabels([
        f'{env_label(e)}  ({per_env_records[e]["n_winning_bursts"]}/'
        f'{per_env_records[e]["n_marg_bursts"]} winning, '
        f'{per_env_records[e]["n_distinct_winners"]} distinct, '
        f'{per_env_records[e]["n_bursts"]}b)'
        for e in envs
    ])
    ax.invert_yaxis()
    ax.set_xlabel('fraction of training  (burst index / n_bursts)', fontsize=10)
    ax.set_title(
        'Layer 5 companion: per-env, PER-BURST best mediator\n'
        'Color = winning mediator family at each training-fraction bin '
        '(x normalised across envs; total bursts in row label)\n'
        'White = no marginal arm→outcome edge, or no candidate d-separates',
        fontsize=10.5,
    )
    # Family legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=c, label=f)
        for f, c in FAMILY_COLOR.items() if f != '(no winner)'
    ]
    # Legend below the plot (under the x-label) so it doesn't
    # collide with the 3-line title.
    ax.legend(handles=handles, loc='upper center', fontsize=9,
              bbox_to_anchor=(0.5, -0.13), ncol=5, frameon=False)
    ax.set_title(ax.get_title(), pad=12)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=120, bbox_inches='tight')

    # ─── CSV ───
    with OUT_CSV.open('w') as f:
        f.write('env,n_bursts,n_marg_bursts,n_winning,n_distinct,'
                'family_counts,per_burst_winners\n')
        for env in envs:
            rec = per_env_records[env]
            fc = '|'.join(f'{k}:{v}' for k, v in sorted(rec['family_counts'].items()))
            pbw = '|'.join(
                _candidate_family(w).replace(' (soft-taut)', '') if w else '-'
                for w in rec['winner_per_burst']
            )
            f.write(f'{env_label(env)},{rec["n_bursts"]},'
                    f'{rec["n_marg_bursts"]},{rec["n_winning_bursts"]},'
                    f'{rec["n_distinct_winners"]},"{fc}","{pbw}"\n')

    print(f'saved → {OUT_PNG.name}, {OUT_CSV.name}')


if __name__ == '__main__':
    main()
