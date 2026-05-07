"""Polyak-τ intervention sweep analysis: does do(τ) at fixed sync
period prove staleness causality?

Sweep design (`experiments/configs/polyak_tau_intervention.yaml`):
  τ ∈ {0.001, 0.01, 0.1, 1.0}
  envs: Acrobot, Breakout-MinAtar, FourRooms, MountainCar
        (+ SpaceInvaders only on τ=1.0, partial)
  arms: baseline + DDQN, both with `polyak_update`
  sync_period FIXED at 100, 30 seeds per cell
  total: 1440 designed cells (4 τ × 6 envs × 30 seeds × 2 arms);
  3 sub-sweeps with traces: 4 envs × 30 × 2 = 240 cells each.

Pearl-rung-2 logic. `target_staleness_late` is collinear with
`sync_period` in the periodic-copy regime (ρ ≈ +0.96 within
sync). do(τ) at fixed sync is the canonical break: τ directly
controls how fast the target follows online (high τ → low
staleness, low τ → high staleness), with no other knob touched.
If staleness causes outcome, the τ → outcome curve is monotone
within env. If outcome causes staleness or both are downstream
confounded, the curve has a different shape (or none).

The analysis proceeds in three checks:

1. **Staleness IS varying with τ** — sanity check that the
   intervention bites. If staleness doesn't change with τ,
   the sweep didn't actually intervene.

2. **Per-env outcome curve** — does each env's mean
   `eval_best_burst_mean` move monotonically with log τ? Sign
   of slope reveals the env's response: canonical
   "less-staleness → better-outcome" gives positive slope on
   linear-τ axis; if slope is null or wrong-signed, the env
   isn't responding to the staleness channel.

3. **DDQN-vs-baseline g_outcome by τ** — does DDQN's marginal
   benefit (g_outcome) shift with τ? Under the staleness
   mediation hypothesis (CLAIM 13), DDQN's bias correction
   compensates for staleness; with τ=1.0 (no staleness) DDQN's
   benefit should shrink (bias correction has nothing to
   correct against), with τ low (high staleness) DDQN's
   benefit should be largest.
"""
from __future__ import annotations

import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import json
import math
from pathlib import Path

import numpy as np
import polars as pl

import corroborate_rl.dqn.measurables  # register
from corroborate.runner.runner import _join_required_traces, _measurable_signature
from corroborate.corpus.measurements import build_measurements, load_measurements
from corroborate.measurables import get_registered

DDQN = 'bootstrap=partial(Claim:bootstrap;greedification=Claim:double_greedify)'


def hedges_g(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Paired Hedges' g + small-sample SE. Returns (g, se)."""
    if len(x) != len(y) or len(x) < 3:
        return float('nan'), float('nan')
    d = x - y
    n = len(d)
    sd = float(d.std(ddof=1))
    if sd == 0:
        return float('nan'), float('nan')
    g_raw = float(d.mean()) / sd
    # Hedges' correction
    j = 1.0 - 3.0 / (4.0 * (n - 1) - 1.0) if n > 2 else 1.0
    g = j * g_raw
    se = math.sqrt(1.0 / n + g_raw ** 2 / (2.0 * n))
    return g, se


def main() -> None:
    base = Path('experiments/data/polyak_tau_intervention')
    sub_sweeps = sorted([p for p in base.iterdir() if p.is_dir()])

    required = (
        'target_staleness_late', 'target_staleness_early',
        'eval_best_burst_mean', 'jensen_gap',
        'effective_horizon', 'bootstrap_fraction',
        'env_reward_polarity',
    )
    trace_reads = set()
    for n in required:
        m = get_registered(n)
        if m and hasattr(m, 'reads'):
            trace_reads.update(m.reads)

    # Load each sub-sweep + compute measurables.
    panels: list[pl.DataFrame] = []
    for sub in sub_sweeps:
        runs_path = sub / 'runs.parquet'
        traces_path = sub / 'traces.parquet'
        if not runs_path.exists():
            print(f'{sub.name}: no runs.parquet — skip', flush=True)
            continue
        runs = pl.read_parquet(runs_path)
        if traces_path.exists():
            df = _join_required_traces(runs, traces_path, frozenset(trace_reads))
            build_measurements(
                sub, required=required, runs_df=df,
                measurable_signature_fn=_measurable_signature,
            )
            loaded = load_measurements(sub, columns=list(required))
            present = [c for c in required if c in loaded.columns]
            collide = [c for c in present if c in runs.columns]
            if collide:
                runs = runs.drop(collide)
            df_full = runs.join(loaded.select(['id', *present]), on='id', how='left')
        else:
            # No traces: can't compute staleness, but eval_best_burst_mean
            # is a per-cell scalar already in runs.parquet. Build a
            # staleness-less panel so this sub-sweep contributes to
            # CHECK 2 / CHECK 3 / CHECK 4 (outcome curves) but not
            # CHECK 1 (staleness sanity).
            print(f'{sub.name}: NO traces — outcome-only contribution', flush=True)
            df_full = runs
        # Tag the τ value on each cell.
        tau_v = float(df_full['target_sync.tau'][0])
        df_full = df_full.with_columns(pl.lit(tau_v).alias('tau'))
        panels.append(df_full)
        print(
            f'{sub.name}: tau={tau_v}, '
            f'cells={df_full.height}, '
            f'staleness_finite={int(df_full["target_staleness_late"].is_finite().sum())}',
            flush=True,
        )

    if not panels:
        print('No panels built — abort.', flush=True)
        return

    df = pl.concat(panels, how='diagonal_relaxed')
    print(f'\\nMerged: {df.height} cells across {df["tau"].n_unique()} τ values', flush=True)

    # ====================================================================
    # CHECK 1 — staleness IS varying with τ (sanity)
    # ====================================================================
    print()
    print('=== CHECK 1: target_staleness_late by τ (sanity) ===\n')
    print(f'{"env":<24} {"arm":<10}', end='')
    taus = sorted(df['tau'].unique().to_list())
    for tau_v in taus:
        print(f'  τ={tau_v}'.ljust(14), end='')
    print()
    print('-' * 100)

    for env in sorted(df['env_name'].unique()):
        for arm_label, arm in (('baseline', 'baseline'), ('ddqn', DDQN)):
            sub = df.filter((pl.col('env_name') == env) & (pl.col('arm_key') == arm))
            print(f'{env:<24} {arm_label:<10}', end='')
            for tau_v in taus:
                cell = sub.filter(pl.col('tau') == tau_v)
                if cell.height == 0 or 'target_staleness_late' not in cell.columns:
                    print(f'{"-":>14}', end='')
                else:
                    mean_v = cell['target_staleness_late'].drop_nans().mean()
                    if mean_v is None:
                        print(f'{"-":>14}', end='')
                    else:
                        print(f'{float(mean_v):>10.5f}    ', end='')
            print(flush=True)

    # ====================================================================
    # CHECK 2 — outcome by τ per (env, arm)
    # ====================================================================
    print()
    print('=== CHECK 2: eval_best_burst_mean by τ ===\n')
    print(f'{"env":<24} {"arm":<10}', end='')
    for tau_v in taus:
        print(f'  τ={tau_v}'.ljust(14), end='')
    print()
    print('-' * 100)

    for env in sorted(df['env_name'].unique()):
        for arm_label, arm in (('baseline', 'baseline'), ('ddqn', DDQN)):
            sub = df.filter((pl.col('env_name') == env) & (pl.col('arm_key') == arm))
            print(f'{env:<24} {arm_label:<10}', end='')
            for tau_v in taus:
                cell = sub.filter(pl.col('tau') == tau_v)
                if cell.height == 0:
                    print(f'{"":>14}', end='')
                else:
                    mean_out = float(cell['eval_best_burst_mean'].drop_nans().mean())
                    print(f'{mean_out:>10.4f}    ', end='')
            print(flush=True)

    # ====================================================================
    # CHECK 3 — DDQN's g_outcome by τ
    # ====================================================================
    print()
    print('=== CHECK 3: DDQN paired g(outcome) by τ ===\n')
    print(f'{"env":<24}', end='')
    for tau_v in taus:
        print(f'  τ={tau_v}'.ljust(18), end='')
    print()
    print('-' * 100)

    g_panel = []
    for env in sorted(df['env_name'].unique()):
        print(f'{env:<24}', end='')
        env_row = {'env': env}
        for tau_v in taus:
            sub = df.filter((pl.col('env_name') == env) & (pl.col('tau') == tau_v))
            v = sub.filter(pl.col('arm_key') == 'baseline').sort('seed')
            d = sub.filter(pl.col('arm_key') == DDQN).sort('seed')
            # Pair on seed — within (env, τ) only
            paired = v.select(['seed', 'eval_best_burst_mean']).rename(
                {'eval_best_burst_mean': 'out_v'}
            ).join(
                d.select(['seed', 'eval_best_burst_mean']).rename(
                    {'eval_best_burst_mean': 'out_d'}
                ),
                on='seed', how='inner',
            ).filter(
                pl.col('out_v').is_finite() & pl.col('out_d').is_finite()
            )
            if paired.height < 5:
                print(f'{"-":>16}', end='  ')
                env_row[f'g_tau_{tau_v}'] = None
                continue
            x = paired['out_d'].to_numpy()
            y = paired['out_v'].to_numpy()
            n = paired.height
            # Special case: τ=1.0 collapses both arms (target=online
            # every step → polyak_update is identity). DDQN's bias
            # correction (Q_target(s', argmax_a Q_online(s', a)) vs
            # max_a Q_target) is a no-op because Q_target ≡ Q_online,
            # so the two arms produce identical seeded trajectories.
            # All-zero diff → g=0 by construction; the analysis path
            # would NaN-out, so set explicitly.
            if (x == y).all():
                g, se = 0.0, 0.0
                t = 0.0
                print(f'{g:>+.3f}±{se:.3f} (n={n}, identical)', end='  ')
            else:
                g, se = hedges_g(x, y)
                t = g / se if se > 0 else float('nan')
                print(f'{g:>+.3f}±{se:.3f} (n={n})', end='  ')
            env_row[f'g_tau_{tau_v}'] = float(g)
            env_row[f'se_tau_{tau_v}'] = float(se)
            env_row[f't_tau_{tau_v}'] = float(t) if not math.isnan(t) else None
        print(flush=True)
        g_panel.append(env_row)

    # ====================================================================
    # CHECK 4 — Δ_outcome vs τ slope per env (does DDQN's benefit move
    # monotonically with τ?)
    # ====================================================================
    print()
    print('=== CHECK 4: g_outcome ~ log(τ) slope per env (mediation prediction) ===\n')
    print(f'{"env":<24} {"slope_g_logτ":>15} {"R²":>8} {"|g(τ_min)|":>11} {"|g(τ_max)|":>11} {"reading":<40}')
    print('-' * 100)

    for row in g_panel:
        env = row['env']
        gs = []
        log_taus = []
        for tau_v in taus:
            g = row.get(f'g_tau_{tau_v}')
            if g is not None and not math.isnan(g):
                gs.append(g)
                log_taus.append(math.log10(tau_v))
        if len(gs) < 3:
            print(f'{env:<24} {"":>15} {"":>8} insufficient τ points')
            continue
        gs_arr = np.asarray(gs)
        lt = np.asarray(log_taus)
        # Linear regression slope
        if gs_arr.std() == 0:
            slope = 0.0
            r2 = 0.0
        else:
            n = len(lt)
            cov = float(np.cov(lt, gs_arr, ddof=1)[0, 1])
            var_x = float(np.var(lt, ddof=1))
            slope = cov / var_x if var_x > 0 else float('nan')
            mean_y = float(gs_arr.mean())
            ss_tot = float(((gs_arr - mean_y) ** 2).sum())
            y_pred = slope * (lt - lt.mean()) + mean_y
            ss_res = float(((gs_arr - y_pred) ** 2).sum())
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')

        # Mediation prediction: as τ increases (less staleness), DDQN's
        # benefit DECREASES (bias correction has nothing to correct).
        # That predicts NEGATIVE slope of g vs log(τ). At τ=1.0 the
        # arms collapse algorithmically → g=0 by construction; that
        # data point is included in the slope to anchor the dose-
        # response curve at zero.
        # |g(τ_min)| meaningful at τ=0.001 (high staleness, expected
        # max DDQN benefit); |g(τ_max)| at τ=1.0 (no staleness,
        # expected zero).
        g_min = abs(gs[0])
        g_max = abs(gs[-1])
        # Mediation HELD: monotone decay from a substantive
        # |g(τ_min)| to ~0 |g(τ_max)|, with high goodness-of-fit.
        # Concretely: R² ≥ 0.7, slope ≤ -0.1 (negative dose-response),
        # g(τ_min) > 3·g(τ_max), AND g(τ_min) > 0.3 (large enough
        # to not be noise).
        if r2 >= 0.7 and slope <= -0.1 and g_min >= 0.3 and g_min > 3 * g_max:
            reading = 'MEDIATION HELD: g shrinks with τ↑'
        elif g_min < 0.2:
            reading = 'no DDQN effect at any τ'
        elif slope > 0.1:
            reading = 'DDQN benefit GROWS with τ↑ (anti-mediation)'
        else:
            reading = 'weak/noisy: τ may not be the channel'
        print(f'{env:<24} {slope:>+15.3f} {r2:>8.3f} {g_min:>11.3f} {g_max:>11.3f} {reading:<40}')

    # ====================================================================
    # Save panel
    # ====================================================================
    out = Path('experiments/findings/sync_curve_breakout/polyak_tau_panel.json')
    out.write_text(json.dumps(g_panel, indent=2, default=str))
    print(f'\nwrote: {out}', flush=True)


if __name__ == '__main__':
    main()
