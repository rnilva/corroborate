"""Test: is polarity a measurement-frame property, or does it shape
DDQN's mechanism?

For each polarity-panel env with local traces, compute per-pair
(DDQN − vanilla):
  Δ_jens         — mech step (bias reduction)
  Δ_staleness    — proposed dominant mediator
  Δ_eff_h        — polarity-coupling channel
  Δ_outcome      — outcome change

Then aggregate to env-level mean, normalize by within-cell std for
scale-invariance, and correlate each cross-env with polarity.

Prediction (mechanism-blind hypothesis):
  - mean(Δ_jens / σ_jens_baseline) ⊥ polarity
  - mean(Δ_staleness / σ_stale_baseline) ⊥ polarity
  - mean(Δ_eff_h / σ_eff_h_baseline) ∝ polarity (by polarity definition)
  - mean(Δ_outcome / σ_outcome_baseline) ∝ polarity (downstream)

If predictions hold, polarity is purely a measurement-frame property
that projects DDQN's outcome benefit onto L-space. Mechanism is
polarity-invariant; staleness mediation should not depend on
polarity sign or magnitude.

If predictions fail (mech step polarity-correlated), polarity actually
shapes DDQN's bias-correction itself — a strictly stronger claim.
"""
from __future__ import annotations

import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import json
import math
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats
from corroborate_rl.dqn import measurables as _dqn_measurables  # noqa: F401
from corroborate.measurables import get_registered

DDQN = 'bootstrap=partial(Claim:bootstrap;greedification=Claim:double_greedify)'

# Per-env corpus + pair_keys + polarity (from prior panels).
ENV_SOURCES = [
    {
        'env': 'FourRooms-misc',
        'corpus_dir': Path('experiments/data/capacity_sweep_fourrooms'),
        'pair_keys': ('seed', 'replay.capacity'),
        'polarity': -0.864,  # from polarity_x_undamped_panel.json
    },
    {
        'env': 'Acrobot-v1',
        'corpus_dir': Path('experiments/data/expectile_3way'),
        'pair_keys': ('seed',),
        'polarity': -0.896,
    },
    {
        'env': 'MountainCar-v0',
        'corpus_dir': Path('experiments/data/expectile_3way'),
        'pair_keys': ('seed',),
        'polarity': -0.990,
    },
    {
        'env': 'Breakout-MinAtar',
        'corpus_dir': Path('experiments/data/minatar_1M'),
        'pair_keys': ('seed',),
        'polarity': +0.988,
        'arm_shards': True,  # minatar_1M has tmp/arm*__<env>__<arm>__traces.parquet
    },
    {
        'env': 'SpaceInvaders-MinAtar',
        'corpus_dir': Path('experiments/data/minatar_sync_curve_resume/ddqn_sync3k'),
        'pair_keys': ('seed',),
        'polarity': +0.026,  # SI sync=3000 — different regime than minatar_1M
    },
]


def _load_array(traces: pl.DataFrame, col: str) -> np.ndarray:
    list_len = traces[col].list.len()[0]
    casted = traces.select(pl.col(col).cast(pl.Array(pl.Float64, list_len)))
    return casted[col].to_numpy()


def _mean_late(arr2d: np.ndarray) -> np.ndarray:
    n = arr2d.shape[1]
    lo = int(0.5 * n)
    return arr2d[:, lo:].mean(axis=1)


def compute_per_cell(spec: dict) -> pl.DataFrame:
    """Per-cell (env, arm, seed) DataFrame with jens, eff_h, staleness,
    outcome scalars. Loads runs + traces from local corpus dir and
    handles arm-shard layouts."""
    corpus_dir = spec['corpus_dir']
    env = spec['env']

    runs = pl.read_parquet(corpus_dir / 'runs.parquet')
    runs = runs.filter(pl.col('env_name') == env)
    runs = runs.filter(pl.col('arm_key').is_in(['baseline', DDQN]))
    if 'arm_shards' in spec and spec['arm_shards']:
        # minatar_1M layout: tmp/arm{NNN}__{env}__{arm}__traces.parquet.
        # Polars' glob expansion stops on path special chars; use
        # plain Path.iterdir + suffix filtering instead.
        tmp = corpus_dir / 'tmp'
        env_shards = sorted(
            p for p in tmp.iterdir()
            if p.is_file() and p.suffix == '.parquet'
            and f'__{env}__' in p.name
            and p.name.endswith('__traces.parquet')
        )
        if not env_shards:
            print(f'  {env}: no trace shards found; skipping')
            return None
        traces_frames = []
        for shard in env_shards:
            tf = pl.read_parquet(shard, columns=[
                'id', 'mc_return', 'predicted_q_at_start', 'episode_length',
                'online_max_q_per_step', 'target_max_q_per_step',
            ])
            traces_frames.append(tf)
        traces = pl.concat(traces_frames, how='vertical')
    else:
        traces = pl.read_parquet(corpus_dir / 'traces.parquet', columns=[
            'id', 'mc_return', 'predicted_q_at_start', 'episode_length',
            'online_max_q_per_step', 'target_max_q_per_step',
        ])

    joined = runs.join(traces, on='id', how='inner')
    if len(joined) == 0:
        return None

    # Compute scalars from list-typed cols
    online_max = _load_array(joined, 'online_max_q_per_step')
    target_max = _load_array(joined, 'target_max_q_per_step')
    n_steps = online_max.shape[1]
    lo = int(0.5 * n_steps)
    abs_gap = np.abs(online_max[:, lo:] - target_max[:, lo:])
    denom = np.maximum.reduce([
        np.abs(online_max[:, lo:]), np.abs(target_max[:, lo:]),
        np.full_like(online_max[:, lo:], 1e-6),
    ])
    target_stale_late = (abs_gap / denom).mean(axis=1)

    # eval_best_burst_mean: max-over-bursts of mean-over-episodes
    ebbm = []
    jens = []
    eff_h = []
    bf = []
    for row in joined.iter_rows(named=True):
        mc = np.asarray(row['mc_return'], dtype=np.float64)
        if mc.ndim == 2:
            ebbm.append(float(mc.mean(axis=1).max()))
        else:
            ebbm.append(float('nan'))
        pq = np.asarray(row['predicted_q_at_start'], dtype=np.float64)
        if pq.size > 0 and mc.size > 0:
            jens.append(float(max(0.0, (pq - mc).mean())))
        else:
            jens.append(float('nan'))
        # episode_length per cell — derive bf from done-rate proxy = 1 - 1/L_mean
        el = np.asarray(row['episode_length'], dtype=np.float64).flatten()
        L_mean = float(el.mean()) if el.size > 0 else float('nan')
        bf_v = 1.0 - 1.0 / max(L_mean, 1.0)
        gamma_v = float(row.get('gamma', 0.99))
        bf.append(bf_v)
        eff_h.append(1.0 / max(1.0 - gamma_v * bf_v, 1e-9))

    df = joined.select(['id', 'arm_key'] + list(spec['pair_keys']))
    df = df.with_columns([
        pl.Series('eval_best_burst_mean', ebbm),
        pl.Series('jensen_gap', jens),
        pl.Series('effective_horizon', eff_h),
        pl.Series('bootstrap_fraction', bf),
        pl.Series('target_staleness_late', target_stale_late.tolist()),
    ])
    return df


def pair_deltas(per_cell: pl.DataFrame, pair_keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Pair vanilla + DDQN cells on `pair_keys`, return Δ vectors."""
    v = per_cell.filter(pl.col('arm_key') == 'baseline').select(
        list(pair_keys) + ['jensen_gap', 'effective_horizon', 'bootstrap_fraction',
                           'target_staleness_late', 'eval_best_burst_mean']
    ).rename({c: f'{c}_v' for c in (
        'jensen_gap', 'effective_horizon', 'bootstrap_fraction',
        'target_staleness_late', 'eval_best_burst_mean')})
    d = per_cell.filter(pl.col('arm_key') == DDQN).select(
        list(pair_keys) + ['jensen_gap', 'effective_horizon', 'bootstrap_fraction',
                           'target_staleness_late', 'eval_best_burst_mean']
    ).rename({c: f'{c}_d' for c in (
        'jensen_gap', 'effective_horizon', 'bootstrap_fraction',
        'target_staleness_late', 'eval_best_burst_mean')})
    j = v.join(d, on=list(pair_keys), how='inner')
    if len(j) == 0:
        return {}
    out = {}
    for k in ('jensen_gap', 'effective_horizon', 'bootstrap_fraction',
              'target_staleness_late', 'eval_best_burst_mean'):
        out[f'd_{k}'] = (j[f'{k}_d'] - j[f'{k}_v']).to_numpy()
        out[f'sd_v_{k}'] = float(j[f'{k}_v'].std())
        out[f'mean_v_{k}'] = float(j[f'{k}_v'].mean())
    return out


def within_env_correlations(deltas: dict[str, np.ndarray]) -> dict[str, tuple[float, float]]:
    """Per-env r(Δ_X, Δ_outcome) for X in {jens, staleness, eff_h, bf}.
    Returns {predictor: (r, p)}."""
    d_o = deltas['d_eval_best_burst_mean']
    out: dict[str, tuple[float, float]] = {}
    if d_o.std() == 0 or len(d_o) < 4:
        return out
    for predictor in ('jensen_gap', 'effective_horizon',
                      'bootstrap_fraction', 'target_staleness_late'):
        d_p = deltas[f'd_{predictor}']
        if d_p.std() == 0:
            out[predictor] = (float('nan'), float('nan'))
            continue
        r, p = stats.pearsonr(d_p, d_o)
        out[predictor] = (float(r), float(p))
    return out


def main() -> None:
    print('Per-env Δ panel + within-env r(Δ_predictor, Δ_outcome):\n', flush=True)
    panel = []
    for spec in ENV_SOURCES:
        env = spec['env']
        print(f'  [{env}]', flush=True)
        per_cell = compute_per_cell(spec)
        if per_cell is None:
            print(f'    no data')
            continue
        deltas = pair_deltas(per_cell, spec['pair_keys'])
        if not deltas:
            print(f'    no pairs')
            continue
        n = len(deltas['d_jensen_gap'])

        row = {'env': env, 'polarity': spec['polarity'], 'n_pairs': n}
        for k in ('jensen_gap', 'effective_horizon', 'bootstrap_fraction',
                  'target_staleness_late', 'eval_best_burst_mean'):
            d_arr = deltas[f'd_{k}']
            row[f'mean_d_{k}'] = float(np.nanmean(d_arr))
            sd_v = deltas[f'sd_v_{k}']
            if sd_v > 0:
                row[f'cohen_d_{k}'] = float(np.nanmean(d_arr) / sd_v)
            else:
                row[f'cohen_d_{k}'] = float('nan')

        # Within-env: r(Δ_X, Δ_outcome) for each candidate
        within = within_env_correlations(deltas)
        for predictor, (r, p) in within.items():
            row[f'r_{predictor}_to_outcome'] = r
            row[f'p_{predictor}_to_outcome'] = p

        panel.append(row)
        print(
            f'    n={n}  pol={spec["polarity"]:+.2f}    '
            f'r(Δjens,Δo)={within.get("jensen_gap", (float("nan"),0))[0]:+.3f}  '
            f'r(Δstale,Δo)={within.get("target_staleness_late", (float("nan"),0))[0]:+.3f}  '
            f'r(Δeff_h,Δo)={within.get("effective_horizon", (float("nan"),0))[0]:+.3f}  '
            f'r(Δbf,Δo)={within.get("bootstrap_fraction", (float("nan"),0))[0]:+.3f}',
            flush=True,
        )

    print()
    print('=== Mechanism-blind hypothesis test ===')
    print('Prediction: r(Δstale,Δo) and r(Δjens,Δo) are CONSISTENT-SIGN across polarities;')
    print('           r(Δeff_h,Δo) FLIPS SIGN by polarity (per polarity definition).')
    print()
    print(f'{"env":<24} {"polarity":>9} {"r_jens":>8} {"r_stale":>8} {"r_eff_h":>8} {"r_bf":>8}', flush=True)
    print('-' * 75)
    for r in panel:
        print(
            f'  {r["env"]:<22} {r["polarity"]:>+9.2f} '
            f'{r.get("r_jensen_gap_to_outcome", float("nan")):>+8.3f} '
            f'{r.get("r_target_staleness_late_to_outcome", float("nan")):>+8.3f} '
            f'{r.get("r_effective_horizon_to_outcome", float("nan")):>+8.3f} '
            f'{r.get("r_bootstrap_fraction_to_outcome", float("nan")):>+8.3f}',
            flush=True,
        )

    print()
    print('Cross-env: do within-env correlations correlate with polarity?')
    print('-' * 90, flush=True)
    pols = np.array([r['polarity'] for r in panel])
    for predictor in ('jensen_gap', 'target_staleness_late',
                      'effective_horizon', 'bootstrap_fraction'):
        rs = np.array([r.get(f'r_{predictor}_to_outcome', float('nan')) for r in panel])
        if np.isnan(rs).all():
            continue
        mask = ~np.isnan(rs)
        if mask.sum() < 3:
            continue
        rho_s, p_s = stats.spearmanr(pols[mask], rs[mask])
        sign_consistent = bool(np.all(np.sign(rs[mask]) == np.sign(rs[mask][0])))
        print(
            f'  r(Δ_{predictor:<22}, Δ_outcome): '
            f'ρ_signed_with_polarity={rho_s:+.3f} (p={p_s:.3g})  '
            f'sign-consistent across envs: {sign_consistent}',
            flush=True,
        )

    out = Path('experiments/findings/sync_curve_breakout/polarity_mechanism_test.json')
    out.write_text(json.dumps(panel, indent=2))
    print(f'\nwrote: {out}', flush=True)


if __name__ == '__main__':
    main()
