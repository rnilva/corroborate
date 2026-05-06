"""Diagnose why DDQN hurts at sync=10000 on Breakout.

Three diagnostics across the sync curve:

1. **Cross-seed σ_Q per burst** — tests the Q-amplification hypothesis
   (`findings_q_amplification_cartpole.md`): vanilla's max-bias acts as
   a cross-seed regularizer, so DDQN's "faithful" target tracking shows
   higher per-burst σ across seeds. If DDQN σ_Q > vanilla σ_Q at
   sync=10000 but not at sync=1000-3000, the regime-specific failure
   is structural.

2. **Argmax disagreement rate** — fraction of training steps where
   `argmax(Q_online) != argmax(Q_target)`. This is the rate at which
   DDQN's "decoupling" actually fires. If disagreement is rare at
   sync=10000, DDQN ≈ vanilla algorithmically; if frequent, the
   amplification has a clear engagement source.

3. **TD error per burst** — DDQN's "decoupled" target should be lower
   than vanilla's max-bias target. If TD error is HIGHER for DDQN at
   sync=10000, the targets are being chased to different (bigger) Q
   surfaces than vanilla.

All three are computed on the existing traces.parquet column projections
without retraining anything.
"""
from __future__ import annotations

import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import json
from pathlib import Path

import numpy as np
import polars as pl

ENV = 'Breakout-MinAtar'
DATA = Path('experiments/data')
TREATMENT_ARM = 'bootstrap=partial(Claim:bootstrap;greedification=Claim:double_greedify)'
BASELINE_ARM = 'baseline'
N_BURSTS = 20
STEPS_PER_BURST = 50_000


def _bin_per_burst_mean(arr: np.ndarray) -> np.ndarray:
    """Bin a length-1M step-level array into 20 burst-mean values."""
    a = arr[: N_BURSTS * STEPS_PER_BURST]
    return a.reshape(N_BURSTS, STEPS_PER_BURST).mean(axis=1)


def _bin_per_burst_max(arr: np.ndarray) -> np.ndarray:
    a = arr[: N_BURSTS * STEPS_PER_BURST]
    return a.reshape(N_BURSTS, STEPS_PER_BURST).max(axis=1)


def _argmax_disagree_per_burst(
    online_amax: np.ndarray, target_amax: np.ndarray,
) -> np.ndarray:
    """Per-burst rate at which online's argmax differs from target's."""
    a = online_amax[: N_BURSTS * STEPS_PER_BURST].reshape(N_BURSTS, STEPS_PER_BURST)
    b = target_amax[: N_BURSTS * STEPS_PER_BURST].reshape(N_BURSTS, STEPS_PER_BURST)
    return (a != b).mean(axis=1)


def _load_corpus_diagnostics(
    corpus: str, sync: int,
    *, traces_paths: list[Path] | None = None,
) -> dict[str, dict[int, dict[str, np.ndarray]]]:
    """For one corpus, return:
        {arm_short: {burst: {seed: per-burst diagnostic dict}}}

    `traces_paths`, when given, is a list of trace shards (no top-level
    traces.parquet). Used for `minatar_1M` where the merged file isn't
    archived; only the per-arm tmp/ shards are.
    """
    runs = pl.read_parquet(
        DATA / corpus / 'runs.parquet',
        columns=['id', 'env_name', 'arm_key', 'seed'],
    ).filter(pl.col('env_name') == ENV)
    if traces_paths is None:
        traces_paths = [DATA / corpus / 'traces.parquet']
    trace_cols = [
        'id', 'td_error', 'loss',
        'online_max_q_per_step', 'target_max_q_per_step',
        'online_mean_q_per_step',
        'online_argmax_per_step', 'target_argmax_per_step',
    ]
    trace_dfs = [pl.read_parquet(p, columns=trace_cols) for p in traces_paths]
    traces = pl.concat(trace_dfs, how='vertical_relaxed') if len(trace_dfs) > 1 else trace_dfs[0]
    bk = runs.join(traces, on='id', how='inner')

    # arm_key strings → short labels for printing
    out: dict[str, dict[int, list[dict[str, float]]]] = {
        'vanilla': {b: [] for b in range(N_BURSTS)},
        'ddqn':    {b: [] for b in range(N_BURSTS)},
    }
    for row in bk.iter_rows(named=True):
        arm = 'ddqn' if 'bootstrap=' in row['arm_key'] else 'vanilla'
        td = np.asarray(row['td_error'], dtype=np.float64)
        loss = np.asarray(row['loss'], dtype=np.float64)
        online_max = np.asarray(row['online_max_q_per_step'], dtype=np.float64)
        target_max = np.asarray(row['target_max_q_per_step'], dtype=np.float64)
        online_mean = np.asarray(row['online_mean_q_per_step'], dtype=np.float64)
        online_amax = np.asarray(row['online_argmax_per_step'], dtype=np.int64)
        target_amax = np.asarray(row['target_argmax_per_step'], dtype=np.int64)
        td_pb = _bin_per_burst_mean(np.abs(td))
        loss_pb = _bin_per_burst_mean(loss)
        online_max_pb = _bin_per_burst_mean(online_max)
        target_max_pb = _bin_per_burst_mean(target_max)
        online_mean_pb = _bin_per_burst_mean(online_mean)
        disagree_pb = _argmax_disagree_per_burst(online_amax, target_amax)
        for b in range(N_BURSTS):
            out[arm][b].append({
                'seed': row['seed'],
                'td_error_abs': float(td_pb[b]),
                'loss': float(loss_pb[b]),
                'online_max_q': float(online_max_pb[b]),
                'target_max_q': float(target_max_pb[b]),
                'online_mean_q': float(online_mean_pb[b]),
                'argmax_disagree_rate': float(disagree_pb[b]),
            })
    return out  # type: ignore[return-value]


def main() -> None:
    # sync=100 has no merged traces.parquet — uses the two per-arm
    # tmp shards (Breakout vanilla + Breakout ddqn) restored from cloud.
    sync100_shards = [
        DATA / 'minatar_1M/tmp/arm002__Breakout-MinAtar__vanilla_dqn__traces.parquet',
        DATA / 'minatar_1M/tmp/arm003__Breakout-MinAtar__ddqn__traces.parquet',
    ]
    sources = {
        100:   'minatar_1M',
        1000:  'minatar_sync_curve/ddqn_sync1k',
        3000:  'minatar_sync_curve/ddqn_sync3k',
        10000: 'minatar_sync_intervention',
    }
    shards_for: dict[int, list[Path] | None] = {
        100: sync100_shards,
        1000: None,
        3000: None,
        10000: None,
    }
    table_cols = (
        'online_max_q', 'argmax_disagree_rate', 'td_error_abs', 'loss',
    )
    print(f'Per-arm per-burst diagnostics on {ENV} across the sync curve...')
    summaries: dict[int, dict[str, list[dict[str, float]]]] = {}
    for sync, corpus in sources.items():
        print(f'  loading sync={sync} from {corpus}/...')
        diag = _load_corpus_diagnostics(corpus, sync, traces_paths=shards_for[sync])
        # collapse to per-burst summary statistics
        s: dict[str, list[dict[str, float]]] = {'vanilla': [], 'ddqn': []}
        for arm in ('vanilla', 'ddqn'):
            for b in range(N_BURSTS):
                rows = diag[arm][b]
                if not rows:
                    s[arm].append({})
                    continue
                row = {'burst': b, 'n': len(rows)}
                for col in table_cols + ('online_mean_q',):
                    vs = np.array([r[col] for r in rows])
                    row[f'{col}_mean'] = float(vs.mean())
                    row[f'{col}_sd'] = float(vs.std(ddof=1))
                s[arm].append(row)
        summaries[sync] = s

    # Print headline panel: σ_Q across seeds per burst (Q-amplification).
    print()
    print('A. Cross-seed σ(online_max_q) per burst — Q-amplification test')
    print(f'    "DDQN > vanilla" means DDQN has more cross-seed divergence.')
    print(f'    burst |' + ' | '.join(
        f'sync={s:>5}: van vs ddqn' for s in sources
    ))
    for b in [0, 4, 8, 12, 16, 19]:
        line = f'    {b:>5} |'
        for sync in sources:
            v = summaries[sync]['vanilla'][b].get('online_max_q_sd', float('nan'))
            d = summaries[sync]['ddqn'][b].get('online_max_q_sd', float('nan'))
            ratio = d / v if v else float('nan')
            line += f' {v:>9.2f}/{d:>9.2f} ({ratio:>4.2f}×) |'
        print(line)

    print()
    print('B. Cross-seed σ(td_error_abs) per burst — target-chase noise')
    for b in [0, 4, 8, 12, 16, 19]:
        line = f'    {b:>5} |'
        for sync in sources:
            v = summaries[sync]['vanilla'][b].get('td_error_abs_sd', float('nan'))
            d = summaries[sync]['ddqn'][b].get('td_error_abs_sd', float('nan'))
            ratio = d / v if v else float('nan')
            line += f' {v:>9.4f}/{d:>9.4f} ({ratio:>4.2f}×) |'
        print(line)

    print()
    print('C. Mean argmax_disagree_rate per burst — DDQN mechanism engagement')
    for b in [0, 4, 8, 12, 16, 19]:
        line = f'    {b:>5} |'
        for sync in sources:
            v = summaries[sync]['vanilla'][b].get('argmax_disagree_rate_mean', float('nan'))
            d = summaries[sync]['ddqn'][b].get('argmax_disagree_rate_mean', float('nan'))
            line += f' {v:>9.3f}/{d:>9.3f}        |'
        print(line)

    print()
    print('D. Mean abs(td_error) per burst — algorithm "doing more work" signal')
    for b in [0, 4, 8, 12, 16, 19]:
        line = f'    {b:>5} |'
        for sync in sources:
            v = summaries[sync]['vanilla'][b].get('td_error_abs_mean', float('nan'))
            d = summaries[sync]['ddqn'][b].get('td_error_abs_mean', float('nan'))
            ratio = d / v if v else float('nan')
            line += f' {v:>9.4f}/{d:>9.4f} ({ratio:>4.2f}×) |'
        print(line)

    out = Path('experiments/findings/sync_curve_breakout/mechanism_panel.json')
    out.write_text(json.dumps(summaries, indent=2))
    print()
    print(f'Wrote: {out}')


if __name__ == '__main__':
    main()
