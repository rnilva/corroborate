"""Per-env per-burst trajectory of DDQN vs vanilla.

For each canonical env, prints the per-burst trajectory of:
  Δ_jens                  mech-canonical (per-burst, derived from q − mc)
  Δ_bg_frac               MC-free wedge frequency (per-burst not in cache —
                          we report Δ_q_per_burst as a proxy for the
                          Q-magnitude channel that bg_frac shadows)
  Δ_q                     per-burst Q magnitude
  Δ_out_raw               per-burst raw outcome (mc_return_raw_per_burst_mean)
  Δ_out_disc              per-burst discounted outcome (mc_return__mean_axis_-1)
  d_out_raw(burst)        unpaired Cohen's d on raw outcome at this burst

The summary table (`per_env_panel_full.py`) uses best-burst-raw for
env-level aggregation per substrate convention. This script
unrolls the trajectory inside each env so we can see:
  - WHEN the divergence between arms emerges
  - WHETHER mech precedes or co-evolves with outcome
  - WHETHER there are Q-explosion phases (per
    `findings_q_div_threshold_too_loose`)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]


def _cohens_d_arr(d_arr: np.ndarray, v_arr: np.ndarray) -> float:
    if d_arr.size < 3 or v_arr.size < 3:
        return float('nan')
    pooled = np.sqrt(0.5 * (d_arr.var(ddof=1) + v_arr.var(ddof=1)))
    if pooled == 0:
        return float('nan')
    return float((d_arr.mean() - v_arr.mean()) / pooled)


def main() -> int:
    df = pl.read_parquet(REPO / 'experiments/data/cache/ddqn.parquet').filter(
        ~pl.col('env_name').str.contains('bsuite')
    )
    for env in df.select(pl.col('env_name').unique()).to_series().sort().to_list():
        sub = df.filter(pl.col('env_name') == env)
        v = sub.filter(pl.col('arm_key').str.contains('baseline'))
        d = sub.filter(~pl.col('arm_key').str.contains('baseline'))
        if v.is_empty() or d.is_empty():
            continue
        v_rows = v.to_dicts()
        d_rows = d.to_dicts()
        # Determine n_bursts from first cell
        n_bursts = len(v_rows[0].get('mc_return_raw__mean_axis_-1') or [])
        if n_bursts == 0:
            continue
        print(f'\n=== {env}  (n_v={len(v_rows)} n_d={len(d_rows)}, n_bursts={n_bursts}) ===')
        print(
            f'  {"b":>2} {"Δ_jens":>8} {"Δ_q":>8} {"Δ_out_raw":>10} '
            f'{"Δ_out_disc":>11} {"d_out_raw":>10} {"v_out_raw":>10} {"d_out_raw_arm":>13}'
        )
        for b in range(n_bursts):
            v_jens = np.array([
                np.asarray(r.get('q_per_burst') or [])[b]
                - np.asarray(r.get('mc_return__mean_axis_-1') or [])[b]
                if len(r.get('q_per_burst') or []) > b
                and len(r.get('mc_return__mean_axis_-1') or []) > b
                else np.nan
                for r in v_rows
            ])
            d_jens = np.array([
                np.asarray(r.get('q_per_burst') or [])[b]
                - np.asarray(r.get('mc_return__mean_axis_-1') or [])[b]
                if len(r.get('q_per_burst') or []) > b
                and len(r.get('mc_return__mean_axis_-1') or []) > b
                else np.nan
                for r in d_rows
            ])
            v_q = np.array([
                np.asarray(r.get('q_per_burst') or [])[b]
                if len(r.get('q_per_burst') or []) > b else np.nan
                for r in v_rows
            ])
            d_q = np.array([
                np.asarray(r.get('q_per_burst') or [])[b]
                if len(r.get('q_per_burst') or []) > b else np.nan
                for r in d_rows
            ])
            v_out_raw = np.array([
                np.asarray(r.get('mc_return_raw__mean_axis_-1') or [])[b]
                if len(r.get('mc_return_raw__mean_axis_-1') or []) > b else np.nan
                for r in v_rows
            ])
            d_out_raw = np.array([
                np.asarray(r.get('mc_return_raw__mean_axis_-1') or [])[b]
                if len(r.get('mc_return_raw__mean_axis_-1') or []) > b else np.nan
                for r in d_rows
            ])
            v_out_disc = np.array([
                np.asarray(r.get('mc_return__mean_axis_-1') or [])[b]
                if len(r.get('mc_return__mean_axis_-1') or []) > b else np.nan
                for r in v_rows
            ])
            d_out_disc = np.array([
                np.asarray(r.get('mc_return__mean_axis_-1') or [])[b]
                if len(r.get('mc_return__mean_axis_-1') or []) > b else np.nan
                for r in d_rows
            ])
            dj = float(np.nanmean(d_jens) - np.nanmean(v_jens))
            dq = float(np.nanmean(d_q) - np.nanmean(v_q))
            do_r = float(np.nanmean(d_out_raw) - np.nanmean(v_out_raw))
            do_d = float(np.nanmean(d_out_disc) - np.nanmean(v_out_disc))
            cd_out = _cohens_d_arr(d_out_raw[~np.isnan(d_out_raw)], v_out_raw[~np.isnan(v_out_raw)])
            n_d_better = int((d_out_raw > v_out_raw.mean()).sum())  # rough
            print(
                f'  {b:>2} {dj:>+8.3f} {dq:>+8.3f} {do_r:>+10.3f} '
                f'{do_d:>+11.4f} {cd_out:>+10.2f} '
                f'{float(np.nanmean(v_out_raw)):>+10.3f} '
                f'{n_d_better}/{len(d_rows)}'.rjust(13)
            )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
