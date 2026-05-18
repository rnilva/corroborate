"""Empirical permutation calibration for cluster Finding false-positive rate.

Under the null "arm has no causal effect", how often does each
cluster Finding fire SUPPORTED? Permutes `arm_key` within each
`env_name` stratum (preserves within-env outcome distribution
+ marginal arm counts, breaks the arm-outcome causal link),
re-runs the hypothesis with `use_cache=False`, records the
cluster verdict per Finding. K iterations → empirical FPR.

The replacement for the analytic AND-composition false-positive
rate calculation in the DDQN case study report §3.3 (which
assumed bridge independence — wrong given shared substrate-
realization noise). This script computes the dependent-α
family-wise SUPPORTED rate directly from the framework's own
primitives, no independence assumption needed.

Usage:
    PYTHONPATH=. uv run python scripts/permutation_calibration.py \
        experiments.findings.ddqn_sweeps --k 50 --seed 42

Output: JSON to `docs/PERMUTATION_CALIBRATION_<hypothesis>.json`
with per-Finding cluster-verdict counts.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

from corroborate.graph.causal import (
    ClusterVerdict, PostEvalEntry, composed_verdict, evaluated_graph,
)
from corroborate.runner import run
from corroborate.runner.runner import (
    _default_cache_path,  # pyright: ignore[reportPrivateUsage]
    _validate_hypothesis,  # pyright: ignore[reportPrivateUsage]
)


def permute_arm_within_env(
    df: pl.DataFrame, rng: np.random.Generator,
) -> pl.DataFrame:
    """Shuffle `arm_key` within each `env_name`. Preserves env-
    level outcome distribution + marginal arm counts; breaks
    arm-outcome causality."""
    # Build a permutation of row indices stratified by env_name.
    n = len(df)
    arr = np.arange(n)
    env_col = df['env_name'].to_numpy()
    # For each env, shuffle the indices among the env's rows;
    # then use the resulting permutation to reassign arm_key.
    perm = arr.copy()
    for env in np.unique(env_col):
        idx = np.where(env_col == env)[0]
        shuf = rng.permutation(idx)
        perm[idx] = shuf
    permuted_arm = df['arm_key'].to_numpy()[perm]
    return df.with_columns(pl.Series('arm_key', permuted_arm))


def evaluate_finding_clusters(
    h_module, results
) -> dict[str, ClusterVerdict]:
    """For each Finding in h.FINDINGS, compute composed_verdict
    against the empirical bridges' results."""
    post_eval = {
        name: PostEvalEntry(verdict=ev.verdict, extent_hash=ev.extent_hash)
        for name, ev in results.items()
    }
    g = evaluated_graph(h_module.BRIDGES, post_eval)
    out: dict[str, ClusterVerdict] = {}
    for f in getattr(h_module, 'FINDINGS', ()):
        finding_bridges = f.BRIDGES
        verdict = composed_verdict(g, bridges=finding_bridges)
        out[f.__name__.split('.')[-1]] = verdict
    return out


def main() -> int:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('module', help='hypothesis module path')
    parser.add_argument('--k', type=int, default=50, help='number of permutations')
    parser.add_argument('--seed', type=int, default=42, help='RNG seed')
    parser.add_argument('--out', type=Path, default=None,
                        help='output JSON path (default docs/PERMUTATION_CALIBRATION_<short>.json)')
    args = parser.parse_args()

    h = _validate_hypothesis(importlib.import_module(args.module))
    cache_path = _default_cache_path(h)
    print(f'cache: {cache_path}', flush=True)
    df = pl.read_parquet(cache_path)
    print(f'cells: {len(df)} × {len(df.columns)} cols', flush=True)
    rng = np.random.default_rng(args.seed)

    # Baseline (unpermuted) verdicts — sanity check
    print('--- baseline (unpermuted) ---', flush=True)
    t0 = time.time()
    base_results = run(h, data=df, use_cache=False, write_cache=False,
                      restore_from_cloud=False)
    base_finding_verdicts = evaluate_finding_clusters(h, base_results)
    print(f'  baseline took {time.time()-t0:.1f}s; '
          f'{len(base_finding_verdicts)} findings:', flush=True)
    for fname, v in sorted(base_finding_verdicts.items()):
        print(f'    {v.value:14s} {fname}', flush=True)

    # Permutation loop
    finding_counters: dict[str, Counter] = {
        fname: Counter() for fname in base_finding_verdicts
    }
    n_any_supported = 0  # family-wise: at least 1 Finding SUPPORTED per iter
    iter_times: list[float] = []
    print(f'\n--- {args.k} permutations ---', flush=True)
    for i in range(args.k):
        t0 = time.time()
        permuted = permute_arm_within_env(df, rng)
        results = run(h, data=permuted, use_cache=False, write_cache=False,
                      restore_from_cloud=False)
        v = evaluate_finding_clusters(h, results)
        iter_any_supported = False
        for fname, verdict in v.items():
            finding_counters[fname][verdict.value] += 1
            if verdict == ClusterVerdict.SUPPORTED:
                iter_any_supported = True
        if iter_any_supported:
            n_any_supported += 1
        iter_times.append(time.time() - t0)
        if (i+1) % 5 == 0:
            print(f'  iter {i+1}/{args.k}: {iter_times[-1]:.1f}s '
                  f'(mean {np.mean(iter_times):.1f}s), '
                  f'family-wise running rate {n_any_supported}/{i+1}', flush=True)

    # Summary
    print(f'\n--- Empirical FPR (SUPPORTED-rate under null) ---', flush=True)
    out_data: dict[str, object] = {
        'module': args.module,
        'k': args.k,
        'seed': args.seed,
        'baseline_verdicts': {
            f: v.value for f, v in base_finding_verdicts.items()
        },
        'permutation_counts': {
            fname: dict(counter) for fname, counter in finding_counters.items()
        },
        'fpr_supported': {},
    }
    for fname in sorted(finding_counters):
        c = finding_counters[fname]
        total = sum(c.values())
        n_supported = c.get('supported', 0)
        fpr = n_supported / total if total else 0.0
        out_data['fpr_supported'][fname] = fpr
        baseline_v = base_finding_verdicts[fname].value
        print(f'  {fname:50s} baseline={baseline_v:14s} '
              f'fpr={fpr:.3f} ({n_supported}/{total} supported)', flush=True)
    fw_fpr = n_any_supported / args.k
    out_data['family_wise_fpr'] = fw_fpr
    print(f'\nFamily-wise FPR (P[≥1 Finding SUPPORTED | null]): '
          f'{fw_fpr:.3f} ({n_any_supported}/{args.k})', flush=True)

    out_path = args.out or Path(
        f'docs/PERMUTATION_CALIBRATION_{args.module.split(".")[-1]}.json'
    )
    out_path.write_text(json.dumps(out_data, indent=2))
    print(f'\nwrote {out_path}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
