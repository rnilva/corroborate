"""Substrate-author workflow demo: Panel for exploration.

Three steps that capture the day-to-day reality:

1. **Load** — `Panel.from_corpus` / `Panel.from_corpora` /
   `Panel.from_cache` materialise cells without the ingest dance
   (no sidecar plumbing, no @claim_bridge harness).
2. **Probe** — `panel.diagnostics` surfaces per-stratum cell
   counts, source-corpus map, finite-measurable fractions,
   nonunique-config heterogeneity.
3. **Analyze** — call any @analysis primitive on `panel.cells`
   directly (DataFrame is the canonical analysis input) OR
   hand-roll a polars query, OR use `panel.derive(DerivedSpec)`
   for per-stratum aggregates.

What this demo deliberately does NOT show:

- @claim_bridge authoring. Once exploration finds a signal,
  authoring is "wrap the same code in a @claim_bridge + return
  a Verdict" — see real bridges in
  `experiments/findings/`. Bridges consume @analysis results
  by typed parameter name (pytest-fixture style); the same
  primitives the Day-2 exploration above called directly
  (`cross_stratum_property_slope`, `partial_spearman`, …) are
  the registered analyses a bridge's `holds_when` body
  receives.

Run from repo root:
    JAX_PLATFORMS=cpu uv run --package corroborate_rl \\
        python3 scripts/panel_workflow_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import corroborate_rl.dqn.measurables  # noqa: F401  # pyright: ignore[reportUnusedImport]
import polars as pl

from corroborate.analyses.link.cross_stratum_property_slope import (
    cross_stratum_property_slope,
)
from corroborate.data import DerivedSpec, Panel


def main() -> None:
    # === 1. Load ============================================
    print('=== Load ===\n', flush=True)
    panel = Panel.from_cache('experiments.findings.ddqn_sweeps')
    print(f'Panel.from_cache: {panel.cells.shape}')
    print(f'  stratify_by = {panel.stratify_by}')
    print(f'  scope_chain = {len(panel.scope_chain)} expressions')

    # Narrow to γ=0.999 canonical k=1 — `narrow` extends the
    # scope-provenance chain so the next inspector knows what
    # filter applied.
    narrowed = panel.narrow(pl.col('gamma') == 0.999).narrow(
        pl.col('action_duplicate_k').is_null()
        | (pl.col('action_duplicate_k') == 1),
    )
    print(f'\nnarrow γ=0.999 k=1: {narrowed.cells.shape}')

    # === 2. Probe ===========================================
    print('\n\n=== Probe (diagnostics) ===\n', flush=True)
    diag = narrowed.diagnostics
    sample_keys = sorted(diag.n_cells_per_stratum)[:6]
    print('per-stratum cell counts (first 6):')
    for k in sample_keys:
        n = diag.n_cells_per_stratum[k]
        corpora = diag.corpora_per_stratum[k]
        configs = diag.nonunique_configs_per_stratum[k]
        flag = '⚠' if (len(corpora) > 1 or configs > 1) else ' '
        print(
            f'  {flag} {k!s:60s} n={n:>3d} '
            f'corpora={len(corpora)} configs={configs}'
        )

    # === 3. Analyze ========================================
    print('\n\n=== Analyze ===\n', flush=True)

    # 3a. Hand-rolled polars query on `panel.cells` —
    #     no framework boilerplate needed.
    print('3a. Hand-rolled polars: per-env DDQN vs vanilla outcome means')
    per_env = (
        narrowed.cells
        .filter(pl.col('eval_best_burst_raw_mean').is_finite())
        .group_by(['env_name', 'arm_key'])
        .agg(pl.col('eval_best_burst_raw_mean').mean().alias('mean'))
        .sort(['env_name', 'arm_key'])
    )
    print(per_env.head(6))

    # 3b. Panel.derive — per-stratum aggregate via DerivedSpec.
    #     Same primitive Panel.diagnostics uses internally;
    #     reusable building block for hand-rolled stats.
    print('\n3b. Panel.derive — σ_Λ_a per env (vanilla cells)')
    sigmas = narrowed.derive(DerivedSpec(
        column='lambda_a_late',
        aggregator='std',
        cell_filter=pl.col('arm_key') == 'baseline',
    ))
    for k in sorted(sigmas)[:6]:
        print(f'  {k!s:60s} σ_Λ_a={sigmas[k]:.4f}')

    # 3c. Framework @analysis primitive directly on `panel.cells`
    #     (which is a pl.DataFrame — the canonical analysis input).
    #     No conversion, no shim — just pass the DataFrame.
    print('\n3c. cross_stratum_property_slope on panel.cells')
    result = cross_stratum_property_slope.fn(
        narrowed.cells,
        treatment_arm=(
            'bootstrap=partial(Claim:bootstrap;'
            'greedification=Claim:double_greedify)'
        ),
        baseline_arm='baseline',
        source='eval_best_burst_raw_mean',
        covariate_name='sigma_lambda_a',
        derived_covariate=DerivedSpec(
            column='lambda_a_late',
            aggregator='std',
            cell_filter=pl.col('arm_key') == 'baseline',
        ),
        covariate_key_field='env_name',
        stratify_by=('env_name',),
        scope_predictor='jensen_gap',
        min_baseline_predictor=0.0,
        min_strata=3,
    )
    print(
        f'   ρ={result.rho:+.4f} p={result.p_value:.4g} '
        f'n_strata={result.n_strata}'
    )

    print(
        '\nDay-2 → Day-3: wrap the above analysis in @claim_bridge + '
        'return a Verdict tuple. The framework injects the analysis '
        'result by parameter name (pytest-fixture style).',
    )


if __name__ == '__main__':
    main()
