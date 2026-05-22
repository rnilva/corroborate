"""Day-1 / Day-2 exploration + cache migration for the
Hasselt-chain hypothesis.

Worked example of the substrate-author workflow:

  Load (from canonical γ=0.999 corpora)
    → Narrow (CANONICAL_DORMANCY_SCOPE)
      → Probe (per-stratum diagnostics)
        → Analyze (B1 theorem-edge preview)
          → Promote (write hasselt_clean.parquet + sidecars)

Bridges themselves are the Day-3 surface: they consume @analysis
results by typed parameter name (see `chain.py`). This script is
the *exploration* that decides the scope predicates a substrate
author would then author into `_scope.py` + `chain.py`, AND the
migration that seeds `experiments/data/cache/hasselt_clean.parquet`
from the canonical γ=0.999 corpora without going through the
runner's `--ingest-all` walk.

Run from repo root:
    JAX_PLATFORMS=cpu uv run --package corroborate_rl \\
        python3 -m experiments.findings.hasselt_clean._exploration

Promote (write the cache):
    PROMOTE=1 JAX_PLATFORMS=cpu uv run --package corroborate_rl \\
        python3 -m experiments.findings.hasselt_clean._exploration

The `PROMOTE=1` env-var gate is the safety latch — running
without it walks the workflow but DOESN'T mutate
`experiments/data/cache/`. The promote step writes
`hasselt_clean.parquet` + `.sources.json` + `.hashes.json`
(populated from the `@measurable` registry, which is loaded by
the `corroborate_rl.dqn.measurables` import at the script
header).
"""
from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path

import polars as pl

import corroborate.analyses  # pyright: ignore[reportUnusedImport]  # populate analysis registry
import corroborate_rl.dqn.measurables  # pyright: ignore[reportUnusedImport]  # populate measurable registry

from corroborate.analyses.spearman.partial_spearman import (
    partial_spearman,
)
from corroborate.data import Panel
from experiments.findings.hasselt_clean._scope import (
    CANONICAL_DORMANCY_SCOPE,
    JDG_AVAILABLE_ENVS,
    PREMISE_ACTIVE_PER_CELL,
    PREMISE_ACTIVE_PER_STRATUM,
    VANILLA_ONLY,
)


_DATA_DIR = Path(__file__).resolve().parents[3] / 'experiments' / 'data'


# Canonical γ=0.999 corpora that contribute to the 10-env
# Hasselt-chain panel. Each corpus's `measurements.parquet` has
# `jensen_dormancy_gap` populated (per the 2026-05-22 backfill);
# `Panel.from_corpora` left-joins runs + measurements per corpus
# then diagonal-relaxed concats across corpora so schema drift
# across substrate generations is absorbed.
_CANONICAL_G0999_CORPORA: tuple[Path, ...] = (
    _DATA_DIR / 'minatar_gamma_sweep_k1' / 'g0999_Asterix-MinAtar',
    _DATA_DIR / 'minatar_gamma_sweep_k1' / 'g0999_Breakout-MinAtar',
    _DATA_DIR / 'minatar_gamma_sweep_k1' / 'g0999_Freeway-MinAtar',
    _DATA_DIR / 'minatar_gamma_sweep_k1' / 'g0999_SpaceInvaders-MinAtar',
    _DATA_DIR / 'g0999_panel_extension_snake_only' / 'g0999_Snake-jumanji',
    _DATA_DIR / 'g0999_panel_extension_lunar_cpu' / 'g0999_LunarLander-v2-jax',
    _DATA_DIR / 'metamaze_canonical_verify' / 'g0999',
    _DATA_DIR / 'l2_gamma_sweep_acrobot',
    _DATA_DIR / 'fa_depth_fourrooms',
    _DATA_DIR / 'fa_depth_xenv_gpu',  # carries MountainCar γ=0.999 cells
)


def _present_corpora() -> tuple[Path, ...]:
    """Return only the canonical corpora that exist locally. A
    workstation without every cloud-restored corpus shouldn't
    fail the script; missing-corpus lines go to stderr so the
    substrate author sees the gap."""
    present: list[Path] = []
    for c in _CANONICAL_G0999_CORPORA:
        if (c / 'runs.parquet').exists():
            present.append(c)
        else:
            print(f'  [missing] {c.relative_to(_DATA_DIR)}', file=sys.stderr)
    return tuple(present)


def main() -> None:
    promote = os.environ.get('PROMOTE') == '1'

    # === 1. Load =============================================
    #
    # Panel.from_corpora loads each corpus's runs.parquet +
    # measurements.parquet, stamps the corpus column, and
    # diagonal-relaxed concats them so columns missing in some
    # corpora become null-padded in the combined frame.
    print('=== 1. Load ===\n', flush=True)
    present = _present_corpora()
    print(
        f'canonical γ=0.999 corpora: {len(_CANONICAL_G0999_CORPORA)} '
        f'declared, {len(present)} present locally'
    )
    if not present:
        print(
            'No canonical corpora present — restore via '
            '`corroborate restore <corpus>` or `--ingest-all` '
            'first. Exploration cannot continue.',
        )
        return
    panel = Panel.from_corpora(present)
    print(f'panel: cells={panel.cells.shape}, sources={len(panel.sources)}')

    # Substrate-author boundary glue: the canonical-pool corpora
    # don't carry `action_duplicate_k` (only action-dim-inflated
    # sweeps do). The MODULE_SCOPE predicate is
    # `action_duplicate_k IS NULL OR == 1`; on a frame where the
    # column doesn't exist, polars raises ColumnNotFoundError.
    # Stamp it null (k=1 by convention) so the scope predicate
    # filters cleanly. This is the substrate-side schema-
    # alignment step the runner's diagonal_relaxed-concat
    # ingest would handle implicitly across heterogeneous-
    # schema corpora.
    if 'action_duplicate_k' not in panel.cells.columns:
        panel = replace(
            panel,
            cells=panel.cells.with_columns(
                pl.lit(None).cast(pl.Float64).alias('action_duplicate_k'),
            ),
        )

    # === 2. Narrow ==========================================
    #
    # CANONICAL_DORMANCY_SCOPE is the module-level scope every
    # chain bridge AND-combines with. Authoring it once at the
    # module level (per HYPOTHESIS_AS_GRAPH §3b's cluster-shape
    # rule) keeps every bridge in the cluster operating on the
    # same extent — cluster identity is structural, not
    # accidental.
    in_scope = panel.narrow(CANONICAL_DORMANCY_SCOPE)
    print(f'\nafter CANONICAL_DORMANCY_SCOPE narrow: {in_scope.cells.shape}')
    print(
        f'scope_chain depth: {len(in_scope.scope_chain)} (one expr)',
    )

    # === 3. Probe ===========================================
    #
    # Day-2 substrate-author questions live here:
    # - Which envs from JDG_AVAILABLE_ENVS actually carry data?
    # - How heterogeneous are the per-stratum cohorts?
    # - Are any (env, arm) cohorts dominated by one corpus?
    print('\n\n=== 3. Probe (diagnostics) ===\n', flush=True)
    diag = in_scope.diagnostics
    declared = set(JDG_AVAILABLE_ENVS)
    in_scope_envs: set[str] = set()
    for k in diag.n_cells_per_stratum:
        env_obj = k[0] if k else None
        if isinstance(env_obj, str):
            in_scope_envs.add(env_obj)
    missing = declared - in_scope_envs
    print(
        f'declared dormancy-available envs: {len(declared)}\n'
        f'cache-present: {sorted(in_scope_envs)}\n'
        f'cache-missing: {sorted(missing) if missing else "(none)"}'
    )

    print('\nper-stratum cell counts:')
    typed_keys: list[tuple[str, str]] = []
    for k in diag.n_cells_per_stratum:
        if len(k) == 2 and all(isinstance(v, str) for v in k):
            typed_keys.append((str(k[0]), str(k[1])))
    for env, arm in sorted(typed_keys):
        n = diag.n_cells_per_stratum[(env, arm)]
        corpora = len(diag.corpora_per_stratum[(env, arm)])
        configs = diag.nonunique_configs_per_stratum[(env, arm)]
        flag = '⚠' if (corpora > 1 or configs > 1) else ' '
        arm_short = 'ddqn' if 'double_greedify' in arm else arm
        print(
            f'  {flag} ({env:>22s}, {arm_short:>8s}) '
            f'n={n:>3d} corpora={corpora} configs={configs}'
        )

    # === 4. Sibling predicate sensitivity ===================
    #
    # The chain.py current state uses per-stratum scope for the
    # intervention edges (B3, B4) — the principled choice per
    # finding_hasselt_chain_explicit.py's docstring (per-cell
    # JDG-conditioning is post-treatment selection, equivalent
    # to collider conditioning under do(DDQN)). The per-cell
    # vs per-stratum predicate counts surface whether selection
    # bias would materially differ between them.
    print('\n\n=== 4. Sibling predicate sensitivity ===\n', flush=True)
    per_cell_count = in_scope.cells.filter(
        PREMISE_ACTIVE_PER_CELL,
    ).height
    per_stratum_count = in_scope.cells.filter(
        PREMISE_ACTIVE_PER_STRATUM,
    ).height
    print(
        f'PREMISE_ACTIVE_PER_CELL    surviving: {per_cell_count:>5d}\n'
        f'PREMISE_ACTIVE_PER_STRATUM surviving: {per_stratum_count:>5d}'
    )
    delta = per_stratum_count - per_cell_count
    if delta == 0:
        print(
            '  → predicates equivalent on this panel '
            '(no per-cell selection bias to defend against)',
        )
    else:
        print(
            f'  → predicates differ by {delta:+d} cells — '
            f'per-stratum admits cells the per-cell predicate '
            f'rejects, and vice-versa. Per-stratum is the '
            f'principled choice for the intervention edges '
            f'(see finding docstring §"Per-stratum scope is '
            f'the principled choice").',
        )

    # === 5. B1 theorem-edge preview =========================
    #
    # B1 (`hasselt_floor_predicts_observed_bias__vanilla`)
    # tests stratified partial-Spearman ρ(jensen_dormancy_gap,
    # jensen_gap) under VANILLA_ONLY. Pre-check: ρ is
    # structurally undefined when JDG has zero variance — skip
    # cleanly rather than reporting NaN-as-result.
    print('\n\n=== 5. B1 theorem-edge preview ===\n', flush=True)
    vanilla_panel = in_scope.narrow(VANILLA_ONLY)
    if vanilla_panel.cells.height < 30:
        print(
            f'skipped: vanilla cells = {vanilla_panel.cells.height} '
            f'< 30 (per-env panel underpowered).'
        )
    else:
        jdg_variance = vanilla_panel.cells.filter(
            pl.col('jensen_dormancy_gap') > 0,
        ).height
        if jdg_variance == 0:
            print(
                'skipped: vanilla cohort has '
                'jensen_dormancy_gap == 0 in every cell — ρ is '
                'structurally undefined.\n'
                'The bridge correctly fires POWER_INSUFFICIENT '
                'on this cohort.'
            )
        else:
            b1 = partial_spearman.fn(
                vanilla_panel.cells,
                x='jensen_dormancy_gap',
                y='jensen_gap',
                conditioning=(),
                stratify_by='env_name',
                min_stratum_size=30,
            )
            print(
                f'pooled ρ={b1.rho_pooled:+.4f} '
                f'p={b1.p_value:.4g} '
                f'n_strata={b1.n_strata} '
                f'n_obs_total={b1.n_obs_total} '
                f'(cells with JDG>0: {jdg_variance})'
            )

    # === 6. Promote =========================================
    #
    # `to_cache` writes the narrowed cohort to
    # `experiments/data/cache/hasselt_clean.parquet` + sources
    # sidecar + hashes manifest (populated from the live
    # @measurable registry). Gated on `PROMOTE=1` to avoid
    # silent cache mutation when running the script as a
    # workflow demo.
    print('\n\n=== 6. Promote ===\n', flush=True)
    if not promote:
        print(
            'skipped: set PROMOTE=1 to write '
            'experiments/data/cache/hasselt_clean.parquet '
            '+ sidecars. Dry-run preview:'
        )
        print(
            f'  would write: {in_scope.cells.shape[0]} cells '
            f'across {len(in_scope.sources)} source corpora\n'
            f'  cache target: experiments/data/cache/hasselt_clean.parquet\n'
            f'  sidecar target: experiments/data/cache/hasselt_clean.sources.json\n'
            f'  manifest target: experiments/data/cache/hasselt_clean.hashes.json'
        )
        return
    written = in_scope.to_cache(
        'experiments.findings.hasselt_clean',
    )
    print(f'wrote: {written}')
    print(
        f'sidecar: {written.with_suffix(".sources.json")} '
        f'({len(in_scope.sources)} sources + fresh ingested_at)\n'
        f'manifest: {written.with_suffix(".hashes.json")} '
        f'(populated from the @measurable registry — '
        f'`corroborate_rl.dqn.measurables` imported at script '
        f'header). The runner\'s drift detection now has '
        f'authoritative signatures to validate against.'
    )


if __name__ == '__main__':
    main()
