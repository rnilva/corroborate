"""Runner: drive the LG-SCM, package each run as `RunRow` (+
optional `TraceRow` for multi-burst phased runs).

Scalar entry points (one Y-mean per cell, no per-burst structure):

- `run_cell(scm, *, seed, arm_key, env_name)` — one cell as a
  `RunRow` whose `measurements` carry both the configurational
  leaves (β coefficients, μ, σ) and observation summaries
  (`x_mean`, `z_mean`, `y_mean`).
- `run_arm(scm, *, seeds, ...)` / `run_paired_arms(treatment, baseline,
  *, seeds, ...)` — sweep helpers.

Phased entry points (per-burst Y matrix; for `paired_g_per_burst`
and link analyses):

- `run_phased_cell(scms_per_burst, *, seed, arm_key, env_name)` —
  returns `(RunRow, TraceRow)`. RunRow carries scalar
  measurements (overall `y_mean`, `seed`, `env_name`); TraceRow
  carries the `(n_bursts, n_steps)` Y matrix at the
  author-chosen key `y_per_episode`.
- `merge_cell(run, trace)` — flat merge of RunRow + TraceRow into
  the cell-dict shape per-burst analyses consume. Mirrors the
  production polars left-join of `runs` + `traces` on `id`.
- `run_paired_phased_arms(...)` — phased two-arm sweep returning a
  flat list of cell-dicts.

Cells round-trip through `RunRow.as_dict()` and `TraceRow.as_dict()`,
the same surface real corpora go through, so analyses are
exercised on the production data path. Provenance fields (`id`,
`cycle_id`, `timestamp`) get filled with deterministic synthetic
values so re-runs yield byte-equal output.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from corroborate.bridge.verdict import Verdict
from corroborate.corpus.schema import (
    MeasurementLeaf,
    RunRow,
    TraceLeaf,
    TraceRow,
)

from tests.analytic.lg_scm.composition import (
    LGSCMObservation,
    LinearGaussianSCM,
    simulate,
    simulate_phased,
)


# Synthetic, deterministic timestamp. The framework records a UTC
# isoformat string here in production; tests prefer a fixed value
# so two `run_paired_arms` calls with the same arguments yield
# byte-equal output. Matches the format the schema's `timestamp`
# expects (string).
_FIXED_TIMESTAMP = '2026-01-01T00:00:00Z'


@dataclass(frozen=True, slots=True)
class CellId:
    """Composite cell identifier minted deterministically from
    (arm_key, seed, env_name). Deterministic IDs let pytest
    re-runs produce byte-equal RunRows so equality-based
    fixtures (`test_cells == fresh_cells`) stay sharp."""
    arm_key: str
    seed: int
    env_name: str

    def as_run_id(self) -> str:
        return f'lg-scm/{self.env_name}/{self.arm_key}/seed={self.seed}'


def _config_leaves(scm: LinearGaussianSCM) -> dict[str, MeasurementLeaf]:
    """Project the bundle's structural coefficients to dotted-leaf
    measurements. Fields land at their bare attribute name — the
    SCM has no nested config, so no dotted hierarchy is needed yet.
    """
    return {
        'mu_x': scm.mu_x,
        'sigma_x': scm.sigma_x,
        'beta_xz': scm.beta_xz,
        'sigma_z': scm.sigma_z,
        'beta_zy': scm.beta_zy,
        'sigma_y': scm.sigma_y,
        'n_steps': scm.n_steps,
    }


def _observation_summaries(obs: LGSCMObservation) -> dict[str, MeasurementLeaf]:
    """Project the run's scalar summaries to measurement leaves.
    Trajectories (`obs.x`, `obs.z`, `obs.y`) live on the parallel
    `TraceRow` surface; this scalar projection is what `RunRow`
    persists for paired-g and friends."""
    return {
        'x_mean': obs.x_mean,
        'z_mean': obs.z_mean,
        'y_mean': obs.y_mean,
    }


def run_cell(
    scm: LinearGaussianSCM,
    *,
    seed: int,
    arm_key: str,
    env_name: str = 'lg_scm',
) -> RunRow:
    """Run one cell; return a `RunRow` with closed-form-tractable
    measurements.

    The verdict is fixed to `HELD` because the implementation emits raw
    cells, not bridge results — the analyses under test compute
    their own verdicts from the cell-set. A non-HELD value would
    encode a bridge decision the implementation has no business making.

    `seed`, `env_name` and `arm_key` flow into `measurements` at
    top-level so paired_g's default `pair_by=('seed',)` and
    env-stratified panels work without further plumbing.
    """
    obs = simulate(scm, seed=seed)
    cell_id = CellId(arm_key=arm_key, seed=seed, env_name=env_name)
    measurements: dict[str, MeasurementLeaf] = {
        'env_name': env_name,
        'seed': seed,
    }
    measurements.update(_config_leaves(scm))
    measurements.update(_observation_summaries(obs))
    return RunRow(
        id=cell_id.as_run_id(),
        parent_id=None,
        cycle_id=None,
        timestamp=_FIXED_TIMESTAMP,
        verdict=Verdict.HELD,
        arm_key=arm_key,
        measurements=measurements,
    )


def run_arm(
    scm: LinearGaussianSCM,
    *,
    seeds: Iterable[int],
    arm_key: str,
    env_name: str = 'lg_scm',
) -> list[RunRow]:
    """Run one arm across a list of seeds; return RunRows in seed
    order. The list shape (rather than a generator) is what
    paired_g and friends consume."""
    return [
        run_cell(scm, seed=s, arm_key=arm_key, env_name=env_name)
        for s in seeds
    ]


def run_paired_arms(
    *,
    treatment: LinearGaussianSCM,
    baseline: LinearGaussianSCM,
    seeds: Sequence[int],
    treatment_arm: str = 'treatment',
    baseline_arm: str = 'baseline',
    env_name: str = 'lg_scm',
) -> list[RunRow]:
    """Two-arm paired sweep. Both arms run on the *same* `seeds`
    so paired-Δ noise cancels (same X realisation per seed).

    Returns a flat list with treatment cells then baseline cells.
    The order doesn't matter to paired_g (it pair-keys on
    `('seed',)`) but a stable order keeps fixtures readable.
    """
    rows: list[RunRow] = []
    rows.extend(run_arm(
        treatment, seeds=seeds, arm_key=treatment_arm, env_name=env_name,
    ))
    rows.extend(run_arm(
        baseline, seeds=seeds, arm_key=baseline_arm, env_name=env_name,
    ))
    return rows


def run_multi_env_paired_arms(
    *,
    envs: Mapping[str, tuple[LinearGaussianSCM, LinearGaussianSCM]],
    seeds: Sequence[int],
    treatment_arm: str = 'treatment',
    baseline_arm: str = 'baseline',
) -> list[RunRow]:
    """Multi-env paired sweep. `envs` maps `env_name` to a
    `(treatment_scm, baseline_scm)` tuple — each env has its own
    pair of SCMs so structural coefficients can vary across envs
    while the intervention axis is shared.

    Returns a flat list of RunRows tagged with each cell's env_name.
    Cells from different envs are independently seeded (same `seeds`
    per env), so meta-analyses see each env as its own stratum
    with internal paired-Δ variance.

    Used to drive `meta_regression_paired_g`: the panel of per-env
    paired-g's is the regression's stratum population, with
    env-level covariates supplied separately at the analysis call.
    """
    rows: list[RunRow] = []
    for env_name, (treatment, baseline) in envs.items():
        rows.extend(run_paired_arms(
            treatment=treatment, baseline=baseline, seeds=seeds,
            treatment_arm=treatment_arm, baseline_arm=baseline_arm,
            env_name=env_name,
        ))
    return rows


# ============ Phased (multi-burst) cells ============

# Author-chosen keys for the per-(burst, episode) trajectory
# matrices on TraceRow.leaves. Substrate-named (mirrors how the
# real DQN implementation names its trajectory columns); per-burst
# analyses specify custom sources via
# `reduce_axis(from_key(PER_BURST_Y_KEY), axis=-1, op='mean')` to
# read the per-burst-mean Y, etc.
PER_BURST_Y_KEY = 'y_per_episode'
PER_BURST_Z_KEY = 'z_per_episode'
PER_BURST_X_KEY = 'x_per_episode'


def _matrix_to_trace_leaf(
    matrix: tuple[tuple[float, ...], ...],
) -> TraceLeaf:
    """Project a (n_bursts, n_steps) tuple-of-tuples into the
    nested-list form `TraceRow.leaves` expects after a parquet
    round-trip. Tuples → lists at the boundary so polars'
    nested-list decode is the same shape the test ingests."""
    return [list(burst) for burst in matrix]


def run_phased_cell(
    scms_per_burst: tuple[LinearGaussianSCM, ...],
    *,
    seed: int,
    arm_key: str,
    env_name: str = 'lg_scm',
) -> tuple[RunRow, TraceRow]:
    """Run a multi-burst cell. Returns the `(RunRow, TraceRow)`
    pair: RunRow carries scalar provenance + overall summaries
    (`y_mean` is the unconditional mean across all bursts and
    episodes); TraceRow carries the `(n_bursts, n_steps)` Y matrix
    at `y_per_episode`.

    Per-burst β coefficients are NOT projected to RunRow
    measurements (those are scalar-only); analyses that need them
    read scalar leaves only. The per-burst Y matrix is the path
    `paired_g_per_burst` and the link analyses care about.
    """
    obs = simulate_phased(scms_per_burst, seed=seed)
    cell_id = CellId(arm_key=arm_key, seed=seed, env_name=env_name)
    run_id = cell_id.as_run_id()
    # Representative scalar leaves — first burst's coefficients.
    # When all bursts share coefficients (the constant-phase test),
    # this is just *the* configuration. Phase-flipping tests don't
    # rely on these leaves.
    rep = scms_per_burst[0]
    measurements: dict[str, MeasurementLeaf] = {
        'env_name': env_name,
        'seed': seed,
        'mu_x': rep.mu_x,
        'sigma_x': rep.sigma_x,
        'beta_zy': rep.beta_zy,
        'sigma_z': rep.sigma_z,
        'sigma_y': rep.sigma_y,
        'n_bursts': len(scms_per_burst),
        'n_steps': rep.n_steps,
        'y_mean': obs.y_mean_overall,
    }
    run = RunRow(
        id=run_id,
        parent_id=None,
        cycle_id=None,
        timestamp=_FIXED_TIMESTAMP,
        verdict=Verdict.HELD,
        arm_key=arm_key,
        measurements=measurements,
    )
    trace = TraceRow(
        id=run_id,
        cycle_id=None,
        timestamp=_FIXED_TIMESTAMP,
        leaves={
            PER_BURST_Y_KEY: _matrix_to_trace_leaf(obs.y),
            PER_BURST_Z_KEY: _matrix_to_trace_leaf(obs.z),
            PER_BURST_X_KEY: _matrix_to_trace_leaf(obs.x),
        },
    )
    return run, trace


def merge_cell(run: RunRow, trace: TraceRow) -> Mapping[str, object]:
    """Flat merge of RunRow + TraceRow into the cell-dict shape
    per-burst analyses see. Mirrors the production polars
    left-join of runs + traces on `id`: typed RunRow fields and
    measurements at top level, plus TraceRow leaves at top level.

    The `id` column is shared (joined), so trace's id silently
    wins by overwriting the run's; both are equal by construction
    in `run_phased_cell`. Other lineage fields (`cycle_id`,
    `timestamp`) are likewise duplicated and identical.
    """
    if run.id != trace.id:
        raise ValueError(
            f'merge_cell: id mismatch — run.id={run.id!r}, '
            f'trace.id={trace.id!r}',
        )
    out: dict[str, object] = dict(run.as_dict())
    for k, v in trace.leaves.items():
        out[k] = v
    return out


def run_multi_env_paired_phased_arms(
    *,
    envs: Mapping[str, tuple[
        tuple[LinearGaussianSCM, ...], tuple[LinearGaussianSCM, ...],
    ]],
    seeds: Sequence[int],
    treatment_arm: str = 'treatment',
    baseline_arm: str = 'baseline',
) -> list[Mapping[str, object]]:
    """Multi-env phased paired sweep. `envs` maps `env_name` to a
    `(treatments_per_burst, baselines_per_burst)` pair of SCM
    tuples — each env can have its own per-burst structural
    coefficients while sharing the intervention contract.

    Returns the cell-dict list `paired_g_per_burst` /
    `meta_regression_per_burst` / `mundlak_paired_g_per_burst`
    consume directly. Used to drive panel meta-regression on
    multi-env corpora where the implementation produces both per-burst
    structure (for the panel's burst dimension) AND env-level
    variation (for between-env covariates)."""
    cells: list[Mapping[str, object]] = []
    for env_name, (treatments, baselines) in envs.items():
        cells.extend(run_paired_phased_arms(
            treatments_per_burst=treatments,
            baselines_per_burst=baselines,
            seeds=seeds,
            treatment_arm=treatment_arm,
            baseline_arm=baseline_arm,
            env_name=env_name,
        ))
    return cells


def run_paired_phased_arms(
    *,
    treatments_per_burst: tuple[LinearGaussianSCM, ...],
    baselines_per_burst: tuple[LinearGaussianSCM, ...],
    seeds: Sequence[int],
    treatment_arm: str = 'treatment',
    baseline_arm: str = 'baseline',
    env_name: str = 'lg_scm',
) -> list[Mapping[str, object]]:
    """Phased two-arm sweep. Both arms must share the same
    `n_bursts` (and per-burst `n_steps`); only β coefficients
    differ. Same `seeds` across arms ensures shared noise streams
    so paired Δ cancels every epsilon — every paired Δ is
    structurally `Delta_beta(b) * beta_zy(b) * X_avg(seed, b)`.

    Returns a flat list of cell-dicts (the `Mapping[str, object]`
    shape `paired_g_per_burst.fn(...)` expects), with treatment
    cells first then baseline cells.
    """
    if len(treatments_per_burst) != len(baselines_per_burst):
        raise ValueError(
            f'run_paired_phased_arms: arms must share n_bursts; got '
            f'treatment={len(treatments_per_burst)}, '
            f'baseline={len(baselines_per_burst)}',
        )
    cells: list[Mapping[str, object]] = []
    for s in seeds:
        run_t, trace_t = run_phased_cell(
            treatments_per_burst,
            seed=s, arm_key=treatment_arm, env_name=env_name,
        )
        cells.append(merge_cell(run_t, trace_t))
    for s in seeds:
        run_b, trace_b = run_phased_cell(
            baselines_per_burst,
            seed=s, arm_key=baseline_arm, env_name=env_name,
        )
        cells.append(merge_cell(run_b, trace_b))
    return cells


__all__ = [
    'CellId',
    'PER_BURST_X_KEY',
    'PER_BURST_Y_KEY',
    'PER_BURST_Z_KEY',
    'merge_cell',
    'run_arm',
    'run_cell',
    'run_multi_env_paired_arms',
    'run_multi_env_paired_phased_arms',
    'run_paired_arms',
    'run_paired_phased_arms',
    'run_phased_cell',
]
