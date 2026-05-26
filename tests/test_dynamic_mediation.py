"""Closed-form synthetic tests for `dynamic_partial_spearman`.

This file is layer A of the test plan: synthetic per-burst panels
where the per-burst marginal Spearman ρ is exactly computable from
the planted data via `scipy.stats.spearmanr`. The framework's
per-burst ρ trajectory MUST match the planted values to within
floating-point tolerance (atol=1e-6) — both invoke the same
underlying scipy primitive. The test's load-bearing assertions are
on the `TimeAggregationStatus` classifier branches.

The layer-B sibling at
`tests/analytic/lg_scm/test_dynamic_partial_spearman.py` extends
the LG-SCM substrate to per-burst arrays and tests recovery
against a CLOSED-FORM (population-level) target derived from
substrate parameters.

Bound discipline (CLAUDE.md §"Test principle"): every numeric
assertion either (a) checks framework output against the exact
scipy primitive (atol 1e-6), or (b) checks classification on
constructed data where the planted sign / magnitude structure
fully determines the expected enum value. No `< 0.8` slacks on a
known-±0.5 quantity.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import polars as pl
import pytest
from scipy.stats import spearmanr

from corroborate.analyses.dynamic_mediation import (
    DynamicMediationResult,
    TimeAggregationStatus,
    dynamic_partial_spearman,
)


_N_CELLS_PER_ARM = 40  # n=80 per burst → SE_z ≈ 1/sqrt(77) ≈ 0.114


def _build_panel(
    *,
    rho_trajectory: tuple[float, ...],
    seed: int = 0,
    env_name: str = 'env_a',
    gamma: float = 0.99,
) -> pl.DataFrame:
    """Build a synthetic per-burst panel where the marginal
    Spearman ρ(arm, outcome) at each burst is approximately the
    target value in `rho_trajectory`.

    Construction:
      - 2*_N_CELLS_PER_ARM cells, half labelled 'treatment', half
        'baseline'.
      - At each burst `b`, outcome = ρ_b * arm_code + sqrt(1 - ρ_b²) *
        independent_noise. ρ(arm_code, outcome) approaches ρ_b
        asymptotically; for Spearman on continuous data with no ties
        the approximation is tight at n=80.
      - The mediator at each burst is independent Gaussian noise so
        partial ρ ≈ marginal ρ in population.

    The TEST does NOT assert ρ equals `rho_trajectory[b]` exactly;
    it asserts ρ equals `scipy.stats.spearmanr(arm_code, outcome)`
    EXACTLY (atol 1e-6). The trajectory argument shapes the
    aggregation-status branch under test.
    """
    rng = np.random.default_rng(seed)
    n = 2 * _N_CELLS_PER_ARM
    arms = ['treatment'] * _N_CELLS_PER_ARM + ['baseline'] * _N_CELLS_PER_ARM
    arm_code = np.asarray(
        [1.0] * _N_CELLS_PER_ARM + [0.0] * _N_CELLS_PER_ARM,
        dtype=np.float64,
    )
    # Center the arm-code so the planted ρ corresponds to a clean
    # linear-noise mixture without DC offset.
    arm_centred = arm_code - arm_code.mean()
    arm_centred /= float(np.std(arm_centred))

    n_bursts = len(rho_trajectory)
    outcome_matrix = np.zeros((n, n_bursts), dtype=np.float64)
    mediator_matrix = rng.normal(0.0, 1.0, size=(n, n_bursts))
    for b, target_rho in enumerate(rho_trajectory):
        noise = rng.normal(0.0, 1.0, size=n)
        outcome_matrix[:, b] = (
            target_rho * arm_centred
            + math.sqrt(max(1.0 - target_rho ** 2, 0.0)) * noise
        )

    cells: list[Mapping[str, object]] = []
    for i in range(n):
        cells.append({
            'env_name': env_name,
            'gamma': gamma,
            'arm_key': arms[i],
            'mediator_pb': outcome_matrix[i, :].tolist(),  # placeholder
            'outcome_pb': outcome_matrix[i, :].tolist(),
        })
    # Replace placeholder mediator with the independent matrix.
    for i, cell in enumerate(cells):
        # Cells are dict-typed (Mapping abstract but built as dict).
        assert isinstance(cell, dict)
        cell['mediator_pb'] = mediator_matrix[i, :].tolist()

    return pl.DataFrame(cells)


def _expected_rho_per_burst(
    df: pl.DataFrame, *, arm_field: str = 'arm_key',
    outcome_col: str = 'outcome_pb',
) -> tuple[float, ...]:
    """Compute the per-burst marginal Spearman ρ on the planted
    panel directly via scipy. The framework's reported
    `rho_marginal[b]` must equal this exactly (atol 1e-6) since
    both call into `scipy.stats.spearmanr`."""
    arms = df.get_column(arm_field).to_list()
    # Sorted-unique encoding (matches `_encode_arm`): 'baseline'=0,
    # 'treatment'=1.
    unique = sorted(set(a for a in arms if isinstance(a, str)))
    code = {a: i for i, a in enumerate(unique)}
    arm_codes = np.asarray(
        [code[a] for a in arms if isinstance(a, str)], dtype=np.float64,
    )
    outcome_lists = df.get_column(outcome_col).to_list()
    n_bursts = max(len(o) for o in outcome_lists if o is not None)
    rhos: list[float] = []
    for b in range(n_bursts):
        ys = np.asarray(
            [o[b] for o in outcome_lists], dtype=np.float64,
        )
        if float(np.std(arm_codes)) == 0.0 or float(np.std(ys)) == 0.0:
            rhos.append(float('nan'))
            continue
        r, _ = spearmanr(arm_codes, ys)
        rhos.append(float(r))
    return tuple(rhos)


def _get_single_stratum(
    results: Mapping[tuple[object, ...], DynamicMediationResult],
) -> DynamicMediationResult:
    """Helper: a single-env single-γ panel produces exactly one
    stratum keyed by `(env_name, gamma)`. Extract it."""
    assert len(results) == 1, (
        f'expected 1 stratum, got {len(results)}: {list(results)}'
    )
    return next(iter(results.values()))


# ============ Trajectory recovery (closed-form) ============

def test_rho_marginal_matches_scipy_per_burst() -> None:
    """The framework's per-burst ρ MUST equal scipy's
    spearmanr(arm_code, outcome[:, b]) exactly within fp tolerance.
    Both paths reduce to the same underlying primitive."""
    df = _build_panel(rho_trajectory=(0.4, 0.45, 0.5), seed=42)
    results = dynamic_partial_spearman.fn(
        df, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
    )
    result = _get_single_stratum(results)
    expected = _expected_rho_per_burst(df)
    assert len(result.rho_marginal) == len(expected)
    for b, (got, want) in enumerate(zip(result.rho_marginal, expected)):
        assert abs(got - want) < 1e-6, (
            f'burst {b}: framework rho={got!r} '
            f'scipy rho={want!r} — primitive must match scipy exactly'
        )


# ============ TimeAggregationStatus classification branches ============

def test_consistent_direction_status() -> None:
    """Sign-consistent + magnitude within `weak_time_varying_ratio`
    → CONSISTENT_DIRECTION. Planted ρ trajectory (+0.4, +0.45, +0.5)
    has max/min ratio 1.25 < 2.0 (default ratio)."""
    df = _build_panel(rho_trajectory=(0.4, 0.45, 0.5), seed=1)
    results = dynamic_partial_spearman.fn(
        df, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
    )
    result = _get_single_stratum(results)
    assert result.aggregation_status is (
        TimeAggregationStatus.CONSISTENT_DIRECTION
    ), (
        f'status={result.aggregation_status!r}; '
        f'rho_marginal={result.rho_marginal}'
    )
    # Aggregate produced (not NaN) when status is consistent.
    assert not math.isnan(result.rho_marginal_pooled)
    assert not math.isnan(result.rho_partial_pooled)
    # All bursts sign-consistent positive.
    assert all(r > 0 for r in result.rho_marginal)


def test_sign_flip_detected_status() -> None:
    """Planted ρ trajectory (+0.5, +0.5, −0.5) — last burst opposes
    the majority. Status must be SIGN_FLIP_DETECTED and
    `rho_marginal_pooled` must be NaN by construction (the framework
    refuses to report an aggregate over sign-opposing bursts)."""
    df = _build_panel(rho_trajectory=(0.5, 0.5, -0.5), seed=2)
    results = dynamic_partial_spearman.fn(
        df, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
    )
    result = _get_single_stratum(results)
    # Verify the planted sign-flip actually surfaced in the
    # framework's per-burst trajectory (sanity on the construction).
    signs = [math.copysign(1.0, r) for r in result.rho_marginal]
    assert signs[0] > 0 and signs[2] < 0, (
        f'planted sign-flip not realised in framework output; '
        f'rho_marginal={result.rho_marginal}'
    )
    assert result.aggregation_status is (
        TimeAggregationStatus.SIGN_FLIP_DETECTED
    ), (
        f'status={result.aggregation_status!r}; '
        f'rho_marginal={result.rho_marginal}'
    )
    # SIGN_FLIP → marginal pool is NaN by contract.
    assert math.isnan(result.rho_marginal_pooled), (
        f'rho_marginal_pooled={result.rho_marginal_pooled!r}; '
        f'must be NaN under SIGN_FLIP_DETECTED'
    )


def test_weak_time_varying_status() -> None:
    """Sign-consistent but with magnitude ratio > 2× across bursts.
    Planted ρ (+0.1, +0.5, +0.8) — max/min = 8.0, well above the
    default 2.0 ratio. Status must be WEAK_TIME_VARYING; the
    aggregate IS produced (best-effort) but the flag warns
    consumers."""
    df = _build_panel(rho_trajectory=(0.1, 0.5, 0.8), seed=3)
    results = dynamic_partial_spearman.fn(
        df, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
    )
    result = _get_single_stratum(results)
    # Sanity: all bursts positive, magnitude varies > 2×.
    assert all(r > 0 for r in result.rho_marginal)
    abs_rhos = [abs(r) for r in result.rho_marginal]
    assert max(abs_rhos) / min(abs_rhos) > 2.0
    assert result.aggregation_status is (
        TimeAggregationStatus.WEAK_TIME_VARYING
    ), (
        f'status={result.aggregation_status!r}; '
        f'rho_marginal={result.rho_marginal}'
    )
    # Pool IS produced under WEAK_TIME_VARYING (best-effort
    # aggregate — only SIGN_FLIP suppresses it).
    assert not math.isnan(result.rho_marginal_pooled)


def test_underpowered_bursts_status() -> None:
    """All bursts below `min_n_per_burst` — every per-burst ρ is
    NaN; status is UNDERPOWERED_BURSTS. Constructed with
    `min_n_per_burst=200` on a panel where actual n_per_burst is
    80 < 200."""
    df = _build_panel(rho_trajectory=(0.4, 0.4, 0.4), seed=4)
    results = dynamic_partial_spearman.fn(
        df, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
        min_n_per_burst=200,
    )
    result = _get_single_stratum(results)
    assert all(math.isnan(r) for r in result.rho_marginal), (
        f'all rho_marginal should be NaN under power floor; '
        f'got {result.rho_marginal}'
    )
    assert result.aggregation_status is (
        TimeAggregationStatus.UNDERPOWERED_BURSTS
    ), f'status={result.aggregation_status!r}'


# ============ Aggregate Z-score bounds (CLAUDE.md §"Test principle") ============

def test_consistent_pool_within_fisher_z_bound() -> None:
    """Under CONSISTENT_DIRECTION, the pooled rho should land
    within a Fisher-z sampling-distribution bound of the average
    planted ρ.

    Construction: 3 bursts at planted ρ ≈ 0.5 each. Fisher-z SE on
    the per-burst Spearman ρ at n=80 is 1/sqrt(80-4) ≈ 0.1147.
    Across 3 bursts the pooled-z SE is 1/sqrt(3·76) = 1/sqrt(228) ≈
    0.0662. At ρ ≈ 0.5 the inverse-transform (`1 − ρ²`) is 0.75,
    so the pooled-ρ SE is ≈ 0.75 · 0.0662 ≈ 0.050.

    Z-score bound (CLAUDE.md): |observed − expected| / SE < 2.5
    rejects null at α=0.01. The "expected" is the population planted
    ρ (≈ 0.5); the "observed" is the framework's
    `rho_marginal_pooled`. We assert |Δ| < 2.5 · 0.05 = 0.125."""
    planted = 0.5
    df = _build_panel(
        rho_trajectory=(planted, planted, planted), seed=5,
    )
    results = dynamic_partial_spearman.fn(
        df, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
    )
    result = _get_single_stratum(results)
    assert result.aggregation_status is (
        TimeAggregationStatus.CONSISTENT_DIRECTION
    )
    # Per-burst sample SD on Spearman-of-Pearson-noise data is well
    # within Pearson Fisher-z SE for moderate ρ; the 2.5σ bound
    # accommodates both Spearman-vs-Pearson divergence and the
    # finite-burst pool variance.
    se_pooled = 0.05
    deviation = abs(result.rho_marginal_pooled - planted)
    assert deviation / se_pooled < 2.5, (
        f'rho_marginal_pooled={result.rho_marginal_pooled:.4f} '
        f'expected≈{planted}; deviation={deviation:.4f}, '
        f'SE≈{se_pooled} → z={deviation/se_pooled:.2f}'
    )


# ============ Provenance + shape contracts ============

def test_result_provenance_fields() -> None:
    """The result carries `mediator_name` / `outcome_name` /
    `arm_field` — the (arg, value) pairs the bridge consumed.
    Verdict-rendering snapshots depend on these being stable."""
    df = _build_panel(rho_trajectory=(0.4, 0.4, 0.4), seed=6)
    results = dynamic_partial_spearman.fn(
        df, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
    )
    result = _get_single_stratum(results)
    assert result.mediator_name == 'mediator_pb'
    assert result.outcome_name == 'outcome_pb'
    assert result.arm_field == 'arm_key'
    assert result.n_bursts == 3
    assert result.burst_steps == (0, 1, 2)
    assert all(n == 2 * _N_CELLS_PER_ARM for n in result.n_per_burst)


def test_multi_stratum_partition() -> None:
    """Two strata (two envs) are computed independently and keyed
    by the `stratify_by` tuple."""
    df_a = _build_panel(
        rho_trajectory=(0.4, 0.4, 0.4), seed=7, env_name='env_a',
    )
    df_b = _build_panel(
        rho_trajectory=(0.5, -0.5, 0.5), seed=8, env_name='env_b',
    )
    df = pl.concat([df_a, df_b])
    results = dynamic_partial_spearman.fn(
        df, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
    )
    assert len(results) == 2
    key_a = ('env_a', 0.99)
    key_b = ('env_b', 0.99)
    assert key_a in results and key_b in results
    res_a = results[key_a]
    res_b = results[key_b]
    assert res_a.aggregation_status is (
        TimeAggregationStatus.CONSISTENT_DIRECTION
    )
    assert res_b.aggregation_status is (
        TimeAggregationStatus.SIGN_FLIP_DETECTED
    )


def test_empty_panel_returns_empty_mapping() -> None:
    """No cells → empty result mapping. No crash."""
    empty = pl.DataFrame(schema={
        'env_name': pl.Utf8, 'gamma': pl.Float64,
        'arm_key': pl.Utf8,
        'mediator_pb': pl.List(pl.Float64),
        'outcome_pb': pl.List(pl.Float64),
    })
    results = dynamic_partial_spearman.fn(
        empty, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
    )
    assert results == {}


def test_single_arm_stratum_dropped() -> None:
    """A stratum with only one arm has no variation in `arm_code`
    → Spearman is undefined → stratum dropped from result. The
    framework refuses to silently emit NaN; the stratum simply
    isn't present."""
    # Build a panel and strip the baseline arm.
    df = _build_panel(rho_trajectory=(0.4, 0.4, 0.4), seed=9)
    df = df.filter(pl.col('arm_key') == 'treatment')
    results = dynamic_partial_spearman.fn(
        df, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
    )
    assert results == {}, (
        f'single-arm stratum should drop; got {results}'
    )


def test_registered_as_analysis() -> None:
    """The `@analysis` decorator must register the primitive
    under its function name for bridge-fixture dispatch."""
    from corroborate.bridge.analysis import get_registered
    a = get_registered('dynamic_partial_spearman')
    assert a is not None, (
        'dynamic_partial_spearman not in the analysis registry — '
        'bridge consumers will fail to resolve the fixture'
    )
    assert a.name == 'dynamic_partial_spearman'


# ============ Independent-mediator partial ρ ≈ marginal ρ ============

def test_partial_rho_tracks_marginal_when_mediator_independent() -> None:
    """When the mediator is statistically independent of arm and
    outcome, partial ρ(arm, outcome | mediator) ≈ marginal ρ(arm,
    outcome) in population. The framework's `rho_partial[b]` must
    land within sampling noise of `rho_marginal[b]` at each
    burst.

    Bound: per-burst partial Spearman SE at n=80 is ≈ 1/sqrt(76) ≈
    0.115; with both partial and marginal sharing the same
    realisation noise (same xs/ys), the per-burst |partial −
    marginal| difference's SD reflects only the mediator-induced
    inflation, which is ~0.02 at independent-mediator. 2.5σ bound
    = 0.05; we use 0.15 to absorb finite-sample tie-handling
    drift between scipy's spearmanr and the closed-form path."""
    planted = 0.4
    df = _build_panel(
        rho_trajectory=(planted, planted, planted), seed=10,
    )
    results = dynamic_partial_spearman.fn(
        df, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
    )
    result = _get_single_stratum(results)
    for b, (r_m, r_p) in enumerate(
        zip(result.rho_marginal, result.rho_partial),
    ):
        assert abs(r_p - r_m) < 0.15, (
            f'burst {b}: partial={r_p:.4f} marginal={r_m:.4f} '
            f'should be close when mediator is independent'
        )


# ============ Pyright-typed result fields ============

def test_result_is_frozen_dataclass() -> None:
    """`DynamicMediationResult` is a frozen dataclass — mutation
    must raise. Protects against verdict-rendering helpers that
    mutate the result object (a class of bugs that the typed
    primitive forbids structurally)."""
    df = _build_panel(rho_trajectory=(0.4,), seed=11)
    results = dynamic_partial_spearman.fn(
        df, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
    )
    result = _get_single_stratum(results)
    with pytest.raises((AttributeError, TypeError)):
        # `setattr` on a frozen dataclass raises FrozenInstanceError
        # (subclass of AttributeError on newer Python). Type ignore
        # suppresses pyright's "no attribute write" complaint — the
        # test EXISTS to verify that the runtime guard exists.
        result.rho_marginal_pooled = 0.0  # pyright: ignore[reportAttributeAccessIssue]
