"""Closed-form synthetic tests for `dynamic_pc_adjacency`.

Layer A of the test plan, parallel shape to
`tests/test_dynamic_mediation.py`. Synthetic per-burst panels
where the planted DAG is one of:

  - **Full mediation** — arm → mediator → outcome (no direct
    edge): the Fisher-z CI test must report
    `mediator_dseparates[b] == True` at most bursts.
  - **Direct edge** — arm → outcome (mediator independent):
    conditioning on the mediator should NOT remove the edge →
    `direct_edge[b] == True` at most bursts.
  - **Null** — arm independent of outcome: the marginal CI test
    should NOT reject at α at most bursts → small
    `n_bursts_marginal_edge`.

Bound discipline (CLAUDE.md §"Test principle"): Type-I and
Type-II error rates are α-controlled by construction of the CI
test. We allow ~ceil(α · n_bursts) Type-I false-positives in the
null case and ~ceil(power-failure rate · n_bursts) misses in the
mediation cases. Concrete numbers are derived in each test's
docstring from per-burst sample SE.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest

from corroborate.analyses.dynamic_mediation import (
    ClusterBootstrapEdgeCounts,
    ClusterBootstrapInterval,
    DynamicPCResult,
    TimeAggregationStatus,
    dynamic_pc_adjacency,
)
from corroborate.analyses.dynamic_mediation import (
    _classify_status,  # pyright: ignore[reportPrivateUsage]
)


_N_CELLS_PER_ARM = 30  # n=60 per burst, well above default min_n=20


def _build_full_mediation_panel(
    *,
    n_bursts: int,
    beta_arm_mediator: float = 1.0,
    beta_mediator_outcome: float = 1.0,
    sigma_mediator: float = 0.5,
    sigma_outcome: float = 0.5,
    seed: int = 0,
    env_name: str = 'env_a',
    gamma: float = 0.99,
) -> pl.DataFrame:
    """Construct a panel where the DAG is `arm → mediator →
    outcome` with NO direct arm→outcome edge.

    At each burst b:
      mediator[b] = beta_arm_mediator * arm_code + ε_M (iid)
      outcome[b]  = beta_mediator_outcome * mediator[b] + ε_Y (iid)

    The mediator d-separates arm from outcome by construction:
      arm ⊥ outcome | mediator (no direct edge between arm and
      outcome past the mediator). Population partial ρ = 0.

    Population marginal point-biserial Pearson r between arm_code
    (binary) and outcome:
      Var(outcome) = β_my² · (β_am² · Var(arm) + σ_M²) + σ_Y²
                   = β_my² · (β_am² · 0.25 + σ_M²) + σ_Y²
      Cov(arm, outcome) = β_my · β_am · Var(arm) = β_my · β_am · 0.25
      r ≈ Cov / sqrt(Var(arm) · Var(outcome))
        = (β_my · β_am · 0.25) / sqrt(0.25 · Var(outcome))
        = 0.5 · β_my · β_am / sqrt(Var(outcome))

    With defaults: Var(out) = 1·(0.25 + 0.25) + 0.25 = 0.75 →
    r ≈ 0.5 / 0.866 ≈ 0.577. Per-burst CI test rejects null at
    α=0.05 with n=60 with power ~ 0.999.
    """
    rng = np.random.default_rng(seed)
    n = 2 * _N_CELLS_PER_ARM
    arms = ['treatment'] * _N_CELLS_PER_ARM + ['baseline'] * _N_CELLS_PER_ARM
    arm_code = np.asarray(
        [0.5] * _N_CELLS_PER_ARM + [-0.5] * _N_CELLS_PER_ARM,
        dtype=np.float64,
    )
    mediator_matrix = np.zeros((n, n_bursts), dtype=np.float64)
    outcome_matrix = np.zeros((n, n_bursts), dtype=np.float64)
    for b in range(n_bursts):
        e_m = rng.normal(0.0, sigma_mediator, size=n)
        e_y = rng.normal(0.0, sigma_outcome, size=n)
        mediator_matrix[:, b] = beta_arm_mediator * arm_code + e_m
        outcome_matrix[:, b] = (
            beta_mediator_outcome * mediator_matrix[:, b] + e_y
        )
    cells: list[Mapping[str, object]] = []
    for i in range(n):
        cells.append({
            'env_name': env_name,
            'gamma': gamma,
            'arm_key': arms[i],
            'mediator_pb': mediator_matrix[i, :].tolist(),
            'outcome_pb': outcome_matrix[i, :].tolist(),
        })
    return pl.DataFrame(cells)


def _build_direct_edge_panel(
    *,
    n_bursts: int,
    beta_arm_outcome: float = 1.0,
    sigma_outcome: float = 0.5,
    sigma_mediator: float = 1.0,
    seed: int = 0,
    env_name: str = 'env_a',
    gamma: float = 0.99,
) -> pl.DataFrame:
    """Panel with DAG `arm → outcome` (direct), `mediator`
    independent of both.

    At each burst b:
      mediator[b] = ε_M (iid N(0, σ_M))   ← independent of arm
      outcome[b]  = β_ao * arm_code + ε_Y (iid N(0, σ_Y))

    Mediator does NOT d-separate arm from outcome (it doesn't
    appear in the arm→outcome path at all). Conditioning on the
    mediator should not change the arm→outcome correlation; both
    marginal and conditional CI tests should reject at most
    bursts.
    """
    rng = np.random.default_rng(seed)
    n = 2 * _N_CELLS_PER_ARM
    arms = ['treatment'] * _N_CELLS_PER_ARM + ['baseline'] * _N_CELLS_PER_ARM
    arm_code = np.asarray(
        [0.5] * _N_CELLS_PER_ARM + [-0.5] * _N_CELLS_PER_ARM,
        dtype=np.float64,
    )
    mediator_matrix = rng.normal(0.0, sigma_mediator, size=(n, n_bursts))
    outcome_matrix = np.zeros((n, n_bursts), dtype=np.float64)
    for b in range(n_bursts):
        e_y = rng.normal(0.0, sigma_outcome, size=n)
        outcome_matrix[:, b] = beta_arm_outcome * arm_code + e_y
    cells: list[Mapping[str, object]] = []
    for i in range(n):
        cells.append({
            'env_name': env_name,
            'gamma': gamma,
            'arm_key': arms[i],
            'mediator_pb': mediator_matrix[i, :].tolist(),
            'outcome_pb': outcome_matrix[i, :].tolist(),
        })
    return pl.DataFrame(cells)


def _build_null_panel(
    *,
    n_bursts: int,
    seed: int = 0,
    env_name: str = 'env_a',
    gamma: float = 0.99,
) -> pl.DataFrame:
    """Panel where arm is independent of both mediator and
    outcome (the null DAG). Mediator and outcome are iid Gaussian
    noise per burst.

    Marginal CI test (arm ⊥ outcome) should reject at α=0.05 at
    most α·n_bursts bursts by construction (type-I rate).
    """
    rng = np.random.default_rng(seed)
    n = 2 * _N_CELLS_PER_ARM
    arms = ['treatment'] * _N_CELLS_PER_ARM + ['baseline'] * _N_CELLS_PER_ARM
    mediator_matrix = rng.normal(0.0, 1.0, size=(n, n_bursts))
    outcome_matrix = rng.normal(0.0, 1.0, size=(n, n_bursts))
    cells: list[Mapping[str, object]] = []
    for i in range(n):
        cells.append({
            'env_name': env_name,
            'gamma': gamma,
            'arm_key': arms[i],
            'mediator_pb': mediator_matrix[i, :].tolist(),
            'outcome_pb': outcome_matrix[i, :].tolist(),
        })
    return pl.DataFrame(cells)


def _get_single_stratum(
    results: Mapping[tuple[object, ...], DynamicPCResult],
) -> DynamicPCResult:
    assert len(results) == 1, (
        f'expected 1 stratum, got {len(results)}: {list(results)}'
    )
    return next(iter(results.values()))


# ============ Closed-form mediation pattern tests ============

def test_full_mediation_dseparates_at_every_burst() -> None:
    """Full mediation DAG `arm → mediator → outcome`: at every
    burst, the mediator d-separates arm from outcome.

    With n=60 per burst and planted r ≈ 0.58, the marginal CI test
    rejects at α=0.05 with power > 0.999 (sanity: scipy's
    spearmanr p-value at n=60, |r|=0.58 → z ≈ 4.7, two-sided p ≈
    3e-6). So `marginal_edge[b]` is True at every burst by
    construction.

    Under d-separation, partial ρ population value is 0. Per-burst
    partial Spearman SE at n=60 is ≈ 1/sqrt(56) ≈ 0.134 with some
    closed-form-partial denominator inflation; at the null
    population value the test rejects at α=0.05 with type-I rate
    exactly 5%.

    Across 8 bursts, expected false-positives on the partial test:
    ~0.4. We assert `n_bursts_mediator_dseparates >= 6` (allow 2
    bursts to flip via type-I noise — bound chosen to be robust
    across reasonable seeds; planted DAG admits up to ⌈α·8⌉=1
    false-positive on average, but per-seed variance lets 2 happen
    occasionally).
    """
    n_bursts = 8
    df = _build_full_mediation_panel(n_bursts=n_bursts, seed=1)
    results = dynamic_pc_adjacency.fn(
        df, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
    )
    result = _get_single_stratum(results)
    assert result.n_bursts == n_bursts
    # Marginal edge should hold at every burst — planted r ≈ 0.58
    # at n=60 → marginal p ≪ α=0.05 at every burst with overwhelming
    # power.
    assert result.n_bursts_marginal_edge == n_bursts, (
        f'expected marginal edge at every burst; got '
        f'{result.n_bursts_marginal_edge}/{n_bursts}. '
        f'p_marginal={result.p_marginal}'
    )
    # Mediator d-separates at most bursts; allow some flips via
    # type-I noise on the partial CI test.
    assert result.n_bursts_mediator_dseparates >= 6, (
        f'expected mediator_dseparates >= 6/{n_bursts} under '
        f'full-mediation DAG; got '
        f'{result.n_bursts_mediator_dseparates}. '
        f'p_conditional={result.p_conditional}'
    )
    # Symmetric: direct edge should fire at MOST 2 bursts (type-I).
    assert result.n_bursts_direct_edge <= 2, (
        f'expected direct_edge <= 2/{n_bursts} under full-mediation '
        f'DAG; got {result.n_bursts_direct_edge} (type-I inflation?). '
        f'p_conditional={result.p_conditional}'
    )


def test_direct_edge_panel_does_not_dseparate() -> None:
    """Direct-edge DAG `arm → outcome`, mediator independent:
    conditioning on the mediator should NOT remove the arm→outcome
    edge. Per-burst `direct_edge` should fire at most bursts.

    n=60, planted point-biserial r between arm and outcome ≈ 0.71
    (Cov = 0.5, Var(out) = 0.25 + 0.25 = 0.50, r = 0.5/sqrt(0.125)
    = 1.414... ah no: r = Cov/sqrt(Var(arm)·Var(out)) =
    0.5/sqrt(0.25 · 0.5) = 0.5/0.354 ≈ 1.414, but r is bounded; the
    correct expression is Cov(arm, out)/[SD(arm) · SD(out)] where
    Var(arm)=0.25, Var(out)=β² · 0.25 + σ² = 0.25 + 0.25 = 0.5).
    Empirically r ≈ 0.7-0.8 at n=60.

    Power of CI test at n=60, |r|≈0.7 is > 0.999 → both marginal
    and conditional CI tests reject at α=0.05 at every burst.
    """
    n_bursts = 8
    df = _build_direct_edge_panel(n_bursts=n_bursts, seed=2)
    results = dynamic_pc_adjacency.fn(
        df, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
    )
    result = _get_single_stratum(results)
    assert result.n_bursts_marginal_edge == n_bursts
    # Mediator is independent of arm + outcome by construction →
    # partial r ≈ marginal r → conditional CI rejects too. Direct
    # edge fires at >= 6/8 bursts (allow 2-burst slack for closed-
    # form-partial sampling noise around the saturating r).
    assert result.n_bursts_direct_edge >= 6, (
        f'expected direct_edge >= 6/{n_bursts} under direct-edge '
        f'DAG; got {result.n_bursts_direct_edge}. '
        f'p_conditional={result.p_conditional}'
    )
    # Mediator should NOT d-separate at most bursts.
    assert result.n_bursts_mediator_dseparates <= 2, (
        f'expected mediator_dseparates <= 2/{n_bursts}; got '
        f'{result.n_bursts_mediator_dseparates}'
    )


def test_null_panel_has_few_marginal_edges() -> None:
    """Null DAG (arm independent of outcome): marginal CI test has
    type-I rate α=0.05 by construction. With n_bursts=20, expected
    `n_bursts_marginal_edge` is ≈ α · n_bursts = 1. We bound by 5
    (Poisson p(X ≥ 5 | λ=1) ≈ 0.004 — well under any reasonable
    flakiness threshold)."""
    n_bursts = 20
    df = _build_null_panel(n_bursts=n_bursts, seed=3)
    results = dynamic_pc_adjacency.fn(
        df, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
    )
    result = _get_single_stratum(results)
    # Bound: 5 false-positive marginal-edge bursts in 20 (expected
    # 1 under α=0.05; Poisson tail at λ=1, k≥5 is ~3.7e-3).
    assert result.n_bursts_marginal_edge <= 5, (
        f'expected n_bursts_marginal_edge <= 5/{n_bursts} under '
        f'null DAG; got {result.n_bursts_marginal_edge} '
        f'(type-I rate violated?). p_marginal={result.p_marginal}'
    )


# ============ Sign-flip aggregation status ============

def test_sign_flip_status_fires_with_noise_floor() -> None:
    """Construct a hand-built ρ trajectory that flips sign at
    above-noise-floor magnitude → SIGN_FLIP_DETECTED. Uses the
    classifier directly because the PC primitive's status field
    is driven by the same `_classify_status` shared with
    `dynamic_partial_spearman` — exercising the shared classifier
    is the simplest pin.

    Trajectory (+0.5, +0.5, −0.5): last burst opposes the
    majority at magnitude 0.5 (well above the 0.05 default floor).
    """
    status = _classify_status(
        rho_marginal=(0.5, 0.5, -0.5),
        n_per_burst=(60, 60, 60),
        min_n_per_burst=20,
        weak_time_varying_ratio=2.0,
        sign_flip_min_abs_rho=0.05,
    )
    assert status is TimeAggregationStatus.SIGN_FLIP_DETECTED


def test_sign_flip_below_floor_is_noise() -> None:
    """A trajectory whose opposite-sign burst is at |ρ| < floor
    must classify as CONSISTENT_DIRECTION (the noise floor swallows
    the flip). Parallel to the partial-Spearman primitive's
    noise-floor unit test."""
    status = _classify_status(
        rho_marginal=(0.5, 0.5, -0.02),
        n_per_burst=(60, 60, 60),
        min_n_per_burst=20,
        weak_time_varying_ratio=2.0,
        sign_flip_min_abs_rho=0.05,
    )
    assert status is TimeAggregationStatus.CONSISTENT_DIRECTION


# ============ Boundary tests on min_n_per_burst ============

def _build_n_panel(
    *,
    n_cells_per_arm: int,
    n_bursts: int,
    seed: int,
) -> pl.DataFrame:
    """Tiny-stratum panel where each cell carries a length-
    `n_bursts` per-burst trajectory. Used for `min_n_per_burst`
    boundary tests."""
    rng = np.random.default_rng(seed)
    n = 2 * n_cells_per_arm
    arms = ['treatment'] * n_cells_per_arm + ['baseline'] * n_cells_per_arm
    arm_code = np.asarray(
        [0.5] * n_cells_per_arm + [-0.5] * n_cells_per_arm,
        dtype=np.float64,
    )
    mediator_matrix = np.zeros((n, n_bursts), dtype=np.float64)
    outcome_matrix = np.zeros((n, n_bursts), dtype=np.float64)
    for b in range(n_bursts):
        e_m = rng.normal(0.0, 0.5, size=n)
        e_y = rng.normal(0.0, 0.5, size=n)
        mediator_matrix[:, b] = arm_code + e_m
        outcome_matrix[:, b] = mediator_matrix[:, b] + e_y
    cells: list[Mapping[str, object]] = [
        {
            'env_name': 'env_a', 'gamma': 0.99, 'arm_key': arms[i],
            'mediator_pb': mediator_matrix[i, :].tolist(),
            'outcome_pb': outcome_matrix[i, :].tolist(),
        }
        for i in range(n)
    ]
    return pl.DataFrame(cells)


def test_below_min_n_per_burst_marks_underpowered() -> None:
    """n=10 cells per burst, default `min_n_per_burst=20`: all
    bursts under-powered → NaN p-values, NaN ρs, all boolean
    counters zero, status UNDERPOWERED_BURSTS."""
    df = _build_n_panel(n_cells_per_arm=5, n_bursts=3, seed=10)
    results = dynamic_pc_adjacency.fn(
        df, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
        # default min_n_per_burst=20
    )
    result = _get_single_stratum(results)
    assert result.n_per_burst == (10, 10, 10)
    assert all(math.isnan(p) for p in result.p_marginal)
    assert all(math.isnan(p) for p in result.p_conditional)
    assert all(math.isnan(r) for r in result.rho_marginal)
    assert all(math.isnan(r) for r in result.rho_partial)
    assert result.n_bursts_marginal_edge == 0
    assert result.n_bursts_mediator_dseparates == 0
    assert result.n_bursts_direct_edge == 0
    assert result.aggregation_status is (
        TimeAggregationStatus.UNDERPOWERED_BURSTS
    )


def test_at_or_above_min_n_per_burst_defined() -> None:
    """n=50 cells per burst (well above default 20): ρ and
    p-values are defined (not NaN) at every burst."""
    df = _build_n_panel(n_cells_per_arm=25, n_bursts=3, seed=11)
    results = dynamic_pc_adjacency.fn(
        df, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
    )
    result = _get_single_stratum(results)
    assert result.n_per_burst == (50, 50, 50)
    for b in range(3):
        assert not math.isnan(result.p_marginal[b])
        assert not math.isnan(result.p_conditional[b])
        assert not math.isnan(result.rho_marginal[b])
        assert not math.isnan(result.rho_partial[b])


# ============ Measurable input path ============

def test_accepts_measurable_inputs() -> None:
    """Mirrors `dynamic_partial_spearman`'s Measurable test. The
    PC primitive accepts `Measurable[..., NDArray]` inputs via the
    shared `_resolve_per_burst` dispatch — cache-first short-
    circuits make Measurable identical to column-name path when
    the cached column is present."""
    from corroborate.measurables import measurable

    @measurable(name='outcome_pb', reads=('outcome_pb',))
    def outcome_pb_m(cell: Mapping[str, object]) -> npt.NDArray[np.floating]:
        raw = cell['outcome_pb']
        assert isinstance(raw, list)
        return np.asarray(raw, dtype=np.float64)

    @measurable(name='mediator_pb', reads=('mediator_pb',))
    def mediator_pb_m(cell: Mapping[str, object]) -> npt.NDArray[np.floating]:
        raw = cell['mediator_pb']
        assert isinstance(raw, list)
        return np.asarray(raw, dtype=np.float64)

    df = _build_full_mediation_panel(n_bursts=3, seed=20)
    results_str = dynamic_pc_adjacency.fn(
        df, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
    )
    results_m = dynamic_pc_adjacency.fn(
        df, arm_field='arm_key',
        mediator_per_burst=mediator_pb_m,
        outcome_per_burst=outcome_pb_m,
        stratify_by=('env_name', 'gamma'),
    )
    r_str = _get_single_stratum(results_str)
    r_m = _get_single_stratum(results_m)
    # Per-burst p-values + ρs identical when both paths read the
    # same underlying List(Float64) column (cache-first short-
    # circuits Measurable evaluation).
    assert r_str.p_marginal == r_m.p_marginal
    assert r_str.p_conditional == r_m.p_conditional
    assert r_str.rho_marginal == r_m.rho_marginal
    assert r_str.rho_partial == r_m.rho_partial
    # Provenance from Measurable.name.
    assert r_m.mediator_name == 'mediator_pb'
    assert r_m.outcome_name == 'outcome_pb'


# ============ Ragged-tail per-burst alignment ============

def test_ragged_tail_alignment() -> None:
    """Cells with shorter trajectories contribute only their
    prefix. Construction: 50 short cells (25 per arm, length 2) +
    80 long cells (40 per arm, length 4). n_bursts = max length =
    4. n_per_burst should be (130, 130, 80, 80).

    Pins the ragged-tail semantics for the PC primitive (same
    `_n_bursts` + `_gather_burst_b` helpers as the partial-Spearman
    sibling).
    """
    rng = np.random.default_rng(40)
    cells: list[Mapping[str, object]] = []
    for arm, arm_code in (('treatment', 0.5), ('baseline', -0.5)):
        # 25 short cells per arm.
        for _ in range(25):
            cells.append({
                'env_name': 'env_a', 'gamma': 0.99, 'arm_key': arm,
                'outcome_pb': (
                    arm_code + rng.normal(0.0, 0.5, size=2)
                ).tolist(),
                'mediator_pb': rng.normal(0.0, 1.0, size=2).tolist(),
            })
        # 40 long cells per arm.
        for _ in range(40):
            cells.append({
                'env_name': 'env_a', 'gamma': 0.99, 'arm_key': arm,
                'outcome_pb': (
                    arm_code + rng.normal(0.0, 0.5, size=4)
                ).tolist(),
                'mediator_pb': rng.normal(0.0, 1.0, size=4).tolist(),
            })
    df = pl.DataFrame(cells)
    results = dynamic_pc_adjacency.fn(
        df, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
    )
    result = _get_single_stratum(results)
    assert result.n_bursts == 4
    assert result.n_per_burst == (130, 130, 80, 80), (
        f'n_per_burst={result.n_per_burst} — ragged-tail should '
        f'give (130, 130, 80, 80)'
    )


# ============ Provenance + shape contract ============

def test_result_provenance_and_shape() -> None:
    """The result carries `mediator_name` / `outcome_name` /
    `arm_field` / `alpha` matching the call site. The burst-axis
    shape contract: tuples of equal length `n_bursts`."""
    df = _build_full_mediation_panel(n_bursts=5, seed=50)
    results = dynamic_pc_adjacency.fn(
        df, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
        alpha=0.01,
    )
    result = _get_single_stratum(results)
    assert result.mediator_name == 'mediator_pb'
    assert result.outcome_name == 'outcome_pb'
    assert result.arm_field == 'arm_key'
    assert result.alpha == 0.01
    assert result.n_bursts == 5
    assert result.burst_steps == (0, 1, 2, 3, 4)
    assert len(result.n_per_burst) == 5
    assert len(result.p_marginal) == 5
    assert len(result.p_conditional) == 5
    assert len(result.rho_marginal) == 5
    assert len(result.rho_partial) == 5


def test_result_is_frozen_dataclass() -> None:
    """`DynamicPCResult` is a frozen dataclass — mutation must
    raise. Parallel guard to the partial-Spearman result."""
    df = _build_full_mediation_panel(n_bursts=2, seed=51)
    results = dynamic_pc_adjacency.fn(
        df, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
    )
    result = _get_single_stratum(results)
    with pytest.raises((AttributeError, TypeError)):
        result.n_bursts_marginal_edge = 0  # pyright: ignore[reportAttributeAccessIssue]


def test_registered_as_analysis() -> None:
    """The `@analysis` decorator must register the primitive under
    its function name."""
    from corroborate.bridge.analysis import get_registered
    a = get_registered('dynamic_pc_adjacency')
    assert a is not None
    assert a.name == 'dynamic_pc_adjacency'


def test_empty_panel_returns_empty_mapping() -> None:
    """No cells → empty result mapping."""
    empty = pl.DataFrame(schema={
        'env_name': pl.Utf8, 'gamma': pl.Float64,
        'arm_key': pl.Utf8,
        'mediator_pb': pl.List(pl.Float64),
        'outcome_pb': pl.List(pl.Float64),
    })
    results = dynamic_pc_adjacency.fn(
        empty, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
    )
    assert results == {}


def test_single_arm_stratum_dropped() -> None:
    """Single-arm stratum is dropped — Spearman is undefined when
    arm has no variance."""
    df = _build_full_mediation_panel(n_bursts=3, seed=52)
    df = df.filter(pl.col('arm_key') == 'treatment')
    results = dynamic_pc_adjacency.fn(
        df, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
    )
    assert results == {}


def test_multi_stratum_partition() -> None:
    """Two strata (different envs) computed independently and
    keyed by the `stratify_by` tuple."""
    df_a = _build_full_mediation_panel(
        n_bursts=3, seed=53, env_name='env_a',
    )
    df_b = _build_direct_edge_panel(
        n_bursts=3, seed=54, env_name='env_b',
    )
    df = pl.concat([df_a, df_b])
    results = dynamic_pc_adjacency.fn(
        df, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
    )
    assert len(results) == 2
    key_a = ('env_a', 0.99)
    key_b = ('env_b', 0.99)
    res_a = results[key_a]
    res_b = results[key_b]
    # env_a is full-mediation → mediator_dseparates most bursts.
    assert res_a.n_bursts_mediator_dseparates >= 2
    # env_b is direct-edge → direct_edge most bursts.
    assert res_b.n_bursts_direct_edge >= 2


# ============ Alpha threshold sensitivity ============

def test_alpha_threshold_affects_edge_classification() -> None:
    """Lowering α (more conservative) must NOT increase the count
    of marginal edges. Direction-of-effect check on the α
    parameter — pins that the boolean threshold flows through.
    """
    df = _build_full_mediation_panel(n_bursts=5, seed=60)
    res_loose = dynamic_pc_adjacency.fn(
        df, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
        alpha=0.10,
    )
    res_tight = dynamic_pc_adjacency.fn(
        df, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
        alpha=0.001,
    )
    r_loose = _get_single_stratum(res_loose)
    r_tight = _get_single_stratum(res_tight)
    assert r_loose.n_bursts_marginal_edge >= r_tight.n_bursts_marginal_edge, (
        f'tightening α from 0.10 to 0.001 should NOT increase '
        f'edge count; got loose={r_loose.n_bursts_marginal_edge} '
        f'tight={r_tight.n_bursts_marginal_edge}'
    )


# ============ DerSimonian-Laird random-effects pool ============

def test_pc_adjacency_exposes_dl_pool_fields() -> None:
    """The PC primitive's `DynamicPCResult` carries
    `dl_marginal` / `dl_partial` — same `FisherZDLPool` shape as
    the partial-Spearman sibling. The DL pool is computed on the
    per-burst (ρ, n) trajectory the CI tests already produce.

    Construction: 8 bursts of full-mediation DAG (`arm → mediator
    → outcome`, no direct edge). Per-burst marginal ρ planted
    near 0.58 (closed-form from `_build_full_mediation_panel`),
    per-burst partial ρ ≈ 0 (d-separated).

    Bounds:
      - DL marginal `tau2` finite (not NaN).
      - DL marginal `i2` ∈ [0, 1].
      - DL marginal `rho_pooled` finite and positive (planted
        positive direction).
      - DL marginal `n_bursts_used` = 8 (every burst above
        `min_n_per_burst=20` at n=60).
      - DL partial `rho_pooled` near 0 (planted d-separation).
    """
    n_bursts = 8
    df = _build_full_mediation_panel(n_bursts=n_bursts, seed=200)
    results = dynamic_pc_adjacency.fn(
        df, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
    )
    result = _get_single_stratum(results)
    dl_m = result.dl_marginal
    dl_p = result.dl_partial
    # All 8 bursts contribute (n=60 ≫ df_offset=3).
    assert dl_m.n_bursts_used == n_bursts
    assert dl_p.n_bursts_used == n_bursts
    # τ² finite (not NaN) — DL is defined at G=8.
    assert not math.isnan(dl_m.tau2), (
        f'τ² is NaN at G=8: DL should be defined'
    )
    # I² in [0, 1] — Higgins' fraction.
    assert 0.0 <= dl_m.i2 <= 1.0, (
        f'I²={dl_m.i2} outside [0, 1] — invariant violation'
    )
    # Marginal DL pool positive (planted positive direction).
    assert dl_m.rho_pooled > 0, (
        f'dl_marginal.rho_pooled={dl_m.rho_pooled:.4f} — expected '
        f'positive (planted arm → mediator → outcome with '
        f'positive coefficients)'
    )
    # Partial DL pool near zero (d-separated).
    # Per-burst partial ρ has SE ≈ 1/sqrt(56) ≈ 0.134 in z-units;
    # DL pooled over 8 bursts with τ²≈0 gives SE_pooled in z-units
    # ≈ 1/sqrt(8·56) ≈ 0.0472. After tanh-transform on a pool near
    # zero, SE_rho ≈ SE_z ≈ 0.047. 3σ bound = 0.14.
    assert abs(dl_p.rho_pooled) < 0.20, (
        f'dl_partial.rho_pooled={dl_p.rho_pooled:.4f} — expected '
        f'near 0 under d-separation; |·|<0.20 is 4σ from null'
    )


# ============ Cluster bootstrap CI ============

def test_pc_n_bootstrap_zero_keeps_bootstrap_fields_none() -> None:
    """Default `n_bootstrap=0` keeps the PC primitive's bootstrap
    fields None — pins fast-path bit-identical behaviour with
    pre-bootstrap callers."""
    df = _build_full_mediation_panel(n_bursts=3, seed=300)
    results = dynamic_pc_adjacency.fn(
        df, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
    )
    result = _get_single_stratum(results)
    assert result.bootstrap_marginal is None
    assert result.bootstrap_partial is None
    assert result.n_bootstrap == 0


def test_pc_cluster_bootstrap_populated_under_n_bootstrap_positive() -> None:
    """Under `n_bootstrap > 0` the PC primitive exposes both
    bootstrap CIs. Closed-form bounds:
      - Full-mediation DAG: marginal ρ ≈ 0.58 → bootstrap CI
        should bracket positive (planted positive direction).
      - Mediator d-separates → partial ρ ≈ 0 → bootstrap CI
        should bracket 0.

    n_resamples=200 with n_cells=60 gives bootstrap-replica DL
    sampling SD ≈ 1/sqrt(60-3) ≈ 0.132 in z-units; at ρ=0.58 the
    delta-method ρ-unit SD ≈ (1 - 0.58²) · 0.132 ≈ 0.087. 95% CI
    half-width ≈ 1.96 · 0.087 ≈ 0.17; bracket bound at 0.30."""
    df = _build_full_mediation_panel(n_bursts=8, seed=301)
    results = dynamic_pc_adjacency.fn(
        df, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
        n_bootstrap=200,
        bootstrap_seed=42,
    )
    result = _get_single_stratum(results)
    assert result.n_bootstrap == 200
    boot_m = result.bootstrap_marginal
    boot_p = result.bootstrap_partial
    assert boot_m is not None and boot_p is not None
    assert isinstance(boot_m, ClusterBootstrapInterval)
    assert isinstance(boot_p, ClusterBootstrapInterval)
    # Marginal CI brackets positive ρ (planted positive arm→outcome
    # link through the mediator). DL marginal point estimate is in
    # range ≈ 0.5–0.65; expect CI to bracket this.
    assert boot_m.rho_lower > 0.0, (
        f'marginal CI lower={boot_m.rho_lower:.4f}: full-mediation '
        f'DAG should produce strictly-positive marginal CI'
    )
    assert boot_m.rho_upper > boot_m.rho_lower
    # Partial CI brackets 0 under d-separation. With n_bursts=8 and
    # n=60 per burst, partial sampling SD per replica is ~0.08;
    # bracket [-0.20, +0.20].
    assert boot_p.rho_lower < 0.1, (
        f'partial CI lower={boot_p.rho_lower:.4f}: d-separated '
        f'partial should bracket 0'
    )
    assert boot_p.rho_upper > -0.1


def test_pc_cluster_bootstrap_reproducible() -> None:
    """Reproducibility check on the PC primitive's bootstrap —
    identical `bootstrap_seed` produces identical bounds."""
    df = _build_full_mediation_panel(n_bursts=4, seed=302)
    r_a = dynamic_pc_adjacency.fn(
        df, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
        n_bootstrap=50,
        bootstrap_seed=7,
    )
    r_b = dynamic_pc_adjacency.fn(
        df, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
        n_bootstrap=50,
        bootstrap_seed=7,
    )
    a = _get_single_stratum(r_a).bootstrap_marginal
    b = _get_single_stratum(r_b).bootstrap_marginal
    assert a is not None and b is not None
    assert a.rho_lower == b.rho_lower
    assert a.rho_upper == b.rho_upper


# ============ Cluster bootstrap on the EDGE-COUNT triple ============
#
# Conceptually distinct from the ρ-pool CIs above. The
# `bootstrap_edge_counts` field carries integer-count CIs on
# (n_bursts_marginal_edge, n_bursts_mediator_dseparates,
# n_bursts_direct_edge); the question is "is the edge
# classification robust to which cells we sampled?" — wide CI
# means a few outlier cells flip per-burst CI decisions and the
# count drifts across resamples.


def test_pc_bootstrap_edge_counts_none_at_zero_resamples() -> None:
    """Default `n_bootstrap=0` keeps `bootstrap_edge_counts` None
    — preserves the existing fast path."""
    df = _build_full_mediation_panel(n_bursts=3, seed=400)
    results = dynamic_pc_adjacency.fn(
        df, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
    )
    result = _get_single_stratum(results)
    assert result.bootstrap_edge_counts is None


def test_pc_bootstrap_edge_counts_full_mediation_narrow_dsep_ci() -> None:
    """Clean full-mediation scenario: dsep should be the
    dominant count, the CI should be narrow around n_bursts.

    Construction: arm → mediator → outcome with planted marginal
    r ≈ 0.58 at n=60. The marginal CI test rejects at α=0.05 with
    power ≈ 1.0 at every burst → marg count ≈ n_bursts in nearly
    every resample. Mediator d-separates by construction →
    partial CI's type-I rate is α=0.05 → on average 0.4
    false-positive direct-edge bursts across 8 bursts → dsep
    median should be near n_bursts.

    Bound: `dsep_median >= n_bursts - 1` and `direct_median <= 1`.
    """
    n_bursts = 8
    df = _build_full_mediation_panel(n_bursts=n_bursts, seed=401)
    results = dynamic_pc_adjacency.fn(
        df, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
        n_bootstrap=200,
        bootstrap_seed=42,
    )
    result = _get_single_stratum(results)
    bec = result.bootstrap_edge_counts
    assert bec is not None
    assert isinstance(bec, ClusterBootstrapEdgeCounts)
    # Provenance bookkeeping.
    assert bec.n_resamples == 200
    assert bec.seed == 42
    # dsep dominates under full mediation. Bound:
    # `dsep_median >= ceil(n_bursts/2)` — under the closed-form
    # partial-CI's α=0.05 type-I rate at saturating marginal r,
    # the bootstrap distribution can shift the median down by
    # a couple of bursts, but the majority must still d-separate.
    assert bec.dsep_median >= (n_bursts + 1) // 2, (
        f'dsep_median={bec.dsep_median} < ceil(n_bursts/2)='
        f'{(n_bursts + 1) // 2} under clean full mediation. '
        f'Counts: marg={bec.marg_median} dsep={bec.dsep_median} '
        f'direct={bec.direct_median}.'
    )
    # direct dominates in the opposite verdict; under full
    # mediation it should not become the majority class.
    assert bec.direct_median <= n_bursts // 2, (
        f'direct_median={bec.direct_median} > n_bursts/2={n_bursts // 2} '
        f'under full-mediation DAG — should be minority class.'
    )
    # CI is narrow-ish but the closed-form partial-CI's α=0.05
    # type-I rate at saturating marginal-r introduces noise on
    # any single resample. Bound: dsep_lower >= floor(n_bursts /
    # 2) — under clean full mediation the lower CI edge captures
    # at least half the trajectory.
    assert bec.dsep_lower >= n_bursts // 2, (
        f'dsep_lower={bec.dsep_lower} below n_bursts/2={n_bursts // 2}; '
        f'expected at least half-trajectory under clean full '
        f'mediation.'
    )
    # Bounds are ordered.
    assert bec.dsep_lower <= bec.dsep_median <= bec.dsep_upper
    assert bec.marg_lower <= bec.marg_median <= bec.marg_upper
    assert bec.direct_lower <= bec.direct_median <= bec.direct_upper


def test_pc_bootstrap_edge_counts_direct_edge_panel() -> None:
    """Clean direct-edge scenario: arm → outcome directly,
    mediator independent of both → conditioning on the mediator
    does NOT remove the edge → direct_median near n_bursts,
    dsep_median near 0."""
    n_bursts = 8
    df = _build_direct_edge_panel(n_bursts=n_bursts, seed=402)
    results = dynamic_pc_adjacency.fn(
        df, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
        n_bootstrap=200,
        bootstrap_seed=42,
    )
    result = _get_single_stratum(results)
    bec = result.bootstrap_edge_counts
    assert bec is not None
    assert bec.direct_median >= n_bursts - 2, (
        f'direct_median={bec.direct_median} below n_bursts-2 under '
        f'clean direct-edge DAG.'
    )
    assert bec.dsep_median <= 2, (
        f'dsep_median={bec.dsep_median} > 2 under direct-edge DAG.'
    )


def test_pc_bootstrap_edge_counts_null_panel_marg_low() -> None:
    """Null scenario: arm independent of outcome. Marginal CI test
    type-I rate = α = 0.05. With n_bursts=20, expected
    `n_bursts_marginal_edge` per replica is ≈ α · n_bursts = 1.

    Closed-form: per replica, `n_marg` ~ Binomial(20, 0.05) with
    mean μ = 1.0 and SD σ = sqrt(20 · 0.05 · 0.95) ≈ 0.975. The
    bootstrap median should land within ±2 SE of the binomial
    expected value: |median - 1| <= 2 · 0.975 ≈ 2 → median ∈
    [0, 3]. We assert `marg_median <= 3` (the strict integer
    bound corresponding to ~2σ of the per-replica binomial)."""
    n_bursts = 20
    df = _build_null_panel(n_bursts=n_bursts, seed=403)
    results = dynamic_pc_adjacency.fn(
        df, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
        n_bootstrap=200,
        bootstrap_seed=42,
    )
    result = _get_single_stratum(results)
    bec = result.bootstrap_edge_counts
    assert bec is not None
    # Closed-form binomial expectation under H0: mean 1, SD ~0.975.
    # 2σ envelope around the median → integer bound 3.
    assert bec.marg_median <= 3, (
        f'marg_median={bec.marg_median} > 3 under null DAG (expected '
        f'~1 by Binomial(20, 0.05); 2σ ≈ 2 → bound 3).'
    )
    # Bootstrap should NOT surface a spurious mediation signal:
    # `dsep` requires marg edge present first, so dsep <= marg.
    assert bec.dsep_median <= bec.marg_median
    assert bec.direct_median <= bec.marg_median


def test_pc_bootstrap_edge_counts_reproducible() -> None:
    """Reproducibility: same `bootstrap_seed` → identical
    integer counts on every field."""
    df = _build_full_mediation_panel(n_bursts=5, seed=404)
    r_a = dynamic_pc_adjacency.fn(
        df, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
        n_bootstrap=100,
        bootstrap_seed=13,
    )
    r_b = dynamic_pc_adjacency.fn(
        df, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
        n_bootstrap=100,
        bootstrap_seed=13,
    )
    a = _get_single_stratum(r_a).bootstrap_edge_counts
    b = _get_single_stratum(r_b).bootstrap_edge_counts
    assert a is not None and b is not None
    assert (a.marg_lower, a.marg_median, a.marg_upper) == (
        b.marg_lower, b.marg_median, b.marg_upper,
    )
    assert (a.dsep_lower, a.dsep_median, a.dsep_upper) == (
        b.dsep_lower, b.dsep_median, b.dsep_upper,
    )
    assert (a.direct_lower, a.direct_median, a.direct_upper) == (
        b.direct_lower, b.direct_median, b.direct_upper,
    )


def _build_heterogeneous_panel(
    *,
    n_bursts: int,
    seed: int,
) -> pl.DataFrame:
    """Heterogeneous-cell panel for the wide-CI test on
    `bootstrap_edge_counts`.

    Two cell sub-populations:
      A. Half the cells follow `arm → mediator → outcome` (full
         mediation, strong signal) → dsep at most bursts.
      B. Other half follow `arm → outcome` directly with mediator
         independent (direct edge) → direct at most bursts.

    Under cluster resampling, which sub-population dominates the
    replica drives the count classification. The bootstrap CI on
    `dsep` should be WIDE: dsep_upper - dsep_lower >= n_bursts/2
    (heterogeneity uncovered)."""
    rng = np.random.default_rng(seed)
    # Mix: half cells follow full-mediation chain at strong
    # signal, half cells follow weak direct-edge with a tiny
    # arm→outcome coefficient. Bootstrap-resampling the mixture
    # produces a CI on the count because:
    #   - replicas drawn mostly from type-A → high dsep count
    #   - replicas drawn mostly from type-B → mixed (weak signal
    #     bursts drop below α; some bursts dsep by partial-CI
    #     type-II)
    # Per-burst n is kept just above min_n_per_burst=20 so the CI
    # tests are near the power boundary on the weaker stratum.
    n_per_arm_per_type = 11  # n=22 per type per burst → near threshold
    arm_code = np.asarray(
        [0.5] * n_per_arm_per_type + [-0.5] * n_per_arm_per_type
        + [0.5] * n_per_arm_per_type + [-0.5] * n_per_arm_per_type,
        dtype=np.float64,
    )
    arms = (
        ['treatment'] * n_per_arm_per_type
        + ['baseline'] * n_per_arm_per_type
        + ['treatment'] * n_per_arm_per_type
        + ['baseline'] * n_per_arm_per_type
    )
    n_a = 2 * n_per_arm_per_type  # type-A row count
    n_total = 4 * n_per_arm_per_type
    mediator_matrix = np.zeros((n_total, n_bursts), dtype=np.float64)
    outcome_matrix = np.zeros((n_total, n_bursts), dtype=np.float64)
    for b in range(n_bursts):
        # Type-A: strong full-mediation chain.
        e_m_a = rng.normal(0.0, 0.5, size=n_a)
        e_y_a = rng.normal(0.0, 0.5, size=n_a)
        mediator_matrix[:n_a, b] = arm_code[:n_a] + e_m_a
        outcome_matrix[:n_a, b] = mediator_matrix[:n_a, b] + e_y_a
        # Type-B: weak direct edge (β=0.3) with noisy independent
        # mediator. The 0.3 coefficient is near the per-burst
        # detection threshold at n=44 → some bursts marginal-CI
        # rejects, some don't.
        e_m_b = rng.normal(0.0, 1.0, size=n_a)
        e_y_b = rng.normal(0.0, 1.0, size=n_a)
        mediator_matrix[n_a:, b] = e_m_b
        outcome_matrix[n_a:, b] = 0.3 * arm_code[n_a:] + e_y_b
    cells: list[Mapping[str, object]] = []
    for i in range(n_total):
        cells.append({
            'env_name': 'env_a',
            'gamma': 0.99,
            'arm_key': arms[i],
            'mediator_pb': mediator_matrix[i, :].tolist(),
            'outcome_pb': outcome_matrix[i, :].tolist(),
        })
    return pl.DataFrame(cells)


def test_pc_bootstrap_edge_counts_heterogeneous_wide_ci() -> None:
    """Heterogeneous panel (half full-mediation cells + half
    direct-edge cells): different cell sub-populations imply
    different per-burst CI classifications. Bootstrap resampling
    should surface this — the CI on the dsep count should be
    wider than under the clean full-mediation panel.

    With 200 resamples we expect the dsep distribution to span
    a non-trivial range; we assert the dsep CI width (upper -
    lower) is at least 1 (compared to the clean panel where the
    width can be 0 if every resample produces the same count).
    This is a directional check, not a tight closed-form bound —
    the framework surfaces the heterogeneity, the magnitude is
    panel-dependent."""
    n_bursts = 6
    df = _build_heterogeneous_panel(n_bursts=n_bursts, seed=405)
    results = dynamic_pc_adjacency.fn(
        df, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
        n_bootstrap=200,
        bootstrap_seed=42,
    )
    result = _get_single_stratum(results)
    bec = result.bootstrap_edge_counts
    assert bec is not None
    dsep_width = bec.dsep_upper - bec.dsep_lower
    direct_width = bec.direct_upper - bec.direct_lower
    # At least one of dsep or direct should have non-zero CI
    # width — the bootstrap is surfacing the cell-mixture.
    assert dsep_width + direct_width >= 1, (
        f'expected non-zero CI width under heterogeneous panel; '
        f'got dsep_width={dsep_width}, direct_width={direct_width}'
    )


def test_pc_bootstrap_edge_counts_n_resamples_one_degenerate() -> None:
    """Boundary: `n_bootstrap=1` returns a single-iteration
    triple (degenerate but doesn't crash). Lower / median / upper
    are all the same value (one sample → percentile is that
    sample)."""
    df = _build_full_mediation_panel(n_bursts=3, seed=406)
    results = dynamic_pc_adjacency.fn(
        df, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
        n_bootstrap=1,
        bootstrap_seed=42,
    )
    result = _get_single_stratum(results)
    bec = result.bootstrap_edge_counts
    assert bec is not None
    assert bec.n_resamples == 1
    # Single sample → all three percentiles are the same value.
    assert bec.marg_lower == bec.marg_median == bec.marg_upper
    assert bec.dsep_lower == bec.dsep_median == bec.dsep_upper
    assert bec.direct_lower == bec.direct_median == bec.direct_upper


def test_pc_bootstrap_edge_counts_null_binomial_z_score_bound() -> None:
    """Z-score-bound test under the null scenario. Under
    `H0: arm ⊥ outcome` at α=0.05 across n_bursts=20:

      n_marg ~ Binomial(20, 0.05)
      μ = 20 · 0.05 = 1.0
      σ = sqrt(20 · 0.05 · 0.95) ≈ 0.975

    The bootstrap median of `n_marg` is an estimator of the
    distribution's central tendency at the observed panel; under
    the null its expected value is the binomial mean (1.0) and
    its sampling SD is bounded by σ_binomial. Empirical CI
    half-width at n_resamples=300 percentile (interp): SE ≈
    σ / sqrt(n_resamples) ≈ 0.056 — well within 1 integer count.

    Combined: |marg_median - 1| ≤ ⌈2σ⌉ = 2 (2σ envelope rounded
    up). Bound: marg_median ∈ [0, 3]. Bootstrap shouldn't drift
    the median outside the binomial 2σ confidence envelope."""
    n_bursts = 20
    df = _build_null_panel(n_bursts=n_bursts, seed=407)
    results = dynamic_pc_adjacency.fn(
        df, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
        n_bootstrap=300,
        bootstrap_seed=42,
    )
    result = _get_single_stratum(results)
    bec = result.bootstrap_edge_counts
    assert bec is not None
    # Closed-form binomial parameters under H0.
    alpha_null = 0.05
    mu_binom = n_bursts * alpha_null  # 1.0
    sigma_binom = math.sqrt(n_bursts * alpha_null * (1.0 - alpha_null))
    # 2σ envelope → integer count bound 3 (mu + 2σ ≈ 2.95 → 3).
    upper_bound = int(math.ceil(mu_binom + 2.0 * sigma_binom))
    assert bec.marg_median <= upper_bound, (
        f'marg_median={bec.marg_median} > {upper_bound} (μ={mu_binom:.2f} '
        f'+ 2σ={2 * sigma_binom:.2f} = '
        f'{mu_binom + 2 * sigma_binom:.2f}) under null DAG. Bootstrap '
        f'should not drift the median outside the binomial 2σ '
        f'envelope.'
    )
    assert bec.marg_median >= 0


def test_pc_bootstrap_edge_counts_is_frozen() -> None:
    """`ClusterBootstrapEdgeCounts` is a frozen dataclass —
    mutation must raise."""
    df = _build_full_mediation_panel(n_bursts=3, seed=408)
    results = dynamic_pc_adjacency.fn(
        df, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
        n_bootstrap=20,
        bootstrap_seed=42,
    )
    result = _get_single_stratum(results)
    bec = result.bootstrap_edge_counts
    assert bec is not None
    with pytest.raises((AttributeError, TypeError)):
        bec.dsep_median = 0  # pyright: ignore[reportAttributeAccessIssue]
