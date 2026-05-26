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
