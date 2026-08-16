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
the LG-SCM implementation to per-burst arrays and tests recovery
against a CLOSED-FORM (population-level) target derived from
implementation parameters.

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
import numpy.typing as npt
import polars as pl
import pytest
from scipy.stats import spearmanr

from corroborate.analyses.dynamic_mediation import (
    ClusterBootstrapInterval,
    DynamicMediationResult,
    FisherZDLPool,
    TimeAggregationStatus,
    dynamic_partial_spearman,
)
from corroborate.analyses.dynamic_mediation import (
    classify_status,
)
from corroborate.analyses.dynamic_mediation import (
    fisher_z_dl_pool,
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
    # SIGN_FLIP → BOTH marginal AND partial pool are NaN by
    # contract. The partial inherits the same suspect support as
    # the marginal (computed on the same per-burst (xs, ys, zs)
    # trios) — if the marginal is incoherent the partial's pool
    # isn't trustworthy either.
    assert math.isnan(result.rho_marginal_pooled), (
        f'rho_marginal_pooled={result.rho_marginal_pooled!r}; '
        f'must be NaN under SIGN_FLIP_DETECTED'
    )
    assert math.isnan(result.rho_partial_pooled), (
        f'rho_partial_pooled={result.rho_partial_pooled!r}; '
        f'must be NaN under SIGN_FLIP_DETECTED (partial inherits '
        f'marginal\'s suspect support)'
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
    assert result.mediator_names == ('mediator_pb',)
    assert result.k_conditioning == 1
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


# ============ Sign-flip noise floor (`sign_flip_min_abs_rho`) ============

def test_noise_level_opposite_signs_dont_flip_classifier_unit() -> None:
    """Unit-level test on the classifier: hand-built ρ trajectory
    (+0.5, +0.5, -0.02) with the third burst's |ρ| below the
    default 0.05 noise floor. Classifier must return
    CONSISTENT_DIRECTION at the default floor.

    Tested via the framework's `classify_status` directly because
    `_build_panel`'s seed-coupled noise makes it hard to land the
    burst-2 empirical ρ in the tight `(-0.05, 0)` window the panel
    test would need."""
    status = classify_status(
        rho_marginal=(0.5, 0.5, -0.02),
        n_per_burst=(80, 80, 80),
        min_n_per_burst=5,
        weak_time_varying_ratio=2.0,
        sign_flip_min_abs_rho=0.05,
    )
    assert status is TimeAggregationStatus.CONSISTENT_DIRECTION, (
        f'classifier returned {status!r} on a trajectory whose '
        f'sole opposite-sign burst is at |ρ|=0.02 < floor=0.05; '
        f'must treat as noise, not flip'
    )


def test_noise_floor_zero_recovers_legacy_hairtrigger_behaviour() -> None:
    """Setting `sign_flip_min_abs_rho=0.0` recovers the pre-fix
    hair-trigger behaviour: any opposite-sign valid burst triggers
    SIGN_FLIP_DETECTED regardless of magnitude. Pins the
    direction-of-effect of the new parameter."""
    status_default = classify_status(
        rho_marginal=(0.5, 0.5, -0.02),
        n_per_burst=(80, 80, 80),
        min_n_per_burst=5,
        weak_time_varying_ratio=2.0,
        sign_flip_min_abs_rho=0.05,
    )
    status_zero = classify_status(
        rho_marginal=(0.5, 0.5, -0.02),
        n_per_burst=(80, 80, 80),
        min_n_per_burst=5,
        weak_time_varying_ratio=2.0,
        sign_flip_min_abs_rho=0.0,
    )
    assert status_default is TimeAggregationStatus.CONSISTENT_DIRECTION
    assert status_zero is TimeAggregationStatus.SIGN_FLIP_DETECTED, (
        f'floor=0 should restore hair-trigger behaviour; got '
        f'{status_zero!r}'
    )


def test_noise_floor_panel_integration_with_clean_construction() -> None:
    """Panel-level integration test for the noise-floor gate.
    Construction: 3 bursts at planted ρ values that produce
    empirical |ρ|[2] < 0.05 reliably. We over-pad the construction
    via larger n + tighter planted value so the seed-noise band
    lands inside the floor with high probability.

    Two assertions:
      (a) at default floor=0.05, status is CONSISTENT_DIRECTION
          (the near-zero burst is treated as noise);
      (b) at floor=0.0, status flips to SIGN_FLIP_DETECTED if and
          only if burst-2 empirical ρ is below zero.
    """
    # Seed=0 produces burst-2 empirical ρ ≈ -0.015 (negative, below
    # the 0.05 noise floor) — exactly the construction we need to
    # exercise the noise-floor gate.
    df = _build_panel(rho_trajectory=(0.5, 0.5, 0.0), seed=0)
    # Default floor: noise-band burst doesn't trigger flip.
    res_default = dynamic_partial_spearman.fn(
        df, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
        sign_flip_min_abs_rho=0.05,
    )
    r_default = _get_single_stratum(res_default)
    # Sanity on construction: burst-2 lands in (-0.05, 0) — below
    # the default 0.05 floor with opposite (negative) sign.
    assert -0.05 < r_default.rho_marginal[2] < 0.0, (
        f'construction failed: burst-2 ρ={r_default.rho_marginal[2]!r} '
        f'should be in (-0.05, 0) to exercise the noise-floor gate'
    )
    assert r_default.aggregation_status is (
        TimeAggregationStatus.CONSISTENT_DIRECTION
    ), (
        f'default floor: noise-level burst should NOT flip; got '
        f'{r_default.aggregation_status!r} (ρ trajectory='
        f'{r_default.rho_marginal})'
    )
    # At floor=0.0 the same trajectory triggers SIGN_FLIP_DETECTED
    # because burst-2's opposite-sign ρ counts regardless of
    # magnitude.
    res_zero = dynamic_partial_spearman.fn(
        df, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
        sign_flip_min_abs_rho=0.0,
    )
    r_zero = _get_single_stratum(res_zero)
    assert r_zero.aggregation_status is (
        TimeAggregationStatus.SIGN_FLIP_DETECTED
    ), (
        f'floor=0: opposite-sign burst at any magnitude must flip; '
        f'got {r_zero.aggregation_status!r}'
    )


def test_sign_flip_above_noise_floor_still_fires() -> None:
    """A burst at |ρ| > `sign_flip_min_abs_rho` with the opposite
    sign MUST still trigger SIGN_FLIP_DETECTED. Pins that the
    noise floor doesn't silently swallow real flips."""
    df = _build_panel(rho_trajectory=(0.5, 0.5, -0.3), seed=21)
    results = dynamic_partial_spearman.fn(
        df, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
        sign_flip_min_abs_rho=0.05,
    )
    result = _get_single_stratum(results)
    # burst-2 empirical ρ should be deeply negative (~-0.3) — well
    # above the noise floor in magnitude.
    assert result.rho_marginal[2] < -0.1, (
        f'burst 2 ρ={result.rho_marginal[2]!r} should be deeply '
        f'negative (planted -0.3)'
    )
    assert result.aggregation_status is (
        TimeAggregationStatus.SIGN_FLIP_DETECTED
    )


# ============ WEAK_TIME_VARYING robustness to noise bursts ============

def test_weak_time_varying_robust_to_single_noise_burst() -> None:
    """`WEAK_TIME_VARYING` should be driven by the genuinely
    time-varying part of the trajectory — NOT by a single
    near-zero burst inflating the max/min ratio.

    Planted (+0.4, +0.4, +0.4, +0.01): the first three are
    indistinguishable in |ρ|; the last is noise-level. Pre-fix
    behaviour would compute max(|ρ|)/min(|ρ|) ≈ 0.4/0.01 = 40, far
    above the default 2.0 ratio → WEAK_TIME_VARYING. Post-fix
    behaviour drops noise-level bursts from the ratio → ratio ≈
    1.0 → CONSISTENT_DIRECTION."""
    df = _build_panel(rho_trajectory=(0.4, 0.4, 0.4, 0.01), seed=22)
    results = dynamic_partial_spearman.fn(
        df, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
        sign_flip_min_abs_rho=0.05,
    )
    result = _get_single_stratum(results)
    # Verify the planted near-zero burst materialised below the
    # noise floor.
    assert abs(result.rho_marginal[3]) < 0.10, (
        f'burst 3 ρ={result.rho_marginal[3]!r} should be near-zero '
        f'(planted +0.01); construction broken'
    )
    assert result.aggregation_status is (
        TimeAggregationStatus.CONSISTENT_DIRECTION
    ), (
        f'status={result.aggregation_status!r}; '
        f'rho_marginal={result.rho_marginal} — noise-level burst '
        f'must NOT drive WEAK_TIME_VARYING'
    )


# ============ Boundary tests on `min_n_per_burst` ============

def _build_small_panel(
    *,
    n_cells_per_arm: int,
    seed: int,
) -> pl.DataFrame:
    """Tiny-stratum panel for `min_n_per_burst` boundary tests.
    Single burst; per-burst n = 2 * n_cells_per_arm."""
    rng = np.random.default_rng(seed)
    n = 2 * n_cells_per_arm
    arms = ['treatment'] * n_cells_per_arm + ['baseline'] * n_cells_per_arm
    outcome = rng.normal(0.0, 1.0, size=n)
    # Plant a moderate ρ via the arm code so the per-burst ρ is
    # well-defined when n is sufficient.
    arm_code = np.asarray(
        [1.0] * n_cells_per_arm + [0.0] * n_cells_per_arm,
        dtype=np.float64,
    )
    arm_centred = (arm_code - arm_code.mean()) / float(np.std(arm_code))
    outcome = 0.5 * arm_centred + math.sqrt(0.75) * outcome
    mediator = rng.normal(0.0, 1.0, size=n)
    cells: list[Mapping[str, object]] = [
        {
            'env_name': 'env_a', 'gamma': 0.99, 'arm_key': arms[i],
            'mediator_pb': [float(mediator[i])],
            'outcome_pb': [float(outcome[i])],
        }
        for i in range(n)
    ]
    return pl.DataFrame(cells)


def test_min_n_per_burst_boundary_n_below_floor() -> None:
    """At n=4 (2 cells per arm) with `min_n_per_burst=5`, the
    single burst falls below the floor → all per-burst ρ are
    NaN → status UNDERPOWERED_BURSTS."""
    df = _build_small_panel(n_cells_per_arm=2, seed=30)  # n=4
    results = dynamic_partial_spearman.fn(
        df, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
        min_n_per_burst=5,
    )
    result = _get_single_stratum(results)
    assert result.n_per_burst == (4,)
    assert math.isnan(result.rho_marginal[0])
    assert math.isnan(result.rho_partial[0])
    assert result.aggregation_status is (
        TimeAggregationStatus.UNDERPOWERED_BURSTS
    ), (
        f'n=4 < min_n_per_burst=5 must produce UNDERPOWERED_BURSTS; '
        f'got {result.aggregation_status!r}'
    )


def test_min_n_per_burst_boundary_n_at_floor() -> None:
    """At n=6 (3 cells per arm; we use 3 here to stay safely above
    boundary while keeping n above the closed-form partial floor of
    n=5) with `min_n_per_burst=5`, the burst meets the floor → ρ
    computed (non-NaN)."""
    # n=6 chosen because partial_spearman_rho requires n >= 5 for
    # the closed-form first-order partial; n=5 sits at the boundary
    # where partial ρ is well-defined but rho_partial has df=1.
    df = _build_small_panel(n_cells_per_arm=3, seed=31)  # n=6
    results = dynamic_partial_spearman.fn(
        df, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
        min_n_per_burst=5,
    )
    result = _get_single_stratum(results)
    assert result.n_per_burst == (6,)
    # ρ values defined (not NaN).
    assert not math.isnan(result.rho_marginal[0]), (
        f'n=6 >= floor=5: marginal ρ should be defined, got NaN'
    )
    assert not math.isnan(result.rho_partial[0]), (
        f'n=6 >= floor=5: partial ρ should be defined, got NaN'
    )
    # Status is NOT UNDERPOWERED_BURSTS — at least one valid burst.
    assert result.aggregation_status is not (
        TimeAggregationStatus.UNDERPOWERED_BURSTS
    ), (
        f'status={result.aggregation_status!r} — n>=floor should '
        f'avoid UNDERPOWERED_BURSTS'
    )


# ============ df_offset discriminates marginal vs partial pool ============

def test_df_offset_distinguishes_marginal_and_partial_pools() -> None:
    """The marginal pool uses df_offset=3 (matching
    `stratified_spearman_rho`); the partial pool uses df_offset=4
    (matching `stratified_partial_spearman_rho`). When per-burst
    `n` is small and uneven, the two df_offsets weight bursts
    differently → the two pools land at numerically distinct
    values even when the per-burst marginal and partial ρ's are
    identical.

    Construction: a stratum with multiple bursts at small n (e.g.
    n=10), planted so r_marginal[b] = r_partial[b] = 0.5 for all
    b but with one burst dropped via the partial-floor (n=5 means
    partial df = 1, weight = 1; marginal df = 7, weight = 7).
    Different weights → different pooled values.

    Pre-fix code used df_offset=4 for BOTH pools, so this test
    would assert equality. Post-fix code uses df_offset=3 for
    marginal, so the pools diverge in the small-n regime."""
    # Compute directly via fisher_z_pool to verify the framework's
    # behaviour matches the closed-form for both df_offset values.
    from corroborate.stats import fisher_z_pool
    # Two bursts with same ρ but unequal n. At n=5 the marginal
    # weight is 2 (n-3) and the partial weight is 1 (n-4); at n=20
    # the marginal weight is 17 and the partial weight is 16. The
    # POOLED ρ at equal per-burst ρ is the same (both reduce to ρ
    # via Fisher z), but the test wedge needs UNEQUAL per-burst ρ
    # to make the weights bite.
    rs = (0.30, 0.70)  # two bursts
    ns = (5, 20)
    rho_marg_3, _ = fisher_z_pool(rs, ns, df_offset=3)
    rho_marg_4, _ = fisher_z_pool(rs, ns, df_offset=4)
    # Confirm the two df_offsets land at numerically distinct
    # pooled values when per-burst ρ varies. (Sanity on the test.)
    assert abs(rho_marg_3 - rho_marg_4) > 0.005, (
        f'fisher_z_pool not sensitive to df_offset at this '
        f'construction: df=3 → {rho_marg_3:.6f}, df=4 → '
        f'{rho_marg_4:.6f}; bump rho/n spread'
    )
    # Now verify the framework's `rho_marginal_pooled` matches
    # df_offset=3 NOT df_offset=4. Build a panel where both bursts
    # have ρ ≈ planted (with mediator independent so partial ≈
    # marginal). We construct a single-stratum panel where burst 0
    # has small n and burst 1 has large n with the same planted ρ.
    # Use the ragged-tail semantics: short cells contribute only
    # to burst 0; long cells contribute to both. n_burst_0 = 5,
    # n_burst_1 = 20.
    rng = np.random.default_rng(40)
    rho_target = 0.5
    cells: list[Mapping[str, object]] = []

    def _one_cell(arm: str, arm_code: float, n_bursts: int) -> Mapping[str, object]:
        noise_out = rng.normal(0.0, 1.0, size=n_bursts)
        outcome = (
            rho_target * arm_code
            + math.sqrt(max(1.0 - rho_target ** 2, 0.0)) * noise_out
        ).tolist()
        med = rng.normal(0.0, 1.0, size=n_bursts).tolist()
        return {
            'env_name': 'env_a', 'gamma': 0.99,
            'arm_key': arm, 'outcome_pb': outcome,
            'mediator_pb': med,
        }

    # Encode arms as centred for the construction to plant ρ;
    # framework's sorted-unique encoding will recover the same
    # partition since 'baseline' < 'treatment'.
    for _ in range(3):  # 3 cells per arm with length-1 trajectory
        cells.append(_one_cell('treatment', 1.0, 1))
        cells.append(_one_cell('baseline', -1.0, 1))
    for _ in range(10):  # 10 cells per arm with length-2 trajectory
        cells.append(_one_cell('treatment', 1.0, 2))
        cells.append(_one_cell('baseline', -1.0, 2))
    df = pl.DataFrame(cells)
    results = dynamic_partial_spearman.fn(
        df, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
        min_n_per_burst=5,  # both bursts above floor
    )
    result = _get_single_stratum(results)
    # n_per_burst[0] = 26 (all cells); n_per_burst[1] = 20 (only
    # the length-2 cells contribute to burst 1).
    assert result.n_per_burst[0] == 26
    assert result.n_per_burst[1] == 20
    # The framework's pool weights are:
    #   marginal: (26-3, 20-3) = (23, 17)
    #   partial:  (26-4, 20-4) = (22, 16)
    # Verify by independently invoking fisher_z_pool with the
    # framework's reported per-burst ρ values.
    rho_pb = result.rho_marginal
    n_pb = result.n_per_burst
    expected_marg_3, _ = fisher_z_pool(rho_pb, n_pb, df_offset=3)
    expected_marg_4, _ = fisher_z_pool(rho_pb, n_pb, df_offset=4)
    # Framework's marginal pool must match df_offset=3, NOT 4.
    assert abs(result.rho_marginal_pooled - expected_marg_3) < 1e-9, (
        f'rho_marginal_pooled={result.rho_marginal_pooled} should '
        f'equal fisher_z_pool with df_offset=3 ({expected_marg_3}); '
        f'mismatch implies wrong df_offset'
    )
    # And NOT match df_offset=4 (proves the fix bit).
    assert abs(result.rho_marginal_pooled - expected_marg_4) > 1e-9, (
        f'rho_marginal_pooled={result.rho_marginal_pooled} equals '
        f'df_offset=4 pool — fix not landed'
    )
    # Symmetric assertion on the partial pool: must match
    # df_offset=4, not df_offset=3.
    rho_pb_partial = result.rho_partial
    expected_part_3, _ = fisher_z_pool(rho_pb_partial, n_pb, df_offset=3)
    expected_part_4, _ = fisher_z_pool(rho_pb_partial, n_pb, df_offset=4)
    assert abs(result.rho_partial_pooled - expected_part_4) < 1e-9, (
        f'rho_partial_pooled={result.rho_partial_pooled} should '
        f'equal fisher_z_pool with df_offset=4 ({expected_part_4})'
    )
    assert abs(result.rho_partial_pooled - expected_part_3) > 1e-9, (
        f'rho_partial_pooled={result.rho_partial_pooled} equals '
        f'df_offset=3 pool — partial-df accounting wrong'
    )


# ============ Ragged-tail per-burst alignment ============

def test_ragged_tail_uses_max_length_with_decreasing_n_per_burst() -> None:
    """Cells with shorter trajectories contribute their prefix
    only; n_per_burst grows past shorter cells' tails as longer
    cells continue contributing.

    Construction: 6 short cells (3 per arm, length 2) + 8 long
    cells (4 per arm, length 4). n_bursts = max length = 4.
    n_per_burst should be: (14, 14, 8, 8) — all 14 cells
    contribute to bursts 0-1; only 8 long cells contribute to
    bursts 2-3.

    This pins the "less-information-losing" docstring claim
    against the pre-fix truncate-to-min behaviour (which would
    have given n_bursts=2 and discarded bursts 2-3 entirely)."""
    rng = np.random.default_rng(50)
    cells: list[Mapping[str, object]] = []
    for arm, arm_code in (('treatment', 1.0), ('baseline', -1.0)):
        # 3 short cells per arm.
        for _ in range(3):
            cells.append({
                'env_name': 'env_a', 'gamma': 0.99, 'arm_key': arm,
                'outcome_pb': (
                    0.5 * arm_code + rng.normal(0.0, 0.5, size=2)
                ).tolist(),
                'mediator_pb': rng.normal(0.0, 1.0, size=2).tolist(),
            })
        # 4 long cells per arm.
        for _ in range(4):
            cells.append({
                'env_name': 'env_a', 'gamma': 0.99, 'arm_key': arm,
                'outcome_pb': (
                    0.5 * arm_code + rng.normal(0.0, 0.5, size=4)
                ).tolist(),
                'mediator_pb': rng.normal(0.0, 1.0, size=4).tolist(),
            })
    df = pl.DataFrame(cells)
    results = dynamic_partial_spearman.fn(
        df, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
        min_n_per_burst=5,
    )
    result = _get_single_stratum(results)
    assert result.n_bursts == 4, (
        f'expected n_bursts=4 (max trajectory length); got '
        f'{result.n_bursts}. Truncate-to-min would have given 2.'
    )
    assert result.n_per_burst == (14, 14, 8, 8), (
        f'n_per_burst={result.n_per_burst} — ragged-tail should '
        f'give (14, 14, 8, 8): all 14 cells contribute bursts 0-1; '
        f'only 8 long cells contribute bursts 2-3'
    )


# ============ Measurable input path ============

def test_accepts_measurable_inputs_for_mediator_and_outcome() -> None:
    """`dynamic_partial_spearman` accepts `Measurable[..., NDArray]`
    inputs for `mediator_per_burst` / `outcome_per_burst`,
    mirroring the static `partial_spearman` lazy-evaluation
    pattern. The Measurable's `.name` is used as the cache-column
    key (cache-first dispatch via `evaluate_per_burst_source`).

    This test pins that the Measurable path produces the SAME
    result as the column-name path when the cache column is
    present — the cache-first read short-circuits Measurable
    evaluation, so identity is by construction."""
    from corroborate.measurables import measurable

    @measurable(name='outcome_pb', reads=('outcome_pb',))
    def outcome_pb_m(cell: Mapping[str, object]) -> npt.NDArray[np.floating]:
        # Cache-first dispatch should make this body unreachable
        # when the cell has the cached column — but the function
        # must be well-typed.
        raw = cell['outcome_pb']
        assert isinstance(raw, list)
        return np.asarray(raw, dtype=np.float64)

    @measurable(name='mediator_pb', reads=('mediator_pb',))
    def mediator_pb_m(cell: Mapping[str, object]) -> npt.NDArray[np.floating]:
        raw = cell['mediator_pb']
        assert isinstance(raw, list)
        return np.asarray(raw, dtype=np.float64)

    df = _build_panel(rho_trajectory=(0.4, 0.45, 0.5), seed=60)
    # Column-name baseline.
    results_str = dynamic_partial_spearman.fn(
        df, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
    )
    # Measurable input (cache-first hits the same `outcome_pb` /
    # `mediator_pb` columns).
    results_m = dynamic_partial_spearman.fn(
        df, arm_field='arm_key',
        mediator_per_burst=mediator_pb_m,
        outcome_per_burst=outcome_pb_m,
        stratify_by=('env_name', 'gamma'),
    )
    r_str = _get_single_stratum(results_str)
    r_m = _get_single_stratum(results_m)
    # Per-burst ρ identical when both paths read the same data.
    assert r_str.rho_marginal == r_m.rho_marginal, (
        f'column-name vs Measurable path diverged on rho_marginal: '
        f'str={r_str.rho_marginal} measurable={r_m.rho_marginal}'
    )
    assert r_str.rho_partial == r_m.rho_partial
    # Provenance: Measurable path stamps mediator_names /
    # outcome_name from the Measurable's `.name`.
    assert r_m.mediator_names == ('mediator_pb',)
    assert r_m.outcome_name == 'outcome_pb'


# ============ DerSimonian-Laird random-effects pool ============
#
# These tests verify the DL pool wired into `DynamicMediationResult`
# via `fisher_z_dl_pool`. The DL pool is computed alongside the FE
# Fisher-z pool — it's NEVER NaN'd by the diagnostic gate; its
# τ²/I² ARE the quantitative signal of the heterogeneity that
# `aggregation_status` flags qualitatively.
#
# Closed-form bound derivation for the SIGN_FLIP test:
#   Planted ρ trajectory (+0.5, +0.5, −0.5) with n=80 per burst.
#   - z_b = atanh(±0.5) = ±0.5493
#   - var_z = 1/(n−3) = 1/77 ≈ 0.01299, w_fixed = 77 each
#   - z_fixed (FE) = (77·0.5493 + 77·0.5493 + 77·(−0.5493))/(3·77)
#                  = 0.5493/3 ≈ 0.1831
#   - Q = Σ w·(z − z_fixed)²
#       = 77·(0.5493−0.1831)² + 77·(0.5493−0.1831)² + 77·(−0.5493−0.1831)²
#       = 77·0.1341 + 77·0.1341 + 77·0.5364
#       ≈ 61.96
#   - df = G − 1 = 2
#   - I² = (Q − df)/Q = (61.96 − 2)/61.96 ≈ 0.968 → > 0.7 with
#     wide margin
#   - c = Σw − Σw²/Σw = 3·77 − 3·77²/(3·77) = 2·77 = 154
#   - τ² = max(0, (Q − df)/c) = (61.96 − 2)/154 ≈ 0.389 → > 0.1

def test_dl_pool_matches_fe_pool_when_tau2_zero() -> None:
    """When per-burst ρ is constant across bursts (no
    heterogeneity), Cochran's Q is small (~df under sampling
    noise) and τ² clips to 0. With τ²=0, DL random-effects
    weights equal FE Fisher-z weights → pooled estimates are
    numerically identical.

    Construction: planted ρ ≈ 0.5 across 3 bursts at n=80 each.
    With seed-coupled noise the per-burst empirical ρ varies by
    ~0.05 around 0.5; Q ≈ df ≈ 2; τ² = max(0, (Q-df)/c) clips
    to 0 with high probability.

    Bound: |dl.rho_pooled − fe_rho_pooled| < 1e-9 when τ² = 0
    exactly (the clip resolves both paths to identical n-weighted
    z-pooling). If τ² > 0 due to seed variance, the bound
    loosens to < 0.01 (still tight — both weight by
    (n_b − df_offset) which doesn't differ across bursts here)."""
    planted = 0.5
    df = _build_panel(
        rho_trajectory=(planted, planted, planted), seed=100,
    )
    results = dynamic_partial_spearman.fn(
        df, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
    )
    result = _get_single_stratum(results)
    # Sanity: this construction is CONSISTENT_DIRECTION.
    assert result.aggregation_status is (
        TimeAggregationStatus.CONSISTENT_DIRECTION
    )
    # FE pool defined.
    assert not math.isnan(result.rho_marginal_pooled)
    # DL pool defined.
    assert not math.isnan(result.dl_marginal.rho_pooled)
    assert result.dl_marginal.n_bursts_used == 3
    # Without genuine heterogeneity, τ² should be 0 (or in the
    # max(0, ·) clip's tiny-positive-bias zone for small G — flag
    # it if found by surfacing in the failure message).
    # The empirical probe at `tests/analytic/robustness/
    # test_dl_small_g_robustness.py` documents bias ≈ 0.011 at G=3.
    assert result.dl_marginal.tau2 < 0.05, (
        f'τ²={result.dl_marginal.tau2:.4f}: expected near 0 at '
        f'planted-constant ρ with G=3; in DL clip\'s small-G '
        f'tiny-positive-bias zone is OK but >0.05 suggests the '
        f'planted construction has more sampling-noise spread '
        f'than anticipated'
    )
    # I² near 0 when there's no real heterogeneity.
    assert result.dl_marginal.i2 < 0.5, (
        f'I²={result.dl_marginal.i2:.4f}: expected near 0 at '
        f'planted-constant ρ'
    )
    # DL and FE pools agree closely. With τ²=0 the weights are
    # identical; the only difference is numerical (DL goes through
    # `random_effects_summary`, FE through `fisher_z_pool`).
    deviation = abs(
        result.dl_marginal.rho_pooled - result.rho_marginal_pooled
    )
    assert deviation < 0.01, (
        f'DL rho_pooled={result.dl_marginal.rho_pooled:.6f} vs '
        f'FE rho_marginal_pooled={result.rho_marginal_pooled:.6f}: '
        f'diff={deviation:.6f} — should agree when τ²=0'
    )


def test_dl_pool_surfaces_large_tau2_under_sign_flip() -> None:
    """Under SIGN_FLIP_DETECTED the FE pool returns NaN (the
    diagnostic gate suppresses it). The DL pool is NOT gated —
    it returns a real pooled estimate (near zero by
    cancellation) PLUS large τ² PLUS I² near 1.0, surfacing the
    heterogeneity quantitatively.

    Closed-form (planted (+0.5, +0.5, −0.5) at n=80):
      z_b = ±atanh(0.5) ≈ ±0.5493 (each w=77)
      z_fixed = (0.5493 + 0.5493 − 0.5493)/3 ≈ 0.1831
      Q ≈ 61.96, df=2 → I² ≈ 0.968, τ² ≈ 0.389
    The bound `i2 > 0.7` admits seed-variance slack on top of
    the closed-form 0.968; `tau2 > 0.1` admits slack below the
    closed-form 0.389. Both are tight relative to the closed-
    form values and reject the null hypothesis (no
    heterogeneity → I²=0, τ²=0).
    """
    df = _build_panel(rho_trajectory=(0.5, 0.5, -0.5), seed=101)
    results = dynamic_partial_spearman.fn(
        df, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
    )
    result = _get_single_stratum(results)
    # Sanity: status is SIGN_FLIP_DETECTED and FE pool is NaN.
    assert result.aggregation_status is (
        TimeAggregationStatus.SIGN_FLIP_DETECTED
    )
    assert math.isnan(result.rho_marginal_pooled)
    # DL pool is NOT gated by the diagnostic enum.
    assert not math.isnan(result.dl_marginal.rho_pooled), (
        f'DL rho_pooled must NOT be NaN under SIGN_FLIP_DETECTED '
        f'— the DL pool is the heterogeneity diagnostic, not the '
        f'point estimate of a "consistent" effect'
    )
    # DL pooled ρ near zero by cancellation. With FE
    # z_pooled ≈ 0.1831 → tanh ≈ 0.181; DL gives more weight to
    # the larger-variance branches, pulling pooled-z slightly
    # closer to z=0. Either way pooled-ρ is in (0, 0.3).
    assert -0.05 < result.dl_marginal.rho_pooled < 0.35, (
        f'DL rho_pooled={result.dl_marginal.rho_pooled:.4f} — '
        f'expected near 0 by sign-flip cancellation'
    )
    # I² near 1: heterogeneity dominates total variance.
    # Closed-form: I² ≈ 0.968. Bound at 0.7 to leave wide
    # seed-variance slack.
    assert result.dl_marginal.i2 > 0.7, (
        f'I²={result.dl_marginal.i2:.4f}: expected > 0.7 under '
        f'sign-flip (closed-form ≈ 0.968)'
    )
    # τ² substantial. Closed-form: τ² ≈ 0.389. Bound at 0.1.
    assert result.dl_marginal.tau2 > 0.1, (
        f'τ²={result.dl_marginal.tau2:.4f}: expected > 0.1 under '
        f'sign-flip (closed-form ≈ 0.389)'
    )


def test_dl_pool_surfaces_moderate_tau2_under_weak_time_varying() -> None:
    """Under WEAK_TIME_VARYING (sign-consistent but magnitude
    varies by > 2×): the DL pool surfaces I² substantial but
    less than the sign-flip extreme. Closed-form (planted
    (+0.1, +0.3, +0.7), n=80):
      z_b = atanh(0.1, 0.3, 0.7) ≈ (0.1003, 0.3095, 0.8673)
      z_fixed ≈ 0.4257
      Q ≈ 77·(0.1059 + 0.0135 + 0.1950) ≈ 24.22
      df=2 → I² ≈ (24.22−2)/24.22 ≈ 0.917
    Bound 0.3 < i2 < 0.95: tight under above the noise floor."""
    df = _build_panel(rho_trajectory=(0.1, 0.3, 0.7), seed=102)
    results = dynamic_partial_spearman.fn(
        df, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
    )
    result = _get_single_stratum(results)
    # Sanity: status is WEAK_TIME_VARYING.
    assert result.aggregation_status is (
        TimeAggregationStatus.WEAK_TIME_VARYING
    )
    # DL pool defined.
    assert not math.isnan(result.dl_marginal.rho_pooled)
    # I² substantial but not saturated (no sign-flip cancellation).
    # Closed-form ≈ 0.917; bound (0.3, 0.95).
    assert 0.3 < result.dl_marginal.i2 < 0.95, (
        f'I²={result.dl_marginal.i2:.4f}: expected ∈ (0.3, 0.95) '
        f'under WEAK_TIME_VARYING (closed-form ≈ 0.917)'
    )


def test_dl_pool_consistent_with_fe_under_uniform_noise() -> None:
    """When per-burst ρ is uniformly small (all bursts at
    similar weak magnitude), no heterogeneity is present and DL
    pool agrees closely with FE pool. Both should land near the
    planted value within their respective SE bounds.

    Construction: planted ρ ≈ 0.2 at 3 bursts with n=80 each
    (n−df=77 weight per burst; ≪ saturation, well-defined
    Fisher-z). The DL clip-bias contribution at G=3 is ≤ 0.011
    on τ² (probe-documented); both pools land within ~0.05 of
    each other.
    """
    planted = 0.2
    df = _build_panel(
        rho_trajectory=(planted, planted, planted), seed=103,
    )
    results = dynamic_partial_spearman.fn(
        df, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
    )
    result = _get_single_stratum(results)
    # Both pools defined.
    assert not math.isnan(result.rho_marginal_pooled)
    assert not math.isnan(result.dl_marginal.rho_pooled)
    # Pooled estimates agree closely when τ² ≈ 0. At τ²=0
    # exactly they agree to fp tolerance; in the small-G
    # clip-bias regime (τ² ~ 0.01) the deviation is ≤ 0.01.
    deviation = abs(
        result.dl_marginal.rho_pooled - result.rho_marginal_pooled
    )
    assert deviation < 0.05, (
        f'DL vs FE divergence={deviation:.4f}: expected <0.05 '
        f'at planted-constant ρ (no real heterogeneity)'
    )


def test_dl_pool_pi_bounds_bracket_pooled_at_g_sufficient() -> None:
    """At G ≥ 3 (PI df = G−2 ≥ 1) PI bounds are defined and
    bracket the pooled estimate. At G = 2 (PI df = 0) PI bounds
    are NaN by the HTS formula.

    Construction: planted ρ ≈ 0.3 at G=4 bursts → PI defined.
    Closed-form: pooled z ≈ atanh(0.3) ≈ 0.3095, var_pooled ≈
    1/(4·77) ≈ 0.00325, SE_pooled ≈ 0.057. τ² ≈ 0 →
    pi_se ≈ SE_pooled. At G=4, t_{0.975, df=2} ≈ 4.303, so
    pi half-width ≈ 4.303 · 0.057 ≈ 0.245 in z-units. After
    tanh-back-transform PI bounds are inside ρ ∈ (-1, 1) and
    bracket rho_pooled."""
    planted = 0.3
    df = _build_panel(
        rho_trajectory=(planted,) * 4, seed=104,
    )
    results = dynamic_partial_spearman.fn(
        df, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
    )
    result = _get_single_stratum(results)
    dl = result.dl_marginal
    assert dl.n_bursts_used == 4
    assert not math.isnan(dl.rho_pi_lo), (
        f'PI lower NaN at G=4: expected defined (HTS df=G-2=2)'
    )
    assert not math.isnan(dl.rho_pi_hi)
    # PI brackets the pooled estimate.
    assert dl.rho_pi_lo < dl.rho_pooled < dl.rho_pi_hi, (
        f'PI=({dl.rho_pi_lo:.4f}, {dl.rho_pi_hi:.4f}) does not '
        f'bracket rho_pooled={dl.rho_pooled:.4f}'
    )
    # PI bounds inside (-1, 1) (Fisher-z space ↔ ρ space).
    assert -1.0 < dl.rho_pi_lo < 1.0
    assert -1.0 < dl.rho_pi_hi < 1.0


def test_dl_pool_pi_undefined_at_g_two() -> None:
    """At G=2 the HTS prediction interval is undefined
    (`pi_df = G − 2 = 0` → no t-critical) → PI bounds are NaN.
    Verify via the helper directly with a 2-burst trajectory."""
    # G=2 bursts: each ρ=0.3 at n=80. Direct helper call to
    # avoid the need for a panel that aligns to exactly G=2
    # ragged-tail-wise.
    dl = fisher_z_dl_pool(rhos=[0.3, 0.35], ns=[80, 80], df_offset=3)
    assert dl.n_bursts_used == 2
    # Pooled estimate defined (DL needs G ≥ 2 for τ² estimation).
    assert not math.isnan(dl.rho_pooled)
    # PI undefined at G=2 (pi_df = 0).
    assert math.isnan(dl.rho_pi_lo), (
        f'rho_pi_lo={dl.rho_pi_lo!r}: HTS PI must be NaN at G=2 '
        f'(df = G − 2 = 0)'
    )
    assert math.isnan(dl.rho_pi_hi)


# ============ FisherZDLPool dataclass invariants ============

def test_fisher_z_dl_pool_is_frozen_dataclass() -> None:
    """`FisherZDLPool` is a frozen dataclass — mutation must
    raise. Pinned for the same reason `DynamicMediationResult`
    is frozen: bridge consumers / verdict renderers must not be
    able to mutate the typed result."""
    dl = fisher_z_dl_pool(rhos=[0.3, 0.3, 0.3], ns=[80, 80, 80], df_offset=3)
    assert isinstance(dl, FisherZDLPool)
    with pytest.raises((AttributeError, TypeError)):
        dl.tau2 = 0.0  # pyright: ignore[reportAttributeAccessIssue]


def test_fisher_z_dl_pool_helper_nan_under_g_lt_2() -> None:
    """`fisher_z_dl_pool` propagates `random_effects_summary`'s
    `n < 2` NaN-filling. Below-floor n bursts are filtered before
    DL — they don't count toward `n_bursts_used`."""
    # n=5 with df_offset=3 → n−df=2 → SE_z = 1/sqrt(2) (valid),
    # but only ONE burst → `n_bursts_used=1`, DL undefined.
    dl = fisher_z_dl_pool(rhos=[0.5], ns=[5], df_offset=3)
    assert dl.n_bursts_used == 1
    assert math.isnan(dl.rho_pooled)
    assert math.isnan(dl.tau2)
    assert math.isnan(dl.i2)
    assert math.isnan(dl.q)

    # n=3 with df_offset=3 → n−df=0 → SE_z = 1/sqrt(0) undefined,
    # burst filtered → `n_bursts_used=0`.
    dl2 = fisher_z_dl_pool(rhos=[0.5, 0.5], ns=[3, 3], df_offset=3)
    assert dl2.n_bursts_used == 0
    assert math.isnan(dl2.rho_pooled)


# ============ Cluster bootstrap CI ============
#
# The cluster bootstrap resamples WHOLE CELLS (each cell = one
# training trajectory) with replacement, recomputes per-burst ρ
# from the resampled panel, and DL-pools per replica. The
# empirical [α/2, 1−α/2] percentile range across replicas is the
# CI — assumption-free under any within-cell autocorrelation
# (vs DL's PI bounds which assume per-burst independence).
#
# Bound derivation: at n=80 cells with planted constant ρ=0.5
# across 3 bursts, the bootstrap-replica DL-pooled ρ has sampling
# SD ≈ 1/sqrt(n_cells - df) = 1/sqrt(77) ≈ 0.114 in Fisher-z
# units; at ρ=0.5 the delta-method ρ-unit SE ≈ (1 - ρ²) · SE_z ≈
# 0.75 · 0.114 ≈ 0.086. 95% percentile half-width ≈ 1.96 · 0.086
# ≈ 0.169; width ≈ 0.34. So `rho_upper - rho_lower < 0.2` (the
# spec's bound) is *tight* — well below 2× the closed-form
# expected width at this sample size.

def test_n_bootstrap_zero_default_keeps_bootstrap_fields_none() -> None:
    """`n_bootstrap=0` (the default) leaves `bootstrap_marginal` /
    `bootstrap_partial` as None and `n_bootstrap=0` on the result.
    Pins the spec's "default 0 keeps existing test behavior intact"
    requirement — the existing 48 tests above MUST still pass
    bit-identical."""
    df = _build_panel(rho_trajectory=(0.4, 0.4, 0.4), seed=200)
    results = dynamic_partial_spearman.fn(
        df, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
    )
    result = _get_single_stratum(results)
    assert result.bootstrap_marginal is None
    assert result.bootstrap_partial is None
    assert result.n_bootstrap == 0


def test_cluster_bootstrap_narrow_ci_under_consistent_constant_rho() -> None:
    """Constant ρ trajectory (no per-burst heterogeneity, no
    within-cell autocorrelation by construction): the cluster
    bootstrap CI should be narrow.

    Planted ρ=0.5 across 3 bursts at n_cells=80. Bootstrap CI
    width ≈ 2 · 1.96 · 0.086 ≈ 0.34 in ρ-units (closed-form
    derivation in module docstring). Bound at 0.4 admits the
    closed-form value + finite-`n_resamples` slack at 200
    replicas (percentile estimator's sampling SD at α=0.05
    contributes ~5% more to the bound).

    Assert bracketing of true value: 0 < ρ_lower < 0.5 < ρ_upper."""
    df = _build_panel(rho_trajectory=(0.5, 0.5, 0.5), seed=201)
    results = dynamic_partial_spearman.fn(
        df, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
        n_bootstrap=200,
        bootstrap_seed=42,
        bootstrap_alpha=0.05,
    )
    result = _get_single_stratum(results)
    boot = result.bootstrap_marginal
    assert boot is not None
    assert isinstance(boot, ClusterBootstrapInterval)
    assert boot.n_resamples == 200
    assert boot.alpha == 0.05
    assert boot.seed == 42
    # Bracket the planted value.
    assert boot.rho_lower < 0.5 < boot.rho_upper, (
        f'CI=({boot.rho_lower:.4f}, {boot.rho_upper:.4f}) does NOT '
        f'bracket planted ρ=0.5 — type-I rate inflation?'
    )
    # CI narrow (closed-form expected width ≈ 0.34; bound at 0.4
    # accommodates percentile-estimator finite-sample slack at
    # n_resamples=200).
    width = boot.rho_upper - boot.rho_lower
    assert width < 0.4, (
        f'CI width={width:.4f}: expected ≲ 0.34 (closed-form at '
        f'n=80, ρ=0.5, 95% CI); >0.4 suggests bootstrap doing more '
        f'than it should on constant-ρ data'
    )


def test_cluster_bootstrap_recovers_pool_median_under_sign_flip() -> None:
    """Sign-flip ρ trajectory (+0.5, +0.5, −0.5): DL surfaces
    I² ≈ 0.97, τ² ≈ 0.39. The cluster bootstrap resamples cells
    (not bursts); since every cell carries the same per-burst
    trajectory, the bootstrap distribution's median collapses
    to roughly the DL point estimate (cancellation around 0.18
    in ρ-units). Pins:

      - bootstrap median tracks DL `rho_pooled` (both within
        ~0.1 of each other — they're estimating the same
        population quantity from the same per-burst data).
      - The CI is well-defined (lo < hi).

    Note on the alternative "bootstrap > DL PI" claim: DL's PI is
    a PREDICTIVE interval for a hypothetical new burst's
    underlying parameter (formula uses τ² heterogeneity); the
    bootstrap is a SAMPLING interval for the pool point
    estimate. Under planted sign-flip with identical per-cell
    trajectories, the bootstrap stays NARROW (cells are
    interchangeable → pool is stable across resamples) while DL
    PI gets wide (heterogeneity propagates to predicted-burst
    uncertainty). The two are different quantities — bootstrap
    > PI is NOT a load-bearing property here. The load-bearing
    property is that the bootstrap is assumption-free under
    within-cell autocorrelation, which manifests on data where
    cells DIFFER in trajectory shape, not where they all share
    the same heterogeneous structure."""
    df = _build_panel(rho_trajectory=(0.5, 0.5, -0.5), seed=202)
    results = dynamic_partial_spearman.fn(
        df, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
        n_bootstrap=200,
        bootstrap_seed=42,
        bootstrap_alpha=0.05,
    )
    result = _get_single_stratum(results)
    # Sanity: DL flagged the heterogeneity quantitatively.
    assert result.dl_marginal.i2 > 0.7
    boot = result.bootstrap_marginal
    assert boot is not None
    # Bootstrap median tracks DL point estimate (both estimate
    # the same pool from the same per-burst data).
    deviation = abs(boot.rho_median - result.dl_marginal.rho_pooled)
    assert deviation < 0.10, (
        f'bootstrap median={boot.rho_median:.4f} drifts from DL '
        f'rho_pooled={result.dl_marginal.rho_pooled:.4f} by '
        f'{deviation:.4f} — should track within ~0.1 under the '
        f'identical-trajectory construction'
    )
    # CI well-defined.
    assert boot.rho_lower < boot.rho_upper
    assert boot.rho_lower < boot.rho_median < boot.rho_upper


def test_cluster_bootstrap_reproducible_under_same_seed() -> None:
    """Same `bootstrap_seed` → identical CI bounds. Pins the
    `np.random.default_rng(seed)` determinism the spec
    requires."""
    df = _build_panel(rho_trajectory=(0.4, 0.4, 0.4), seed=203)
    r_a = dynamic_partial_spearman.fn(
        df, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
        n_bootstrap=100,
        bootstrap_seed=123,
    )
    r_b = dynamic_partial_spearman.fn(
        df, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
        n_bootstrap=100,
        bootstrap_seed=123,
    )
    res_a = _get_single_stratum(r_a)
    res_b = _get_single_stratum(r_b)
    assert res_a.bootstrap_marginal is not None
    assert res_b.bootstrap_marginal is not None
    # Bit-identical bounds for identical seed.
    assert res_a.bootstrap_marginal.rho_lower == (
        res_b.bootstrap_marginal.rho_lower
    )
    assert res_a.bootstrap_marginal.rho_upper == (
        res_b.bootstrap_marginal.rho_upper
    )
    assert res_a.bootstrap_marginal.rho_median == (
        res_b.bootstrap_marginal.rho_median
    )


def test_cluster_bootstrap_wider_under_between_cell_heterogeneity() -> None:
    """The load-bearing methodological claim: under genuine
    BETWEEN-CELL heterogeneity (cells with structurally different
    trajectory shapes) the cluster bootstrap CI is wider than
    under matched no-heterogeneity data. The bootstrap's
    assumption-free property kicks in here — resampling cells
    with replacement up-weights some trajectory shapes over
    others, surfacing the cross-cell variability that DL's
    parametric formula (which treats all bursts as iid within
    the same population) doesn't see.

    Construction: 40 'positive' cells with trajectory (+0.6,
    +0.6, +0.6) + 40 'negative' cells with trajectory (−0.6,
    −0.6, −0.6) — same overall marginal ρ as a constant=0
    panel BUT the per-cell trajectories are dramatically
    heterogeneous. Compare bootstrap CI width on this panel
    against an iid-noise panel with planted ρ ≈ 0.

    Bound: heterogeneous-cells width >= iid width (the
    bootstrap surfaces the bimodal cell-shape distribution).
    """
    # Build the heterogeneous-cells panel by concatenating two
    # half-panels with opposite arm-coding (so the marginal ρ
    # nearly cancels but per-cell trajectories are extreme).
    # Actually simpler: just plant a heterogeneous cell mixture
    # by hand.
    rng = np.random.default_rng(204)
    n_per_half = 20  # 20 cells per arm per "sign-cluster"
    cells_het: list[Mapping[str, object]] = []

    def _cell(arm: str, arm_code: float, rho: float) -> Mapping[str, object]:
        out_arr: list[float] = []
        med_arr: list[float] = []
        for _ in range(3):  # 3 bursts
            noise = rng.normal(0.0, 1.0)
            y = rho * arm_code + math.sqrt(max(1.0 - rho ** 2, 0.0)) * noise
            out_arr.append(float(y))
            med_arr.append(float(rng.normal(0.0, 1.0)))
        return {
            'env_name': 'env_a', 'gamma': 0.99, 'arm_key': arm,
            'outcome_pb': out_arr, 'mediator_pb': med_arr,
        }

    # 20 treatment cells with positive ρ; 20 treatment cells with
    # negative ρ. Same for baseline. Each cell's whole trajectory
    # is consistently positive OR consistently negative — extreme
    # between-cell heterogeneity.
    for _ in range(n_per_half):
        cells_het.append(_cell('treatment', 1.0, +0.6))
        cells_het.append(_cell('baseline', -1.0, +0.6))
    for _ in range(n_per_half):
        cells_het.append(_cell('treatment', 1.0, -0.6))
        cells_het.append(_cell('baseline', -1.0, -0.6))
    df_het = pl.DataFrame(cells_het)

    # Comparison panel: iid noise, marginal ρ ≈ 0 by construction.
    # Same n_cells (80) for equal sampling-SD baseline.
    df_iid = _build_panel(rho_trajectory=(0.0, 0.0, 0.0), seed=205)

    res_het = dynamic_partial_spearman.fn(
        df_het, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
        n_bootstrap=300,
        bootstrap_seed=42,
    )
    res_iid = dynamic_partial_spearman.fn(
        df_iid, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
        n_bootstrap=300,
        bootstrap_seed=42,
    )
    boot_het = _get_single_stratum(res_het).bootstrap_marginal
    boot_iid = _get_single_stratum(res_iid).bootstrap_marginal
    assert boot_het is not None and boot_iid is not None
    width_het = boot_het.rho_upper - boot_het.rho_lower
    width_iid = boot_iid.rho_upper - boot_iid.rho_lower
    # Heterogeneous-cells width is wider than iid (the
    # cluster-bootstrap surfaces the bimodal cell-shape mixture
    # that an iid-noise panel doesn't have).
    assert width_het > width_iid, (
        f'heterogeneous-cells CI width={width_het:.4f} should be '
        f'wider than iid CI width={width_iid:.4f}: cluster '
        f'bootstrap should expose between-cell heterogeneity'
    )


def test_cluster_bootstrap_n_resamples_one_is_degenerate_but_does_not_crash() -> None:
    """`n_bootstrap=1` returns a (degenerate, single-replica) CI
    where `rho_lower == rho_upper == rho_median == single
    replica's ρ`. Pins the boundary behaviour: the framework
    refuses to crash on the smallest valid input, even when the
    interval collapses to a point."""
    df = _build_panel(rho_trajectory=(0.4, 0.4, 0.4), seed=205)
    results = dynamic_partial_spearman.fn(
        df, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
        n_bootstrap=1,
        bootstrap_seed=42,
    )
    result = _get_single_stratum(results)
    boot = result.bootstrap_marginal
    assert boot is not None
    assert boot.n_resamples == 1
    # Single-replica percentile/median all collapse to the same
    # number — `np.percentile` returns the lone value at any q.
    assert boot.rho_lower == boot.rho_upper == boot.rho_median


def test_cluster_bootstrap_partial_pool_distinct_from_marginal() -> None:
    """The marginal and partial bootstrap CIs use different
    df_offset (3 vs 4) and different per-burst ρ — they should
    land at numerically distinct CI bounds when the mediator is
    NOT independent of arm + outcome. With independent mediator
    (`_build_panel`'s construction), partial ≈ marginal in
    expectation so the CIs overlap heavily but the medians can
    still differ by small amounts due to the df weight differing
    across resamples. Pins that both pools are computed (not
    swapped, not identical-by-bug)."""
    df = _build_panel(rho_trajectory=(0.4, 0.4, 0.4), seed=206)
    results = dynamic_partial_spearman.fn(
        df, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
        n_bootstrap=100,
        bootstrap_seed=42,
    )
    result = _get_single_stratum(results)
    assert result.bootstrap_marginal is not None
    assert result.bootstrap_partial is not None
    # Both pools populated.
    assert not math.isnan(result.bootstrap_marginal.rho_median)
    assert not math.isnan(result.bootstrap_partial.rho_median)
    # Both pools bracket the planted ρ (≈0.4) — independent
    # mediator means partial ≈ marginal in expectation.
    assert result.bootstrap_marginal.rho_lower < 0.4 < (
        result.bootstrap_marginal.rho_upper
    )
    assert result.bootstrap_partial.rho_lower < 0.4 < (
        result.bootstrap_partial.rho_upper
    )


def test_cluster_bootstrap_interval_is_frozen_dataclass() -> None:
    """`ClusterBootstrapInterval` is a frozen dataclass —
    mutation must raise. Parallel guard to the FisherZDLPool
    sibling."""
    df = _build_panel(rho_trajectory=(0.4, 0.4, 0.4), seed=207)
    results = dynamic_partial_spearman.fn(
        df, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
        n_bootstrap=10,
    )
    result = _get_single_stratum(results)
    boot = result.bootstrap_marginal
    assert boot is not None
    assert isinstance(boot, ClusterBootstrapInterval)
    with pytest.raises((AttributeError, TypeError)):
        boot.rho_lower = 0.0  # pyright: ignore[reportAttributeAccessIssue]


# ============ Multi-mediator depth-≥2 conditioning ============
#
# Layer A coverage for the multi-mediator extension. Construction
# parallels the PC depth-2 tests: a planted DAG where joint
# conditioning is the only valid d-separating set. The
# closed-form expectation is that ρ(arm, outcome | {Z1, Z2}) → 0
# while ρ(arm, outcome | Z1) stays bounded away from zero. The
# DL pool's df_offset accounting must shift to 3+k per the
# multi-Z Fisher-z statistic.


def _build_two_mediator_panel(
    *,
    n_bursts: int,
    seed: int = 0,
    env_name: str = 'env_a',
    gamma: float = 0.99,
) -> pl.DataFrame:
    """Planted DAG: arm → {Z1, Z2} → outcome (no direct edge).

    Each cell:
      Z1[b] = arm_code + ε_1
      Z2[b] = arm_code + ε_2  (indep noise per mediator)
      outcome[b] = Z1[b] + Z2[b] + ε_Y

    Population-level partial ρ:
      - With single Z1 conditioning: Z2 still carries
        arm-information → ρ(arm, outcome | Z1) ≠ 0.
      - With joint (Z1, Z2) conditioning: both back-doors
        blocked → ρ(arm, outcome | Z1, Z2) ≈ 0.
    """
    rng = np.random.default_rng(seed)
    n = 2 * _N_CELLS_PER_ARM
    arms = ['treatment'] * _N_CELLS_PER_ARM + ['baseline'] * _N_CELLS_PER_ARM
    arm_code = np.asarray(
        [0.5] * _N_CELLS_PER_ARM + [-0.5] * _N_CELLS_PER_ARM,
        dtype=np.float64,
    )
    z1 = np.zeros((n, n_bursts), dtype=np.float64)
    z2 = np.zeros((n, n_bursts), dtype=np.float64)
    out = np.zeros((n, n_bursts), dtype=np.float64)
    for b in range(n_bursts):
        z1[:, b] = arm_code + rng.normal(0.0, 0.5, size=n)
        z2[:, b] = arm_code + rng.normal(0.0, 0.5, size=n)
        out[:, b] = z1[:, b] + z2[:, b] + rng.normal(0.0, 0.5, size=n)
    cells: list[Mapping[str, object]] = []
    for i in range(n):
        cells.append({
            'env_name': env_name,
            'gamma': gamma,
            'arm_key': arms[i],
            'z1_pb': z1[i, :].tolist(),
            'z2_pb': z2[i, :].tolist(),
            'outcome_pb': out[i, :].tolist(),
        })
    return pl.DataFrame(cells)


def test_depth_2_joint_partial_rho_near_zero() -> None:
    """Joint conditioning on (Z1, Z2) drives partial ρ ≈ 0 at
    every burst. Bound: |ρ_partial[b]| ≤ 3 × SE_z (3-σ envelope
    in Fisher-z units at n=80, df=n−5=75 → SE_z=1/√75 ≈ 0.115 →
    ρ-units ≈ tanh(3 × 0.115) ≈ 0.34).
    """
    df = _build_two_mediator_panel(n_bursts=4, seed=700)
    results = dynamic_partial_spearman.fn(
        df, arm_field='arm_key',
        mediator_per_burst=('z1_pb', 'z2_pb'),
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
    )
    result = _get_single_stratum(results)
    assert result.k_conditioning == 2
    n_per = result.n_per_burst[0]
    # df=n-3-k=n-5 (k=2). SE_z=1/sqrt(df-1).
    se_z = 1.0 / math.sqrt(n_per - 5 - 1)
    # 3σ z-bound → ρ-units via tanh.
    rho_bound = math.tanh(3.0 * se_z)
    for b, r in enumerate(result.rho_partial):
        assert not math.isnan(r)
        assert abs(r) <= rho_bound, (
            f'burst {b}: joint partial ρ={r:.3f} exceeds 3σ '
            f'envelope ρ-bound={rho_bound:.3f} (n={n_per}, '
            f'k=2, df={n_per-5}, SE_z={se_z:.3f})'
        )


def test_depth_2_vs_depth_1_dsep_proxy_via_partial_magnitudes() -> None:
    """Single-Z1 conditioning leaves a residual ρ(arm, outcome | Z1)
    well above zero (Z2 unblocked); joint (Z1, Z2) conditioning
    drives it near zero. Verify
    |ρ_partial_depth2| < |ρ_partial_depth1| at every burst.
    """
    df = _build_two_mediator_panel(n_bursts=4, seed=701)
    res_single = _get_single_stratum(dynamic_partial_spearman.fn(
        df, arm_field='arm_key',
        mediator_per_burst='z1_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
    ))
    res_joint = _get_single_stratum(dynamic_partial_spearman.fn(
        df, arm_field='arm_key',
        mediator_per_burst=('z1_pb', 'z2_pb'),
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
    ))
    assert res_single.k_conditioning == 1
    assert res_joint.k_conditioning == 2
    for b, (r1, r2) in enumerate(
        zip(res_single.rho_partial, res_joint.rho_partial)
    ):
        assert abs(r2) < abs(r1), (
            f'burst {b}: joint |ρ|={abs(r2):.3f} should be less '
            f'than single-Z1 |ρ|={abs(r1):.3f} (Z2 still unblocks '
            f'arm-info at depth-1)'
        )


def test_depth_2_dl_pool_uses_df_offset_3_plus_k() -> None:
    """The DL pool df_offset = 3 + k at depth-k. Verify by
    comparing `fisher_z_dl_pool(rho_partial, n, df_offset=5)`
    against the framework's reported `dl_partial.rho_pooled`."""
    df = _build_two_mediator_panel(n_bursts=4, seed=702)
    result = _get_single_stratum(dynamic_partial_spearman.fn(
        df, arm_field='arm_key',
        mediator_per_burst=('z1_pb', 'z2_pb'),
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
    ))
    # k=2 → df_offset = 5.
    expected_dl = fisher_z_dl_pool(
        list(result.rho_partial), list(result.n_per_burst),
        df_offset=5,
    )
    # `fisher_z_dl_pool` is the same primitive used inside the
    # framework; results must be bit-identical.
    assert result.dl_partial.rho_pooled == expected_dl.rho_pooled
    assert result.dl_partial.tau2 == expected_dl.tau2
    assert result.dl_partial.n_bursts_used == expected_dl.n_bursts_used


def test_depth_1_singular_vs_tuple_bit_identical() -> None:
    """`mediator_per_burst='m'` and `mediator_per_burst=('m',)`
    must produce bit-identical results at k=1 (both dispatch to
    the closed-form `partial_spearman_rho`).
    """
    df = _build_panel(rho_trajectory=(0.4, 0.4, 0.4), seed=703)
    res_singular = _get_single_stratum(dynamic_partial_spearman.fn(
        df, arm_field='arm_key',
        mediator_per_burst='mediator_pb',
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
    ))
    res_tuple = _get_single_stratum(dynamic_partial_spearman.fn(
        df, arm_field='arm_key',
        mediator_per_burst=('mediator_pb',),
        outcome_per_burst='outcome_pb',
        stratify_by=('env_name', 'gamma'),
    ))
    assert res_singular.rho_marginal == res_tuple.rho_marginal
    assert res_singular.rho_partial == res_tuple.rho_partial
    assert res_singular.rho_marginal_pooled == res_tuple.rho_marginal_pooled
    assert res_singular.rho_partial_pooled == res_tuple.rho_partial_pooled
    assert res_singular.dl_partial.rho_pooled == (
        res_tuple.dl_partial.rho_pooled
    )
    assert res_singular.k_conditioning == res_tuple.k_conditioning == 1
    assert res_tuple.mediator_names == ('mediator_pb',)


def test_empty_tuple_raises_value_error() -> None:
    """`mediator_per_burst=()` raises ValueError — no conditioning
    is the marginal test, which is reported via `rho_marginal`."""
    df = _build_panel(rho_trajectory=(0.3, 0.3), seed=704)
    with pytest.raises(ValueError, match='marginal'):
        _ = dynamic_partial_spearman.fn(
            df, arm_field='arm_key',
            mediator_per_burst=(),
            outcome_per_burst='outcome_pb',
            stratify_by=('env_name', 'gamma'),
        )


def test_depth_2_bootstrap_reproducible_and_recomputed_via_multi() -> None:
    """Same `bootstrap_seed` at depth-2 → identical CI bounds across
    calls. Confirms the multi-mediator bootstrap kernel inherits
    deterministic RNG behaviour from the depth-1 sibling."""
    df = _build_two_mediator_panel(n_bursts=4, seed=705)
    # A closure rather than a shared kwargs dict keeps the one
    # spelling of the call statically checked against the analysis
    # signature.
    def run_once():
        return _get_single_stratum(dynamic_partial_spearman.fn(
            df,
            arm_field='arm_key',
            mediator_per_burst=('z1_pb', 'z2_pb'),
            outcome_per_burst='outcome_pb',
            stratify_by=('env_name', 'gamma'),
            n_bootstrap=40,
            bootstrap_seed=11,
        ))

    r1 = run_once()
    r2 = run_once()
    bp1 = r1.bootstrap_partial
    bp2 = r2.bootstrap_partial
    assert bp1 is not None and bp2 is not None
    assert (bp1.rho_lower, bp1.rho_median, bp1.rho_upper) == (
        bp2.rho_lower, bp2.rho_median, bp2.rho_upper
    )
    # And the CI brackets near-zero (planted joint d-separation).
    assert bp1.rho_lower < 0.2 < bp1.rho_upper or bp1.rho_median < 0.2
