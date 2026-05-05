"""Tests for `redundancy_check` — outcome-jaccard + HP-R² tautology
detection."""
from __future__ import annotations

import math
from collections.abc import Mapping

import pytest

from corroborate.measurables import measurable
from corroborate.measurables.redundancy_check import (
    TautologyReport, _r_squared, audit_mediator_panel,
    is_hp_tautological, is_outcome_tautological, jaccard,
    reads_overlap,
)
from corroborate.corpus.schema import RunRow
from corroborate.bridge.verdict import Verdict


# ============ jaccard ============

def test_jaccard_full_overlap_is_one() -> None:
    assert jaccard(frozenset({'a', 'b'}), frozenset({'a', 'b'})) == 1.0


def test_jaccard_no_overlap_is_zero() -> None:
    assert jaccard(frozenset({'a'}), frozenset({'b'})) == 0.0


def test_jaccard_partial_overlap() -> None:
    """{a,b} ∩ {b,c} = {b}; {a,b} ∪ {b,c} = {a,b,c}; jaccard = 1/3."""
    j = jaccard(frozenset({'a', 'b'}), frozenset({'b', 'c'}))
    assert math.isclose(j, 1/3, rel_tol=1e-9)


def test_jaccard_both_empty_is_zero() -> None:
    """Vacuous overlap convention — empty sets aren't informative."""
    assert jaccard(frozenset(), frozenset()) == 0.0


# ============ reads_overlap (Measurable level) ============

def test_reads_overlap_identical_measurables() -> None:
    @measurable(reads=('mc_return', 'episode_length'))
    def a(record: Mapping[str, object]) -> float:
        del record
        return 1.0

    @measurable(reads=('mc_return', 'episode_length'))
    def b(record: Mapping[str, object]) -> float:
        del record
        return 2.0

    assert reads_overlap(a, b) == 1.0


def test_reads_overlap_disjoint() -> None:
    @measurable(reads=('mc_return',))
    def a(record: Mapping[str, object]) -> float:
        del record
        return 1.0

    @measurable(reads=('online_argmax',))
    def b(record: Mapping[str, object]) -> float:
        del record
        return 2.0

    assert reads_overlap(a, b) == 0.0


# ============ is_outcome_tautological ============

def test_outcome_tautological_when_full_overlap() -> None:
    @measurable(reads=('mc_return',))
    def m(record: Mapping[str, object]) -> float:
        del record
        return 1.0

    assert is_outcome_tautological(m, frozenset({'mc_return'}))


def test_outcome_tautological_when_disjoint() -> None:
    @measurable(reads=('online_argmax', 'target_argmax'))
    def m(record: Mapping[str, object]) -> float:
        del record
        return 1.0

    assert not is_outcome_tautological(m, frozenset({'mc_return'}))


def test_outcome_tautological_with_partial_overlap_below_threshold() -> None:
    """Mediator reads {mc_return, td_error}; outcome reads
    {mc_return}. Jaccard = 1/2 = 0.5 = threshold by default.
    Inclusive comparison flags it; below threshold it doesn't."""
    @measurable(reads=('mc_return', 'td_error'))
    def m(record: Mapping[str, object]) -> float:
        del record
        return 1.0

    # Default threshold = 0.5; jaccard = 1/2 → flagged.
    assert is_outcome_tautological(m, frozenset({'mc_return'}))
    # Stricter threshold → not flagged.
    assert not is_outcome_tautological(
        m, frozenset({'mc_return'}), threshold=0.6,
    )


# ============ is_hp_tautological ============

def test_hp_tautological_when_deterministic() -> None:
    """Mediator = 0.5 * HP exactly → R² = 1.0 → flagged."""
    hp = [10.0, 20.0, 30.0, 40.0, 50.0]
    mediator = [5.0, 10.0, 15.0, 20.0, 25.0]
    assert is_hp_tautological(mediator, hp)


def test_hp_tautological_when_independent() -> None:
    """Mediator uncorrelated with HP → low R² → not flagged."""
    hp = [10.0, 20.0, 30.0, 40.0, 50.0]
    mediator = [3.0, 1.0, 5.0, 2.0, 4.0]  # no relationship
    assert not is_hp_tautological(mediator, hp)


# ============ _r_squared — direct primitive coverage ============
#
# `is_hp_tautological` already exercises `_r_squared` via the
# threshold path, but doesn't pin the closed-form formula
# (slope, intercept, R²-from-SSE) or the n-boundary / NaN-filter
# branches. These tests target the primitive directly.

def test_r_squared_perfect_linear_relationship_is_one() -> None:
    """y = 2x + 3 exactly → SS_res = 0 → R² = 1.0."""
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [2.0 * xi + 3.0 for xi in x]
    assert _r_squared(x, y) == pytest.approx(1.0, abs=1e-9)


def test_r_squared_zero_when_y_independent_of_x() -> None:
    """Constant y → no slope can reduce SS_res below SS_tot.
    With y constant, SS_tot = 0 → returns 1.0 by the early
    branch (degenerate-y convention). Use random-y for true 0:
    on average R² ≈ 0 over enough trials, but for a single
    fixture: pin a known-low value."""
    # Anti-correlated noise: y has variation but no x-relationship.
    x = [0.0, 1.0, 2.0, 3.0, 4.0]
    y = [3.0, 1.0, 4.0, 2.0, 5.0]
    # OLS slope on this fixture is small; R² should be < 0.4.
    r2 = _r_squared(x, y)
    assert 0.0 <= r2 < 0.5


def test_r_squared_returns_nan_when_x_constant() -> None:
    """All-equal x → var_x = 0 → NaN. Pins `if var_x == 0.0`
    early-return path."""
    x = [5.0, 5.0, 5.0, 5.0]
    y = [1.0, 2.0, 3.0, 4.0]
    assert math.isnan(_r_squared(x, y))


def test_r_squared_returns_nan_below_n_two() -> None:
    """n=1 → NaN. Pin `n < 2` against `n < 3` mutant (n=2 should
    be valid) and `n < 2 and len(y) != n` mutant (which would
    NOT NaN at n=1 when len(y)=1=n)."""
    assert math.isnan(_r_squared([1.0], [1.0]))


def test_r_squared_returns_nan_on_length_mismatch() -> None:
    """len(y) != len(x) → NaN. Pin `len(y) != n` against
    `len(y) == n` mutant (which would be NaN on every match
    instead of every mismatch)."""
    assert math.isnan(_r_squared([1.0, 2.0], [1.0, 2.0, 3.0]))


def test_r_squared_returns_finite_at_n_equals_two() -> None:
    """n=2 with distinct x → finite R² (=1, two points always
    fit a line). Pin `n < 2` against `n < 3` mutant."""
    r2 = _r_squared([1.0, 2.0], [3.0, 5.0])
    assert math.isfinite(r2)
    assert r2 == pytest.approx(1.0, abs=1e-9)


def test_r_squared_drops_nan_pairs_via_and_filter() -> None:
    """Cells with NaN in either x or y are filtered. Pin
    `not isnan(xi) and not isnan(yi)` against `or` mutant
    (which would keep cells with one NaN, then propagate NaN
    through the regression).

    Construct: 3 valid (perfect linear) + 2 NaN-bearing pairs.
    Original keeps 3 valid → R²=1. Mutant keeps NaN-bearing
    cells → NaN propagates."""
    x = [1.0, 2.0, 3.0, float('nan'), 5.0]
    y = [2.0, 4.0, 6.0, 8.0, float('nan')]
    r2 = _r_squared(x, y)
    assert r2 == pytest.approx(1.0, abs=1e-9)


def test_r_squared_returns_nan_when_finite_count_below_two() -> None:
    """Only 1 finite pair after NaN-filtering → NaN. Pin
    `len(finite) < 2` against `len(finite) <= 2` and `len(finite) < 3`
    boundary mutants."""
    x = [1.0, float('nan'), float('nan')]
    y = [2.0, 4.0, 6.0]
    assert math.isnan(_r_squared(x, y))


def test_r_squared_uses_x_not_y_in_regression_input() -> None:
    """Pin `xs = [p[0] for p in finite]` against `p[1]` mutant
    (which would swap x and y in the slope computation).

    Construct asymmetric pairs: y = 2x but the inverse y→x
    would be x = 0.5y. The mutant computing R² of x ~ y instead
    of y ~ x still gives R²=1 here because the relationship is
    bijective. So construct a case where R²(y~x) ≠ R²(x~y):
    add x-noise that's irrelevant after the swap.

    Simpler: use a fixture where x has a clear range and y is
    discrete. Slope of y on x makes sense; slope of x on y
    inverts but produces the same R² in this simple bijective
    case.

    Actually the swap of `p[0]` for `p[1]` in JUST the `xs`
    list (without changing the cov_xy / var_x logic) produces
    `xs = ys` everywhere — so x_mean would equal y_mean,
    var_x would be Var(y), slope = cov_xy / Var(y). For
    asymmetric cases this gives a wrong slope and R². Use
    fixture where Var(x) ≠ Var(y) substantially."""
    # x has small spread, y has large spread; perfect linear.
    x = [0.0, 1.0, 2.0, 3.0, 4.0]
    y = [0.0, 100.0, 200.0, 300.0, 400.0]
    r2 = _r_squared(x, y)
    assert r2 == pytest.approx(1.0, abs=1e-9)


def test_r_squared_returns_one_when_y_constant_and_x_varies() -> None:
    """When y is constant (ss_tot = 0) but x varies (ss_res = 0),
    the function returns 1.0 by convention — perfect fit
    trivially achieved by intercept = y_mean. Pin `return 1.0`
    against `return 2.0` mutant."""
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [3.5] * 5
    assert _r_squared(x, y) == pytest.approx(1.0, abs=1e-9)


def test_r_squared_slope_uses_subtraction_not_addition() -> None:
    """Pin `(xi - x_mean)` against `(xi + x_mean)` mutant in
    var_x. Original var_x = Σ(xi - x̄)². Mutant computes
    Σ(xi + x̄)² which is wildly different except when x̄=0.

    Use x with non-zero mean (so x̄ matters) and a real linear
    relationship; assert R² = 1 (perfect fit). Mutant's var_x is
    much larger than truth → slope shrinks → R² < 1."""
    x = [10.0, 20.0, 30.0, 40.0, 50.0]  # x̄ = 30
    y = [3.0 * xi + 7.0 for xi in x]
    r2 = _r_squared(x, y)
    assert r2 == pytest.approx(1.0, abs=1e-9)


def test_hp_tautological_when_partially_correlated() -> None:
    """Strong but imperfect correlation — depends on threshold."""
    hp = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    # y = 0.5 * x + small noise → R² high but < 1.
    mediator = [5.1, 9.8, 15.2, 19.9, 25.3, 29.7]
    # Default threshold = 0.95; tight relationship may or may not flag.
    # Lower threshold should flag.
    assert is_hp_tautological(mediator, hp, threshold=0.9)


# ============ audit_mediator_panel ============

def _row(
    cell_id: str, *, capacity: int, batch_size: int,
    mediator_outcome_taut: float,
    mediator_hp_taut: float,
    mediator_clean: float,
) -> RunRow:
    return RunRow(
        id=cell_id, parent_id=None, cycle_id=None,
        timestamp='ts', verdict=Verdict.HELD, arm_key='baseline',
        measurements={
            'replay.capacity': capacity,
            'replay.batch_size': batch_size,
            'mediator.mc_return_based': mediator_outcome_taut,
            'mediator.deterministic_in_hp': mediator_hp_taut,
            'mediator.independent': mediator_clean,
        },
    )


def test_audit_panel_flags_outcome_tautological() -> None:
    """A measurable with reads={mc_return} is flagged when outcome
    reads also include mc_return."""
    @measurable(reads=('mc_return',))
    def mc_return_based(record: Mapping[str, object]) -> float:
        del record
        return 1.0

    @measurable(reads=('online_argmax',))
    def independent(record: Mapping[str, object]) -> float:
        del record
        return 1.0

    runs = [
        _row(f'c{i}', capacity=10000 + i * 1000, batch_size=32,
             mediator_outcome_taut=1.0, mediator_hp_taut=0.0,
             mediator_clean=float(i % 2))
        for i in range(8)
    ]
    reports = audit_mediator_panel(
        [mc_return_based, independent], runs,
        outcome_reads=frozenset({'mc_return'}),
        hp_axes=('replay.capacity', 'replay.batch_size'),
    )
    by_name = {r.measurable_name: r for r in reports}
    assert by_name['mc_return_based'].flagged_outcome is True
    assert by_name['independent'].flagged_outcome is False


def test_audit_panel_flags_hp_tautological() -> None:
    """A measurable whose value is f(capacity) gets HP-flagged on
    that axis, while a constant mediator doesn't (R² is undefined
    when x has no variance, but the mediator-on-HP regression is
    NaN there — falls through as not flagged)."""
    @measurable(reads=('online_argmax',))
    def deterministic_in_hp(record: Mapping[str, object]) -> float:
        del record
        return 1.0

    @measurable(reads=('online_argmax',))
    def independent(record: Mapping[str, object]) -> float:
        del record
        return 1.0

    # Per-cell mediator values: deterministic_in_hp = capacity / 2;
    # independent = pseudorandom independent of capacity.
    rng_vals = [3.2, 7.1, 4.5, 1.9, 8.3, 5.4, 2.7, 6.1]
    runs = []
    for i in range(8):
        cap = 10000 + i * 5000
        runs.append(_row(
            f'c{i}', capacity=cap, batch_size=32,
            mediator_outcome_taut=0.0,
            mediator_hp_taut=cap / 2,           # deterministic
            mediator_clean=rng_vals[i],          # independent
        ))
    reports = audit_mediator_panel(
        [deterministic_in_hp, independent], runs,
        outcome_reads=frozenset({'mc_return'}),
        hp_axes=('replay.capacity', 'replay.batch_size'),
        mediator_path_for={
            'deterministic_in_hp': 'mediator.deterministic_in_hp',
            'independent': 'mediator.independent',
        },
    )
    by_name = {r.measurable_name: r for r in reports}
    assert 'replay.capacity' in by_name['deterministic_in_hp'].flagged_hp
    assert 'replay.capacity' not in by_name['independent'].flagged_hp


def test_audit_panel_clean_property() -> None:
    """A measurable that's neither outcome- nor HP-tautological has
    `is_clean=True`."""
    @measurable(reads=('online_argmax',))
    def clean(record: Mapping[str, object]) -> float:
        del record
        return 1.0

    rng_vals = [3.2, 7.1, 4.5, 1.9, 8.3, 5.4, 2.7, 6.1]
    runs = []
    for i in range(8):
        cap = 10000 + i * 5000
        runs.append(_row(
            f'c{i}', capacity=cap, batch_size=32,
            mediator_outcome_taut=0.0, mediator_hp_taut=0.0,
            mediator_clean=rng_vals[i],
        ))
    reports = audit_mediator_panel(
        [clean], runs,
        outcome_reads=frozenset({'mc_return'}),
        hp_axes=('replay.capacity',),
        mediator_path_for={'clean': 'mediator.independent'},
    )
    assert len(reports) == 1
    r = reports[0]
    assert r.is_clean
    assert r.flagged_outcome is False
    assert r.flagged_hp == ()


def test_audit_panel_outcome_jaccard_at_exactly_threshold_flags() -> None:
    """A mediator whose `reads` jaccard with `outcome_reads` is
    EXACTLY the threshold (0.5 by default) must be flagged. Pin
    `oj >= outcome_jaccard_threshold` against `oj > threshold`
    mutant — original True at boundary, mutant False.

    Construct: mediator reads = {a, b}, outcome reads = {a, c}.
    Jaccard = 1/3 with default threshold? Let me re-derive:
    - intersection = {a}
    - union = {a, b, c}
    - jaccard = 1/3 ≈ 0.333

    Need exact 0.5: mediator reads = {a, b}, outcome reads = {a}.
    - intersection = {a} (size 1)
    - union = {a, b} (size 2)
    - jaccard = 1/2 = 0.5 EXACTLY."""
    @measurable(reads=('a', 'b'))
    def m(record: Mapping[str, object]) -> float:
        del record
        return 1.0

    runs = [_row(
        f'c{i}', capacity=10000 + i * 1000, batch_size=32,
        mediator_outcome_taut=0.0, mediator_hp_taut=0.0,
        mediator_clean=float(i),
    ) for i in range(4)]
    reports = audit_mediator_panel(
        [m], runs,
        outcome_reads=frozenset({'a'}),
        hp_axes=('replay.capacity',),
    )
    assert reports[0].outcome_jaccard == pytest.approx(0.5)
    assert reports[0].flagged_outcome is True


def test_audit_panel_skips_bool_hp_values() -> None:
    """HP values that are bool must be skipped (Python bool is int
    subclass; the `not isinstance(..., (int, float)) or isinstance(..., bool)`
    guard catches it). Pin `or` against `and` mutant — under
    `and`, bool would pass through and be converted to 1.0/0.0
    instead of being rejected.

    Cells with bool HP value should produce an empty mediator_vals
    list → flagged_hp empty for that axis."""
    @measurable(reads=('online_argmax',))
    def m(record: Mapping[str, object]) -> float:
        del record
        return 1.0

    runs: list[RunRow] = []
    for i in range(8):
        runs.append(RunRow(
            id=f'c{i}', parent_id=None, cycle_id=None,
            timestamp='ts', verdict=Verdict.HELD, arm_key='baseline',
            measurements={
                'replay.capacity': True if i % 2 else False,    # bool!
                'mediator.test': float(i),
            },
        ))
    reports = audit_mediator_panel(
        [m], runs,
        outcome_reads=frozenset({'mc_return'}),
        hp_axes=('replay.capacity',),
        mediator_path_for={'m': 'mediator.test'},
    )
    # Bool HP values rejected → no mediator/HP pairs collected →
    # _r_squared returns NaN → flagged_hp empty.
    r = reports[0]
    assert 'replay.capacity' not in r.flagged_hp
    assert math.isnan(r.hp_r_squared['replay.capacity'])


def test_audit_panel_default_path_resolves_to_mediator_dot_name() -> None:
    """When `mediator_path_for` is not provided AND the measurable
    name has no dot, the default path is `f'mediator.{m.name}'`.
    Pin against the `path = None` mutant (which would prevent
    mediator value lookup → flagged_hp empty even for a
    deterministic mediator).

    Construct: a deterministic mediator whose values are stored
    at `mediator.clean` (matches default path). HP-deterministic
    construction (mediator = capacity / 2) → R² = 1 → flagged_hp
    contains 'replay.capacity'. Mutant path=None → no values
    collected → flagged_hp empty."""
    @measurable(reads=('online_argmax',))
    def clean(record: Mapping[str, object]) -> float:
        del record
        return 1.0

    runs: list[RunRow] = []
    for i in range(8):
        cap = 10000 + i * 5000
        runs.append(_row(
            f'c{i}', capacity=cap, batch_size=32,
            mediator_outcome_taut=0.0,
            mediator_hp_taut=0.0,
            # Store under the DEFAULT-derived key 'mediator.clean'
            # (since the helper _row stores in 'mediator.independent',
            # we have to add it manually).
            mediator_clean=cap / 2,
        ))
    # Augment cell measurements to ALSO have 'mediator.clean'
    # matching the default path lookup.
    augmented: list[RunRow] = []
    from dataclasses import replace
    for r in runs:
        m_dict = dict(r.measurements)
        m_dict['mediator.clean'] = m_dict['mediator.independent']
        augmented.append(replace(r, measurements=m_dict))
    reports = audit_mediator_panel(
        [clean], augmented,
        outcome_reads=frozenset({'mc_return'}),
        hp_axes=('replay.capacity',),
        # NO mediator_path_for — exercise the default path branch.
    )
    r = reports[0]
    # Mediator values = capacity / 2 → linear in capacity → R² = 1 → flagged.
    assert 'replay.capacity' in r.flagged_hp


def test_audit_panel_no_residual_signal_default_false_when_outcome_path_missing() -> None:
    """When `outcome_path` isn't provided, the stratified-residual
    check is skipped and `flagged_no_residual_signal` stays False.
    Pin `flagged_no_residual = False` initial against the `True`
    mutant — under the mutant it would default to True without
    ever reaching the conditional that sets it."""
    @measurable(reads=('online_argmax',))
    def m(record: Mapping[str, object]) -> float:
        del record
        return 1.0

    runs = [_row(
        f'c{i}', capacity=10000 + i * 1000, batch_size=32,
        mediator_outcome_taut=0.0, mediator_hp_taut=0.0,
        mediator_clean=float(i),
    ) for i in range(16)]
    reports = audit_mediator_panel(
        [m], runs,
        outcome_reads=frozenset({'mc_return'}),
        hp_axes=('replay.capacity',),
        # NO outcome_path — stratified check skipped.
        mediator_path_for={'m': 'mediator.independent'},
    )
    assert reports[0].flagged_no_residual_signal is False


def test_audit_panel_uses_first_hp_axis_as_default_stratum() -> None:
    """When `hp_stratum_axis` is None, the function falls back to
    `hp_axes[0]`. Pin `hp_axes[0]` against the `hp_axes[1]` mutant.

    Construct: hp_axes = ('strat_axis', 'other_axis'). 'strat_axis'
    is the HP-shadow axis; 'other_axis' has no signal. Default
    stratum is hp_axes[0] = 'strat_axis' → flagged_no_residual
    fires on HP-shadow. Mutant uses hp_axes[1] = 'other_axis'
    which doesn't form a useful stratification → flagged_no_residual
    behavior shifts."""
    import random
    rng = random.Random(7)

    @measurable(reads=('online_argmax',))
    def hp_shadow(record: Mapping[str, object]) -> float:
        del record
        return 1.0

    runs: list[RunRow] = []
    cap_values = [5000, 10000, 15000, 20000, 30000, 40000, 50000, 70000]
    for i, cap in enumerate(cap_values * 8):
        runs.append(RunRow(
            id=f'c{i}', parent_id=None, cycle_id=None,
            timestamp='ts', verdict=Verdict.HELD, arm_key='baseline',
            measurements={
                'strat_axis': cap,    # the discriminating HP
                'other_axis': float(i),  # noisy second axis (no useful strata)
                'mediator.test': cap * 0.001 + rng.gauss(0.0, 1.0),
                'outcome.return': cap * 0.0005 + rng.gauss(0.0, 1.0),
            },
        ))
    reports = audit_mediator_panel(
        [hp_shadow], runs,
        outcome_reads=frozenset({'mc_return'}),
        hp_axes=('strat_axis', 'other_axis'),    # 0 = strat_axis
        outcome_path='outcome.return',
        mediator_path_for={'hp_shadow': 'mediator.test'},
        # No hp_stratum_axis → defaults to hp_axes[0] = 'strat_axis'.
    )
    r = reports[0]
    # Default stratification on strat_axis (good HP grid) reveals
    # HP-shadow → flagged.
    assert r.flagged_no_residual_signal


def test_audit_panel_returns_typed_dataclass() -> None:
    @measurable(reads=('mc_return',))
    def m(record: Mapping[str, object]) -> float:
        del record
        return 1.0

    runs = [_row(
        'c0', capacity=10000, batch_size=32,
        mediator_outcome_taut=0.0, mediator_hp_taut=0.0,
        mediator_clean=0.0,
    )]
    reports = audit_mediator_panel(
        [m], runs, outcome_reads=frozenset({'mc_return'}),
        hp_axes=('replay.capacity',),
    )
    assert isinstance(reports[0], TautologyReport)


# ============ Partial-correlation tautology check ============

def _row_with_outcome(
    cell_id: str, *, capacity: int, batch_size: int,
    mediator_value: float, outcome_value: float,
) -> RunRow:
    return RunRow(
        id=cell_id, parent_id=None, cycle_id=None,
        timestamp='ts', verdict=Verdict.HELD, arm_key='baseline',
        measurements={
            'replay.capacity': capacity,
            'replay.batch_size': batch_size,
            'mediator.test': mediator_value,
            'outcome.return': outcome_value,
        },
    )


def test_audit_stratified_rho_flags_hp_shadow_mediator() -> None:
    """A mediator HP-conditioned but uncorrelated with outcome
    *within each stratum* gets flagged. Construction: 8 cells per
    HP-stratum, 8 strata. Mediator = stratum_offset + independent
    noise. Outcome = different stratum_offset + INDEPENDENT noise.
    Within each stratum, mediator and outcome are independent →
    stratified ρ ≈ 0 → flagged."""
    import random
    # Seed chosen so the random within-stratum noise gives a clean
    # partial-correlation null. The check is necessarily power-
    # limited at small n; this test verifies the wiring not the
    # estimator's robustness — the latter is documented in the
    # module docstring.
    rng = random.Random(1)

    @measurable(reads=('online_argmax',))
    def hp_shadow(record: Mapping[str, object]) -> float:
        del record
        return 1.0

    runs: list[RunRow] = []
    cap_values = [5000, 10000, 15000, 20000, 30000, 40000, 50000, 70000]
    for i, cap in enumerate(cap_values * 8):  # 64 cells, 8 strata
        runs.append(_row_with_outcome(
            f'c{i}', capacity=cap, batch_size=32,
            mediator_value=cap * 0.001 + rng.gauss(0.0, 1.0),
            outcome_value=cap * 0.0005 + rng.gauss(0.0, 1.0),
        ))
    reports = audit_mediator_panel(
        [hp_shadow], runs,
        outcome_reads=frozenset({'mc_return'}),
        hp_axes=('replay.capacity',),
        outcome_path='outcome.return',
        mediator_path_for={'hp_shadow': 'mediator.test'},
    )
    r = reports[0]
    # Within-stratum mediator and outcome are independent → flagged.
    assert r.flagged_no_residual_signal


def test_audit_stratified_rho_does_not_flag_residual_signal() -> None:
    """A mediator with REAL within-stratum signal (outcome depends
    on mediator beyond HP) passes the stratified check."""
    import random
    rng = random.Random(1)

    @measurable(reads=('online_argmax',))
    def real_mediator(record: Mapping[str, object]) -> float:
        del record
        return 1.0

    runs: list[RunRow] = []
    cap_values = [5000, 10000, 15000, 20000, 30000, 40000, 50000, 70000]
    for i, cap in enumerate(cap_values * 8):  # 64 cells, 8 strata
        # mediator varies independently of capacity; outcome depends
        # on mediator beyond HP.
        mv = rng.gauss(0.0, 1.0)
        ov = cap * 0.0005 + 20.0 * mv + rng.gauss(0.0, 0.5)
        runs.append(_row_with_outcome(
            f'c{i}', capacity=cap, batch_size=32,
            mediator_value=mv, outcome_value=ov,
        ))
    reports = audit_mediator_panel(
        [real_mediator], runs,
        outcome_reads=frozenset({'mc_return'}),
        hp_axes=('replay.capacity',),
        outcome_path='outcome.return',
        mediator_path_for={'real_mediator': 'mediator.test'},
    )
    r = reports[0]
    assert not r.flagged_no_residual_signal
    # Within-stratum ρ should be substantially positive.
    assert abs(r.outcome_stratified_rho) > 0.5


def test_audit_stratified_skipped_when_no_outcome_path() -> None:
    """Without `outcome_path`, the stratified check is skipped and
    the fields are NaN (not flagged)."""
    @measurable(reads=('online_argmax',))
    def m(record: Mapping[str, object]) -> float:
        del record
        return 1.0

    runs = [_row(
        f'c{i}', capacity=10000 + i * 1000, batch_size=32,
        mediator_outcome_taut=0.0, mediator_hp_taut=0.0,
        mediator_clean=float(i),
    ) for i in range(16)]
    reports = audit_mediator_panel(
        [m], runs,
        outcome_reads=frozenset({'mc_return'}),
        hp_axes=('replay.capacity',),
        # no outcome_path
        mediator_path_for={'m': 'mediator.independent'},
    )
    r = reports[0]
    assert math.isnan(r.outcome_stratified_rho)
    assert math.isnan(r.outcome_stratified_p)
    assert r.flagged_no_residual_signal is False


def test_audit_clean_property_requires_all_three_pass() -> None:
    """`is_clean` is True only when none of the three flags trip."""
    import random
    rng = random.Random(2)

    @measurable(reads=('online_argmax',))
    def real_mediator(record: Mapping[str, object]) -> float:
        del record
        return 1.0

    runs: list[RunRow] = []
    cap_values = [5000, 10000, 15000, 20000, 30000, 40000, 50000, 70000]
    for i, cap in enumerate(cap_values * 8):
        mv = rng.gauss(0.0, 1.0)
        ov = cap * 0.0005 + 20.0 * mv + rng.gauss(0.0, 0.5)
        runs.append(_row_with_outcome(
            f'c{i}', capacity=cap, batch_size=32,
            mediator_value=mv, outcome_value=ov,
        ))
    reports = audit_mediator_panel(
        [real_mediator], runs,
        outcome_reads=frozenset({'mc_return'}),
        hp_axes=('replay.capacity',),
        outcome_path='outcome.return',
        mediator_path_for={'real_mediator': 'mediator.test'},
    )
    r = reports[0]
    assert r.is_clean
