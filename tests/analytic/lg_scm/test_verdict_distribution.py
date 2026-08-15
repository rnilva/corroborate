"""Closed-form assertions on `verdict_distribution_per_env` —
specifically on the *non-trivial* logic: dominant-verdict
resolution (strict-majority / tie / unanimous), case-folding of
verdict strings, fall-through to the `other` bucket on unknown
strings, and arm filtering.

NOT tested here (intentionally): verbatim count tally and
held/violation fractions. Those are I/O — the test would just
read back what the test stamped, exercising no framework logic.
The audit pass dropped them.

The implementation produces real LG-SCM cells; only the categorical
verdict column is synthetic (the analysis takes verdict strings
as opaque categories — the LG-SCM doesn't structurally produce
verdict mixes, so synthetic stamping IS the test setup, not a
mock)."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from corroborate.analyses.diagnostic.verdict_distribution import (
    verdict_distribution_per_env,
)
from corroborate.bridge.verdict import Verdict
from corroborate.corpus.schema import MeasurementLeaf, RunRow

from tests.analytic.lg_scm.composition import LinearGaussianSCM
from tests.analytic.lg_scm.runner import run_arm


_N_PER_ENV = 30
_VERDICT_COLUMN = 'mech_verdict'


def _scm() -> LinearGaussianSCM:
    return LinearGaussianSCM(
        mu_x=1.0, sigma_x=0.5, beta_xz=0.5, sigma_z=0.1,
        beta_zy=1.5, sigma_y=0.1, n_steps=200,
    )


def _stamp(rows: list[RunRow], verdicts: list[str]) -> list[RunRow]:
    """Augment each RunRow's measurements with a per-cell verdict
    string. Length must match."""
    if len(rows) != len(verdicts):
        raise ValueError(
            f'_stamp: rows ({len(rows)}) / verdicts ({len(verdicts)}) '
            f'length mismatch',
        )
    out: list[RunRow] = []
    for r, v in zip(rows, verdicts):
        m: dict[str, MeasurementLeaf] = dict(r.measurements)
        m[_VERDICT_COLUMN] = v
        out.append(replace(r, measurements=m))
    return out


def _build_dominance_corpus() -> list[Mapping[str, object]]:
    """Three envs designed to force every branch of the dominant-
    resolution logic:

      env_strict_majority — held=24, power_insufficient=6 → 'held'
      env_tie             — held=15, invariant_violation=15 → ''
      env_unanimous       — invariant_violation=30 → 'invariant_violation'
    """
    held = Verdict.HELD.value
    pi = Verdict.POWER_INSUFFICIENT.value
    iv = Verdict.INVARIANT_VIOLATION.value
    plan = {
        'env_strict_majority': [held] * 24 + [pi] * 6,
        'env_tie': [held] * 15 + [iv] * 15,
        'env_unanimous': [iv] * 30,
    }
    rows: list[RunRow] = []
    for env_name, verdicts in plan.items():
        env_rows = run_arm(
            _scm(), seeds=range(_N_PER_ENV),
            arm_key='single', env_name=env_name,
        )
        rows.extend(_stamp(env_rows, verdicts))
    return [r.as_dict() for r in rows]


# ============ Dominant logic — the real classification work ============

def test_dominant_returns_power_insufficient_string_when_dominant() -> None:
    """A corpus where POWER_INSUFFICIENT is the dominant verdict
    must report exactly the lowercase enum string `'power_insufficient'`.
    Pin the literal string in `_dominant`'s tuple — mutations to
    `'POWER_INSUFFICIENT'` (uppercase) or `'XXpower_insufficientXX'`
    (mangled) would break round-trip with the enum value."""
    pi = Verdict.POWER_INSUFFICIENT.value
    held = Verdict.HELD.value
    rows = run_arm(
        _scm(), seeds=range(_N_PER_ENV),
        arm_key='single', env_name='env_pi_dominant',
    )
    rows = _stamp(rows, [pi] * 24 + [held] * 6)
    cells: list[Mapping[str, object]] = [r.as_dict() for r in rows]
    result = verdict_distribution_per_env.fn(
        cells, arm_filter='single', verdict_column=_VERDICT_COLUMN,
    )
    assert result.per_env['env_pi_dominant'].dominant == 'power_insufficient'


def test_dominant_returns_label_when_top_count_is_one() -> None:
    """When the highest bucket count is exactly 1 (e.g., a
    single-cell env), `_dominant` must still return the label
    of that bucket, not ''. Pin
    `if sorted_pairs[0][1] == 0: return ''` against
    `== 1` mutant which would also short-circuit at top=1."""
    held = Verdict.HELD.value
    rows = run_arm(
        _scm(), seeds=range(1),
        arm_key='single', env_name='env_single_cell',
    )
    rows = _stamp(rows, [held])
    cells: list[Mapping[str, object]] = [r.as_dict() for r in rows]
    result = verdict_distribution_per_env.fn(
        cells, arm_filter='single', verdict_column=_VERDICT_COLUMN,
    )
    counts = result.for_env('env_single_cell')
    assert counts is not None
    assert counts.dominant == 'held'


def test_dominant_returns_other_string_when_other_bucket_wins() -> None:
    """The 'other' bucket holds unknown verdict strings. When
    that bucket dominates, the returned label must be the literal
    `'other'`. Pin against `'OTHER'` / `'XXotherXX'` mutations."""
    rows = run_arm(
        _scm(), seeds=range(_N_PER_ENV),
        arm_key='single', env_name='env_other_dominant',
    )
    rows = _stamp(rows, ['unknown_verdict'] * 24 + [Verdict.HELD.value] * 6)
    cells: list[Mapping[str, object]] = [r.as_dict() for r in rows]
    result = verdict_distribution_per_env.fn(
        cells, arm_filter='single', verdict_column=_VERDICT_COLUMN,
    )
    assert result.per_env['env_other_dominant'].dominant == 'other'


def test_dominant_resolves_strict_majority_tie_and_unanimous() -> None:
    """`dominant` returns:
      - the strict-majority bucket name when one is highest
      - `''` on a tie between the top two
      - `''` when all counts are zero (empty env — not present here)

    Three of the four logical branches are exercised on this
    corpus. The tie case is the most subtle — a regression that
    used `>=` instead of `>` in the dominance comparison would
    return whichever bucket appeared first, masking the tie."""
    cells = _build_dominance_corpus()
    result = verdict_distribution_per_env.fn(
        cells, arm_filter='single', verdict_column=_VERDICT_COLUMN,
    )
    by_env = result.per_env
    assert by_env['env_strict_majority'].dominant == Verdict.HELD.value
    assert by_env['env_unanimous'].dominant == Verdict.INVARIANT_VIOLATION.value
    assert by_env['env_tie'].dominant == '', (
        f"env_tie dominant = {by_env['env_tie'].dominant!r}, "
        f"expected '' — strict tie between held and "
        f'invariant_violation should NOT silently pick one'
    )


# ============ Case folding + 'other' bucket ============

def test_unknown_verdict_strings_route_to_other_bucket() -> None:
    """The analysis classifies four canonical verdicts (held,
    invariant_violation, power_insufficient) and routes everything
    else to `other`. A regression that crashed on unknown strings
    or routed them to held would fail here.

    Includes case variants: the docstring promises case-folding,
    so 'HELD' and 'Held' should both land in the `held` bucket."""
    rows = run_arm(_scm(), seeds=range(20),
                   arm_key='single', env_name='env_mixed')
    # Mix: 5 canonical lower, 5 upper-case (should fold), 5 unknown,
    # 5 mixed-case canonical.
    plan = (
        ['held'] * 5
        + ['HELD'] * 5
        + ['totally_made_up'] * 5
        + ['Held'] * 5
    )
    rows = _stamp(rows, plan)
    cells: list[Mapping[str, object]] = [r.as_dict() for r in rows]
    result = verdict_distribution_per_env.fn(
        cells, arm_filter='single', verdict_column=_VERDICT_COLUMN,
    )
    counts = result.for_env('env_mixed')
    assert counts is not None
    assert counts.held == 15, (
        f'held = {counts.held}, expected 15 (5 lower + 5 upper + '
        f'5 mixed-case, all should case-fold to held)'
    )
    assert counts.other == 5, (
        f'other = {counts.other}, expected 5 (one unknown verdict '
        f'string repeated 5 times)'
    )
    assert counts.invariant_violation == 0
    assert counts.power_insufficient == 0
    assert counts.total == 20


# ============ Arm filter discrimination ============

def test_arm_filter_excludes_other_arms() -> None:
    """Cells from a different arm should NOT contribute. Build
    cells under arm_key='single' AND arm_key='other'; filtering on
    'single' must drop the 'other' cells silently. A regression
    that ignored arm_filter (or matched substrings) would leak."""
    rows: list[RunRow] = []
    rows.extend(_stamp(
        run_arm(_scm(), seeds=range(30), arm_key='single', env_name='env_x'),
        [Verdict.HELD.value] * 30,
    ))
    rows.extend(_stamp(
        run_arm(_scm(), seeds=range(30), arm_key='other', env_name='env_x'),
        [Verdict.INVARIANT_VIOLATION.value] * 30,
    ))
    cells = [r.as_dict() for r in rows]
    result = verdict_distribution_per_env.fn(
        cells, arm_filter='single', verdict_column=_VERDICT_COLUMN,
    )
    counts = result.for_env('env_x')
    assert counts is not None
    assert counts.held == 30
    assert counts.invariant_violation == 0, (
        f'invariant_violation = {counts.invariant_violation}; the '
        f"30 cells under arm_key='other' should be filtered out"
    )
    assert counts.total == 30


# ============ Empty env returns empty dominant ============

def test_empty_env_returns_no_per_env_entry() -> None:
    """When no cells match the arm_filter for any env, the result's
    per_env mapping is empty. Asking for a non-present env returns
    None — not an error, not a phantom entry. Bridges that consume
    `for_env` rely on this `None` distinction to map to
    POWER_INSUFFICIENT."""
    rows = _stamp(
        run_arm(_scm(), seeds=range(30), arm_key='other', env_name='env_y'),
        [Verdict.HELD.value] * 30,
    )
    cells = [r.as_dict() for r in rows]
    result = verdict_distribution_per_env.fn(
        cells, arm_filter='single', verdict_column=_VERDICT_COLUMN,
    )
    assert result.per_env == {}, (
        f'per_env = {dict(result.per_env)!r}; arm_filter=single '
        f'matches no cell, panel must be empty'
    )
    assert result.for_env('env_y') is None
