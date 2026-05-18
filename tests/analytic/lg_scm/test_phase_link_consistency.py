"""Direct tests on `phase_link_consistency` — the per-burst panel
reduction reporting the fraction of bursts with a significant
correlation in the predicted direction.

End-to-end LG-SCM tests at `test_paired_link_per_burst.py`
exercise the public path; this file pins the primitive's
filter / count / division boundaries that the integration tests
don't isolate."""
from __future__ import annotations

import math

import pytest

from corroborate.analyses.link.paired_link_per_burst import (
    PerBurstLinkResult,
    PerBurstLinkStratum,
    phase_link_consistency,
)


def _stratum(
    env: str, burst: int, r: float, p: float,
) -> PerBurstLinkStratum:
    """Builder: per-burst stratum with only r/p meaningful for
    `phase_link_consistency`. Other fields filled with finite
    placeholder values."""
    return PerBurstLinkStratum(
        env_name=env, burst_index=burst,
        r=r, p=p, slope=0.0,
        mean_d_predictor=0.0, mean_d_target=0.0,
        sd_d_target=1.0, n_pairs=20,
    )


def _result(strata: list[PerBurstLinkStratum]) -> PerBurstLinkResult:
    return PerBurstLinkResult(
        strata=tuple(strata),
        target='Δy', predictor='Δx',
        treatment_arm='ddqn', baseline_arm='vanilla',
        pair_by=('seed',),
    )


def test_phase_link_one_when_all_bursts_match() -> None:
    """4 bursts, all with strong significant positive r: ratio = 1.
    Pin the `1 for s in panel` count (vs `2 for ...` mutant which
    would give 2.0) and `/ len(panel)` (vs `* len(panel)` mutant
    which would give 16.0)."""
    res = _result([
        _stratum('env', i, r=0.9, p=0.001) for i in range(4)
    ])
    assert phase_link_consistency(res) == pytest.approx(1.0)


def test_phase_link_zero_when_no_bursts_match() -> None:
    """All bursts in the wrong direction: ratio = 0."""
    res = _result([
        _stratum('env', i, r=-0.9, p=0.001) for i in range(4)
    ])
    assert phase_link_consistency(res) == pytest.approx(0.0)


def test_phase_link_half_when_half_match() -> None:
    """2 of 4 bursts match → 0.5. Combined check on count + division."""
    res = _result([
        _stratum('env', 0, r=0.9, p=0.001),
        _stratum('env', 1, r=-0.9, p=0.001),
        _stratum('env', 2, r=0.9, p=0.001),
        _stratum('env', 3, r=-0.9, p=0.001),
    ])
    assert phase_link_consistency(res) == pytest.approx(0.5)


def test_phase_link_strict_p_significance_at_boundary_excludes() -> None:
    """p exactly at significance is NOT counted (strict `<`).
    Pin `p < significance` against `p <= significance` mutant
    (which would count boundary bursts)."""
    sig = 0.05
    res = _result([
        _stratum('env', 0, r=0.9, p=sig),         # boundary (excluded)
        _stratum('env', 1, r=0.9, p=sig - 0.001),  # well inside (included)
    ])
    plc = phase_link_consistency(res, significance=sig)
    assert plc == pytest.approx(0.5)


def test_phase_link_strict_r_sign_at_zero_excludes() -> None:
    """r=0 (no signal) is NOT counted as matching, even with
    p<significance somehow. Pin `(s.r * expected_sign) > 0`
    against `>= 0` mutant.

    Note: with r=0 actual p would be 1, but a constructed fixture
    can have r=0 paired with a low p — direct primitive test.
    The match fails on the strict `> 0` boundary."""
    res = _result([
        _stratum('env', 0, r=0.0, p=0.001),
        _stratum('env', 1, r=0.5, p=0.001),
    ])
    plc = phase_link_consistency(res)
    assert plc == pytest.approx(0.5)


def test_phase_link_r_in_zero_to_one_counts_as_match() -> None:
    """r=0.5 with significant p must count as matching (vs `> 1`
    mutant which would only count r > 1, impossible since r ∈ [-1, 1])."""
    res = _result([
        _stratum('env', 0, r=0.5, p=0.001),
    ])
    assert phase_link_consistency(res) == pytest.approx(1.0)


def test_phase_link_filters_by_env_name_when_provided() -> None:
    """`env_name='envA'` keeps only envA strata. Pin `s.env_name == env_name`
    against `!= env_name` mutant (would invert filter)."""
    res = _result([
        _stratum('envA', 0, r=0.9, p=0.001),    # match
        _stratum('envA', 1, r=0.9, p=0.001),    # match
        _stratum('envB', 0, r=-0.9, p=0.001),   # different env, no match
        _stratum('envB', 1, r=-0.9, p=0.001),   # different env, no match
    ])
    plc = phase_link_consistency(res, env_name='envA')
    assert plc == pytest.approx(1.0)
    plc_b = phase_link_consistency(res, env_name='envB')
    assert plc_b == pytest.approx(0.0)


def test_phase_link_returns_nan_when_panel_empty_for_env() -> None:
    """Filter to non-existent env → empty panel → NaN. Pins:

    - `tuple(s for s in result.strata if s.env_name == env_name)`
      against `tuple(None)` mutant (TypeError) and
      `s.env_name != env_name` (would keep wrong strata)
    - `return float('nan')` against `float(None)` mutant (TypeError)"""
    res = _result([
        _stratum('envA', 0, r=0.9, p=0.001),
    ])
    plc = phase_link_consistency(res, env_name='envZ')
    assert math.isnan(plc)


def test_phase_link_negative_expected_sign_inverts_match_direction() -> None:
    """`expected_sign=-1` counts negative r as matching. Validates
    the sign-multiplication pattern works in both directions."""
    res = _result([
        _stratum('env', 0, r=-0.9, p=0.001),    # match under sign=-1
        _stratum('env', 1, r=-0.9, p=0.001),    # match
        _stratum('env', 2, r=+0.9, p=0.001),    # no match
    ])
    plc = phase_link_consistency(res, expected_sign=-1)
    assert plc == pytest.approx(2 / 3)
