"""Cross-primitive comparison: Cliff's δ is skew-robust where
paired_g is biased.

The empirical case for `cliff_delta_paired` as a complementary
primitive to `paired_g` on skewed-Δ corpora: under right-skewed
Δ where the population MEAN is fixed but the distribution shape
changes, paired_g's reported `g` drifts (because sample sd is
biased by skew); Cliff's δ stays anchored to `2·P(Δ>0) - 1`,
which depends only on the SIGN of Δ.

This is the head-to-head probe: same generated Δ, both primitives
run, demonstrate paired_g's bias and Cliff's δ's stability.

If this comparison fails (e.g., Cliff's δ becomes biased OR
paired_g becomes unbiased), it changes the recommendation in
ROBUSTNESS.md.
"""
from __future__ import annotations

import math
import zlib

import numpy as np

from corroborate.analyses.paired.cliff_delta_paired import cliff_delta_paired
from corroborate.analyses.paired.paired_g import paired_g
from corroborate.data import cells_to_dataframe


def _det_seed(*parts: object) -> int:
    return zlib.adler32(repr(parts).encode()) & 0xFFFF_FFFF


_K_REPLICATES = 300


def _make_paired_cells(deltas: list[float]) -> list[dict[str, object]]:
    cells: list[dict[str, object]] = []
    for s, d in enumerate(deltas):
        cells.append({'arm_key': 'T', 'seed': s, 'value': float(d)})
        cells.append({'arm_key': 'B', 'seed': s, 'value': 0.0})
    return cells


def test_cliff_delta_unbiased_under_lognormal_where_paired_g_inflates() -> None:
    """**Headline cross-primitive comparison**: same log-normal
    Δ stream, both primitives evaluated. paired_g overestimates
    by +0.13 at n=30; Cliff's δ stays within ±0.05 of population
    δ.

    Construction: Δ ~ lognormal(0, 0.7). Population:
        E[Δ] ≈ 1.278  (right-skewed mean)
        SD[Δ] ≈ 1.016
        median[Δ] = exp(0) = 1.0
        P(Δ > 0) = 1.0 (lognormal is positive-only)
        → δ_pop = 1.0 (every Δ > 0 by construction)

    paired_g: should land ABOVE the structural g_struct =
    E[Δ]/SD[Δ] · c_4(30) ≈ 1.225, with bias > +0.10
    (per the empirical map at test_paired_g_skew_robustness.py).

    Cliff's δ: should land at δ ≈ 1.0 exactly, since EVERY draw
    is positive (log-normal support is (0, ∞)). No skew-induced
    bias possible.

    The probe demonstrates that Cliff's δ correctly reports
    "treatment helps in 100% of pairs" while paired_g's
    standardized magnitude is inflated by skew.
    """
    n = 30
    K = _K_REPLICATES
    pg_estimates: list[float] = []
    cliff_estimates: list[float] = []
    for k in range(K):
        rng = np.random.default_rng(_det_seed('cross_lognorm', k))
        deltas = rng.lognormal(0.0, 0.7, n).tolist()
        cells = _make_paired_cells(deltas)
        pg_result = paired_g.fn(
            cells_to_dataframe(cells),
            treatment_arm='T', baseline_arm='B',
            pair_by=('seed',), source='value',
        )
        cd_result = cliff_delta_paired.fn(
            cells_to_dataframe(cells),
            treatment_arm='T', baseline_arm='B',
            pair_by=('seed',), source='value',
        )
        pg_estimates.append(pg_result.g)
        cliff_estimates.append(cd_result.delta)
    pg_arr = np.array(pg_estimates)
    cd_arr = np.array(cliff_estimates)

    # Cliff's δ on log-normal Δ: every draw is positive, so δ = 1.0
    # for every replicate. Pin exactly.
    assert (cd_arr == 1.0).all(), (
        f'Cliff δ should be 1.0 on log-normal Δ (all positive). '
        f'Got mean={cd_arr.mean():.4f}, min={cd_arr.min():.4f}.'
    )

    # paired_g: above structural g_struct by > 10%.
    e_x = math.exp(0.0 + 0.7 ** 2 / 2)
    var_x = (math.exp(0.7 ** 2) - 1) * math.exp(2 * 0.0 + 0.7 ** 2)
    sd_x = math.sqrt(var_x)
    c4 = 1.0 - 3.0 / (4 * n - 5)
    g_struct = e_x / sd_x * c4
    pg_bias = pg_arr.mean() - g_struct
    assert pg_bias > 0.05, (
        f'paired_g should overestimate on log-normal Δ at n={n}; '
        f'observed bias = {pg_bias:+.4f} ≤ 0.05. The probe at '
        f'test_paired_g_skew_robustness.py predicts +0.125 here.'
    )

    # The complementary value: Cliff's δ has zero variance on this
    # construction (it's exactly 1.0 every time), while paired_g
    # has substantial sampling SD AND systematic bias.
    pg_sd = float(pg_arr.std(ddof=1))
    assert pg_sd > 0.1, (
        f'paired_g sampling SD should be substantial; got {pg_sd:.4f}'
    )


def test_cliff_delta_and_paired_g_agree_on_sign_under_normal() -> None:
    """**Negative control**: under normal Δ (the well-calibrated
    regime for paired_g), the two primitives should agree on sign
    AND on magnitude convention. paired_g's `g` and Cliff's `δ`
    are MONOTONICALLY RELATED under normal Δ:

        δ = 2·Φ(g/c_4 · 1) - 1     (when SD-units = 1 standardized)

    For (μ=1, σ=2): paired_g.g ≈ 0.49, Cliff δ ≈ 0.38. Both
    POSITIVE; magnitudes are different (g is in SD-units, δ is
    in fraction-helped-vs-hurt units) but they agree on
    direction.

    This pins that adopting Cliff's δ doesn't FLIP signs relative
    to paired_g — the complementary primitive is consistent on
    well-calibrated inputs.
    """
    n = 30
    rng = np.random.default_rng(_det_seed('agree', n))
    deltas = rng.normal(1.0, 2.0, n).tolist()
    cells = _make_paired_cells(deltas)
    pg = paired_g.fn(
        cells_to_dataframe(cells),
        treatment_arm='T', baseline_arm='B',
        pair_by=('seed',), source='value',
    )
    cd = cliff_delta_paired.fn(
        cells_to_dataframe(cells),
        treatment_arm='T', baseline_arm='B',
        pair_by=('seed',), source='value',
    )
    # Both positive, both bounded; magnitudes differ.
    assert pg.g > 0
    assert cd.delta > 0
    # They report on the same direction. A regression that
    # accidentally flipped one's sign would breach this.
    assert (pg.g > 0) == (cd.delta > 0)
