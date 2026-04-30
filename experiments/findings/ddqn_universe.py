"""DDQN measurement graph — universal-dataset bridges.

Built from scratch on the paired-delta cells universe
(`experiments/data/ddqn_universal/paired_delta_cells.parquet`).
Each cell already encodes the (ddqn − vanilla_dqn) contrast as
scalars; bridges here filter by per-cell scope predicates and
report whether DDQN's outcome benefit holds *on the in-scope
subset*. The corpus-by-corpus zoo in `dqn_bridges.py` is not
imported.

Two terminal claims:

  1. **DDQN helps where vanilla is failing AND has bias to undo.**
     Empirical scope predicate: `jensen_gap_vanilla ≥ 5.0` AND
     `outcome_final_vanilla ≤ 0.5`. HELD when ≥50% of in-scope
     cells see Δoutcome > 0 AND pooled g > +0.10.

  2. **DDQN's mechanism premise being dormant ⇒ DDQN does NOT
     help.** Scope predicate: `dormancy_gap_avg > 0`. HELD as a
     refutation of the help claim — corroborated when ≤15% of
     in-scope cells see Δoutcome > 0. INVARIANT_VIOLATION when
     DDQN unexpectedly helps despite dormancy.

The framework's own dormancy invariant is therefore the
NECESSARY condition for DDQN's outcome benefit (bridge 2);
the empirical predicate (bridge 1) is the SUFFICIENT-ish
condition that gets us to ~50% helped rate.
"""
from __future__ import annotations

import math

import corroborate.analyses  # pyright: ignore[reportUnusedImport]  # populate registry
from corroborate.analyses.universe_scope import UniverseScopeResult
from corroborate.claim_bridge import (
    Direction, Tier, claim_bridge,
)
from corroborate.verdict import Verdict


@claim_bridge
def ddqn_helps_within_empirical_scope(
    universe_scope: UniverseScopeResult,
    *,
    source: str = 'arm.ddqn',
    target: str = 'outcome.eval_best_burst_mean',
    direction: Direction = Direction.DIRECT,
    tier: Tier = Tier.ASSOCIATIONAL,
    outcome_col: str = 'delta_outcome_best',
    delta_jensen_col: str = 'delta_jensen_gap',
    filter_min_pairs: tuple[tuple[str, float], ...] = (
        ('log_obs_dim', 5.0),
        ('bias_late_vanilla', 5.0),
        ('mc_peak_burst_vanilla', 3.0),
    ),
    filter_max_pairs: tuple[tuple[str, float], ...] = (),
) -> Verdict:
    """Universal scope claim: on cells where (a) the env is
    high-obs-dim (log_obs_dim ≥ 5 — pixel-input MinAtar +
    FourRooms-class), (b) the vanilla baseline shows substantial
    late-phase Jensen-bias (≥5), and (c) vanilla shows
    non-trivial learning dynamics (mc peaks at burst ≥ 3, not
    burst 0-1 saturation), DDQN's outcome delta is positive in
    the majority of cells with substantial pooled effect.

    Time-sliced features were chosen over cell-mean ones (rev 9
    motivation: mechanism operates early; outcome stabilizes
    later). Concretely the time-sliced composite achieves
    helped=57%, g=+0.36, n=124 — vs the cell-mean composite at
    helped=52%, g=+0.17, n=256.

    HELD when helped_fraction ≥ 0.5 AND g_outcome ≥ +0.20.
    NO_EFFECT otherwise; POWER_INSUFFICIENT when n_in_scope <
    100."""
    del source, target, direction, tier
    del outcome_col, delta_jensen_col
    del filter_min_pairs, filter_max_pairs
    if universe_scope.n_in_scope < 100:
        return Verdict.POWER_INSUFFICIENT
    if math.isnan(universe_scope.helped_fraction):
        return Verdict.POWER_INSUFFICIENT
    if (
        universe_scope.helped_fraction >= 0.5
        and universe_scope.g_outcome >= 0.20
    ):
        return Verdict.HELD
    return Verdict.NO_EFFECT


@claim_bridge
def ddqn_refuted_when_dormancy_fires(
    universe_scope: UniverseScopeResult,
    *,
    source: str = 'invariant.jensen_dormancy_gap',
    target: str = 'outcome.eval_best_burst_mean',
    direction: Direction = Direction.INVERSE,
    tier: Tier = Tier.ASSOCIATIONAL,
    outcome_col: str = 'delta_outcome_best',
    delta_jensen_col: str = 'delta_jensen_gap',
    filter_min_pairs: tuple[tuple[str, float], ...] = (
        ('dormancy_gap_avg', 1e-9),
    ),
    filter_max_pairs: tuple[tuple[str, float], ...] = (),
) -> Verdict:
    """Framework's-own dormancy invariant as a NECESSARY
    condition for DDQN's outcome benefit. The bridge predicts:
    when premise is dormant (dormancy_gap > 0 — observed bias
    below the structural Jensen floor), DDQN almost never helps
    outcome.

    HELD when helped_fraction < 0.15 (refutation corroborated —
    DDQN is reliably DOESN'T help under dormancy).
    INVARIANT_VIOLATION when helped_fraction > 0.40 (dormancy
    invariant fails as a falsifier — DDQN helps even when the
    framework says premise is dormant)."""
    del source, target, direction, tier
    del outcome_col, delta_jensen_col
    del filter_min_pairs, filter_max_pairs
    if universe_scope.n_in_scope < 50:
        return Verdict.POWER_INSUFFICIENT
    if math.isnan(universe_scope.helped_fraction):
        return Verdict.POWER_INSUFFICIENT
    if universe_scope.helped_fraction < 0.15:
        return Verdict.HELD
    if universe_scope.helped_fraction > 0.40:
        return Verdict.INVARIANT_VIOLATION
    return Verdict.NO_EFFECT


# === The DDQN measurement graph (built from scratch) ===
DDQN_UNIVERSE_BRIDGES = (
    ddqn_helps_within_empirical_scope,
    ddqn_refuted_when_dormancy_fires,
)
"""The two terminal bridges that close the DDQN study on the
universal paired-delta cells. Run against the universal dataset
to refresh both verdicts; corpus-by-corpus footnotes are no
longer load-bearing."""


__all__ = [
    'DDQN_UNIVERSE_BRIDGES',
    'ddqn_helps_within_empirical_scope',
    'ddqn_refuted_when_dormancy_fires',
]
