"""Framework-as-instrument: `tautology_audit` flags
HP-shadow / outcome-jaccard / clean mediators correctly on a
substrate where the structural answer is known by construction.

Three mediator candidates are synthesized over a γ-grid corpus
(γ acts as the HP axis); each is engineered to fail (or pass)
exactly the audit check the framework should detect:

1. **`mediator.gamma_shadow`**: noisy version of the γ HP itself
   (`γ + N(0, σ_g²)`). Has high MARGINAL correlation with the
   outcome (γ drives the across-env mean), but within-stratum
   (fixed γ) is pure noise — uncorrelated with within-env
   outcome variation. The substrate-side prediction:
   - `flagged_outcome` = False (reads `'gamma'` ⫫ outcome reads)
   - `flagged_hp = ()`             (R² with γ is bounded; σ_g
     chosen so R² < 0.95 threshold)
   - `flagged_no_residual_signal` = True (within-stratum ρ ≈ 0)

2. **`mediator.outcome_shadow`**: the outcome itself plus tiny
   noise, with `reads=('mc_return',)` — the SAME trace key the
   outcome aggregates from. The substrate-side prediction:
   - `flagged_outcome` = True (jaccard with outcome_reads = 1.0)

3. **`mediator.clean`**: a real per-trajectory mediator
   (`x_0(seed) + N(0, σ_c²)`) that depends on within-env stochastic
   structure, NOT on γ. Predicted to pass all three checks:
   - `flagged_outcome` = False (reads disjoint)
   - `flagged_hp = ()`             (R² with γ ≈ 0)
   - `flagged_no_residual_signal` = False (within-stratum ρ > 0)

The framework's `tautology_audit` should return
`clean_names == ('mediator.clean',)` exactly. THIS is the
framework-as-instrument question: given mediators where the
structural verdict is known by construction, does the
three-check audit logic flag the right candidates?

A regression that swapped any check, conflated outcome-shadow
with HP-shadow, or that mishandled the stratified-ρ pooling
would breach the per-check verdict on at least one mediator.
"""
from __future__ import annotations

import math
import zlib

from corroborate.analyses.tautology_audit import tautology_audit


def _det_seed(*parts: object) -> int:
    return zlib.adler32(repr(parts).encode()) & 0xFFFF_FFFF


_GAMMA_GRID: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9)
_N_SEEDS_PER_ENV = 30
_SIGMA_X0 = 1.0
_SIGMA_OUTCOME_NOISE = 0.1
# Wide enough to drop R²(γ_shadow on γ) well below 0.95 while
# keeping marginal corr(γ_shadow, outcome) high. With γ-variance
# ≈ 0.02 over the [0.5, 0.9] grid, a per-seed noise of 0.5
# gives R² ≈ 0.07 — safely below the 0.95 HP-deterministic
# threshold but high enough to dominate within-stratum.
_SIGMA_GAMMA_SHADOW = 0.5
_SIGMA_CLEAN = 0.3
# Outcome shadow tracks the outcome closely.
_SIGMA_OUTCOME_SHADOW = 0.05


def _outcome_value(gamma: float, x_0: float, *, noise: float) -> float:
    """Substrate outcome: a γ-and-x_0-driven quantity. The
    base level scales with γ (so γ drives across-env mean)
    and the within-env variation comes from the per-seed x_0.
    """
    return 5.0 * gamma + x_0 + noise


def _generate_audit_cells() -> list[dict[str, object]]:
    cells: list[dict[str, object]] = []
    import numpy as np
    for gamma in _GAMMA_GRID:
        env_name = f'env_gamma_{gamma:g}'
        for s in range(_N_SEEDS_PER_ENV):
            rng_x0 = np.random.default_rng(seed=_det_seed('aud_x0', gamma, s))
            rng_o = np.random.default_rng(seed=_det_seed('aud_o', gamma, s))
            rng_g = np.random.default_rng(seed=_det_seed('aud_g', gamma, s))
            rng_c = np.random.default_rng(seed=_det_seed('aud_c', gamma, s))
            rng_os = np.random.default_rng(seed=_det_seed('aud_os', gamma, s))

            x_0 = float(_SIGMA_X0 * rng_x0.standard_normal())
            outcome = _outcome_value(
                gamma=gamma, x_0=x_0,
                noise=float(_SIGMA_OUTCOME_NOISE * rng_o.standard_normal()),
            )
            gamma_shadow_val = float(
                gamma + _SIGMA_GAMMA_SHADOW * rng_g.standard_normal(),
            )
            clean_val = float(x_0 + _SIGMA_CLEAN * rng_c.standard_normal())
            outcome_shadow_val = float(
                outcome + _SIGMA_OUTCOME_SHADOW * rng_os.standard_normal(),
            )

            # RunRow.from_row_dict expects a FLAT dict — typed
            # provenance fields at top level + each measurement
            # path as its own top-level key (no nested dict).
            cells.append({
                'id': f'g{gamma:g}_s{s}',
                'parent_id': None,
                'cycle_id': None,
                'timestamp': 'ts',
                'verdict': 'held',     # any value; not consumed by audit
                'arm_key': 'single',
                'env_name': env_name,
                'seed': s,
                'gamma': gamma,
                'outcome.v_terminal': outcome,
                'mediator.gamma_shadow': gamma_shadow_val,
                'mediator.clean': clean_val,
                'mediator.outcome_shadow': outcome_shadow_val,
            })
    return cells


_MEDIATORS: tuple[dict[str, object], ...] = (
    # gamma_shadow reads only the HP value `gamma` — disjoint
    # from outcome reads → no jaccard flag.
    {'name': 'mediator.gamma_shadow', 'reads': ('gamma',)},
    # clean reads a substrate-derived per-trajectory quantity —
    # disjoint from outcome reads.
    {'name': 'mediator.clean', 'reads': ('x_0_per_episode',)},
    # outcome_shadow reads the SAME trace column the outcome
    # aggregates from → jaccard = 1.0 → outcome-flagged.
    {'name': 'mediator.outcome_shadow', 'reads': ('mc_return',)},
)

_OUTCOME_READS = ('mc_return',)


def _audit_result():
    return tautology_audit.fn(
        _generate_audit_cells(),
        measurables=_MEDIATORS,
        outcome_path='outcome.v_terminal',
        outcome_reads=_OUTCOME_READS,
        hp_axes=('gamma',),
        hp_stratum_axis='gamma',
        arm_filter='single',
    )


# ============ outcome-shadow detection ============

def test_audit_flags_outcome_shadow_via_jaccard() -> None:
    """`mediator.outcome_shadow` reads `'mc_return'` — the same
    trace key the outcome aggregates from. Jaccard = 1.0 ≥ 0.5
    threshold → `flagged_outcome=True`. Pin the structural
    reads-jaccard check.
    """
    result = _audit_result()
    rep = result.by_name('mediator.outcome_shadow')
    assert rep is not None
    assert rep.flagged_outcome is True, (
        f'outcome_shadow.flagged_outcome = {rep.flagged_outcome}, '
        f'expected True (jaccard = {rep.outcome_jaccard:.4f}, '
        f'threshold 0.5).'
    )
    # Structural jaccard = 1.0 (identical reads sets).
    assert rep.outcome_jaccard == 1.0, (
        f'outcome_jaccard = {rep.outcome_jaccard:.4f}, expected '
        f'1.0 (identical reads).'
    )


def test_audit_does_not_flag_clean_via_jaccard() -> None:
    """`mediator.clean` reads disjoint trace keys → jaccard = 0.
    Pin the structural negative case for the jaccard check.
    """
    result = _audit_result()
    rep = result.by_name('mediator.clean')
    assert rep is not None
    assert rep.flagged_outcome is False
    assert rep.outcome_jaccard == 0.0


# ============ HP-shadow detection (stratified ρ) ============

def test_audit_flags_gamma_shadow_via_stratified_rho() -> None:
    """`mediator.gamma_shadow` is `γ + N(0, σ²)` — high marginal
    correlation with the outcome (γ drives the across-env mean
    of the outcome) but within-stratum (fixed γ) is pure noise.

    Stratified Spearman ρ(γ_shadow, outcome | env=γ) pools
    per-stratum ρ. Within each stratum, γ_shadow is constant +
    iid noise; the outcome varies via x_0(seed). They're
    independent within stratum → stratified ρ ≈ 0 → with α=0.05
    and the default ρ-threshold 0.1, `flagged_no_residual_signal`
    fires.

    A regression that conflated marginal-vs-stratified
    correlation, or that mis-pooled the per-stratum ρ via
    Fisher z, would silently let γ_shadow pass.
    """
    result = _audit_result()
    rep = result.by_name('mediator.gamma_shadow')
    assert rep is not None
    assert rep.flagged_no_residual_signal is True, (
        f'gamma_shadow.flagged_no_residual_signal = '
        f'{rep.flagged_no_residual_signal}, expected True. '
        f'within-stratum ρ = {rep.outcome_stratified_rho:.4f}, '
        f'p = {rep.outcome_stratified_p:.4f}.'
    )
    # |ρ| should be well below the 0.1 threshold (within-stratum
    # signal is pure noise).
    assert abs(rep.outcome_stratified_rho) < 0.1


def test_audit_does_not_flag_clean_via_stratified_rho() -> None:
    """`mediator.clean = x_0 + noise` and outcome both depend on
    x_0 within stratum → stratified ρ should be substantially
    positive. Pin that the clean mediator is NOT HP-shadow-flagged.
    """
    result = _audit_result()
    rep = result.by_name('mediator.clean')
    assert rep is not None
    assert rep.flagged_no_residual_signal is False
    # Within-stratum ρ should be well above the threshold.
    assert rep.outcome_stratified_rho > 0.5, (
        f'clean.outcome_stratified_rho = '
        f'{rep.outcome_stratified_rho:.4f}, expected > 0.5 '
        f'(both clean and outcome depend on x_0 within env).'
    )


# ============ HP-deterministic check (R²) ============

def test_audit_does_not_flag_gamma_shadow_via_r_squared() -> None:
    """`gamma_shadow = γ + N(0, 0.5)`. With γ-grid variance ≈
    0.02 over [0.5, 0.9] and per-seed noise σ = 0.5, R²(γ_shadow
    on γ) ≈ 0.02 / (0.02 + 0.25) ≈ 0.07. Well below the 0.95
    HP-deterministic threshold.

    This is the structural negative for the HP-R² check on
    γ_shadow — the mediator IS γ-shadowed but in a way that the
    R² test does NOT detect (the noise is too large). The
    stratified-ρ check IS the load-bearing one for HP-shadow
    detection here; pin that R² doesn't fire (lest someone
    later inflate the threshold to 0.05 and mis-flag everything).
    """
    result = _audit_result()
    rep = result.by_name('mediator.gamma_shadow')
    assert rep is not None
    assert rep.flagged_hp == (), (
        f'gamma_shadow.flagged_hp = {rep.flagged_hp!r}, '
        f'expected (). HP-R² on γ should be too low to flag '
        f'(R² ≈ 0.07 vs threshold 0.95).'
    )


def test_audit_does_not_flag_clean_via_r_squared() -> None:
    """`mediator.clean = x_0 + noise` is structurally independent
    of γ. R²(clean on γ) ≈ 0. Pin that the HP-R² check passes
    on a clean mediator.
    """
    result = _audit_result()
    rep = result.by_name('mediator.clean')
    assert rep is not None
    assert rep.flagged_hp == ()


# ============ Aggregate clean-names ============

def test_audit_clean_names_returns_only_the_clean_mediator() -> None:
    """The aggregate `is_clean_names` surface — bridges typically
    call this — should return exactly `('mediator.clean',)`.
    Three of the four checks fire on at least one of the two
    flagged mediators; only `clean` survives all three.

    A regression that conflated `is_clean` with any-flag-set, or
    that returned ALL passes when one fired, would breach.
    """
    result = _audit_result()
    assert result.clean_names == ('mediator.clean',), (
        f'clean_names = {result.clean_names!r}, expected '
        f"('mediator.clean',). Three audit checks fire on the "
        f'two engineered-tautological mediators; only the '
        f'structurally-clean mediator passes all three.'
    )
