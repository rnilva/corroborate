"""Framework-as-instrument: `dowhy.backdoor_ate` recovers the
structural treatment effect under a Z-confounded outcome on a
tabular-MDP-style substrate.

Setup (the canonical confounded SCM):

    Z ∼ N(0, 1)                                 (confounder)
    T | Z ∼ Bernoulli(sigmoid(α·Z))             (T biased by Z)
    V = β_T · T + β_Z · Z + N(0, σ²)            (outcome)

V can be read as the policy-evaluated value of a degenerate
1-step TabularMDP where the per-cell reward depends on (T, Z):
the Bellman backup gives `V[s_0] = E[r] = β_T·T + β_Z·Z` exactly
(the implementation's `policy_evaluation` would return precisely
this in the limit of γ → 0). The substrate-grounding here is
that the closed-form `V` IS the policy evaluation result; the
framework's backdoor_ate must recover the STRUCTURAL `β_T`
despite the confounding.

Closed-form structural ATE = `β_T` (the effect of `do(T=1)` vs
`do(T=0)` after blocking the Z backdoor).

Naive (unadjusted) estimate of `E[V|T=1] − E[V|T=0]`:

    naive = β_T + β_Z · (E[Z|T=1] − E[Z|T=0])
          = β_T + β_Z · ΔZ                    (positive bias when α > 0)

For α=2, β_T=1, β_Z=1: empirically ΔZ ≈ 1.0 → naive ≈ 2.0
(100% inflated) — backdoor adjustment for Z must collapse this
to ≈ 1.0.

DAG: Z → T, Z → V, T → V. DoWhy's `identify_effect` finds the
backdoor adjustment set `{Z}`; `backdoor.linear_regression`
estimator regresses V on (T, Z), recovering β_T as the T
coefficient.

THIS is the framework-as-instrument question: given a confounded
corpus where the structural ATE is known by construction, does
the framework's backdoor adjustment recover the structural value
within sampling SE — and demonstrably WITHOUT the confounding
bias the naive estimator would produce?
"""
from __future__ import annotations

import math
import zlib

import numpy as np

from corroborate.analyses.dowhy import backdoor_ate


def _det_seed(*parts: object) -> int:
    return zlib.adler32(repr(parts).encode()) & 0xFFFF_FFFF


_BETA_T = 1.0       # structural treatment effect (the ATE to recover)
_BETA_Z = 1.0       # confounder strength
_SIGMA = 0.3        # outcome noise
_ALPHA = 2.0        # selection bias strength (T biased by Z)
_N_CELLS = 500


def _generate_confounded_cells() -> list[dict[str, object]]:
    """Per-cell: sample Z, then T|Z, then V = β_T·T + β_Z·Z + ε.
    Cells flow through `RunRow.from_row_dict` via the framework's
    cell adapter — provenance fields at top level + measurement
    paths flat (the persistence shape from CLAUDE.md)."""
    cells: list[dict[str, object]] = []
    for s in range(_N_CELLS):
        rng = np.random.default_rng(seed=_det_seed('backdoor_z', s))
        z = float(rng.standard_normal())
        # Treatment assignment biased by Z via logistic.
        u = float(rng.standard_normal())
        # Use a logit-shift form: T ~ Bernoulli(sigmoid(α·Z + ε_logit)).
        # Equivalently: T = 1 iff α·Z + ε > 0 with ε ~ Logistic(0,1).
        # For simplicity use Gaussian residual + threshold:
        treatment = 1.0 if (_ALPHA * z + u) > 0 else 0.0
        # Outcome: structural β_T·T + β_Z·Z + iid Gaussian noise.
        outcome_v = (
            _BETA_T * treatment
            + _BETA_Z * z
            + _SIGMA * float(rng.standard_normal())
        )
        cells.append({
            'id': f'c{s}',
            'parent_id': None,
            'cycle_id': None,
            'timestamp': 'ts',
            'verdict': 'held',
            'arm_key': 'single',
            'env_name': 'confounded_v',
            'seed': s,
            'confounder_z': z,
            'treatment': treatment,
            'outcome_v': outcome_v,
        })
    return cells


def _naive_difference_in_means(
    cells: list[dict[str, object]],
) -> float:
    """Closed-form-known biased baseline: simple
    `E[V|T=1] − E[V|T=0]` with NO adjustment. Used as a
    structural sanity check that the unadjusted estimate IS
    biased — pinning that `backdoor_ate`'s adjustment is doing
    real work."""
    v_treat = [
        float(c['outcome_v']) for c in cells if c['treatment'] == 1.0
    ]
    v_base = [
        float(c['outcome_v']) for c in cells if c['treatment'] == 0.0
    ]
    return (sum(v_treat) / len(v_treat)) - (sum(v_base) / len(v_base))


# ============ Backdoor adjustment recovers the structural ATE ============

def test_backdoor_ate_recovers_structural_effect_under_confounding() -> None:
    """The framework's `backdoor_ate` with the correct DAG
    (Z → T, Z → V, T → V) should recover β_T = 1.0 within
    sampling SE.

    OLS slope SE on `T` after residualizing on Z scales as
    σ / √(n · Var(T | Z)). Var(T | Z) ≈ 0.16-0.20 for α=2
    Bernoulli(sigmoid(α·Z)) with Z ∼ N(0,1); σ_residual ≈
    `sqrt(σ²)` = 0.3. Thus SE(β̂_T) ≈ 0.3 / √(500 · 0.18) ≈
    0.032. 4·SE bound = 0.128.
    """
    cells = _generate_confounded_cells()
    dag: list[tuple[str, str]] = [
        ('confounder_z', 'treatment'),
        ('confounder_z', 'outcome_v'),
        ('treatment', 'outcome_v'),
    ]
    result = backdoor_ate.fn(
        cells,
        treatment='treatment',
        outcome='outcome_v',
        dag=dag,
    )
    assert result.identified, (
        f'backdoor_ate did not identify the effect; '
        f'estimand_str={result.estimand_str!r}'
    )
    assert result.n_rows == _N_CELLS
    bound = 0.15
    assert abs(result.ate - _BETA_T) < bound, (
        f'ATE = {result.ate:.4f}, structural β_T = {_BETA_T:.4f} '
        f'(4·SE bound ≈ {bound:.4f}). Backdoor adjustment for Z '
        f'must recover the structural treatment effect.'
    )


def test_naive_estimate_is_biased_above_structural_ate() -> None:
    """Negative-control demonstration: WITHOUT adjustment, the
    raw `E[V|T=1] − E[V|T=0]` is biased UPWARD by β_Z · ΔZ
    where ΔZ > 0 due to the α > 0 selection on Z. This is the
    confounding the backdoor adjustment is supposed to remove.

    Pin that:
    - the naive estimate is significantly different from the
      structural β_T (sanity that confounding IS present)
    - the framework's backdoor_ate (with adjustment) is
      substantially closer to β_T than the naive estimate

    A test that PASSED test_backdoor_ate_recovers without
    showing that naive is biased would not have demonstrated
    the framework actually did adjustment — it could have
    coincidentally received unconfounded data. This test
    rules that out.
    """
    cells = _generate_confounded_cells()
    naive = _naive_difference_in_means(cells)
    # With α=2, β_Z=1, ΔZ ≈ 1.0 → naive ≈ 2.0, biased above 1.0.
    # The naive estimate must exceed β_T by a meaningful margin
    # (well outside its own sampling SE ≈ 0.03).
    assert naive > _BETA_T + 0.5, (
        f'naive = {naive:.4f}, expected > {_BETA_T + 0.5:.4f}. '
        f'Without confounding, the naive estimate would equal '
        f'β_T; the inflation IS the confounding bias the '
        f'framework s backdoor adjustment must remove.'
    )
    # And the framework's adjusted ATE should be substantially
    # closer to β_T than the naive estimate. Concretely:
    # |ate_adjusted - β_T| < |naive - β_T| / 3.
    dag: list[tuple[str, str]] = [
        ('confounder_z', 'treatment'),
        ('confounder_z', 'outcome_v'),
        ('treatment', 'outcome_v'),
    ]
    result = backdoor_ate.fn(
        cells,
        treatment='treatment',
        outcome='outcome_v',
        dag=dag,
    )
    naive_bias = abs(naive - _BETA_T)
    adjusted_bias = abs(result.ate - _BETA_T)
    assert adjusted_bias < naive_bias / 3.0, (
        f'adjusted_bias = {adjusted_bias:.4f}, naive_bias = '
        f'{naive_bias:.4f}. Backdoor adjustment should reduce '
        f'the bias by at least 3× — otherwise it isn t doing '
        f'meaningful adjustment.'
    )


def test_backdoor_n_rows_matches_input() -> None:
    """The framework's `BackdoorResult.n_rows` should equal the
    number of cells projected to the DataFrame. Pin against a
    regression that silently dropped rows during projection
    (which would inflate ATE SE and pass tests by widening,
    not by computing correctly).
    """
    cells = _generate_confounded_cells()
    dag: list[tuple[str, str]] = [
        ('confounder_z', 'treatment'),
        ('confounder_z', 'outcome_v'),
        ('treatment', 'outcome_v'),
    ]
    result = backdoor_ate.fn(
        cells,
        treatment='treatment',
        outcome='outcome_v',
        dag=dag,
    )
    assert result.n_rows == _N_CELLS, (
        f'n_rows = {result.n_rows}, expected {_N_CELLS}'
    )
