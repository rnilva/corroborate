"""Robustness probe: `backdoor_ate` under DAG misspecification.

The framework's `backdoor_ate` requires the user to author the
DAG that DoWhy uses to compute the adjustment set. **The framework
has no way to verify the authored DAG is correct.** A user who
misspecifies the DAG gets a silently-biased ATE estimate with
`identified=True`, which downstream bridges interpret as a
trustworthy effect.

This probe quantifies the bias under three categories of DAG
authoring error, against the LG-SCM implementation where the structural
ATE = β_xz · β_zy = 0.75:

1. **Correct DAG** (`X → Y`): baseline. ATE recovered within
   sampling (≈ 0.75).
2. **Mediator-treated-as-confounder** (`Z → X, Z → Y, X → Y`): the
   user mistakes the mediator Z for a confounder. DoWhy adjusts
   for Z, removing the X variance that's transmitted through Z.
   Estimated ATE collapses to the DIRECT-X→Y effect (which is
   structurally 0 in our SCM). **Silent wrong verdict** with
   `identified=True`.
3. **Mediator-as-collider** (`X → Z, Y → Z, X → Y`): the user
   wrongly thinks Y causes Z (reversed mediator edge). DoWhy
   does NOT condition on Z (it's a collider, not on the
   adjustment set), so the estimate is unaffected.
4. **Reversed treatment/outcome** (`Y → X`): DoWhy correctly
   reports unidentified. ATE = NaN, identified=False.

Substrate-author guidance: DoWhy's `identified=True` + a
plausible-looking ATE is NOT a guarantee that the DAG is right.
The framework cannot verify DAG correctness; implementation author
must validate the DAG against domain knowledge BEFORE trusting
the ATE. The silent-wrong-verdict case (#2) is the most
dangerous because it produces a confident-looking 0 estimate.

Empirical numbers anchored to deterministic seed=0 across the
LG-SCM corpus (5 envs × 30 cells × 200 steps). Reproducible
across processes.
"""
from __future__ import annotations

import math
import sys
from collections.abc import Mapping

import corroborate.analyses  # noqa: F401  # pyright: ignore[reportUnusedImport]

from corroborate.analyses.dowhy import backdoor_ate
from corroborate.data import cells_to_dataframe

# Reuse the implementation from the existing dowhy test.
sys.path.insert(0, 'tests/analytic/lg_scm')
from tests.analytic.lg_scm.test_dowhy import (  # noqa: E402
    _EXPECTED_ATE, _build_observational_corpus,
)


_CELLS: list[Mapping[str, object]] = list(_build_observational_corpus())


def test_correct_dag_recovers_structural_ate() -> None:
    """**Baseline**: with the correct DAG `[(x_mean, y_mean)]`,
    DoWhy's backdoor adjustment recovers the structural ATE
    within 5% relative error.

    Pin: |ate - 0.75| / 0.75 < 0.05. A regression in the SCM
    or DoWhy dispatch would fail this; serves as harness validation."""
    result = backdoor_ate.fn(
        cells_to_dataframe(_CELLS), treatment='x_mean', outcome='y_mean',
        dag=[('x_mean', 'y_mean')],
    )
    assert result.identified
    rel_err = abs(result.ate - _EXPECTED_ATE) / _EXPECTED_ATE
    assert rel_err < 0.05, (
        f'baseline ATE recovery: ate = {result.ate:.4f}, '
        f'expected {_EXPECTED_ATE:.4f}, rel_err = {rel_err:.4f}.'
    )


def test_mediator_treated_as_confounder_silently_zeroes_ate() -> None:
    """**The dangerous case**: implementation author treats Z (a
    mediator) as a confounder. DAG `Z → X, Z → Y, X → Y` tells
    DoWhy to adjust for Z when computing the X → Y effect.

    Adjustment removes the X variance transmitted through Z (which
    is the WHOLE structural pathway in our SCM); only the DIRECT
    X → Y effect remains, and it's structurally 0. DoWhy reports
    `ate ≈ 0` with `identified=True`.

    Empirical: ate ≈ 0.012 (truth = 0.75; bias ≈ -0.74). A
    downstream bridge body that treats `ate < 0.1` as NO_EFFECT
    will silently miss the real total effect.

    Pin: ate < 0.05 (clearly close to zero) AND identified=True
    (the silent-wrong-verdict mode). Implementation guidance: NEVER
    include the mediator in the adjustment set when computing
    total effect."""
    result = backdoor_ate.fn(
        cells_to_dataframe(_CELLS), treatment='x_mean', outcome='y_mean',
        dag=[
            ('z_mean', 'x_mean'),    # mediator wrongly treated as cause of X
            ('z_mean', 'y_mean'),    # mediator wrongly treated as cause of Y
            ('x_mean', 'y_mean'),    # direct path (effectively 0 here)
        ],
    )
    assert result.identified, (
        f'mediator-as-confounder DAG: identified={result.identified}; '
        f'expected True (the silent-wrong-verdict pathway requires '
        f'identified=True; otherwise it would surface as POWER_INSUFFICIENT).'
    )
    assert abs(result.ate) < 0.05, (
        f'mediator-as-confounder DAG: ate = {result.ate:.4f}, '
        f'expected < 0.05 (truth = {_EXPECTED_ATE:.4f}; '
        f'adjustment for the mediator zeroes the total effect). '
        f'A bias > 0.05 would mean DoWhy didn\'t fully adjust — '
        f'still reporting a confidently wrong answer.'
    )


def test_reversed_treatment_outcome_returns_unidentified() -> None:
    """**Safe case**: DAG with the wrong direction (`Y → X`).
    DoWhy correctly reports unidentified.

    The framework's NaN-fallback path fires, NaN propagates
    downstream, the bridge gets a clear "couldn't compute"
    signal. NOT a silent-wrong-verdict situation."""
    result = backdoor_ate.fn(
        cells_to_dataframe(_CELLS), treatment='x_mean', outcome='y_mean',
        dag=[('y_mean', 'x_mean')],
    )
    assert not result.identified
    assert math.isnan(result.ate)


def test_mediator_as_collider_unaffected_by_misspecification() -> None:
    """**Negative control**: if the user writes the DAG with a
    REVERSED mediator edge (`X → Z, Y → Z` instead of the correct
    `X → Z → Y`), Z becomes a collider and DoWhy does NOT
    condition on it. The X → Y direct edge in the DAG is what
    DoWhy uses; the estimate is unaffected by the misspecified
    Z edges.

    Pin: ate ≈ 0.75 (within sampling). This shows that not all
    DAG errors are catastrophic — getting the colliders/mediators
    right is what matters. The asymmetry reinforces the
    implementation-author guidance: DAG correctness check should focus
    on identifying mediators."""
    result = backdoor_ate.fn(
        cells_to_dataframe(_CELLS), treatment='x_mean', outcome='y_mean',
        dag=[
            ('x_mean', 'z_mean'),    # correct: X causes Z
            ('y_mean', 'z_mean'),    # WRONG: Y doesn't cause Z, but
                                      # this makes Z a collider not
                                      # a mediator
            ('x_mean', 'y_mean'),    # direct edge — DoWhy uses this
        ],
    )
    assert result.identified
    rel_err = abs(result.ate - _EXPECTED_ATE) / _EXPECTED_ATE
    assert rel_err < 0.05, (
        f'mediator-as-collider DAG: ate = {result.ate:.4f}, '
        f'expected ≈ {_EXPECTED_ATE:.4f}. Some DAG errors are '
        f'safe (collider-direction reversal); the mediator-as-'
        f'confounder error in the previous test IS dangerous. '
        f'The asymmetry is the substrate-author guidance.'
    )


def test_dag_misspecification_summary_table() -> None:
    """Self-documenting test: emits a stderr summary of the
    bias map for the four DAG variants. Useful to implementation
    authors as a quick reference; the test asserts only that
    each DoWhy call returned a result (no crash) and the
    summary printed.

    The point IS the printout (sample run):

        DAG variant                    ate     identified
        correct                        0.752   True
        mediator-as-confounder         0.012   True   ← SILENT WRONG
        mediator-as-collider           0.752   True
        reversed treatment/outcome     nan     False

    Implementation authors who run the test suite see this table and
    know which DAG errors are dangerous."""
    variants = [
        ('correct', [('x_mean', 'y_mean')]),
        (
            'mediator-as-confounder',
            [('z_mean', 'x_mean'), ('z_mean', 'y_mean'),
             ('x_mean', 'y_mean')],
        ),
        (
            'mediator-as-collider',
            [('x_mean', 'z_mean'), ('y_mean', 'z_mean'),
             ('x_mean', 'y_mean')],
        ),
        ('reversed', [('y_mean', 'x_mean')]),
    ]
    sys.stderr.write('\n=== DoWhy DAG misspecification bias map ===\n')
    sys.stderr.write(f'{"variant":<32}{"ate":>10}{"identified":>14}\n')
    for name, dag in variants:
        result = backdoor_ate.fn(
            cells_to_dataframe(_CELLS), treatment='x_mean', outcome='y_mean', dag=dag,
        )
        ate_str = (
            f'{result.ate:>10.4f}' if result.identified
            else f'{"nan":>10}'
        )
        sys.stderr.write(
            f'{name:<32}{ate_str}{str(result.identified):>14}\n',
        )
    # The assertion is a sanity check that all four ran; the
    # value of the test is the table.
    assert True
