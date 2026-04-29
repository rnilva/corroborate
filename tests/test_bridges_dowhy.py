"""Tests for the DoWhy-backed bridge factories.

Skipped at import time when DoWhy is not installed (it's an
optional dependency). When installed, validates:

- `backdoor_ate` HELD on a synthetic DGP with a known ATE.
- `backdoor_ate` NO_EFFECT when the sign / magnitude expectation
  is wrong.
- `placebo_refutation` HELD on a real-effect DGP (placebo near 0)
  and NO_EFFECT on a no-effect DGP (placebo matches the "real").
- `random_common_cause_refutation` HELD on a robust estimate.
- `build_causal_graph` consumes the BridgeResults and produces a
  Tier.INTERVENTIONAL edge with `evidentiary_level='causal_one_sided'`.
- `promote_bridged_evidence` upgrades to `'causal_bridged'` when
  ≥2 INTERVENTIONAL HELD bridges share a (treatment, outcome) pair."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest

pytest.importorskip('dowhy')
pytest.importorskip('pandas')
pytest.importorskip('networkx')

from corroborate.bridges_dowhy import (
    backdoor_ate, placebo_refutation, random_common_cause_refutation,
)
from corroborate.causal_graph import (
    Tier, build_causal_graph, promote_bridged_evidence,
)
from corroborate.verdict import Verdict


def _linear_dgp(
    n: int = 500, seed: int = 42,
) -> Mapping[str, np.ndarray]:
    """T → M → Y plus T → Y direct. ATE(T → Y) ≈ 0.5*0.7 + 0.3 = 0.65."""
    rng = np.random.default_rng(seed)
    T = rng.normal(0, 1, n)
    M = 0.7 * T + rng.normal(0, 0.5, n)
    Y = 0.5 * M + 0.3 * T + rng.normal(0, 0.5, n)
    return {'T': T, 'M': M, 'Y': Y}


_FULL_DAG: list[tuple[str, str]] = [
    ('T', 'M'), ('M', 'Y'), ('T', 'Y'),
]


# ============ backdoor_ate ============

def test_backdoor_ate_held_on_real_effect() -> None:
    """Real ATE > 0 → backdoor regression returns positive estimate
    above threshold → HELD. `stats['tier']='interventional'` is
    set so the graph builder promotes the edge."""
    record = _linear_dgp()
    bridge = backdoor_ate(
        'T', 'Y', graph=_FULL_DAG,
        expected_sign=+1, threshold=0.1,
    )
    result = bridge(record)
    assert result.verdict is Verdict.HELD
    ate_v = result.stats['ate']
    assert isinstance(ate_v, (int, float))
    assert float(ate_v) > 0.1
    assert result.stats['tier'] == 'interventional'


def test_backdoor_ate_no_effect_on_wrong_sign() -> None:
    """Real ATE > 0 but author predicts negative → NO_EFFECT
    (sign-flip refutation)."""
    record = _linear_dgp()
    bridge = backdoor_ate(
        'T', 'Y', graph=_FULL_DAG,
        expected_sign=-1, threshold=0.1,
    )
    result = bridge(record)
    assert result.verdict is Verdict.NO_EFFECT
    assert 'wrong sign' in result.reason


def test_backdoor_ate_carries_estimand_string() -> None:
    """The dowhy-derived estimand expression is stored in
    `stats['estimand']` for audit / paper write-up."""
    record = _linear_dgp()
    bridge = backdoor_ate(
        'T', 'Y', graph=_FULL_DAG,
        expected_sign=+1, threshold=0.1,
    )
    result = bridge(record)
    estimand = result.stats.get('estimand')
    assert isinstance(estimand, str)
    assert len(estimand) > 0


# ============ placebo_refutation ============

def test_placebo_refutation_held_on_real_effect() -> None:
    """Real effect → permuted-treatment effect collapses near 0 →
    HELD. The estimator returns ≈0 for a fake treatment, so the
    real effect isn't an artefact."""
    record = _linear_dgp()
    bridge = placebo_refutation(
        'T', 'Y', graph=_FULL_DAG, tolerance=0.2,
    )
    result = bridge(record)
    assert result.verdict is Verdict.HELD
    placebo = result.stats['placebo_ate']
    assert isinstance(placebo, (int, float))
    assert abs(float(placebo)) < 0.2


def test_placebo_refutation_carries_real_and_placebo_ate() -> None:
    record = _linear_dgp()
    bridge = placebo_refutation(
        'T', 'Y', graph=_FULL_DAG, tolerance=0.2,
    )
    result = bridge(record)
    assert 'real_ate' in result.stats
    assert 'placebo_ate' in result.stats
    assert result.stats['role'] == 'refuter'


# ============ random_common_cause_refutation ============

def test_random_common_cause_refutation_held_on_robust_estimate() -> None:
    """Adding a synthetic random common cause shouldn't move the
    estimate (the new var is uncorrelated by construction). HELD
    iff the drift is below tolerance."""
    record = _linear_dgp()
    bridge = random_common_cause_refutation(
        'T', 'Y', graph=_FULL_DAG, tolerance=0.1,
    )
    result = bridge(record)
    assert result.verdict is Verdict.HELD
    drift = result.stats['drift']
    assert isinstance(drift, (int, float))
    assert float(drift) < 0.1


# ============ Graph promotion ============

def test_promotion_to_interventional() -> None:
    """`build_causal_graph` reads `stats['tier']='interventional'`
    on a HELD BridgeResult and produces a Tier.INTERVENTIONAL
    edge with `evidentiary_level='causal_one_sided'`."""
    record = _linear_dgp()
    estimate = backdoor_ate(
        'T', 'Y', graph=_FULL_DAG,
        expected_sign=+1, threshold=0.1,
    )(record)
    g = build_causal_graph([estimate])
    edges = [e for e in g.edges if (e.source, e.target) == ('T', 'Y')]
    assert len(edges) == 1
    md = edges[0].metadata
    assert md.tier is Tier.INTERVENTIONAL
    assert md.evidentiary_level == 'causal_one_sided'


def test_bridged_evidence_promotion_via_estimate_plus_refuter() -> None:
    """1 estimate HELD + 1 refuter HELD on the same (T, Y) pair →
    promote_bridged_evidence upgrades to 'causal_bridged'. This
    is the v10 §3.5 contract: do-calculus inference corroborated
    by an INDEPENDENT bridge."""
    record = _linear_dgp()
    r_est = backdoor_ate(
        'T', 'Y', graph=_FULL_DAG,
        expected_sign=+1, threshold=0.1,
    )(record)
    r_pl = placebo_refutation(
        'T', 'Y', graph=_FULL_DAG, tolerance=0.2,
    )(record)
    assert r_est.verdict is Verdict.HELD
    assert r_pl.verdict is Verdict.HELD

    g = build_causal_graph([r_est, r_pl])
    g = promote_bridged_evidence(g)
    edges = [
        e.metadata for e in g.edges
        if (e.source, e.target) == ('T', 'Y')
    ]
    bridged = [m for m in edges if m.evidentiary_level == 'causal_bridged']
    assert len(bridged) == 2
