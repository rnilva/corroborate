"""Tests for `Scope` + `build_scope`.

Validates:
- `Scope` is frozen, slim (6 fields), gap-grounded by construction.
- `is_in_scope` predicate routes correctly across discovery
  vs committed regimes.
- `build_scope` reads per-cell gap from baseline runs at
  `gap_path`, aggregates per-stratum mean, runs meta_regression.
- `log_scale=True` regresses on log10 of gap magnitude.
- Strata with NaN gap or insufficient g/se are dropped.
- Missing target raises loudly.
- Discovery vs committed mode round-trip on `Scope.threshold`."""
from __future__ import annotations

import math
from collections.abc import Mapping

import pytest

from corroborate.causal_graph import Direction, Tier
from corroborate.claim import claim
from corroborate.claim_bridge import Bridge as ClaimBridge
from corroborate.graph import Graph
from corroborate.hypothesis import Hypothesis, PredictedDirection
from corroborate.hypothesis_verdict import hypothesis_subgraph_verdict
from corroborate.intervention import DoEffect, Intervention
from corroborate.meta_regression import MetaRegressionResult
from corroborate.schema import RunRow
from corroborate.scope import Scope, build_scope
from corroborate.verdict import Verdict


@claim
def _stub_arm(x: int) -> int:
    return x


_TREATMENT_ARMS = (Intervention(slot_path='stub', replacement=_stub_arm),)
_TEST_DO = DoEffect(treatment_arm='treat', baseline_arm='baseline')


def _intervention_edge(
    target: str, predicted_direction: PredictedDirection,
) -> ClaimBridge:
    return ClaimBridge(
        name=f'do->{target}',
        source=_TEST_DO, target=target,
        tier=Tier.INTERVENTIONAL, direction=Direction.DIRECT,
        predicted_direction=predicted_direction,
    )


def _coupling_edge(
    source: str, target: str, predicted_direction: PredictedDirection,
) -> ClaimBridge:
    return ClaimBridge(
        name=f'{source}->{target}',
        source=source, target=target,
        tier=Tier.ASSOCIATIONAL, direction=Direction.DIRECT,
        predicted_direction=predicted_direction,
    )


def _run(
    cell_id: str, *, env: str, seed: int,
    arm_key: str, mech: float, outcome_v: float, gap: float,
) -> RunRow:
    return RunRow(
        id=cell_id, parent_id=None, cycle_id=None,
        timestamp='2026-04-29T00:00:00Z',
        verdict=Verdict.HELD,
        arm_key=arm_key,
        measurements={
            'env_name': env, 'seed': seed,
            'mechanism.q': mech,
            'outcome.r': outcome_v,
            'jensen_gap': gap,
            'intervention_name': (
                'treat' if arm_key != 'baseline' else 'baseline'
            ),
        },
    )


def _three_edge_hypothesis() -> Hypothesis[Mapping[str, object]]:
    return Hypothesis(
        name='ddqn_test',
        intervention={},
        intervention_arms=_TREATMENT_ARMS,
        edges=(
            _intervention_edge('mechanism.q', 'a_lt_b'),
            _intervention_edge('outcome.r', 'a_gt_b'),
            _coupling_edge('mechanism.q', 'outcome.r', 'a_gt_b'),
        ),
    )


def _build_corpus(
    envs: tuple[str, ...], seeds_per_env: int = 8,
) -> tuple[list[RunRow], list[RunRow]]:
    """Synthetic corpus with env-varying mechanism + outcome and
    per-(env, seed) noise. Each cell carries a `jensen_gap`
    value: baseline cells have a per-env mean that varies with i,
    treatment cells have a smaller (DDQN reduces the gap)."""
    import random
    rng = random.Random(0)
    treatment: list[RunRow] = []
    baseline: list[RunRow] = []
    treatment_arm_key = (
        Hypothesis(
            name='_', intervention={}, intervention_arms=_TREATMENT_ARMS,
        ).arm_key()
    )
    for i, env in enumerate(envs):
        env_baseline_gap = 1.0 + i * 0.5  # baseline gap per env
        for seed in range(seeds_per_env):
            base_mech = 1.0 + i * 0.3 + seed * 0.01
            base_outcome = 0.5 + seed * 0.01
            mech_reduction = 0.5 - i * 0.05
            outcome_lift = 0.4 - i * 0.05
            t_noise = rng.gauss(0.0, 0.05)
            b_noise = rng.gauss(0.0, 0.05)
            treatment.append(_run(
                f't{env}{seed}', env=env, seed=seed,
                arm_key=treatment_arm_key,
                mech=base_mech - mech_reduction + t_noise,
                outcome_v=base_outcome + outcome_lift + t_noise,
                gap=env_baseline_gap * 0.5 + t_noise,
            ))
            baseline.append(_run(
                f'b{env}{seed}', env=env, seed=seed,
                arm_key='baseline',
                mech=base_mech + b_noise,
                outcome_v=base_outcome + b_noise,
                gap=env_baseline_gap + b_noise,
            ))
    return treatment, baseline


# ============ Scope dataclass ============

def test_scope_is_frozen_with_expected_fields() -> None:
    """`Scope` is a 6-field frozen dataclass — gap_name +
    threshold are the gap-grounding additions over a generic
    cleavage record."""
    fields = Scope.__dataclass_fields__
    assert set(fields.keys()) == {
        'hypothesis_name', 'gap_name', 'cleavage', 'chain',
        'alpha', 'threshold',
    }


def test_is_in_scope_discovery_mode_always_true_for_finite() -> None:
    """In discovery mode (`threshold=None`) the predicate is
    True for any finite gap — no commitment yet."""
    scope = Scope(
        hypothesis_name='_', gap_name='gap',
        cleavage=MetaRegressionResult(
            n_strata=0, intercept=0.0,
            coefficients=(), r_squared=0.0,
        ),
        chain=Graph(),
        alpha=0.05, threshold=None,
    )
    assert scope.is_in_scope(0.0) is True
    assert scope.is_in_scope(1e9) is True
    assert scope.is_in_scope(float('nan')) is False


def test_is_in_scope_committed_mode_compares_to_threshold() -> None:
    """In committed mode the predicate is `gap_value <= threshold`."""
    scope = Scope(
        hypothesis_name='_', gap_name='gap',
        cleavage=MetaRegressionResult(
            n_strata=0, intercept=0.0,
            coefficients=(), r_squared=0.0,
        ),
        chain=Graph(),
        alpha=0.05, threshold=1.0,
    )
    assert scope.is_in_scope(0.5) is True
    assert scope.is_in_scope(1.0) is True
    assert scope.is_in_scope(1.5) is False
    assert scope.is_in_scope(float('nan')) is False


# ============ build_scope ============

def test_build_scope_target_outcome() -> None:
    """`build_scope` with `target='outcome.r'`: cleavage carries
    the per-env outcome g regressed on the per-env baseline gap
    aggregated from `jensen_gap`."""
    envs = ('A', 'B', 'C', 'D', 'E', 'F')
    treatment, baseline = _build_corpus(envs)
    h = _three_edge_hypothesis()
    v = hypothesis_subgraph_verdict(
        h, treatment, baseline,
        pair_by=('seed',), group_by='env_name',
    )

    scope = build_scope(
        v, baseline,
        gap_path='jensen_gap',
        gap_name='jensen_overestimation_gap',
        target='outcome.r',
    )
    assert isinstance(scope, Scope)
    assert scope.hypothesis_name == 'ddqn_test'
    assert scope.gap_name == 'jensen_overestimation_gap'
    assert isinstance(scope.cleavage, MetaRegressionResult)
    assert isinstance(scope.chain, Graph)
    assert scope.alpha == 0.05
    assert scope.threshold is None
    assert scope.cleavage.n_strata == len(envs)
    # Single covariate, named after gap_name.
    assert len(scope.cleavage.coefficients) == 1
    assert scope.cleavage.coefficients[0].name == 'jensen_overestimation_gap'


def test_build_scope_log_scale_uses_log_prefix_name() -> None:
    """`log_scale=True` regresses on log10(gap); covariate name
    becomes `log_<gap_name>`."""
    envs = ('A', 'B', 'C', 'D', 'E', 'F')
    treatment, baseline = _build_corpus(envs)
    h = _three_edge_hypothesis()
    v = hypothesis_subgraph_verdict(
        h, treatment, baseline,
        pair_by=('seed',), group_by='env_name',
    )
    scope = build_scope(
        v, baseline,
        gap_path='jensen_gap',
        gap_name='jensen_overestimation_gap',
        target='outcome.r',
        log_scale=True,
    )
    assert scope.cleavage.coefficients[0].name == (
        'log_jensen_overestimation_gap'
    )


def test_build_scope_threshold_metadata_round_trips() -> None:
    """Passing a threshold stores it on the Scope (commitment
    metadata; doesn't affect the regression)."""
    envs = ('A', 'B', 'C', 'D', 'E', 'F')
    treatment, baseline = _build_corpus(envs)
    h = _three_edge_hypothesis()
    v = hypothesis_subgraph_verdict(
        h, treatment, baseline,
        pair_by=('seed',), group_by='env_name',
    )
    scope = build_scope(
        v, baseline,
        gap_path='jensen_gap',
        gap_name='jensen_overestimation_gap',
        target='outcome.r',
        threshold=2.0,
    )
    assert scope.threshold == 2.0
    assert scope.is_in_scope(1.0) is True
    assert scope.is_in_scope(3.0) is False


def test_build_scope_drops_strata_with_nan_gap() -> None:
    """A stratum where every baseline cell has NaN gap is dropped
    — meta_regression would crash on NaN-bearing rows otherwise."""
    envs = ('A', 'B', 'C', 'D', 'E', 'F')
    treatment, baseline = _build_corpus(envs)

    # Inject NaN gap into all baseline cells of env 'C'.
    patched_baseline = []
    for r in baseline:
        if r.measurements.get('env_name') == 'C':
            patched = dict(r.measurements)
            patched['jensen_gap'] = float('nan')
            r = RunRow(
                id=r.id, parent_id=r.parent_id, cycle_id=r.cycle_id,
                timestamp=r.timestamp, verdict=r.verdict,
                arm_key=r.arm_key, measurements=patched,
            )
        patched_baseline.append(r)

    h = _three_edge_hypothesis()
    v = hypothesis_subgraph_verdict(
        h, treatment, patched_baseline,
        pair_by=('seed',), group_by='env_name',
    )
    scope = build_scope(
        v, patched_baseline,
        gap_path='jensen_gap',
        gap_name='jensen_overestimation_gap',
        target='outcome.r',
    )
    # 6 envs total, 'C' has NaN gap → dropped → 5 strata.
    assert scope.cleavage.n_strata == 5


def test_build_scope_chain_is_verdict_graph() -> None:
    """`Scope.chain` is the verdict's typed CausalGraph (no copy,
    no rebuild)."""
    envs = ('A', 'B', 'C', 'D', 'E', 'F')
    treatment, baseline = _build_corpus(envs)
    h = _three_edge_hypothesis()
    v = hypothesis_subgraph_verdict(
        h, treatment, baseline,
        pair_by=('seed',), group_by='env_name',
    )
    scope = build_scope(
        v, baseline,
        gap_path='jensen_gap',
        gap_name='jensen_overestimation_gap',
        target='outcome.r',
    )
    assert scope.chain is v.graph


def test_build_scope_missing_target_raises() -> None:
    """A hypothesis with no edge whose target matches `target`
    can't be scoped on that target — raise loudly."""
    envs = ('A', 'B', 'C', 'D')
    treatment, baseline = _build_corpus(envs)
    h: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='mech_only',
        intervention={},
        intervention_arms=_TREATMENT_ARMS,
        edges=(
            _intervention_edge('mechanism.q', 'a_lt_b'),
        ),
    )
    v = hypothesis_subgraph_verdict(
        h, treatment, baseline,
        pair_by=('seed',), group_by='env_name',
    )
    with pytest.raises(ValueError, match="target='outcome.r'"):
        _ = build_scope(
            v, baseline,
            gap_path='jensen_gap',
            gap_name='jensen_overestimation_gap',
            target='outcome.r',
        )


def test_build_scope_target_mechanism_uses_mechanism_row() -> None:
    """`target='mechanism.q'` regresses the mechanism g on the gap."""
    envs = ('A', 'B', 'C', 'D', 'E', 'F')
    treatment, baseline = _build_corpus(envs)
    h = _three_edge_hypothesis()
    v = hypothesis_subgraph_verdict(
        h, treatment, baseline,
        pair_by=('seed',), group_by='env_name',
    )
    s_out = build_scope(
        v, baseline,
        gap_path='jensen_gap',
        gap_name='jensen_overestimation_gap',
        target='outcome.r',
    )
    s_mech = build_scope(
        v, baseline,
        gap_path='jensen_gap',
        gap_name='jensen_overestimation_gap',
        target='mechanism.q',
    )
    # Different regression targets → different intercepts.
    assert not math.isclose(
        s_out.cleavage.intercept, s_mech.cleavage.intercept,
    )
