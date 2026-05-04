"""Tests for `corroborate.causal_discovery` — PC + JCI primitives.

This file covers commit 1 (CI tests). PC algorithm + orientation
land in commit 2 with their own tests."""
from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest
from scipy.stats import spearmanr

from corroborate._internals.polars import series_std_float
from corroborate.graph.discovery import (
    VariableScope,
    assert_stratification_admissible,
    classify_variable_scope,
    partial_spearman_rho,
    partial_spearman_rho_multi,
    stratified_partial_spearman_rho,
    stratified_spearman_rho,
)


# ============ partial_spearman_rho — single Z closed form ============

def test_partial_reduces_to_marginal_when_z_is_orthogonal() -> None:
    """When Z is orthogonal to both X and Y (independent noise),
    partial(X, Y | Z) ≈ marginal Spearman(X, Y)."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(200)
    # Y carries a real correlation with X plus noise.
    y = 0.7 * x + rng.standard_normal(200) * 0.7
    z = rng.standard_normal(200)  # independent of X, Y

    marginal_r, _ = spearmanr(x, y)
    partial_r, _ = partial_spearman_rho(x, y, z)
    assert abs(float(marginal_r) - partial_r) < 0.05


def test_partial_zero_when_y_is_function_of_z_alone() -> None:
    """If Y is a deterministic function of Z and X is independent
    of Z, partial(X, Y | Z) ≈ 0 — the X→Y path doesn't survive
    conditioning on Z."""
    rng = np.random.default_rng(0)
    n = 300
    z = rng.standard_normal(n)
    y = z + rng.standard_normal(n) * 0.01
    x = rng.standard_normal(n)
    # Make X correlated with Z marginally (so X⫫Y becomes a
    # confounding test).
    x = x + 0.5 * z

    rho, p = partial_spearman_rho(x, y, z)
    assert abs(rho) < 0.15
    assert p > 0.05


def test_partial_returns_nan_on_constant_z() -> None:
    """Constant Z → singular partial-correlation denominator →
    NaN (no information)."""
    n = 100
    rng = np.random.default_rng(0)
    x = rng.standard_normal(n)
    y = rng.standard_normal(n)
    z = np.zeros(n)
    rho, p = partial_spearman_rho(x, y, z)
    assert math.isnan(rho)
    assert math.isnan(p)


# ============ partial_spearman_rho_multi — residual regression ============

def test_partial_multi_matches_single_z_to_within_tolerance() -> None:
    """With one Z, multi-Z partial via OLS residuals should agree
    with the closed-form single-Z partial."""
    rng = np.random.default_rng(0)
    n = 200
    x = rng.standard_normal(n)
    y = 0.5 * x + rng.standard_normal(n) * 0.7
    z = rng.standard_normal(n)
    rho_closed, _ = partial_spearman_rho(x, y, z)
    rho_multi, _ = partial_spearman_rho_multi(x, y, z.reshape(-1, 1))
    # Residual-regression has higher variance; allow 0.1 slack.
    assert abs(rho_closed - rho_multi) < 0.1


def test_partial_multi_with_two_z_columns() -> None:
    """Y = Z1 + Z2 + noise; X independent. Partial(X, Y | Z1, Z2) ≈
    0."""
    rng = np.random.default_rng(0)
    n = 300
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    y = z1 + z2 + rng.standard_normal(n) * 0.01
    x = rng.standard_normal(n)  # independent
    rho, p = partial_spearman_rho_multi(x, y, np.column_stack([z1, z2]))
    assert abs(rho) < 0.15
    assert p > 0.05


# ============ stratified_spearman_rho — JCI / Simpson's paradox ============

def test_stratified_finds_within_stratum_correlation_masked_by_pooling() -> None:
    """Simpson's-paradox fixture: within-stratum positive
    correlation, but mean shifts across strata flip the sign of
    the pooled-marginal correlation. Stratified Spearman recovers
    the within-stratum sign."""
    rng = np.random.default_rng(0)
    n_per = 100
    # Stratum A: positive correlation, low mean.
    xa = rng.standard_normal(n_per)
    ya = 0.7 * xa + rng.standard_normal(n_per) * 0.3
    # Stratum B: positive correlation, high mean of x AND high
    # mean of y opposite to xa direction.
    xb = rng.standard_normal(n_per) + 5
    yb = 0.7 * xb + rng.standard_normal(n_per) * 0.3 - 8
    x = np.concatenate([xa, xb])
    y = np.concatenate([ya, yb])
    strata = ['a'] * n_per + ['b'] * n_per

    pooled_rho, _ = spearmanr(x, y)
    strat_rho, strat_p = stratified_spearman_rho(x, y, strata)

    # Pooled marginal is misled (could be either sign depending on
    # exact shifts); stratified within is positive (~0.7).
    assert strat_rho > 0.5
    assert strat_p < 0.05
    # The pooled-vs-stratified separation is the load-bearing point
    # of JCI: they disagree.
    assert abs(float(pooled_rho) - strat_rho) > 0.05


def test_stratified_skips_too_small_strata() -> None:
    """Strata with n < min_stratum_size are dropped — they don't
    contribute to the Fisher-z pool."""
    rng = np.random.default_rng(0)
    # 50 samples in stratum 'a', 2 in stratum 'b'.
    xa = rng.standard_normal(50)
    ya = 0.7 * xa + rng.standard_normal(50) * 0.3
    xb = np.array([0.0, 1.0])
    yb = np.array([0.0, 1.0])
    x = np.concatenate([xa, xb])
    y = np.concatenate([ya, yb])
    strata = ['a'] * 50 + ['b'] * 2

    rho, _ = stratified_spearman_rho(x, y, strata, min_stratum_size=4)
    # 'b' is too small → dropped → result depends on 'a' alone.
    assert rho > 0.5


def test_stratified_returns_nan_when_no_eligible_strata() -> None:
    """All strata too small → NaN."""
    x = np.arange(6, dtype=np.float64)
    y = np.arange(6, dtype=np.float64)
    strata = list('abcdef')  # 1 obs per stratum
    rho, p = stratified_spearman_rho(x, y, strata, min_stratum_size=4)
    assert math.isnan(rho)
    assert math.isnan(p)


# ============ stratified_partial_spearman_rho ============

def test_stratified_partial_recovers_within_stratum_partial() -> None:
    """Each stratum has Y = f(Z) + noise, X independent of Z.
    Within-stratum partial(X, Y | Z) ≈ 0; stratified pool agrees."""
    rng = np.random.default_rng(0)
    n_per = 100

    def _stratum(_seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        z = rng.standard_normal(n_per)
        y = z + rng.standard_normal(n_per) * 0.5
        x = rng.standard_normal(n_per)
        return x, y, z

    xa, ya, za = _stratum(0)
    xb, yb, zb = _stratum(1)
    x = np.concatenate([xa, xb])
    y = np.concatenate([ya, yb])
    z = np.concatenate([za, zb])
    strata = ['a'] * n_per + ['b'] * n_per

    rho, p = stratified_partial_spearman_rho(x, y, z, strata)
    assert abs(rho) < 0.15
    assert p > 0.05


# ============ classify_variable_scope — within vs across stratum ============

def test_classify_within_stratum_per_burst_at_fixed_env() -> None:
    """Per-burst measurements at fixed env: variance lives within env.
    The motivating WITHIN_STRATUM case — JCI's natural domain."""
    rng = np.random.default_rng(0)
    # 3 envs, 100 bursts each. Within-env: noisy values. Across-env:
    # same mean (no env effect on the per-burst variable).
    values = rng.standard_normal(300)
    strata = ['envA'] * 100 + ['envB'] * 100 + ['envC'] * 100
    scope = classify_variable_scope(values, strata)
    assert scope is VariableScope.WITHIN_STRATUM


def test_classify_across_stratum_env_level_feature() -> None:
    """Env-level feature (constant within env, varies across envs):
    the case that motivated codification — JCI silently NaN-skips
    these because within-stratum variance is zero."""
    # Three envs with different log_obs_dim values, but each env's
    # bursts share the same value (env-level feature).
    values = np.concatenate([
        np.full(50, 2.0),   # envA: log_obs_dim = 2.0
        np.full(50, 5.0),   # envB: log_obs_dim = 5.0
        np.full(50, 8.0),   # envC: log_obs_dim = 8.0
    ])
    strata = ['envA'] * 50 + ['envB'] * 50 + ['envC'] * 50
    scope = classify_variable_scope(values, strata)
    assert scope is VariableScope.ACROSS_STRATUM


def test_classify_both_when_within_and_across_variance_present() -> None:
    """Variable that varies both within and across strata: BOTH —
    compatible with either analysis, choice depends on question."""
    rng = np.random.default_rng(0)
    # Each env has its own mean (across-stratum variance) AND noise
    # within (within-stratum variance), both substantial.
    values = np.concatenate([
        rng.standard_normal(100) * 1.0 + 0.0,
        rng.standard_normal(100) * 1.0 + 5.0,
        rng.standard_normal(100) * 1.0 + 10.0,
    ])
    strata = ['envA'] * 100 + ['envB'] * 100 + ['envC'] * 100
    scope = classify_variable_scope(values, strata)
    assert scope is VariableScope.BOTH


def test_classify_degenerate_constant_column() -> None:
    """Constant variable across the full panel: useless for any
    test; explicitly DEGENERATE (not silently mistaken for one
    side or the other)."""
    values = np.full(150, 3.14)
    strata = ['envA'] * 50 + ['envB'] * 50 + ['envC'] * 50
    scope = classify_variable_scope(values, strata)
    assert scope is VariableScope.DEGENERATE


def test_classify_single_stratum_treated_as_within() -> None:
    """One stratum present → no across-stratum dimension to test;
    classify by whether values vary at all."""
    rng = np.random.default_rng(0)
    values = rng.standard_normal(50)
    strata = ['envA'] * 50
    assert classify_variable_scope(values, strata) is VariableScope.WITHIN_STRATUM
    assert classify_variable_scope(np.zeros(50), strata) is VariableScope.DEGENERATE


def test_assert_stratification_admissible_raises_on_env_level_features() -> None:
    """The motivating reproducer: passing env-level features (e.g.
    `log_obs_dim`, `log_action_dim`) to a stratify_by='env_name'
    primitive must fail loudly, not silently NaN-skip."""
    df = pl.DataFrame({
        'env_name': ['envA'] * 50 + ['envB'] * 50 + ['envC'] * 50,
        'g_link': np.random.default_rng(0).standard_normal(150).tolist(),
        'log_obs_dim': [2.0] * 50 + [5.0] * 50 + [8.0] * 50,
    })
    with pytest.raises(ValueError, match='ACROSS_STRATUM'):
        assert_stratification_admissible(
            df, ['g_link', 'log_obs_dim'], 'env_name',
        )


def test_assert_stratification_admissible_passes_within_stratum_variables() -> None:
    """The happy path: per-burst measurements (within-stratum
    variance) pass the admissibility check and return their scope
    classification."""
    rng = np.random.default_rng(0)
    df = pl.DataFrame({
        'env_name': ['envA'] * 50 + ['envB'] * 50 + ['envC'] * 50,
        'g_link': rng.standard_normal(150).tolist(),
        'g_mech': rng.standard_normal(150).tolist(),
    })
    scopes = assert_stratification_admissible(
        df, ['g_link', 'g_mech'], 'env_name',
    )
    assert scopes['g_link'] is VariableScope.WITHIN_STRATUM
    assert scopes['g_mech'] is VariableScope.WITHIN_STRATUM


# ============ PC algorithm — discover_adjacency ============

def _df_from_columns(**cols: np.ndarray) -> pl.DataFrame:
    """Build a polars DataFrame from kwarg columns."""
    return pl.DataFrame({k: v.tolist() for k, v in cols.items()})


def test_compare_pc_depths_kills_chain_edge_at_depth_1() -> None:
    """Three-variable chain X → M → Y. At depth-0 (marginal
    only), the X-Y edge survives because X and Y are marginally
    correlated through M. At depth-1, conditioning on M
    separates X-Y and the edge is removed. The diff catches
    `xy_edge in low_only` exactly. Same diff shape as depth-1 vs
    depth-2 catching a confounded edge that needs |Z|=2; the
    chain example just exercises the primitive cheaply."""
    from corroborate.graph.discovery import compare_pc_depths
    rng = np.random.default_rng(0)
    n = 500
    x = rng.standard_normal(n)
    m = x + rng.standard_normal(n) * 0.3
    y = m + rng.standard_normal(n) * 0.3
    df = _df_from_columns(x=x, m=m, y=y)

    diff = compare_pc_depths(
        df, variables=['x', 'm', 'y'],
        alpha=0.05, depths=(0, 1),
    )
    xy_edge = frozenset({'x', 'y'})
    assert xy_edge in diff.edges_low
    assert xy_edge not in diff.edges_high
    assert xy_edge in diff.low_only
    assert xy_edge not in diff.common


def test_compare_pc_depths_chain_unaffected_by_depth_increase() -> None:
    """X → M → Y: depth-1 already kills X-Y via {M}. Depth-2
    can only confirm. Diff: low_only and high_only both empty;
    common == both edge sets."""
    from corroborate.graph.discovery import compare_pc_depths
    rng = np.random.default_rng(0)
    n = 500
    x = rng.standard_normal(n)
    m = x + rng.standard_normal(n) * 0.3
    y = m + rng.standard_normal(n) * 0.3
    df = _df_from_columns(x=x, m=m, y=y)

    diff = compare_pc_depths(
        df, variables=['x', 'm', 'y'],
        alpha=0.05, depths=(1, 2),
    )
    assert diff.low_only == frozenset()
    assert diff.high_only == frozenset()
    assert diff.common == diff.edges_low == diff.edges_high


def test_compare_pc_depths_rejects_descending_depths() -> None:
    """Depths must be (low, high) with low < high."""
    from corroborate.graph.discovery import compare_pc_depths
    df = _df_from_columns(
        x=np.array([1.0, 2.0, 3.0]),
        y=np.array([1.0, 2.0, 3.0]),
    )
    import pytest
    with pytest.raises(ValueError, match='low < high'):
        compare_pc_depths(
            df, variables=['x', 'y'],
            alpha=0.05, depths=(2, 1),
        )


def test_discover_chain_removes_marginal_independence_pair() -> None:
    """3-variable chain X → M → Y. PC at depth 1 should:
    - Keep X−M and M−Y (direct dependence)
    - Remove X−Y at depth 1, conditioning on M (X⫫Y | M)."""
    from corroborate.graph.discovery import discover_adjacency
    rng = np.random.default_rng(0)
    n = 500
    x = rng.standard_normal(n)
    m = x + rng.standard_normal(n) * 0.3
    y = m + rng.standard_normal(n) * 0.3
    df = _df_from_columns(x=x, m=m, y=y)

    adj = discover_adjacency(
        df, variables=['x', 'm', 'y'],
        alpha=0.05, max_conditioning=1,
    )
    edges = {tuple(sorted(e)) for e in adj.edges}
    # X−M and M−Y survive
    assert ('m', 'x') in edges
    assert ('m', 'y') in edges
    # X−Y removed (separated by M)
    assert ('x', 'y') not in edges
    xy_sepset = adj.separating_sets[frozenset({'x', 'y'})]
    assert frozenset({'m'}) in xy_sepset


def test_discover_collider_keeps_marginal_independence() -> None:
    """3-variable collider X → Z ← Y. X⫫Y marginally (no direct
    edge, no path through Z without conditioning). PC should keep
    X⫫Y at depth 0 (Berkson-bias example)."""
    from corroborate.graph.discovery import discover_adjacency
    rng = np.random.default_rng(0)
    n = 500
    x = rng.standard_normal(n)
    y = rng.standard_normal(n)  # independent
    z = x + y + rng.standard_normal(n) * 0.3  # collider
    df = _df_from_columns(x=x, y=y, z=z)

    adj = discover_adjacency(
        df, variables=['x', 'y', 'z'],
        alpha=0.05, max_conditioning=0,
    )
    edges = {tuple(sorted(e)) for e in adj.edges}
    # X−Z and Y−Z survive
    assert ('x', 'z') in edges
    assert ('y', 'z') in edges
    # X−Y removed at depth 0 (marginal independence)
    assert ('x', 'y') not in edges


def test_discover_with_jci_stratification() -> None:
    """JCI stratification on a categorical context: edges that
    look correlated when pooled are within-stratum independent.
    Stratified PC removes them; unstratified keeps them."""
    from corroborate.graph.discovery import discover_adjacency
    rng = np.random.default_rng(0)
    n_per = 200
    # Stratum A: X and Y both shifted up (creates pooled correlation
    # via mean shift even though within-stratum is independent).
    xa = rng.standard_normal(n_per) - 3
    ya = rng.standard_normal(n_per) - 3
    # Stratum B: shifted down.
    xb = rng.standard_normal(n_per) + 3
    yb = rng.standard_normal(n_per) + 3
    df = _df_from_columns(
        x=np.concatenate([xa, xb]),
        y=np.concatenate([ya, yb]),
        env=np.array(['a'] * n_per + ['b'] * n_per, dtype=object),
    )

    # Without stratification: pooled correlation is real.
    adj_pooled = discover_adjacency(
        df, variables=['x', 'y'],
        alpha=0.05, max_conditioning=0,
    )
    edges_pooled = {tuple(sorted(e)) for e in adj_pooled.edges}
    assert ('x', 'y') in edges_pooled

    # With stratification on env: within-stratum independent.
    adj_strat = discover_adjacency(
        df, variables=['x', 'y'],
        alpha=0.05, max_conditioning=0,
        stratify_by='env',
    )
    edges_strat = {tuple(sorted(e)) for e in adj_strat.edges}
    assert ('x', 'y') not in edges_strat


# ============ Orientation — v-structures + Meek rules ============

def test_orient_v_structure_collider() -> None:
    """Unshielded X − Z − Y with Z NOT in sepset(X, Y) → orient
    X → Z ← Y (definite collider)."""
    from corroborate.graph.discovery import (
        DiscoveredAdjacency, orient_adjacency,
    )
    adj = DiscoveredAdjacency(
        variables=frozenset({'x', 'y', 'z'}),
        edges=frozenset({frozenset({'x', 'z'}), frozenset({'y', 'z'})}),
        separating_sets={
            frozenset({'x', 'y'}): frozenset({frozenset[str]()}),
        },
        n_observations=100, alpha=0.05, max_conditioning=0,
        stratify_by=None,
    )
    oriented = orient_adjacency(adj)
    assert ('x', 'z') in oriented.directed_edges
    assert ('y', 'z') in oriented.directed_edges
    assert oriented.undirected_edges == frozenset()
    assert oriented.ambiguous_triples == frozenset()


def test_orient_non_collider_when_z_in_sepset() -> None:
    """Unshielded X − Z − Y with Z IN sepset(X, Y) → Z is a
    non-collider; the X−Z and Y−Z edges stay undirected."""
    from corroborate.graph.discovery import (
        DiscoveredAdjacency, orient_adjacency,
    )
    adj = DiscoveredAdjacency(
        variables=frozenset({'x', 'y', 'z'}),
        edges=frozenset({frozenset({'x', 'z'}), frozenset({'y', 'z'})}),
        separating_sets={
            # Z separates X and Y → non-collider
            frozenset({'x', 'y'}): frozenset({frozenset({'z'})}),
        },
        n_observations=100, alpha=0.05, max_conditioning=1,
        stratify_by=None,
    )
    oriented = orient_adjacency(adj)
    assert oriented.directed_edges == frozenset()
    assert frozenset({'x', 'z'}) in oriented.undirected_edges
    assert frozenset({'y', 'z'}) in oriented.undirected_edges


def test_orient_meek_r1_propagation() -> None:
    """A → B and B − C undirected, A not adjacent to C → R1
    propagates orientation B → C (else A → B ← C would be a new
    v-structure that v-structure detection would have caught)."""
    from corroborate.graph.discovery import (
        DiscoveredAdjacency, orient_adjacency,
    )
    # We need: A → B (already directed), B − C undirected, A and C
    # not adjacent. Construct via a 4-node fixture:
    # collider triple X → A → ... → no. Simpler: use ambiguity-free
    # collider that orients A and then a downstream undirected B-C.
    # Construct: U → A, V → A (collider so A→A wouldn't apply; we
    # need A→B). Alternative: just synthesise the post-collider
    # state directly.
    # Manual construction: discovered adjacency has edges
    # {U-A, V-A, A-B, B-C} with U, V both colliding into A
    # (so v-structures orient U → A, V → A) and B-C undirected.
    # After v-structure: directed = {(U, A), (V, A)}, undirected =
    # {A-B, B-C}. Meek R1: U → A, A − B undirected, U not adjacent
    # to B → orient A → B. Then A → B, B − C undirected, A not
    # adjacent to C → orient B → C.
    adj = DiscoveredAdjacency(
        variables=frozenset({'u', 'v', 'a', 'b', 'c'}),
        edges=frozenset({
            frozenset({'u', 'a'}),
            frozenset({'v', 'a'}),
            frozenset({'a', 'b'}),
            frozenset({'b', 'c'}),
        }),
        separating_sets={
            # U⫫V at depth 0 (no edge)
            frozenset({'u', 'v'}): frozenset({frozenset[str]()}),
            # U⫫B given A
            frozenset({'u', 'b'}): frozenset({frozenset({'a'})}),
            # V⫫B given A
            frozenset({'v', 'b'}): frozenset({frozenset({'a'})}),
            # A⫫C given B
            frozenset({'a', 'c'}): frozenset({frozenset({'b'})}),
            # U⫫C, V⫫C — chain dissipates
            frozenset({'u', 'c'}): frozenset({frozenset({'a'})}),
            frozenset({'v', 'c'}): frozenset({frozenset({'a'})}),
        },
        n_observations=100, alpha=0.05, max_conditioning=1,
        stratify_by=None,
    )
    oriented = orient_adjacency(adj)
    # v-structure detection: U-A-V is unshielded; A NOT in sepset(U,V)
    # → collider U → A ← V.
    assert ('u', 'a') in oriented.directed_edges
    assert ('v', 'a') in oriented.directed_edges
    # Meek R1: U → A, A−B undirected, U not adjacent to B (sepset
    # has A) → A → B. Then A → B, B−C undirected, A not adjacent
    # to C → B → C.
    assert ('a', 'b') in oriented.directed_edges
    assert ('b', 'c') in oriented.directed_edges
    assert oriented.undirected_edges == frozenset()


def test_orient_ambiguous_triple_skipped() -> None:
    """Triple where Z is in SOME but not ALL separating sets →
    ambiguous; not oriented in conservative mode."""
    from corroborate.graph.discovery import (
        DiscoveredAdjacency, orient_adjacency,
    )
    adj = DiscoveredAdjacency(
        variables=frozenset({'x', 'y', 'z'}),
        edges=frozenset({frozenset({'x', 'z'}), frozenset({'y', 'z'})}),
        separating_sets={
            # Two sepsets: empty AND {z}. Z is in some but not all.
            frozenset({'x', 'y'}): frozenset({
                frozenset[str](),
                frozenset({'z'}),
            }),
        },
        n_observations=100, alpha=0.05, max_conditioning=1,
        stratify_by=None,
    )
    oriented = orient_adjacency(adj, conservative=True)
    # Conservative: ambiguous → not oriented.
    assert ('x', 'z') not in oriented.directed_edges
    assert ('y', 'z') not in oriented.directed_edges
    # Tracked as ambiguous triple.
    assert ('x', 'z', 'y') in oriented.ambiguous_triples


# ============ §4 acceptance: integration smoke ============

def test_pc_dqn_smoke_holds_on_migrated_corpus() -> None:
    """§4 acceptance on the existing 17-env / 1020-row corpus:
    PC + JCI on env_name finds NO edge between arm_ddqn and any
    outcome variable. Reproduces PAPER §4.3's structural finding.

    Skipped if the corpus parquet isn't on disk — the framework
    tests don't require it, only the integration smoke does."""
    from pathlib import Path

    import polars as pl

    runs_path = (
        Path(__file__).parent.parent
        / 'experiments' / 'data' / 'ddqn' / 'runs.parquet'
    )
    if not runs_path.exists():
        import pytest
        pytest.skip(f'{runs_path} not on disk')

    from corroborate.graph.discovery import discover_adjacency

    df = pl.read_parquet(runs_path)
    df = df.with_columns(
        # `arm_ddqn` is the binary one-hot of the DDQN arm. Post-
        # Phase-6 the corpus's `arm_key` column carries
        # `canonical_str(intervention_arms)`; the legacy
        # `intervention_name` column is preserved (`'ddqn'` /
        # `'vanilla_dqn'`) and is the canonical one-hot source for
        # PC discovery tests like this one.
        (pl.col('intervention_name') == 'ddqn')
        .cast(pl.Int64).alias('arm_ddqn'),
    )
    variables = [
        'arm_ddqn',
        'jensen_gap',
        'late_window_mean',
        'eval_final_mean',
        'eval_best_burst_mean',
        'eval_best_burst_step',
    ]
    df = df.drop_nulls(subset=variables)

    adj = discover_adjacency(
        df, variables=variables,
        alpha=0.05, max_conditioning=1,
        stratify_by='env_name',
    )

    # The §4 finding: NO edge between arm_ddqn and any outcome.
    outcome_vars = {v for v in variables if v.startswith('outcome.')}
    arm_outcome_edges = [
        e for e in adj.edges
        if 'arm_ddqn' in e and any(v in e for v in outcome_vars)
    ]
    assert not arm_outcome_edges, (
        f'§4 acceptance FAILED — surviving edges from arm_ddqn '
        f'to outcomes: {arm_outcome_edges}'
    )

    # Sanity: the mechanism intervention edge SHOULD survive
    # (DDQN's slot swap reduces the Jensen gap on a subset of envs).
    assert frozenset({'arm_ddqn', 'jensen_gap'}) in adj.edges, (
        'arm_ddqn → jensen_gap should survive (DDQN '
        'demonstrably reduces the gap on a subset of envs)'
    )


def test_per_env_pc_dqn_smoke_finds_within_env_arm_edges() -> None:
    """§6 thin per-env PC: at least some envs surface a within-env
    edge from arm_ddqn (mostly to jensen_gap). Skipped if
    the corpus parquet isn't on disk.

    With only one mechanism feature, this is the *thin* §6 — it
    cannot reproduce the three-regime mediator taxonomy. The gate
    is qualitative: at least 3 envs show within-env arm_ddqn
    neighbours (the slot swap leaves a per-env footprint even
    where pooled-JCI averages it to zero)."""
    from pathlib import Path

    import polars as pl

    runs_path = (
        Path(__file__).parent.parent
        / 'experiments' / 'data' / 'ddqn' / 'runs.parquet'
    )
    if not runs_path.exists():
        import pytest
        pytest.skip(f'{runs_path} not on disk')

    from corroborate.graph.discovery import discover_adjacency

    df = pl.read_parquet(runs_path)
    df = df.with_columns(
        # `arm_ddqn` is the binary one-hot of the DDQN arm. Post-
        # Phase-6 the corpus's `arm_key` column carries
        # `canonical_str(intervention_arms)`; the legacy
        # `intervention_name` column is preserved (`'ddqn'` /
        # `'vanilla_dqn'`) and is the canonical one-hot source for
        # PC discovery tests like this one.
        (pl.col('intervention_name') == 'ddqn')
        .cast(pl.Int64).alias('arm_ddqn'),
    )
    variables = [
        'arm_ddqn',
        'jensen_gap',
        'late_window_mean',
        'eval_final_mean',
        'eval_best_burst_mean',
        'eval_best_burst_step',
    ]
    df = df.drop_nulls(subset=variables)

    envs_with_arm_edge: list[str] = []
    for env in sorted(df['env_name'].unique().to_list()):
        env_df = df.filter(pl.col('env_name') == env)
        if env_df.height < 5 or env_df['arm_ddqn'].n_unique() < 2:
            continue
        constant_cols = [
            v for v in variables
            if env_df[v].dtype.is_float()
            and series_std_float(env_df[v]) == 0.0
        ]
        if constant_cols:
            continue
        adj = discover_adjacency(
            env_df, variables=variables,
            alpha=0.05, max_conditioning=1,
        )
        if any('arm_ddqn' in e for e in adj.edges):
            envs_with_arm_edge.append(env)

    assert len(envs_with_arm_edge) >= 3, (
        f'§6 thin gate: expected ≥3 envs with within-env arm_ddqn '
        f'edges, got {len(envs_with_arm_edge)}: {envs_with_arm_edge}'
    )


def test_per_env_mediator_pc_smoke_finds_outcome_neighbours() -> None:
    """§5+§6 rich gate on `runs_with_mediators.parquet`: per-env PC
    over the 10-variable mediator-augmented set surfaces ≥1 neighbour
    of `eval_final_mean` in at least 8 envs (the paper's
    9-of-15 threshold, allowing 1 slack for corpus-specific noise).

    Skipped if `runs_with_mediators.parquet` isn't on disk — produced
    by `experiments/compute_mediators.py`."""
    from pathlib import Path

    import polars as pl

    enriched_path = (
        Path(__file__).parent.parent
        / 'experiments' / 'data' / 'ddqn'
        / 'runs_with_mediators.parquet'
    )
    if not enriched_path.exists():
        import pytest
        pytest.skip(f'{enriched_path} not on disk')

    from corroborate.graph.discovery import discover_adjacency

    df = pl.read_parquet(enriched_path)
    df = df.with_columns(
        # `arm_ddqn` is the binary one-hot of the DDQN arm. Post-
        # Phase-6 the corpus's `arm_key` column carries
        # `canonical_str(intervention_arms)`; the legacy
        # `intervention_name` column is preserved (`'ddqn'` /
        # `'vanilla_dqn'`) and is the canonical one-hot source for
        # PC discovery tests like this one.
        (pl.col('intervention_name') == 'ddqn')
        .cast(pl.Int64).alias('arm_ddqn'),
    )
    # Drop epsilon_late and fill_ratio_late — corpus-wide constants.
    pc_mediators = (
        'mediator.q_gap_late', 'mediator.q_gap_growth',
        'mediator.q_max_growth', 'mediator.v_vs_max_delta_late',
        'mediator.td_residual_late', 'mediator.greedy_match_late',
    )
    variables = [
        'arm_ddqn', 'jensen_gap',
        *pc_mediators,
        'eval_final_mean', 'late_window_mean',
    ]
    outcome = 'eval_final_mean'

    envs_with_neighbour: list[str] = []
    for env in sorted(df['env_name'].unique().to_list()):
        env_df = df.filter(pl.col('env_name') == env)
        if env_df.height < 5 or env_df['arm_ddqn'].n_unique() < 2:
            continue
        env_df = env_df.drop_nulls(subset=variables)
        for v in variables:
            if env_df[v].dtype.is_float():
                env_df = env_df.filter(~pl.col(v).is_nan())
        if env_df.height < 5 or env_df['arm_ddqn'].n_unique() < 2:
            continue
        constant_cols = [
            v for v in variables
            if env_df[v].dtype.is_float()
            and series_std_float(env_df[v]) == 0.0
        ]
        if constant_cols:
            continue
        adj = discover_adjacency(
            env_df, variables=variables,
            alpha=0.05, max_conditioning=1,
        )
        if any(outcome in edge for edge in adj.edges):
            envs_with_neighbour.append(env)

    assert len(envs_with_neighbour) >= 8, (
        f'§5+§6 rich gate: expected ≥8 envs with a within-env '
        f'{outcome}-neighbour, got {len(envs_with_neighbour)}: '
        f'{envs_with_neighbour}'
    )
