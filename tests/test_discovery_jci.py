"""Regression tests for JCI-stratified PC discovery."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl
import pytest

from corroborate.graph import discovery


def test_depth_two_pc_uses_stratified_multi_conditioning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A JCI context must remain active for every conditioning depth."""
    data = pl.DataFrame({
        "x": np.arange(12, dtype=np.float64),
        "y": np.arange(12, dtype=np.float64),
        "z1": np.arange(12, dtype=np.float64),
        "z2": np.arange(12, dtype=np.float64),
        "context": ["a"] * 6 + ["b"] * 6,
    })
    multi_calls: list[tuple[int, int, tuple[object, ...]]] = []

    def dependent_marginal(
        _x: np.ndarray,
        _y: np.ndarray,
        _strata: Sequence[object],
    ) -> tuple[float, float]:
        return 0.5, 0.0

    def dependent_single_condition(
        _x: np.ndarray,
        _y: np.ndarray,
        _z: np.ndarray,
        _strata: Sequence[object],
    ) -> tuple[float, float]:
        return 0.5, 0.0

    def independent_multi_condition(
        x: np.ndarray,
        _y: np.ndarray,
        z: np.ndarray,
        strata: Sequence[object],
    ) -> tuple[float, float]:
        multi_calls.append((len(x), z.shape[1], tuple(strata)))
        return 0.0, 1.0

    def unstratified_multi_must_not_run(
        _x: np.ndarray,
        _y: np.ndarray,
        _z: np.ndarray,
    ) -> tuple[float, float]:
        raise AssertionError(
            "depth-two JCI discovery discarded its stratification context",
        )

    monkeypatch.setattr(
        discovery,
        "stratified_spearman_rho",
        dependent_marginal,
    )
    monkeypatch.setattr(
        discovery,
        "stratified_partial_spearman_rho",
        dependent_single_condition,
    )
    monkeypatch.setattr(
        discovery,
        "stratified_partial_spearman_rho_multi",
        independent_multi_condition,
    )
    monkeypatch.setattr(
        discovery,
        "partial_spearman_rho_multi",
        unstratified_multi_must_not_run,
    )

    result = discovery.discover_adjacency(
        data,
        variables=("x", "y", "z1", "z2"),
        alpha=0.05,
        max_conditioning=2,
        stratify_by="context",
    )

    assert multi_calls
    assert all(n == 12 and k == 2 for n, k, _ in multi_calls)
    assert all(
        contexts == tuple(data["context"].to_list())
        for _, _, contexts in multi_calls
    )
    assert result.edges == frozenset()
    assert result.stratify_by == "context"


def test_depth_two_jci_removes_context_confounding() -> None:
    """Depth-two CI must remove an edge hidden by pooled confounding."""
    generator = np.random.default_rng(0)
    n_per_context = 100
    context = ["a"] * n_per_context + ["b"] * n_per_context
    context_shift = np.concatenate((
        -np.ones(n_per_context, dtype=np.float64),
        np.ones(n_per_context, dtype=np.float64),
    ))
    z1 = generator.normal(size=2 * n_per_context)
    z2 = generator.normal(size=2 * n_per_context)
    x = (
        0.4 * z1
        + 0.4 * z2
        + 4.0 * context_shift
        + generator.normal(scale=0.7, size=2 * n_per_context)
    )
    y = (
        0.4 * z1
        + 0.4 * z2
        + 4.0 * context_shift
        + generator.normal(scale=0.7, size=2 * n_per_context)
    )
    conditioning = np.column_stack((z1, z2))

    _, p_stratified = discovery.stratified_partial_spearman_rho_multi(
        x,
        y,
        conditioning,
        context,
    )
    _, p_pooled = discovery.partial_spearman_rho_multi(
        x,
        y,
        conditioning,
    )
    assert p_stratified >= 0.05
    assert p_pooled < 0.05

    data = pl.DataFrame({
        "x": x,
        "y": y,
        "z1": z1,
        "z2": z2,
        "context": context,
    })
    depth_one = discovery.discover_adjacency(
        data,
        variables=("x", "y", "z1", "z2"),
        alpha=0.05,
        max_conditioning=1,
        stratify_by="context",
    )
    result = discovery.discover_adjacency(
        data,
        variables=("x", "y", "z1", "z2"),
        alpha=0.05,
        max_conditioning=2,
        stratify_by="context",
    )

    xy = frozenset(("x", "y"))
    assert xy in depth_one.edges
    assert xy not in result.edges
    assert result.separating_sets[xy] == frozenset({
        frozenset(("z1", "z2")),
    })
