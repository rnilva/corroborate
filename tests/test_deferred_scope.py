"""Focused contracts for panel-derived bridge scopes."""
from __future__ import annotations

from collections.abc import Iterable, Mapping

import polars as pl

from corroborate._internals.polars import as_rows
from corroborate.analyses.panel.stratum_panel import StratumPanel
from corroborate.bridge.analysis import analysis
from corroborate.bridge.bridge import claim_bridge, evaluate
from corroborate.bridge.deferred_scope import scope_from_panel
from corroborate.bridge.verdict import Verdict


def _empty_panel(
    rows: Iterable[Mapping[str, object]],
) -> StratumPanel:
    materialized = list(rows)
    strata = tuple(
        (name,)
        for name in sorted({str(row['env_name']) for row in materialized})
    )
    empty_counts = tuple(0 for _ in strata)
    return StratumPanel(
        stratify_by=('env_name',),
        strata=strata,
        measurables=(),
        treatment_arm='treatment',
        baseline_arm='baseline',
        aggregator='mean',
        n_treatment=empty_counts,
        n_baseline=empty_counts,
        means_treatment={},
        means_baseline={},
        stds_treatment={},
        stds_baseline={},
        n_treatment_per_measurable={},
        n_baseline_per_measurable={},
        spearman_within={},
    )


def test_static_scope_is_applied_before_panel_and_in_final_expr() -> None:
    """The static predicate shapes the panel's input first, and is
    still ANDed into the returned expression so an excluded row in
    a surviving stratum cannot leak back into evaluation. The probe
    follows the canonical-union convention: it accepts either cells
    shape and normalises at its own entry.
    """
    panel_inputs: list[list[Mapping[str, object]]] = []

    @analysis
    def _row_consuming_panel(
        cells: pl.DataFrame | Iterable[Mapping[str, object]],
    ) -> StratumPanel:
        rows = list(as_rows(cells))
        panel_inputs.append(rows)
        return _empty_panel(rows)

    cells: list[dict[str, object]] = [
        {'id': 1, 'env_name': 'A', 'eligible': True},
        {'id': 2, 'env_name': 'A', 'eligible': False},
        {'id': 3, 'env_name': 'B', 'eligible': False},
    ]
    scope = scope_from_panel(
        panel_analysis=_row_consuming_panel,
        panel_kwargs={},
        keep=lambda _panel, _index: True,
        static_scope=pl.col('eligible'),
    )

    resolved = scope.resolve(cells)

    assert panel_inputs == [[cells[0]]]
    admitted = pl.from_dicts(cells).filter(resolved)
    assert admitted['id'].to_list() == [1]


def test_panel_analysis_receives_prefiltered_dataframe() -> None:
    """With a static scope, `resolve` builds the DataFrame once
    (the filter needs it) and hands the FILTERED frame over
    directly — the analysis's own entry normalisation owns any
    further conversion."""
    seen: list[type[object]] = []

    @analysis
    def _dataframe_panel(
        cells: pl.DataFrame | Iterable[Mapping[str, object]],
    ) -> StratumPanel:
        seen.append(type(cells))
        assert isinstance(cells, pl.DataFrame)
        return _empty_panel(cells.to_dicts())

    scope = scope_from_panel(
        panel_analysis=_dataframe_panel,
        panel_kwargs={},
        keep=lambda _panel, _index: True,
        static_scope=pl.col('env_name') == 'A',
    )
    _ = scope.resolve([{'env_name': 'A'}])
    assert seen == [pl.DataFrame]


def test_rows_pass_through_untouched_without_static_scope() -> None:
    """Without a static scope there is nothing to filter, so the
    caller's rows reach the panel analysis as the SAME object — no
    rows → DataFrame → rows round trip for row-consuming panel
    builders (regression guard for the conversion the canonical
    union made tempting)."""
    received: list[object] = []

    @analysis
    def _identity_probe_panel(
        cells: pl.DataFrame | Iterable[Mapping[str, object]],
    ) -> StratumPanel:
        received.append(cells)
        return _empty_panel(as_rows(cells))

    scope = scope_from_panel(
        panel_analysis=_identity_probe_panel,
        panel_kwargs={},
        keep=lambda _panel, _index: True,
    )
    rows: list[dict[str, object]] = [{'env_name': 'A'}]
    _ = scope.resolve(rows)
    assert received == [rows]
    assert received[0] is rows


def test_missing_static_column_is_null_padded_before_panel() -> None:
    panel_inputs: list[list[Mapping[str, object]]] = []

    @analysis
    def _missing_column_panel(
        cells: pl.DataFrame | Iterable[Mapping[str, object]],
    ) -> StratumPanel:
        rows = list(as_rows(cells))
        panel_inputs.append(rows)
        return _empty_panel(rows)

    scope = scope_from_panel(
        panel_analysis=_missing_column_panel,
        panel_kwargs={},
        keep=lambda _panel, _index: True,
        static_scope=pl.col('optional_flag') == 1,
    )
    _ = scope.resolve([{'env_name': 'A'}])
    assert panel_inputs == [[]]


def test_generator_cells_survive_deferred_scope_and_analysis() -> None:
    @analysis
    def _count_deferred_cells(
        cells: pl.DataFrame | Iterable[Mapping[str, object]],
    ) -> int:
        return len(list(as_rows(cells)))

    @analysis
    def _all_strata_panel(
        cells: pl.DataFrame | Iterable[Mapping[str, object]],
    ) -> StratumPanel:
        return _empty_panel(as_rows(cells))

    scope = scope_from_panel(
        panel_analysis=_all_strata_panel,
        panel_kwargs={},
        keep=lambda _panel, _index: True,
    )

    @claim_bridge(source='x', target='x', scope=scope)
    def generator_bridge(_count_deferred_cells: int) -> Verdict:
        return (
            Verdict.HELD
            if _count_deferred_cells == 2
            else Verdict.POWER_INSUFFICIENT
        )

    rows: list[dict[str, object]] = [
        {'id': 'a', 'env_name': 'A', 'x': 1.0},
        {'id': 'b', 'env_name': 'A', 'x': 2.0},
    ]
    result = evaluate(generator_bridge, (row for row in rows))
    assert result.n_cells_in_scope == 2
    assert result.verdict is Verdict.HELD
