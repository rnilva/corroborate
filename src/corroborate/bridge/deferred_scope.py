"""Deferred scope — bridge scopes that resolve at evaluation time
by computing a panel from cells and applying a per-stratum
predicate.

The framework's regular scope predicates are `pl.Expr` — declarative
and computed at decorator time. Some scopes legitimately depend
on per-stratum aggregates of the cells themselves (e.g. "restrict
to envs where MC_disc ↔ MC_raw coupling is below 0.7"), which
require deferred resolution.

`scope_from_panel` returns a `DeferredScope` that wraps:
1. A panel-building analysis function (typically `stratum_panel`)
2. The kwargs to feed it
3. A `keep` predicate over per-stratum panel data
4. The stratify column to filter cells by
5. An optional `static_scope` that's AND-combined with the dynamic
   `is_in(surviving_strata)` filter

At bridge evaluation time, `Bridge.evaluate()` calls
`DeferredScope.resolve(cells)` to produce a `pl.Expr` and applies
it like any other scope.

This closes the framework's "use the correlation edge directly"
gap (γ in the original discussion)."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import polars as pl

from corroborate.analyses.panel.stratum_panel import StratumPanel
from corroborate.bridge.analysis import Analysis


@dataclass(frozen=True, slots=True)
class DeferredScope:
    """A bridge scope that resolves at evaluation time.

    `panel_analysis` is the `@analysis`-decorated panel builder
    (typically `stratum_panel`). `panel_kwargs` are the kwargs to
    pass it. `keep` is a per-stratum predicate. `stratify_column`
    names the column whose values are filtered on. `static_scope`
    is AND-combined with the dynamic filter."""
    panel_analysis: Analysis[Mapping[str, object], StratumPanel]
    panel_kwargs: Mapping[str, object]
    keep: Callable[[StratumPanel, int], bool]
    stratify_column: str
    static_scope: pl.Expr | None = None

    def resolve(
        self,
        cells: list[dict[str, object]],
    ) -> pl.Expr:
        """Build the panel from `cells`, apply `keep` per stratum,
        return `pl.col(stratify_column).is_in([surviving]) & static_scope`."""
        panel = self.panel_analysis.fn(cells, **self.panel_kwargs)
        try:
            stratify_idx = panel.stratify_by.index(self.stratify_column)
        except ValueError as exc:
            raise ValueError(
                f'DeferredScope: stratify_column {self.stratify_column!r} '
                f'not in panel.stratify_by {panel.stratify_by!r}',
            ) from exc
        surviving: list[object] = []
        for i in range(panel.n_strata):
            if self.keep(panel, i):
                surviving.append(panel.strata[i][stratify_idx])
        # Empty surviving → predicate evaluates to False for all
        # cells (is_in on an empty list). That's the correct
        # behaviour: zero strata pass.
        dynamic_expr = pl.col(self.stratify_column).is_in(surviving)
        if self.static_scope is None:
            return dynamic_expr
        return self.static_scope & dynamic_expr


def scope_from_panel(
    *,
    panel_analysis: Analysis[Mapping[str, object], StratumPanel],
    panel_kwargs: Mapping[str, object],
    keep: Callable[[StratumPanel, int], bool],
    stratify_column: str = 'env_name',
    static_scope: pl.Expr | None = None,
) -> DeferredScope:
    """Construct a `DeferredScope` for a bridge.

    Use case: scope a downstream bridge to strata where an
    upstream panel statistic crosses a threshold. The classic
    example is "restrict link bridge to envs where MC_disc and
    MC_raw decouple":

    ```python
    @claim_bridge(
        source=...,
        target=...,
        scope=scope_from_panel(
            panel_analysis=stratum_panel,
            panel_kwargs={
                'measurables': ('eval_best_burst_mean',
                                'eval_best_burst_raw_mean'),
                'treatment_arm': DDQN_ARM,
                'baseline_arm': VANILLA_ARM,
            },
            keep=lambda p, i: p.spearman_within[
                pair_key('eval_best_burst_mean',
                         'eval_best_burst_raw_mean')
            ][i] < 0.7,
            stratify_column='env_name',
            static_scope=DDQN_RELEVANT_SCOPE,
        ),
    )
    def my_bridge(...): ...
    ```

    Note: the `panel_analysis` runs with the full corpus + the
    `static_scope` filter applied BEFORE panel build (the runner
    pre-filters with static_scope). The `keep` predicate then
    decides which strata survive on top of that. This gives
    proper edge-conditioning: upstream-fixture's per-stratum data
    drives downstream-bridge's cell admission."""
    return DeferredScope(
        panel_analysis=panel_analysis,
        panel_kwargs=panel_kwargs,
        keep=keep,
        stratify_column=stratify_column,
        static_scope=static_scope,
    )


__all__ = ['DeferredScope', 'scope_from_panel']
