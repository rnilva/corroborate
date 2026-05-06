"""`tautology_audit` — three-check verdict on candidate mediators.

The shape FINDINGS revisions 3 and 5 consume: for each candidate
mediator, three independent checks against the corpus determine
whether it's a *real* mediator or a false positive:

1. **Structural**: reads-set jaccard with the outcome's source
   columns. High overlap → outcome-tautological (the mediator
   is largely a restatement of the outcome).
2. **Empirical (HP-R²)**: per-HP-axis OLS R² of the mediator on
   each HP. High R² on any axis → the mediator is essentially
   deterministic in that HP and shadows whatever the HP itself
   does to outcome.
3. **Empirical (stratified ρ)**: Spearman ρ(mediator, outcome)
   pooled within HP strata. |ρ| below threshold + p above α →
   the mediator's apparent outcome correlation is HP-mediated
   (no residual signal once you pool within an HP regime).

Returns one tag per measurable: `clean` | `OUTCOME` | `HP` |
`SHADOW` (or any union when a mediator fails multiple checks).
Bridges consume the per-measurable verdict and assert claims
about which mediator candidates survive the audit and which
get unmasked as artifacts of HP shadow / outcome leakage.

Wraps `corroborate.redundancy_check.audit_mediator_panel`. The
analysis converts the cell collection to `RunRow` objects at
the boundary; the existing audit is preserved unchanged."""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from corroborate.bridge.analysis import analysis
from corroborate.measurables.redundancy_check import (
    TautologyReport, audit_mediator_panel,
)
from corroborate.corpus.schema import RunRow


@dataclass(frozen=True, slots=True)
class _MeasurableSpec:
    """Minimal `_NameReadsProtocol` shape — what
    `audit_mediator_panel` requires per mediator."""
    name: str
    reads: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AuditResult:
    """Output of `tautology_audit`: per-measurable
    `TautologyReport` plus a `by_name` lookup. Bridges consume
    the lookup to make per-measurable claims."""
    reports: tuple[TautologyReport, ...]

    def by_name(self, name: str) -> TautologyReport | None:
        """Return the report for `name`, or None if not in the
        panel. Bridges typically call this with the measurable
        their claim is about."""
        for r in self.reports:
            if r.measurable_name == name:
                return r
        return None

    @property
    def clean_names(self) -> tuple[str, ...]:
        """Names of mediators that passed all three checks
        (structural + HP-R² + stratified-ρ)."""
        return tuple(r.measurable_name for r in self.reports if r.is_clean)


@analysis
def tautology_audit(
    cells: Iterable[Mapping[str, object]],
    *,
    measurables: Sequence[Mapping[str, object]],
    outcome_path: str,
    outcome_reads: tuple[str, ...],
    hp_axes: tuple[str, ...],
    hp_stratum_axis: str | None = None,
    arm_filter: str | None = None,
    arm_field: str = 'arm_key',
    outcome_jaccard_threshold: float = 0.5,
    hp_r_squared_threshold: float = 0.95,
    stratified_rho_threshold: float = 0.1,
    stratified_alpha: float = 0.05,
    mediator_path_for: Mapping[str, str] | None = None,
) -> AuditResult:
    """Three-check tautology audit on a panel of mediators.

    `measurables` is a sequence of `{name: str, reads:
    tuple[str, ...]}` dicts (one entry per mediator). Each
    measurable's `reads` names the trace-column inputs from
    which the mediator is computed; the audit's structural
    check compares this against `outcome_reads`.

    `arm_filter`, when supplied, restricts the audit to one
    arm — the audit then reports mediator quality conditional
    on that arm's data only.

    `mediator_path_for` is an optional `name → cell-dict-key`
    override. By default `audit_mediator_panel` looks up each
    mediator's value at `RunRow.measurements[f'mediator.{name}']`
    (or at `name` when `name` already contains a dot). When the
    cells expose mediator scalars under a different key
    convention — e.g. bare names like `target_staleness_late`
    rather than `mediator.target_staleness_late` — pass an
    explicit map so the audit reads them without forcing the
    caller to rename keys. Forwarded verbatim to
    `audit_mediator_panel`."""
    spec_list: list[_MeasurableSpec] = []
    for m in measurables:
        name_v = m.get('name')
        reads_v = m.get('reads')
        if not isinstance(name_v, str):
            raise TypeError(
                f'measurable spec missing string `name`: {m!r}',
            )
        if (
            not isinstance(reads_v, tuple)
            or not all(isinstance(s, str) for s in reads_v)
        ):
            raise TypeError(
                f'measurable spec {name_v!r} `reads` must be '
                f'tuple[str, ...]; got {reads_v!r}',
            )
        spec_list.append(_MeasurableSpec(name=name_v, reads=reads_v))

    runs: list[RunRow] = []
    for cell in cells:
        if arm_filter is not None and cell.get(arm_field) != arm_filter:
            continue
        runs.append(RunRow.from_row_dict(cell))

    reports = audit_mediator_panel(
        spec_list,
        runs,
        outcome_reads=frozenset(outcome_reads),
        hp_axes=hp_axes,
        outcome_path=outcome_path,
        hp_stratum_axis=hp_stratum_axis,
        mediator_path_for=mediator_path_for,
        outcome_jaccard_threshold=outcome_jaccard_threshold,
        hp_r_squared_threshold=hp_r_squared_threshold,
        stratified_rho_threshold=stratified_rho_threshold,
        stratified_alpha=stratified_alpha,
    )
    return AuditResult(reports=reports)


__all__ = ['AuditResult', 'tautology_audit']
