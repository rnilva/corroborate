"""Structural protocols for analysis Result dataclass shapes.

The framework's analyses each return a frozen dataclass shaped
to that analysis's verdict, but bridge predicates and helpers
often want to operate on the SHARED fields across families
(`n_pairs` + `pair_by` for paired analyses; `n_strata` +
`stratify_by` for stratified ones). Two Protocols let pyright
type those helpers without forcing a base class on every Result
or falling back to `Any` / `object`.

The pattern mirrors `StratumGProtocol[K]` in
`stats/meta_regression.py:42`. Per CLAUDE.md, read-only
properties (not bare attrs) are required to match frozen
dataclass fields — writable Protocol fields don't structurally
match immutable concrete fields.
"""
from __future__ import annotations

from typing import Protocol


class PairedEffectResult(Protocol):
    """Common surface across paired-shape analyses.

    `PairedGResult`, `BootstrapPairedGResult`, `CliffDeltaPairedResult`,
    and other paired primitives structurally satisfy this.
    Suitable for typed helpers like `verdict_from_paired(r:
    PairedEffectResult) -> Verdict` that read only the shared
    identifying surface without depending on a specific Result.
    """
    @property
    def n_pairs(self) -> int: ...
    @property
    def measurable(self) -> str: ...
    @property
    def treatment_arm(self) -> str: ...
    @property
    def baseline_arm(self) -> str: ...
    @property
    def pair_by(self) -> tuple[str, ...]: ...


class StratifiedResult(Protocol):
    """Common surface across stratified-shape analyses.

    `StratifiedArmDiffPooledResult`, `StratumEffectPanel`,
    `MetaRegressionResult`, and other primitives that aggregate
    over strata structurally satisfy this. Bridges that
    introspect the stratification axis (env, config, burst)
    can type-check against this without binding to a concrete
    Result class.
    """
    @property
    def n_strata(self) -> int: ...
    @property
    def measurable(self) -> str: ...
    @property
    def treatment_arm(self) -> str: ...
    @property
    def baseline_arm(self) -> str: ...
    @property
    def stratify_by(self) -> tuple[str, ...]: ...
