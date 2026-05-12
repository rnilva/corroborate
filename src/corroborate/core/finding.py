"""Finding — a typed subgraph of a Hypothesis's evaluated graph.

A Finding identifies a subset of bridges from a parent Hypothesis
that together support a single asserted claim. The Protocol
mirrors `Hypothesis` (in `corroborate.core.hypothesis`): module-
level attributes form a declarative claim; the framework's
evaluator computes the verdict via `composed_verdict`.

Parallel to Hypothesis:
- `Hypothesis.BRIDGES` → authored topology (the full graph).
- `Finding.BRIDGES` → subgraph specification (a subset of the
  parent's bridges; defines which edges the finding claims about).

Composition is uniform across subgraph topologies. A cluster
(parallel edges between `(s, t)`), a chain (serial edges through
intermediate nodes), and an envelope (parallel edges on `(s, t)`
with disjoint scopes) all compose the same way: every named
bridge must admit on the post-evaluated parent graph for the
Finding's verdict to be SUPPORTED.

Findings are discovered through `Hypothesis.FINDINGS`. No reverse
pointer needed on the Finding — the parent owns the relationship,
so each Finding module imports only the bridges it cites (no
parent-package import → no circular import risk).

The framework discovers and evaluates Findings via the
`run_hypothesis.py` rollup; per-finding debug is `python
scripts/run_hypothesis.py <hypothesis_module> --filter
<bridge_name>` against the parent. There's no per-finding
script-mode entrypoint — the rollup is the single truth path."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict


@runtime_checkable
class Finding(Protocol):
    """The framework's typed Finding contract.

    Conforming objects (typically modules under
    `experiments/findings/<hypothesis>/finding_*.py`) expose:

    - `EXPECTED: ClusterVerdict` — verdict the author asserts;
      drift detection compares against the actual `composed_verdict`
      result. **`EXPECTED` pins the EMPIRICAL state, not the
      theoretical claim.** If the current cache can't decisively
      land the theoretical claim's verdict (data still landing,
      bridges authored but corpus thin, …), the author pins
      `EXPECTED` to whatever the framework actually computes
      *today* and names the gap in `BLOCKED_ON`. Drift then fires
      iff state CHANGES — improvement OR regression — both of
      which warrant operator attention.
    - `BRIDGES: tuple[Bridge, ...]` — the subgraph specification:
      which bridges of the parent `Hypothesis.BRIDGES` are in this
      finding. Imported by Python name so a bridge rename is an
      `ImportError` at load, not silent drift. The subset
      invariant (`f.BRIDGES ⊆ h.BRIDGES`) is enforced at
      `_validate_hypothesis` time.
    - `BLOCKED_ON: str | None` — non-empty (None vs str) reflects
      whether `EXPECTED` matches the theoretical claim or pins a
      sub-optimal empirical state pending data. When non-None the
      string names the gap (which corpora missing, what would
      unblock). Renderer surfaces this so operators reading drift
      can tell "regression-now-occurring" from "permanent-state-
      pending-on-data." A non-None `BLOCKED_ON` paired with a
      terminal `EXPECTED` (SUPPORTED / REFUTED) is author
      contradiction — the renderer surfaces a warning.
    - `__name__: str` — Python identity (module's dotted path).

    Prose claim (the theoretical assertion) lives in the module
    docstring — not a Protocol field, because no framework
    operation needs to compare against it; the renderer quotes
    `__doc__` first line as context."""
    EXPECTED: ClusterVerdict
    BRIDGES: tuple[Bridge, ...]
    BLOCKED_ON: str | None
    __name__: str
