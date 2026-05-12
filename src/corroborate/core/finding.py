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
parent-package import → no circular import risk)."""
from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType
from typing import Protocol, runtime_checkable

from corroborate.bridge.bridge import Bridge
from corroborate.bridge.verdict import Verdict
from corroborate.graph.causal import (
    ClusterVerdict, PostEvalEntry, composed_verdict, evaluated_graph,
)


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
      `ImportError` at load, not silent drift.
    - `BLOCKED_ON: str` — non-empty when `EXPECTED` doesn't match
      the theoretical claim because data hasn't caught up. Names
      the gap (which corpora are missing, which CIs aren't tight
      yet, …). Empty when `EXPECTED` matches the theoretical
      claim directly. Renderer surfaces this so operators
      reading drift can tell "regression-now-occurring" from
      "permanent-state-pending-on-data."
    - `__name__: str` — Python identity (module's dotted path).

    Prose claim (the theoretical assertion) lives in the module
    docstring — not a Protocol field, because no framework
    operation needs to compare against it; the renderer quotes
    `__doc__` first line as context."""
    EXPECTED: ClusterVerdict
    BRIDGES: tuple[Bridge, ...]
    BLOCKED_ON: str
    __name__: str


# ============ Smoke runner ============


def run_finding(mod: ModuleType, *, repo_root: Path | None = None) -> bool:
    """Smoke runner for a single Finding module.

    Loads the parent hypothesis's snapshot, builds the post-eval
    graph, computes the cluster verdict, prints the drift signal,
    returns True iff the actual verdict matches `EXPECTED`.

    Per-finding `_main()` reduces to:
        `if __name__ == '__main__': run_finding(sys.modules[__name__])`

    Invocation: `python -m experiments.findings.<H>.finding_X`.
    The `-m` form sets `__spec__.name` to the dotted path; without
    it the script's `__name__` is `'__main__'` and parent-package
    discovery fails. Conventional parent-package discovery: the
    finding's dotted path's second-to-last segment IS the
    hypothesis short-name (`experiments.findings.<H>.finding_*` →
    `<H>`)."""
    if not isinstance(mod, Finding):
        raise TypeError(
            f'{mod.__name__!r} does not satisfy the Finding Protocol: '
            f'missing one of `EXPECTED: ClusterVerdict`, '
            f'`BRIDGES: tuple[Bridge, ...]` at module level.',
        )
    import importlib
    # When invoked via `python -m <dotted>`, `__name__` is
    # `'__main__'` but `__spec__.name` carries the dotted path.
    # Prefer the spec; fall back to __name__ for normal imports.
    spec = getattr(mod, '__spec__', None)
    dotted = spec.name if spec is not None else mod.__name__
    parts = dotted.split('.')
    if len(parts) < 2:
        raise ValueError(
            f'finding module {dotted!r} has no parent package; '
            f'invoke as `python -m experiments.findings.<H>.finding_X` '
            f'so `__spec__.name` carries the dotted path.',
        )
    parent_short = parts[-2]
    parent_module_name = '.'.join(parts[:-1])
    parent = importlib.import_module(parent_module_name)
    parent_bridges: tuple[Bridge, ...] = parent.BRIDGES

    root = repo_root or Path.cwd()
    run_json = root / 'experiments/findings' / f'{parent_short}.run.json'
    snapshot = json.loads(run_json.read_text())
    post_eval = {
        b['bridge_name']: PostEvalEntry(
            verdict=Verdict(b['verdict']),
            extent_hash=int(b['extent_hash']),
        )
        for b in snapshot['bridges']
    }
    g = evaluated_graph(parent_bridges, post_eval)
    verdict = composed_verdict(g, bridges=mod.BRIDGES)
    drift = verdict != mod.EXPECTED
    short_name = parts[-1]
    doc = (mod.__doc__ or '').strip().split('\n', 1)[0]

    print(f'{short_name}:')
    print(f'  doc:        {doc}')
    print(f'  verdict:    {verdict.value}')
    print(f'  expected:   {mod.EXPECTED.value}')
    print(f'  drift:      {drift}')
    if mod.BLOCKED_ON:
        print(f'  blocked-on: {mod.BLOCKED_ON}')
    return not drift
