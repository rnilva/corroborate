"""Audit the most recent run.json: classify each verdict.

For each bridge, surface:
- HELD / NO_EFFECT — final verdicts
- POWER_INSUFFICIENT subcategories:
  * scope-empty (n_in_scope=0): bridge's scope filter found no
    matching cells; usually means traces missing on the scoped
    corpus and the trace-dependent measurable couldn't be
    computed.
  * data-pair-empty: scope had cells but pair_by yielded zero
    paired (treatment, baseline) tuples.
  * assumption-violated: analysis returned but its assumption
    (linear-mediation `in_unit_interval`, regression slope finite,
    …) is violated; bridge body returns POWER_INSUFFICIENT to
    flag the violation.
  * underpowered: n_pairs below the bridge's `n_pairs_floor`.
  * analysis-specific: meta-regression / dowhy result with no
    n_pairs fingerprint — typically means coefficient or refuter
    didn't pass thresholds.

Reports actionable: scope-empty + data-pair-empty often resolve
with `--restore` (download traces from R2). Assumption-violated
verdicts are honest; the framework is refusing to call HELD when
its assumptions don't hold.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def categorize(b: dict[str, object]) -> str:
    name = str(b.get('bridge_name', '?'))
    verdict = str(b.get('verdict', '?'))
    n = int(b.get('n_cells_in_scope', 0) or 0)
    if verdict in ('held', 'no_effect'):
        return verdict
    n_pairs: int | None = None
    in_unit_interval: bool | None = None
    proportion: float | None = None
    for k, v in (b.get('analysis_results') or {}).items():
        del k
        if isinstance(v, dict):
            n_pairs = v.get('n_pairs', n_pairs)
            in_unit_interval = v.get('in_unit_interval', in_unit_interval)
            proportion = v.get('proportion', proportion)
    del name
    if verdict == 'power_insufficient':
        if n == 0:
            return 'pwr/scope-empty'
        if n_pairs == 0:
            return 'pwr/data-pair-empty'
        if in_unit_interval is False:
            return f'pwr/assumption-violated (proportion={proportion:.3f})'
        if n_pairs is not None:
            return f'pwr/underpowered (n_pairs={n_pairs})'
        return 'pwr/analysis-specific'
    return verdict


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1
                else 'experiments/findings/ddqn_universe.run.json')
    if not path.exists():
        print(f'no run.json at {path}', file=sys.stderr)
        sys.exit(1)
    r = json.loads(path.read_text())
    bridges = r.get('bridges') or []
    cats: dict[str, list[str]] = {}
    print(f'{"bridge":<55} {"n_in_scope":>10} {"category":<40}')
    print('-' * 110)
    for b in bridges:
        cat = categorize(b)
        name = str(b.get('bridge_name', '?'))
        n = int(b.get('n_cells_in_scope', 0) or 0)
        print(f'{name:<55} {n:>10} {cat:<40}')
        cats.setdefault(cat, []).append(name)
    print()
    print('=== Summary ===')
    for cat in sorted(cats):
        print(f'  {cat:<45} {len(cats[cat])}')


if __name__ == '__main__':
    main()
