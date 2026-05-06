"""Snapshot regression test: each committed `experiments/findings/
<short>.run.json` is loaded, its hypothesis re-run in a subprocess,
and per-bridge verdicts asserted to match the snapshot.

This is the sentinel against accidental bridge edits flipping
HELD ↔ NO_EFFECT. Effect-size *magnitudes* drift slightly across
runs (scipy / statsmodels float-precision noise) and are NOT
compared — only the verdict identity.

Skipped when the referenced cache parquet is absent locally and
cloud restore is disabled (CI-friendly when caches aren't shipped
with the repo).

Subprocess isolation is load-bearing: hypothesis modules
auto-register their measurables via `claim_bridge` decoration, and
two findings modules can author distinct `Measurable` instances with
the same name (e.g., both compute `reduce_axis('mc_return',
axis=-1, op='mean')`). Importing both into the same Python process
trips the registry's identity check. Re-running each in a fresh
subprocess matches the production invocation pattern (separate
`scripts/run_hypothesis.py` processes).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
FINDINGS_DIR = REPO_ROOT / 'experiments' / 'findings'


def _committed_reports() -> list[Path]:
    if not FINDINGS_DIR.is_dir():
        return []
    return sorted(p for p in FINDINGS_DIR.glob('*.run.json'))


# Subprocess child script: import the named hypothesis, re-run via
# runner.run (no cache write, no report write, no cloud restore),
# print verdicts as JSON to stdout. Allows clean module-registration
# state per test case.
_CHILD_SCRIPT = r'''
import json
import sys

from corroborate.runner.runner import _default_cache_path, run


def _main(module_name: str) -> int:
    import importlib
    module = importlib.import_module(module_name)
    cache_path = _default_cache_path(module)
    if not cache_path.exists():
        sys.stdout.write(json.dumps({'_skip': str(cache_path)}))
        return 0
    fresh = run(
        module_name,
        use_cache=True,
        write_cache=False,
        restore_from_cloud=False,
        write_report=False,
    )
    payload = {name: ev.verdict.value for name, ev in fresh.items()}
    sys.stdout.write(json.dumps(payload))
    return 0


if __name__ == '__main__':
    sys.exit(_main(sys.argv[1]))
'''


def _run_in_subprocess(module_name: str) -> dict[str, str] | None:
    """Run the hypothesis in a fresh Python subprocess.
    Returns dict[bridge_name, verdict_value] on success, or None
    when the cache is absent (skip signal). Raises on subprocess
    failure."""
    env = os.environ.copy()
    env.setdefault('JAX_PLATFORMS', 'cpu')
    env['PYTHONPATH'] = str(REPO_ROOT) + ':' + env.get('PYTHONPATH', '')
    proc = subprocess.run(
        [sys.executable, '-c', _CHILD_SCRIPT, module_name],
        capture_output=True, text=True, env=env, cwd=str(REPO_ROOT),
        check=False, timeout=300,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f'subprocess for {module_name!r} exited {proc.returncode}; '
            f'stderr:\n{proc.stderr}',
        )
    payload = json.loads(proc.stdout)
    if isinstance(payload, dict) and '_skip' in payload:
        return None
    return payload


@pytest.mark.parametrize('report_path', _committed_reports(),
                         ids=lambda p: p.stem)
def test_committed_report_verdicts_unchanged(report_path: Path) -> None:
    """For each `experiments/findings/<short>.run.json`: load,
    re-run the named hypothesis (in a subprocess for module-registry
    isolation), assert per-bridge verdict matches.

    Skipped when the referenced cache parquet is absent."""
    snapshot = json.loads(report_path.read_text())
    module_name = snapshot['hypothesis_module']
    fresh_verdicts = _run_in_subprocess(module_name)
    if fresh_verdicts is None:
        pytest.skip(
            f'cache for {module_name!r} absent — snapshot regression '
            f'needs a built cache (commit caches OR restore from cloud)',
        )

    expected: dict[str, str] = {
        entry['bridge_name']: entry['verdict']
        for entry in snapshot['bridges']
    }
    expected_errored = {
        entry['bridge_name']: entry['error_type']
        for entry in snapshot.get('errored_bridges', [])
    }

    # Every bridge in the snapshot must be present + same verdict.
    mismatches: list[str] = []
    for bridge_name, expected_verdict in expected.items():
        if bridge_name not in fresh_verdicts:
            if bridge_name in expected_errored:
                # Bridge was errored in snapshot. If it now succeeds,
                # the snapshot is stale (real progress). The sentinel
                # doesn't fail on this — it's improvement, not drift.
                continue
            mismatches.append(
                f'  {bridge_name}: snapshot={expected_verdict}, '
                f'now=MISSING (bridge dropped or renamed)',
            )
            continue
        actual = fresh_verdicts[bridge_name]
        if actual != expected_verdict:
            mismatches.append(
                f'  {bridge_name}: snapshot={expected_verdict}, now={actual}',
            )
    new_bridges = set(fresh_verdicts) - set(expected) - set(expected_errored)
    if new_bridges:
        # Newly-added bridges aren't a regression by themselves, but
        # the snapshot needs a refresh to capture them. Surface as
        # an INFO message in the failure (xfail-style would be
        # nicer but pytest doesn't allow soft-fail with info easily).
        mismatches.append(
            f'  [info] {len(new_bridges)} bridge(s) in fresh run not in '
            f'snapshot: {sorted(new_bridges)} — re-generate the report '
            f'to update the baseline.',
        )

    assert not mismatches, (
        f'Verdict landscape drifted from snapshot {report_path.name}:\n'
        + '\n'.join(mismatches)
        + '\n\nIf the change is intentional, regenerate the report:\n'
        + f'  PYTHONPATH=. uv run python scripts/run_hypothesis.py '
        + f'{module_name} --no-restore'
    )


def test_findings_dir_present() -> None:
    """Sanity: the findings dir exists. If the dir is missing the
    parametrize above silently produces zero tests; this catches that."""
    assert FINDINGS_DIR.is_dir(), f'{FINDINGS_DIR} missing'
