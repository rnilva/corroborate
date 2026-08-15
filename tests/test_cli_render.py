"""CLI-level tests for `corroborate hypothesis ... --render`.

Mirrors the `tests/test_cli_preflight_gating.py` pattern: drive
`corroborate.cli.hypothesis.main` with a real dotted module path,
patch `run` so no data pipeline executes, and assert on the
rendered artifact / exit code. The render content itself is
covered by `tests/test_graph_render.py`; here we cover the wiring
— one command in, one figure out, clean failure when the run
produced nothing renderable.
"""
from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from unittest.mock import patch

from corroborate.bridge.bridge import Bridge, BridgeEvaluation
from corroborate.bridge.verdict import Verdict
from tests.probes.stub_hypothesis_with_bridge import BRIDGES

_HYP_PRINT_PATH = 'corroborate.cli.hypothesis._print_verdicts'
_MODULE_WITH_BRIDGE = 'tests.probes.stub_hypothesis_with_bridge'


def _evaluation_for(bridge: Bridge) -> BridgeEvaluation:
    return BridgeEvaluation(
        bridge_name=bridge.name,
        verdict=Verdict.HELD,
        analysis_results=MappingProxyType({}),
        n_cells_in_scope=8,
        source_name=bridge.source_name,
        target_name=bridge.target_name,
        extent_hash=99,
    )


def test_render_flag_writes_figure_from_run(tmp_path: Path) -> None:
    """`--render out.svg` after a run writes the evidence figure with
    every display decision defaulted from the run: edge label from
    the bridge name, verdict styling from the evaluation, title from
    the hypothesis module name."""
    bridge = BRIDGES[0]
    results = {bridge.name: _evaluation_for(bridge)}
    out = tmp_path / 'evidence.svg'
    with patch('corroborate.cli.hypothesis.run', return_value=results), \
         patch(_HYP_PRINT_PATH):
        from corroborate.cli.hypothesis import main
        code = main([
            _MODULE_WITH_BRIDGE,
            '--no-report', '--no-cache',
            '--render', str(out),
        ])
    assert code == 0
    svg = out.read_text(encoding='utf-8')
    assert svg.startswith('<?xml')
    assert 'return edge' in svg  # edge label defaulted from bridge name
    assert 'HELD' in svg  # verdict styling from the evaluation
    assert _MODULE_WITH_BRIDGE in svg  # title from the module name


def test_render_fails_cleanly_with_no_bridges(tmp_path: Path) -> None:
    """A hypothesis module with zero bridges has nothing to draw —
    exit 1 with a message, no file written."""
    out = tmp_path / 'evidence.svg'
    with patch('corroborate.cli.hypothesis.run', return_value={}), \
         patch(_HYP_PRINT_PATH):
        from corroborate.cli.hypothesis import main
        code = main([
            'tests.probes.stub_hypothesis',
            '--no-report', '--no-cache',
            '--render', str(out),
        ])
    assert code == 1
    assert not out.exists()


def test_render_fails_cleanly_on_unsupported_suffix(
    tmp_path: Path,
) -> None:
    bridge = BRIDGES[0]
    results = {bridge.name: _evaluation_for(bridge)}
    out = tmp_path / 'evidence.png'
    with patch('corroborate.cli.hypothesis.run', return_value=results), \
         patch(_HYP_PRINT_PATH):
        from corroborate.cli.hypothesis import main
        code = main([
            _MODULE_WITH_BRIDGE,
            '--no-report', '--no-cache',
            '--render', str(out),
        ])
    assert code == 1
    assert not out.exists()
