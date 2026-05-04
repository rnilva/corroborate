"""corroborate — executable scientific claims with intervention,
falsification, and causal discovery as composable capabilities.

The top-level surface re-exports the four names users invoke
manually at the top of every file: the three framework
decorators and the trace context manager. Everything else lives
in subpackages and is imported via the explicit qualified path:

    from corroborate import claim, claim_bridge, measurable, trace_context
    from corroborate.core import Hypothesis, Intervention, canonical_str
    from corroborate.bridge import Verdict
    from corroborate.measurables import Measurable, register
    from corroborate.corpus import RunRow, TraceRow
    from corroborate.runner import run_module, sweep
    from corroborate.stats import hedges_g_paired
    from corroborate.graph import CausalGraph
"""
from corroborate.bridge.bridge import claim_bridge
from corroborate.core.claim import claim, trace_context
from corroborate.measurables.measurable import measurable

__version__ = '0.0.1'

__all__ = [
    'claim',
    'claim_bridge',
    'measurable',
    'trace_context',
]
