"""Deterministic vector renderers for Corroborate's two graph layers.

The framework carries two graphs with different meanings:

* a :class:`~corroborate.graph.computation.ComputationGraph` records
  observed value flow between ``@claim`` calls; and
* an evaluated evidence graph records authored scientific edges and
  their exact bridge verdicts.

This module deliberately keeps those surfaces separate.  Both can be
rendered as DOT or as dependency-free SVG.  The SVG layout is intended
for compact findings and intervention fragments rather than arbitrary
large network visualisation.

:func:`render_evidence` is the one-call entry: it consumes exactly what
a hypothesis run produces (the module's bridges plus the run's verdict
mapping) and defaults every display decision from the run itself.  The
``evidence_graph_to_*`` / ``computation_graph_to_*`` functions remain
the fine-control surface for publication figures.
"""
from __future__ import annotations

import textwrap
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from html import escape as xml_escape
from pathlib import Path
from typing import Literal

from corroborate.bridge.bridge import Bridge, BridgeEvaluation
from corroborate.bridge.verdict import Verdict
from corroborate.core.intervention import DoEffect
from corroborate.graph.causal import (
    ClusterVerdict,
    Direction,
    PostEvalEntry,
    Tier,
    composed_verdict,
    evaluated_graph,
)
from corroborate.graph.computation import ComputationGraph


@dataclass(frozen=True, slots=True)
class _RenderNode:
    key: str
    label: str
    kind: Literal['ordinary', 'intervention'] = 'ordinary'


@dataclass(frozen=True, slots=True)
class _RenderEdge:
    source: str
    target: str
    name: str
    label_lines: tuple[str, ...]
    verdict: Verdict | None = None
    tier: Tier | None = None
    direction: Direction | None = None


@dataclass(frozen=True, slots=True)
class _RenderSpec:
    nodes: tuple[_RenderNode, ...]
    edges: tuple[_RenderEdge, ...]
    title: str | None = None
    aggregate_verdict: ClusterVerdict | None = None


_VERDICT_STYLES: Mapping[Verdict | None, tuple[str, str, str]] = {
    Verdict.HELD: ('#087f5b', 'solid', '#e6fcf5'),
    Verdict.HELD_WITH_SCOPE_FLAG: ('#1971c2', 'solid', '#e7f5ff'),
    Verdict.NO_EFFECT: ('#c92a2a', 'solid', '#fff5f5'),
    Verdict.POWER_INSUFFICIENT: ('#d97706', 'dashed', '#fff9db'),
    Verdict.INVARIANT_VIOLATION: ('#7b2cbf', 'dotted', '#f8f0fc'),
    Verdict.INADMISSIBLE: ('#6b7280', 'dotted', '#f3f4f6'),
    None: ('#64748b', 'dashed', '#f8fafc'),
}


def _shorten(text: str, *, limit: int = 42) -> str:
    if len(text) <= limit:
        return text
    return f'{text[:limit - 1]}…'


def _default_node_label(key: str) -> str:
    if key.startswith('do(') and key.endswith(')'):
        arms = key[3:-1].split('|')
        compact = ' ↔ '.join(_shorten(arm, limit=22) for arm in arms)
        return f'do({compact})'
    return _shorten(key)


def _direction_label(direction: Direction | None) -> str:
    return {
        Direction.DIRECT: '+',
        Direction.INVERSE: '−',
        Direction.AT_MOST: '≤',
        Direction.AT_LEAST: '≥',
        None: '',
    }[direction]


def _tier_label(tier: Tier | None) -> str:
    if tier is None:
        return 'structural'
    return {
        Tier.INTERVENTIONAL: 'interventional',
        Tier.ASSOCIATIONAL: 'associational',
        Tier.INVARIANT: 'invariant',
    }[tier]


def _verdict_label(verdict: Verdict | None) -> str:
    if verdict is None:
        return 'NOT EVALUATED'
    return verdict.value.replace('_', ' ').upper()


def _aggregate_verdict(
    bridge_tuple: tuple[Bridge, ...],
    evaluations: Mapping[str, BridgeEvaluation],
) -> ClusterVerdict | None:
    """Compose the rendered bridge set's verdict from its inputs.

    The badge is framework-derived, never caller-supplied: the same
    ``composed_verdict`` a Finding walk uses, applied to exactly the
    bridges being drawn.  Rendering a subset (a Finding's ``BRIDGES``)
    therefore shows that subset's composed verdict, and an unevaluated
    graph honestly shows UNDERPOWERED."""
    if not bridge_tuple:
        return None
    post_eval = {
        name: PostEvalEntry(
            verdict=evaluation.verdict,
            extent_hash=evaluation.extent_hash,
        )
        for name, evaluation in evaluations.items()
    }
    return composed_verdict(
        evaluated_graph(bridge_tuple, post_eval),
        bridges=bridge_tuple,
    )


def _evidence_spec(
    bridges: Iterable[Bridge],
    evaluations: Mapping[str, BridgeEvaluation],
    *,
    node_labels: Mapping[str, str] | None,
    edge_labels: Mapping[str, str] | None,
    edge_summaries: Mapping[str, str] | None,
    title: str | None,
) -> _RenderSpec:
    bridge_tuple = tuple(bridges)
    by_name: dict[str, Bridge] = {}
    for bridge in bridge_tuple:
        if bridge.name in by_name:
            raise ValueError(
                f'duplicate bridge name {bridge.name!r}; rendering by '
                'scientific edge identity would be ambiguous',
            )
        by_name[bridge.name] = bridge

    selected = bridge_tuple
    labels = node_labels or {}
    display_edge_labels = edge_labels or {}
    summaries = edge_summaries or {}
    nodes: dict[str, _RenderNode] = {}
    edges: list[_RenderEdge] = []
    for bridge in selected:
        source = bridge.source_name
        target = bridge.target_name
        source_kind: Literal['ordinary', 'intervention'] = (
            'intervention' if isinstance(bridge.source, DoEffect)
            else 'ordinary'
        )
        nodes[source] = _RenderNode(
            key=source,
            label=labels.get(source, _default_node_label(source)),
            kind=source_kind,
        )
        nodes[target] = _RenderNode(
            key=target,
            label=labels.get(target, _default_node_label(target)),
        )

        evaluation = evaluations.get(bridge.name)
        verdict = evaluation.verdict if evaluation is not None else None
        tier = (
            Tier.INTERVENTIONAL
            if isinstance(bridge.source, DoEffect)
            else bridge.tier
        )
        direction = bridge.direction
        descriptor = _tier_label(tier)
        direction_text = _direction_label(direction)
        if direction_text:
            descriptor = f'{descriptor} · {direction_text}'
        lines = [
            display_edge_labels.get(
                bridge.name,
                bridge.name.replace('_', ' '),
            ),
            descriptor,
            _verdict_label(verdict),
        ]
        summary = summaries.get(bridge.name)
        if summary:
            lines.append(summary)
        elif evaluation is not None and evaluation.n_cells_in_scope >= 0:
            lines.append(f'extent: {evaluation.n_cells_in_scope} cells')
        edges.append(_RenderEdge(
            source=source,
            target=target,
            name=bridge.name,
            label_lines=tuple(lines),
            verdict=verdict,
            tier=tier,
            direction=direction,
        ))

    return _RenderSpec(
        nodes=tuple(sorted(nodes.values(), key=lambda node: node.key)),
        edges=tuple(sorted(
            edges,
            key=lambda edge: (edge.source, edge.target, edge.name),
        )),
        title=title,
        aggregate_verdict=_aggregate_verdict(bridge_tuple, evaluations),
    )


def _computation_spec(
    graph: ComputationGraph,
    *,
    node_labels: Mapping[str, str] | None,
    title: str | None,
) -> _RenderSpec:
    labels = node_labels or {}
    nodes = tuple(
        _RenderNode(
            key=key,
            label=labels.get(key, _default_node_label(key)),
        )
        for key in sorted(graph.nodes)
    )
    edges: list[_RenderEdge] = []
    for index, edge in enumerate(sorted(
        graph.edges,
        key=lambda item: (
            item.source,
            item.target,
            item.metadata.reader_arg,
            item.metadata.source_path,
        ),
    )):
        source_path = (
            f'.{edge.metadata.source_path}'
            if edge.metadata.source_path else '<return>'
        )
        edges.append(_RenderEdge(
            source=edge.source,
            target=edge.target,
            name=f'computation-{index}',
            label_lines=(
                f'.{edge.metadata.reader_arg} ← {source_path}',
                'observed identity flow',
            ),
        ))
    return _RenderSpec(nodes=nodes, edges=tuple(edges), title=title)


def _dot_quote(text: str) -> str:
    escaped = (
        text.replace('\\', '\\\\')
        .replace('"', '\\"')
        .replace('\n', '\\n')
    )
    return f'"{escaped}"'


def _spec_to_dot(spec: _RenderSpec) -> str:
    node_ids = {
        node.key: f'n{index}' for index, node in enumerate(spec.nodes)
    }
    lines = [
        'digraph corroborate {',
        '  graph [rankdir=LR, bgcolor="transparent", pad=0.25, '
        'nodesep=0.55, ranksep=1.15];',
        '  node [shape=box, style="rounded,filled", fontname="Arial", '
        'fontsize=11, color="#94a3b8", fillcolor="#ffffff", margin="0.16,0.10"];',
        '  edge [fontname="Arial", fontsize=9, arrowsize=0.8, penwidth=1.8];',
    ]
    if spec.title:
        title = spec.title
        if spec.aggregate_verdict is not None:
            title = f'{title} — {spec.aggregate_verdict.value.upper()}'
        lines.append(f'  label={_dot_quote(title)}; labelloc="t"; fontsize=16;')
    for node in spec.nodes:
        attributes = [f'label={_dot_quote(node.label)}']
        if node.kind == 'intervention':
            attributes.extend([
                'shape=hexagon',
                'color="#2563eb"',
                'fillcolor="#eff6ff"',
            ])
        lines.append(
            f'  {node_ids[node.key]} [{", ".join(attributes)}];',
        )
    for edge in spec.edges:
        colour, line_style, _ = _VERDICT_STYLES[edge.verdict]
        # Tier and verdict are orthogonal.  Positive associational evidence
        # must remain visually distinguishable from an interventional edge.
        # Dotted blocker/invalid styles take precedence.
        if edge.tier is Tier.ASSOCIATIONAL and line_style == 'solid':
            line_style = 'dashed'
        dot_style = {
            'solid': 'solid',
            'dashed': 'dashed',
            'dotted': 'dotted',
        }[line_style]
        label = '\n'.join(edge.label_lines)
        lines.append(
            f'  {node_ids[edge.source]} -> {node_ids[edge.target]} '
            f'[label={_dot_quote(label)}, color={_dot_quote(colour)}, '
            f'fontcolor={_dot_quote(colour)}, style={dot_style}, '
            f'tooltip={_dot_quote(edge.name)}];',
        )
    lines.append('}')
    return '\n'.join(lines) + '\n'


def evidence_graph_to_dot(
    bridges: Iterable[Bridge],
    evaluations: Mapping[str, BridgeEvaluation],
    *,
    node_labels: Mapping[str, str] | None = None,
    edge_labels: Mapping[str, str] | None = None,
    edge_summaries: Mapping[str, str] | None = None,
    title: str | None = None,
) -> str:
    """Render exact evaluated bridge edges as deterministic DOT.

    ``bridges`` is the selection surface: pass a subset (a Finding's
    ``BRIDGES``) to draw a subgraph — no separate name filter exists.
    Verdict styling and the aggregate-verdict badge are derived from
    ``evaluations``, never caller-supplied.  The surviving display
    sidecars each carry information the run does not possess:
    ``node_labels`` replaces code-facing measurable keys with public
    copy; ``edge_labels`` replaces long Python bridge identifiers with
    concise public copy; ``edge_summaries`` supplies a per-bridge
    headline estimate, because the graph model intentionally does not
    guess which field of an arbitrary analysis result is scientific;
    ``title`` is a custom caption overriding the run-derived default.
    """
    return _spec_to_dot(_evidence_spec(
        bridges,
        evaluations,
        node_labels=node_labels,
        edge_labels=edge_labels,
        edge_summaries=edge_summaries,
        title=title,
    ))


def computation_graph_to_dot(
    graph: ComputationGraph,
    *,
    node_labels: Mapping[str, str] | None = None,
    title: str | None = None,
) -> str:
    """Render an observed ``@claim`` identity-flow graph as DOT."""
    return _spec_to_dot(_computation_spec(
        graph,
        node_labels=node_labels,
        title=title,
    ))


def _wrap_label(label: str, *, width: int) -> tuple[str, ...]:
    wrapped = textwrap.wrap(
        label,
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )
    if not wrapped:
        return ('',)
    if len(wrapped) <= 3:
        return tuple(wrapped)
    return (*wrapped[:2], _shorten(' '.join(wrapped[2:]), limit=width))


def _node_ranks(spec: _RenderSpec) -> Mapping[str, int]:
    """Longest-path ranks for a DAG, with a deterministic cycle fallback."""
    keys = tuple(node.key for node in spec.nodes)
    outgoing: dict[str, set[str]] = {key: set() for key in keys}
    indegree = {key: 0 for key in keys}
    for edge in spec.edges:
        if edge.source == edge.target or edge.target in outgoing[edge.source]:
            continue
        outgoing[edge.source].add(edge.target)
        indegree[edge.target] += 1

    ranks = {key: 0 for key in keys}
    ready = sorted(key for key, degree in indegree.items() if degree == 0)
    visited: set[str] = set()
    while ready:
        current = ready.pop(0)
        visited.add(current)
        for target in sorted(outgoing[current]):
            ranks[target] = max(ranks[target], ranks[current] + 1)
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
                ready.sort()

    # A cyclic component has no topological rank.  Place its remaining
    # nodes in deterministic successive columns; back-edges remain visible.
    next_rank = max(ranks.values(), default=0)
    for key in sorted(set(keys) - visited):
        next_rank += 1
        ranks[key] = next_rank
    return ranks


def _svg_multiline_text(
    lines: tuple[str, ...],
    *,
    x: float,
    y: float,
    line_height: float,
    css_class: str,
    anchor: Literal['start', 'middle', 'end'] = 'middle',
) -> str:
    spans = []
    start_y = y - line_height * (len(lines) - 1) / 2
    for index, line in enumerate(lines):
        spans.append(
            f'<tspan x="{x:.1f}" y="{start_y + index * line_height:.1f}">'
            f'{xml_escape(line)}</tspan>',
        )
    return (
        f'<text class="{css_class}" text-anchor="{anchor}">'
        f'{"".join(spans)}</text>'
    )


def _spec_to_svg(spec: _RenderSpec) -> str:
    node_width = 244.0
    node_height = 76.0
    # Reserve a real annotation lane: evidence edges commonly carry a label,
    # tier, verdict, and one estimate, and must not be drawn through nodes.
    horizontal_gap = 520.0
    vertical_gap = 180.0
    margin_x = 62.0
    title_height = 86.0 if (spec.title or spec.aggregate_verdict) else 32.0
    ranks = _node_ranks(spec)
    by_rank: dict[int, list[_RenderNode]] = defaultdict(list)
    for node in spec.nodes:
        by_rank[ranks[node.key]].append(node)
    for nodes in by_rank.values():
        nodes.sort(key=lambda node: node.label)

    max_rank = max(by_rank, default=0)
    max_rows = max((len(nodes) for nodes in by_rank.values()), default=1)
    canvas_width = (
        2 * margin_x + node_width + max_rank * horizontal_gap
    )
    canvas_height = max(
        300.0,
        title_height + 44.0 + max_rows * vertical_gap,
    )
    positions: dict[str, tuple[float, float]] = {}
    for rank, nodes in sorted(by_rank.items()):
        x = margin_x + node_width / 2 + rank * horizontal_gap
        available = canvas_height - title_height - 24.0
        for index, node in enumerate(nodes):
            y = title_height + available * (index + 1) / (len(nodes) + 1)
            positions[node.key] = (x, y)

    edge_groups: dict[tuple[str, str], list[_RenderEdge]] = defaultdict(list)
    for edge in spec.edges:
        edge_groups[(edge.source, edge.target)].append(edge)

    edge_fragments: list[str] = []
    marker_defs: list[str] = []
    for edge_index, edge in enumerate(spec.edges):
        colour, line_style, label_fill = _VERDICT_STYLES[edge.verdict]
        if edge.tier is Tier.ASSOCIATIONAL and line_style == 'solid':
            line_style = 'dashed'
        marker_id = f'arrow-{edge_index}'
        marker_defs.append(
            f'<marker id="{marker_id}" markerWidth="10" markerHeight="10" '
            'refX="9" refY="5" orient="auto" markerUnits="strokeWidth">'
            f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{colour}"/>'
            '</marker>',
        )
        group = edge_groups[(edge.source, edge.target)]
        group_index = group.index(edge)
        offset = (group_index - (len(group) - 1) / 2) * 26.0
        source_x, source_y = positions[edge.source]
        target_x, target_y = positions[edge.target]
        dash = {
            'solid': '',
            'dashed': ' stroke-dasharray="9 7"',
            'dotted': ' stroke-dasharray="2 7"',
        }[line_style]

        if edge.source == edge.target:
            x1 = source_x + node_width / 2 - 12.0
            y1 = source_y - 14.0 + offset
            path = (
                f'M {x1:.1f} {y1:.1f} '
                f'C {x1 + 92:.1f} {y1 - 72:.1f}, '
                f'{x1 + 92:.1f} {y1 + 72:.1f}, '
                f'{x1:.1f} {y1 + 28:.1f}'
            )
            label_x = x1 + 86.0
            label_y = y1
        else:
            forward = target_x >= source_x
            x1 = source_x + (node_width / 2 if forward else -node_width / 2)
            x2 = target_x - (node_width / 2 if forward else -node_width / 2)
            y1 = source_y + offset
            y2 = target_y + offset
            bend = max(72.0, abs(x2 - x1) * 0.42)
            sign = 1.0 if forward else -1.0
            path = (
                f'M {x1:.1f} {y1:.1f} '
                f'C {x1 + sign * bend:.1f} {y1:.1f}, '
                f'{x2 - sign * bend:.1f} {y2:.1f}, '
                f'{x2:.1f} {y2:.1f}'
            )
            label_x = (x1 + x2) / 2
            # Fan-out graphs share one source y-coordinate. Moving labels
            # toward their individual targets preserves a distinct annotation
            # lane for each edge instead of stacking every label at midline.
            label_y = y1 + 0.68 * (y2 - y1)

        wrapped_lines: list[str] = []
        for line in edge.label_lines:
            wrapped_lines.extend(_wrap_label(line, width=33))
        label_lines = tuple(wrapped_lines)
        label_width = min(
            270.0,
            max(120.0, 7.0 * max((len(line) for line in label_lines), default=1)),
        )
        label_height = 15.0 * len(label_lines) + 16.0
        edge_fragments.append(
            f'<g data-bridge="{xml_escape(edge.name)}">'
            f'<path d="{path}" fill="none" stroke="{colour}" '
            f'stroke-width="2.3"{dash} marker-end="url(#{marker_id})"/>'
            f'<rect x="{label_x - label_width / 2:.1f}" '
            f'y="{label_y - label_height / 2:.1f}" '
            f'width="{label_width:.1f}" height="{label_height:.1f}" '
            f'rx="8" fill="{label_fill}" fill-opacity="0.96" '
            f'stroke="{colour}" stroke-opacity="0.28"/>'
            f'{_svg_multiline_text(label_lines, x=label_x, y=label_y, line_height=15.0, css_class="edge-label")}'
            '</g>',
        )

    node_fragments: list[str] = []
    for node in spec.nodes:
        x, y = positions[node.key]
        lines = _wrap_label(node.label, width=26)
        if node.kind == 'intervention':
            fill = '#eff6ff'
            stroke = '#2563eb'
        else:
            fill = '#ffffff'
            stroke = '#94a3b8'
        node_fragments.append(
            f'<g data-node="{xml_escape(node.key)}">'
            f'<rect x="{x - node_width / 2:.1f}" '
            f'y="{y - node_height / 2:.1f}" '
            f'width="{node_width:.1f}" height="{node_height:.1f}" '
            f'rx="18" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
            f'{_svg_multiline_text(lines, x=x, y=y, line_height=18.0, css_class="node-label")}'
            '</g>',
        )

    title_fragment = ''
    if spec.title:
        title_fragment = (
            f'<text x="{margin_x:.1f}" y="35" class="title" '
            f'text-anchor="start">{xml_escape(spec.title)}</text>'
        )
    badge_fragment = ''
    if spec.aggregate_verdict is not None:
        badge_text = spec.aggregate_verdict.value.replace('_', ' ').upper()
        badge_width = max(132.0, 8.0 * len(badge_text) + 30.0)
        badge_x = canvas_width - margin_x - badge_width
        badge_fragment = (
            f'<rect x="{badge_x:.1f}" y="15" width="{badge_width:.1f}" '
            'height="32" rx="16" fill="#f1f5f9" stroke="#64748b"/>'
            f'<text x="{badge_x + badge_width / 2:.1f}" y="36" '
            f'class="badge" text-anchor="middle">{xml_escape(badge_text)}</text>'
        )

    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {canvas_width:.1f} {canvas_height:.1f}" '
        f'width="{canvas_width:.1f}" height="{canvas_height:.1f}" '
        'role="img">\n'
        '<style>'
        '.title{font:700 20px Arial,sans-serif;fill:#0f172a}'
        '.badge{font:700 11px Arial,sans-serif;fill:#334155;letter-spacing:.5px}'
        '.node-label{font:600 13px Arial,sans-serif;fill:#0f172a}'
        '.edge-label{font:500 10.5px Arial,sans-serif;fill:#334155}'
        '</style>\n'
        f'<defs>{"".join(marker_defs)}</defs>\n'
        f'{title_fragment}{badge_fragment}'
        f'{"".join(edge_fragments)}'
        f'{"".join(node_fragments)}'
        '</svg>\n'
    )


def evidence_graph_to_svg(
    bridges: Iterable[Bridge],
    evaluations: Mapping[str, BridgeEvaluation],
    *,
    node_labels: Mapping[str, str] | None = None,
    edge_labels: Mapping[str, str] | None = None,
    edge_summaries: Mapping[str, str] | None = None,
    title: str | None = None,
) -> str:
    """Render exact evaluated bridge edges as standalone vector SVG.

    Same surface as :func:`evidence_graph_to_dot` — see its docstring
    for the per-parameter justifications."""
    return _spec_to_svg(_evidence_spec(
        bridges,
        evaluations,
        node_labels=node_labels,
        edge_labels=edge_labels,
        edge_summaries=edge_summaries,
        title=title,
    ))


def computation_graph_to_svg(
    graph: ComputationGraph,
    *,
    node_labels: Mapping[str, str] | None = None,
    title: str | None = None,
) -> str:
    """Render an observed ``@claim`` identity-flow graph as SVG."""
    return _spec_to_svg(_computation_spec(
        graph,
        node_labels=node_labels,
        title=title,
    ))


_RENDER_SUFFIXES: Mapping[str, Literal['svg', 'dot']] = {
    '.svg': 'svg',
    '.dot': 'dot',
}


def render_evidence(
    bridges: Iterable[Bridge],
    evaluations: Mapping[str, BridgeEvaluation],
    out: Path,
    *,
    title: str | None = None,
) -> Path:
    """One-call evidence render from exactly what a hypothesis run holds.

    ``bridges`` (the module's ``BRIDGES``) and ``evaluations`` (the
    verdict mapping ``corroborate.runner.run`` returns) are the whole
    input; every display decision defaults from the run itself — edge
    labels from bridge names, verdict styling from the evaluations,
    the aggregate-verdict badge from the framework's own
    ``composed_verdict``.  The output format follows the suffix of
    ``out`` (``.svg`` or ``.dot``); the rendered file is written there
    and the path returned.

    ``title`` is the only optional knob: the CLI passes the hypothesis
    module name, library callers may caption differently or omit it.
    For per-node/per-edge display copy, reach for the fine-control
    functions (:func:`evidence_graph_to_svg` and siblings) instead.
    """
    bridge_tuple = tuple(bridges)
    if not bridge_tuple:
        raise ValueError(
            'nothing renderable: the bridge set is empty — the '
            'hypothesis module declares no BRIDGES (or an empty '
            'subset was passed)',
        )
    fmt = _RENDER_SUFFIXES.get(out.suffix.lower())
    if fmt is None:
        raise ValueError(
            f'unsupported render suffix {out.suffix!r} on {out} — '
            f'use .svg (standalone vector) or .dot (Graphviz source)',
        )
    if fmt == 'svg':
        rendered = evidence_graph_to_svg(
            bridge_tuple, evaluations, title=title,
        )
    else:
        rendered = evidence_graph_to_dot(
            bridge_tuple, evaluations, title=title,
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    _ = out.write_text(rendered, encoding='utf-8')
    return out


__all__ = [
    'computation_graph_to_dot',
    'computation_graph_to_svg',
    'evidence_graph_to_dot',
    'evidence_graph_to_svg',
    'render_evidence',
]
