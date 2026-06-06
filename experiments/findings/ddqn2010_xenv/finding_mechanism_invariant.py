"""Mechanism direction-consistency: paired_dqn reduces jensen_gap at
every MinAtar env (γ=0.999) — UNDERPOWERED at n=4.

Per env the de-biasing is real and large (vanilla jensen_gap
289 / 57 / 34 / 0.12 → paired 0 at all four). The cross-env claim is a
DIRECTION sign-test (the magnitudes are too heterogeneous, I²≈0.99, for
a defensible pooled magnitude — the DL prediction interval brackets
zero). At 4/4 same-direction the one-tailed binomial is p=0.0625 > 0.05,
so the framework correctly returns POWER_INSUFFICIENT, not HELD: one
more same-direction env certifies SUPPORTED. EXPECTED is pinned to the
honest underpowered state; BLOCKED_ON names the gap.
"""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn2010_xenv.bridges import (
    paired_reduces_jens_consistently__minatar4,
)

EXPECTED: ClusterVerdict = ClusterVerdict.UNDERPOWERED
BLOCKED_ON: str | None = (
    '4/4 envs reduce jensen_gap (consistent direction) but binomial '
    'p=0.0625 at n=4 envs; needs >=5 same-direction envs for SUPPORTED. '
    'Per-env de-biasing is real (vanilla jens 289/57/34/0.12 -> paired 0).'
)
BRIDGES: tuple[Bridge, ...] = (paired_reduces_jens_consistently__minatar4,)
