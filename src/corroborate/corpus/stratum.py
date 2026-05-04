"""Parametric stratum-summary type shared across analyses.

`StratumG[K]` is the canonical (stratum-id, g, se, n_pairs)
record produced by per-stratum paired-Hedges'-g analyses. The
type parameter `K` lets `paired_g_pooled` strata key by env
(`StratumG[str]`), `paired_g_per_burst` strata key by (env,
burst) (`StratumG[tuple[str, int]]`), and downstream
meta-regression panels work uniformly over the panel.

`StratumObservation` (in `corroborate.meta_regression`) is the
covariate-bearing sibling — same g/se/n shape plus a
`covariates: Mapping[str, float]` field — kept distinct so the
no-covariate panel-build path doesn't drag a permanently-empty
mapping along."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StratumG[K]:
    """Per-stratum paired Hedges' g result. `K` is the stratum-id
    type — `str` for env-strata, `tuple[str, int]` for (env,
    burst), or whatever the substrate uses."""
    stratum_id: K
    g: float
    se: float
    n_pairs: int


# Convenience aliases for the two common shapes. Substrate code
# can use these names directly or address `StratumG[K]`.
type PerEnvG = StratumG[str]
type PerBurstStratum = StratumG[tuple[str, int]]


__all__ = ['PerBurstStratum', 'PerEnvG', 'StratumG']
