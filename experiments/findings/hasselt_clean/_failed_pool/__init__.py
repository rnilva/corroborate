"""Pedagogical: the random-effects pool attempt that surfaced
the cross-env exchangeability failure.

This subdirectory preserves the *original* B3 / B4 intervention
bridges that used `stratified_arm_diff_pooled` (random-effects
pool of Cohen's d across envs) under PI-based discipline. Both
fire NO_EFFECT/NULL_EFFECT on the canonical-dormancy panel —
not because DDQN has no effect (B3's pooled d=-1.90 is huge),
but because the pool's prediction interval brackets zero under
extreme cross-env heterogeneity (I²=0.97).

The methodological reading: **random-effects pooling assumes the
strata are exchangeable draws from a population, with model
`g_i ~ N(μ, τ²)`**. RL environments aren't exchangeable in any
useful sense — they differ in network class (CNN vs MLP),
Q-magnitude (Asterix d=-8.9 vs Acrobot d=-0.01 — 800× scale),
reward sparsity, episode horizon. Pooling with only `do(DDQN)`
as covariate forces all this structural heterogeneity into τ²,
and the PI test honestly refuses extrapolation.

The framework's verdict layer made this visible: the pool
bridges fire NO_EFFECT under PI-based discipline, prompting
the author to ask "what claim shape IS appropriate?" The
substantive answer (cross-env consistency / sign-test, not
pop-mean pool) lives in the parent module's main `chain.py`.

Kept here for pedagogy. Each bridge's `EXPECTED` pinned to its
actual verdict (REFUTED at the Finding cluster level) — the
framework's drift detection then ensures the pedagogical
artifact stays empirically anchored."""
from __future__ import annotations

from experiments.findings.hasselt_clean._failed_pool import (
    finding_chain_pool_inadequate,
)
from experiments.findings.hasselt_clean._failed_pool.chain_pool import (
    BRIDGES as BRIDGES,
)


FINDINGS = (
    finding_chain_pool_inadequate,
)
