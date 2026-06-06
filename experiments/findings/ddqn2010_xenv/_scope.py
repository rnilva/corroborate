"""Scope + program-contrast constants for the cross-env DDQN-2010 study.

Four MinAtar envs at γ=0.999, three programs/arms each:
  vanilla    — single-net DQN          (program='dqn',        arm_key='baseline')
  ddqn2016   — shared-target double DQN (program='dqn',        arm_key '…double_greedify')
  paired     — van Hasselt 2010        (program='paired_dqn',  arm_key='baseline')

The treatment is a PROGRAM swap (`paired_dqn`), so the framework's typed
`RunRow.program` column is the natural contrast key — bridges set
`arm_field='program'`, `treatment_arm='paired_dqn'`, `baseline_arm='dqn'`.
vanilla and ddqn2016 share `program='dqn'`, so the paired-vs-vanilla
contrast scopes OUT the ddqn2016 arm (its `…double_greedify` arm_key);
within scope, `program='dqn'` ⟺ vanilla.

Provenance note: vanilla+ddqn2016 come from the canonical
`minatar_gamma_sweep_k1/g0999_*` corpora for SI / Breakout / Freeway.
Asterix's canonical corpus lacks a per-burst-eval `measurements.parquet`,
so Asterix's vanilla is the MATCHED `paired_vanilla` run that lives in
`asterix_g0999_ddqn2010_paired` (same seeds/HPs/n_episodes as its
paired_dqn arm — a cleaner matched baseline; no Asterix ddqn2016 arm).
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

_DATA = Path(__file__).resolve().parents[3] / 'experiments' / 'data'

ENVS: tuple[str, ...] = (
    'Asterix-MinAtar', 'SpaceInvaders-MinAtar',
    'Breakout-MinAtar', 'Freeway-MinAtar',
)

CANONICAL: dict[str, Path] = {
    e: _DATA / 'minatar_gamma_sweep_k1' / f'g0999_{e}' for e in ENVS
}
PAIRED: dict[str, Path] = {
    'Asterix-MinAtar': _DATA / 'asterix_g0999_ddqn2010_paired',
    'SpaceInvaders-MinAtar': _DATA / 'si_g0999_ddqn2010_paired',
    'Breakout-MinAtar': _DATA / 'breakout_g0999_ddqn2010_paired',
    'Freeway-MinAtar': _DATA / 'freeway_g0999_ddqn2010_paired',
}

TREATMENT_PROGRAM = 'paired_dqn'
BASELINE_PROGRAM = 'dqn'
_DDQN2016 = pl.col('arm_key').str.contains('double_greedify')

# Hypothesis-module universe: γ=0.999, these 4 MinAtar envs.
MODULE_SCOPE: pl.Expr = (
    (pl.col('gamma') == 0.999) & pl.col('env_name').is_in(list(ENVS))
)

# paired ↔ vanilla contrast: keep both `program` values, drop ddqn2016
# (so `program='dqn'` ⟺ vanilla within scope).
PAIRED_VS_VANILLA_SCOPE: pl.Expr = (
    MODULE_SCOPE
    & ~_DDQN2016
    & pl.col('program').is_in([TREATMENT_PROGRAM, BASELINE_PROGRAM])
)


def display_arm() -> pl.Expr:
    """3-way label for human-readable exploration output ONLY (the
    authoritative contrast is `program` + `PAIRED_VS_VANILLA_SCOPE`)."""
    return (
        pl.when(pl.col('program') == TREATMENT_PROGRAM).then(pl.lit('paired'))
        .when(_DDQN2016).then(pl.lit('ddqn2016'))
        .when(pl.col('arm_key') == 'baseline').then(pl.lit('vanilla'))
        .otherwise(None)
    )
