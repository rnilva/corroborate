"""Direct tests on `per_env_paired_g_panel` — the env-stratifying
wrapper around `paired_g`. The core math sits in `paired_g.fn`,
extensively covered elsewhere; this file pins:

- env_filter: when non-empty, only those envs are returned
  (vs `not in` mutant inverting the filter, vs `lambda env: None`
  mutant excluding everything, vs `key_filter=None` mutant
  ignoring the filter)
- pair_by + arm_field kwargs propagate to `paired_g.fn`"""
from __future__ import annotations

import pytest

from corroborate.analyses.paired_g import per_env_paired_g_panel

from tests.analytic.lg_scm.composition import LinearGaussianSCM
from tests.analytic.lg_scm.runner import run_paired_arms


def _scm() -> LinearGaussianSCM:
    return LinearGaussianSCM(
        mu_x=1.0, sigma_x=0.5, beta_xz=0.5, sigma_z=0.1,
        beta_zy=1.5, sigma_y=0.1, n_steps=200,
    )


def _multi_env_corpus() -> list:
    """Three envs (env_A, env_B, env_C) × two arms × 10 seeds."""
    rows = []
    for env in ('env_A', 'env_B', 'env_C'):
        rows.extend(run_paired_arms(
            treatment=_scm(), baseline=_scm(),
            seeds=range(10),
            treatment_arm='ddqn', baseline_arm='vanilla',
            env_name=env,
        ))
    return [r.as_dict() for r in rows]


def test_panel_with_env_filter_returns_only_listed_envs() -> None:
    """`env_filter=('env_A', 'env_C')` keeps only those two envs.
    Pin:

    - `lambda env: env in env_filter` (vs `not in` mutant which
      would invert and return only env_B)
    - `lambda env: None` mutant (would return empty panel)
    - `key_filter=key_filter` (vs `key_filter=None` mutant that
      would ignore the filter and return all 3 envs)"""
    cells = _multi_env_corpus()
    panel = per_env_paired_g_panel(
        cells,
        treatment_arm='ddqn', baseline_arm='vanilla',
        source='y_mean',
        env_filter=('env_A', 'env_C'),
    )
    env_names = {s.stratum_id for s in panel}
    assert env_names == {'env_A', 'env_C'}


def test_panel_with_no_env_filter_returns_all_envs() -> None:
    """Empty `env_filter` (default) → key_filter=None → every
    env in the corpus contributes a stratum."""
    cells = _multi_env_corpus()
    panel = per_env_paired_g_panel(
        cells,
        treatment_arm='ddqn', baseline_arm='vanilla',
        source='y_mean',
    )
    env_names = {s.stratum_id for s in panel}
    assert env_names == {'env_A', 'env_B', 'env_C'}


def test_panel_with_non_default_pair_by_propagates_to_paired_g() -> None:
    """`pair_by` propagates through to paired_g.fn. Pin against
    the kwarg-drop mutant: drop reverts to default `('seed',)`,
    which would still produce paired cells.

    Construct: cells where pair_by=('seed', 'cell_id') makes
    every cell unique → 0 paired cells per env. Default
    `('seed',)` would still pair on seed → 10 pairs per env.
    Different n_pairs distinguishes the kwarg drop."""
    cells = _multi_env_corpus()
    # Inject a unique tag per cell so ('seed', 'tag') gives no pairs.
    tagged = [
        {**c, 'tag': c['id']}    # tag is unique per cell
        for c in cells
    ]
    panel = per_env_paired_g_panel(
        tagged,
        treatment_arm='ddqn', baseline_arm='vanilla',
        source='y_mean',
        pair_by=('seed', 'tag'),
    )
    # Each cell gets its own (seed, tag) tuple → no paired matches.
    assert all(s.n_pairs == 0 for s in panel)


def test_panel_with_non_default_arm_field_propagates() -> None:
    """`arm_field` propagates through to paired_g.fn. Pin against
    the kwarg-drop mutant — drop reverts to default 'arm_key'.

    Construct: cells WITHOUT 'arm_key', with arm in 'treatment'
    field instead. Original passes arm_field='treatment' through
    → finds arms → pairs. Mutant drops kwarg → paired_g uses
    default 'arm_key' → no arm-key field → no pairs."""
    cells = _multi_env_corpus()
    # Move arm_key → treatment, drop arm_key entirely.
    rekeyed = [
        {k: v for k, v in c.items() if k != 'arm_key'} | {'treatment': c['arm_key']}
        for c in cells
    ]
    panel = per_env_paired_g_panel(
        rekeyed,
        treatment_arm='ddqn', baseline_arm='vanilla',
        source='y_mean',
        arm_field='treatment',
    )
    assert len(panel) == 3
    assert all(s.n_pairs > 0 for s in panel)
