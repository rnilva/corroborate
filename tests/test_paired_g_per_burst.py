"""Real-corpus reproduction smoke for the per-burst paired-g panel
analysis.

The synthetic-input + composition tests previously living here
were superseded by the closed-form analytic substrate at
`tests/analytic/lg_scm/test_paired_g_per_burst.py`, which exercises
the full `Measurable → reduce_axis → paired_g_per_burst` pipeline
on real LG-SCM cells with `rel_err` bounds against
`mu_x · sqrt(n_steps) / sigma_x · c_4`.

What stays here: registry probe + corpus-binding smokes that
reproduce specific FINDINGS observations on real DDQN data.
Those tests are NOT analytical — they're regression guards
against shifts in the persisted corpus + analysis path."""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

import corroborate.analyses  # noqa: F401  # pyright: ignore[reportUnusedImport]

from corroborate.analyses.paired.paired_g_per_burst import (
    DEFAULT_PER_BURST_SOURCE, panel_for_env, paired_g_per_burst,
)


def test_analysis_registered_with_bridge_required_signature() -> None:
    """`paired_g_per_burst` populates the registry on import AND
    its callable signature exposes the parameters that bridges
    rely on (`treatment_arm`, `baseline_arm`, `pair_by`, `source`).

    The `assert get_registered(...) is not None` check alone passes
    for ANY function decorated with `@analysis` — a stub-passable
    D-class assertion. Pin the four parameter names that bridges
    in the registry actually inject; renaming any of them in
    `paired_g_per_burst.fn` would silently break bridge dispatch
    while leaving the registry probe green.
    """
    import inspect
    from corroborate.bridge.analysis import get_registered
    registered = get_registered('paired_g_per_burst')
    assert registered is not None
    sig = inspect.signature(registered.fn)
    required = {'treatment_arm', 'baseline_arm', 'pair_by', 'source'}
    missing = required - set(sig.parameters)
    assert not missing, (
        f'paired_g_per_burst.fn missing parameters: {missing}. '
        f'Bridges inject these by name; renaming silently breaks '
        f'dispatch.'
    )


# ============ Real-corpus reproduction smoke ============

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPECTILE_RUNS = (
    REPO_ROOT / 'experiments' / 'data' / 'expectile_3way'
    / 'runs.parquet'
)
EXPECTILE_TRACES = (
    REPO_ROOT / 'experiments' / 'data' / 'expectile_3way'
    / 'traces.parquet'
)


@pytest.fixture(scope='module')
def expectile_3way_cells() -> list[dict[str, object]]:
    """Project to needed columns to keep memory bounded; join
    runs + traces on `id`."""
    if not (EXPECTILE_RUNS.exists() and EXPECTILE_TRACES.exists()):
        pytest.skip('expectile_3way corpus not available')
    runs = pl.read_parquet(
        EXPECTILE_RUNS,
        columns=['id', 'arm_key', 'env_name', 'seed'],
    )
    traces = pl.read_parquet(
        EXPECTILE_TRACES,
        columns=['id', 'mc_return', 'predicted_q_at_start'],
    )
    return list(
        runs.join(traces, on='id', how='inner').iter_rows(named=True),
    )


# DDQN arm in the canonical-arm-key naming used by the current
# `expectile_3way` corpus (post-Phase-6 typed-contract migration).
# Pre-migration tests passed `treatment_arm='ddqn'` / `'vanilla_dqn'`;
# the corpus stores the `combined_arm_key`-derived canonical string.
_DDQN_ARM = (
    'bootstrap=partial(Claim:bootstrap;greedification=Claim:double_greedify)'
)
_VANILLA_ARM = 'baseline'


def test_real_corpus_catch_bsuite_zero_across_bursts(
    expectile_3way_cells: list[dict[str, object]],
) -> None:
    """FINDINGS revision 12: 'DDQN at n=1 has *exactly* zero effect
    (g = +0.00) on Catch — both arms saturate at mc_return ≈ +0.92
    and converge to the same near-optimal policy.'

    Per-burst panel should show g ≈ 0 across every burst on Catch.
    """
    result = paired_g_per_burst.fn(
        expectile_3way_cells,
        treatment_arm=_DDQN_ARM,
        baseline_arm=_VANILLA_ARM,
        pair_by=('seed',),
        source=DEFAULT_PER_BURST_SOURCE,
    )
    catch_panel = panel_for_env(result, 'Catch-bsuite')
    assert len(catch_panel) > 0
    for s in catch_panel:
        assert abs(s.g) < 0.1, (
            f'Catch burst {s.burst_index}: expected g ≈ 0, '
            f'got {s.g:+.3f}'
        )


def test_real_corpus_fourrooms_positive_across_bursts(
    expectile_3way_cells: list[dict[str, object]],
) -> None:
    """FINDINGS revision 9 (and revision 12 for the (C-A) cell):
    'g_link(C-A) ≈ +0.79 across bursts on FourRooms — DDQN
    benefit is stable throughout.'

    Per-burst panel should show positive g across every burst on
    FourRooms (DDQN improves outcome relative to vanilla)."""
    result = paired_g_per_burst.fn(
        expectile_3way_cells,
        treatment_arm=_DDQN_ARM,
        baseline_arm=_VANILLA_ARM,
        pair_by=('seed',),
        source=DEFAULT_PER_BURST_SOURCE,
    )
    fr_panel = panel_for_env(result, 'FourRooms-misc')
    assert len(fr_panel) > 0
    positive_bursts = sum(1 for s in fr_panel if s.g > 0)
    assert positive_bursts >= len(fr_panel) - 1, (
        f'expected positive g on most FourRooms bursts; '
        f'got {positive_bursts}/{len(fr_panel)} positive'
    )
    mean_g = sum(s.g for s in fr_panel) / len(fr_panel)
    assert mean_g > 0.3, (
        f'expected mean burst g > 0.3 (FINDINGS reports ~+0.79); '
        f'got {mean_g:.3f}'
    )
