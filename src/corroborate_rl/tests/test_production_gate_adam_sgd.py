"""Production-gate end-to-end smoke: Adam ≥ SGD on CartPole.

Drives the canonical analysis pipeline at minimal scale — substrate
sweep (`run_dqn_arm` × 2 arms) → cell records → registered
`paired_g` analysis — and asserts a directional verdict against
a known-result anchor: Adam dominates plain SGD on DQN-CartPole at
small step budgets (per the literature; folklore among DQN
practitioners since Mnih 2015 onward, since Adam's per-parameter
adaptive scaling absorbs Q's heavy-tail gradient distribution
better than vanilla SGD's single global lr).

This test is a regression net for the analysis-pipeline plumbing,
not a scientific result:
- Did the substrate sweep run at all?
- Did the runner stamp arms with stable `arm_key` values?
- Did the registered `eval_best_burst_mean` measurable surface
  a usable scalar in `cell.run.measurements`?
- Does `paired_g` consume cell dicts and emit a typed result?
- Is the verdict's directional sign right (positive g)?

If pyright + unit tests pass but this test changes verdict, an
analytical-layer regression slipped through. Catch it before the
corpus run does.

Marked `@pytest.mark.slow` — JAX compile dominates wall time
(~10-20 s on GPU, ~30-60 s on CPU). Run via `pytest -m slow` or
the empty-marker invocation that includes both cohorts."""
from __future__ import annotations

from functools import partial

import pytest

# Side-effect imports: register substrate measurables (so
# `eval_best_burst_mean` resolves through the registry inside
# `paired_g.fn`) and framework analyses (so the @analysis-decorated
# `paired_g` is wired into the registry).
import corroborate.analyses  # noqa: F401  # pyright: ignore[reportUnusedImport]
import corroborate_rl.dqn.measurables  # noqa: F401  # pyright: ignore[reportUnusedImport]

from corroborate.analyses.paired_g import paired_g
from corroborate_rl.cell_runner import ArmResult, run_dqn_arm
from corroborate_rl.dqn.claims.optimizer import adam, sgd
from corroborate_rl.dqn.dqn import dqn
from corroborate_rl.dqn.measurables import dqn_default_measurables
from corroborate_rl.env_catalogue import get


_N_SEEDS: int = 5
_TOTAL_STEPS: int = 5_000


def _cells_from_arm(
    arm_key: str, result: ArmResult,
) -> list[dict[str, object]]:
    """Flatten one arm's `ArmResult` into the cell-dict shape
    `paired_g.fn` consumes — top-level `arm_key` plus the run's
    measurements (which carry `env_name`, `seed`, the registered
    measurables' scalars, etc.)."""
    out: list[dict[str, object]] = []
    for cell in result.cells:
        row: dict[str, object] = {'arm_key': arm_key}
        row.update(cell.run.measurements)
        out.append(row)
    return out


@pytest.mark.slow
def test_adam_dominates_sgd_on_cartpole() -> None:
    """At identical lr=1e-3 and 5k training steps × 5 seeds on
    CartPole-v1, Adam reaches a higher `eval_best_burst_mean`
    than vanilla SGD with the same baseline. paired_g's `g` must
    be positive (Adam > SGD) and `n_pairs` must equal the seed
    count (no dropout from missing measurements)."""
    spec = get('CartPole-v1')
    seeds = tuple(range(_N_SEEDS))
    measurables = dqn_default_measurables()

    adam_claim = partial(
        dqn,
        total_steps=_TOTAL_STEPS,
        eval_every=_TOTAL_STEPS,
        n_episodes=5,
        optimizer=partial(adam, lr=1e-3),
    )
    adam_result = run_dqn_arm(
        spec, seeds, claim=adam_claim, arm_key='adam',
        measurables=measurables,
    )

    sgd_claim = partial(
        dqn,
        total_steps=_TOTAL_STEPS,
        eval_every=_TOTAL_STEPS,
        n_episodes=5,
        optimizer=partial(sgd, lr=1e-3),
    )
    sgd_result = run_dqn_arm(
        spec, seeds, claim=sgd_claim, arm_key='sgd',
        measurables=measurables,
    )

    cells = (
        _cells_from_arm('adam', adam_result)
        + _cells_from_arm('sgd', sgd_result)
    )

    # `eval_best_burst_mean` is a registered measurable; paired_g
    # resolves it through the registry to read each cell's scalar.
    result = paired_g.fn(
        cells,
        source='eval_best_burst_mean',
        treatment_arm='adam',
        baseline_arm='sgd',
        pair_by=('seed',),
    )

    assert result.n_pairs == _N_SEEDS, (
        f'expected {_N_SEEDS} matched (adam, sgd) pairs at the '
        f'5 seed indices; got n_pairs={result.n_pairs}. '
        f'Likely cause: arm_key stamping changed shape or a '
        f'measurable failed to surface a scalar.'
    )
    assert result.g > 0.0, (
        f'expected Adam > SGD on CartPole at lr=1e-3, 5k steps '
        f'(g > 0); got g={result.g:+.3f} ± {result.se:.3f}. '
        f'A regression to non-positive g indicates either an '
        f'analytical-layer bug or a substrate change that broke '
        f'optimizer convergence.'
    )
