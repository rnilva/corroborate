"""Schema-contract smoke for the `expectile_3way` YAML sweep:
every authored slot resolves to the typed handle the substrate's
Claim graph holds.

The reference Python construction below is one realisation of
the same schema — useful for catching loader regressions, but
the canonical contract is the YAML schema itself: a
Module Claim slot resolves to a `ClaimBase` instance, a slot
binding to a `partial`-of-FnClaim resolves with the right inner
FnClaim, and `claim_graph_signature` is stable across loads."""
from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from pathlib import Path

import pytest

from corroborate.hypothesis import Hypothesis
from corroborate.intervention import Intervention
from corroborate.rl.dqn.collect import EnvConfig
from corroborate.registry import Registry
from corroborate.rl.dqn.yaml_sweep import (
    DQNSweep, default_dqn_registry, load_sweep,
)
from corroborate.signature import claim_graph_signature


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = (
    REPO_ROOT / 'experiments' / 'configs' / 'expectile_3way.yaml'
)


# ---------- Python-authored reference ----------

def _python_hypothesis(
    name: str,
) -> Hypothesis[Mapping[str, object]]:
    """Canonical Python recipe for the expectile_3way cohort —
    the reference that `expectile_3way.yaml` must match
    structurally."""
    from corroborate.rl.dqn.claims.bootstrap import (
        bootstrap, double_greedify, expectile_greedify,
    )
    from corroborate.rl.dqn.claims.optimizer import (
        Adam, WarmedUpdate,
    )
    from corroborate.rl.dqn.claims.q_network import MLP
    from corroborate.rl.dqn.claims.replay import Replay
    base: dict[str, object] = {
        'total_steps': 200_000,
        'eval_every': 20_000,
        'n_episodes': 5,
        'gamma': 0.99,
        'replay': Replay(capacity=50_000, batch_size=32),
        'optimizer': WarmedUpdate(
            inner=Adam(lr=1e-4), warmup_steps=100,
        ),
        'sync_period': 100,
        'q_network': MLP(hidden=(64, 64)),
    }
    if name == 'vanilla_dqn':
        return Hypothesis(
            name='vanilla_dqn', intervention=base, predicted_direction=None,
            intervention_arms=(),
        )
    if name == 'ddqn':
        boot = partial(bootstrap, greedification=double_greedify)
        base['bootstrap'] = boot
        return Hypothesis(
            name='ddqn', intervention=base, predicted_direction='a_gt_b',
            intervention_arms=(
                Intervention(slot_path='bootstrap', replacement=boot),
            ),
        )
    if name == 'expectile_dqn':
        boot = partial(
            bootstrap,
            greedification=partial(expectile_greedify, tau=0.7),
        )
        base['bootstrap'] = boot
        return Hypothesis(
            name='expectile_dqn', intervention=base, predicted_direction='a_gt_b',
            intervention_arms=(
                Intervention(slot_path='bootstrap', replacement=boot),
            ),
        )
    raise ValueError(name)


# ---------- fixtures ----------

@pytest.fixture
def reg() -> Registry:
    return default_dqn_registry()


@pytest.fixture
def sweep(reg: Registry) -> DQNSweep:
    s = load_sweep(MANIFEST_PATH, reg=reg)
    assert s.arms_shape == 'chunked'
    return s


@pytest.fixture
def yaml_hypotheses(
    sweep: DQNSweep, reg: Registry,
) -> tuple[Hypothesis[Mapping[str, object]], ...]:
    return sweep.build_hypotheses(reg=reg)


# ---------- envelope checks ----------

def test_sweep_envelope_fields(sweep: DQNSweep) -> None:
    assert sweep.name == 'expectile_3way'
    assert sweep.out_dir == Path(
        'experiments/data/expectile_3way',
    )
    assert sweep.archive_remote is None
    assert sweep.arms_shape == 'chunked'


def test_sweep_envs_tuple_matches(sweep: DQNSweep) -> None:
    expected_envs = (
        EnvConfig('Catch-bsuite', n_seeds=30, chunk_size=15),
        EnvConfig('DiscountingChain-bsuite', n_seeds=30, chunk_size=15),
        EnvConfig('MountainCar-v0', n_seeds=30, chunk_size=15),
        EnvConfig('Acrobot-v1', n_seeds=30, chunk_size=15),
        EnvConfig('FourRooms-misc', n_seeds=30, chunk_size=15),
    )
    assert sweep.envs == expected_envs


def test_sweep_hypothesis_count(
    yaml_hypotheses: tuple[Hypothesis[Mapping[str, object]], ...],
) -> None:
    assert len(yaml_hypotheses) == 3
    assert [h.name for h in yaml_hypotheses] == [
        'vanilla_dqn', 'ddqn', 'expectile_dqn',
    ]


# ---------- per-hypothesis schema-contract checks ----------

@pytest.fixture
def hypothesis_pairs(
    yaml_hypotheses: tuple[Hypothesis[Mapping[str, object]], ...],
) -> dict[str, tuple[
    Hypothesis[Mapping[str, object]],
    Hypothesis[Mapping[str, object]],
]]:
    yaml_by_name = {h.name: h for h in yaml_hypotheses}
    return {
        name: (yaml_by_name[name], _python_hypothesis(name))
        for name in ('vanilla_dqn', 'ddqn', 'expectile_dqn')
    }


@pytest.mark.parametrize(
    'h_name', ['vanilla_dqn', 'ddqn', 'expectile_dqn'],
)
def test_predicted_direction_matches(
    hypothesis_pairs: dict[str, tuple[
        Hypothesis[Mapping[str, object]],
        Hypothesis[Mapping[str, object]],
    ]],
    h_name: str,
) -> None:
    yaml_h, py_h = hypothesis_pairs[h_name]
    assert yaml_h.predicted_direction == py_h.predicted_direction


@pytest.mark.parametrize(
    'h_name', ['vanilla_dqn', 'ddqn', 'expectile_dqn'],
)
def test_intervention_leaves_match(
    hypothesis_pairs: dict[str, tuple[
        Hypothesis[Mapping[str, object]],
        Hypothesis[Mapping[str, object]],
    ]],
    h_name: str,
) -> None:
    yaml_h, py_h = hypothesis_pairs[h_name]
    for k in (
        'total_steps', 'eval_every', 'n_episodes', 'gamma',
        'sync_period',
    ):
        assert yaml_h.intervention[k] == py_h.intervention[k], (
            f'{h_name}.intervention[{k!r}] differs: '
            f'yaml={yaml_h.intervention[k]!r} '
            f'python={py_h.intervention[k]!r}'
        )


@pytest.mark.parametrize(
    'h_name', ['vanilla_dqn', 'ddqn', 'expectile_dqn'],
)
def test_module_claim_slots_equal(
    hypothesis_pairs: dict[str, tuple[
        Hypothesis[Mapping[str, object]],
        Hypothesis[Mapping[str, object]],
    ]],
    h_name: str,
) -> None:
    """Frozen-dataclass equality on Module Claim slots —
    Replay/WarmedUpdate/Adam/MLP all `==` between paths."""
    yaml_h, py_h = hypothesis_pairs[h_name]
    for k in ('q_network', 'optimizer', 'replay'):
        assert yaml_h.intervention[k] == py_h.intervention[k]


@pytest.mark.parametrize(
    'h_name', ['ddqn', 'expectile_dqn'],
)
def test_bootstrap_signature_matches(
    hypothesis_pairs: dict[str, tuple[
        Hypothesis[Mapping[str, object]],
        Hypothesis[Mapping[str, object]],
    ]],
    h_name: str,
) -> None:
    """The headline contract: `claim_graph_signature` of the
    intervened-on slot is identical across YAML- and
    Python-authored paths. If they differ, downstream corpus
    rows tagged with the signature land in different
    structural-identity buckets."""
    yaml_h, py_h = hypothesis_pairs[h_name]
    sig_yaml = claim_graph_signature(
        yaml_h.intervention['bootstrap'],
    )
    sig_python = claim_graph_signature(
        py_h.intervention['bootstrap'],
    )
    assert sig_yaml == sig_python


@pytest.mark.parametrize(
    'h_name', ['vanilla_dqn', 'ddqn', 'expectile_dqn'],
)
def test_arm_key_matches(
    hypothesis_pairs: dict[str, tuple[
        Hypothesis[Mapping[str, object]],
        Hypothesis[Mapping[str, object]],
    ]],
    h_name: str,
) -> None:
    """Pairing key for HypothesisComparisonRow. Drift here would
    place YAML and Python rows in different arms."""
    yaml_h, py_h = hypothesis_pairs[h_name]
    assert yaml_h.arm_key() == py_h.arm_key()


def test_signatures_distinct_across_arms(
    yaml_hypotheses: tuple[Hypothesis[Mapping[str, object]], ...],
) -> None:
    """The signature is not constant — it actually distinguishes
    the three arms. ddqn != expectile_dqn at the bootstrap slot,
    confirming the registry-resolved partials aren't collapsing
    into the same hash."""
    by_name = {h.name: h for h in yaml_hypotheses}
    sig_ddqn = claim_graph_signature(
        by_name['ddqn'].intervention['bootstrap'],
    )
    sig_expectile = claim_graph_signature(
        by_name['expectile_dqn'].intervention['bootstrap'],
    )
    assert sig_ddqn != sig_expectile
