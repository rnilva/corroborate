"""End-to-end parity smoke: the YAML-loaded `expectile_3way`
manifest produces a Hypothesis tuple that is structurally
identical to the Python-authored `_hypothesis()` recipe in
`experiments/collect_expectile_3way.py`.

Identity contract:
- Per-hypothesis: `name`, `predicted_direction`, `arm_key()`,
  intervention leaves equal, intervention slot Claims equal
  (frozen-dataclass equality), and `claim_graph_signature` of
  every callable slot value matches between paths.
- Per-manifest: `name`, `out_dir`, `archive_remote`, `arms_shape`
  match; `envs` tuple equals.

If the YAML schema or the loader drifts from the Python authoring
shape, the smoke catches it before any sweep runs."""
from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from pathlib import Path

import pytest

from corroborate.hypothesis import Hypothesis
from corroborate.intervention import Intervention
from corroborate.rl.dqn.collect import EnvConfig
from corroborate.rl.dqn.yaml_sweep import (
    DQNExperimentManifest, default_dqn_registry, load_manifest,
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
    """Mirrors `experiments/collect_expectile_3way.py::_hypothesis`
    exactly. Inlined so the test does not import a script (which
    triggers the script's `os.environ` setup as a side effect)."""
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
            name='vanilla_dqn', intervention=base,
            bridges=(), predicted_direction=None,
            intervention_arms=(),
        )
    if name == 'ddqn':
        boot = partial(bootstrap, greedification=double_greedify)
        base['bootstrap'] = boot
        return Hypothesis(
            name='ddqn', intervention=base,
            bridges=(), predicted_direction='a_gt_b',
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
            name='expectile_dqn', intervention=base,
            bridges=(), predicted_direction='a_gt_b',
            intervention_arms=(
                Intervention(slot_path='bootstrap', replacement=boot),
            ),
        )
    raise ValueError(name)


# ---------- fixtures ----------

@pytest.fixture
def manifest() -> DQNExperimentManifest:
    reg = default_dqn_registry()
    return load_manifest(MANIFEST_PATH, reg=reg)


# ---------- envelope checks ----------

def test_manifest_envelope_fields(
    manifest: DQNExperimentManifest,
) -> None:
    assert manifest.name == 'expectile_3way'
    assert manifest.out_dir == Path(
        'experiments/data/expectile_3way',
    )
    assert manifest.archive_remote is None
    assert manifest.arms_shape == 'chunked'


def test_manifest_envs_tuple_matches(
    manifest: DQNExperimentManifest,
) -> None:
    expected_envs = (
        EnvConfig('Catch-bsuite', n_seeds=30, chunk_size=15),
        EnvConfig('DiscountingChain-bsuite', n_seeds=30, chunk_size=15),
        EnvConfig('MountainCar-v0', n_seeds=30, chunk_size=15),
        EnvConfig('Acrobot-v1', n_seeds=30, chunk_size=15),
        EnvConfig('FourRooms-misc', n_seeds=30, chunk_size=15),
    )
    assert manifest.envs == expected_envs


def test_manifest_hypothesis_count(
    manifest: DQNExperimentManifest,
) -> None:
    assert len(manifest.hypotheses) == 3
    assert [h.name for h in manifest.hypotheses] == [
        'vanilla_dqn', 'ddqn', 'expectile_dqn',
    ]


# ---------- per-hypothesis parity (parametrised) ----------

@pytest.fixture
def hypothesis_pairs(
    manifest: DQNExperimentManifest,
) -> dict[str, tuple[
    Hypothesis[Mapping[str, object]],
    Hypothesis[Mapping[str, object]],
]]:
    yaml_by_name = {h.name: h for h in manifest.hypotheses}
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
    manifest: DQNExperimentManifest,
) -> None:
    """The signature is not constant — it actually distinguishes
    the three arms. ddqn != expectile_dqn at the bootstrap slot,
    confirming the registry-resolved partials aren't collapsing
    into the same hash."""
    by_name = {h.name: h for h in manifest.hypotheses}
    sig_ddqn = claim_graph_signature(
        by_name['ddqn'].intervention['bootstrap'],
    )
    sig_expectile = claim_graph_signature(
        by_name['expectile_dqn'].intervention['bootstrap'],
    )
    assert sig_ddqn != sig_expectile
