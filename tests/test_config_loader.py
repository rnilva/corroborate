"""YAML → Hypothesis loader smoke. The contract is structural
identity: a YAML-loaded Hypothesis's slot Claims pass
`claim_graph_signature` equality with the equivalent
Python-authored Hypothesis, and `arm_key()` agrees.

If the loader and the hand-authored path drift, the YAML schema
needs a fix; the smoke catches that drift at config-load time."""
from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from pathlib import Path

import pytest

from corroborate.runner.config_loader import load_hypothesis, resolve
from corroborate.core.hypothesis import Hypothesis
from corroborate.core.intervention import Intervention
from corroborate.runner.registry import Registry
from corroborate.core.signature import claim_graph_signature


DQN_CLAIM_MODULES = (
    'corroborate.rl.dqn.claims.bootstrap',
    'corroborate.rl.dqn.claims.action_select',
    'corroborate.rl.dqn.claims.replay',
    'corroborate.rl.dqn.claims.q_network',
    'corroborate.rl.dqn.claims.optimizer',
    'corroborate.rl.dqn.claims.target_sync',
    'corroborate.rl.dqn.claims.loss',
)


@pytest.fixture
def reg() -> Registry:
    r = Registry()
    r.add_modules(DQN_CLAIM_MODULES)
    return r


# ---------- low-level resolve() smokes ----------

def test_resolve_passes_scalar_through(reg: Registry) -> None:
    assert resolve(0.99, reg=reg) == 0.99
    assert resolve(50000, reg=reg) == 50000
    assert resolve(True, reg=reg) is True
    assert resolve('plain', reg=reg) == 'plain'


def test_resolve_tuple_ifies_list(reg: Registry) -> None:
    """Config-bundle fields like `MLP.hidden: tuple[int, ...]`
    need YAML lists coerced to tuples or `replace(MLP(),
    hidden=[64,64])` raises on the frozen-dataclass equality
    check downstream."""
    assert resolve([64, 64], reg=reg) == (64, 64)


def test_resolve_class_instantiates_module(reg: Registry) -> None:
    from corroborate.rl.dqn.claims.q_network import MLP
    out = resolve({'class': 'MLP', 'hidden': [128]}, reg=reg)
    assert out == MLP(hidden=(128,))


def test_resolve_class_instantiates_container(reg: Registry) -> None:
    from corroborate.rl.dqn.claims.replay import Replay
    out = resolve(
        {'class': 'Replay', 'capacity': 50000, 'batch_size': 32},
        reg=reg,
    )
    assert out == Replay(capacity=50000, batch_size=32)


def test_resolve_fn_no_kwargs_returns_bare_fnclaim(
    reg: Registry,
) -> None:
    """Bare `{fn: name}` resolves to the FnClaim itself, not a
    partial. Identity holds (registry returns the cached
    instance)."""
    from corroborate.rl.dqn.claims.bootstrap import double_greedify
    out = resolve({'fn': 'double_greedify'}, reg=reg)
    assert out is double_greedify


def test_resolve_fn_with_kwargs_returns_partial(
    reg: Registry,
) -> None:
    from corroborate.rl.dqn.claims.bootstrap import expectile_greedify
    out = resolve(
        {'fn': 'expectile_greedify', 'tau': 0.7}, reg=reg,
    )
    assert isinstance(out, partial)
    assert out.func is expectile_greedify
    assert out.keywords == {'tau': 0.7}


def test_resolve_nested_fn_in_fn_kwargs(reg: Registry) -> None:
    """Nested case: an FnClaim factory takes another FnClaim
    factory as a kwarg. Real example:
    `partial(warmed_update, inner=partial(adam, lr=...), warmup_steps=100)`
    — same shape but at factory-of-factories depth."""
    from corroborate.rl.dqn.claims.optimizer import adam, warmed_update
    out = resolve(
        {
            'fn': 'warmed_update',
            'inner': {'fn': 'adam', 'lr': 0.0001},
            'warmup_steps': 100,
        },
        reg=reg,
    )
    assert isinstance(out, partial)
    assert out.func is warmed_update
    inner_partial = out.keywords['inner']
    assert isinstance(inner_partial, partial)
    assert inner_partial.func is adam
    assert inner_partial.keywords == {'lr': 0.0001}
    assert out.keywords['warmup_steps'] == 100


def test_resolve_partial_of_fn_with_partial_kwarg(
    reg: Registry,
) -> None:
    """The expectile_3way intervention's bootstrap binding:
    `partial(bootstrap, greedification=partial(expectile_greedify,
    tau=0.7))`. Two-level nested partial."""
    from corroborate.rl.dqn.claims.bootstrap import (
        bootstrap, expectile_greedify,
    )
    out = resolve(
        {
            'fn': 'bootstrap',
            'greedification': {
                'fn': 'expectile_greedify', 'tau': 0.7,
            },
        },
        reg=reg,
    )
    assert isinstance(out, partial)
    assert out.func is bootstrap
    inner = out.keywords['greedification']
    assert isinstance(inner, partial)
    assert inner.func is expectile_greedify
    assert inner.keywords == {'tau': 0.7}


def test_resolve_unknown_class_raises(reg: Registry) -> None:
    with pytest.raises(KeyError, match='no class'):
        resolve({'class': 'NotARealModule'}, reg=reg)


def test_resolve_unknown_fn_raises(reg: Registry) -> None:
    with pytest.raises(KeyError, match='no FnClaim'):
        resolve({'fn': 'not_a_real_claim'}, reg=reg)


# ---------- hypothesis-level round-trip smoke ----------

def _expectile_yaml() -> str:
    return """
name: expectile_dqn
predicted_direction: a_gt_b
intervention:
  total_steps: 200000
  eval_every: 20000
  n_episodes: 5
  gamma: 0.99
  sync_period: 100
  replay: {class: Replay, capacity: 50000, batch_size: 32}
  optimizer:
    fn: warmed_update
    inner: {fn: adam, lr: 0.0001}
    warmup_steps: 100
  q_network: {class: MLP, hidden: [64, 64]}
  bootstrap:
    fn: bootstrap
    greedification: {fn: expectile_greedify, tau: 0.7}
intervention_arms:
  - slot_path: bootstrap
    replacement:
      fn: bootstrap
      greedification: {fn: expectile_greedify, tau: 0.7}
""".strip()


def _expectile_python() -> Hypothesis[Mapping[str, object]]:
    """Canonical Python recipe for the expectile arm — the
    reference the YAML loader must match."""
    from corroborate.rl.dqn.claims.bootstrap import (
        bootstrap, expectile_greedify,
    )
    from corroborate.rl.dqn.claims.optimizer import adam, warmed_update
    from corroborate.rl.dqn.claims.q_network import MLP
    from corroborate.rl.dqn.claims.replay import Replay
    boot = partial(
        bootstrap,
        greedification=partial(expectile_greedify, tau=0.7),
    )
    intervention: dict[str, object] = {
        'total_steps': 200_000,
        'eval_every': 20_000,
        'n_episodes': 5,
        'gamma': 0.99,
        'sync_period': 100,
        'replay': Replay(capacity=50_000, batch_size=32),
        'optimizer': partial(
            warmed_update,
            inner=partial(adam, lr=1e-4),
            warmup_steps=100,
        ),
        'q_network': MLP(hidden=(64, 64)),
        'bootstrap': boot,
    }
    return Hypothesis(
        name='expectile_dqn',
        intervention=intervention,
        predicted_direction='a_gt_b',
        intervention_arms=(
            Intervention(slot_path='bootstrap', replacement=boot),
        ),
    )


@pytest.fixture
def expectile_yaml_path(tmp_path: Path) -> Path:
    p = tmp_path / 'expectile.yaml'
    _ = p.write_text(_expectile_yaml())
    return p


def test_yaml_hypothesis_matches_python_authored(
    reg: Registry, expectile_yaml_path: Path,
) -> None:
    h_yaml = load_hypothesis(expectile_yaml_path, reg=reg)
    h_python = _expectile_python()
    assert h_yaml.name == h_python.name
    assert h_yaml.predicted_direction == h_python.predicted_direction


def test_yaml_intervention_leaves_match_python(
    reg: Registry, expectile_yaml_path: Path,
) -> None:
    h_yaml = load_hypothesis(expectile_yaml_path, reg=reg)
    h_python = _expectile_python()
    for k in (
        'total_steps', 'eval_every', 'n_episodes', 'gamma',
        'sync_period',
    ):
        assert h_yaml.intervention[k] == h_python.intervention[k]


def test_yaml_module_claim_slots_equal_python_constructions(
    reg: Registry, expectile_yaml_path: Path,
) -> None:
    h_yaml = load_hypothesis(expectile_yaml_path, reg=reg)
    h_python = _expectile_python()
    # Slot equality compared via `canonical_str` — `functools
    # .partial` instances don't define value equality, but the
    # canonical-string fingerprint does, by construction.
    from corroborate._internals.canonical import canonical_str
    for k in ('q_network', 'optimizer', 'replay'):
        assert canonical_str(h_yaml.intervention[k]) == canonical_str(
            h_python.intervention[k],
        )


def test_yaml_bootstrap_partial_signature_matches_python(
    reg: Registry, expectile_yaml_path: Path,
) -> None:
    """The headline contract: the slot Claim's `claim_graph_signature`
    is identical across YAML- and Python-built paths. If the YAML
    schema diverges from the Python authoring shape (different
    nesting, missing kwarg, wrong default elision), this signature
    diverges and the YAML run is structurally non-comparable to
    the Python run."""
    h_yaml = load_hypothesis(expectile_yaml_path, reg=reg)
    h_python = _expectile_python()
    sig_yaml = claim_graph_signature(h_yaml.intervention['bootstrap'])
    sig_python = claim_graph_signature(
        h_python.intervention['bootstrap'],
    )
    assert sig_yaml == sig_python


def test_yaml_arm_key_matches_python(
    reg: Registry, expectile_yaml_path: Path,
) -> None:
    """`arm_key()` is the canonical fingerprint of intervention_arms.
    If YAML and Python paths disagree on it, downstream pairing in
    `HypothesisComparisonRow.from_cells` lands rows in different
    arms — silently, not loudly. Asserting equality up front
    prevents that drift."""
    h_yaml = load_hypothesis(expectile_yaml_path, reg=reg)
    h_python = _expectile_python()
    assert h_yaml.arm_key() == h_python.arm_key()


def test_vanilla_yaml_has_empty_arms(
    reg: Registry, tmp_path: Path,
) -> None:
    """Baseline / vanilla arm: no interventions, omitted
    `intervention_arms` defaults to empty tuple → `arm_key()`
    returns 'baseline'."""
    p = tmp_path / 'vanilla.yaml'
    _ = p.write_text("""
name: vanilla_dqn
predicted_direction: null
intervention:
  total_steps: 200000
  gamma: 0.99
  q_network: {class: MLP, hidden: [64, 64]}
""".strip())
    h = load_hypothesis(p, reg=reg)
    assert h.intervention_arms == ()
    assert h.arm_key() == 'baseline'


def test_invalid_predicted_direction_raises(
    reg: Registry, tmp_path: Path,
) -> None:
    p = tmp_path / 'bad.yaml'
    _ = p.write_text("""
name: bad
predicted_direction: bogus
intervention: {}
""".strip())
    with pytest.raises(ValueError, match='predicted_direction'):
        _ = load_hypothesis(p, reg=reg)


def test_intervention_arm_with_non_callable_replacement_raises(
    reg: Registry, tmp_path: Path,
) -> None:
    p = tmp_path / 'bad_arm.yaml'
    _ = p.write_text("""
name: bad
intervention: {}
intervention_arms:
  - slot_path: gamma
    replacement: 0.99
""".strip())
    with pytest.raises(TypeError, match='must resolve to a callable'):
        _ = load_hypothesis(p, reg=reg)
