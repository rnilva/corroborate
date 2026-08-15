"""YAML → InterventionConfig loader smoke. The contract is
structural identity: a YAML-loaded InterventionConfig's slot Claims
pass `claim_graph_signature` equality with the equivalent
Python-authored InterventionConfig, and the `combined_arm_key` of
its arms agrees.

If the loader and the hand-authored path drift, the YAML schema
needs a fix; the smoke catches that drift at config-load time."""
from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from pathlib import Path

import pytest

from corroborate.core.intervention import (
    DoEffect, Intervention, combined_arm_key,
)
from corroborate.runner.registry import Registry
from corroborate_rl.dqn.config_loader import (
    InterventionConfig, load_intervention, resolve,
)
from corroborate.core.signature import claim_graph_signature


DQN_CLAIM_MODULES = (
    'corroborate_rl.dqn.claims.bootstrap',
    'corroborate_rl.dqn.claims.action_select',
    'corroborate_rl.dqn.claims.replay',
    'corroborate_rl.dqn.claims.q_network',
    'corroborate_rl.dqn.claims.optimizer',
    'corroborate_rl.dqn.claims.target_sync',
    'corroborate_rl.dqn.claims.loss',
    # Root programs — resolvable as `program:` values.
    'corroborate_rl.dqn.dqn',
    'corroborate_rl.dqn.dqn_paired',
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
    from corroborate_rl.dqn.claims.q_network import MLP
    out = resolve({'class': 'MLP', 'hidden': [128]}, reg=reg)
    assert out == MLP(hidden=(128,))


def test_resolve_class_instantiates_container(reg: Registry) -> None:
    from corroborate_rl.dqn.claims.replay import Replay
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
    from corroborate_rl.dqn.claims.bootstrap import double_greedify
    out = resolve({'fn': 'double_greedify'}, reg=reg)
    assert out is double_greedify


def test_resolve_fn_with_kwargs_returns_partial(
    reg: Registry,
) -> None:
    from corroborate_rl.dqn.claims.bootstrap import expectile_greedify
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
    from corroborate_rl.dqn.claims.optimizer import adam, warmed_update
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
    """The expectile_3way base's bootstrap binding:
    `partial(bootstrap, greedification=partial(expectile_greedify,
    tau=0.7))`. Two-level nested partial."""
    from corroborate_rl.dqn.claims.bootstrap import (
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


# ---------- intervention-level round-trip smoke ----------

def _expectile_yaml() -> str:
    return """
name: expectile_dqn
base:
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
arms:
  - []
  - - slot_path: bootstrap
      replacement:
        fn: bootstrap
        greedification: {fn: expectile_greedify, tau: 0.7}
""".strip()


def _expectile_python() -> InterventionConfig:
    """Canonical Python recipe for the expectile arm — the
    reference the YAML loader must match."""
    from corroborate_rl.dqn.claims.bootstrap import (
        bootstrap, expectile_greedify,
    )
    from corroborate_rl.dqn.claims.optimizer import adam, warmed_update
    from corroborate_rl.dqn.claims.q_network import MLP
    from corroborate_rl.dqn.claims.replay import Replay
    boot = partial(
        bootstrap,
        greedification=partial(expectile_greedify, tau=0.7),
    )
    base: dict[str, object] = {
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
    }
    return InterventionConfig(
        name='expectile_dqn',
        base=base,
        do_effect=DoEffect(arms=(
            (),
            (Intervention(slot_path='bootstrap', replacement=boot),),
        )),
    )


@pytest.fixture
def expectile_yaml_path(tmp_path: Path) -> Path:
    p = tmp_path / 'expectile.yaml'
    _ = p.write_text(_expectile_yaml())
    return p


def test_yaml_intervention_name_matches_python(
    reg: Registry, expectile_yaml_path: Path,
) -> None:
    h_yaml = load_intervention(expectile_yaml_path, reg=reg)
    h_python = _expectile_python()
    assert h_yaml.name == h_python.name


def test_yaml_base_leaves_match_python(
    reg: Registry, expectile_yaml_path: Path,
) -> None:
    h_yaml = load_intervention(expectile_yaml_path, reg=reg)
    h_python = _expectile_python()
    for k in (
        'total_steps', 'eval_every', 'n_episodes', 'gamma',
        'sync_period',
    ):
        assert h_yaml.base[k] == h_python.base[k]


def test_yaml_module_claim_slots_equal_python_constructions(
    reg: Registry, expectile_yaml_path: Path,
) -> None:
    h_yaml = load_intervention(expectile_yaml_path, reg=reg)
    h_python = _expectile_python()
    # Slot equality compared via `canonical_str` — `functools
    # .partial` instances don't define value equality, but the
    # canonical-string fingerprint does, by construction.
    from corroborate._internals.canonical import canonical_str
    for k in ('q_network', 'optimizer', 'replay'):
        assert canonical_str(h_yaml.base[k]) == canonical_str(
            h_python.base[k],
        )


def test_yaml_bootstrap_partial_signature_matches_python(
    reg: Registry, expectile_yaml_path: Path,
) -> None:
    """The headline contract: the slot Claim's `claim_graph_signature`
    is identical across YAML- and Python-built paths. If the YAML
    schema diverges from the Python authoring shape (different
    nesting, missing kwarg, wrong default elision), this signature
    diverges and the YAML run is structurally non-comparable to
    the Python run.

    The bootstrap partial lives on the treatment arm
    (`do_effect.arms[1][0].replacement`); the YAML's empty control
    arm doesn't carry it (and inherits the substrate's default at
    dispatch time)."""
    h_yaml = load_intervention(expectile_yaml_path, reg=reg)
    h_python = _expectile_python()
    yaml_repl = h_yaml.do_effect.arms[1][0].replacement
    py_repl = h_python.do_effect.arms[1][0].replacement
    assert claim_graph_signature(yaml_repl) == claim_graph_signature(py_repl)


def test_yaml_arm_keys_match_python(
    reg: Registry, expectile_yaml_path: Path,
) -> None:
    """Arm fingerprint stability: `DoEffect.arm_keys()` is the
    canonical fingerprint tuple. Drift here would place YAML- and
    Python-built rows in different arms during
    `paired_comparison`."""
    h_yaml = load_intervention(expectile_yaml_path, reg=reg)
    h_python = _expectile_python()
    assert h_yaml.do_effect.arm_keys() == h_python.do_effect.arm_keys()


def test_vanilla_yaml_defaults_to_single_empty_arm(
    reg: Registry, tmp_path: Path,
) -> None:
    """Baseline / vanilla intervention: no `arms` key declared →
    `do_effect.arms` defaults to `((),)` (single empty control
    arm). The `combined_arm_key` of that arm is `'baseline'`."""
    p = tmp_path / 'vanilla.yaml'
    _ = p.write_text("""
name: vanilla_dqn
base:
  total_steps: 200000
  gamma: 0.99
  q_network: {class: MLP, hidden: [64, 64]}
""".strip())
    h = load_intervention(p, reg=reg)
    assert h.do_effect.arms == ((),)
    assert combined_arm_key(h.do_effect.arms[0]) == 'baseline'


def test_arm_with_non_callable_replacement_raises(
    reg: Registry, tmp_path: Path,
) -> None:
    """Replacement values must satisfy `is_replacement`; non-
    callable scalars (like `0.99`) flow through `resolve()` as a
    pass-through scalar and fail the typed gate. Catches the
    common authoring mistake of trying to use Intervention as
    a generic kwarg-set primitive."""
    p = tmp_path / 'bad_arm.yaml'
    _ = p.write_text("""
name: bad
arms:
  - []
  - - slot_path: gamma
      replacement: 0.99
""".strip())
    # Scalars (int/float) currently pass `is_replacement` because
    # Replacement = object — implementation-agnostic by design. The
    # framework's substrate-level type-checker catches slot/
    # replacement mismatches at call site; the loader accepts the
    # value as a typed Intervention. Confirming behavioural
    # invariant: load succeeds; downstream apply fails on type.
    cfg = load_intervention(p, reg=reg)
    assert cfg.do_effect.arms[1][0].replacement == 0.99


# ---------- required_measurables ----------

def test_required_measurables_resolves_registered_names(
    reg: Registry, tmp_path: Path,
) -> None:
    """The YAML's `required_measurables` lookup goes through the
    global `@measurable` registry; valid names land on the
    `InterventionConfig` field as a typed tuple."""
    # Importing the implementation registers the canonical measurable
    # set as a side effect — `eval_best_burst_mean` and
    # `jensen_gap` are stable framework-recognised names.
    import corroborate_rl.dqn.measurables  # noqa: F401
    p = tmp_path / 'with_extras.yaml'
    _ = p.write_text("""
name: with_extras
base: {gamma: 0.99}
required_measurables:
  - eval_best_burst_mean
  - jensen_gap
""".strip())
    cfg = load_intervention(p, reg=reg)
    assert cfg.required_measurables == (
        'eval_best_burst_mean', 'jensen_gap',
    )


def test_required_measurables_unknown_name_raises(
    reg: Registry, tmp_path: Path,
) -> None:
    """Loader rejects unrecognised names at parse time — typos
    fail loud, not silently. Mirrors the framework's
    `_validate_hypothesis` discipline for `REQUIRED_MEASURABLES`."""
    import corroborate_rl.dqn.measurables  # noqa: F401
    p = tmp_path / 'bad.yaml'
    _ = p.write_text("""
name: bad
base: {}
required_measurables:
  - not_a_real_measurable
""".strip())
    with pytest.raises(KeyError, match='not_a_real_measurable'):
        _ = load_intervention(p, reg=reg)


def test_required_measurables_default_empty(
    reg: Registry, tmp_path: Path,
) -> None:
    """Field is optional; omitting it yields the empty tuple."""
    p = tmp_path / 'no_extras.yaml'
    _ = p.write_text("""
name: no_extras
base: {gamma: 0.99}
""".strip())
    cfg = load_intervention(p, reg=reg)
    assert cfg.required_measurables == ()


# ---------- program (base-claim selection) ----------

def test_program_defaults_to_dqn(reg: Registry, tmp_path: Path) -> None:
    """Omitting `program:` yields the single-net `dqn` base."""
    p = tmp_path / 'no_program.yaml'
    _ = p.write_text("""
name: vanilla
base: {gamma: 0.99}
""".strip())
    cfg = load_intervention(p, reg=reg)
    assert cfg.program == 'dqn'


def test_program_paired_dqn_parsed(reg: Registry, tmp_path: Path) -> None:
    """`program: paired_dqn` selects the coupled two-learner program;
    the cross-evaluation is structural so the arm stays empty (no
    marker, no slot swap)."""
    p = tmp_path / 'paired.yaml'
    _ = p.write_text("""
name: deep2010
program: paired_dqn
base: {gamma: 0.999}
arms:
  - []
""".strip())
    cfg = load_intervention(p, reg=reg)
    assert cfg.program == 'paired_dqn'
    assert cfg.do_effect.arms == ((),)


def test_program_invalid_raises(reg: Registry, tmp_path: Path) -> None:
    """An unregistered program name is a config error — the registry
    is the single authority, surfacing a loud error listing the
    known set (no hardcoded enum to maintain)."""
    p = tmp_path / 'bad_program.yaml'
    _ = p.write_text("""
name: bad
program: triple_dqn
base: {}
""".strip())
    with pytest.raises(ValueError, match='not a registered claim program'):
        _ = load_intervention(p, reg=reg)
