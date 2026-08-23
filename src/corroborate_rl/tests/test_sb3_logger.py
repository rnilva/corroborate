"""`corroborate_rl.sb3_logger` — Logger output → cells.

Fixtures are hand-written progress files (the reader has no SB3
dependency to exercise); assertions target the reader's actual
transformations: verbatim keys, dump-aligned null padding,
config-agreement enforcement, registry membership, and the
zero-semantics walk."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from corroborate.data.panel import Panel
from corroborate_rl.sb3_logger import (
    load_sb3_logger_runs,
    progress_columns,
    read_progress,
)


def _write_json(path: Path, dumps: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        '\n'.join(json.dumps(d) for d in dumps) + '\n',
    )


_DUMPS: list[dict[str, object]] = [
    {'config/gamma': 0.99, 'config/seed': 0,
     'rollout/ep_rew_mean': 14.75, 'time/total_timesteps': 59},
    {'rollout/ep_rew_mean': 16.9, 'time/total_timesteps': 135},
    {'rollout/ep_rew_mean': 21.0, 'train/loss': 0.41,
     'eval/mean_reward': 168.6, 'time/total_timesteps': 2000},
]


def test_read_progress_json_preserves_keys_and_absence(
    tmp_path: Path,
) -> None:
    file = tmp_path / 'progress.json'
    _write_json(file, _DUMPS)
    dumps = read_progress(file)
    assert len(dumps) == 3
    assert dumps[0]['config/gamma'] == 0.99
    assert 'train/loss' not in dumps[0]
    assert dumps[2]['eval/mean_reward'] == 168.6


def test_read_progress_csv_round_trips_absence_and_numbers(
    tmp_path: Path,
) -> None:
    file = tmp_path / 'progress.csv'
    _ = file.write_text(
        'config/gamma,rollout/ep_rew_mean,train/loss\n'
        '0.99,14.75,\n'
        ',16.9,0.41\n',
    )
    dumps = read_progress(file)
    assert dumps[0] == {'config/gamma': 0.99, 'rollout/ep_rew_mean': 14.75}
    assert dumps[1] == {'rollout/ep_rew_mean': 16.9, 'train/loss': 0.41}


def test_progress_columns_splits_config_from_records() -> None:
    columns, registry = progress_columns(
        [{k: v for k, v in d.items() if isinstance(v, (int, float))}
         for d in _DUMPS],
    )
    assert registry == {'config/gamma', 'config/seed'}
    assert columns['config/gamma'] == 0.99  # scalar, not array
    # Record arrays align dump-by-dump, null where not recorded.
    assert columns['train/loss'] == [None, None, 0.41]
    assert columns['eval/mean_reward'] == [None, None, 168.6]
    assert columns['time/total_timesteps'] == [59, 135, 2000]


def test_progress_columns_rejects_disagreeing_config() -> None:
    with pytest.raises(ValueError, match='config/gamma'):
        _ = progress_columns([
            {'config/gamma': 0.99},
            {'config/gamma': 0.8},
        ])


def test_progress_columns_promotes_mixed_int_float() -> None:
    columns, _ = progress_columns([
        {'train/loss': 1},
        {'train/loss': 0.5},
    ])
    assert columns['train/loss'] == [1.0, 0.5]


def test_progress_columns_rejects_mixed_numeric_and_text() -> None:
    with pytest.raises(ValueError, match='train/loss'):
        _ = progress_columns([
            {'train/loss': 1.0},
            {'train/loss': 'diverged'},
        ])


def test_load_walks_every_progress_file(tmp_path: Path) -> None:
    _write_json(tmp_path / 'a-s0' / 'progress.json', _DUMPS)
    _write_json(
        tmp_path / 'a-s1' / 'progress.json',
        [{'config/gamma': 0.8, 'config/lr': 1e-3,
          'rollout/ep_rew_mean': 9.0, 'time/total_timesteps': 60}],
    )
    panel = load_sb3_logger_runs(tmp_path)
    assert isinstance(panel, Panel)
    df = panel.cells.sort('id')
    assert df['id'].to_list() == ['a-s0', 'a-s1']
    # Diagonal concat: heterogeneous keys null-pad across rows.
    assert df['config/lr'].to_list() == [None, 1e-3]
    assert df['config/gamma'].to_list() == [0.99, 0.8]
    # Registry is the union of every log's config keys.
    assert panel.leaves == {'config/gamma', 'config/seed', 'config/lr'}
    assert panel.sources[0].corpus == tmp_path.name


def test_load_prefers_json_over_csv_in_one_directory(
    tmp_path: Path,
) -> None:
    run = tmp_path / 'run'
    _write_json(run / 'progress.json', [{'train/loss': 0.5}])
    _ = (run / 'progress.csv').write_text('train/loss\n0.9\n')
    panel = load_sb3_logger_runs(tmp_path)
    assert panel.cells['train/loss'].to_list() == [[0.5]]


def test_load_rejects_reserved_key(tmp_path: Path) -> None:
    _write_json(tmp_path / 'run' / 'progress.json', [{'id': 'x'}])
    with pytest.raises(ValueError, match="reserved key 'id'"):
        _ = load_sb3_logger_runs(tmp_path)


def test_load_raises_on_empty_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='no progress'):
        _ = load_sb3_logger_runs(tmp_path)
