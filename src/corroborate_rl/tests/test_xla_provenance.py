"""Tests for `_xla_command_buffer_enabled` + `_xla_deterministic_ops`
provenance readers in `cell_runner`.

The corrected `_xla_command_buffer_enabled` must report XLA's EFFECTIVE
runtime cmdbuf state (not just "explicit-disable string absent"). The
key edge case: `det=True` with no explicit cmdbuf flag → XLA implicitly
disables cmdbuf → reader must return False (the prior implementation
returned True, falsely advertising cmdbuf=ON for bit-deterministic runs).
"""
from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from corroborate_rl.cell_runner import (
    _xla_command_buffer_enabled,
    _xla_deterministic_ops,
)


@contextmanager
def xla_flags(value: str | None) -> Iterator[None]:
    """Set XLA_FLAGS for the duration of the context. None unsets."""
    prior = os.environ.get('XLA_FLAGS')
    if value is None:
        os.environ.pop('XLA_FLAGS', None)
    else:
        os.environ['XLA_FLAGS'] = value
    try:
        yield
    finally:
        if prior is None:
            os.environ.pop('XLA_FLAGS', None)
        else:
            os.environ['XLA_FLAGS'] = prior


# ============ _xla_deterministic_ops ============

def test_det_unset() -> None:
    with xla_flags(None):
        assert _xla_deterministic_ops() is False


def test_det_false_explicit() -> None:
    with xla_flags('--xla_gpu_deterministic_ops=false'):
        assert _xla_deterministic_ops() is False


def test_det_true_explicit() -> None:
    with xla_flags('--xla_gpu_deterministic_ops=true'):
        assert _xla_deterministic_ops() is True


def test_det_true_among_other_flags() -> None:
    with xla_flags(
        '--some_other_flag=42 --xla_gpu_deterministic_ops=true '
        '--xla_gpu_enable_command_buffer=FUSION',
    ):
        assert _xla_deterministic_ops() is True


# ============ _xla_command_buffer_enabled ============

def test_cmdbuf_default_no_det() -> None:
    """Config (a): det=F, cmdbuf unset → XLA default ON.

    The worst-drift config the user empirically observed."""
    with xla_flags(None):
        assert _xla_command_buffer_enabled() is True


def test_cmdbuf_default_no_det_with_unrelated_flag() -> None:
    """det=F, cmdbuf unset, but XLA_FLAGS has unrelated content → True."""
    with xla_flags('--some_unrelated_flag=value'):
        assert _xla_command_buffer_enabled() is True


def test_cmdbuf_explicit_disable_no_det() -> None:
    """Config (b): det=F, cmdbuf explicitly disabled (empty value)
    → False. Prior cloud's anomalous d=-0.28 config."""
    with xla_flags('--xla_gpu_enable_command_buffer='):
        assert _xla_command_buffer_enabled() is False


def test_cmdbuf_explicit_disable_then_space() -> None:
    """Empty value followed by space (mid-flag) → still disabled."""
    with xla_flags(
        '--xla_gpu_enable_command_buffer= --other_flag=1',
    ):
        assert _xla_command_buffer_enabled() is False


def test_det_true_implicit_disable() -> None:
    """Config (c): det=T, cmdbuf unset → XLA implicitly disables
    cmdbuf (the fix). Prior implementation incorrectly reported True
    here. This is the implementation's default GPU configuration."""
    with xla_flags('--xla_gpu_deterministic_ops=true'):
        assert _xla_command_buffer_enabled() is False


def test_det_true_with_explicit_cmdbuf_enable_overrides() -> None:
    """Config (d): det=T but explicit `--xla_gpu_enable_command_buffer=
    FUSION,...` → the explicit value wins over det's implicit disable.
    This recovers cmdbuf at the cost of non-determinism."""
    with xla_flags(
        '--xla_gpu_deterministic_ops=true '
        '--xla_gpu_enable_command_buffer=FUSION,CUSTOM_CALL,CUBLAS,CUDNN',
    ):
        assert _xla_command_buffer_enabled() is True


def test_det_true_with_explicit_cmdbuf_disable() -> None:
    """det=T + cmdbuf explicit disable → False (explicit wins;
    redundant with implicit but reported consistently)."""
    with xla_flags(
        '--xla_gpu_deterministic_ops=true '
        '--xla_gpu_enable_command_buffer=',
    ):
        assert _xla_command_buffer_enabled() is False


def test_det_false_explicit_cmdbuf_enabled() -> None:
    """det=F + cmdbuf explicit enable (e.g., default FUSION list).
    Reports True; equivalent to config (a) but operator declared it."""
    with xla_flags(
        '--xla_gpu_deterministic_ops=false '
        '--xla_gpu_enable_command_buffer=FUSION',
    ):
        assert _xla_command_buffer_enabled() is True


@pytest.mark.parametrize('det_value', ['true', 'TRUE', 'True'])
def test_det_case_sensitive(det_value: str) -> None:
    """XLA's flag parser is case-sensitive for `true`. Reader follows
    the same convention (matches lowercase `=true` substring)."""
    with xla_flags(f'--xla_gpu_deterministic_ops={det_value}'):
        # Only lowercase 'true' counts.
        expected = det_value == 'true'
        assert _xla_deterministic_ops() is expected
